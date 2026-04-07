"""
Configurable GPT model supporting GPT-2 and LLaMA-style architectures.

Supports:
- RMSNorm or LayerNorm
- RoPE or learned positional embeddings
- ReLU-squared or SwiGLU MLP
- Weight tying between embedding and LM head
- Automatic parameter labeling for Muon-family optimizers (muon_class attribute)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    model_dim: int = 768
    max_seq_len: int = 1024
    dropout: float = 0.0
    norm_type: str = "rmsnorm"      # "rmsnorm" or "layernorm"
    mlp_type: str = "relusqr"       # "relusqr" or "swiglu"
    pos_encoding: str = "rope"      # "rope" or "learned"
    bias: bool = False
    tie_weights: bool = True

    @property
    def hidden_dim(self) -> int:
        """MLP hidden dimension: 4x for relusqr, 8/3*4 rounded for swiglu."""
        if self.mlp_type == "swiglu":
            # Common SwiGLU sizing: 8/3 * model_dim, rounded to nearest 256
            raw = int(8 / 3 * self.model_dim)
            return ((raw + 255) // 256) * 256
        return 4 * self.model_dim


PRESET_CONFIGS = {
    "tiny": GPTConfig(
        num_layers=6, num_heads=6, head_dim=64, model_dim=384, max_seq_len=512,
    ),
    "small": GPTConfig(
        num_layers=12, num_heads=12, head_dim=64, model_dim=768, max_seq_len=1024,
    ),
    "medium": GPTConfig(
        num_layers=24, num_heads=16, head_dim=64, model_dim=1024, max_seq_len=1024,
    ),
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


def build_norm(dim: int, norm_type: str) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")


# ---------------------------------------------------------------------------
# Rotary Position Embeddings
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Standard RoPE with precomputed cos/sin cache."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(torch.bfloat16), persistent=False)

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to tensor of shape (B, T, H, D)."""
    cos = cos[None, :x.size(1), None, :]
    sin = sin[None, :x.size(1), None, :]
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# MLP variants
# ---------------------------------------------------------------------------

class ReluSquaredMLP(nn.Module):
    """FFN with ReLU-squared activation: relu(x)^2. Standard in Muon papers."""

    def __init__(self, model_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.up = nn.Linear(model_dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, model_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.relu(self.up(x)).square())


class SwiGLUMLP(nn.Module):
    """SwiGLU FFN as used in LLaMA."""

    def __init__(self, model_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(model_dim, hidden_dim, bias=bias)
        self.up = nn.Linear(model_dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, model_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.model_dim = config.model_dim
        assert config.num_heads * config.head_dim == config.model_dim, \
            f"num_heads * head_dim ({config.num_heads}*{config.head_dim}) must equal model_dim ({config.model_dim})"

        self.qkv = nn.Linear(config.model_dim, 3 * config.model_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.model_dim, config.model_dim, bias=config.bias)
        self.dropout = config.dropout

        # Use RoPE for Q,K
        self.use_rope = config.pos_encoding == "rope"

    def forward(self, x: Tensor, cos: Tensor | None = None, sin: Tensor | None = None) -> Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, D)

        if self.use_rope and cos is not None:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # (B, H, T, D) for scaled_dot_product_attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = build_norm(config.model_dim, config.norm_type)
        self.attn = CausalSelfAttention(config)
        self.norm2 = build_norm(config.model_dim, config.norm_type)

        if config.mlp_type == "relusqr":
            self.mlp = ReluSquaredMLP(config.model_dim, config.hidden_dim, config.bias)
        elif config.mlp_type == "swiglu":
            self.mlp = SwiGLUMLP(config.model_dim, config.hidden_dim, config.bias)
        else:
            raise ValueError(f"Unknown mlp_type: {config.mlp_type}")

    def forward(self, x: Tensor, cos: Tensor | None = None, sin: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.model_dim)

        if config.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(config.max_seq_len, config.model_dim)
        else:
            self.pos_emb = None

        if config.pos_encoding == "rope":
            self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
        else:
            self.rope = None

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.final_norm = build_norm(config.model_dim, config.norm_type)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()
        self._label_params()

    def _init_weights(self):
        """Initialize weights following GPT-2 conventions."""
        cfg = self.config
        std = 0.02
        residual_std = std / math.sqrt(2 * cfg.num_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "out_proj" in name or "down" in name:
                    nn.init.normal_(module.weight, mean=0.0, std=residual_std)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def _label_params(self):
        """
        Tag each parameter with ``muon_class`` and ``_diag_name`` attributes
        so the optimizer factory and diagnostic logger can identify them.

        Matrix params (2D Linear weights, excluding embeddings/lm_head) get
        ``muon_class = "matrix"``. Everything else gets ``muon_class = "nonmatrix"``.
        """
        embedding_ids = {id(self.tok_emb.weight)}
        if not self.config.tie_weights:
            embedding_ids.add(id(self.lm_head.weight))

        for name, p in self.named_parameters():
            p._diag_name = name
            if p.ndim == 2 and id(p) not in embedding_ids:
                p.muon_class = "matrix"
            else:
                p.muon_class = "nonmatrix"

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            input_ids: (B, T) int tensor of token ids.
            targets: (B, T) int tensor of next-token targets (optional).

        Returns:
            dict with keys:
                'logits': (B, T, V) float tensor
                'loss': scalar (present only if targets is provided)
        """
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)

        if self.pos_emb is not None:
            positions = torch.arange(T, device=input_ids.device)
            x = x + self.pos_emb(positions)

        cos, sin = None, None
        if self.rope is not None:
            cos, sin = self.rope(T)

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="mean",
            )
            result["loss"] = loss

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "GPT":
        """Create a GPT model from a named preset (tiny, small, medium)."""
        if name not in PRESET_CONFIGS:
            raise ValueError(f"Unknown preset: {name}. Choose from {list(PRESET_CONFIGS.keys())}")
        cfg = PRESET_CONFIGS[name]
        # Apply overrides
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                raise ValueError(f"Unknown GPTConfig field: {k}")
        return cls(cfg)
