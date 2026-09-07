"""Moonlight / DeepSeek-V3-Small style MoE language model.

This module implements the pieces needed for optimizer-comparison experiments
on the public Moonlight-16B-A3B shape:

* MLA attention with compressed KV states,
* SwiGLU dense and expert MLPs,
* top-k routed experts plus shared experts,
* optional router aux loss and aux-free style routing-bias updates,
* parameter labels compatible with the Muon/OrScale optimizer factory.

It is intentionally a training model, not a HuggingFace checkpoint loader.
The architecture constants mirror the public Moonlight config, while keeping
the code small enough to run tiny CPU/GPU smoke tests in this repository.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from orscale.model.gpt import RMSNorm


@dataclass
class MoonlightMoEConfig:
    vocab_size: int = 163840
    num_layers: int = 27
    num_heads: int = 16
    hidden_size: int = 2048
    max_seq_len: int = 8192

    # DeepSeek MLA dimensions. q_head_dim = qk_nope_head_dim + qk_rope_head_dim.
    q_lora_rank: int | None = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 10000.0

    # MLP / MoE dimensions.
    intermediate_size: int = 10944
    moe_intermediate_size: int = 1408
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    num_experts_per_tok: int = 6
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1

    # Router behavior.
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.446
    router_aux_loss_alpha: float = 0.0
    router_bias_update_rate: float = 0.0

    # Training/runtime.
    attention_dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = False
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.006
    loss_chunk_size: int = 2048
    checkpoint_mlp: bool = False
    checkpoint_moe: bool = False

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


MOONLIGHT_MOE_PRESET_CONFIGS = {
    "moonlight_16b_a3b": MoonlightMoEConfig(),
    # Tiny preset for tests and local wiring checks. It preserves the same
    # structural choices while keeping routing and attention cheap.
    "moonlight_tiny_moe": MoonlightMoEConfig(
        vocab_size=256,
        num_layers=3,
        num_heads=2,
        hidden_size=64,
        max_seq_len=64,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        intermediate_size=160,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        router_aux_loss_alpha=1e-3,
        routed_scaling_factor=1.5,
        loss_chunk_size=0,
    ),
}


class MoonlightRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
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
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [B, T, H, D] or [B, T, 1, D]
    cos = cos[None, : x.size(1), None, :]
    sin = sin[None, : x.size(1), None, :]
    return x * cos + _rotate_half(x) * sin


class MoonlightMLA(nn.Module):
    """Multi-head latent attention as used by DeepSeek-style Moonlight models."""

    def __init__(self, config: MoonlightMoEConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.q_head_dim = config.q_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.softmax_scale = self.q_head_dim ** -0.5
        self.dropout = config.attention_dropout

        if config.q_lora_rank is None:
            self.q_proj = nn.Linear(
                config.hidden_size,
                config.num_heads * self.q_head_dim,
                bias=config.bias,
            )
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
        else:
            self.q_proj = None
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.bias)
            self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank,
                config.num_heads * self.q_head_dim,
                bias=config.bias,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.bias,
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=config.bias,
        )
        self.o_proj = nn.Linear(
            config.num_heads * config.v_head_dim,
            config.hidden_size,
            bias=config.bias,
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape

        if self.q_proj is not None:
            q = self.q_proj(x)
        else:
            assert self.q_a_proj is not None
            assert self.q_a_layernorm is not None
            assert self.q_b_proj is not None
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(bsz, seq_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(
            compressed_kv,
            [self.config.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(
            bsz,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, value_states = torch.split(
            kv,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )

        q_pe = _apply_rope(q_pe, cos, sin)
        k_pe = _apply_rope(k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim), cos, sin)
        k_pe = k_pe.expand(-1, -1, self.num_heads, -1)

        query_states = torch.cat((q_nope, q_pe), dim=-1).transpose(1, 2)
        key_states = torch.cat((k_nope, k_pe), dim=-1).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.softmax_scale,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output)


class MoonlightSwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoonlightMoEGate(nn.Module):
    def __init__(self, config: MoonlightMoEConfig):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.aux_loss_alpha = config.router_aux_loss_alpha
        self.bias_update_rate = config.router_bias_update_rate

        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))
        self.register_buffer("routing_bias", torch.zeros(self.num_experts), persistent=True)
        self.register_buffer(
            "_pending_expert_fraction_sum",
            torch.zeros(self.num_experts),
            persistent=False,
        )
        self.register_buffer(
            "_pending_update_count",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = F.linear(hidden_states, self.weight)
        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits)
        elif self.scoring_func == "softmax":
            scores = F.softmax(logits, dim=-1)
        else:
            raise ValueError(f"Unsupported scoring_func: {self.scoring_func}")

        scores_for_choice = scores + self.routing_bias.to(scores.dtype)
        _, selected_experts = torch.topk(scores_for_choice, k=self.top_k, dim=-1)
        routing_weights = scores.gather(-1, selected_experts)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
        routing_weights = routing_weights * self.routed_scaling_factor

        aux_loss = self._aux_loss(scores, selected_experts)
        return routing_weights, selected_experts, aux_loss

    def _aux_loss(self, scores: Tensor, selected_experts: Tensor) -> Tensor:
        if self.aux_loss_alpha <= 0.0:
            return scores.new_zeros(())
        selected = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        expert_fraction = selected.sum(dim=1).mean(dim=0) / float(self.top_k)
        score_probs = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        score_fraction = score_probs.float().mean(dim=0)
        aux = self.num_experts * torch.sum(expert_fraction * score_fraction)
        return aux.to(scores.dtype) * self.aux_loss_alpha

    @torch.no_grad()
    def queue_routing_bias_update(self, selected_experts: Tensor) -> None:
        if self.bias_update_rate <= 0.0:
            return
        counts = torch.bincount(
            selected_experts.reshape(-1),
            minlength=self.num_experts,
        ).float()
        expert_fraction = counts / max(1, selected_experts.numel())
        self._pending_expert_fraction_sum.add_(
            expert_fraction.to(self._pending_expert_fraction_sum.device)
        )
        self._pending_update_count.add_(1.0)

    @torch.no_grad()
    def apply_queued_routing_bias_update(self) -> None:
        if self.bias_update_rate <= 0.0 or self._pending_update_count.item() == 0.0:
            return

        expert_fraction_sum = self._pending_expert_fraction_sum.float().clone()
        update_count = self._pending_update_count.float().clone()
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(expert_fraction_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(update_count, op=dist.ReduceOp.SUM)

        expert_fraction = expert_fraction_sum / update_count.clamp_min(1.0)
        target = 1.0 / float(self.num_experts)
        signs = torch.sign(expert_fraction - target)
        signs = signs - signs.mean()
        self.routing_bias.add_(signs.to(self.routing_bias.dtype), alpha=self.bias_update_rate)
        self._pending_expert_fraction_sum.zero_()
        self._pending_update_count.zero_()


class MoonlightSparseMoE(nn.Module):
    def __init__(self, config: MoonlightMoEConfig):
        super().__init__()
        self.config = config
        self.gate = MoonlightMoEGate(config)
        self.experts = nn.ModuleList(
            [
                MoonlightSwiGLUMLP(
                    config.hidden_size,
                    config.moe_intermediate_size,
                    bias=config.bias,
                )
                for _ in range(config.n_routed_experts)
            ]
        )
        shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = MoonlightSwiGLUMLP(
            config.hidden_size,
            shared_intermediate,
            bias=config.bias,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        orig_shape = x.shape
        flat_x = x.reshape(-1, orig_shape[-1])
        routing_weights, selected_experts, aux_loss = self.gate(flat_x)

        final = torch.zeros_like(flat_x)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, route_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_input = flat_x.index_select(0, token_idx)
            expert_output = expert(expert_input)
            expert_output = expert_output * routing_weights[token_idx, route_idx].unsqueeze(-1)
            final.index_add_(0, token_idx, expert_output)

        final = final + self.shared_experts(flat_x)
        return final.view(orig_shape), aux_loss, selected_experts


class MoonlightDecoderLayer(nn.Module):
    def __init__(self, config: MoonlightMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MoonlightMLA(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        use_moe = (
            layer_idx >= config.first_k_dense_replace
            and (layer_idx - config.first_k_dense_replace) % config.moe_layer_freq == 0
        )
        self.mlp = (
            MoonlightSparseMoE(config)
            if use_moe
            else MoonlightSwiGLUMLP(
                config.hidden_size,
                config.intermediate_size,
                bias=config.bias,
            )
        )

    def _mlp_residual(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        normed = self.post_attention_layernorm(x)
        if isinstance(self.mlp, MoonlightSparseMoE):
            return self.mlp(normed)
        empty_selected = torch.empty(0, dtype=torch.long, device=normed.device)
        return self.mlp(normed), normed.new_zeros(()), empty_selected

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        if (
            self.training
            and (
                self.config.checkpoint_mlp
                or (self.config.checkpoint_moe and isinstance(self.mlp, MoonlightSparseMoE))
            )
        ):
            mlp_out, aux_loss, selected_experts = _grad_checkpoint(
                self._mlp_residual,
                x,
                use_reentrant=False,
            )
        else:
            mlp_out, aux_loss, selected_experts = self._mlp_residual(x)
        if (
            self.training
            and isinstance(self.mlp, MoonlightSparseMoE)
            and selected_experts.numel() > 0
        ):
            self.mlp.gate.queue_routing_bias_update(selected_experts.detach())
        x = x + mlp_out
        return x, aux_loss


class MoonlightMoEForCausalLM(nn.Module):
    def __init__(self, config: MoonlightMoEConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = MoonlightRotaryEmbedding(
            config.qk_rope_head_dim,
            config.max_seq_len,
            base=config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [MoonlightDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()
        self._label_params()

    def _init_weights(self) -> None:
        std = self.config.initializer_range
        residual_std = std / math.sqrt(2 * self.config.num_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                init_std = residual_std if name.endswith(("o_proj", "down_proj")) else std
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, MoonlightMoEGate):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def _label_params(self) -> None:
        nonmatrix_ids = {id(self.embed_tokens.weight), id(self.lm_head.weight)}
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nonmatrix_ids.add(id(module.weight))

        for name, p in self.named_parameters():
            p._diag_name = name
            if p.ndim == 2 and id(p) not in nonmatrix_ids:
                p.muon_class = "matrix"
            else:
                p.muon_class = "nonmatrix"

    def _chunked_loss(self, x: Tensor, targets: Tensor) -> Tensor:
        flat_x = x.reshape(-1, x.size(-1))
        flat_targets = targets.reshape(-1)
        chunk_size = int(self.config.loss_chunk_size)

        if chunk_size <= 0 or flat_x.size(0) <= chunk_size:
            logits = self.lm_head(flat_x)
            return F.cross_entropy(logits, flat_targets, reduction="mean")

        loss_sum = flat_x.new_zeros(())
        for start in range(0, flat_x.size(0), chunk_size):
            end = min(start + chunk_size, flat_x.size(0))
            logits = self.lm_head(flat_x[start:end])
            loss_sum = loss_sum + F.cross_entropy(
                logits,
                flat_targets[start:end],
                reduction="sum",
            )
        return loss_sum / flat_targets.numel()

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
    ) -> dict[str, Tensor]:
        _, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(seq_len)

        aux_loss = x.new_zeros(())
        for layer in self.layers:
            x, layer_aux = layer(x, cos, sin)
            aux_loss = aux_loss + layer_aux
        aux_loss = aux_loss / max(1, self._num_moe_layers())

        x = self.norm(x)
        if targets is not None:
            lm_loss = self._chunked_loss(x, targets)
            return {
                "loss": lm_loss + aux_loss,
                "lm_loss": lm_loss,
                "aux_loss": aux_loss,
            }

        return {"logits": self.lm_head(x), "aux_loss": aux_loss}

    def _num_moe_layers(self) -> int:
        return sum(isinstance(layer.mlp, MoonlightSparseMoE) for layer in self.layers)

    @torch.no_grad()
    def apply_router_bias_updates(self) -> None:
        for layer in self.layers:
            if isinstance(layer.mlp, MoonlightSparseMoE):
                layer.mlp.gate.apply_queued_routing_bias_update()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_activated_parameters(self) -> int:
        """Approximate active parameters per token, excluding embeddings/head."""
        total = 0
        for layer in self.layers:
            total += sum(p.numel() for p in layer.self_attn.parameters())
            total += sum(p.numel() for p in layer.input_layernorm.parameters())
            total += sum(p.numel() for p in layer.post_attention_layernorm.parameters())
            if isinstance(layer.mlp, MoonlightSparseMoE):
                expert_params = sum(p.numel() for p in layer.mlp.experts[0].parameters())
                total += self.config.num_experts_per_tok * expert_params
                total += sum(p.numel() for p in layer.mlp.shared_experts.parameters())
                total += layer.mlp.gate.weight.numel()
            else:
                total += sum(p.numel() for p in layer.mlp.parameters())
        total += sum(p.numel() for p in self.norm.parameters())
        return total

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "MoonlightMoEForCausalLM":
        if name not in MOONLIGHT_MOE_PRESET_CONFIGS:
            raise ValueError(
                f"Unknown Moonlight MoE preset: {name}. "
                f"Choose from {list(MOONLIGHT_MOE_PRESET_CONFIGS.keys())}"
            )
        cfg = copy.deepcopy(MOONLIGHT_MOE_PRESET_CONFIGS[name])
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                raise ValueError(f"Unknown MoonlightMoEConfig field: {key}")
        return cls(cfg)
