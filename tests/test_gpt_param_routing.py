"""
Tests for GPT parameter routing between Muon-family and AdamW.

Following Keller Jordan's Muon blog ("Empirical considerations") and the
Moonlight paper (arXiv:2502.16982, Sec 2.2), the following parameters MUST
be optimized with AdamW rather than Muon, even when they are 2D tensors:

    - token embedding weight (nn.Embedding),
    - learned positional embedding weight (nn.Embedding), if used,
    - final LM head / classifier weight (output layer),
    - all scalar / vector parameters (RMSNorm, LayerNorm gains, biases).

These tests verify that both the model's ``_label_params`` and
``build_optimizer`` agree on this routing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from orscale.model.gpt import GPT, GPTConfig, RMSNorm
from orscale.optim import _split_params, build_optimizer


def _make_cfg(pos_encoding: str = "rope", tie_weights: bool = True, bias: bool = False) -> GPTConfig:
    return GPTConfig(
        vocab_size=256,
        num_layers=2,
        num_heads=2,
        head_dim=16,
        model_dim=32,
        max_seq_len=16,
        norm_type="rmsnorm",
        mlp_type="relusqr",
        pos_encoding=pos_encoding,
        bias=bias,
        tie_weights=tie_weights,
    )


def test_tok_emb_uses_adamw():
    model = GPT(_make_cfg())
    assert model.tok_emb.weight.muon_class == "nonmatrix"


def test_learned_pos_emb_uses_adamw():
    """Regression: pos_emb is an nn.Embedding and must go to AdamW even though
    its weight is 2D and has the same shape family as a hidden Linear."""
    model = GPT(_make_cfg(pos_encoding="learned"))
    assert model.pos_emb is not None
    assert model.pos_emb.weight.muon_class == "nonmatrix"


def test_untied_lm_head_uses_adamw():
    model = GPT(_make_cfg(tie_weights=False))
    assert model.lm_head.weight.muon_class == "nonmatrix"
    assert id(model.lm_head.weight) != id(model.tok_emb.weight)


def test_tied_lm_head_uses_adamw():
    model = GPT(_make_cfg(tie_weights=True))
    assert model.lm_head.weight is model.tok_emb.weight
    assert model.lm_head.weight.muon_class == "nonmatrix"


def test_rmsnorm_params_use_adamw():
    model = GPT(_make_cfg())
    for name, p in model.named_parameters():
        if "norm" in name:
            assert p.muon_class == "nonmatrix", f"{name} should be nonmatrix (got {p.muon_class})"


def test_rmsnorm_preserves_activation_dtype():
    norm = RMSNorm(8)
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)

    out = norm(x)

    assert out.dtype == x.dtype


def test_chunked_loss_matches_full_logits_loss():
    torch.manual_seed(0)
    cfg = _make_cfg(pos_encoding="learned", tie_weights=False)
    cfg.loss_chunk_size = 3
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(x, y)
    logits = model(x)["logits"]
    expected = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="mean",
    )

    assert "logits" not in out
    torch.testing.assert_close(out["loss"], expected)


def test_hidden_linear_weights_use_muon():
    model = GPT(_make_cfg())
    hidden_linear_names = ["qkv.weight", "out_proj.weight", "up.weight", "down.weight"]
    seen = set()
    for name, p in model.named_parameters():
        for key in hidden_linear_names:
            if name.endswith(key):
                assert p.muon_class == "matrix", f"{name} should be matrix (got {p.muon_class})"
                seen.add(key)
    assert seen == set(hidden_linear_names), f"missing hidden linear params: {set(hidden_linear_names) - seen}"


def test_split_params_routes_learned_pos_emb_to_adamw():
    """End-to-end check via build_optimizer's param splitter."""
    model = GPT(_make_cfg(pos_encoding="learned"))
    matrix_params, nonmatrix_params = _split_params(model)
    matrix_ids = {id(p) for p in matrix_params}
    nonmatrix_ids = {id(p) for p in nonmatrix_params}

    assert id(model.tok_emb.weight) in nonmatrix_ids
    assert id(model.pos_emb.weight) in nonmatrix_ids
    assert id(model.pos_emb.weight) not in matrix_ids


def test_build_optimizer_splits_muon_and_adamw_correctly():
    """build_optimizer should return [muon_opt, adamw_opt] with the right params."""
    model = GPT(_make_cfg(pos_encoding="learned", tie_weights=False))
    opts = build_optimizer(
        "muon",
        model,
        {"lr": 0.02, "momentum": 0.95, "adamw_lr": 1e-3, "ns_iters": 5},
    )
    assert isinstance(opts, list) and len(opts) == 2
    muon_opt, adamw_opt = opts

    muon_param_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}
    adamw_param_ids = {id(p) for g in adamw_opt.param_groups for p in g["params"]}

    # These MUST use AdamW.
    assert id(model.tok_emb.weight) in adamw_param_ids
    assert id(model.pos_emb.weight) in adamw_param_ids
    assert id(model.lm_head.weight) in adamw_param_ids
    # All RMSNorm / LayerNorm weights (1D) must be in AdamW too.
    for name, p in model.named_parameters():
        if "norm" in name:
            assert id(p) in adamw_param_ids, f"{name} should be in AdamW group"

    # Hidden matrix weights must use Muon.
    for name, p in model.named_parameters():
        if name.endswith(("qkv.weight", "out_proj.weight", "up.weight", "down.weight")):
            assert id(p) in muon_param_ids, f"{name} should be in Muon group"
            assert id(p) not in adamw_param_ids


def test_forward_backward_works_after_routing():
    """Sanity: model still trains end-to-end with split optimizers."""
    torch.manual_seed(0)
    model = GPT(_make_cfg(pos_encoding="learned", tie_weights=False))
    opts = build_optimizer(
        "muon",
        model,
        {"lr": 0.02, "momentum": 0.95, "adamw_lr": 1e-3, "ns_iters": 5},
    )

    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))

    losses = []
    for _ in range(5):
        out = model(x, y)
        for o in opts:
            o.zero_grad()
        out["loss"].backward()
        for o in opts:
            o.step()
        losses.append(out["loss"].item())

    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
