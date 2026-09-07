"""Smoke tests for the Moonlight MoE training model."""

from __future__ import annotations

import torch

from orscale.model.moonlight_moe import MoonlightMoEForCausalLM, MoonlightSparseMoE
from orscale.optim import _split_params, build_optimizer
from scripts.train import build_model


def _make_model() -> MoonlightMoEForCausalLM:
    torch.manual_seed(0)
    return MoonlightMoEForCausalLM.from_preset("moonlight_tiny_moe")


def test_moonlight_moe_forward_backward_with_aux_loss():
    model = _make_model().train()
    x = torch.randint(0, model.config.vocab_size, (2, 16))
    y = torch.randint(0, model.config.vocab_size, (2, 16))

    out = model(x, y)

    assert set(out) == {"loss", "lm_loss", "aux_loss"}
    assert out["loss"].ndim == 0
    assert out["lm_loss"].ndim == 0
    assert out["aux_loss"].ndim == 0
    assert out["aux_loss"].item() > 0.0

    out["loss"].backward()
    assert model.embed_tokens.weight.grad is not None
    assert any(
        p.grad is not None
        for layer in model.layers
        if isinstance(layer.mlp, MoonlightSparseMoE)
        for p in layer.mlp.experts.parameters()
    )


def test_moonlight_moe_param_routing_matches_muon_rules():
    model = _make_model()

    assert model.embed_tokens.weight.muon_class == "nonmatrix"
    assert model.lm_head.weight.muon_class == "nonmatrix"

    gate_weights = [
        p
        for name, p in model.named_parameters()
        if name.endswith("gate.weight")
    ]
    expert_weights = [
        p
        for name, p in model.named_parameters()
        if ".experts." in name and p.ndim == 2
    ]
    assert gate_weights
    assert expert_weights
    assert all(p.muon_class == "matrix" for p in gate_weights + expert_weights)

    matrix_params, nonmatrix_params = _split_params(model)
    matrix_ids = {id(p) for p in matrix_params}
    nonmatrix_ids = {id(p) for p in nonmatrix_params}

    assert id(model.embed_tokens.weight) in nonmatrix_ids
    assert id(model.lm_head.weight) in nonmatrix_ids
    assert all(id(p) in matrix_ids for p in gate_weights + expert_weights)


def test_moonlight_moe_build_optimizer_step_runs():
    model = _make_model().train()
    opts = build_optimizer(
        "orscale_lm",
        model,
        {
            "lr": 1e-3,
            "momentum": 0.95,
            "weight_decay": 0.1,
            "adamw_lr": 1e-3,
            "adamw_weight_decay": 0.1,
            "r_min": 0.1,
            "r_max": 5.0,
        },
    )
    x = torch.randint(0, model.config.vocab_size, (2, 16))
    y = torch.randint(0, model.config.vocab_size, (2, 16))

    out = model(x, y)
    out["loss"].backward()
    for opt in opts:
        opt.step()
        opt.zero_grad(set_to_none=True)

    assert out["loss"].isfinite().item()


def test_moonlight_moe_parameter_count_reports_total_and_active():
    model = _make_model()

    total = model.count_parameters()
    active = model.count_activated_parameters()

    assert total > 0
    assert active > 0
    assert total > active


def test_train_build_model_can_cast_params_to_bfloat16():
    model = build_model(
        {
            "architecture": "moonlight_moe",
            "preset": "moonlight_tiny_moe",
            "param_dtype": "bfloat16",
        },
        torch.device("cpu"),
    )

    assert next(model.parameters()).dtype == torch.bfloat16


def test_router_bias_update_is_deferred_until_after_backward_with_checkpoint():
    model = MoonlightMoEForCausalLM.from_preset(
        "moonlight_tiny_moe",
        checkpoint_moe=True,
        router_bias_update_rate=1e-2,
    ).train()
    x = torch.randint(0, model.config.vocab_size, (2, 16))
    y = torch.randint(0, model.config.vocab_size, (2, 16))

    gates = [
        layer.mlp.gate
        for layer in model.layers
        if isinstance(layer.mlp, MoonlightSparseMoE)
    ]
    before = [gate.routing_bias.detach().clone() for gate in gates]

    out = model(x, y)
    out["loss"].backward()

    for gate, old_bias in zip(gates, before):
        torch.testing.assert_close(gate.routing_bias, old_bias)
        assert gate._pending_update_count.item() == 1.0

    model.apply_router_bias_updates()

    for gate, old_bias in zip(gates, before):
        assert not torch.equal(gate.routing_bias, old_bias)
        assert gate._pending_update_count.item() == 0.0
