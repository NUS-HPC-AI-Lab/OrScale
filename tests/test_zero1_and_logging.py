"""Tests for ZeRO-1 wrapper and selected diagnostics logging."""

from __future__ import annotations

import torch
import torch.nn as nn

from orscale.diagnostics.logger import DiagnosticLogger
from orscale.optim import build_optimizer
from orscale.optim.zero1 import Zero1Optimizer, make_zero1_partitions


def test_zero1_wrapper_steps_and_zeros_all_replicated_grads():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 4, bias=False))
    partitions = make_zero1_partitions(model, world_size=1)
    local_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt = Zero1Optimizer([local_opt], partitions)

    x = torch.randn(2, 8)
    loss = model(x).square().mean()
    loss.backward()
    before = [p.detach().clone() for p in model.parameters()]

    opt.step()
    opt.zero_grad(set_to_none=True)

    for old, new in zip(before, model.parameters()):
        assert not torch.equal(old, new)
        assert new.grad is None
    state = opt.state_dict()
    assert state["world_size"] == 1
    assert len(state["partitions"]) == len(list(model.parameters()))


def test_build_optimizer_ignores_zero1_when_not_distributed():
    model = nn.Linear(8, 4, bias=False)
    model.weight.muon_class = "matrix"
    opts = build_optimizer(
        "muon_moonlight",
        model,
        {"lr": 1e-3, "zero_stage": 1},
    )

    assert isinstance(opts, list)
    assert not getattr(opts[0], "is_zero1", False)


def test_diagnostics_selected_metrics_without_per_param_spam():
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    model.weight._diag_name = "layers.0.self_attn.o_proj.weight"
    model.weight.muon_class = "matrix"
    opts = build_optimizer(
        "orscale_lm",
        model,
        {"lr": 1e-3, "weight_decay": 0.1, "r_min": 0.1, "r_max": 5.0},
    )

    x = torch.randn(4, 8)
    loss = model(x).square().mean()
    loss.backward()
    for opt in opts:
        opt.step()

    logger = DiagnosticLogger(
        model=model,
        optimizers=opts,
        log_every=1,
        use_wandb=False,
        log_per_param=False,
        selected_param_patterns=["layers.0.*.weight"],
    )
    metrics = logger.collect(step=1)

    assert metrics is not None
    assert "diagnostics/layers.0.self_attn.o_proj.weight/trust_ratio_clipped" not in metrics
    assert "diagnostics/selected/layers.0.self_attn.o_proj.weight/trust_ratio_clipped" in metrics
    assert "diagnostics/_summary/trust_ratio_clipped_mean" in metrics
    assert "diagnostics/_summary/trust_ratio_clipped_p95" in metrics
    assert "selected_r" in logger.format_console_summary(metrics)
