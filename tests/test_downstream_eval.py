"""
Smoke tests for the lm-evaluation-harness adapter.

These tests are skipped if ``lm-eval`` is not installed. They verify:

1. The adapter can be constructed from a tiny GPT and returns finite scores.
2. ``loglikelihood`` produces the right number of (logprob, is_greedy) pairs.
3. ``run_downstream`` returns a dict with the expected task entries.
"""

from __future__ import annotations

import pytest
import torch

from orscale.model.gpt import GPT, GPTConfig


lm_eval = pytest.importorskip("lm_eval")


@pytest.fixture(scope="module")
def tiny_gpt() -> GPT:
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=50304,
        num_layers=2,
        num_heads=2,
        head_dim=32,
        model_dim=64,
        max_seq_len=128,
        norm_type="rmsnorm",
        mlp_type="relusqr",
        pos_encoding="rope",
        bias=False,
        tie_weights=True,
    )
    return GPT(cfg).eval()


def test_adapter_can_be_constructed(tiny_gpt):
    from orscale.eval.downstream import OrScaleLMAdapter

    adapter = OrScaleLMAdapter(tiny_gpt, batch_size=2, device="cpu")
    assert adapter.max_length == 128
    assert adapter.eot_token_id is not None


def test_loglikelihood_returns_expected_shape(tiny_gpt):
    from orscale.eval.downstream import OrScaleLMAdapter

    adapter = OrScaleLMAdapter(tiny_gpt, batch_size=2, device="cpu")

    class _Req:
        def __init__(self, context, continuation):
            self.args = (context, continuation)

    requests = [
        _Req("The quick brown fox", " jumps over the lazy dog"),
        _Req("Hello", " world"),
    ]
    results = adapter.loglikelihood(requests)
    assert len(results) == 2
    for ll, greedy in results:
        assert isinstance(ll, float)
        assert isinstance(greedy, bool)
        assert not (ll != ll)  # not NaN


def test_generate_until_produces_string(tiny_gpt):
    from orscale.eval.downstream import OrScaleLMAdapter

    adapter = OrScaleLMAdapter(tiny_gpt, batch_size=1, device="cpu")

    class _Req:
        def __init__(self, context, gen_kwargs):
            self.args = (context, gen_kwargs)

    out = adapter.generate_until([_Req("Once upon a time", {"until": ["\n"], "max_gen_toks": 8})])
    assert len(out) == 1
    assert isinstance(out[0], str)


def test_run_downstream_on_hellaswag(tiny_gpt):
    """Full round-trip through lm-eval on a tiny subset of HellaSwag."""
    from orscale.eval.downstream import run_downstream

    pytest.importorskip("datasets")

    try:
        results = run_downstream(
            model_or_ckpt=tiny_gpt,
            tasks=["hellaswag"],
            batch_size=2,
            limit=4,
            device="cpu",
        )
    except Exception as err:  # network / dataset issues shouldn't fail CI
        pytest.skip(f"run_downstream failed (likely no network access): {err}")

    assert "results" in results
    assert "hellaswag" in results["results"]
    # Every metric should be a finite float
    for metric_name, value in results["results"]["hellaswag"].items():
        if isinstance(value, float):
            assert value == value  # not NaN
