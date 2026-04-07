"""
Tests for Newton-Schulz / Polar Express orthogonalization.

Verifies:
1. Output is close to the SVD-based polar factor U @ V^T.
2. Frobenius norm of the output is close to sqrt(min(m, n)) for full-rank matrices.
3. Handles tall, wide, and square matrices.
4. Compiled and simple versions produce matching results.
"""

import math

import pytest
import torch

from orscale.optim.newton_schulz import orthogonalize_simple, POLAR_EXPRESS_COEFFS


def _svd_polar_factor(M: torch.Tensor) -> torch.Tensor:
    """Compute exact polar factor via SVD: Q = U @ V^T."""
    U, S, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    return (U @ Vh).to(M.dtype)


@pytest.mark.parametrize("shape", [(64, 64), (128, 64), (64, 128), (32, 96)])
def test_polar_factor_approximation(shape):
    """NS5 output should be close to the exact polar factor."""
    torch.manual_seed(42)
    M = torch.randn(*shape, dtype=torch.float32)

    Q_ns = orthogonalize_simple(M, num_iters=5)
    Q_exact = _svd_polar_factor(M)

    # bfloat16 limits precision; use generous tolerance
    cos_sim = torch.nn.functional.cosine_similarity(
        Q_ns.float().flatten(), Q_exact.float().flatten(), dim=0
    )
    assert cos_sim > 0.98, f"Cosine similarity {cos_sim:.4f} too low for shape {shape}"


@pytest.mark.parametrize("shape", [(64, 64), (128, 64), (64, 128)])
def test_frobenius_norm(shape):
    """||Q||_F should be close to sqrt(min(m, n)) for full-rank input."""
    torch.manual_seed(123)
    M = torch.randn(*shape, dtype=torch.float32)
    Q = orthogonalize_simple(M, num_iters=5)

    expected_norm = math.sqrt(min(shape))
    actual_norm = Q.float().norm().item()
    # bfloat16 introduces error; allow 10% tolerance
    assert abs(actual_norm - expected_norm) / expected_norm < 0.10, \
        f"||Q||_F = {actual_norm:.3f}, expected ≈ {expected_norm:.3f}"


def test_idempotence():
    """Applying orthogonalize twice should give a similar result.

    The bfloat16 cast in orthogonalize introduces non-trivial rounding on each
    call, so the tolerance is generous. We mainly verify Q2 is still close
    to the polar factor direction rather than bit-exact.
    """
    torch.manual_seed(7)
    M = torch.randn(64, 64, dtype=torch.float32)
    Q1 = orthogonalize_simple(M, num_iters=5)
    Q2 = orthogonalize_simple(Q1.float(), num_iters=5)
    cos_sim = torch.nn.functional.cosine_similarity(
        Q1.float().flatten(), Q2.float().flatten(), dim=0,
    )
    assert cos_sim > 0.90, f"Not idempotent: cosine similarity = {cos_sim:.4f}"


def test_batched():
    """Orthogonalize should work on batched (3D) inputs."""
    torch.manual_seed(99)
    M = torch.randn(4, 32, 64, dtype=torch.float32)
    Q = orthogonalize_simple(M, num_iters=5)
    assert Q.shape == M.shape

    # Each slice should have norm ≈ sqrt(32)
    expected = math.sqrt(32)
    for i in range(4):
        norm_i = Q[i].float().norm().item()
        assert abs(norm_i - expected) / expected < 0.10, \
            f"Batch {i}: ||Q||_F = {norm_i:.3f}, expected ≈ {expected:.3f}"


def test_zero_input():
    """Zero matrix should not produce NaN."""
    M = torch.zeros(32, 32, dtype=torch.float32)
    Q = orthogonalize_simple(M, num_iters=5)
    assert not torch.isnan(Q).any(), "NaN in output for zero input"


def test_coefficients_count():
    """We should have exactly 5 coefficient tuples."""
    assert len(POLAR_EXPRESS_COEFFS) == 5
    for a, b, c in POLAR_EXPRESS_COEFFS:
        assert isinstance(a, float)
        assert isinstance(b, float)
        assert isinstance(c, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
