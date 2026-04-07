"""
Newton-Schulz / Polar Express orthogonalization for Muon-family optimizers.

Computes an approximate polar factor (nearest orthogonal matrix) of a given matrix
using 5 iterations of a polynomial Newton-Schulz method. The polynomial coefficients
are from the Polar Express paper (Amsel et al., 2025): https://arxiv.org/abs/2505.16932

The key property: for M = U S V^T (SVD), orthogonalize(M) ≈ U V^T.
After orthogonalization, ||Q||_F ≈ sqrt(min(m, n)) for full-rank matrices.
"""

import torch
from torch import Tensor

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

_compile = getattr(torch, "compile", None)
_maybe_compile = (
    _compile(dynamic=False, fullgraph=True)
    if _compile is not None
    else lambda fn: fn
)


@_maybe_compile
def orthogonalize(M: Tensor, num_iters: int = 5) -> Tensor:
    """
    Compute the approximate polar factor of M via Newton-Schulz iteration.

    For M of shape (..., m, n), returns Q of the same shape where Q ≈ U @ V^T
    (the orthogonal polar factor). Works on batched inputs.

    The method always operates on wide matrices internally (n >= m) for
    efficiency, transposing if needed.

    Args:
        M: Input matrix of shape (..., m, n). Must be 2D or 3D.
        num_iters: Number of Newton-Schulz iterations (default 5).

    Returns:
        Q: Approximate polar factor, same shape as M.
    """
    transposed = M.size(-2) > M.size(-1)
    X = M.bfloat16()
    if transposed:
        X = X.mT

    # Scale so spectral norm <= 1 (required for convergence)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    X = X.contiguous()

    # Allocate work buffers
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    matmul_fn = torch.bmm if X.ndim > 2 else torch.mm
    addmm_fn = torch.baddbmm if X.ndim > 2 else torch.addmm

    for a, b, c in POLAR_EXPRESS_COEFFS[:num_iters]:
        # A = X @ X^T (small square matrix)
        if X.ndim > 2:
            torch.bmm(X, X.mT, out=A)
        else:
            torch.mm(X, X.mT, out=A)

        # B = b*A + c*(A@A)
        if X.ndim > 2:
            torch.baddbmm(A, A, A, beta=b, alpha=c, out=B)
        else:
            torch.addmm(A, A, A, beta=b, alpha=c, out=B)

        # C = a*X + B @ X
        addmm_fn(X, B, X, beta=a, out=C)
        X, C = C, X

    if transposed:
        X = X.mT
    return X


def orthogonalize_simple(M: Tensor, num_iters: int = 5) -> Tensor:
    """
    Non-compiled version for debugging / testing. Same algorithm as orthogonalize()
    but without @torch.compile, making it easier to inspect intermediates.
    """
    transposed = M.size(-2) > M.size(-1)
    X = M.bfloat16()
    if transposed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)

    for a, b, c in POLAR_EXPRESS_COEFFS[:num_iters]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X
