"""Contrastive and Optimal-Transport losses for EEG-OT-CLIP alignment.

Implements the composite objective described in the architecture docs:

.. math::

    \\mathcal{L} = \\lambda_1 \\mathcal{L}_{\\mathrm{InfoNCE}}
                 + \\lambda_2 \\mathcal{L}_{\\mathrm{OT}}

Both losses expect L2-normalised batch embeddings ``(N, D)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def info_nce_loss(a: Tensor, b: Tensor, temperature: float = 0.07) -> Tensor:
    """Symmetric InfoNCE between paired embeddings ``(a_i, b_i)``.

    Args:
        a: L2-normalised embeddings ``(N, D)`` (e.g. projected EEG).
        b: L2-normalised embeddings ``(N, D)`` (e.g. projected text), paired with ``a``.
        temperature: Softmax temperature (smaller → sharper).

    Returns:
        Scalar loss ``(CE(a→b) + CE(b→a)) / 2``.
    """
    if a.shape != b.shape:
        raise ValueError(f'a and b must share shape; got {tuple(a.shape)} vs {tuple(b.shape)}')
    if a.ndim != 2:
        raise ValueError(f'expected (N, D) tensors; got shape {tuple(a.shape)}')
    n = a.shape[0]
    if n == 0:
        return a.new_zeros(())
    logits = (a @ b.T) / temperature
    labels = torch.arange(n, device=a.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.T, labels)
    return (loss_ab + loss_ba) * 0.5


def sinkhorn_ot_loss(
    a: Tensor,
    b: Tensor,
    epsilon: float = 0.05,
    n_iters: int = 20,
) -> Tensor:
    """Entropic OT cost between two batches via log-domain Sinkhorn.

    Cost is ``1 - cosine`` (valid when ``a`` / ``b`` are L2-normalised). Sinkhorn
    iterations produce a soft transport plan ``P``; the loss is ``(P * cost).sum()``.

    Args:
        a: L2-normalised embeddings ``(N, D)``.
        b: L2-normalised embeddings ``(M, D)`` (typically ``M == N``).
        epsilon: Entropic regularisation strength.
        n_iters: Number of Sinkhorn scaling iterations.

    Returns:
        Scalar differentiable OT cost.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f'expected 2-D embeddings; got {tuple(a.shape)}, {tuple(b.shape)}')
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return a.new_zeros(())

    # Cosine similarity for unit vectors; cost in [0, 2].
    cost = 1.0 - (a @ b.T).clamp(-1.0, 1.0)
    # Log-domain kernel: K = exp(-C / eps)
    log_k = -cost / max(epsilon, 1e-8)

    # Uniform marginals in log space.
    log_mu = torch.full((n,), -torch.log(torch.tensor(float(n), device=a.device)), device=a.device)
    log_nu = torch.full((m,), -torch.log(torch.tensor(float(m), device=a.device)), device=a.device)
    log_u = torch.zeros(n, device=a.device, dtype=a.dtype)
    log_v = torch.zeros(m, device=a.device, dtype=a.dtype)

    for _ in range(max(1, n_iters)):
        # u ← mu ⊘ (K v); v ← nu ⊘ (Kᵀ u)  — in log space via logsumexp.
        log_u = log_mu - torch.logsumexp(log_k + log_v[None, :], dim=1)
        log_v = log_nu - torch.logsumexp(log_k.T + log_u[None, :], dim=1)

    # Transport plan P_ij = u_i K_ij v_j
    log_p = log_u[:, None] + log_k + log_v[None, :]
    plan = torch.exp(log_p)
    return (plan * cost).sum()
