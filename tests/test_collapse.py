"""Tests for the anti-collapse / anti-cone geometry: ZCA whitening and the uniformity term."""

from __future__ import annotations

import numpy as np
import torch

from zte.evaluation.metrics import anisotropy, effective_rank, whiten_features
from zte.models.objectives import vicreg_terms


def _cone(n: int = 400, d: int = 64, seed: int = 0) -> np.ndarray:
    """A near-degenerate cone: every vector shares a big common direction plus small noise."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=d)
    return (3.0 * common + 0.3 * rng.normal(size=(n, d))).astype(np.float32)


def test_whitening_kills_the_cone_and_fills_dimensions() -> None:
    x = _cone()
    assert anisotropy(x) > 0.9  # a cone: nearly all vectors point the same way
    w = whiten_features(x)
    # centring removes the shared direction -> anisotropy collapses toward 0
    assert abs(anisotropy(w)) < 0.1
    # whitening equalises variance across the used dimensions -> covariance ~ identity
    cov = np.cov(w.T)
    off = cov - np.diag(np.diag(cov))
    assert np.abs(off).mean() < 0.05
    # effective rank does not fall (a full-rank Gaussian stays near full-rank after whitening)
    g = np.random.default_rng(1).normal(size=(400, 64)).astype(np.float32)
    assert effective_rank(whiten_features(g)) >= effective_rank(g) - 1.0


def test_uniformity_term_spreads_normalised_embeddings() -> None:
    torch.manual_seed(0)
    common = torch.randn(64)
    emb = (3.0 * common + 0.3 * torch.randn(400, 64)).clone().requires_grad_(True)
    opt = torch.optim.Adam([emb], lr=0.05)

    def aniso(x: torch.Tensor) -> float:
        with torch.no_grad():
            u = torch.nn.functional.normalize(x, dim=-1)
            g = u @ u.t()
            return float((g.sum() - g.diag().sum()) / (400 * 399))

    a0 = aniso(emb)
    for _ in range(150):
        opt.zero_grad()
        loss, metrics = vicreg_terms(emb, gamma=1.0, var_weight=0.0, cov_weight=0.0, aniso_weight=1.0)
        assert 'uniformity_loss' in metrics
        loss.backward()
        opt.step()
    assert aniso(emb) < a0 - 0.3  # the uniformity term genuinely spreads the directions


def test_whiten_features_handles_tiny_input() -> None:
    x = np.zeros((1, 8), dtype=np.float32)
    out = whiten_features(x)
    assert out.shape == (1, 8) and np.isfinite(out).all()
