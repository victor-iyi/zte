"""Tests for the method levers in docs/METHODS.md.

Covers the geometry fix (all-but-the-top + CSLS), rank-percentile / frequency-matched retrieval,
the sharpened contrastive terms (alignment, debiased InfoNCE), the collapse-proof data2vec auxiliary,
the subject-adversary ramp, FiLM subject conditioning (held-out-safe), learned spatial attention, and
the phase-scramble surrogate control.
"""

from __future__ import annotations

import numpy as np
import torch

from zte.config import ModelConfig, ObjectiveConfig
from zte.evaluation import metrics as M
from zte.inference.retrieval import NearestNeighborIndex
from zte.models.embedding import build_model
from zte.models.objectives import alignment_penalty, build_objective, debiased_infonce
from zte.models.spatial import ScalpGeometry, SpatialAttention


def _hubby_clusters(n_groups: int = 6, per: int = 20, dim: int = 32, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A clustered embedding set with a strong shared 'hub' axis (high anisotropy)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_groups, dim))
    emb = np.repeat(centers, per, axis=0) + 0.4 * rng.standard_normal((n_groups * per, dim))
    emb += 3.0  # inject a shared direction -> anisotropy near 1
    gids = np.repeat(np.arange(n_groups), per)
    return emb.astype(np.float32), gids


# --------------------------------------------------------------------------- #
# Tier 1.1 — geometry fix
# --------------------------------------------------------------------------- #
def test_all_but_the_top_removes_shared_axis() -> None:
    """ABTT should collapse the anisotropy injected by a shared common direction."""
    emb, _ = _hubby_clusters()
    assert M.anisotropy(emb) > 0.5
    fixed = M.all_but_the_top(emb, 1)
    assert M.anisotropy(fixed) < 0.2
    assert M.all_but_the_top(emb, 0).shape == emb.shape  # no-op path


def test_csls_index_runs_and_reranks() -> None:
    """CSLS is a valid, self-consistent re-ranking (never crashes, changes hub scores)."""
    emb, gids = _hubby_clusters()
    import pandas as pd

    plain = NearestNeighborIndex(emb, pd.DataFrame({'g': gids}))
    csls = NearestNeighborIndex(emb, pd.DataFrame({'g': gids}), csls=True, csls_k=5)
    idx_p, _ = plain.query(emb[:5], k=3, self_indices=np.arange(5))
    idx_c, sim_c = csls.query(emb[:5], k=3, self_indices=np.arange(5))
    assert idx_c.shape == idx_p.shape == (5, 3)
    assert np.isfinite(sim_c).all()
    assert csls.r_bank is not None and csls.r_bank.shape == (len(emb),)


def test_content_retrieval_rank_percentile_and_matched() -> None:
    """Rank-percentile lands in [0, 1]; matched retrieval pools bins with within-bin chance."""
    emb, gids = _hubby_clusters()
    out = M.content_retrieval(emb, gids, return_ranks=True)
    assert 0.0 <= out['rank_percentile'] <= 1.0
    assert out['median_rank'] >= 1.0
    bins = np.tile(np.repeat(np.arange(3), 7)[:20], 6)[: len(gids)].astype(float)
    matched = M.matched_content_retrieval(emb, gids, bins)
    assert matched['n_bins'] >= 1.0 and 0.0 <= matched['chance_top1'] <= 1.0


# --------------------------------------------------------------------------- #
# Tier 2 — sharpened contrastive terms
# --------------------------------------------------------------------------- #
def test_alignment_and_debiased_are_finite() -> None:
    """The alignment and debiased-InfoNCE helpers return finite scalars."""
    n, d = 12, 8
    center = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    context = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    pos = torch.zeros(n, n, dtype=torch.bool)
    # A positive cycle: every anchor has exactly one positive (matches the real code's has_pos gate,
    # so plain InfoNCE stays finite for the tau_plus->0 comparison).
    pos[torch.arange(n), (torch.arange(n) + 1) % n] = True
    a = alignment_penalty(center, context, pos)
    assert torch.isfinite(a) and a.item() >= 0.0

    logits = center @ context.t() / 0.07
    cand = ~torch.eye(n, dtype=torch.bool)
    loss = debiased_infonce(logits, pos, cand, temperature=0.07, tau_plus=0.1)
    assert torch.isfinite(loss)
    # tau_plus -> 0 approaches plain InfoNCE over the candidates (non-candidates masked, as in the code).
    neg_inf = float('-inf')
    plain = (
        torch.logsumexp(logits.masked_fill(~cand, neg_inf), dim=1)
        - torch.logsumexp(logits.masked_fill(~pos, neg_inf), dim=1)
    ).mean()
    near = debiased_infonce(logits, pos, cand, temperature=0.07, tau_plus=1e-6)
    assert abs(near.item() - plain.item()) < 1e-2


# --------------------------------------------------------------------------- #
# Tier 1.2 / 2.3 — adversary ramp + data2vec auxiliary
# --------------------------------------------------------------------------- #
def test_adversary_ramp_and_data2vec_frozen_target() -> None:
    """The GRL lambda ramps 0->1 and the data2vec auxiliary uses a truly frozen target."""
    obj = ObjectiveConfig(
        name='skipgram',
        subject_adversary_weight=0.1,
        subject_adversary_warmup_ratio=0.5,
        data2vec_aux_weight=0.5,
    )
    mdl = ModelConfig(embed_dim=64, hidden_dim=32, n_layers=1, factored=True, content_dim=32)
    model = build_model(mdl, in_dim=40)
    o = build_objective(obj, model, feature_dim=40)
    o.attach_auxiliary(feature_dim=40)
    assert o.data2vec_head is not None and o.data2vec_proj is not None
    # frozen projection: no trainable params
    assert all(not p.requires_grad for p in o.data2vec_proj.parameters())

    o.set_progress(0, 100)
    assert o._adv_lambda() == 0.0  # noqa: SLF001
    o.set_progress(50, 100)
    assert abs(o._adv_lambda() - 1.0) < 1e-6  # noqa: SLF001 — warmup_ratio*total=50, ramp complete
    o.set_progress(None, None)
    assert o._adv_lambda() == 1.0  # noqa: SLF001 — eval / legacy -> full strength


# --------------------------------------------------------------------------- #
# Tier 3.1 — FiLM + spatial attention
# --------------------------------------------------------------------------- #
def test_film_is_identity_at_init_and_holdout_safe() -> None:
    """Zero-init FiLM is the identity, so an unseen (held-out) subject id injects no noise."""
    mdl = ModelConfig(embed_dim=64, hidden_dim=32, n_layers=1, subject_film=True, n_subjects=4)
    model = build_model(mdl, in_dim=40).eval()
    batch = {
        'features': torch.randn(3, 5, 40),
        'pad_mask': torch.ones(3, 5, dtype=torch.bool),
        'presence': torch.ones(3, 5, dtype=torch.bool),
        'subject': torch.tensor([0, 1, 3]),  # id 3 stands in for an untrained held-out subject
    }
    with torch.no_grad():
        h_film = model.token_hidden(batch)
        model.subject_film = None
        h_plain = model.token_hidden(batch)
    assert torch.allclose(h_film, h_plain, atol=1e-6)


def test_spatial_attention_preserves_shape() -> None:
    """coords_2d lies in [0, 1]^2 and SpatialAttention preserves the token shape."""
    geo = ScalpGeometry.fibonacci_fallback(16)
    c2 = geo.coords_2d
    assert c2.shape == (16, 2) and c2.min() >= 0.0 and c2.max() <= 1.0
    sa = SpatialAttention(geo, feat_dim=5, n_freqs=4)
    x = torch.randn(3, 16, 5)
    assert sa(x).shape == x.shape

    mdl = ModelConfig(
        embed_dim=64,
        hidden_dim=32,
        n_layers=1,
        spatial_encoding='spatial_attention',
        spatial_attn_freqs=6,
    )
    model = build_model(mdl, in_dim=80, n_channels=10, bp_features_per_channel=8)
    out = model.token_hidden(
        {
            'features': torch.randn(2, 4, 80),
            'pad_mask': torch.ones(2, 4, dtype=torch.bool),
            'presence': torch.ones(2, 4, dtype=torch.bool),
            'subject': torch.zeros(2, dtype=torch.long),
        }
    )
    assert out.shape == (2, 4, 32)


# --------------------------------------------------------------------------- #
# Tier 3.2 — phase-scramble surrogate control
# --------------------------------------------------------------------------- #
def test_phase_scramble_preserves_power_spectrum() -> None:
    """The surrogate keeps each channel's power spectrum but destroys the phase/time structure."""
    from zte.data.features.transforms import phase_scramble

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8, 128)).astype(np.float32)  # (words, channels, time)
    y = phase_scramble(x)
    assert y.shape == x.shape and np.isfinite(y).all()
    px = np.abs(np.fft.rfft(x, axis=-1))
    py = np.abs(np.fft.rfft(y, axis=-1))
    assert np.allclose(px, py, atol=1e-3)  # power spectrum preserved
    assert not np.allclose(x, y)  # the time-domain signal is genuinely scrambled
