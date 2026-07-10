"""Regression tests for the performance-review improvements.

These cover the fixes that turned the honest negative result into a set of actionable levers:
anti-collapse (VICReg), subject invariance (adversary, cross-subject positives, per-subject
normalisation), the masked-objective repair (the exported head is now trained), leakage-aware
splitting/normalisation, and objective-aware inference routing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from zte.config import (
    DatasetConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainConfig,
    ZTEConfig,
)
from zte.data.dataset import ZuCoDataset
from zte.data.synthetic import generate_synthetic_zuco
from zte.data.torch_dataset import make_dataloader
from zte.data.transforms import FeatureNormalizer
from zte.models.embedding import build_model
from zte.models.heads import SubjectAdversary, gradient_reverse
from zte.models.objectives import build_objective, vicreg_terms


@pytest.fixture(scope='module')
def three_subject_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic tree with three subjects (needed for subject-invariance tests)."""
    out = tmp_path_factory.mktemp('zuco3')
    generate_synthetic_zuco(
        out, subjects=('ZAB', 'ZDM', 'ZPH'), tasks=('SR', 'NR'), n_sentences=10, show_progress=False
    )
    return out


def _dataset(root: Path, tmp: Path, **ds_kwargs: object) -> ZuCoDataset:
    cfg = DatasetConfig(root=str(root), cache_dir=str(tmp / 'cache'), **ds_kwargs)  # type: ignore[arg-type]
    return ZuCoDataset(cfg).build(show_progress=False)


# --------------------------------------------------------------------------- #
# Anti-collapse (VICReg)
# --------------------------------------------------------------------------- #


def test_vicreg_variance_penalises_collapse() -> None:
    """A near-constant batch incurs a large variance-hinge loss; a spread one does not."""
    collapsed = torch.zeros(64, 16) + 0.01 * torch.randn(64, 16)
    spread = torch.randn(64, 16) * 2.0
    loss_c, m_c = vicreg_terms(collapsed, gamma=1.0, var_weight=1.0, cov_weight=0.0)
    loss_s, m_s = vicreg_terms(spread, gamma=1.0, var_weight=1.0, cov_weight=0.0)
    assert float(loss_c) > float(loss_s)
    assert m_c['emb_std'] < m_s['emb_std']


def test_vicreg_covariance_penalises_correlated_dims() -> None:
    """Perfectly correlated dimensions incur a positive covariance penalty."""
    base = torch.randn(128, 1)
    correlated = base.repeat(1, 8)  # all dims identical -> large off-diagonal covariance
    loss, metrics = vicreg_terms(correlated, gamma=1.0, var_weight=0.0, cov_weight=1.0)
    assert float(loss) > 0.0
    assert metrics['vicreg_cov'] > 0.0


# --------------------------------------------------------------------------- #
# Subject invariance
# --------------------------------------------------------------------------- #


def test_gradient_reversal_flips_sign() -> None:
    """The gradient-reversal layer is identity forward and sign-flipped backward."""
    x = torch.randn(4, 3, requires_grad=True)
    y = gradient_reverse(x, lambda_=2.0).sum()
    y.backward()
    assert torch.allclose(x.grad, torch.full_like(x, -2.0))


def test_subject_adversary_shapes() -> None:
    """The adversary maps hiddens to per-subject logits."""
    adv = SubjectAdversary(in_dim=32, n_subjects=5)
    logits = adv(torch.randn(10, 32), lambda_=1.0)
    assert logits.shape == (10, 5)


def test_adversary_built_only_when_enabled(three_subject_dir: Path, tmp_path: Path) -> None:
    """The subject adversary is present iff its weight is positive."""
    ds = _dataset(three_subject_dir, tmp_path)
    model = build_model(
        ModelConfig(embed_dim=48, hidden_dim=32, n_subjects=3), in_dim=ds.features.shape[1]
    )
    off = build_objective(ObjectiveConfig(name='skipgram'), model)
    on = build_objective(ObjectiveConfig(name='skipgram', subject_adversary_weight=0.5), model)
    assert off.subject_adversary is None
    assert on.subject_adversary is not None


def test_cross_subject_positives_run(three_subject_dir: Path, tmp_path: Path) -> None:
    """Skip-gram with cross-subject positives produces a finite loss over a stimulus-grouped batch."""
    ds = _dataset(three_subject_dir, tmp_path)
    model = build_model(
        ModelConfig(embed_dim=48, hidden_dim=32, n_subjects=3), in_dim=ds.features.shape[1]
    )
    obj = build_objective(ObjectiveConfig(name='skipgram', cross_subject_positives=True), model)
    sp = ds.split('by_sentence', val_fraction=0.1, test_fraction=0.1, seed=0)
    loader = make_dataloader(ds.to_torch(split=sp['train']), batch_size=8, group_by_stimulus=True)
    loss, metrics = obj.compute(model, next(iter(loader)))
    assert torch.isfinite(loss)
    assert metrics['cross_subject'] == 1.0


# --------------------------------------------------------------------------- #
# Masked objective repair
# --------------------------------------------------------------------------- #


def test_masked_latent_trains_projection_head(three_subject_dir: Path, tmp_path: Path) -> None:
    """The exported 768-d projection head now receives a gradient under masked/latent training."""
    ds = _dataset(three_subject_dir, tmp_path)
    model = build_model(
        ModelConfig(embed_dim=48, hidden_dim=32, n_subjects=3), in_dim=ds.features.shape[1]
    )
    obj = build_objective(
        ObjectiveConfig(name='masked', masked_target='latent'),
        model,
        feature_dim=ds.features.shape[1],
    )
    sp = ds.split('by_sentence', val_fraction=0.1, test_fraction=0.1, seed=0)
    loader = make_dataloader(ds.to_torch(split=sp['train']), batch_size=8)
    loss, _ = obj.compute(model, next(iter(loader)))
    loss.backward()
    grad = sum(
        float(p.grad.abs().sum()) for p in model.projection.parameters() if p.grad is not None
    )
    assert grad > 0.0


def test_ema_decay_ramps() -> None:
    """`post_step` ramps the teacher decay from `ema_decay` toward `ema_decay_end`."""
    cfg = ModelConfig(embed_dim=48, hidden_dim=32, n_subjects=3)
    model = build_model(cfg, in_dim=64)
    obj = build_objective(
        ObjectiveConfig(name='masked', masked_target='latent', ema_decay=0.9, ema_decay_end=0.99),
        model,
        feature_dim=64,
    )
    # Move a parameter so an EMA step is observable, then compare early vs late decay effect.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)
    before = next(iter(obj.teacher.module.parameters())).clone()
    obj.post_step(model, step=0, total_steps=100)  # decay ~0.9 -> moves a lot
    early_delta = (next(iter(obj.teacher.module.parameters())) - before).abs().sum()
    # A late step uses decay ~0.99 -> moves less for the same student/teacher gap.
    with torch.no_grad():
        for tp in obj.teacher.module.parameters():
            tp.zero_()
        for p in model.parameters():
            p.zero_().add_(0.5)
    before2 = next(iter(obj.teacher.module.parameters())).clone()
    obj.post_step(model, step=99, total_steps=100)
    late_delta = (next(iter(obj.teacher.module.parameters())) - before2).abs().sum()
    assert float(early_delta) > float(late_delta)


# --------------------------------------------------------------------------- #
# Leakage-aware splitting / normalisation
# --------------------------------------------------------------------------- #


def test_by_stimulus_split_is_text_disjoint(three_subject_dir: Path, tmp_path: Path) -> None:
    """`by_stimulus` never lets the same sentence text span train and test."""
    ds = _dataset(three_subject_dir, tmp_path)
    sp = ds.split('by_stimulus', val_fraction=0.2, test_fraction=0.2, seed=1)
    keys = lambda idx: set(ds.words.iloc[idx]['stimulus_key'])
    assert keys(sp['train']).isdisjoint(keys(sp['test']))
    assert keys(sp['train']).isdisjoint(keys(sp['val']))


def test_refit_normalizer_uses_train_only(three_subject_dir: Path, tmp_path: Path) -> None:
    """Re-fitting on the train split changes the features and the stored statistics."""
    ds = _dataset(three_subject_dir, tmp_path, normalize='zscore_channel')
    sp = ds.split('by_sentence', val_fraction=0.1, test_fraction=0.1, seed=0)
    before = ds.features.copy()
    ds.refit_normalizer(sp['train'])
    assert not np.allclose(before, ds.features)


def test_refit_normalizer_noop_when_fit_all(three_subject_dir: Path, tmp_path: Path) -> None:
    """`normalizer_fit='all'` keeps the legacy whole-dataset fit (refit is a no-op)."""
    ds = _dataset(three_subject_dir, tmp_path, normalize='zscore_channel', normalizer_fit='all')
    sp = ds.split('by_sentence', val_fraction=0.1, test_fraction=0.1, seed=0)
    before = ds.features.copy()
    ds.refit_normalizer(sp['train'])
    assert np.allclose(before, ds.features)


def test_per_subject_normalizer_state_roundtrip() -> None:
    """The per-subject normaliser serialises and re-applies its per-subject statistics."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 5)).astype(np.float32)
    subjects = np.array(['A'] * 15 + ['B'] * 15)
    norm = FeatureNormalizer('zscore_subject').fit(x, subjects=subjects)
    out = norm.transform(x, subjects=subjects)
    restored = FeatureNormalizer.from_state(norm.state)
    assert np.allclose(out, restored.transform(x, subjects=subjects))
    # Unknown subject falls back to the global pooled stats rather than crashing.
    unknown = restored.transform(x[:2], subjects=np.array(['Z', 'Z']))
    assert np.isfinite(unknown).all()


# --------------------------------------------------------------------------- #
# Objective-aware inference routing
# --------------------------------------------------------------------------- #


def test_embed_sentence_routes_per_objective(three_subject_dir: Path, tmp_path: Path) -> None:
    """skip-gram sentence embedding skips the transformer; masked/cpc route through it."""
    ds = _dataset(three_subject_dir, tmp_path)
    model = build_model(
        ModelConfig(embed_dim=48, hidden_dim=32, n_subjects=3), in_dim=ds.features.shape[1]
    )
    loader = make_dataloader(ds.to_torch(), batch_size=6, shuffle=False)
    batch = next(iter(loader))
    sg = model.embed_sentence(batch, objective='skipgram')
    ms = model.embed_sentence(batch, objective='masked')
    assert sg.shape == ms.shape == (batch['subject'].shape[0], 48)
    # Different routing (transformer skipped vs applied) yields different vectors.
    assert not torch.allclose(sg, ms)


def test_full_run_with_all_improvements(three_subject_dir: Path, tmp_path: Path) -> None:
    """End-to-end tiny run with VICReg + adversary + cross-subject positives + per-subject norm."""
    cfg = ZTEConfig(
        dataset=DatasetConfig(
            root=str(three_subject_dir),
            cache_dir=str(tmp_path / 'cache'),
            normalize='zscore_subject',
        ),
        model=ModelConfig(
            embed_dim=48, hidden_dim=32, n_layers=2, projection_hidden=32, n_subjects=3
        ),
        objective=ObjectiveConfig(
            name='skipgram',
            variance_weight=1.0,
            covariance_weight=1.0,
            subject_adversary_weight=0.3,
            cross_subject_positives=True,
        ),
        train=TrainConfig(
            epochs=1, batch_size=8, device='cpu', precision='fp32', tensorboard=False
        ),
    )
    from zte.training.pipeline import run_training

    ds = ZuCoDataset(cfg.dataset).build(show_progress=False)
    art = run_training(cfg, ds)
    assert np.isfinite(art.history['train_loss'][-1])
