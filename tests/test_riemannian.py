"""Tests for Unit E: Riemannian per-subject covariance whitening."""

from __future__ import annotations

import numpy as np

from zte.data.transforms import FeatureNormalizer


def _two_subject_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate two subject data."""
    rng = np.random.default_rng(seed)
    d = 12
    # Two subjects with different covariance structure (the "forward-model fingerprint").
    a = rng.normal(size=(200, d)) @ rng.normal(size=(d, d)) + 3.0
    b = rng.normal(size=(200, d)) @ rng.normal(size=(d, d)) - 2.0
    x = np.vstack([a, b]).astype(np.float32)
    subjects = np.array(['A'] * 200 + ['B'] * 200)
    return x, subjects


def _identity_gap(cov: np.ndarray) -> float:
    """Compute the mean absolute deviation of a covariance from the identity."""
    return float(np.abs(cov - np.eye(len(cov))).mean())


def test_riemannian_whitens_each_subject_toward_identity() -> None:
    """Test that Riemannian whitening each subject toward the identity."""
    x, subjects = _two_subject_data()
    norm = FeatureNormalizer(mode='riemannian')
    y = norm.fit_transform(x, subjects=subjects)
    for code in ('A', 'B'):
        raw_gap = _identity_gap(np.cov(x[subjects == code], rowvar=False))
        white_gap = _identity_gap(np.cov(y[subjects == code], rowvar=False))
        assert white_gap < 0.2 * raw_gap  # whitening massively reduces the fingerprint
        off = np.cov(y[subjects == code], rowvar=False)
        off = off - np.diag(np.diag(off))
        assert np.abs(off).mean() < 0.1  # decorrelated (shrinkage leaves variance < 1)


def test_riemannian_inverse_roundtrip() -> None:
    """Test that Riemannian inverse roundtrip."""
    x, subjects = _two_subject_data(1)
    norm = FeatureNormalizer(mode='riemannian')
    y = norm.fit_transform(x, subjects=subjects)
    recon = norm.inverse_transform(y, subjects=subjects)
    assert np.allclose(recon, x, atol=1e-2)


def test_riemannian_unknown_subject_uses_global_fallback() -> None:
    """Test that Riemannian unknown subject uses global fallback."""
    x, subjects = _two_subject_data(2)
    norm = FeatureNormalizer(mode='riemannian').fit(x, subjects=subjects)
    # A never-seen subject falls back to the global map (finite, no crash).
    y = norm.transform(x[:10], subjects=np.array(['UNSEEN'] * 10))
    assert np.isfinite(y).all()


def test_riemannian_state_roundtrip() -> None:
    """Test that Riemannian state roundtrip."""
    x, subjects = _two_subject_data(3)
    norm = FeatureNormalizer(mode='riemannian').fit(x, subjects=subjects)
    restored = FeatureNormalizer.from_state(norm.state)
    y1 = norm.transform(x, subjects=subjects)
    y2 = restored.transform(x, subjects=subjects)
    assert np.allclose(y1, y2, atol=1e-5)


def test_calibrate_new_subject_riemannian() -> None:
    """Zero-shot new-brain calibration: a stranger's baseline whitens them into the shared space."""
    x, subjects = _two_subject_data(4)
    norm = FeatureNormalizer(mode='riemannian').fit(x, subjects=subjects)
    # A genuinely new subject with its own covariance structure.
    rng = np.random.default_rng(9)
    new = (rng.normal(size=(200, 12)) @ rng.normal(size=(12, 12)) + 7.0).astype(np.float32)
    before = _identity_gap(
        np.cov(norm.transform(new, subjects=np.array(['NEW'] * 200)), rowvar=False)
    )
    norm.calibrate_subject(new, 'NEW')  # unlabelled baseline
    after = _identity_gap(
        np.cov(norm.transform(new, subjects=np.array(['NEW'] * 200)), rowvar=False)
    )
    assert after < 0.3 * before  # calibration whitens the new brain into the shared frame


def test_calibrate_new_subject_zscore() -> None:
    """Test that calibrate new subject zscore."""
    x, subjects = _two_subject_data(5)
    norm = FeatureNormalizer(mode='zscore_subject').fit(x, subjects=subjects)
    new = (np.random.default_rng(3).normal(size=(100, 12)) * 5 + 20).astype(np.float32)
    norm.calibrate_subject(new, 'NEW')
    y = norm.transform(new, subjects=np.array(['NEW'] * 100))
    assert np.abs(y.mean(axis=0)).mean() < 0.2 and np.abs(y.std(axis=0) - 1.0).mean() < 0.2


def test_riemannian_falls_back_when_too_wide() -> None:
    """Test that Riemannian falls back when too wide."""
    norm = FeatureNormalizer(mode='riemannian')
    norm._RIEMANN_MAX_DIM = 4  # force the guard
    x = np.random.default_rng(0).normal(size=(50, 8)).astype(np.float32)
    subjects = np.array(['A'] * 25 + ['B'] * 25)
    norm.fit(x, subjects=subjects)
    assert norm.mode == 'zscore_subject'  # degraded gracefully
