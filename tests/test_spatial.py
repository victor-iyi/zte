"""Tests for electrode spatial (spherical-harmonic) positional encoding.

These cover both the mathematics (orthonormality, known values, the addition theorem, rotation behaviour) and the
integration into the ZTE frontends (shape preservation, opt-in activation, checkpoint round-trip).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import eval_legendre

from zte.config import ModelConfig
from zte.data.dataset import ZuCoDataset
from zte.data.torch_dataset import collate_sentences
from zte.models.embedding import build_model
from zte.models.spatial import (
    ScalpGeometry,
    SpatialChannelMixer,
    SphericalHarmonicEncoding,
    degree_of_column,
    n_harmonics,
    real_spherical_harmonics,
    resolve_geometry,
)


# --------------------------------------------------------------------------- #
# Mathematics
# --------------------------------------------------------------------------- #
def _fibonacci_sphere(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns near-uniform `(theta, phi)` samples on the full sphere for numerical integration."""
    idx = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (idx + 0.5) / n
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = (np.pi * (3.0 - np.sqrt(5.0)) * idx) % (2.0 * np.pi)
    return theta, phi


def test_harmonic_count_and_degrees() -> None:
    """The basis has (l_max + 1)^2 columns with the expected per-degree multiplicities."""
    for l_max in range(5):
        assert n_harmonics(l_max) == (l_max + 1) ** 2
        deg = degree_of_column(l_max)
        assert deg.shape[0] == (l_max + 1) ** 2
        for l in range(l_max + 1):
            assert int((deg == l).sum()) == 2 * l + 1


def test_y00_is_constant() -> None:
    """Y_0^0 equals 1 / sqrt(4 pi) everywhere (the sphere's constant mode)."""
    theta, phi = _fibonacci_sphere(64)
    y = real_spherical_harmonics(theta, phi, 0)
    assert y.shape == (64, 1)
    np.testing.assert_allclose(y[:, 0], 1.0 / np.sqrt(4.0 * np.pi), rtol=1e-6)


def test_orthonormality_by_monte_carlo() -> None:
    """The real harmonics are orthonormal under the sphere's uniform measure.

    Uniform samples with weight 4 pi / N approximate the surface integral, so `(4 pi / N) * Y^T Y` -> identity.
    """
    theta, phi = _fibonacci_sphere(40000)
    y = real_spherical_harmonics(theta, phi, 4)  # 25 harmonics
    gram = (4.0 * np.pi / y.shape[0]) * (y.T @ y)
    np.testing.assert_allclose(gram, np.eye(y.shape[1]), atol=2e-2)


def test_addition_theorem() -> None:
    """Per-degree harmonic inner products equal the Legendre kernel of the geodesic angle.

    `sum_m Y_l^m(a) Y_l^m(b) = (2l + 1) / (4 pi) * P_l(cos gamma)`, the spherical addition theorem -- the property that
    makes the encoding a geodesic-distance kernel over electrodes.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(size=3)
    b = rng.normal(size=3)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cos_gamma = float(np.dot(a, b))
    theta = np.arccos(np.clip([a[2], b[2]], -1.0, 1.0))
    phi = np.arctan2([a[1], b[1]], [a[0], b[0]])

    l_max = 6
    y = real_spherical_harmonics(theta, phi, l_max)  # (2, n_harmonics)
    deg = degree_of_column(l_max)
    for l in range(l_max + 1):
        cols = deg == l
        lhs = float(y[0, cols] @ y[1, cols])
        rhs = (2 * l + 1) / (4.0 * np.pi) * eval_legendre(l, cos_gamma)
        assert lhs == pytest.approx(rhs, abs=1e-9)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_from_xyz_projects_to_unit_sphere() -> None:
    """from_xyz recentres and normalises arbitrary coordinates onto the unit sphere."""
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(20, 3)) * 40.0 + np.array([10.0, -5.0, 3.0])  # off-centre, mm-like
    geo = ScalpGeometry.from_xyz(raw)
    np.testing.assert_allclose(np.linalg.norm(geo.xyz, axis=1), 1.0, atol=1e-6)
    assert geo.approximate is False


def test_geodesic_angles_symmetry_and_diagonal() -> None:
    """Geodesic angle matrix is symmetric with a zero diagonal."""
    geo = ScalpGeometry.fibonacci_fallback(30)
    ang = geo.geodesic_angles()
    assert ang.shape == (30, 30)
    np.testing.assert_allclose(np.diag(ang), 0.0, atol=1e-6)
    np.testing.assert_allclose(ang, ang.T, atol=1e-7)


def test_fallback_is_flagged_approximate() -> None:
    """The coordinate-free fallback honestly flags itself approximate."""
    geo = ScalpGeometry.fibonacci_fallback(105)
    assert geo.approximate is True
    assert geo.n_channels == 105


def test_from_csv_xyz_and_spherical(tmp_path: Path) -> None:
    """from_csv reads both x,y,z and theta,phi montages and agrees for the same points."""
    theta = np.array([0.3, 1.1, 2.0])
    phi = np.array([0.0, 1.5, -2.0])
    xyz = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=1)
    xyz_csv = tmp_path / 'xyz.csv'
    xyz_csv.write_text(
        'channel,x,y,z\n' + '\n'.join(f'{i},{p[0]},{p[1]},{p[2]}' for i, p in enumerate(xyz)),
        encoding='utf-8',
    )
    sph_csv = tmp_path / 'sph.csv'
    sph_csv.write_text(
        'channel,theta,phi\n' + '\n'.join(f'{i},{t},{p}' for i, (t, p) in enumerate(zip(theta, phi))),
        encoding='utf-8',
    )
    g_xyz = ScalpGeometry.from_csv(xyz_csv, 3)
    g_sph = ScalpGeometry.from_csv(sph_csv, 3)
    np.testing.assert_allclose(g_xyz.xyz, g_sph.xyz, atol=1e-6)
    assert g_xyz.approximate is False


def test_from_csv_missing_channel_raises(tmp_path: Path) -> None:
    """A montage missing a channel index is rejected."""
    csv = tmp_path / 'bad.csv'
    csv.write_text('channel,x,y,z\n0,1,0,0\n', encoding='utf-8')
    with pytest.raises(ValueError, match='missing channels'):
        ScalpGeometry.from_csv(csv, 3)


def test_resolve_geometry_falls_back_without_coords() -> None:
    """resolve_geometry returns the approximate fallback when no montage is supplied."""
    geo = resolve_geometry(64, montage_csv=None)
    assert geo.approximate is True
    assert geo.n_channels == 64


# --------------------------------------------------------------------------- #
# nn.Module encoding
# --------------------------------------------------------------------------- #
def test_encoding_shape_and_rotation_of_degree_gram() -> None:
    """Encoding has the right shape and rotating the head preserves each degree's Gram block.

    Rotations act within a degree (Wigner-D), so the per-degree Gram matrix `Y_l Y_l^T` over electrodes is
    rotation-invariant -- the defining equivariance property. We check it on the raw harmonics for degree 2.
    """
    geo = ScalpGeometry.fibonacci_fallback(40)
    enc = SphericalHarmonicEncoding(geo, l_max=4, out_dim=16)
    out = enc()
    assert out.shape == (40, 16)

    # Rotate coordinates by a fixed rotation about z, rebuild harmonics, compare degree-2 Gram.
    ang = 0.7
    rot = np.array([[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0], [0.0, 0.0, 1.0]])
    rotated = ScalpGeometry.from_xyz(geo.xyz @ rot.T, normalize=False)
    y0 = geo.spherical_harmonics(4)
    y1 = rotated.spherical_harmonics(4)
    deg = degree_of_column(4)
    for l in range(5):
        cols = deg == l
        g0 = y0[:, cols] @ y0[:, cols].T
        g1 = y1[:, cols] @ y1[:, cols].T
        np.testing.assert_allclose(g0, g1, atol=1e-6)


def test_per_degree_gain_is_learnable() -> None:
    """log_scale is a leaf parameter that receives gradients."""
    geo = ScalpGeometry.fibonacci_fallback(20)
    enc = SphericalHarmonicEncoding(geo, l_max=3, out_dim=8, learnable=True)
    out = enc().sum()
    out.backward()
    assert enc.log_scale.grad is not None
    assert enc.log_scale.requires_grad is True


def test_frozen_gain_has_no_grad() -> None:
    """learnable=False freezes the per-degree gains."""
    geo = ScalpGeometry.fibonacci_fallback(20)
    enc = SphericalHarmonicEncoding(geo, l_max=3, out_dim=8, learnable=False)
    assert enc.log_scale.requires_grad is False


def test_channel_mixer_preserves_shape() -> None:
    """SpatialChannelMixer maps (..., C, D) -> (..., C, D)."""
    geo = ScalpGeometry.fibonacci_fallback(16)
    mixer = SpatialChannelMixer(feat_dim=12, geometry=geo, l_max=3, n_heads=4, mix=True)
    x = torch.randn(5, 7, 16, 12)  # (batch, seq, channels, feat)
    out = mixer(x)
    assert out.shape == x.shape


# --------------------------------------------------------------------------- #
# Model integration
# --------------------------------------------------------------------------- #
def _batch(dataset: ZuCoDataset, n: int = 4) -> dict:
    torch_ds = dataset.to_torch(representation='both')
    samples = [torch_ds[i] for i in range(min(n, len(torch_ds)))]
    return collate_sentences(samples)


def test_band_power_spatial_forward(small_dataset: ZuCoDataset) -> None:
    """A band-power model with spatial encoding runs and adds parameters vs the baseline."""
    batch = _batch(small_dataset)
    in_dim = small_dataset.features.shape[1]
    n_ch = small_dataset.band_power_raw.shape[2]
    n_bp = small_dataset.band_power_raw.shape[1]

    base = build_model(ModelConfig(frontend='band_power_mlp', embed_dim=32, hidden_dim=32), in_dim=in_dim)
    spatial = build_model(
        ModelConfig(
            frontend='band_power_mlp',
            embed_dim=32,
            hidden_dim=32,
            spatial_encoding='spherical_harmonics',
            spatial_harmonic_degree=4,
        ),
        in_dim=in_dim,
        n_channels=n_ch,
        bp_features_per_channel=n_bp,
    )
    out = spatial(batch, contextual=True)
    b, length = batch['pad_mask'].shape
    assert out.shape == (b, length, 32)
    assert sum(p.numel() for p in spatial.parameters()) > sum(p.numel() for p in base.parameters())
    assert spatial.frontend.spatial is not None


def test_raw_conformer_spatial_forward(small_dataset: ZuCoDataset) -> None:
    """A raw-conformer model with spatial encoding consumes (B, L, C, T) windows."""
    batch = _batch(small_dataset)
    c, t = small_dataset.raw_eeg.shape[1], small_dataset.raw_eeg.shape[2]
    model = build_model(
        ModelConfig(
            frontend='raw_conformer',
            embed_dim=32,
            hidden_dim=32,
            conformer_filters=16,
            spatial_encoding='spherical_harmonics',
            spatial_harmonic_degree=3,
        ),
        raw_shape=(c, t),
        n_channels=c,
    )
    out = model(batch, contextual=True)
    assert out.shape[-1] == 32
    assert model.frontend.spatial_mixer is not None


def test_spatial_none_is_unchanged(small_dataset: ZuCoDataset) -> None:
    """spatial_encoding='none' (default) installs no mixer."""
    in_dim = small_dataset.features.shape[1]
    model = build_model(
        ModelConfig(frontend='band_power_mlp', embed_dim=32, hidden_dim=32),
        in_dim=in_dim,
        n_channels=small_dataset.band_power_raw.shape[2],
        bp_features_per_channel=small_dataset.band_power_raw.shape[1],
    )
    assert model.frontend.spatial is None


def test_spatial_checkpoint_roundtrip(small_dataset: ZuCoDataset) -> None:
    """A spatial model's state_dict reloads into a freshly-built spatial model."""
    in_dim = small_dataset.features.shape[1]
    n_ch = small_dataset.band_power_raw.shape[2]
    n_bp = small_dataset.band_power_raw.shape[1]
    cfg = ModelConfig(
        frontend='band_power_mlp',
        embed_dim=32,
        hidden_dim=32,
        spatial_encoding='spherical_harmonics',
        spatial_harmonic_degree=4,
    )
    model = build_model(cfg, in_dim=in_dim, n_channels=n_ch, bp_features_per_channel=n_bp)
    state = model.state_dict()
    clone = build_model(cfg, in_dim=in_dim, n_channels=n_ch, bp_features_per_channel=n_bp)
    clone.load_state_dict(state)
    batch = _batch(small_dataset)
    model.eval()
    clone.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch, contextual=True), clone(batch, contextual=True))
