"""Spatial positional encoding of EEG electrodes on the scalp.

The sequence positional encodings in `zte.models.transformer` (RoPE, sinusoidal, ALiBi) encode *when* a word occurred -- position along a
one-dimensional token axis. They are the Fourier basis of the **line** under translation, which is exactly why relative distance falls out of a dot
product. EEG electrodes are not arranged along a line: they sit at fixed points on the **scalp**, a (topological) sphere. Their arbitrary channel
*index* carries no geometry, so a model that only sees the index has to memorise which column is Oz and which is Fp1 rather than being told that Oz sits
at the back and Fp1 at the front.

This module supplies the mathematically correct spatial analogue. The natural generalisation of sinusoids-on-a-line to a sphere is the family of
**real spherical harmonics** `Y_l^m(theta, phi)` -- the eigenfunctions of the Laplace-Beltrami operator on `S^2` and a complete orthonormal basis for
square-integrable functions on the sphere. They are to the sphere what `sin`/`cos` are to the line:

- **Multi-resolution frequency ladder.** Degree `l` is angular frequency. `l = 0` is constant; `l = 1` are the three dipolar left-right /
  front-back / up-down gradients; higher `l` resolves progressively finer scalp patterns. This mirrors the geometric frequency ladder
  `10000**(2i/d)` of sinusoidal position encoding, but on the sphere.
- **Rotation structure.** Rotating the head (an element of `SO(3)`) mixes harmonics *within* a degree via the Wigner-D matrices and never across
  degrees -- the spherical analogue of "translation acts by a phase shift", which is what makes the encoding a faithful, equivariant position code.
- **Geodesic locality (addition theorem).** `sum_m Y_l^m(a) Y_l^m(b) = (2l + 1) / (4 pi) * P_l(cos gamma)` where `gamma` is the great-circle angle
  between electrodes `a` and `b`. So the inner product between two electrodes' harmonic feature vectors is a function of the geodesic distance
  between them: nearby electrodes get similar encodings, and a per-degree weighting yields a *learnable, rotation-invariant kernel of scalp distance*.

The harmonics are exact for any electrode coordinates. Accuracy therefore lives entirely in the coordinates (`ScalpGeometry`): supply a real montage
(`channel,x,y,z` CSV, or MNE's `GSN-HydroCel-128`) for geometric truth; the coordinate-free fallback is smooth and well separated but is flagged
`approximate=True`, exactly like `zte.data.regions.RegionMap`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zte.logging_utils import get_logger

_LOG = get_logger('models.spatial')


# --------------------------------------------------------------------------- #
# Real spherical harmonics
# --------------------------------------------------------------------------- #
def real_spherical_harmonics(theta: np.ndarray, phi: np.ndarray, l_max: int) -> np.ndarray:
    """Evaluates the real spherical-harmonic basis up to degree `l_max`.

    The returned columns are the orthonormal real harmonics `Y_l^m` for every degree `0 <= l <= l_max` and order `-l <= m <= l`, ordered
    `(l, m)` with `m` ascending from `-l` to `l` within each degree. There are `(l_max + 1) ** 2` columns in total.

    The real basis is the standard tesseral convention (Condon-Shortley phase included, so it matches the complex `Y_l^m` from which it is derived):

    - `m = 0`: `Re(Y_l^0)`
    - `m > 0`: `sqrt(2) * (-1) ** m * Re(Y_l^m)`
    - `m < 0`: `sqrt(2) * (-1) ** m * Im(Y_l^{|m|})`

    Args:
        theta (np.ndarray): Polar angle / colatitude in radians (`0` at `+z`, `pi` at `-z`), shape `(n_points,)`.
        phi (np.ndarray): Azimuthal angle in radians (`0` along `+x`), shape `(n_points,)`.
        l_max (int): Maximum harmonic degree (inclusive). Must be `>= 0`.

    Returns:
        np.ndarray: `(n_points, (l_max + 1) ** 2)` float64 matrix of real harmonics.

    Raises:
        ValueError: If `l_max` is negative.
    """
    if l_max < 0:
        raise ValueError(f'l_max must be >= 0, got {l_max}.')
    theta = np.asarray(theta, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()
    n = theta.shape[0]
    cols: list[np.ndarray] = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            cols.append(_real_ylm(l, m, theta, phi))
    return np.stack(cols, axis=1) if cols else np.zeros((n, 0), dtype=np.float64)


def _real_ylm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Returns a single real harmonic column `Y_l^m` evaluated at `(theta, phi)`."""
    from scipy import special  # local import: heavy, and only needed to build the basis

    # scipy renamed sph_harm -> sph_harm_y (signature (n, m, theta, phi)) around 1.15.
    sph = getattr(special, 'sph_harm_y', None)
    if sph is not None:
        complex_y = sph(l, abs(m), theta, phi)
    else:  # pragma: no cover - legacy scipy
        complex_y = special.sph_harm(abs(m), l, phi, theta)  # (m, n, phi, theta) order
    if m == 0:
        return np.real(complex_y)
    sign = (-1.0) ** m
    if m > 0:
        return math.sqrt(2.0) * sign * np.real(complex_y)
    return math.sqrt(2.0) * sign * np.imag(complex_y)


def n_harmonics(l_max: int) -> int:
    """Returns the number of real harmonics up to degree `l_max` (`(l_max + 1) ** 2`)."""
    return (l_max + 1) ** 2


def degree_of_column(l_max: int) -> np.ndarray:
    """Returns the harmonic degree `l` of each column produced by `real_spherical_harmonics`.

    Args:
        l_max (int): Maximum harmonic degree used to build the basis.

    Returns:
        np.ndarray: `((l_max + 1) ** 2,)` int array; entry `k` is the degree of column `k`.
    """
    return np.concatenate([np.full(2 * l + 1, l, dtype=int) for l in range(l_max + 1)])


# --------------------------------------------------------------------------- #
# Electrode geometry
# --------------------------------------------------------------------------- #
def _fit_unit_sphere(xyz: np.ndarray) -> np.ndarray:
    """Centres electrode coordinates on the best-fit sphere and projects them onto the unit sphere.

    Solves the linear least-squares sphere fit (`|p - c| = r`) for the centre `c`, subtracts it, and normalises every point to unit length so the
    result lies on `S^2` regardless of the input units or origin.

    Args:
        xyz (np.ndarray): `(n_channels, 3)` electrode coordinates in any units/origin.

    Returns:
        np.ndarray: `(n_channels, 3)` unit-norm coordinates.
    """
    p = np.asarray(xyz, dtype=np.float64)
    # Linear sphere fit: |p|^2 = 2 p.c + (r^2 - |c|^2)  ->  A [cx, cy, cz, k]^T = b.
    a = np.concatenate([2.0 * p, np.ones((p.shape[0], 1))], axis=1)
    b = (p**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    centre = sol[:3]
    centred = p - centre[None, :]
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return centred / norms


@dataclass(slots=True)
class ScalpGeometry:
    """Electrode positions on the unit sphere, plus their spherical coordinates.

    The channel order **must** match the channel axis of the EEG tensors (`(n_words, n_bp_features, n_channels)` band power or
    `(n_words, n_channels, time_steps)` raw), so that harmonic column `c` describes electrode `c`.

    Attributes:
        xyz (np.ndarray): `(n_channels, 3)` unit-norm electrode coordinates (`+x` right, `+y` front, `+z` up by convention).
        labels (tuple[str, ...] | None): Optional electrode labels (e.g. `'E1'`, `'Oz'`) aligned with `xyz`.
        approximate (bool): `True` when the coordinates are the coordinate-free fallback rather than a real montage. Mirrors
            `zte.data.regions.RegionMap`: every downstream computation is exact for whatever coordinates are supplied; only this
            flag records whether the *coordinates themselves* are a montage or a placeholder.
    """

    xyz: np.ndarray
    labels: tuple[str, ...] | None = None
    approximate: bool = False

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f'xyz must be (n_channels, 3), got {xyz.shape}.')
        object.__setattr__(self, 'xyz', xyz)

    @property
    def n_channels(self) -> int:
        """Number of electrodes."""
        return int(self.xyz.shape[0])

    @property
    def theta(self) -> np.ndarray:
        """Polar angle / colatitude in radians (`arccos z`), shape `(n_channels,)`."""
        return np.arccos(np.clip(self.xyz[:, 2], -1.0, 1.0))

    @property
    def phi(self) -> np.ndarray:
        """Azimuthal angle in radians (`atan2(y, x)`), shape `(n_channels,)`."""
        return np.arctan2(self.xyz[:, 1], self.xyz[:, 0])

    def geodesic_angles(self) -> np.ndarray:
        """Returns the `(n_channels, n_channels)` great-circle angle (radians) between every electrode pair."""
        gram = np.clip(self.xyz @ self.xyz.T, -1.0, 1.0)
        return np.arccos(gram)

    def spherical_harmonics(self, l_max: int) -> np.ndarray:
        """Evaluates the real harmonic basis up to `l_max` at every electrode.

        Args:
            l_max (int): Maximum harmonic degree.

        Returns:
            np.ndarray: `(n_channels, (l_max + 1) ** 2)` harmonic feature matrix.
        """
        return real_spherical_harmonics(self.theta, self.phi, l_max)

    # -- constructors ------------------------------------------------------- #
    @classmethod
    def from_xyz(
        cls,
        xyz: np.ndarray,
        labels: tuple[str, ...] | None = None,
        approximate: bool = False,
        normalize: bool = True,
    ) -> ScalpGeometry:
        """Builds a geometry from raw 3-D coordinates (any units/origin).

        Args:
            xyz (np.ndarray): `(n_channels, 3)` coordinates.
            labels (tuple[str, ...] | None): Optional electrode labels.
            approximate (bool): Whether these coordinates are a placeholder rather than a real montage.
            normalize (bool): Fit-and-project onto the unit sphere (recommended for real montages in mm).

        Returns:
            ScalpGeometry: The constructed geometry.
        """
        pts = _fit_unit_sphere(xyz) if normalize else np.asarray(xyz, dtype=np.float64)
        return cls(xyz=pts, labels=labels, approximate=approximate)

    @classmethod
    def from_csv(cls, path: str | Path, n_channels: int) -> ScalpGeometry:
        """Loads exact electrode coordinates from a montage CSV.

        The CSV must have a header and either Cartesian columns `channel,x,y,z` or spherical columns `channel,theta,phi` (radians, `theta` the
        colatitude). `channel` is the 0-based index into the EEG channel axis. Every channel in `range(n_channels)` must be present.

        Args:
            path (str | Path): Path to the montage CSV.
            n_channels (int): Expected channel count (rows must cover `0 .. n_channels - 1`).

        Returns:
            ScalpGeometry: An exact (`approximate=False`) geometry.

        Raises:
            ValueError: If required columns are missing or a channel index is absent/out of range.
        """
        import csv

        rows: dict[int, dict[str, str]] = {}
        with Path(path).open(encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            fields = {f.strip().lower() for f in (reader.fieldnames or [])}
            for row in reader:
                low = {k.strip().lower(): v for k, v in row.items()}
                rows[int(low['channel'])] = low
        missing = [c for c in range(n_channels) if c not in rows]
        if missing:
            raise ValueError(
                f'Montage CSV {path} is missing channels {missing[:8]}... (of {n_channels}).'
            )
        labels: list[str] = []
        if {'x', 'y', 'z'} <= fields:
            xyz = np.array(
                [
                    [float(rows[c]['x']), float(rows[c]['y']), float(rows[c]['z'])]
                    for c in range(n_channels)
                ]
            )
        elif {'theta', 'phi'} <= fields:
            th = np.array([float(rows[c]['theta']) for c in range(n_channels)])
            ph = np.array([float(rows[c]['phi']) for c in range(n_channels)])
            xyz = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
        else:
            raise ValueError(
                f'Montage CSV {path} needs either x,y,z or theta,phi columns; found {sorted(fields)}.'
            )
        if 'label' in fields:
            labels = [str(rows[c].get('label', '')) for c in range(n_channels)]
        return cls.from_xyz(xyz, labels=tuple(labels) or None, approximate=False, normalize=True)

    @classmethod
    def fibonacci_fallback(cls, n_channels: int) -> ScalpGeometry:
        """Builds a coordinate-free placeholder geometry (flagged `approximate=True`).

        Places `n_channels` points on a spherical cap covering the scalp using a Fibonacci (golden-angle) spiral, which is near-uniform and
        well separated -- so harmonics are well conditioned -- with the index running anterior -> posterior to respect the common EGI channel
        ordering that `zte.data.regions.default_region_map` also assumes. This is **not** a real montage: it exists so the encoding is usable
        without coordinates, and it honestly reports `approximate=True`. Supply `from_csv` or `from_mne` for geometric accuracy.

        Args:
            n_channels (int): Number of electrodes to place.

        Returns:
            ScalpGeometry: An approximate geometry (`approximate=True`).
        """
        idx = np.arange(n_channels, dtype=np.float64)
        # Colatitude sweeps the scalp cap (0 at vertex to ~150 deg), anterior->posterior with index.
        cap = math.radians(150.0)
        theta = cap * (idx + 0.5) / n_channels
        golden = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
        phi = golden * idx
        xyz = np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=1
        )
        _LOG.info(
            'ScalpGeometry: using APPROXIMATE coordinate-free electrode positions '
            '(Fibonacci cap) for %d channels. Supply a montage CSV (channel,x,y,z) or '
            'MNE GSN-HydroCel-128 for geometric accuracy.',
            n_channels,
        )
        return cls(xyz=xyz, labels=None, approximate=True)

    @classmethod
    def from_mne(
        cls, montage: str = 'GSN-HydroCel-128', keep: list[str] | None = None
    ) -> ScalpGeometry:  # pragma: no cover - optional dependency
        """Builds an exact geometry from an MNE standard montage, if MNE is installed.

        Args:
            montage (str): An MNE standard montage name (default the EGI net ZuCo used).
            keep (list[str] | None): Electrode labels to retain, in the exact order of the EEG channel axis. `None` keeps all EEG channels in the
                montage's own order -- which will **not** match the ZuCo 105-channel subset, so pass the retained labels explicitly for real data.

        Returns:
            ScalpGeometry: An exact (`approximate=False`) geometry.

        Raises:
            ImportError: If MNE is not installed.
            KeyError: If a requested label is absent from the montage.
        """
        try:
            import mne  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                'ScalpGeometry.from_mne requires the optional dependency `mne`. '
                'Install it, or export a montage CSV and use ScalpGeometry.from_csv.'
            ) from exc
        mont = mne.channels.make_standard_montage(montage)
        pos = mont.get_positions()['ch_pos']  # dict label -> (3,) metres
        labels = keep if keep is not None else list(pos.keys())
        xyz = np.array([pos[label] for label in labels], dtype=np.float64)
        return cls.from_xyz(xyz, labels=tuple(labels), approximate=False, normalize=True)


def resolve_geometry(n_channels: int, montage_csv: str | Path | None = None) -> ScalpGeometry:
    """Returns the best available electrode geometry for `n_channels`.

    Loads a real montage from `montage_csv` when supplied (and it contains coordinates); otherwise falls back to the approximate Fibonacci cap and
    logs that geometry is not exact.

    Args:
        n_channels (int): Number of EEG channels.
        montage_csv (str | Path | None): Optional montage CSV path (`channel,x,y,z` or `channel,theta,phi`). A region-only
            `channel,region` CSV is ignored here with a warning, since it carries no coordinates.

    Returns:
        ScalpGeometry: A real geometry when coordinates are available, else the approximate fallback.
    """
    if montage_csv is not None:
        try:
            return ScalpGeometry.from_csv(montage_csv, n_channels)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            _LOG.warning(
                'Could not load coordinates from montage_csv=%s (%s); falling back to '
                'approximate electrode geometry.',
                montage_csv,
                exc,
            )
    return ScalpGeometry.fibonacci_fallback(n_channels)


# --------------------------------------------------------------------------- #
# nn.Module: electrode positional encoding
# --------------------------------------------------------------------------- #
class SphericalHarmonicEncoding(nn.Module):
    """Fixed-geometry, learnable-projection spherical-harmonic positional encoding for electrodes.

    Precomputes the real harmonic matrix `Y` of shape `(n_channels, n_harmonics)` from a `ScalpGeometry` and holds it as a non-trainable
    buffer (the geometry is data, not a parameter). Each degree `l` gets a learnable scalar gain `exp(log_scale_l)` (shared across its `2l + 1`
    orders), and a linear map projects the scaled harmonics to `out_dim`. Because the raw harmonic inner product between two electrodes equals a
    per-degree-weighted Legendre kernel of their geodesic angle (the addition theorem), the learnable per-degree gains let the model choose *which
    spatial scales matter* -- a rotation-invariant, geodesic-aware electrode code -- while the projection adapts it to the frontend width.

    Attributes:
        n_channels (int): Number of electrodes encoded.
        l_max (int): Maximum harmonic degree.
        out_dim (int): Width of the produced positional encoding.
        approximate_geometry (bool): Whether the underlying coordinates were the approximate fallback.
    """

    def __init__(
        self,
        geometry: ScalpGeometry,
        l_max: int,
        out_dim: int,
        learnable: bool = True,
    ) -> None:
        """Builds the encoding for a fixed electrode geometry.

        Args:
            geometry (ScalpGeometry): Electrode positions; its channel order defines the encoding's channel order.
            l_max (int): Maximum harmonic degree. Larger resolves finer scalp detail; `(l_max + 1) ** 2` harmonics are used.
            out_dim (int): Output width (typically the frontend's per-channel feature width).
            learnable (bool): If `True`, per-degree gains and the projection are trainable; if `False`, gains are fixed to 1 and only the
                projection is a (still trainable) linear map. Set both off by freezing parameters externally for a purely fixed code.
        """
        super().__init__()
        self.n_channels = geometry.n_channels
        self.l_max = int(l_max)
        self.out_dim = int(out_dim)
        self.approximate_geometry = bool(geometry.approximate)

        harmonics = geometry.spherical_harmonics(self.l_max).astype(np.float32)
        self.register_buffer('harmonics', torch.from_numpy(harmonics), persistent=True)
        degrees = torch.from_numpy(degree_of_column(self.l_max)).long()
        self.register_buffer('degrees', degrees, persistent=True)

        n_deg = self.l_max + 1
        self.log_scale = nn.Parameter(torch.zeros(n_deg), requires_grad=learnable)
        self.proj = nn.Linear(n_harmonics(self.l_max), self.out_dim)

    def forward(self) -> torch.Tensor:
        """Returns the per-channel positional encoding.

        Returns:
            torch.Tensor: `(n_channels, out_dim)` electrode positional encoding on the module's device/dtype.
        """
        gains = torch.exp(self.log_scale)[self.degrees]  # type: ignore[index]  # (n_harmonics,)
        scaled = self.harmonics * gains[None, :]  # type: ignore[operator]  # (n_channels, n_harmonics)
        return self.proj(scaled)

    def extra_repr(self) -> str:
        """String summary for `print(model)`."""
        approx = ', approximate_geometry=True' if self.approximate_geometry else ''
        return f'n_channels={self.n_channels}, l_max={self.l_max}, out_dim={self.out_dim}{approx}'


def _largest_divisor(value: int, target: int) -> int:
    """Returns the largest divisor of `value` that is `<= target` (min 1)."""
    for h in range(min(target, value), 0, -1):
        if value % h == 0:
            return h
    return 1


class SpatialChannelMixer(nn.Module):
    """Adds spherical-harmonic electrode encoding to per-channel features and (optionally) mixes across electrodes.

    Consumes a tensor whose last two axes are `(n_channels, feat_dim)` -- each electrode is a token carrying `feat_dim` features (raw samples, or
    band-power values). It adds the electrode's spherical-harmonic positional encoding (projected to `feat_dim`) to its features, exactly as a
    sequence transformer adds sequence position; then, when `mix` is set, a single pre-norm multi-head self-attention over the channel axis lets each
    electrode attend to the others with full knowledge of scalp geometry (a position-aware spatial filter), with a residual connection. Output shape
    equals input shape, so the mixer drops in ahead of any channel-consuming frontend without changing downstream dimensions.

    Attributes:
        feat_dim (int): Per-electrode feature width (the last axis).
        mix (bool): Whether cross-electrode self-attention is applied on top of the additive encoding.
        approximate_geometry (bool): Whether the electrode coordinates were the approximate fallback.
    """

    def __init__(
        self,
        feat_dim: int,
        geometry: ScalpGeometry,
        l_max: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        learnable: bool = True,
        mix: bool = True,
    ) -> None:
        """Builds the mixer.

        Args:
            feat_dim (int): Per-electrode feature width.
            geometry (ScalpGeometry): Electrode positions (channel order must match the input's channel axis).
            l_max (int): Maximum harmonic degree for the positional encoding.
            n_heads (int): Requested attention heads for the spatial mixing (clamped to a divisor of `feat_dim`).
            dropout (float): Attention dropout.
            learnable (bool): Whether the encoding's per-degree gains/projection are trainable.
            mix (bool): If `True`, add cross-electrode self-attention; if `False`, only add the positional encoding.
        """
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.mix = bool(mix)
        self.approximate_geometry = bool(geometry.approximate)
        self.pos = SphericalHarmonicEncoding(geometry, l_max, feat_dim, learnable=learnable)
        if self.mix:
            self.norm = nn.LayerNorm(feat_dim)
            heads = _largest_divisor(feat_dim, n_heads)
            self.attn = nn.MultiheadAttention(feat_dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Injects electrode geometry into a per-channel feature tensor.

        Args:
            x (torch.Tensor): `(..., n_channels, feat_dim)` with arbitrary leading (batch/sequence) dims.

        Returns:
            torch.Tensor: Same shape as `x`, with electrode positional encoding added and (optionally) spatial self-attention applied.
        """
        lead = x.shape[:-2]
        c, d = x.shape[-2:]
        flat = x.reshape(-1, c, d)
        flat = flat + self.pos().to(flat.dtype)[None]
        if self.mix:
            h = self.norm(flat)
            attended, _ = self.attn(h, h, h, need_weights=False)
            flat = flat + attended
        return flat.reshape(*lead, c, d)
