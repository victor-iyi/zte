"""Spatial positional encoding of EEG electrodes on the scalp -- see docs/SPATIAL_ENCODING.md for the derivation.

Electrodes sit on a sphere, not a line, so the spatial analogue of the sequence encodings in `zte.models.transformer` is the real
spherical-harmonic basis. The harmonics are exact for any coordinates; accuracy lives entirely in `ScalpGeometry`, which flags
`approximate=True` when no real montage is supplied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from zte.logging_utils import get_logger

_LOG = get_logger('models.spatial')


# -- Real spherical harmonics ----------------------------------------------- #
def real_spherical_harmonics(theta: np.ndarray, phi: np.ndarray, l_max: int) -> np.ndarray:
    """Evaluates the real spherical-harmonic basis up to degree `l_max`.

    Columns are the standard tesseral real harmonics (Condon-Shortley phase included), ordered `(l, m)` with `m` ascending from `-l`
    to `l` within each degree -- `(l_max + 1) ** 2` columns in total.

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


# -- Electrode geometry ----------------------------------------------------- #
def _fit_unit_sphere(xyz: np.ndarray) -> np.ndarray:
    """Centres electrode coordinates on the best-fit sphere and projects them onto the unit sphere.

    Least-squares solves `|p - c| = r` for the centre, subtracts it and normalises, so the result lies on `S^2` for any input units or origin.

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

    The channel order must match the channel axis of the EEG tensors, so that harmonic column `c` describes electrode `c`.

    Attributes:
        xyz (np.ndarray): `(n_channels, 3)` unit-norm electrode coordinates (`+x` right, `+y` front, `+z` up by convention).
        labels (tuple[str, ...] | None): Optional electrode labels (e.g. `'E1'`, `'Oz'`) aligned with `xyz`.
        approximate (bool): `True` when the coordinates are the coordinate-free fallback rather than a real montage; downstream maths
            is exact either way, only the coordinates differ.
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

    @property
    def coords_2d(self) -> np.ndarray:
        """Azimuthal-equidistant 2-D scalp projection in `[0, 1] ** 2`, shape `(n_channels, 2)`.

        The standard EEG topomap flattening -- radius is colatitude from the vertex, azimuth is `phi` -- as consumed by `SpatialAttention`
        and `zte.evaluation.plots.scalp_topomap`.

        Returns:
            np.ndarray: `(n_channels, 2)` coordinates in `[0, 1]`, `+y` toward the front of the head.
        """
        r = self.theta / np.pi  # 0 at the vertex, -> 1 at the base of the scalp cap
        x, y = r * np.cos(self.phi), r * np.sin(self.phi)
        return np.stack([(x + 1.0) * 0.5, (y + 1.0) * 0.5], axis=1)

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

        # Index every row by its channel column, lower-casing the header.
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

        # Cartesian columns are used as-is; spherical columns are converted to unit vectors.
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

        A Fibonacci (golden-angle) spiral over the scalp cap: near-uniform and well separated, so the harmonics stay well conditioned, with
        the index running anterior -> posterior to match the EGI channel ordering `zte.data.montage.regions` assumes. Not a real montage --
        use `from_csv` or `from_mne` for geometric accuracy.

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


# -- nn.Module: electrode positional encoding ------------------------------- #
class SphericalHarmonicEncoding(nn.Module):
    """Fixed-geometry, learnable-projection spherical-harmonic positional encoding for electrodes.

    The `(n_channels, n_harmonics)` harmonic matrix is a non-trainable buffer (geometry is data, not a parameter); each degree gets a
    learnable gain `exp(log_scale_l)` shared across its `2l + 1` orders, and a linear map projects to `out_dim`. By the addition theorem
    the per-degree gains select which geodesic scales matter, giving a rotation-invariant electrode code.

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
            learnable (bool): If `True`, per-degree gains are trainable; if `False` they are fixed to 1 (the projection stays trainable).
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

    Each electrode in the input's `(n_channels, feat_dim)` trailing axes is a token; its positional encoding is added exactly as a sequence
    transformer adds sequence position, then `mix` runs one pre-norm multi-head self-attention over the channel axis. Output shape equals
    input shape, so it drops in ahead of any channel-consuming frontend.

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
        grad_checkpoint: bool = False,
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
            grad_checkpoint (bool): Recompute the attention in the backward pass instead of storing it.
        """
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.mix = bool(mix)
        self.grad_checkpoint = bool(grad_checkpoint)
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
            # Every (sentence, word) token becomes its own attention problem, so the q/k/v activations
            # scale with batch x sentence length -- gigabytes before the first backward on a raw window.
            # Recomputing them in the backward pass is numerically identical and frees most of that.
            if self.grad_checkpoint and self.training and flat.requires_grad:
                flat = flat + checkpoint(self._mix, flat, use_reentrant=False)
            else:
                flat = flat + self._mix(flat)
        return flat.reshape(*lead, c, d)

    def _mix(self, flat: torch.Tensor) -> torch.Tensor:
        """Cross-electrode self-attention over the channel axis of `(n_tokens, n_channels, feat_dim)`."""
        h = self.norm(flat)
        attended, _ = self.attn(h, h, h, need_weights=False)
        return attended


class SpatialAttention(nn.Module):
    """Défossez-style learned spatial attention over 2-D electrode coordinates.

    An alternative to `SpatialChannelMixer`: each output electrode is a geometry-derived weighted combination of the inputs,
    `out_o = in_o + sum_c softmax_c(z_o(pos_c)) . in_c`, where `z_o` reads a 2-D Fourier embedding of the flattened scalp coordinate.
    Input and output channel counts are equal, so it drops into a frontend exactly like `SpatialChannelMixer`.

    Attributes:
        n_channels (int): Number of electrodes.
        approximate_geometry (bool): Whether the coordinates were the approximate fallback.
    """

    def __init__(
        self, geometry: ScalpGeometry, feat_dim: int, n_freqs: int = 8, dropout: float = 0.0
    ) -> None:  # pylint: disable=unused-argument
        """Builds the spatial-attention mixer.

        Args:
            geometry (ScalpGeometry): Electrode positions; its channel order defines the channel axis.
            feat_dim (int): Per-electrode feature width (kept for interface parity with the mixer).
            n_freqs (int): Fourier frequencies per axis; `2 * n_freqs ** 2` positional features.
            dropout (float): Dropout applied to the attention matrix.
        """
        super().__init__()
        self.n_channels = geometry.n_channels
        self.approximate_geometry = bool(geometry.approximate)
        coords = torch.from_numpy(geometry.coords_2d.astype(np.float32))  # (C, 2)
        k = torch.arange(1, n_freqs + 1, dtype=torch.float32)
        kx, ky = torch.meshgrid(k, k, indexing='ij')
        freqs = 2.0 * math.pi * torch.stack([kx.reshape(-1), ky.reshape(-1)], dim=1)  # (F, 2)
        self.register_buffer('coords', coords, persistent=True)
        self.register_buffer('freqs', freqs, persistent=True)
        self.attn_map = nn.Linear(2 * freqs.shape[0], self.n_channels)  # z_o(pos_c) read-out
        self.drop = nn.Dropout(dropout)

    def _pos_embed(self) -> torch.Tensor:
        """Returns the `(n_channels, 2 * n_freqs ** 2)` 2-D Fourier embedding of the scalp coordinates."""
        proj = self.coords @ self.freqs.t()  # type: ignore[operator]  # (C, F)
        return torch.cat([proj.cos(), proj.sin()], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mixes electrode features by geometry-derived attention.

        Args:
            x (torch.Tensor): `(..., n_channels, feat_dim)` with arbitrary leading dims.

        Returns:
            torch.Tensor: Same shape as `x`.
        """
        logits = self.attn_map(self._pos_embed().to(x.dtype))  # (C_in, C_out)
        attn = self.drop(torch.softmax(logits, dim=0).t())  # (C_out, C_in), columns sum to 1
        lead = x.shape[:-2]
        c, d = x.shape[-2:]
        flat = x.reshape(-1, c, d)
        mixed = torch.einsum('oc,ncd->nod', attn, flat)  # (N, C_out=C, d)
        return (flat + mixed).reshape(*lead, c, d)
