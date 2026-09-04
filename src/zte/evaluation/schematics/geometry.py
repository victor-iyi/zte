"""The electrode-geometry schematics: the real montage, the harmonic basis and kernel, transport, the adapter."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.contour import ContourSet
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
from scipy.special import eval_legendre

from zte.data.montage.regions import SCALP_REGIONS
from zte.evaluation.schematics._style import (
    DIVERGING_CMAP,
    DOUBLE_COLUMN_IN,
    EEG,
    FILL,
    HEAD_RADIUS_M,
    INK,
    INK_2,
    MUTED,
    RED,
    REGION_CMAP,
    SEQUENTIAL_CMAP,
    SINGLE_COLUMN_IN,
    Axes,
    arrow,
    blank,
    box,
    head,
    montage,
    tensor_slab,
    topomap,
)
from zte.lens.montage import azimuthal_xy
from zte.models.spatial import ScalpGeometry, degree_of_column, real_spherical_harmonics

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# The GSN-105 cap reaches 23 degrees below the equator, so a vertex-centred projection throws 29 electrodes outside
# the scalp circle and mne then paints the field out to their hull. Lowering the projection origin (mne's `sphere`
# z-offset) keeps every electrode inside the outline, where the field clips at the scalp.
_ORIGIN_DROP: Final[float] = 0.45
"""How far below the sphere centre, in head radii, the top-down projection is taken from."""

# Every electrode sits inside the head circle with a small margin, so the ellipse-cut origin and the head outline share
# one radius and a dot never touches the outline.
_HEAD_MARGIN: Final[float] = 0.985
"""Scale on the projected radius so the outermost electrode stays inside the scalp."""

# Highest degree the model uses; the basis figure shows degrees 0-3 because 49 heads cannot be read at column width.
_L_MAX: Final[int] = 6
"""Maximum harmonic degree of the spatial code."""

# Illustrative gains, not a measurement: a low-pass profile is what a learnable kernel of scalp distance looks like
# once the smooth degrees are favoured; every figure that draws gains draws these, so the figures agree.
_DEGREE_GAINS: Final[tuple[float, ...]] = (1.0, 1.2, 1.3, 1.05, 0.7, 0.42, 0.25)
"""Per-degree gains `g_l`, `l = 0 .. 6`, used by every gain glyph."""

_REGION_INDEX: Final[dict[str, int]] = {name: i for i, name in enumerate(SCALP_REGIONS)}
"""Anterior-to-posterior rank of each scalp region."""
_REGION_SHORT: Final[tuple[str, ...]] = ('Fp', 'F', 'FC', 'C', 'CP', 'P', 'PO', 'O')
"""The standard EEG abbreviation of each region, in `SCALP_REGIONS` order."""

_EEG_TINT: Final[str] = '#e6f0fb'
"""Fill for trainable EEG-side blocks."""
_DASH: Final[tuple[float, tuple[float, float]]] = (0.0, (3.0, 2.0))
"""The dash pattern of a generated-parameter path."""

# The signature is 105 per-channel log scales followed by the upper triangle of the 8 x 8 region tangent matrix.
_N_SCALE: Final[int] = 105
_N_TANGENT: Final[int] = 36
_N_SIGNATURE: Final[int] = _N_SCALE + _N_TANGENT


# ---- Geometry shared by the figures ---- #


def _display_xyz(xyz: np.ndarray) -> np.ndarray:
    """Unit coordinates seen from a projection origin lowered by `_ORIGIN_DROP`, so the cap fits the scalp circle."""
    unit = xyz / np.clip(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-8, None)
    shifted = unit + np.array([0.0, 0.0, _ORIGIN_DROP])

    return shifted / np.linalg.norm(shifted, axis=1, keepdims=True)


def _scalp_xy(xyz: np.ndarray) -> np.ndarray:
    """Top-down positions with the scalp circle at radius one, nose up."""
    return azimuthal_xy(_display_xyz(xyz)) * (2.0 / math.pi) * _HEAD_MARGIN


def _mne_xy(xyz: np.ndarray) -> np.ndarray:
    """The same positions in the metres mne draws its topomap in."""
    return _scalp_xy(xyz) * HEAD_RADIUS_M


def _scalp(
    ax: Axes,
    values: np.ndarray,
    xyz: np.ndarray,
    *,
    cmap: Colormap,
    vlim: tuple[float, float],
    contours: int = 4,
    dots: bool = True,
) -> None:
    """A topomap on the lowered projection, mne's outlines, contours and sensor dots thinned to the house weights."""
    # The margin scales the whole cap down slightly; mne's projection scales by the sphere radius, so the margin is
    # applied by lifting the polar angle a little towards the vertex.
    display = _display_xyz(xyz)
    theta = np.arccos(np.clip(display[:, 2], -1.0, 1.0)) * _HEAD_MARGIN
    phi = np.arctan2(display[:, 1], display[:, 0])
    lifted = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=1)
    topomap(ax, values, lifted, cmap=cmap, vlim=vlim, contours=contours, dots=dots)
    for collection in ax.collections:
        if isinstance(collection, ContourSet):
            collection.set_linewidth(0.3)
            collection.set_linestyle('solid')
            collection.set_edgecolor(INK)
            collection.set_alpha(0.45)
        else:
            collection.set_sizes([1.4])
            collection.set_facecolor(INK)
            collection.set_edgecolor('none')
            collection.set_linewidth(0.0)
    for line in ax.lines:
        line.set_linewidth(0.6)
        line.set_color(INK_2)
    ax.axis('off')


def _half_max_ring(ax: Axes) -> None:
    """One contour at half the peak of the topomap already on `ax`: the kernel's half-width, drawn on the scalp."""
    image = ax.images[-1]
    field = np.ma.masked_invalid(image.get_array())
    x0, x1, y0, y1 = image.get_extent()
    rows, cols = field.shape
    ring = ax.contour(
        np.linspace(x0, x1, cols),
        np.linspace(y0, y1, rows),
        field,
        levels=[0.5],
        colors=[INK],
        linewidths=0.35,
        alpha=0.6,
        zorder=4,
    )
    ring.set_clip_path(image.get_clip_path())


def _canvas(width: float, height: float) -> tuple[Figure, Axes]:
    """A figure whose single axes maps one data unit to one inch, so every position below is a printed length."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    blank(ax, (0.0, width), (0.0, height))

    return fig, ax


def _inset(fig: Figure, x: float, y: float, w: float, h: float) -> Axes:
    """A transparent axes placed by inches from the figure's bottom-left corner."""
    width, height = fig.get_size_inches()
    ax = fig.add_axes((x / width, y / height, w / width, h / height))
    ax.patch.set_alpha(0.0)

    return ax


def _degree_colour(degree: int) -> tuple[float, float, float, float]:
    """The sequential-blue step that stands for harmonic degree `degree` in every figure."""
    return SEQUENTIAL_CMAP(0.15 + 0.85 * degree / _L_MAX)


def _kernel(gamma: np.ndarray, gains: Sequence[float]) -> np.ndarray:
    """The code inner product two electrodes at geodesic angle `gamma` share, by the addition theorem."""
    total = np.zeros_like(gamma)
    for degree, gain in enumerate(gains):
        total += gain**2 * (2 * degree + 1) / (4 * math.pi) * eval_legendre(degree, np.cos(gamma))

    return total


def _region_rank(regions: Sequence[str]) -> np.ndarray:
    """Each channel's anterior-to-posterior region index."""
    return np.array([_REGION_INDEX.get(r, 0) for r in regions])


def _anterior_order(xyz: np.ndarray, regions: Sequence[str]) -> np.ndarray:
    """Channel order sorted by region and then front to back, so a covariance shows its blocks."""
    return np.lexsort((-xyz[:, 1], _region_rank(regions)))


def _reader_covariance(xyz: np.ndarray, *, seed: int, scale: float) -> np.ndarray:
    """A synthetic, geometry-true channel covariance: volume conduction plus broad sources, average referenced."""
    rng = np.random.default_rng(seed)
    geometry = ScalpGeometry.from_xyz(xyz, normalize=True)
    n = geometry.n_channels
    # Neighbouring electrodes see the same tissue, distant ones share only the broad dipolar patterns; each reader's
    # cap sits and amplifies differently, and the average reference is what makes far pairs anticorrelate.
    local = np.exp(-((geometry.geodesic_angles() / 0.55) ** 2))
    patterns = geometry.spherical_harmonics(2)[:, 1:]
    weights = rng.uniform(0.4, 1.6, patterns.shape[1])
    sources = (patterns * weights) @ patterns.T
    gain = np.exp(0.25 * rng.standard_normal(n))
    cov = scale * (gain[:, None] * (local + 0.9 * sources) * gain[None, :])
    centre = np.eye(n) - 1.0 / n
    cov = centre @ cov @ centre

    return (cov + cov.T) / 2 + 1e-3 * scale * np.eye(n)


def _signature(cov: np.ndarray, regions: Sequence[str]) -> np.ndarray:
    """The 141-number signature: per-channel log scales, then the log-Euclidean tangent of the region covariance."""
    log_scale = 0.5 * np.log(np.diag(cov))
    rank = _region_rank(regions)
    members = [np.flatnonzero(rank == r) for r in range(len(SCALP_REGIONS))]
    pooled = np.array([[cov[np.ix_(a, b)].mean() for b in members] for a in members])
    eigenvalues, eigenvectors = np.linalg.eigh(pooled)
    tangent = (eigenvectors * np.log(np.clip(eigenvalues, 1e-9, None))) @ eigenvectors.T
    upper = tangent[np.triu_indices(len(members))]

    return np.concatenate([log_scale, upper])


def _standardised(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-spread copy, so a signed colormap has a neutral midpoint to sit on."""
    return (values - values.mean()) / max(float(values.std()), 1e-9)


def _strip(
    ax: Axes, values: np.ndarray, x: float, y: float, w: float, h: float, *, cmap: Colormap, vlim: tuple[float, float]
) -> None:
    """A row of `values` as coloured cells filling the rectangle at `(x, y)`, with a thin ink frame."""
    ax.imshow(
        values[None, :],
        extent=(x, x + w, y, y + h),
        cmap=cmap,
        vmin=vlim[0],
        vmax=vlim[1],
        interpolation='nearest',
        zorder=2,
    )
    ax.add_patch(Rectangle((x, y), w, h, facecolor='none', edgecolor=INK_2, linewidth=0.5, zorder=3))


def _column(
    ax: Axes, values: np.ndarray, x: float, y: float, w: float, h: float, *, cmap: Colormap, vlim: tuple[float, float]
) -> None:
    """A column of `values` as coloured cells, first value at the top."""
    ax.imshow(
        values[:, None],
        extent=(x, x + w, y, y + h),
        cmap=cmap,
        vmin=vlim[0],
        vmax=vlim[1],
        interpolation='nearest',
        zorder=2,
    )
    ax.add_patch(Rectangle((x, y), w, h, facecolor='none', edgecolor=INK_2, linewidth=0.5, zorder=3))


def _signature_strip(ax: Axes, signature: np.ndarray, x: float, y: float, w: float, h: float, *, dims: bool) -> None:
    """The signature as two framed segments, 105 then 36, each standardised so the split is visible."""
    split = w * _N_SCALE / _N_SIGNATURE
    scales = _standardised(signature[:_N_SCALE])
    tangent = _standardised(signature[_N_SCALE:])
    _strip(ax, scales, x, y, split - 0.01, h, cmap=DIVERGING_CMAP, vlim=(-2.5, 2.5))
    _strip(ax, tangent, x + split + 0.01, y, w - split - 0.01, h, cmap=DIVERGING_CMAP, vlim=(-2.5, 2.5))
    if dims:
        ax.text(x + split / 2, y - 0.04, '105', ha='center', va='top', fontsize=5.5, color=INK_2)
        ax.text(x + (w + split) / 2, y - 0.04, '36', ha='center', va='top', fontsize=5.5, color=INK_2)


def _reader_glyph(ax: Axes, x: float, y: float, r: float, *, held_out: bool) -> None:
    """A small head seen from above: ink for a training reader, red with a question mark for the held-out one."""
    head(ax, x, y, r, color=RED if held_out else INK_2)
    if held_out:
        ax.text(x, y - 0.01 * r, '?', ha='center', va='center', fontsize=9 * r / 0.2, color=RED, fontweight='bold')


def _region_key(ax: Axes, x: float, y0: float, y1: float, w: float) -> None:
    """The ordinal region key: eight swatches from anterior (top) to posterior (bottom), abbreviated beside them."""
    step = (y1 - y0) / len(SCALP_REGIONS)
    for i, short in enumerate(_REGION_SHORT):
        top = y1 - i * step
        ax.add_patch(Rectangle((x, top - step), w, step, facecolor=REGION_CMAP(i), edgecolor='none'))
        ax.text(x + w + 0.05, top - step / 2, short, ha='left', va='center', fontsize=5.5, color=INK_2)
    ax.add_patch(Rectangle((x, y0), w, y1 - y0, facecolor='none', edgecolor=INK_2, linewidth=0.5))


def _dashed_route(ax: Axes, points: Sequence[tuple[float, float]]) -> None:
    """A generated-parameter path: a dashed orthogonal polyline ending in a short solid arrowhead."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=EEG, linewidth=0.7, linestyle=_DASH, solid_capstyle='butt', zorder=2)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    length = math.hypot(x1 - x0, y1 - y0) or 1.0
    stub = 0.06
    arrow(ax, x1 - (x1 - x0) / length * stub, y1 - (y1 - y0) / length * stub, x1, y1, color=EEG)


def _operator(ax: Axes, x: float, y: float, r: float, *, product: bool) -> None:
    """An elementwise-product (dot in a circle) or sum (plus in a circle) node on a data path."""
    ax.add_patch(Circle((x, y), r, facecolor='white', edgecolor=INK, linewidth=0.7, zorder=3))
    if product:
        ax.add_patch(Circle((x, y), 0.22 * r, facecolor=INK, edgecolor='none', zorder=4))
    else:
        ax.plot([x - 0.55 * r, x + 0.55 * r], [y, y], color=INK, linewidth=0.7, zorder=4)
        ax.plot([x, x], [y - 0.55 * r, y + 0.55 * r], color=INK, linewidth=0.7, zorder=4)


def _gain_bars(ax: Axes) -> None:
    """The per-degree gains as a bar row on a tiny axes, with the unit line every gain starts from."""
    degrees = range(len(_DEGREE_GAINS))
    ax.bar(degrees, _DEGREE_GAINS, width=0.7, color=[_degree_colour(d) for d in degrees], linewidth=0)
    ax.axhline(1.0, color=INK_2, linewidth=0.4, linestyle=_DASH)
    ax.set_xticks(degrees)
    ax.set_yticks([0, 1])
    ax.set_ylim(0, 1.45)
    ax.set_xlabel(r'$\ell$', fontsize=6.5, labelpad=1)
    ax.set_ylabel(r'$g_\ell$', fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=5.5, length=1.5, width=0.5, pad=1.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(0.5)


def _covariance_cut(
    ax: Axes,
    ox: float,
    oy: float,
    cuts: Sequence[tuple[np.ndarray, str]],
    pair: tuple[str, str],
    *,
    reach: float,
) -> None:
    """Two-electrode cuts of the readers' covariances as ellipses on a shared origin, scaled to fit `reach`."""
    ax.plot([ox - reach, ox + reach], [oy, oy], color=INK_2, linewidth=0.4, zorder=1)
    ax.plot([ox, ox], [oy - reach, oy + reach], color=INK_2, linewidth=0.4, zorder=1)
    ax.text(ox + reach + 0.03, oy, pair[0], ha='left', va='center', fontsize=5.5, color=INK_2)
    ax.text(ox, oy + reach + 0.03, pair[1], ha='center', va='bottom', fontsize=5.5, color=INK_2)
    largest = max(math.sqrt(float(np.linalg.eigvalsh(cut).max())) for cut, _ in cuts)
    scale = 0.9 * reach / largest
    for cut, colour in cuts:
        eigenvalues, eigenvectors = np.linalg.eigh(cut)
        angle = math.degrees(math.atan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        ax.add_patch(
            Ellipse(
                (ox, oy),
                2 * scale * math.sqrt(eigenvalues[1]),
                2 * scale * math.sqrt(eigenvalues[0]),
                angle=angle,
                facecolor=colour,
                alpha=0.12,
                edgecolor='none',
                zorder=2,
            )
        )
        ax.add_patch(
            Ellipse(
                (ox, oy),
                2 * scale * math.sqrt(eigenvalues[1]),
                2 * scale * math.sqrt(eigenvalues[0]),
                angle=angle,
                facecolor='none',
                edgecolor=colour,
                linewidth=0.8,
                zorder=3,
            )
        )


def _unit_circle(ax: Axes, cx: float, cy: float, r: float) -> None:
    """The whitened cut: one ink circle with both readers' rings lying on it."""
    ax.plot([cx - 1.4 * r, cx + 1.4 * r], [cy, cy], color=INK_2, linewidth=0.4, zorder=1)
    ax.plot([cx, cx], [cy - 1.4 * r, cy + 1.4 * r], color=INK_2, linewidth=0.4, zorder=1)
    ax.add_patch(Circle((cx, cy), r, facecolor='none', edgecolor=INK, linewidth=0.9, zorder=3))
    ax.add_patch(
        Circle((cx, cy), r, facecolor='none', edgecolor=EEG, linewidth=0.7, linestyle=(0.0, (2.5, 2.5)), zorder=4)
    )
    ax.add_patch(
        Circle((cx, cy), r, facecolor='none', edgecolor=RED, linewidth=0.7, linestyle=(2.5, (2.5, 2.5)), zorder=4)
    )


def _colorbar(fig: Figure, cax: Axes, cmap: Colormap, vlim: tuple[float, float], *, horizontal: bool) -> None:
    """A thin shared colorbar with three ticks and hairline furniture."""
    mappable = ScalarMappable(norm=Normalize(vlim[0], vlim[1]), cmap=cmap)
    ticks = [vlim[0], 0.5 * (vlim[0] + vlim[1]), vlim[1]]
    bar = fig.colorbar(mappable, cax=cax, orientation='horizontal' if horizontal else 'vertical', ticks=ticks)
    bar.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1.5)


# ---- The figures ---- #


def montage_map() -> Figure:
    """The 105 retained EGI electrodes on the head, coloured by anterior-to-posterior region.

    Returns:
        Figure: A single-column figure.
    """
    width, height = SINGLE_COLUMN_IN, 3.35
    fig, ax = _canvas(width, height)
    xyz, _, regions = montage()
    cx, cy, radius = 1.55, 1.62, 1.42
    xy = _scalp_xy(xyz) * radius
    head(ax, cx, cy, radius)
    ax.scatter(
        cx + xy[:, 0],
        cy + xy[:, 1],
        s=18,
        c=[REGION_CMAP(r) for r in _region_rank(regions)],
        edgecolors='white',
        linewidths=0.4,
        zorder=3,
    )
    _region_key(ax, 3.2, cy - radius, cy + radius, 0.08)
    ax.text(cx - radius, cy - radius - 0.02, '105 electrodes', ha='left', va='top', fontsize=6.5, color=INK_2)

    return fig


def montage_map_labels() -> Figure:
    """The montage with every electrode numbered, for looking a channel up.

    Returns:
        Figure: A single-column figure.
    """
    width, height = SINGLE_COLUMN_IN, 3.6
    fig, ax = _canvas(width, height)
    xyz, labels, regions = montage()
    cx, cy, radius = 1.75, 1.72, 1.6
    xy = _scalp_xy(xyz) * radius
    head(ax, cx, cy, radius)
    ax.scatter(
        cx + xy[:, 0],
        cy + xy[:, 1],
        s=9,
        c=[REGION_CMAP(r) for r in _region_rank(regions)],
        edgecolors='none',
        zorder=3,
    )
    # A number sits above its dot on the front half and below it on the back half, so the labels lean away from the
    # densest ring around the vertex instead of into it.
    for (x, y), label in zip(xy, labels, strict=True):
        above = y >= 0
        ax.text(
            cx + x,
            cy + y + (0.045 if above else -0.045),
            label.removeprefix('E'),
            ha='center',
            va='bottom' if above else 'top',
            fontsize=5.0,
            color=INK_2,
            zorder=4,
        )

    return fig


def harmonic_basis() -> Figure:
    """Real spherical harmonics on the real head, degrees 0 to 3, with the learnable per-degree gains.

    Returns:
        Figure: A double-column figure.
    """
    xyz, labels, _ = montage()
    geometry = ScalpGeometry.from_xyz(xyz, labels=tuple(labels), normalize=True)
    basis = real_spherical_harmonics(geometry.theta, geometry.phi, 3)
    degrees = degree_of_column(3)
    cell, left, bottom = 0.955, 0.30, 0.27
    rows = 4
    width, height = DOUBLE_COLUMN_IN, bottom + rows * cell + 0.06
    fig, canvas = _canvas(width, height)

    # The pyramid: row l holds orders -l .. l centred on m = 0, each degree scaled to its own maximum.
    for degree in range(rows):
        limit = max(float(np.abs(basis[:, degrees == degree]).max()), 1e-9)
        for order in range(-degree, degree + 1):
            x = left + (order + 3) * cell + 0.02 * cell
            y = bottom + (rows - 1 - degree) * cell + 0.02 * cell
            ax = _inset(fig, x, y, 0.96 * cell, 0.96 * cell)
            column = degree * degree + order + degree
            # A constant field has no level to contour; asking for one draws interpolation noise.
            _scalp(
                ax, basis[:, column] / limit, xyz, cmap=DIVERGING_CMAP, vlim=(-1.0, 1.0), contours=4 if degree else 0
            )
        canvas.text(
            left - 0.04,
            bottom + (rows - 1 - degree) * cell + cell / 2,
            rf'$\ell={degree}$',
            ha='right',
            va='center',
            fontsize=7,
            color=INK,
        )
    for order in range(-3, 4):
        canvas.text(
            left + (order + 3) * cell + cell / 2,
            bottom - 0.05,
            f'{order:+d}' if order else '0',
            ha='center',
            va='top',
            fontsize=6.5,
            color=INK,
        )
    canvas.text(left - 0.04, bottom - 0.05, r'$m$', ha='right', va='top', fontsize=7, color=INK)

    # The two empty top corners carry the gains and the one shared colorbar.
    top_row = bottom + (rows - 1) * cell
    _gain_bars(_inset(fig, left + 0.55, top_row + 0.28, 2 * cell - 0.7, 0.5))
    cax = _inset(fig, left + 5 * cell + 0.2, top_row + 0.5, 2 * cell - 0.6, 0.08)
    _colorbar(fig, cax, DIVERGING_CMAP, (-1.0, 1.0), horizontal=True)
    canvas.text(
        left + 6 * cell, top_row + 0.66, r'$Y_\ell^m\,/\,\max|Y_\ell|$', ha='center', va='bottom', fontsize=7, color=INK
    )

    return fig


def harmonic_kernel() -> Figure:
    """The addition theorem: Legendre curves of the geodesic angle, and the resulting kernel drawn on the head.

    Returns:
        Figure: A single-column figure.
    """
    width, height = SINGLE_COLUMN_IN, 1.95
    fig, canvas = _canvas(width, height)

    # (a) P_l(cos gamma) for every degree the model uses, in the degree colours.
    ax = _inset(fig, 0.40, 0.36, 1.55, 1.42)
    gamma = np.linspace(0.0, math.pi, 400)
    for degree in range(_L_MAX + 1):
        ax.plot(np.degrees(gamma), eval_legendre(degree, np.cos(gamma)), color=_degree_colour(degree), linewidth=0.8)
    ax.axhline(0.0, color=MUTED, linewidth=0.4)
    ax.set_xlim(0, 180)
    ax.set_ylim(-1.05, 1.08)
    ax.set_xticks([0, 90, 180], ['0°', '90°', '180°'])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r'$\gamma$', fontsize=7, labelpad=1)
    ax.set_ylabel(r'$P_\ell(\cos\gamma)$', fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6, length=2, width=0.5, pad=1.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(0.5)
    for degree in range(_L_MAX + 1):
        ax.add_patch(Rectangle((74 + 8 * degree, 0.72), 8, 0.12, facecolor=_degree_colour(degree), edgecolor='none'))
    ax.text(71, 0.78, r'$\ell$', ha='right', va='center', fontsize=6.5, color=INK)
    ax.text(74, 0.88, '0', ha='left', va='bottom', fontsize=5.5, color=INK_2)
    ax.text(130, 0.88, '6', ha='right', va='bottom', fontsize=5.5, color=INK_2)

    # (b) The kernel from the vertex electrode to every other electrode, with the angle to one of them marked.
    xyz, _, _ = montage()
    geometry = ScalpGeometry.from_xyz(xyz, normalize=True)
    reference = int(geometry.theta.argmin())
    angles = geometry.geodesic_angles()[reference]
    kernel = _kernel(angles, _DEGREE_GAINS)
    kernel /= kernel[reference]
    hx = _inset(fig, 2.12, 0.42, 1.28, 1.28)
    # mne's tick-chosen levels include zero, which wanders along the rim where the truncated kernel hovers near it.
    _scalp(hx, kernel, xyz, cmap=DIVERGING_CMAP, vlim=(-1.0, 1.0), contours=0)
    _half_max_ring(hx)
    pos = _mne_xy(xyz)
    posterior = np.flatnonzero(geometry.xyz[:, 1] < -0.3)
    partner = int(posterior[np.abs(angles[posterior] - math.radians(68)).argmin()])
    hx.plot(
        [pos[reference, 0], pos[partner, 0]],
        [pos[reference, 1], pos[partner, 1]],
        color=INK,
        linewidth=0.5,
        linestyle=_DASH,
        zorder=5,
    )
    hx.scatter(*pos[reference], s=16, facecolors='none', edgecolors=INK, linewidths=0.7, zorder=6)
    hx.scatter(*pos[partner], s=16, facecolors='none', edgecolors=INK, linewidths=0.7, zorder=6)
    mid = 0.5 * (pos[reference] + pos[partner])
    hx.text(mid[0] + 0.014, mid[1] + 0.004, r'$\gamma$', ha='left', va='center', fontsize=7, color=INK, zorder=6)
    cax = _inset(fig, 2.45, 0.25, 0.62, 0.06)
    _colorbar(fig, cax, DIVERGING_CMAP, (-1.0, 1.0), horizontal=True)
    canvas.text(
        2.76,
        1.84,
        r'$k(\gamma)=\sum_\ell g_\ell^{2}\,\frac{2\ell+1}{4\pi}\,P_\ell(\cos\gamma)$',
        ha='center',
        va='center',
        fontsize=6.5,
        color=INK,
    )
    canvas.text(0.04, height - 0.04, 'a', ha='left', va='top', fontsize=8, fontweight='bold', color=INK)
    canvas.text(2.06, height - 0.04, 'b', ha='left', va='top', fontsize=8, fontweight='bold', color=INK)

    return fig


def covariance_transport() -> Figure:
    """Two readers' channel covariances, the two-electrode cut whitening collapses, and the identity both land on.

    Returns:
        Figure: A double-column figure.
    """
    width, height = DOUBLE_COLUMN_IN, 2.45
    fig, ax = _canvas(width, height)
    xyz, labels, regions = montage()
    order = _anterior_order(xyz, regions)
    sorted_regions = [regions[i] for i in order]
    readers = [
        (_reader_covariance(xyz, seed=4, scale=1.0)[np.ix_(order, order)], EEG, False, r'$R_s$'),
        (_reader_covariance(xyz, seed=11, scale=1.8)[np.ix_(order, order)], RED, True, r"$R_{s'}$"),
    ]
    # The diagonal dominates a covariance; letting it saturate leaves the colour range for the off-diagonal blocks.
    limit = 0.7 * max(float(np.abs(cov).max()) for cov, _, _, _ in readers)
    n = xyz.shape[0]

    # (a) The covariances, channels sorted front to back so the region strip explains the blocks.
    size, y0 = 1.36, 0.68
    for i, (cov, colour, held_out, symbol) in enumerate(readers):
        x0 = 0.36 + i * (size + 0.22)
        ax.imshow(
            cov / limit,
            extent=(x0, x0 + size, y0, y0 + size),
            cmap=DIVERGING_CMAP,
            vmin=-1.0,
            vmax=1.0,
            interpolation='nearest',
            zorder=2,
        )
        ax.add_patch(Rectangle((x0, y0), size, size, facecolor='none', edgecolor=INK_2, linewidth=0.5, zorder=3))
        _column(
            ax,
            _region_rank(sorted_regions).astype(float),
            x0 - 0.09,
            y0,
            0.06,
            size,
            cmap=REGION_CMAP,
            vlim=(-0.5, len(SCALP_REGIONS) - 0.5),
        )
        _reader_glyph(ax, x0 + size / 2 - 0.2, y0 + size + 0.16, 0.1, held_out=held_out)
        ax.text(x0 + size / 2 + 0.02, y0 + size + 0.16, symbol, ha='left', va='center', fontsize=8, color=colour)
        _signature_strip(ax, _signature(cov, sorted_regions), x0, 0.34, size, 0.14, dims=True)
        ax.text(x0 - 0.04, 0.41, r'$\sigma$', ha='right', va='center', fontsize=7, color=colour)

    # (b) The cut through the vertex electrode and its most correlated neighbour, before and after whitening.
    reference = int(ScalpGeometry.from_xyz(xyz, normalize=True).theta.argmin())
    position = int(np.flatnonzero(order == reference)[0])
    first = readers[0][0]
    partner = int(np.argsort(first[position] / np.sqrt(first[position, position] * np.diag(first)))[-2])
    pair = (labels[order[position]], labels[order[partner]])
    cuts = [(cov[np.ix_([position, partner], [position, partner])], colour) for cov, colour, _, _ in readers]
    ox, oy = 3.95, y0 + size / 2
    _covariance_cut(ax, ox, oy, cuts, pair, reach=0.42)
    arrow(ax, ox + 0.52, oy, ox + 0.8, oy)
    ax.text(ox + 0.66, oy + 0.06, r'$R_s^{-1/2}$', ha='center', va='bottom', fontsize=7, color=INK)
    _unit_circle(ax, 5.05, oy, 0.24)

    # (c) The identity, on the same colour scale.
    ix = 5.52
    arrow(ax, 5.05 + 0.24 * 1.4 + 0.02, oy, ix - 0.02, oy)
    ax.imshow(
        np.eye(n),
        extent=(ix, ix + size, y0, y0 + size),
        cmap=DIVERGING_CMAP,
        vmin=-1.0,
        vmax=1.0,
        interpolation='nearest',
        zorder=2,
    )
    ax.add_patch(Rectangle((ix, y0), size, size, facecolor='none', edgecolor=INK_2, linewidth=0.5, zorder=3))
    ax.text(ix + size / 2, y0 + size + 0.16, r'$R_s^{-1/2}R_sR_s^{-1/2}=I$', ha='center', va='center', fontsize=7.5)
    cax = _inset(fig, ix + size + 0.08, y0, 0.055, size)
    _colorbar(fig, cax, DIVERGING_CMAP, (-1.0, 1.0), horizontal=False)
    for letter, x in (('a', 0.04), ('b', 3.42), ('c', 5.42)):
        ax.text(x, height - 0.04, letter, ha='left', va='top', fontsize=8, fontweight='bold', color=INK)

    return fig


def transport_manifold_only() -> Figure:
    """The transport as geometry alone: two readers on the SPD manifold carried to the identity, and the planar cut.

    Returns:
        Figure: A single-column figure.
    """
    width, height = SINGLE_COLUMN_IN, 1.75
    fig, ax = _canvas(width, height)
    xyz, labels, regions = montage()
    order = _anterior_order(xyz, regions)
    readers = [
        (_reader_covariance(xyz, seed=4, scale=1.0)[np.ix_(order, order)], EEG, r'$R_s$'),
        (_reader_covariance(xyz, seed=11, scale=1.8)[np.ix_(order, order)], RED, r"$R_{s'}$"),
    ]

    # The manifold: a dome with both readers below the crown and the geodesics that carry them to I.
    cx, cy, a, b = 0.95, 0.42, 0.82, 0.95
    t = np.linspace(0.0, math.pi, 120)
    ax.add_patch(
        Polygon(
            np.stack([cx + a * np.cos(t), cy + b * np.sin(t)], axis=1),
            closed=True,
            facecolor=_EEG_TINT,
            edgecolor=EEG,
            linewidth=0.6,
            zorder=1,
        )
    )
    ax.plot(cx + a * np.cos(t), cy - 0.16 * np.sin(t), color=EEG, linewidth=0.5, linestyle=_DASH, zorder=1)
    ax.text(cx - a + 0.02, cy + 0.02, 'SPD(105)', ha='left', va='bottom', fontsize=5.5, color=INK_2)
    crown = (cx, cy + b)
    points = [(cx - 0.52, cy + 0.42, EEG, readers[0][2]), (cx + 0.5, cy + 0.32, RED, readers[1][2])]
    for x, y, colour, symbol in points:
        ax.scatter(x, y, s=14, color=colour, edgecolors='white', linewidths=0.4, zorder=4)
        ax.text(x, y - 0.09, symbol, ha='center', va='top', fontsize=7, color=colour)
        arrow_rad = -0.3 if x < cx else 0.3
        ax.annotate(
            '',
            xy=crown,
            xytext=(x, y),
            arrowprops={
                'arrowstyle': '-|>',
                'connectionstyle': f'arc3,rad={arrow_rad}',
                'color': INK_2,
                'linewidth': 0.7,
                'mutation_scale': 7,
                'shrinkA': 3,
                'shrinkB': 4,
            },
            zorder=3,
        )
    ax.scatter(*crown, s=22, facecolors='white', edgecolors=INK, linewidths=0.9, zorder=5)
    ax.text(crown[0], crown[1] + 0.07, r'$I$', ha='center', va='bottom', fontsize=8, color=INK)
    ax.text(cx, cy + 0.55, r'$R^{-1/2}$', ha='center', va='center', fontsize=7, color=INK)

    # The planar cut, the same pair of electrodes as the transport figure.
    reference = int(ScalpGeometry.from_xyz(xyz, normalize=True).theta.argmin())
    position = int(np.flatnonzero(order == reference)[0])
    first = readers[0][0]
    partner = int(np.argsort(first[position] / np.sqrt(first[position, position] * np.diag(first)))[-2])
    pair = (labels[order[position]], labels[order[partner]])
    cuts = [(cov[np.ix_([position, partner], [position, partner])], colour) for cov, colour, _ in readers]
    ox, oy = 2.2, 0.9
    _covariance_cut(ax, ox, oy, cuts, pair, reach=0.38)
    arrow(ax, ox + 0.5, oy, ox + 0.72, oy)
    ax.text(ox + 0.61, oy + 0.06, r'$R^{-1/2}$', ha='center', va='bottom', fontsize=7, color=INK)
    _unit_circle(ax, ox + 0.98, oy, 0.19)

    return fig


def signature_adapter() -> Figure:
    """A stranger's signature drives a hypernetwork that emits electrode gains and a FiLM affine around the frontend.

    Returns:
        Figure: A double-column figure.
    """
    width, height = DOUBLE_COLUMN_IN, 2.72
    fig, ax = _canvas(width, height)
    xyz, _, regions = montage()
    rng = np.random.default_rng(5)
    y_data, y_adapt = 2.05, 0.78

    # The data path: the whitened window, the electrode gains on the head, the frontend, the token, FiLM, onwards.
    tensor_slab(ax, 0.25, y_data - 0.28, 0.95, 0.56, 0.12, dims=('350', '105', ''))
    ax.text(0.78, y_data + 0.4, r'$R^{-1/2}x$', ha='center', va='bottom', fontsize=7, color=EEG)
    arrow(ax, 1.27, y_data, 1.53, y_data, color=EEG)
    geometry = ScalpGeometry.from_xyz(xyz, normalize=True)
    smooth = geometry.spherical_harmonics(2)[:, 1:] @ rng.standard_normal(8)
    gains = 1.0 + 0.3 * smooth / float(np.abs(smooth).max())
    hx = _inset(fig, 1.54, y_data - 0.39, 0.78, 0.78)
    _scalp(hx, gains, xyz, cmap=DIVERGING_CMAP, vlim=(0.7, 1.3), contours=3)
    ax.text(1.93, y_data - 0.44, r'$\odot\;g$', ha='center', va='top', fontsize=7.5, color=INK)
    arrow(ax, 2.32, y_data, 2.6, y_data, color=EEG)
    box(ax, 3.1, y_data, 1.0, 0.5, 'conformer', fill=_EEG_TINT, edge=EEG)
    arrow(ax, 3.6, y_data, 3.86, y_data, color=EEG)
    token = np.abs(rng.standard_normal(256))
    _column(ax, token, 3.88, y_data - 0.35, 0.12, 0.7, cmap=SEQUENTIAL_CMAP, vlim=(0.0, 2.5))
    ax.text(3.94, y_data + 0.4, r'$h$', ha='center', va='bottom', fontsize=8, color=EEG)
    ax.text(3.94, y_data - 0.4, '256', ha='center', va='top', fontsize=5.5, color=INK_2)
    arrow(ax, 4.02, y_data, 4.28, y_data, color=EEG)
    _operator(ax, 4.4, y_data, 0.1, product=True)
    ax.plot([4.5, 4.7], [y_data, y_data], color=EEG, linewidth=0.7, zorder=2)
    _operator(ax, 4.8, y_data, 0.1, product=False)
    arrow(ax, 4.9, y_data, 5.16, y_data, color=EEG)
    _column(
        ax,
        np.abs(token * (1 + 0.2 * rng.standard_normal(256))),
        5.18,
        y_data - 0.35,
        0.12,
        0.7,
        cmap=SEQUENTIAL_CMAP,
        vlim=(0.0, 2.5),
    )
    ax.text(5.24, y_data + 0.4, r"$h'$", ha='center', va='bottom', fontsize=8, color=EEG)
    ax.text(4.6, y_data + 0.42, r'$h\odot(1+\gamma)+\beta$', ha='center', va='bottom', fontsize=6.5, color=INK)
    arrow(ax, 5.32, y_data, 5.58, y_data, color=EEG)
    box(ax, 6.15, y_data, 1.1, 0.5, 'transformer', sub='× 4', fill=FILL, edge=INK_2)
    arrow(ax, 6.7, y_data, 7.0, y_data, color=EEG)

    # The adapter path: the held-out reader's own covariance, its signature, the hypernetwork.
    _reader_glyph(ax, 0.5, y_adapt, 0.2, held_out=True)
    arrow(ax, 0.74, y_adapt, 0.98, y_adapt, color=RED)
    order = _anterior_order(xyz, regions)
    cov = _reader_covariance(xyz, seed=11, scale=1.8)[np.ix_(order, order)]
    ax.imshow(
        cov / float(np.abs(cov).max()),
        extent=(1.0, 1.46, y_adapt - 0.23, y_adapt + 0.23),
        cmap=DIVERGING_CMAP,
        vmin=-1.0,
        vmax=1.0,
        interpolation='nearest',
        zorder=2,
    )
    ax.add_patch(Rectangle((1.0, y_adapt - 0.23), 0.46, 0.46, facecolor='none', edgecolor=RED, linewidth=0.6, zorder=3))
    ax.text(1.23, y_adapt + 0.27, r"$R_{s'}$", ha='center', va='bottom', fontsize=7, color=RED)
    arrow(ax, 1.5, y_adapt, 1.74, y_adapt, color=RED)
    signature = _signature(cov, [regions[i] for i in order])
    _signature_strip(ax, signature, 1.76, y_adapt - 0.09, 1.6, 0.18, dims=False)
    split = 1.76 + 1.6 * _N_SCALE / _N_SIGNATURE
    ax.text((1.76 + split) / 2, y_adapt - 0.13, 'log scale · 105', ha='center', va='top', fontsize=5.5, color=INK_2)
    ax.text((split + 3.36) / 2, y_adapt - 0.13, 'tangent · 36', ha='center', va='top', fontsize=5.5, color=INK_2)
    ax.text(
        2.56, y_adapt + 0.13, r"$\sigma_{s'}\in\mathbb{R}^{141}$", ha='center', va='bottom', fontsize=6.5, color=RED
    )
    arrow(ax, 3.4, y_adapt, 3.64, y_adapt, color=RED)
    ax.add_patch(
        Polygon(
            [[3.66, y_adapt - 0.25], [3.66, y_adapt + 0.25], [4.5, y_adapt + 0.15], [4.5, y_adapt - 0.15]],
            closed=True,
            facecolor=_EEG_TINT,
            edgecolor=EEG,
            linewidth=0.7,
            zorder=2,
        )
    )
    ax.text(4.06, y_adapt + 0.05, 'hypernet', ha='center', va='center', fontsize=6.5, color=INK, zorder=3)
    ax.text(4.06, y_adapt - 0.08, '128', ha='center', va='center', fontsize=5.5, color=INK_2, zorder=3)
    ax.text(4.08, y_adapt - 0.34, r'init: $g=1,\ \gamma=\beta=0$', ha='center', va='top', fontsize=5.5, color=INK_2)

    # Generated parameters travel dashed: up to the gain head, and up to the two FiLM vectors under their nodes.
    _dashed_route(ax, [(3.9, y_adapt + 0.22), (3.9, 1.32), (1.93, 1.32), (1.93, y_data - 0.4)])
    _dashed_route(ax, [(4.5, y_adapt), (4.6, y_adapt), (4.6, 1.1), (4.4, 1.1), (4.4, 1.3)])
    _dashed_route(ax, [(4.6, 1.1), (4.8, 1.1), (4.8, 1.3)])
    for x, symbol in ((4.4, r'$\gamma$'), (4.8, r'$\beta$')):
        _column(ax, rng.standard_normal(256), x - 0.06, 1.32, 0.12, 0.42, cmap=DIVERGING_CMAP, vlim=(-2.5, 2.5))
        ax.plot([x, x], [1.74, y_data - 0.1], color=EEG, linewidth=0.7, zorder=2)
        ax.text(x + 0.1, 1.53, symbol, ha='left', va='center', fontsize=7, color=INK)

    return fig


def harmonic_code_on_head() -> Figure:
    """One electrode's harmonic code, the degree gains that weight it, and the kernel it induces, in one glance.

    Returns:
        Figure: A single-column figure.
    """
    width, height = SINGLE_COLUMN_IN, 1.9
    fig, ax = _canvas(width, height)
    xyz, _, _ = montage()
    geometry = ScalpGeometry.from_xyz(xyz, normalize=True)
    # A left-parietal electrode: far enough from the vertex that every degree contributes visibly.
    target = np.array([-0.55, -0.45, 0.7])
    chosen = int((geometry.xyz @ (target / np.linalg.norm(target))).argmax())
    basis = real_spherical_harmonics(geometry.theta, geometry.phi, 3)
    degrees = degree_of_column(3)
    code = basis[chosen].copy()
    for degree in range(4):
        code[degrees == degree] /= max(float(np.abs(basis[:, degrees == degree]).max()), 1e-9)

    # The head, with the chosen electrode picked out and led to its code.
    cx, cy, radius = 0.62, 1.0, 0.5
    xy = _scalp_xy(xyz) * radius
    head(ax, cx, cy, radius)
    ax.scatter(cx + xy[:, 0], cy + xy[:, 1], s=4, color=INK_2, linewidths=0, zorder=3)
    ax.scatter(cx + xy[chosen, 0], cy + xy[chosen, 1], s=20, color=EEG, edgecolors='white', linewidths=0.4, zorder=4)
    ax.text(cx + xy[chosen, 0] - 0.06, cy + xy[chosen, 1], r'$c$', ha='right', va='center', fontsize=7, color=EEG)
    col_x, col_y, col_w, cell = 1.36, 0.3, 0.14, 0.08
    ax.plot(
        [cx + xy[chosen, 0], col_x],
        [cy + xy[chosen, 1], col_y + 16 * cell],
        color=INK_2,
        linewidth=0.5,
        linestyle=_DASH,
        zorder=2,
    )
    _column(ax, code, col_x, col_y, col_w, 16 * cell, cmap=DIVERGING_CMAP, vlim=(-1.0, 1.0))
    ax.text(col_x + col_w / 2, col_y + 16 * cell + 0.05, r'$Y_\ell^m(c)$', ha='center', va='bottom', fontsize=7)
    top = col_y + 16 * cell
    brace_x = col_x + col_w + 0.03
    for degree in range(4):
        y_top = top - degree * degree * cell
        y_bottom = y_top - (2 * degree + 1) * cell
        ax.plot(
            [brace_x, brace_x + 0.04, brace_x + 0.04, brace_x],
            [y_top, y_top, y_bottom, y_bottom],
            color=INK_2,
            linewidth=0.5,
        )
        ax.text(brace_x + 0.08, (y_top + y_bottom) / 2, str(degree), ha='left', va='center', fontsize=5.5, color=INK_2)
    ax.text(brace_x + 0.08, top + 0.05, r'$\ell$', ha='left', va='bottom', fontsize=7, color=INK)

    # The gains that weight each degree, then the kernel the weighted code induces from that electrode.
    bar_x, bar_y = 1.98, 0.62
    for degree in range(4):
        gain = _DEGREE_GAINS[degree]
        ax.add_patch(
            Rectangle(
                (bar_x + 0.11 * degree, bar_y), 0.09, 0.45 * gain, facecolor=_degree_colour(degree), edgecolor='none'
            )
        )
        ax.text(
            bar_x + 0.11 * degree + 0.045, bar_y - 0.03, str(degree), ha='center', va='top', fontsize=5.5, color=INK_2
        )
    ax.plot([bar_x - 0.02, bar_x + 0.44], [bar_y, bar_y], color=INK_2, linewidth=0.5)
    ax.plot([bar_x - 0.02, bar_x + 0.44], [bar_y + 0.45, bar_y + 0.45], color=INK_2, linewidth=0.4, linestyle=_DASH)
    ax.text(bar_x + 0.21, bar_y + 0.45 * 1.3 + 0.05, r'$g_\ell$', ha='center', va='bottom', fontsize=7, color=INK)
    ax.text(1.85, bar_y + 0.22, r'$\times$', ha='center', va='center', fontsize=8, color=INK)
    arrow(ax, 2.46, bar_y + 0.22, 2.6, bar_y + 0.22)
    kernel = _kernel(geometry.geodesic_angles()[chosen], _DEGREE_GAINS[:4])
    kernel /= kernel[chosen]
    hx = _inset(fig, 2.42, 0.38, 1.0, 1.0)
    _scalp(hx, kernel, xyz, cmap=DIVERGING_CMAP, vlim=(-1.0, 1.0), contours=0)
    _half_max_ring(hx)
    pos = _mne_xy(xyz)
    hx.scatter(*pos[chosen], s=16, facecolors='none', edgecolors=INK, linewidths=0.7, zorder=6)
    ax.text(2.92, 1.42, r'$k(c,\cdot)$', ha='center', va='bottom', fontsize=7, color=INK)
    ax.text(
        1.9,
        0.1,
        r"$k(c,c')=\sum_\ell g_\ell^{2}\,\frac{2\ell+1}{4\pi}\,P_\ell(\cos\gamma_{cc'})$",
        ha='center',
        va='center',
        fontsize=6.5,
        color=INK,
    )

    return fig


SCHEMATICS = {
    'montage_map': montage_map,
    'montage_map_labels': montage_map_labels,
    'harmonic_basis': harmonic_basis,
    'harmonic_kernel': harmonic_kernel,
    'harmonic_code_on_head': harmonic_code_on_head,
    'covariance_transport': covariance_transport,
    'transport_manifold_only': transport_manifold_only,
    'signature_adapter': signature_adapter,
}
"""This family's data-free schematics, by name."""
