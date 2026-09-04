"""The shared visual language of the schematics: palette by role, journal widths, and the drawing helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from zte.data.montage.montage import packaged_montage_csv
from zte.lens.montage import azimuthal_xy, load_montage_csv
from zte.logging_utils import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_LOG = get_logger('evaluation.schematics')

# Matplotlib's axes classes are untyped upstream; named once so the drawing helpers stay readable.
type Axes = Any
"""A matplotlib axes."""

# IEEE Transactions column widths: a figure is set at one of these two so it is never rescaled in the template.
SINGLE_COLUMN_IN: Final[float] = 3.5
"""Width of a single-column figure, in inches."""
DOUBLE_COLUMN_IN: Final[float] = 7.16
"""Width of a double-column figure, in inches."""

# A validated categorical order (adjacent pairs pass the colour-vision-deficiency gate); slots are assigned by role
# and never cycled, so the same entity keeps its colour across every schematic.
BLUE: Final[str] = '#2a78d6'
ORANGE: Final[str] = '#eb6834'
AQUA: Final[str] = '#1baf7a'
YELLOW: Final[str] = '#eda100'
VIOLET: Final[str] = '#4a3aa7'
RED: Final[str] = '#e34948'
INK: Final[str] = '#0b0b0b'
INK_2: Final[str] = '#52514e'
MUTED: Final[str] = '#9a9891'
SURFACE: Final[str] = '#fcfcfb'
FILL: Final[str] = '#f0efec'

# Roles: the EEG side is blue, the frozen text side is orange, the objective is aqua, a nuisance or control is red.
EEG: Final[str] = BLUE
TEXT: Final[str] = ORANGE
OBJECTIVE: Final[str] = AQUA
FROZEN: Final[str] = MUTED

# One-hue sequential ramp for magnitude and a blue-red diverging ramp with a neutral midpoint for signed values.
_SEQUENTIAL: Final[tuple[str, ...]] = ('#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b')
_DIVERGING: Final[tuple[str, ...]] = ('#0d366b', '#3987e5', '#9ec5f4', '#f0efec', '#f3a3a2', '#e34948', '#8a1f1f')
_ORDINAL: Final[tuple[str, ...]] = (
    '#86b6ef',
    '#6da7ec',
    '#5598e7',
    '#3987e5',
    '#2a78d6',
    '#256abf',
    '#1c5cab',
    '#184f95',
)

SEQUENTIAL_CMAP: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list('zte_sequential', _SEQUENTIAL)
"""Single-hue magnitude colormap."""
DIVERGING_CMAP: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list('zte_diverging', _DIVERGING)
"""Blue-red signed colormap with a neutral midpoint."""
REGION_CMAP: Final[ListedColormap] = ListedColormap(_ORDINAL, name='zte_regions')
"""Ordinal colours for the eight anterior-to-posterior scalp regions."""

# The plotted radius of the scalp in metres, the size mne draws its head outline for.
HEAD_RADIUS_M: Final[float] = 0.095

# Matplotlib types every rc key as one Literal of all its parameters, which no local dict can spell; `Any` keys keep
# every `rc_context` call site checkable.
RC: Final[dict[Any, Any]] = {
    'font.family': 'sans-serif',
    'font.size': 7.5,
    'axes.titlesize': 7.5,
    'axes.labelsize': 7.0,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'axes.edgecolor': INK_2,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
}


# ---- Drawing helpers ---- #


def box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str = '',
    *,
    sub: str | None = None,
    fill: str = FILL,
    edge: str = INK_2,
    dashed: bool = False,
    size: float = 7.0,
    ink: str = INK,
) -> None:
    """Draws a rounded box centred at `(x, y)` with a one-line label and an optional smaller line beneath it."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle='round,pad=0.0,rounding_size=0.12',
        facecolor=fill,
        edgecolor=edge,
        linewidth=0.7,
        linestyle='--' if dashed else '-',
    )
    ax.add_patch(patch)
    if label and sub:
        ax.text(x, y + 0.12 * h, label, ha='center', va='center', fontsize=size, color=ink)
        ax.text(x, y - 0.24 * h, sub, ha='center', va='center', fontsize=size - 1.5, color=INK_2)
    elif label:
        ax.text(x, y, label, ha='center', va='center', fontsize=size, color=ink)


def arrow(ax: Axes, x0: float, y0: float, x1: float, y1: float, *, color: str = INK_2, style: str = '-|>') -> None:
    """Draws a thin arrow between two points."""
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, linewidth=0.7, color=color, shrinkA=0, shrinkB=0
        )
    )


def blank(ax: Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Turns an axes into a drawing canvas."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')


def figure(width: float, height: float) -> tuple[Figure, Axes]:
    """A single-axes canvas at a journal width."""
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    return fig, ax


def traces(ax: Axes, x0: float, y0: float, w: float, h: float, n: int = 6, seed: int = 0) -> None:
    """A stack of `n` schematic EEG traces inside a box."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 120)
    for i in range(n):
        base = y0 - h / 2 + (i + 0.5) * h / n
        wave = 0.2 * (h / n) * np.sin(2 * math.pi * (2 + i) * t + rng.uniform(0, 6)) + 0.12 * (
            h / n
        ) * rng.standard_normal(t.size)
        ax.plot(x0 - w / 2 + 0.05 * w + 0.9 * w * t, base + wave, color=EEG, linewidth=0.5)


def sinusoids(ax: Axes, x0: float, y0: float, w: float, h: float, n: int = 4) -> None:
    """A small bank of band-pass filters."""
    t = np.linspace(0, 1, 100)
    for i in range(n):
        base = y0 - h / 2 + (i + 0.5) * h / n
        ax.plot(
            x0 - w / 2 + 0.06 * w + 0.88 * w * t,
            base + 0.3 * (h / n) * np.sin(2 * math.pi * (1.5 + 1.5 * i) * t),
            color=INK_2,
            linewidth=0.5,
        )


def head(ax: Axes, cx: float, cy: float, r: float, *, color: str = INK_2) -> None:
    """A head outline seen from above: circle, nose and ears."""
    ax.add_patch(Circle((cx, cy), r, facecolor='none', edgecolor=color, linewidth=0.7))
    ax.add_patch(
        Polygon(
            [[cx - 0.1 * r, cy + 0.98 * r], [cx, cy + 1.15 * r], [cx + 0.1 * r, cy + 0.98 * r]],
            closed=False,
            fill=False,
            edgecolor=color,
            linewidth=0.7,
        )
    )
    for sx in (-1, 1):
        ax.add_patch(
            Polygon(
                [[cx + sx * 0.98 * r, cy + 0.15 * r], [cx + sx * 1.08 * r, cy], [cx + sx * 0.98 * r, cy - 0.15 * r]],
                closed=False,
                fill=False,
                edgecolor=color,
                linewidth=0.7,
            )
        )


def montage() -> tuple[np.ndarray, list[str], list[str]]:
    """The packaged ZuCo-105 montage: unit-sphere coordinates, labels and regions."""
    montage = load_montage_csv(packaged_montage_csv(), 105)
    if montage is None:
        raise RuntimeError('the packaged montage could not be read')

    return montage.xyz, montage.labels, montage.regions


def topomap(
    ax: Axes,
    values: np.ndarray,
    xyz: np.ndarray,
    *,
    cmap: Any,
    vlim: tuple[float, float],
    labels: Sequence[str] | None = None,
    contours: int = 6,
    dots: bool = True,
) -> Any:
    """Draws a scalp map of `values` on real coordinates: mne's topomap when importable, the in-house one otherwise."""
    unit = xyz / np.clip(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-8, None)
    try:
        import mne

        names = list(labels) if labels is not None else [f'ch{c:03d}' for c in range(len(values))]
        info = mne.create_info(names, sfreq=500.0, ch_types='eeg')
        info.set_montage(
            mne.channels.make_dig_montage(
                ch_pos=dict(zip(names, unit * HEAD_RADIUS_M, strict=True)), coord_frame='head'
            )
        )
        image, _ = mne.viz.plot_topomap(
            values,
            info,
            axes=ax,
            show=False,
            cmap=cmap,
            vlim=vlim,
            sphere=(0.0, 0.0, 0.0, HEAD_RADIUS_M),
            contours=contours,
            sensors=dots,
            outlines='head',
        )
        return image
    except ImportError:
        xy = azimuthal_xy(unit)
        xy = xy / max(float(np.abs(xy).max()), 1e-9) * 0.95
        image = ax.tricontourf(xy[:, 0], xy[:, 1], values, levels=24, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        if dots:
            ax.scatter(xy[:, 0], xy[:, 1], s=2.5, color=INK, linewidths=0)
        head(ax, 0.0, 0.0, 1.0)
        blank(ax, (-1.2, 1.2), (-1.2, 1.25))
        return image


# ---- Glyphs shared by every family ---- #


def tensor_slab(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    depth: float,
    *,
    fill: str = '#e6f0fb',
    edge: str = EEG,
    dims: tuple[str, str, str] | None = None,
    size: float = 5.5,
) -> None:
    """Draws a feature-map slab: a rectangle extruded by `depth` along the isometric diagonal, dims on its edges.

    Args:
        ax (Axes): The canvas.
        x (float): Left edge of the front face.
        y (float): Bottom edge of the front face.
        w (float): Front-face width (time, by convention).
        h (float): Front-face height (channels or filters).
        depth (float): Extrusion length (the third axis); zero draws a flat rectangle.
        fill (str, optional): Front-face fill. The top and side faces are drawn darker.
        edge (str, optional): Stroke colour.
        dims (tuple[str, str, str] | None, optional): Labels for width, height and depth, written along those edges.
        size (float, optional): Label font size.
    """
    dx, dy = 0.45 * depth, 0.35 * depth
    if depth > 0:
        ax.add_patch(
            Polygon(
                [[x, y + h], [x + dx, y + h + dy], [x + w + dx, y + h + dy], [x + w, y + h]],
                closed=True,
                facecolor=_shade(fill, 0.85),
                edgecolor=edge,
                linewidth=0.6,
            )
        )
        ax.add_patch(
            Polygon(
                [[x + w, y], [x + w + dx, y + dy], [x + w + dx, y + h + dy], [x + w, y + h]],
                closed=True,
                facecolor=_shade(fill, 0.7),
                edgecolor=edge,
                linewidth=0.6,
            )
        )
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=0.6))
    if dims is not None:
        width_label, height_label, depth_label = dims
        if width_label:
            ax.text(x + w / 2, y - 0.08 * h - 0.05, width_label, ha='center', va='top', fontsize=size, color=INK_2)
        if height_label:
            ax.text(x - 0.06, y + h / 2, height_label, ha='right', va='center', fontsize=size, color=INK_2, rotation=90)
        if depth_label and depth > 0:
            ax.text(x + w + dx + 0.06, y + h + dy / 2, depth_label, ha='left', va='center', fontsize=size, color=INK_2)


def _shade(colour: str, factor: float) -> tuple[float, float, float]:
    """Darkens a hex colour by `factor` (1 leaves it unchanged)."""
    r, g, b = (int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5))

    return (r * factor, g * factor, b * factor)


def plate(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    copies: int = 3,
    offset: float = 0.08,
    label: str = '',
    fill: str = '#e6f0fb',
    edge: str = EEG,
    times: str | None = None,
) -> None:
    """Draws a repeated block as stacked offset copies, front copy labelled, with an optional `x N` at the corner.

    Args:
        ax (Axes): The canvas.
        x (float): Front copy's centre x.
        y (float): Front copy's centre y.
        w (float): Copy width.
        h (float): Copy height.
        copies (int, optional): How many copies to stack. Defaults to 3.
        offset (float, optional): Diagonal offset between copies. Defaults to 0.08.
        label (str, optional): Text on the front copy.
        fill (str, optional): Front-copy fill; the copies behind are drawn lighter.
        edge (str, optional): Stroke colour.
        times (str | None, optional): The repetition count written at the top-right corner, e.g. `'x 4'`.
    """
    for i in reversed(range(copies)):
        shift = i * offset
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2 + shift, y - h / 2 + shift),
                w,
                h,
                boxstyle='round,pad=0.0,rounding_size=0.1',
                facecolor=fill if i == 0 else 'white',
                edgecolor=edge,
                linewidth=0.6,
            )
        )
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=7, color=INK)
    if times:
        ax.text(
            x + w / 2 + (copies - 1) * offset + 0.08,
            y + h / 2 + (copies - 1) * offset,
            times,
            ha='left',
            va='center',
            fontsize=7,
            color=INK,
        )


def lock(ax: Axes, x: float, y: float, size: float = 0.25, *, color: str = FROZEN) -> None:
    """Draws a padlock glyph centred at `(x, y)`, the mark of a frozen module."""
    body_w, body_h = size, 0.75 * size
    ax.add_patch(Rectangle((x - body_w / 2, y - body_h / 2), body_w, body_h, facecolor=color, edgecolor='none'))
    ax.add_patch(Circle((x, y + body_h / 2), 0.32 * size, facecolor='none', edgecolor=color, linewidth=1.1))
    ax.add_patch(
        Rectangle((x - 0.32 * size - 0.02, y), 0.64 * size + 0.04, body_h / 2, facecolor='white', edgecolor='none')
    )
    ax.add_patch(Rectangle((x - body_w / 2, y - body_h / 2), body_w, body_h, facecolor=color, edgecolor='none'))


def snowflake(ax: Axes, x: float, y: float, size: float = 0.22, *, color: str = FROZEN) -> None:
    """Draws a six-spoke snowflake glyph centred at `(x, y)`, the other mark of a frozen module."""
    for k in range(3):
        angle = math.pi * k / 3
        dx, dy = size * math.cos(angle), size * math.sin(angle)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, linewidth=1.0, solid_capstyle='round')
        for sign in (-1, 1):
            tip = (x + sign * 0.6 * dx, y + sign * 0.6 * dy)
            for turn in (-1, 1):
                a2 = angle + turn * math.pi / 3
                ax.plot(
                    [tip[0], tip[0] + sign * 0.35 * size * math.cos(a2)],
                    [tip[1], tip[1] + sign * 0.35 * size * math.sin(a2)],
                    color=color,
                    linewidth=0.8,
                    solid_capstyle='round',
                )


def dim_label(ax: Axes, x0: float, y0: float, x1: float, y1: float, text: str, *, size: float = 5.5) -> None:
    """A thin dimension line with end ticks and a centred label, for annotating a size along an edge."""
    ax.plot([x0, x1], [y0, y1], color=INK_2, linewidth=0.5)
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length * 0.05, dx / length * 0.05
    for px, py in ((x0, y0), (x1, y1)):
        ax.plot([px - nx, px + nx], [py - ny, py + ny], color=INK_2, linewidth=0.5)
    ax.text(
        (x0 + x1) / 2 + 2.2 * nx, (y0 + y1) / 2 + 2.2 * ny, text, ha='center', va='center', fontsize=size, color=INK_2
    )


def bracket(ax: Axes, x: float, y0: float, y1: float, text: str, *, side: int = 1, size: float = 7.0) -> None:
    """A vertical brace beside a span, with `text` outside it: the way a repeated group is counted."""
    tick = 0.12 * side
    ax.plot([x, x + tick, x + tick, x], [y0, y0, y1, y1], color=INK_2, linewidth=0.6)
    ax.text(
        x + 2.2 * tick, (y0 + y1) / 2, text, ha='left' if side > 0 else 'right', va='center', fontsize=size, color=INK
    )


# ---- Writing ---- #


@dataclass(slots=True, frozen=True, kw_only=True)
class Rendered:
    """One schematic's written files.

    Attributes:
        name (str): The schematic's name.
        paths (tuple[Path, ...]): Every file written for it, one per format.
    """

    name: str
    paths: tuple[Path, ...]


def save_figure(fig: Figure, out_dir: Path, name: str, formats: Sequence[str] = ('png', 'svg')) -> Rendered:
    """Writes one figure in every requested format and closes it.

    Args:
        fig (Figure): The figure to write.
        out_dir (Path): Destination directory, created if needed.
        name (str): File stem.
        formats (Sequence[str], optional): Extensions to write; PNG is rasterised at 300 dpi. Defaults to PNG and
            SVG.

    Returns:
        Rendered: The written paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in formats:
        path = out_dir / f'{name}.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight', pad_inches=0.02)
        paths.append(path)
    plt.close(fig)

    return Rendered(name=name, paths=tuple(paths))
