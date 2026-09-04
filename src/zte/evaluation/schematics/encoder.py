"""The encoder schematics: the pipeline, the stack, one transformer block and the conformer frontend."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path

from zte.evaluation.schematics._style import (
    DIVERGING_CMAP,
    DOUBLE_COLUMN_IN,
    EEG,
    FILL,
    FROZEN,
    INK,
    INK_2,
    OBJECTIVE,
    SEQUENTIAL_CMAP,
    SINGLE_COLUMN_IN,
    TEXT,
    Axes,
    arrow,
    blank,
    box,
    bracket,
    figure,
    head,
    montage,
    plate,
    sinusoids,
    snowflake,
    tensor_slab,
    traces,
)
from zte.lens.montage import azimuthal_xy

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Light tints of the two side hues, and of the objective hue, so black text and thin strokes stay legible on them.
EEG_TINT: Final[str] = '#e6f0fb'
"""Fill of a trainable EEG-side module."""
TEXT_TINT: Final[str] = '#fdeee7'
"""Fill of a text-side module or vector."""
POSITIVE_TINT: Final[str] = '#d2f0e3'
"""Fill of a positive cell in the similarity square."""
GRID: Final[str] = '#d8d7d2'
"""Stroke of the similarity square's cell grid."""

# One stroke family: the helpers draw boxes and arrows at 0.7 pt, so lanes and glyph strokes sit just under it.
LANE_LW: Final[float] = 0.6
"""Stroke of a residual lane, a bus or a glyph outline."""

# The batch drawn in the similarity square: readings ordered so that copies of one sentence are adjacent, giving
# the 2 x 2, 3 x 3 and 1 x 1 positive blocks that are the multi-positive mask itself.
SENTENCES: Final[str] = 'aabbbc'
"""Sentence letter of each reading in the drawn batch."""
READERS: Final[str] = 'ABABCA'
"""Reader tag of each reading in the drawn batch."""


# ---- Glyphs ---- #


def _plus(ax: Axes, x: float, y: float, r: float = 0.15) -> None:
    """A drawn residual-add node: a white circle with a plus made of two strokes."""
    ax.add_patch(Circle((x, y), r, facecolor='white', edgecolor=INK_2, linewidth=0.7, zorder=3))
    ax.plot([x - 0.55 * r, x + 0.55 * r], [y, y], color=INK_2, linewidth=0.7, zorder=4)
    ax.plot([x, x], [y - 0.55 * r, y + 0.55 * r], color=INK_2, linewidth=0.7, zorder=4)


def _times(ax: Axes, x: float, y: float, r: float = 0.15) -> None:
    """A drawn gain node: a white circle with a cross made of two strokes."""
    ax.add_patch(Circle((x, y), r, facecolor='white', edgecolor=INK_2, linewidth=0.7, zorder=3))
    d = 0.42 * r
    ax.plot([x - d, x + d], [y - d, y + d], color=INK_2, linewidth=0.7, zorder=4)
    ax.plot([x - d, x + d], [y + d, y - d], color=INK_2, linewidth=0.7, zorder=4)


def _lane(
    ax: Axes,
    points: list[tuple[float, float]],
    *,
    color: str = INK_2,
    dashed: bool = False,
    radius: float = 0.12,
    headed: bool = True,
) -> None:
    """An orthogonal route with rounded elbows and an arrowhead at its end."""
    vertices: list[tuple[float, float]] = [points[0]]
    codes: list[np.uint8] = [Path.MOVETO]
    for previous, corner, following in zip(points[:-2], points[1:-1], points[2:], strict=True):
        ux, uy = corner[0] - previous[0], corner[1] - previous[1]
        vx, vy = following[0] - corner[0], following[1] - corner[1]
        lu, lv = math.hypot(ux, uy), math.hypot(vx, vy)
        r = min(radius, lu / 2, lv / 2)
        vertices += [
            (corner[0] - r * ux / lu, corner[1] - r * uy / lu),
            corner,
            (corner[0] + r * vx / lv, corner[1] + r * vy / lv),
        ]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    vertices.append(points[-1])
    codes.append(Path.LINETO)
    ax.add_patch(
        FancyArrowPatch(
            path=Path(vertices, codes),
            arrowstyle='-|>' if headed else '-',
            mutation_scale=7,
            linewidth=LANE_LW,
            color=color,
            linestyle=(0, (3, 2)) if dashed else '-',
            zorder=2,
        )
    )


def _vector(
    ax: Axes,
    x: float,
    y: float,
    h: float,
    *,
    cells: int = 6,
    w: float = 0.15,
    fill: str = EEG_TINT,
    edge: str = EEG,
    dim: str = '',
    symbol: str = '',
    horizontal: bool = False,
) -> None:
    """A vector drawn as a bar of cells centred at `(x, y)`, its length written beside it and its symbol beyond."""
    if horizontal:
        w, h = h, w
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fill, edgecolor=edge, linewidth=0.6, zorder=3))
    for k in range(1, cells):
        if horizontal:
            cx = x - w / 2 + k * w / cells
            ax.plot([cx, cx], [y - h / 2, y + h / 2], color=edge, linewidth=0.4, zorder=4)
        else:
            cy = y - h / 2 + k * h / cells
            ax.plot([x - w / 2, x + w / 2], [cy, cy], color=edge, linewidth=0.4, zorder=4)
    if dim:
        if horizontal:
            ax.text(x + w / 2 + 0.08, y, dim, ha='left', va='center', fontsize=5.5, color=INK_2)
        else:
            ax.text(x, y - h / 2 - 0.1, dim, ha='center', va='top', fontsize=5.5, color=INK_2)
    if symbol:
        if horizontal:
            ax.text(x - w / 2 - 0.1, y, symbol, ha='right', va='center', fontsize=8, color=edge, style='italic')
        else:
            ax.text(x, y + h / 2 + 0.12, symbol, ha='center', va='bottom', fontsize=8, color=edge, style='italic')


def _cards(ax: Axes, x: float, y: float, w: float, h: float, text: str, *, edge: str = TEXT) -> None:
    """A batch of sentences as three offset cards, the front one carrying a real ZuCo sentence."""
    for k in (2, 1, 0):
        shift = 0.07 * k
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2 + shift, y - h / 2 + shift),
                w,
                h,
                boxstyle='round,pad=0.0,rounding_size=0.08',
                facecolor='white',
                edgecolor=edge,
                linewidth=0.6,
                zorder=3 - k,
            )
        )
    ax.text(x, y, text, ha='center', va='center', fontsize=5.5, color=INK, family='monospace', zorder=4)


def _trapezoid(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str,
    edge: str,
    dashed: bool = False,
    label: str = '',
    upward: bool = False,
) -> None:
    """An encoder as a trapezoid narrowing in the flow direction, centred at `(x, y)`."""
    narrow = 0.71
    if upward:
        points = [
            [x - w / 2, y - h / 2],
            [x + w / 2, y - h / 2],
            [x + narrow * w / 2, y + h / 2],
            [x - narrow * w / 2, y + h / 2],
        ]
    else:
        points = [
            [x - w / 2, y - h / 2],
            [x - w / 2, y + h / 2],
            [x + w / 2, y + narrow * h / 2],
            [x + w / 2, y - narrow * h / 2],
        ]
    ax.add_patch(
        Polygon(
            points,
            closed=True,
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.7,
            linestyle=(0, (3, 2)) if dashed else '-',
            zorder=3,
        )
    )
    if label:
        ax.text(x - (0 if upward else 0.04 * w), y, label, ha='center', va='center', fontsize=6.5, color=INK, zorder=4)


def _stacked(
    ax: Axes,
    x0: float,
    y0: float,
    w: float,
    h: float,
    *,
    copies: int = 3,
    offset: float = 0.09,
    fill: str = FILL,
    edge: str = INK_2,
    rounding: float = 0.25,
    times: str = '',
) -> None:
    """A repeated container drawn as stacked offset copies, the front one holding the contents drawn after it."""
    for k in reversed(range(copies)):
        shift = k * offset
        ax.add_patch(
            FancyBboxPatch(
                (x0 + shift, y0 + shift),
                w,
                h,
                boxstyle=f'round,pad=0.0,rounding_size={rounding}',
                facecolor=fill if k == 0 else 'white',
                edgecolor=edge,
                linewidth=0.7,
                zorder=1,
            )
        )
    if times:
        shift = (copies - 1) * offset
        ax.text(x0 + w + shift + 0.1, y0 + h + shift, times, ha='left', va='center', fontsize=7, color=INK)


def _kernel(ax: Axes, x: float, y: float, w: float, h: float, dim: str) -> None:
    """A convolution kernel drawn inside its input map, its width in samples written above it in the EEG hue."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=EEG, alpha=0.35, edgecolor=EEG, linewidth=0.6, zorder=5))
    ax.text(x + w / 2, y + h + 0.16, dim, ha='center', va='bottom', fontsize=5.5, color=EEG, zorder=6)


def _attention_map(ax: Axes, x: float, y: float, s: float, *, dims: str = '350') -> None:
    """A tiny self-attention map over the window's samples: banded near the diagonal, its side written on two edges."""
    rng = np.random.default_rng(4)
    n = 48
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    scores = np.exp(-np.abs(i - j) / 5.0) + 0.25 * rng.uniform(0, 1, (n, n))
    ax.imshow(scores, extent=(x, x + s, y, y + s), cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1.25, zorder=3)
    ax.add_patch(Rectangle((x, y), s, s, facecolor='none', edgecolor=EEG, linewidth=0.6, zorder=4))
    ax.text(x + s / 2, y - 0.08, dims, ha='center', va='top', fontsize=5.5, color=INK_2)
    ax.text(x - 0.08, y + s / 2, dims, ha='right', va='center', fontsize=5.5, color=INK_2, rotation=90)


def _harmonic_head(ax: Axes, cx: float, cy: float, r: float) -> None:
    """A head seen from above, tinted by a real spherical harmonic evaluated on the real 105-electrode montage."""
    xyz, _, _ = montage()
    xy = azimuthal_xy(xyz)
    xy = xy / float(np.abs(xy).max()) * 0.92 * r
    unit = xyz / np.clip(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-8, None)
    # Y_2^2 is proportional to x^2 - y^2: four lobes, unmistakably a spatial code rather than a blob.
    values = unit[:, 0] ** 2 - unit[:, 1] ** 2
    limit = float(np.abs(values).max())
    field = ax.tricontourf(
        cx + xy[:, 0], cy + xy[:, 1], values, levels=12, cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit, zorder=2
    )
    field.set_clip_path(Circle((cx, cy), r, transform=ax.transData))
    head(ax, cx, cy, r)


def _transport(ax: Axes, x: float, y: float, w: float, h: float) -> None:
    """The whitening transport as a glyph: a tilted covariance ellipse becoming the unit circle."""
    box(ax, x, y, w, h, fill=FILL, edge=INK_2)
    ax.add_patch(
        Ellipse(
            (x - 0.26 * w, y), 0.4 * h, 0.22 * h, angle=35, facecolor='none', edgecolor=EEG, linewidth=0.7, zorder=4
        )
    )
    arrow(ax, x - 0.06 * w, y, x + 0.1 * w, y, color=INK_2)
    ax.add_patch(Circle((x + 0.26 * w, y), 0.15 * h, facecolor='none', edgecolor=EEG, linewidth=0.7, zorder=4))


def _pool_glyph(
    ax: Axes,
    x: float,
    y: float,
    *,
    cell: float = 0.22,
    pitch: float = 0.27,
    n: int = 5,
    absent: int = 1,
    vertical: bool = False,
) -> float:
    """Masked attention pooling: a line of word tokens, one absent, fanning into one vector; returns the exit edge."""
    reach = 0.85
    out = (x, y + reach) if vertical else (x + reach, y)
    for k in range(n):
        offset = (k - (n - 1) / 2) * pitch
        cx, cy = (x + offset, y) if vertical else (x, y - offset)
        skipped = k == absent
        ax.add_patch(
            Rectangle(
                (cx - cell / 2, cy - cell / 2),
                cell,
                cell,
                facecolor=FILL if skipped else EEG_TINT,
                edgecolor=INK_2 if skipped else EEG,
                linewidth=0.6,
                linestyle=(0, (2, 1.5)) if skipped else '-',
                zorder=3,
            )
        )
        if skipped:
            continue
        if vertical:
            ax.plot([cx, out[0]], [cy + cell / 2, out[1] - cell / 2], color=EEG, linewidth=0.45, zorder=2)
        else:
            ax.plot([cx + cell / 2, out[0] - cell / 2], [cy, out[1]], color=EEG, linewidth=0.45, zorder=2)
    ax.add_patch(
        Rectangle(
            (out[0] - cell / 2, out[1] - cell / 2),
            cell,
            cell,
            facecolor=EEG_TINT,
            edgecolor=EEG,
            linewidth=0.6,
            zorder=3,
        )
    )

    return (out[1] if vertical else out[0]) + cell / 2


# `tags` names the edge the reader tags sit on ('bottom' under the columns, 'right' beside the rows, '' for none) and
# must be the EEG side of the square: a text vector has no reader.
def _similarity_square(
    ax: Axes,
    x0: float,
    y0: float,
    cell: float,
    *,
    header: float = 0.24,
    gap: float = 0.1,
    row_hue: str = TEXT,
    col_hue: str = EEG,
    tags: str = 'bottom',
) -> None:
    """The similarity square with its multi-positive blocks tinted, tower-tinted headers and reader tags."""
    n = len(SENTENCES)
    side = n * cell
    for i in range(n):
        for j in range(n):
            positive = SENTENCES[i] == SENTENCES[j]
            ax.add_patch(
                Rectangle(
                    (x0 + j * cell, y0 + side - (i + 1) * cell),
                    cell,
                    cell,
                    facecolor=POSITIVE_TINT if positive else 'white',
                    edgecolor=GRID,
                    linewidth=0.4,
                    zorder=2,
                )
            )
    ax.add_patch(Rectangle((x0, y0), side, side, facecolor='none', edgecolor=OBJECTIVE, linewidth=0.8, zorder=3))

    # Row headers (left) and column headers (top), each tinted like the tower it came from.
    row_fill = TEXT_TINT if row_hue == TEXT else EEG_TINT
    col_fill = TEXT_TINT if col_hue == TEXT else EEG_TINT
    for k, letter in enumerate(SENTENCES):
        ry = y0 + side - (k + 0.5) * cell
        ax.add_patch(
            Rectangle(
                (x0 - gap - header, ry - cell / 2),
                header,
                cell,
                facecolor=row_fill,
                edgecolor=row_hue,
                linewidth=0.5,
                zorder=3,
            )
        )
        ax.text(
            x0 - gap - header / 2, ry, letter, ha='center', va='center', fontsize=5.5, color=row_hue, style='italic'
        )
        cx = x0 + (k + 0.5) * cell
        ax.add_patch(
            Rectangle(
                (cx - cell / 2, y0 + side + gap),
                cell,
                header,
                facecolor=col_fill,
                edgecolor=col_hue,
                linewidth=0.5,
                zorder=3,
            )
        )
        ax.text(
            cx,
            y0 + side + gap + header / 2,
            letter,
            ha='center',
            va='center',
            fontsize=5.5,
            color=col_hue,
            style='italic',
        )
        if tags == 'bottom':
            ax.text(cx, y0 - 0.08, READERS[k], ha='center', va='top', fontsize=5.5, color=INK_2)
        elif tags == 'right':
            ax.text(x0 + side + 0.08, ry, READERS[k], ha='left', va='center', fontsize=5.5, color=INK_2)


def _chip(ax: Axes, x: float, y: float, text: str, *, size: float = 0.28) -> None:
    """One tinted cell with its meaning beside it: the only legend the square needs."""
    ax.add_patch(Rectangle((x, y - size / 2), size, size, facecolor=POSITIVE_TINT, edgecolor=OBJECTIVE, linewidth=0.6))
    ax.text(x + size + 0.12, y, text, ha='left', va='center', fontsize=6, color=INK_2)


def _note(ax: Axes, x: float, y: float, text: str, *, size: float = 5.5, color: str = INK_2, **kwargs: object) -> None:
    """A small grey annotation."""
    ax.text(x, y, text, fontsize=size, color=color, **kwargs)


# ---- The pipeline ---- #


def _pipeline(words: bool) -> Figure:
    """The whole encoder left to right meeting the frozen text tower at the square; `words` toggles the block names."""
    fig, ax = figure(DOUBLE_COLUMN_IN, 1.95)
    blank(ax, (0.15, 19.85), (0.3, 5.75))
    y = 4.35

    # Raw window: a slab whose front face is real EEG, dims on its edges, depth = the L words of a sentence.
    tensor_slab(ax, 0.3, y - 0.375, 2.0, 0.75, 0.24, fill=FILL, edge=INK_2, dims=('350', '105', 'L'))
    traces(ax, 1.3, y, 1.85, 0.62, n=6)
    arrow(ax, 2.41, y, 2.85, y, color=EEG)

    # Transport by the reader's own covariance root: not a trainable module, so it keeps the neutral fill.
    _transport(ax, 3.3, y, 0.9, 0.7)
    _note(ax, 3.3, y - 0.5, r'$R_s^{-1/2}$', size=6, ha='center', va='top')
    arrow(ax, 3.75, y, 4.19, y, color=EEG)

    # The harmonic electrode code is added, the way a sequence model adds position: a head joined to a plus.
    _plus(ax, 4.35, y, 0.16)
    ax.plot([4.35, 4.35], [y + 0.16, 4.86], color=INK_2, linewidth=LANE_LW)
    _harmonic_head(ax, 4.35, 5.22, 0.36)
    _note(ax, 4.82, 5.22, r'$Y_\ell^{\,m}$', size=6.5, ha='left', va='center')
    arrow(ax, 4.51, y, 4.95, y, color=EEG)

    # Conformer frontend: (40, 350) map -> 40 -> 256, one token per word.
    ax.add_patch(
        FancyBboxPatch(
            (4.95, 3.55),
            3.7,
            1.55,
            boxstyle='round,pad=0.0,rounding_size=0.2',
            facecolor='white',
            edgecolor=EEG,
            linewidth=0.7,
            zorder=1,
        )
    )
    if words:
        _note(ax, 5.1, 4.97, 'conformer', size=6.5, color=EEG, ha='left', va='center')
    tensor_slab(ax, 5.45, y - 0.18, 1.8, 0.36, 0.2, dims=('350', '40', ''))
    _kernel(ax, 5.45 + 0.7 * 1.8, y - 0.18, 25 / 350 * 1.8, 0.36, '25')
    arrow(ax, 7.34, y, 7.75, y, color=EEG)
    _vector(ax, 7.825, y, 0.45, cells=4, dim='40')
    arrow(ax, 7.9, y, 8.35, y, color=EEG)
    _vector(ax, 8.425, y, 0.95, cells=6, dim='256')
    arrow(ax, 8.5, y, 9.7, y, color=EEG)
    _note(ax, 9.2, y + 0.12, 'L × 256', ha='center', va='bottom')

    # Four rotary blocks over the L tokens, drawn as a plate.
    plate(ax, 10.6, y, 1.8, 1.0, times='× 4')
    if words:
        ax.text(10.6, y + 0.13, 'transformer', ha='center', va='center', fontsize=7, color=INK)
        _note(ax, 10.6, y - 0.2, 'RoPE', size=5.5, ha='center', va='center')
    arrow(ax, 11.5, y, 12.39, y, color=EEG)

    # Masked attention pool and the projection to the unit sphere.
    x_pool_out = _pool_glyph(ax, 12.5, y)
    if words:
        _note(ax, 12.92, y - 0.78, 'pool', size=6, ha='center', va='top')
    arrow(ax, x_pool_out, y, 13.95, y, color=EEG)
    box(ax, 14.55, y, 1.2, 0.65, 'proj' if words else '', fill=EEG_TINT, edge=EEG)
    _note(ax, 14.55, y - 0.48, '256 → 512 → 768', ha='center', va='top')
    arrow(ax, 15.15, y, 15.6, y, color=EEG)
    _vector(ax, 15.675, y, 1.35, cells=8, dim='768', symbol='z')

    # The EEG embeddings run along a bus into the square's column headers.
    square_x0, square_y0, cell = 17.3, 0.95, 0.4
    side = len(SENTENCES) * cell
    ax.plot([15.75, square_x0 + side - cell / 2], [y, y], color=EEG, linewidth=LANE_LW, zorder=2)
    for k in range(len(SENTENCES)):
        cx = square_x0 + (k + 0.5) * cell
        arrow(ax, cx, y, cx, square_y0 + side + 0.1 + 0.24, color=EEG)
    _similarity_square(ax, square_x0, square_y0, cell)
    _note(
        ax,
        square_x0 + side / 2,
        y + 0.42,
        r'$S_{ij} = z_i^{\top} t_j\,/\,\tau$',
        size=6.5,
        color=OBJECTIVE,
        ha='center',
        va='center',
    )

    # The frozen text tower: sentence cards, a grey dashed trapezoid with a snowflake, the text embeddings on a bus.
    ty = square_y0 + cell / 2
    _cards(ax, 1.25, ty, 1.8, 0.8, '“He was elected …”')
    arrow(ax, 2.22, ty, 2.7, ty, color=TEXT)
    _trapezoid(ax, 3.35, ty, 1.3, 1.0, fill=TEXT_TINT, edge=FROZEN, dashed=True, label='text\nencoder' if words else '')
    snowflake(ax, 3.9, ty + 0.62, 0.15)
    arrow(ax, 4.0, ty, 4.5, ty, color=TEXT)
    _vector(ax, 4.575, ty, 1.35, cells=8, fill=TEXT_TINT, edge=TEXT, dim='768', symbol='t')
    bus_x = square_x0 - 0.1 - 0.24 - 0.42
    _lane(ax, [(4.65, ty), (bus_x, ty), (bus_x, square_y0 + side - cell / 2)], color=TEXT, headed=False)
    for k in range(len(SENTENCES)):
        ry = square_y0 + side - (k + 0.5) * cell
        arrow(ax, bus_x, ry, square_x0 - 0.1 - 0.24, ry, color=TEXT)

    # The positive definition is the claim: the same sentence read by anyone, so identity has nowhere to live.
    if words:
        _chip(ax, 6.3, 2.3, 'same sentence, any reader')
    _note(
        ax,
        10.9,
        2.3,
        r'$\mathcal{L} = \frac{1}{2}\,[\,\mathrm{InfoNCE}(S) + \mathrm{InfoNCE}(S^{\top})\,]$',
        size=6.5,
        color=OBJECTIVE,
        ha='left',
        va='center',
    )

    return fig


def encoder_pipeline() -> Figure:
    """The whole encoder left to right, with the frozen text tower meeting it at the multi-positive square.

    Returns:
        Figure: A double-column figure.
    """
    return _pipeline(words=True)


def encoder_overview_minimal() -> Figure:
    """The pipeline with every block name removed: glyphs, dimensions and symbols only.

    Returns:
        Figure: A double-column figure.
    """
    return _pipeline(words=False)


# ---- The stack ---- #


def _width(features: int) -> float:
    """Box width for a feature count, on a log scale so 40 and 768 both fit one column."""
    return 1.3 + 1.0 * math.log2(features / 40)


def encoder_stack() -> Figure:
    """The encoder as one column, input at the bottom, box width following the feature width, shapes in a gutter.

    Returns:
        Figure: A single-column figure.
    """
    fig, ax = figure(SINGLE_COLUMN_IN, 4.9)
    blank(ax, (0.3, 9.7), (0.2, 14.2))
    x, gutter, h = 5.0, 8.3, 0.46
    w105, w40, w256, w512, w768 = _width(105), _width(40), _width(256), _width(512), _width(768)

    def shape(y: float, text: str) -> None:
        ax.text(gutter, y, text, ha='left', va='center', fontsize=6, color=INK_2)

    def beside(y: float, text: str) -> None:
        ax.text(x - 0.12, y, text, ha='right', va='center', fontsize=5.5, color=INK_2)

    # Raw window, then the transport by the reader's covariance root.
    tensor_slab(ax, x - w105 / 2, 0.4, w105, 0.72, 0.0, fill=FILL, edge=INK_2)
    traces(ax, x, 0.76, w105 - 0.2, 0.58, n=6)
    shape(0.76, '105 × 350')
    arrow(ax, x, 1.12, x, 1.7 - h / 2, color=EEG)
    box(ax, x, 1.7, w105, h, r'$R_s^{-1/2}$', fill=FILL, edge=INK_2)
    shape(1.7, '105 × 350')

    # Per-electrode gain from the signature hypernetwork, then the harmonic code added and mixed with a residual.
    arrow(ax, x, 1.7 + h / 2, x, 2.35 - 0.14, color=EEG)
    _times(ax, x, 2.35, 0.14)
    arrow(ax, x, 2.49, x, 3.0 - 0.14, color=EEG)
    _plus(ax, x, 3.0, 0.14)
    ax.plot([3.84, x - 0.14], [3.0, 3.0], color=INK_2, linewidth=LANE_LW)
    _harmonic_head(ax, 3.56, 3.0, 0.27)
    ax.text(3.2, 3.0, r'$Y_\ell^{\,m}$', ha='right', va='center', fontsize=6, color=INK_2)
    arrow(ax, x, 3.14, x, 3.7 - h / 2, color=EEG)
    beside(3.3, 'LN')
    box(ax, x, 3.7, w105, h, 'attention', fill=EEG_TINT, edge=EEG)
    shape(3.7, '105 × 350')
    lane_x = x + w105 / 2 + 0.3
    _lane(ax, [(x, 3.3), (lane_x, 3.3), (lane_x, 4.3), (x + 0.14, 4.3)])
    arrow(ax, x, 3.7 + h / 2, x, 4.3 - 0.14, color=EEG)
    _plus(ax, x, 4.3, 0.14)

    # Conformer frontend: two convolutions, two attention layers at width 40, mean over time, the lift to 256.
    arrow(ax, x, 4.44, x, 4.95 - h / 2, color=EEG)
    box(ax, x, 4.95, w40, h, 'conv 25', fill=EEG_TINT, edge=EEG)
    shape(4.95, '40 × 350')
    arrow(ax, x, 4.95 + h / 2, x, 5.65 - h / 2, color=EEG)
    beside(5.3, 'GELU')
    box(ax, x, 5.65, w40, h, 'conv 1', fill=EEG_TINT, edge=EEG)
    shape(5.65, '40 × 350')
    arrow(ax, x, 5.65 + h / 2, x, 6.55 - 0.31, color=EEG)
    beside(6.0, 'GELU · LN')
    _stacked(ax, x - 0.85, 6.55 - 0.31, 1.7, 0.62, fill=EEG_TINT, edge=EEG, times='× 2')
    ax.text(x, 6.55, 'attention', ha='center', va='center', fontsize=6.5, color=INK)
    shape(6.55, '40 × 350')
    arrow(ax, x, 6.55 + 0.31, x, 7.5 - h / 2, color=EEG)
    beside(7.14, 'mean')
    box(ax, x, 7.5, w256, h, 'linear', fill=EEG_TINT, edge=EEG)
    shape(7.5, '256')

    # FiLM from the same hypernetwork, then the four rotary blocks with their residual lanes on the right.
    arrow(ax, x, 7.5 + h / 2, x, 8.15 - 0.14, color=EEG)
    _times(ax, x, 8.15, 0.14)
    plate_lane = x + w256 / 2 + 0.3
    px0, py0, pw, ph = x - w256 / 2 - 0.25, 8.4, w256 + 0.25 + 0.55, 2.95
    _stacked(ax, px0, py0, pw, ph, rounding=0.3)
    ax.text(px0 - 0.15, py0 + ph / 2, '× 4', ha='right', va='center', fontsize=11, color=INK)
    shape(py0 + ph / 2, 'L × 256')
    arrow(ax, x, 8.29, x, 8.95 - h / 2, color=EEG)
    beside(8.5, 'LN')
    _lane(ax, [(x, 8.5), (plate_lane, 8.5), (plate_lane, 9.6), (x + 0.14, 9.6)])
    box(ax, x, 8.95, w256, h, 'attention', fill=EEG_TINT, edge=EEG)
    arrow(ax, x, 8.95 + h / 2, x, 9.6 - 0.14, color=EEG)
    _plus(ax, x, 9.6, 0.14)
    arrow(ax, x, 9.74, x, 10.3 - h / 2, color=EEG)
    beside(9.9, 'LN')
    _lane(ax, [(x, 9.9), (plate_lane, 9.9), (plate_lane, 10.95), (x + 0.14, 10.95)])
    box(ax, x, 10.3, w256, h, 'feed-forward', fill=EEG_TINT, edge=EEG)
    arrow(ax, x, 10.3 + h / 2, x, 10.95 - 0.14, color=EEG)
    _plus(ax, x, 10.95, 0.14)

    # Masked attention pool and the projection head.
    arrow(ax, x, 11.09, x, 11.75 - h / 2, color=EEG)
    box(ax, x, 11.75, w256, h, 'pool', fill=EEG_TINT, edge=EEG)
    shape(11.75, '256')
    arrow(ax, x, 11.75 + h / 2, x, 12.45 - h / 2, color=EEG)
    box(ax, x, 12.45, w512, h, 'linear', fill=EEG_TINT, edge=EEG)
    shape(12.45, '512')
    arrow(ax, x, 12.45 + h / 2, x, 13.15 - h / 2, color=EEG)
    beside(12.8, 'GELU')
    box(ax, x, 13.15, w768, h, 'linear', fill=EEG_TINT, edge=EEG)
    shape(13.15, '768')
    arrow(ax, x, 13.15 + h / 2, x, 13.75, color=EEG)
    ax.text(x, 13.9, 'z', ha='center', va='center', fontsize=8, color=EEG, style='italic')
    shape(13.9, '‖z‖ = 1')

    # The signature hypernetwork: fed from the raw window, emitting the gain and the FiLM as dashed weight arrows.
    hx, hy = 1.35, 5.3
    _lane(ax, [(x - w105 / 2, 0.76), (hx, 0.76), (hx, 4.05)])
    for k, level in enumerate((0.3, 0.75, 0.5, 0.9)):
        ax.add_patch(
            Rectangle(
                (hx - 0.4 + 0.2 * k, 4.15),
                0.2,
                0.2,
                facecolor=SEQUENTIAL_CMAP(level),
                edgecolor=INK_2,
                linewidth=0.4,
                zorder=3,
            )
        )
    ax.text(hx - 0.5, 4.25, 'σ', ha='right', va='center', fontsize=6.5, color=INK_2, style='italic')
    arrow(ax, hx, 4.35, hx, hy - h / 2, color=EEG)
    box(ax, hx, hy, 1.25, h, 'hypernet', fill=EEG_TINT, edge=EEG, size=6.5)
    _lane(ax, [(hx + 0.625, hy), (2.35, hy), (2.35, 2.35), (x - 0.14, 2.35)], dashed=True)
    _lane(ax, [(2.35, hy), (2.35, 8.15), (x - 0.14, 8.15)], dashed=True)

    return fig


# ---- One transformer block ---- #


def _rope_glyph(ax: Axes, x: float, y: float, r: float = 0.3) -> None:
    """Rotary position: a query and a key each turned by its own position, so their score depends on `m - n` only."""
    for k, (cx, angle, name) in enumerate(((x - 0.5, 0.45, 'q'), (x + 0.5, 1.35, 'k'))):
        ax.add_patch(Circle((cx, y), r, facecolor='white', edgecolor=INK_2, linewidth=LANE_LW, zorder=3))
        ax.plot([cx, cx + r * math.cos(angle)], [y, y + r * math.sin(angle)], color=EEG, linewidth=0.8, zorder=4)
        ax.plot([cx, cx + 0.6 * r], [y, y], color=INK_2, linewidth=0.4, zorder=4)
        ax.text(cx, y - r - 0.08, rf'${name}_{"mn"[k]}$', ha='center', va='top', fontsize=6, color=INK_2)
    ax.text(x, y + r + 0.1, r'$(m - n)\,\theta$', ha='center', va='bottom', fontsize=6, color=INK_2)


def transformer_block() -> Figure:
    """One contextualiser block: pre-norm, rotary self-attention, feed-forward, both residual lanes drawn.

    Returns:
        Figure: A single-column figure.
    """
    fig, ax = figure(SINGLE_COLUMN_IN, 3.75)
    blank(ax, (0.8, 9.4), (0.0, 9.3))
    x, w = 4.4, 3.2
    lane_x = 1.85

    # The plate encloses one block; the multiplier sits outside it at mid-height, larger than the labels.
    ax.add_patch(
        FancyBboxPatch(
            (1.0, 0.75),
            7.3,
            7.7,
            boxstyle='round,pad=0.0,rounding_size=0.35',
            facecolor=FILL,
            edgecolor=INK_2,
            linewidth=0.7,
            zorder=0,
        )
    )
    ax.text(8.45, 4.6, '× 4', ha='left', va='center', fontsize=11, color=INK)

    # Trunk, bottom to top.
    arrow(ax, x, 0.0, x, 1.55 - 0.275, color=EEG)
    _note(ax, x + 0.15, 0.3, 'L × 256', ha='left', va='center')
    box(ax, x, 1.55, w, 0.55, 'LayerNorm', fill='white', edge=INK_2)
    for dx in (-0.75, 0.0, 0.75):
        if dx == 0.0:
            arrow(ax, x, 1.825, x, 3.05 - 0.425, color=EEG)
        else:
            _lane(ax, [(x, 1.825), (x, 2.15), (x + dx, 2.15), (x + dx, 3.05 - 0.425)], color=EEG, radius=0.14)
    box(ax, x, 3.05, w, 0.85, 'attention', fill=EEG_TINT, edge=EEG)
    _rope_glyph(ax, 7.15, 3.05)
    arrow(ax, x, 3.475, x, 4.35 - 0.2, color=EEG)
    _plus(ax, x, 4.35, 0.2)
    arrow(ax, x, 4.55, x, 5.35 - 0.275, color=EEG)
    box(ax, x, 5.35, w, 0.55, 'LayerNorm', fill='white', edge=INK_2)
    arrow(ax, x, 5.625, x, 6.55 - 0.425, color=EEG)
    box(ax, x, 6.55, w, 0.85, 'feed-forward', fill=EEG_TINT, edge=EEG)
    arrow(ax, x, 6.975, x, 7.75 - 0.2, color=EEG)
    _plus(ax, x, 7.75, 0.2)
    arrow(ax, x, 7.95, x, 9.2, color=EEG)
    _note(ax, x + 0.15, 8.9, 'L × 256', ha='left', va='center')

    # Residual lanes: each branches before its LayerNorm (pre-norm made visible) and enters the plus from the side.
    _lane(ax, [(x, 1.0), (lane_x, 1.0), (lane_x, 4.35), (x - 0.2, 4.35)])
    _lane(ax, [(x, 4.85), (lane_x, 4.85), (lane_x, 7.75), (x - 0.2, 7.75)])

    return fig


# ---- The conformer frontend ---- #


def _filter_map(ax: Axes, x: float, y: float, w: float, h: float, depth: float) -> float:
    """A (40, 350) map whose height sits on the outgoing edge, so an arrow can land on its left; returns the exit x."""
    tensor_slab(ax, x, y - h / 2, w, h, depth, dims=('350', '', ''))
    dx = 0.45 * depth
    ax.text(x + w + dx + 0.08, y, '40', ha='left', va='center', fontsize=5.5, color=INK_2, rotation=90)

    return x + w + dx + 0.34


def _frontend(ax: Axes, x0: float, y: float, slab_w: float, *, gap: float, compact: bool) -> None:
    """Draws the frontend's slab sequence from `x0` along the baseline `y`; `compact` drops the post-attention map."""
    h105, h40, depth = slab_w * 105 / 350 * 0.95, slab_w * 40 / 350 * 0.95, 0.2

    # Input map with the temporal kernel drawn inside it, and the filter bank it belongs to beneath.
    tensor_slab(ax, x0, y - h105 / 2, slab_w, h105, depth, fill=FILL, edge=INK_2, dims=('350', '105', ''))
    traces(ax, x0 + slab_w / 2, y, slab_w - 0.2, h105 - 0.16, n=7, seed=3)
    kx = x0 + 0.66 * slab_w
    kw = 25 / 350 * slab_w
    _kernel(ax, kx, y - h105 / 2, kw, h105, '25')
    bank_cx, bank_w, bank_h, bank_y = kx + kw / 2 + 0.45, 1.6 if compact else 1.9, 0.7, 0.45
    sinusoids(ax, bank_cx, bank_y, bank_w, bank_h, n=4)
    bracket(ax, bank_cx + bank_w / 2 + 0.05, bank_y - bank_h / 2, bank_y + bank_h / 2, '40', size=6)
    ax.plot(
        [kx + kw / 2, kx + kw / 2],
        [y - h105 / 2, bank_y + bank_h / 2 + 0.06],
        color=EEG,
        linewidth=0.45,
        linestyle=(0, (1, 1.5)),
    )
    x = x0 + slab_w + 0.12
    arrow(ax, x, y, x + gap, y, color=EEG)
    _note(ax, x + gap / 2, y + 0.1, 'GELU', ha='center', va='bottom')

    # Forty filter outputs with the pointwise kernel inside.
    x += gap
    exit_x = _filter_map(ax, x, y, slab_w, h40, depth)
    _kernel(ax, x + 0.66 * slab_w, y - h40 / 2, 0.05, h40, '1')
    x = exit_x
    arrow(ax, x, y, x + gap + 0.3, y, color=EEG)
    _note(ax, x + (gap + 0.3) / 2, y + 0.1, 'GELU · LN', ha='center', va='bottom')

    # Two attention layers over the 350 samples, drawn as a plate holding a tiny attention map.
    x += gap + 0.3
    pw, ph, glyph = (1.6, 1.4, 0.65) if compact else (1.9, 1.55, 0.8)
    _stacked(ax, x, y - ph / 2 - 0.05, pw, ph, fill=EEG_TINT, edge=EEG, times='× 2')
    _attention_map(ax, x + (pw - glyph) / 2 + 0.1, y - ph / 2 + 0.22, glyph)
    _note(ax, x + pw / 2, y + ph / 2 - 0.3, 'attention', size=6, color=INK, ha='center', va='center')
    x += pw
    if not compact:
        arrow(ax, x, y, x + gap, y, color=EEG)
        x = _filter_map(ax, x + gap, y, slab_w, h40, depth)

    # Mean over time to one 40-vector, and the linear lift to the 256-d token.
    arrow(ax, x, y, x + gap, y, color=EEG)
    _note(ax, x + gap / 2, y + 0.1, 'mean', ha='center', va='bottom')
    x += gap + 0.08
    _vector(ax, x, y, 0.5, cells=4, dim='40')
    x += 0.08
    arrow(ax, x, y, x + gap, y, color=EEG)
    _note(ax, x + gap / 2, y + 0.1, 'linear', ha='center', va='bottom')
    x += gap + 0.08
    _vector(ax, x, y, 1.25, cells=8, dim='256', symbol='h')


def conformer_frontend() -> Figure:
    """The temporal frontend as a slab sequence: (105, 350) -> (40, 350) -> attention -> (40, 350) -> 40 -> 256.

    Returns:
        Figure: A double-column figure.
    """
    fig, ax = figure(DOUBLE_COLUMN_IN, 1.2)
    blank(ax, (0.0, 19.6), (0.0, 3.35))
    _frontend(ax, 0.35, 2.15, 3.4, gap=1.0, compact=False)

    return fig


def conformer_frontend_compact() -> Figure:
    """The frontend at single-column width: the same slabs, without the post-attention map.

    Returns:
        Figure: A single-column figure.
    """
    fig, ax = figure(SINGLE_COLUMN_IN, 1.55)
    blank(ax, (0.0, 10.3), (0.0, 3.35))
    _frontend(ax, 0.3, 2.15, 1.8, gap=0.85, compact=True)

    return fig


# ---- The pipeline, portrait ---- #


def encoder_pipeline_vertical() -> Figure:
    """The pipeline as two towers rising to the square: EEG on the left, the frozen text tower on the right.

    Returns:
        Figure: A single-column figure.
    """
    fig, ax = figure(SINGLE_COLUMN_IN, 5.0)
    blank(ax, (0.0, 10.0), (0.0, 14.3))
    x = 2.4

    # Raw window, transport, harmonic code.
    tensor_slab(ax, x - 0.95, 0.35, 1.9, 0.7, 0.22, fill=FILL, edge=INK_2, dims=('350', '105', 'L'))
    traces(ax, x, 0.7, 1.75, 0.58, n=6)
    arrow(ax, x, 1.05, x, 1.5, color=EEG)
    _transport(ax, x, 1.85, 1.0, 0.7)
    _note(ax, x + 0.62, 1.85, r'$R_s^{-1/2}$', size=6, ha='left', va='center')
    arrow(ax, x, 2.2, x, 2.64, color=EEG)
    _plus(ax, x, 2.8, 0.16)
    ax.plot([x - 0.16, 1.32], [2.8, 2.8], color=INK_2, linewidth=LANE_LW)
    _harmonic_head(ax, 0.95, 2.8, 0.36)
    _note(ax, 0.95, 2.3, r'$Y_\ell^{\,m}$', size=6.5, ha='center', va='top')
    arrow(ax, x, 2.96, x, 3.4, color=EEG)

    # Conformer: (40, 350) map -> 40 -> 256, stacked upward.
    ax.add_patch(
        FancyBboxPatch(
            (x - 1.75, 3.4),
            3.35,
            2.55,
            boxstyle='round,pad=0.0,rounding_size=0.2',
            facecolor='white',
            edgecolor=EEG,
            linewidth=0.7,
            zorder=1,
        )
    )
    _note(ax, x - 1.63, 5.8, 'conformer', size=6.5, color=EEG, ha='left', va='center')
    tensor_slab(ax, x - 0.75, 3.7, 1.5, 0.3, 0.18, dims=('350', '40', ''))
    _kernel(ax, x - 0.75 + 0.62 * 1.5, 3.7, 25 / 350 * 1.5, 0.3, '25')
    arrow(ax, x, 4.07, x, 4.55, color=EEG)
    _vector(ax, x, 4.62, 0.5, cells=4, horizontal=True, dim='40')
    arrow(ax, x, 4.7, x, 5.2, color=EEG)
    _vector(ax, x, 5.27, 1.2, cells=6, horizontal=True, dim='256')
    arrow(ax, x, 5.35, x, 6.45, color=EEG)
    _note(ax, x + 0.15, 6.15, 'L × 256', ha='left', va='center')

    # Four rotary blocks, the masked pool, the projection, z.
    plate(ax, x, 6.95, 1.9, 1.0, times='× 4')
    ax.text(x, 7.08, 'transformer', ha='center', va='center', fontsize=7, color=INK)
    _note(ax, x, 6.75, 'RoPE', size=5.5, ha='center', va='center')
    arrow(ax, x, 7.45, x, 8.04, color=EEG)
    y_out = _pool_glyph(ax, x, 8.15, vertical=True)
    _note(ax, x + 0.95, 8.55, 'pool', size=6, ha='left', va='center')
    arrow(ax, x, y_out, x, 9.6, color=EEG)
    box(ax, x, 9.92, 1.2, 0.65, 'proj', fill=EEG_TINT, edge=EEG)
    _note(ax, x + 0.7, 9.92, '256 → 512 → 768', ha='left', va='center')
    arrow(ax, x, 10.25, x, 10.7, color=EEG)
    _vector(ax, x, 10.77, 1.35, cells=8, horizontal=True, dim='768', symbol='z')

    # The square at the top: EEG readings down the side, texts along the top.
    sq_x0, sq_y0, cell = 4.6, 11.05, 0.36
    side = len(SENTENCES) * cell
    _similarity_square(ax, sq_x0, sq_y0, cell, row_hue=EEG, col_hue=TEXT, tags='right')
    bus_x = sq_x0 - 0.1 - 0.24 - 0.42
    _lane(
        ax,
        [(x, 10.85), (x, 10.85 + 0.35), (bus_x, 10.85 + 0.35), (bus_x, sq_y0 + side - cell / 2)],
        color=EEG,
        headed=False,
    )
    for k in range(len(SENTENCES)):
        ry = sq_y0 + side - (k + 0.5) * cell
        arrow(ax, bus_x, ry, sq_x0 - 0.1 - 0.24, ry, color=EEG)
    _note(
        ax,
        sq_x0 + side / 2,
        sq_y0 + side + 0.34 + 0.55,
        r'$S_{ij} = z_i^{\top} t_j\,/\,\tau$',
        size=6.5,
        color=OBJECTIVE,
        ha='center',
        va='bottom',
    )

    # The frozen text tower on the right, its embeddings combing down into the column headers.
    tx = 7.9
    _cards(ax, tx, 4.9, 1.9, 0.8, '“He was elected …”')
    arrow(ax, tx, 5.33, tx, 5.8, color=TEXT)
    _trapezoid(ax, tx, 6.4, 1.5, 1.2, fill=TEXT_TINT, edge=FROZEN, dashed=True, label='text\nencoder', upward=True)
    snowflake(ax, tx + 0.72, 6.95, 0.15)
    arrow(ax, tx, 7.0, tx, 7.45, color=TEXT)
    _vector(ax, tx, 7.52, 1.35, cells=8, horizontal=True, fill=TEXT_TINT, edge=TEXT, dim='768', symbol='t')
    top_y = sq_y0 + side + 0.1 + 0.24 + 0.45
    _lane(ax, [(tx, 7.6), (tx, top_y), (sq_x0 + cell / 2, top_y)], color=TEXT, headed=False)
    for k in range(len(SENTENCES)):
        cx = sq_x0 + (k + 0.5) * cell
        arrow(ax, cx, top_y, cx, sq_y0 + side + 0.1 + 0.24, color=TEXT)

    # The positive definition and the symmetric loss, in the free corner under the text tower.
    _chip(ax, 5.2, 3.2, 'same sentence,\nany reader')
    _note(
        ax,
        5.2,
        2.3,
        r'$\mathcal{L} = \frac{1}{2}\,[\,\mathrm{InfoNCE}(S)$',
        size=6.5,
        color=OBJECTIVE,
        ha='left',
        va='center',
    )
    _note(ax, 5.2, 1.85, r'$\quad + \mathrm{InfoNCE}(S^{\top})\,]$', size=6.5, color=OBJECTIVE, ha='left', va='center')

    return fig


SCHEMATICS = {
    'encoder_pipeline': encoder_pipeline,
    'encoder_stack': encoder_stack,
    'transformer_block': transformer_block,
    'conformer_frontend': conformer_frontend,
    'encoder_pipeline_vertical': encoder_pipeline_vertical,
    'encoder_overview_minimal': encoder_overview_minimal,
    'conformer_frontend_compact': conformer_frontend_compact,
}
"""This family's data-free schematics, by name."""
