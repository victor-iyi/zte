"""The objective and protocol schematics: two towers, the multi-positive square, the levels, the LOSO ring, the gallery."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from zte.evaluation.schematics._style import (
    DOUBLE_COLUMN_IN,
    EEG,
    FILL,
    FROZEN,
    INK,
    INK_2,
    MUTED,
    OBJECTIVE,
    RED,
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
    snowflake,
    traces,
)
from zte.lens.montage import azimuthal_xy

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# One data unit is a tenth of the single column: the scale the frozen helpers size their corners and glyphs for.
_UNIT_IN: Final[float] = SINGLE_COLUMN_IN / 10.0
"""Inches per data unit on every canvas in this module."""

_EEG_TINT: Final[str] = '#e6f0fb'
"""Fill of anything the EEG tower produces."""
_TEXT_TINT: Final[str] = '#fdeee7'
"""Fill of anything the text tower produces."""
# A pale objective aqua, so a positive cell reads as the loss's colour rather than as either tower's.
_POSITIVE_TINT: Final[str] = '#cfeee2'
"""Fill of a positive cell in the similarity square."""
_GRID: Final[str] = '#d5d4d0'
"""The fine grid between white cells."""

_READERS: Final[tuple[str, ...]] = ('ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH')
"""The twelve ZuCo readers, in protocol order."""
# A batch ordered so readings of one text sit together: the positive blocks then fall on the diagonal.
_BATCH: Final[tuple[tuple[str, str], ...]] = (
    ('s1', 'ZAB'),
    ('s1', 'ZDM'),
    ('s2', 'ZAB'),
    ('s2', 'ZJM'),
    ('s2', 'ZKB'),
    ('s3', 'ZDN'),
    ('s4', 'ZKH'),
    ('s4', 'ZPH'),
)
"""Eight readings as `(text id, reader)`, the rows and columns of the drawn square."""
_WORDS: Final[tuple[str, ...]] = ('The', 'river', 'froze', 'over', 'last', 'night')
"""The sentence the level panels read."""
_SKIPPED: Final[int] = 3
"""The word slot the eye skipped, which has no window."""
_SENTENCE_LINES: Final[tuple[str, ...]] = ('The river', 'froze over', 'last night')
"""The same sentence, wrapped for the text tower's input card."""
_TRUE_RANK: Final[int] = 21
"""Where the true text lands in the drawn gallery, one-based."""


# ---- Canvas and shared glyphs ---- #


def _canvas(width_in: float, y_lim: tuple[float, float]) -> tuple[Figure, Axes]:
    """A blank equal-aspect canvas `width_in` wide whose data units are `_UNIT_IN` inches."""
    fig, ax = figure(width_in, (y_lim[1] - y_lim[0]) * _UNIT_IN)
    blank(ax, (0.0, width_in / _UNIT_IN), y_lim)

    return fig, ax


def _tower(ax: Axes, x: float, y: float, w: float, h_in: float, lines: tuple[str, str], *, frozen: bool) -> None:
    """An encoder trapezoid narrowing toward its output, grey-dashed with a snowflake badge when frozen."""
    h_out = 0.71 * h_in
    ax.add_patch(
        Polygon(
            [[x, y - h_in / 2], [x, y + h_in / 2], [x + w, y + h_out / 2], [x + w, y - h_out / 2]],
            closed=True,
            facecolor=FILL if frozen else _EEG_TINT,
            edgecolor=FROZEN if frozen else EEG,
            linewidth=0.7,
            linestyle='--' if frozen else '-',
        )
    )
    ink = INK_2 if frozen else INK
    ax.text(x + w / 2, y + 0.15, lines[0], ha='center', va='center', fontsize=6.5, color=ink)
    ax.text(x + w / 2, y - 0.15, lines[1], ha='center', va='center', fontsize=6.5, color=ink)
    if frozen:
        snowflake(ax, x + w + 0.14, y + h_out / 2 + 0.14, size=0.13)


def _cards(ax: Axes, x: float, y: float, w: float, h: float) -> None:
    """A three-card stack, the batch idiom; the caller draws the sample on the front card."""
    plate(ax, x, y, w, h, copies=3, offset=0.07, fill='white', edge=INK_2)


def _vector(ax: Axes, x: float, y: float, side: float, label: str, *, text_side: bool) -> None:
    """An embedding as a small tinted square named by its symbol."""
    ax.add_patch(
        Rectangle(
            (x - side / 2, y - side / 2),
            side,
            side,
            facecolor=_TEXT_TINT if text_side else _EEG_TINT,
            edgecolor=TEXT if text_side else EEG,
            linewidth=0.7,
        )
    )
    ax.text(x, y, label, ha='center', va='center', fontsize=7, color=TEXT if text_side else EEG, style='italic')


def _bar(ax: Axes, x: float, y: float, w: float, h: float, *, frozen: bool) -> None:
    """A vector drawn as a vertical bar: blue when learned, grey and dashed when frozen."""
    ax.add_patch(
        Rectangle(
            (x - w / 2, y - h / 2),
            w,
            h,
            facecolor=FILL if frozen else _EEG_TINT,
            edgecolor=FROZEN if frozen else EEG,
            linewidth=0.6,
            linestyle=(0, (2.5, 1.5)) if frozen else '-',
        )
    )


def _tie(ax: Axes, x: float, y0: float, y1: float, *, scale: float = 6.0) -> None:
    """The contrastive pull between a vector and its target: a short two-headed arrow in the objective's colour."""
    ax.add_patch(
        FancyArrowPatch(
            (x, y0),
            (x, y1),
            arrowstyle='<|-|>',
            mutation_scale=scale,
            linewidth=0.7,
            color=OBJECTIVE,
            shrinkA=0,
            shrinkB=0,
        )
    )


def _hbrace(ax: Axes, x0: float, x1: float, y: float, text: str, *, size: float = 6.0) -> None:
    """A horizontal brace opening upward under a span, with `text` beneath it."""
    tick = 0.12
    ax.plot([x0, x0, x1, x1], [y + tick, y, y, y + tick], color=INK_2, linewidth=0.6)
    ax.text((x0 + x1) / 2, y - 0.12, text, ha='center', va='top', fontsize=size, color=INK)


def _montage_xy() -> np.ndarray:
    """The packaged electrode positions projected onto the unit disc."""
    xyz, _, _ = montage()
    xy = azimuthal_xy(xyz)

    return xy / float(np.abs(xy).max())


def _reader(ax: Axes, cx: float, cy: float, r: float, xy: np.ndarray, *, held_out: bool) -> None:
    """One reader: a head outline with the real 105 electrode positions dotted inside, red when held out."""
    head(ax, cx, cy, r, color=RED if held_out else INK_2)
    ax.scatter(
        cx + 0.74 * r * xy[:, 0], cy + 0.74 * r * xy[:, 1], s=2.2 * r, color=RED if held_out else EEG, linewidths=0
    )


def _frozen_box(ax: Axes, x: float, y: float, w: float, h: float, label: str) -> None:
    """A frozen module: grey fill, dashed grey stroke and a snowflake badge at its top-right corner."""
    box(ax, x, y, w, h, label, fill=FILL, edge=FROZEN, dashed=True, ink=INK_2)
    snowflake(ax, x + w / 2 + 0.1, y + h / 2 + 0.12, size=0.13)


# ---- The similarity square ---- #


def _square(ax: Axes, x0: float, y0: float, cell: float, texts: Sequence[str]) -> None:
    """White cells on a fine grid, every same-text cell tinted, framed in the objective's colour."""
    n = len(texts)
    side = n * cell
    ax.add_patch(Rectangle((x0, y0), side, side, facecolor='white', edgecolor='none', zorder=1))
    for i, row_text in enumerate(texts):
        for j, column_text in enumerate(texts):
            if row_text == column_text:
                ax.add_patch(
                    Rectangle(
                        (x0 + j * cell, y0 + side - (i + 1) * cell),
                        cell,
                        cell,
                        facecolor=_POSITIVE_TINT,
                        edgecolor='none',
                        zorder=1.5,
                    )
                )
    for k in range(1, n):
        ax.plot([x0, x0 + side], [y0 + k * cell] * 2, color=_GRID, linewidth=0.4, zorder=2)
        ax.plot([x0 + k * cell] * 2, [y0, y0 + side], color=_GRID, linewidth=0.4, zorder=2)
    ax.add_patch(Rectangle((x0, y0), side, side, facecolor='none', edgecolor=OBJECTIVE, linewidth=0.8, zorder=3))


def _row_headers(
    ax: Axes, x: float, y0: float, cell: float, batch: Sequence[tuple[str, str]], *, tag_gap: float, size: float
) -> None:
    """The reading strip left of the square: a tinted box per row with its text id, the reader tagged beside it."""
    w = 0.62 * cell
    side = len(batch) * cell
    for i, (text, reader) in enumerate(batch):
        cy = y0 + side - (i + 0.5) * cell
        ax.add_patch(Rectangle((x, cy - cell / 2), w, cell, facecolor=_EEG_TINT, edgecolor=EEG, linewidth=0.5))
        ax.text(x + w / 2, cy, text, ha='center', va='center', fontsize=size, color=EEG)
        ax.text(
            x + w + tag_gap / 2,
            cy,
            reader,
            ha='center',
            va='center',
            fontsize=size - 1.0,
            color=INK_2,
            family='monospace',
        )


def _column_headers(ax: Axes, x0: float, y: float, cell: float, texts: Sequence[str], *, size: float) -> None:
    """The text strip above the square: a tinted box per column with its text id."""
    h = 0.62 * cell
    for j, text in enumerate(texts):
        ax.add_patch(Rectangle((x0 + j * cell, y), cell, h, facecolor=_TEXT_TINT, edgecolor=TEXT, linewidth=0.5))
        ax.text(x0 + (j + 0.5) * cell, y + h / 2, text, ha='center', va='center', fontsize=size, color=TEXT)


@dataclass(slots=True, frozen=True, kw_only=True)
class _TwoTowerLayout:
    """Where the two-tower figure's pieces sit, in canvas units; everything else is derived from the square."""

    width_in: float
    y_lim: tuple[float, float]
    batch: tuple[tuple[str, str], ...]
    cell: float
    square_y0: float
    card_x: float
    card_w: float
    card_h: float
    tower_x: float
    tower_w: float
    tower_h: float
    z_x: float
    z_side: float
    eeg_bus_x: float
    tag_gap: float
    annotate: bool


def _two_tower(layout: _TwoTowerLayout) -> Figure:
    """Draws the CLIP-style two-tower figure at the given layout."""
    fig, ax = _canvas(layout.width_in, layout.y_lim)
    batch, cell = layout.batch, layout.cell
    texts = [text for text, _ in batch]
    side = len(batch) * cell
    header = 0.62 * cell
    header_x = layout.eeg_bus_x + 0.5
    sq_x0 = header_x + header + layout.tag_gap
    sq_y0 = layout.square_y0
    sq_x1, sq_y1 = sq_x0 + side, sq_y0 + side
    header_y = sq_y1 + 0.15
    bus_y = header_y + header + 0.45
    eeg_y = sq_y0 + side / 2
    label_size = 6.0 if layout.annotate else 5.5

    # Inputs: a batch of word-window rasters and a batch of sentences, each a card stack.
    _cards(ax, layout.card_x, eeg_y, layout.card_w, layout.card_h)
    traces(ax, layout.card_x, eeg_y, 0.85 * layout.card_w, 0.7 * layout.card_h, n=6)
    ax.text(
        layout.card_x,
        eeg_y - layout.card_h / 2 - 0.1,
        'L × 105 × 350',
        ha='center',
        va='top',
        fontsize=5.5,
        color=INK_2,
        family='monospace',
    )
    _cards(ax, layout.card_x, bus_y, layout.card_w, layout.card_h)
    for k, line in enumerate(_SENTENCE_LINES):
        ax.text(
            layout.card_x,
            bus_y + (1 - k) * 0.24,
            line,
            ha='center',
            va='center',
            fontsize=5,
            color=INK,
            family='monospace',
        )

    # Towers: the same trapezoid twice, so fill and glyph are the only difference between trainable and frozen.
    card_right = layout.card_x + layout.card_w / 2
    tower_right = layout.tower_x + layout.tower_w
    z_left, z_right = layout.z_x - layout.z_side / 2, layout.z_x + layout.z_side / 2
    for y, colour, lines, frozen in ((eeg_y, EEG, ('EEG', 'encoder'), False), (bus_y, TEXT, ('text', 'encoder'), True)):
        arrow(ax, card_right, y, layout.tower_x, y, color=colour)
        _tower(ax, layout.tower_x, y, layout.tower_w, layout.tower_h, lines, frozen=frozen)
        arrow(ax, tower_right, y, z_left, y, color=colour)
    _vector(ax, layout.z_x, eeg_y, layout.z_side, 'z', text_side=False)
    _vector(ax, layout.z_x, bus_y, layout.z_side, 't', text_side=True)
    if layout.annotate:
        for y in (eeg_y, bus_y):
            ax.text(layout.z_x, y - layout.z_side / 2 - 0.08, '768', ha='center', va='top', fontsize=5.5, color=INK_2)

    # Bus and comb: one line along each strip, a short arrow into every header cell.
    row_centres = [sq_y1 - (i + 0.5) * cell for i in range(len(batch))]
    column_centres = [sq_x0 + (j + 0.5) * cell for j in range(len(batch))]
    ax.plot([z_right, layout.eeg_bus_x], [eeg_y, eeg_y], color=EEG, linewidth=0.7)
    ax.plot([layout.eeg_bus_x] * 2, [row_centres[0], row_centres[-1]], color=EEG, linewidth=0.7)
    for cy in row_centres:
        arrow(ax, layout.eeg_bus_x, cy, header_x, cy, color=EEG)
    ax.plot([z_right, column_centres[-1]], [bus_y, bus_y], color=TEXT, linewidth=0.7)
    for cx in column_centres:
        arrow(ax, cx, bus_y, cx, header_y + header, color=TEXT)

    # The square is the objective: nothing is written in it, the tinted blocks are the multi-positive mask.
    _row_headers(ax, header_x, sq_y0, cell, batch, tag_gap=layout.tag_gap, size=label_size)
    _column_headers(ax, sq_x0, header_y, cell, texts, size=label_size)
    _square(ax, sq_x0, sq_y0, cell, texts)
    formula = r'$S = z\,t^{\top} / \tau$'
    if not layout.annotate:
        ax.text((sq_x0 + sq_x1) / 2, sq_y0 - 0.25, formula, ha='center', va='top', fontsize=8, color=OBJECTIVE)

        return fig

    ax.text(sq_x1 + 0.3, header_y + header / 2, formula, ha='left', va='center', fontsize=8, color=OBJECTIVE)
    # Both softmax directions, read off the square's own axes.
    bracket(ax, sq_x1 + 0.2, sq_y0, sq_y1, 'EEG → text', side=1, size=6)
    _hbrace(ax, sq_x0, sq_x1, sq_y0 - 0.2, 'text → EEG')
    # One tinted chip and one white chip say what a block is; nothing else needs a legend.
    chip_x = sq_x1 + 2.0
    for cy, fill, caption in (
        (eeg_y + 0.5, _POSITIVE_TINT, 'same text · any reader'),
        (eeg_y - 0.5, 'white', 'other text'),
    ):
        ax.add_patch(Rectangle((chip_x, cy - cell / 2), cell, cell, facecolor=fill, edgecolor=_GRID, linewidth=0.5))
        ax.text(chip_x + cell + 0.18, cy, caption, ha='left', va='center', fontsize=6, color=INK_2)

    return fig


_DOUBLE_TOWERS: Final[_TwoTowerLayout] = _TwoTowerLayout(
    width_in=DOUBLE_COLUMN_IN,
    y_lim=(0.35, 9.0),
    batch=_BATCH,
    cell=0.68,
    square_y0=1.3,
    card_x=1.5,
    card_w=1.75,
    card_h=1.05,
    tower_x=3.2,
    tower_w=1.25,
    tower_h=1.7,
    z_x=5.95,
    z_side=0.5,
    eeg_bus_x=6.95,
    tag_gap=0.6,
    annotate=True,
)
"""The double-column two-tower figure."""
_SINGLE_TOWERS: Final[_TwoTowerLayout] = _TwoTowerLayout(
    width_in=SINGLE_COLUMN_IN,
    y_lim=(0.2, 6.55),
    batch=_BATCH[:6],
    cell=0.62,
    square_y0=0.95,
    card_x=0.85,
    card_w=1.4,
    card_h=0.9,
    tower_x=1.9,
    tower_w=1.2,
    tower_h=1.45,
    z_x=3.7,
    z_side=0.45,
    eeg_bus_x=4.45,
    tag_gap=0.55,
    annotate=False,
)
"""The single-column two-tower figure."""


def two_tower() -> Figure:
    """The contrastive square: the EEG tower learns to arrive where the frozen text tower already is.

    Returns:
        Figure: A double-column figure.
    """
    return _two_tower(_DOUBLE_TOWERS)


def two_tower_compact() -> Figure:
    """The two-tower figure at single-column width: six readings, no legend, the formula under the square.

    Returns:
        Figure: A single-column figure.
    """
    return _two_tower(_SINGLE_TOWERS)


def alignment_square() -> Figure:
    """The multi-positive mask alone: two readers of one text are positives for each other, everything else is not.

    Returns:
        Figure: A single-column figure.
    """
    fig, ax = _canvas(SINGLE_COLUMN_IN, (0.05, 10.1))
    cell = 1.0
    texts = [text for text, _ in _BATCH]
    header_x, tag_gap = 0.5, 0.65
    sq_x0, sq_y0 = header_x + 0.62 * cell + tag_gap, 0.95
    side = len(_BATCH) * cell
    header_y = sq_y0 + side + 0.15
    _row_headers(ax, header_x, sq_y0, cell, _BATCH, tag_gap=tag_gap, size=6.5)
    _column_headers(ax, sq_x0, header_y, cell, texts, size=6.5)
    _square(ax, sq_x0, sq_y0, cell, texts)
    ax.text(0.22, sq_y0 + side / 2, 'readings', ha='center', va='center', fontsize=6.5, color=EEG, rotation=90)
    ax.text(
        sq_x0 + side / 2, header_y + 0.62 * cell + 0.12, 'texts', ha='center', va='bottom', fontsize=6.5, color=TEXT
    )
    ax.text(
        sq_x0 + side / 2,
        sq_y0 - 0.25,
        r'$P_{ij} = \mathbf{1}[\,\mathrm{text}_i = \mathrm{text}_j\,]$',
        ha='center',
        va='top',
        fontsize=8,
        color=OBJECTIVE,
    )

    return fig


def alignment_square_pooled_vs_strict() -> Figure:
    """The pooled gallery ranks every reader's 8,400 readings; the strict one only the held-out reader's 700.

    Returns:
        Figure: A single-column figure.
    """
    fig, ax = _canvas(SINGLE_COLUMN_IN, (0.15, 8.3))
    block, x0, top = 0.6, 2.6, 7.5
    for i, reader in enumerate(_READERS):
        y1 = top - i * block
        y0 = y1 - block
        held_out = i == 0
        colour = RED if held_out else EEG
        ax.add_patch(Rectangle((x0, y0), block, block, facecolor='white', edgecolor=colour, linewidth=0.7))
        # Every reader read every text, so each reader's block carries the identity diagonal of positives.
        ax.plot([x0, x0 + block], [y1, y0], color=OBJECTIVE, linewidth=0.9)
        ax.text(
            x0 - 0.15, y0 + block / 2, reader, ha='right', va='center', fontsize=5.5, color=colour, family='monospace'
        )
    bottom = top - len(_READERS) * block
    bracket(ax, x0 + block + 0.15, bottom, top, '8,400 queries', side=1, size=6)
    ax.text(x0 + block / 2, top + 0.4, 'pooled', ha='center', va='center', fontsize=7.5, color=INK)
    ax.text(x0 + block / 2, bottom - 0.15, '700 texts', ha='center', va='top', fontsize=5.5, color=INK_2)

    # The strict gallery keeps the held-out block alone: the only rows a stranger's score can come from.
    sx0 = 6.6
    arrow(ax, x0 + block + 0.45, top - block / 2, sx0 - 0.1, top - block / 2, color=RED)
    ax.add_patch(Rectangle((sx0, top - block), block, block, facecolor='white', edgecolor=RED, linewidth=0.7))
    ax.plot([sx0, sx0 + block], [top, top - block], color=OBJECTIVE, linewidth=0.9)
    bracket(ax, sx0 + block + 0.15, top - block, top, '700 queries', side=1, size=6)
    ax.text(sx0 + block / 2, top + 0.4, 'held out', ha='center', va='center', fontsize=7.5, color=INK)
    ax.text(sx0 + block / 2, top - block - 0.15, '700 texts', ha='center', va='top', fontsize=5.5, color=INK_2)

    return fig


# ---- The three alignment levels ---- #


def _level_panel(ax: Axes, x0: float, level: int) -> None:
    """One level panel at `x0`: word cards, the learned vectors, the ties, and the frozen targets beneath."""
    centres = [x0 + 0.75 + i for i in range(len(_WORDS))]
    present = [i for i in range(len(_WORDS)) if i != _SKIPPED]
    card_y, card_h, card_w = 4.25, 0.6, 0.82
    top_y, target_y, bar_h = 2.55, 1.2, 0.7
    ax.text(x0 + 3.25, 5.3, ('sentence', 'word', 'token')[level], ha='center', va='center', fontsize=7.5, color=INK)

    # The word row is identical in every panel; a skipped word has no window, so it is hatched and feeds nothing.
    for i, (cx, word) in enumerate(zip(centres, _WORDS, strict=True)):
        skipped = i == _SKIPPED
        ax.text(cx, 4.85, word, ha='center', va='center', fontsize=5.5, color=MUTED if skipped else INK_2)
        card = FancyBboxPatch(
            (cx - card_w / 2, card_y - card_h / 2),
            card_w,
            card_h,
            boxstyle='round,pad=0.0,rounding_size=0.06',
            facecolor='white' if skipped else _EEG_TINT,
            edgecolor=MUTED if skipped else EEG,
            linewidth=0.6,
            linestyle='--' if skipped else '-',
            hatch='////' if skipped else None,
        )
        card.set_hatch_linewidth(0.5)
        ax.add_patch(card)
        if skipped:
            continue

        traces(ax, cx, card_y, 0.7, 0.44, n=3, seed=i)
        if level == 2:
            for k in range(1, 4):
                sx = cx - card_w / 2 + k * card_w / 4
                ax.plot([sx, sx], [card_y - card_h / 2, card_y + card_h / 2], color=EEG, linewidth=0.4)

    card_bottom = card_y - card_h / 2
    bar_top = top_y + bar_h / 2
    # The frozen row's snowflake sits just left of its first bar, whichever level that bar belongs to.
    first_bar = {0: x0 + 3.25 - 0.15, 1: centres[present[0]] - 0.15, 2: centres[present[0]] - card_w / 2}[level]
    snowflake(ax, first_bar - 0.22, target_y, size=0.12)
    if level == 0:
        # Pool: every present window drops onto one collector, and one arrow leaves it.
        collector_y = card_bottom - 0.45
        centre = x0 + 3.25
        for i in present:
            ax.plot([centres[i]] * 2, [card_bottom, collector_y], color=EEG, linewidth=0.7)
        ax.plot([centres[present[0]], centres[present[-1]]], [collector_y] * 2, color=EEG, linewidth=0.7)
        arrow(ax, centre, collector_y, centre, bar_top, color=EEG)
        _bar(ax, centre, top_y, 0.3, bar_h, frozen=False)
        _tie(ax, centre, top_y - bar_h / 2, target_y + bar_h / 2)
        _bar(ax, centre, target_y, 0.3, bar_h, frozen=True)
    elif level == 1:
        for i in present:
            arrow(ax, centres[i], card_bottom, centres[i], bar_top, color=EEG)
            _bar(ax, centres[i], top_y, 0.3, bar_h, frozen=False)
            _tie(ax, centres[i], top_y - bar_h / 2, target_y + bar_h / 2)
            _bar(ax, centres[i], target_y, 0.3, bar_h, frozen=True)
    else:
        # Four fixed intra-word slices, each tied to one sub-word piece.
        for i in present:
            for k in range(4):
                sx = centres[i] - card_w / 2 + (k + 0.5) * card_w / 4
                ax.plot([sx, sx], [card_bottom, bar_top], color=EEG, linewidth=0.5)
                _bar(ax, sx, top_y, 0.15, bar_h, frozen=False)
                _tie(ax, sx, top_y - bar_h / 2, target_y + bar_h / 2, scale=4.0)
                _bar(ax, sx, target_y, 0.15, bar_h, frozen=True)


def three_levels() -> Figure:
    """Sentence, word and sub-word alignment: three identical panels that differ only in what is tied to what.

    Returns:
        Figure: A double-column figure.
    """
    fig, ax = _canvas(DOUBLE_COLUMN_IN, (0.6, 5.6))
    for level in range(3):
        _level_panel(ax, 0.2 + 6.8 * level, level)

    return fig


# ---- Leave-one-subject-out ---- #


def _ranked_strip(ax: Axes, x: float, y_top: float, w: float, h: float, *, true_index: int) -> float:
    """A column of gallery texts ranked top to bottom, the tail elided, the true text in red; returns its bottom."""
    shown = (5, 3)
    y = y_top
    rank = 0
    for count in shown:
        for _ in range(count):
            true = rank == true_index
            ax.add_patch(
                Rectangle(
                    (x - w / 2, y - h),
                    w,
                    h,
                    facecolor=RED if true else SEQUENTIAL_CMAP(0.55 - 0.06 * rank),
                    edgecolor='white',
                    linewidth=0.5,
                )
            )
            if true:
                ax.text(x + w / 2 + 0.12, y - h / 2, 'true text', ha='left', va='center', fontsize=6, color=RED)
            y -= h
            rank += 1
        if count == shown[0]:
            ax.text(x, y - 0.17, '⋮', ha='center', va='center', fontsize=7, color=INK_2)
            y -= 0.34
    ax.text(x - w / 2 - 0.1, y_top - h / 2, '1', ha='right', va='center', fontsize=5.5, color=INK_2)
    ax.text(x, y - 0.1, '700 texts', ha='center', va='top', fontsize=5.5, color=INK_2)

    return y


def loso_ring() -> Figure:
    """Twelve readers in a ring, one held out with no arrow in; the trained encoder, frozen, then ranks its texts.

    Returns:
        Figure: A single-column figure.
    """
    fig, ax = _canvas(SINGLE_COLUMN_IN, (2.85, 10.05))
    xy = _montage_xy()
    cx, cy, ring, r = 3.45, 6.3, 2.45, 0.3
    box_w, box_h = 1.5, 0.75
    held_centre = (cx, cy + ring + 0.65)
    for i, code in enumerate(_READERS):
        angle = math.pi / 2 - 2 * math.pi * i / len(_READERS)
        ux, uy = math.cos(angle), math.sin(angle)
        held_out = i == 0
        if held_out:
            _reader(ax, *held_centre, r, xy, held_out=True)
            ax.text(
                held_centre[0] - r - 0.12,
                held_centre[1],
                code,
                ha='right',
                va='center',
                fontsize=5,
                color=RED,
                family='monospace',
            )
            continue

        hx, hy = cx + ring * ux, cy + ring * uy
        _reader(ax, hx, hy, r, xy, held_out=False)
        ax.text(
            cx + (ring + 0.62) * ux,
            cy + (ring + 0.62) * uy,
            code,
            ha='center',
            va='center',
            fontsize=5,
            color=INK_2,
            family='monospace',
        )
        # Each arrow stops on the box's own edge along its ray, so all eleven land cleanly.
        reach = min(box_w / 2 / max(abs(ux), 1e-9), box_h / 2 / max(abs(uy), 1e-9)) + 0.05
        arrow(ax, cx + (ring - r - 0.06) * ux, cy + (ring - r - 0.06) * uy, cx + reach * ux, cy + reach * uy, color=EEG)
    box(ax, cx, cy, box_w, box_h, 'encoder', sub='× 12', fill=_EEG_TINT, edge=EEG)
    # Where the twelfth arrow would be, there is only the question.
    ax.text(cx, cy + box_h / 2 + 1.35, '?', ha='center', va='center', fontsize=10, color=RED, weight='bold')

    # The stranger's readings go through the trained encoder, now frozen, and its 700 texts are ranked.
    fx, fy = 8.5, 7.3
    ax.plot([held_centre[0] + r + 0.05, fx], [held_centre[1]] * 2, color=RED, linewidth=0.7)
    arrow(ax, fx, held_centre[1], fx, fy + box_h / 2, color=RED)
    _frozen_box(ax, fx, fy, box_w, box_h, 'encoder')
    strip_top = fy - box_h / 2 - 0.45
    arrow(ax, fx, fy - box_h / 2, fx, strip_top, color=RED)
    _ranked_strip(ax, fx, strip_top, 0.44, 0.34, true_index=2)

    return fig


def loso_fold_strip() -> Figure:
    """The twelve folds as a grid: every reader trains eleven times and is the held-out query set exactly once.

    Returns:
        Figure: A single-column figure.
    """
    fig, ax = _canvas(SINGLE_COLUMN_IN, (0.35, 9.0))
    xy = _montage_xy()
    cell, x0, y1 = 0.5, 2.2, 7.4
    n = len(_READERS)
    for j, code in enumerate(_READERS):
        col_x = x0 + (j + 0.5) * cell
        _reader(ax, col_x, y1 + 0.42, 0.2, xy, held_out=False)
        ax.text(
            col_x, y1 + 0.75, code, ha='center', va='bottom', fontsize=5, color=INK_2, family='monospace', rotation=90
        )
    for i in range(n):
        row_y = y1 - (i + 1) * cell
        ax.text(x0 - 0.15, row_y + cell / 2, str(i + 1), ha='right', va='center', fontsize=5.5, color=INK_2)
        for j in range(n):
            held_out = i == j
            ax.add_patch(
                Rectangle(
                    (x0 + j * cell, row_y),
                    cell,
                    cell,
                    facecolor='white' if held_out else _EEG_TINT,
                    edgecolor=RED if held_out else 'white',
                    linewidth=0.8 if held_out else 0.5,
                    zorder=3 if held_out else 2,
                )
            )
    ax.text(x0 - 0.75, y1 - n * cell / 2, 'fold', ha='center', va='center', fontsize=6.5, color=INK, rotation=90)
    ax.text(x0 + n * cell / 2, y1 + 1.45, 'reader', ha='center', va='bottom', fontsize=6.5, color=INK)
    bracket(ax, x0 + n * cell + 0.15, y1 - n * cell, y1, '700 queries\nper fold', side=1, size=6)

    # Two chips say what a cell is.
    chip_y = y1 - n * cell - 0.7
    for x, fill, edge, caption in ((x0, _EEG_TINT, 'white', 'train'), (x0 + 2.2, 'white', RED, 'held out')):
        ax.add_patch(Rectangle((x, chip_y - cell / 2), cell, cell, facecolor=fill, edgecolor=edge, linewidth=0.8))
        ax.text(x + cell + 0.15, chip_y, caption, ha='left', va='center', fontsize=6, color=INK_2)

    return fig


def retrieval_gallery() -> Figure:
    """Closed-set identification: the true text's rank among 700 is the number, not a generated string.

    Returns:
        Figure: A single-column figure.
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 1.75))
    rng = np.random.default_rng(7)
    scores = np.clip(np.sort(rng.normal(0.22, 0.07, 700))[::-1], 0.02, None)
    ranks = np.arange(1, 701)
    ax.bar(ranks, scores, width=1.0, color=MUTED, linewidth=0)
    # One bar in 700 is a hairline at column width, so the true text is drawn wide enough to be seen.
    true_score = float(scores[_TRUE_RANK - 1])
    ax.bar(_TRUE_RANK, true_score, width=4.0, color=RED, linewidth=0)
    ax.plot(_TRUE_RANK, true_score, 'o', color=RED, markersize=3, markeredgewidth=0)
    ax.text(_TRUE_RANK + 12, true_score + 0.03, 'true text', ha='left', va='bottom', fontsize=6.5, color=RED)
    # A random guess lands, on average, at the gallery's midpoint: rank percentile one half.
    ax.axvline(350.5, color=INK_2, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.text(350.5, float(scores[0]) + 0.02, 'chance', ha='center', va='bottom', fontsize=6, color=INK_2)
    ax.set_xlim(-2, 702)
    ax.set_ylim(0, float(scores[0]) + 0.09)
    ax.set_yticks([])
    ax.set_xticks([_TRUE_RANK, 350, 700])
    ax.get_xticklabels()[0].set_color(RED)
    ax.set_xlabel('rank among 700 texts', fontsize=6.5, color=INK_2, labelpad=2)
    ax.set_ylabel('S', fontsize=7, color=INK_2, style='italic', labelpad=2)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3)

    return fig


SCHEMATICS = {
    'two_tower': two_tower,
    'two_tower_compact': two_tower_compact,
    'alignment_square': alignment_square,
    'alignment_square_pooled_vs_strict': alignment_square_pooled_vs_strict,
    'three_levels': three_levels,
    'loso_ring': loso_ring,
    'loso_fold_strip': loso_fold_strip,
    'retrieval_gallery': retrieval_gallery,
}
"""This family's data-free schematics, by name."""
