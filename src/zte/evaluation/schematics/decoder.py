"""The decoder schematics: the frozen-LM prefix path, its single-column form, and the control ladder."""

from typing import TYPE_CHECKING, Final

from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from zte.evaluation.schematics._style import (
    DOUBLE_COLUMN_IN,
    EEG,
    FILL,
    FROZEN,
    INK,
    INK_2,
    OBJECTIVE,
    RED,
    SINGLE_COLUMN_IN,
    TEXT,
    Axes,
    arrow,
    blank,
    box,
    bracket,
    dim_label,
    figure,
    snowflake,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# The pre-registered brain-independent controls, every one decoded through the identical path.
CONTROLS: Final[tuple[str, ...]] = (
    'null_prefix',
    'noise_prefix',
    'phase',
    'noise',
    'length_only',
    'shuffled_z',
    'mean_prefix',
    'mismatch',
)
"""The eight generation controls, in the order the verdict gate reports them."""

# LayerNorm(768) + 768x128 + eight FiLM (gamma, beta) rows + 128x896 + LayerNorm(896) + the 8x896 null prefix.
BRIDGE_PARAMS: Final[str] = '226,560'
"""Trainable parameters in the prefix bridge at 768 / 896 / 8 slots / bottleneck 128."""

LM_PARAMS: Final[str] = '494 M'
"""Parameters in the frozen Qwen2.5-0.5B, to the nearest million."""

LM_NAME: Final[str] = 'Qwen2.5-0.5B'
"""The frozen language model."""

LM_LAYERS: Final[int] = 24
"""Transformer layers in the frozen language model, drawn as a stack of rows."""

SLOTS: Final[int] = 8
"""Soft-prompt slots the bridge writes into the LM's own sequence."""

OUTPUT_WORDS: Final[tuple[str, ...]] = ('He', 'was', 'elected', '…')
"""The greedy readout, one token chip at a time; the same sentence the encoder schematics read."""

# The bridge's rows, bottom-up as they sit above the prefix bars; the empty entry is the row of eight FiLM squares.
_BRIDGE_ROWS: Final[tuple[str, ...]] = ('LN 896', '128 → 896', '', '768 → 128', 'LN 768')

EEG_TINT: Final[str] = '#e6f0fb'
"""Fill of a trainable EEG-side shape."""

TEXT_TINT: Final[str] = '#fdeee7'
"""Fill of a text-side shape."""

# Two steps darker than `FILL`, so the layer rows read as a stack inside the frozen box without a stroke each.
STACK_FILL: Final[str] = '#d3d2cc'
"""Fill of one frozen transformer layer in the stack."""


# ---- Glyphs ---- #


def _slot(ax: Axes, cx: float, y0: float, w: float, h: float, *, filled: bool) -> None:
    """One sequence position: solid blue when the bridge wrote it, hollow when the frozen model computed it."""
    ax.add_patch(
        Rectangle(
            (cx - w / 2, y0),
            w,
            h,
            facecolor=EEG if filled else 'white',
            edgecolor=EEG if filled else FROZEN,
            linewidth=0.6,
            zorder=3,
        )
    )


def _row(ax: Axes, cx: float, cy: float, w: float, h: float, label: str, *, size: float) -> None:
    """One thin bridge row with its dimension change written inside."""
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle='round,pad=0.0,rounding_size=0.05',
            facecolor='white',
            edgecolor=EEG,
            linewidth=0.6,
            zorder=3,
        )
    )
    ax.text(cx, cy, label, ha='center', va='center', fontsize=size, color=INK_2, zorder=4)


def _layer_stack(ax: Axes, x0: float, x1: float, y0: float, y1: float) -> None:
    """The frozen model's layers as a stack of fill-only rows, so depth is drawn rather than written."""
    pitch = (y1 - y0) / LM_LAYERS
    for k in range(LM_LAYERS):
        ax.add_patch(
            Rectangle((x0, y0 + k * pitch), x1 - x0, 0.6 * pitch, facecolor=STACK_FILL, edgecolor='none', zorder=2)
        )


def _tag(ax: Axes, cx: float, cy: float, w: float, h: float, text: str, *, size: float) -> None:
    """A red control tag: the name of one brain-independent input in monospace."""
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle='round,pad=0.0,rounding_size=0.07',
            facecolor='white',
            edgecolor=RED,
            linewidth=0.6,
        )
    )
    ax.text(cx, cy, text, ha='center', va='center', fontsize=size, color=RED, family='monospace')


def _chip(ax: Axes, cx: float, cy: float, w: float, h: float, text: str, *, size: float) -> None:
    """One decoded token on the text side."""
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle='round,pad=0.0,rounding_size=0.08',
            facecolor=TEXT_TINT,
            edgecolor=TEXT,
            linewidth=0.6,
        )
    )
    ax.text(cx, cy, text, ha='center', va='center', fontsize=size, color=INK)


def _card(
    ax: Axes,
    x0: float,
    x1: float,
    y0: float,
    *,
    bars_y: float,
    bar_h: float,
    pitch: float,
    bar_w: float,
    row_h: float,
    size: float,
    film_label: bool,
) -> tuple[float, float]:
    """The trainable card: five bridge rows over the eight prefix bars; returns the top row's y and the card top."""
    cx = (x0 + x1) / 2
    row_pitch = row_h + 0.09
    first_row = bars_y + bar_h + 0.2 + row_h / 2
    top_row = first_row + (len(_BRIDGE_ROWS) - 1) * row_pitch
    count_y = top_row + row_pitch
    title_y = count_y + 0.32
    y1 = title_y + 0.24
    box(ax, cx, (y0 + y1) / 2, x1 - x0, y1 - y0, fill=EEG_TINT, edge=EEG)

    centres = [cx + (j - (SLOTS - 1) / 2) * pitch for j in range(SLOTS)]
    for bx in centres:
        _slot(ax, bx, bars_y, bar_w, bar_h, filled=True)

    # The eight FiLM squares sit exactly over the eight bars: one code, eight learned views, eight positions.
    row_w = (SLOTS - 1) * pitch + bar_w + 0.1
    y = first_row
    for label in _BRIDGE_ROWS:
        if label:
            _row(ax, cx, y, row_w, row_h, label, size=size)
        else:
            for bx in centres:
                ax.add_patch(
                    Rectangle(
                        (bx - bar_w / 2, y - (row_h - 0.04) / 2),
                        bar_w,
                        row_h - 0.04,
                        facecolor=EEG,
                        edgecolor='none',
                        zorder=3,
                    )
                )
            if film_label:
                ax.text(x0 - 0.1, y, 'FiLM', ha='right', va='center', fontsize=size, color=INK_2)
        y += row_pitch
    ax.text(cx, count_y, BRIDGE_PARAMS, ha='center', va='center', fontsize=size + 0.5, color=INK_2)
    ax.text(cx, title_y, 'bridge', ha='center', va='center', fontsize=size + 1.5, color=INK)

    return top_row, y1


def _frozen_lm(
    ax: Axes,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    interior_x0: float,
    bars_y: float,
    bar_h: float,
    pitch: float,
    bar_w: float,
    stack_y: tuple[float, float],
    size: float,
) -> None:
    """The long grey frozen box: hollow slots along its input, the layer stack above, name and count on top."""
    box(ax, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, fill=FILL, edge=FROZEN, dashed=True)
    cx = interior_x0 + bar_w / 2
    while cx + bar_w / 2 <= x1 - 0.25:
        _slot(ax, cx, bars_y, bar_w, bar_h, filled=False)
        cx += pitch
    stack_x1 = x1 - 1.45
    _layer_stack(ax, interior_x0, stack_x1, *stack_y)
    bracket(ax, stack_x1 + 0.15, stack_y[0], stack_y[1], f'× {LM_LAYERS}', size=size)
    name_y = (stack_y[1] + y1) / 2
    ax.text(interior_x0, name_y, LM_NAME, ha='left', va='center', fontsize=size, color=INK_2)
    ax.text(stack_x1, name_y, LM_PARAMS, ha='right', va='center', fontsize=size, color=INK_2)
    snowflake(ax, x1 - 0.42, name_y, size=0.16)


def _readout(ax: Axes, lm_x1: float, out_y: float, loop_y: float, *, chip_w: float, chip_h: float, size: float) -> None:
    """Greedy decoding out of the LM's right end, one chip per token, each fed back into the sequence."""
    chip_cx = lm_x1 + 0.9 + chip_w / 2 + 0.1
    arrow(ax, lm_x1, out_y, chip_cx - chip_w / 2, out_y, color=TEXT)
    ax.text(
        (lm_x1 + chip_cx - chip_w / 2) / 2, out_y + 0.2, 'greedy', ha='center', va='center', fontsize=size, color=INK_2
    )
    y = out_y
    for word in OUTPUT_WORDS:
        _chip(ax, chip_cx, y, chip_w, chip_h, word, size=size + 1.0)
        y -= chip_h + 0.06
    # No teacher forcing: the token just read out is what enters the next position.
    ax.plot([chip_cx, chip_cx], [y + chip_h / 2 + 0.06, loop_y], color=TEXT, linewidth=0.7)
    arrow(ax, chip_cx, loop_y, lm_x1, loop_y, color=TEXT)


# ---- Builders ---- #


def decoder_bridge() -> Figure:
    """The prefix decoder: a frozen ZTE encoder feeds a small trainable bridge whose eight slots enter a frozen LM.

    Returns:
        Figure: A double-column figure whose trainable-to-frozen area is the argument, with the eight controls beneath.
    """
    fig, ax = figure(DOUBLE_COLUMN_IN, 1.7)
    blank(ax, (0, 20), (0.02, 4.6))
    lm_x0, lm_x1, lm_y0, lm_y1 = 5.6, 16.8, 0.4, 4.3
    bars_y, bar_h, pitch, bar_w = 0.8, 1.0, 0.32, 0.22
    _frozen_lm(
        ax,
        lm_x0,
        lm_x1,
        lm_y0,
        lm_y1,
        interior_x0=7.85,
        bars_y=bars_y,
        bar_h=bar_h,
        pitch=pitch,
        bar_w=bar_w,
        stack_y=(2.05, 3.6),
        size=7.0,
    )
    card_x0, card_x1 = 4.55, 7.45
    z_y, _ = _card(
        ax,
        card_x0,
        card_x1,
        0.6,
        bars_y=bars_y,
        bar_h=bar_h,
        pitch=pitch,
        bar_w=bar_w,
        row_h=0.26,
        size=5.5,
        film_label=True,
    )
    dim_label(ax, card_x1 - 0.24, 0.3, card_x0 + 0.2, 0.3, f'{SLOTS} × 896')

    # The encoder is frozen for the staged run drawn here; it hands over only z.
    enc_cx, enc_w, enc_h = 1.75, 2.8, 1.2
    box(ax, enc_cx, z_y, enc_w, enc_h, 'ZTE encoder', fill=FILL, edge=FROZEN, dashed=True)
    snowflake(ax, enc_cx + enc_w / 2 - 0.25, z_y + enc_h / 2 - 0.25, size=0.14)
    arrow(ax, enc_cx + enc_w / 2, z_y, card_x0, z_y, color=EEG)
    z_x = (enc_cx + enc_w / 2 + card_x0) / 2
    ax.text(z_x, z_y + 0.22, 'z', ha='center', va='center', fontsize=8, color=EEG, style='italic')
    ax.text(z_x, z_y - 0.22, '768', ha='center', va='center', fontsize=5.5, color=INK_2, family='monospace')

    # The controls replace z, or the prefix itself, and go through the identical path.
    tag_w, tag_h = 1.9, 0.34
    for k, name in enumerate(CONTROLS):
        _tag(ax, 1.25 + 2.0 * (k % 2), 2.38 - 0.42 * (k // 2), tag_w, tag_h, name, size=5.5)
    arrow(ax, z_x - 0.4, 2.58, z_x - 0.4, z_y - 0.06, color=RED)

    _readout(ax, lm_x1, z_y, bars_y + bar_h / 2, chip_w=2.1, chip_h=0.36, size=5.5)

    return fig


def decoder_bridge_compact() -> Figure:
    """The prefix decoder at single-column width: z arrives from above, the controls sit beneath the LM.

    Returns:
        Figure: A single-column figure with the same geometry and counts as `decoder_bridge`.
    """
    fig, ax = figure(SINGLE_COLUMN_IN, 1.9)
    blank(ax, (0, 10), (0.0, 5.4))
    lm_x0, lm_x1, lm_y0, lm_y1 = 1.2, 7.7, 1.1, 4.3
    bars_y, bar_h, pitch, bar_w = 1.5, 0.8, 0.22, 0.15
    _frozen_lm(
        ax,
        lm_x0,
        lm_x1,
        lm_y0,
        lm_y1,
        interior_x0=2.62,
        bars_y=bars_y,
        bar_h=bar_h,
        pitch=pitch,
        bar_w=bar_w,
        stack_y=(2.55, 3.5),
        size=6.5,
    )
    card_x0, card_x1 = 0.3, 2.3
    z_y, card_top = _card(
        ax,
        card_x0,
        card_x1,
        1.3,
        bars_y=bars_y,
        bar_h=bar_h,
        pitch=pitch,
        bar_w=bar_w,
        row_h=0.22,
        size=5.0,
        film_label=False,
    )
    card_cx = (card_x0 + card_x1) / 2
    dim_label(ax, card_x1 - 0.16, 0.98, card_x0 + 0.16, 0.98, f'{SLOTS} × 896', size=5.0)
    arrow(ax, card_cx, card_top + 0.55, card_cx, card_top, color=EEG)
    ax.text(card_cx + 0.15, card_top + 0.3, 'z', ha='left', va='center', fontsize=8, color=EEG, style='italic')
    ax.text(
        card_cx - 0.15, card_top + 0.3, '768', ha='right', va='center', fontsize=5.0, color=INK_2, family='monospace'
    )

    tag_w, tag_h = 1.72, 0.3
    for k, name in enumerate(CONTROLS):
        _tag(ax, 1.16 + 1.8 * (k % 4), 0.58 - 0.38 * (k // 4), tag_w, tag_h, name, size=5.0)

    _readout(ax, lm_x1, z_y, bars_y + bar_h / 2, chip_w=1.15, chip_h=0.3, size=5.0)

    return fig


def decoder_controls_ladder() -> Figure:
    """The verdict gate as a ladder: every control is a paired comparison whose interval must clear zero.

    Note:
        Every rung is drawn with the same schematic geometry on purpose. Nothing here is a measurement; the figure
        states the rule, and `zte-decode` reports the values.

    Returns:
        Figure: A single-column figure of the eight rungs, the zero line, and the two remaining clauses beneath.
    """
    fig, ax = figure(SINGLE_COLUMN_IN, 2.6)
    blank(ax, (0, 10), (0.0, 7.1))
    top, step = 6.3, 0.62
    bar_x0, eeg_len, ctrl_len, bar_h = 2.9, 2.1, 1.55, 0.2
    zero_x, dot_x, ci = 6.7, 7.9, 0.7
    ys = [top - step * k for k in range(len(CONTROLS))]
    for y, name in zip(ys, CONTROLS, strict=True):
        _tag(ax, 1.4, y, 2.2, 0.36, name, size=5.5)
        ax.add_patch(Rectangle((bar_x0, y + 0.02), eeg_len, bar_h, facecolor=EEG, edgecolor='none'))
        ax.add_patch(Rectangle((bar_x0, y - 0.02 - bar_h), ctrl_len, bar_h, facecolor=RED, edgecolor='none'))
        ax.plot([dot_x - ci, dot_x + ci], [y, y], color=INK_2, linewidth=0.7)
        for cap in (dot_x - ci, dot_x + ci):
            ax.plot([cap, cap], [y - 0.09, y + 0.09], color=INK_2, linewidth=0.7)
        ax.add_patch(Circle((dot_x, y), 0.09, facecolor=EEG, edgecolor='none', zorder=3))
    # Direct labels on the first rung stand in for a legend.
    ax.text(bar_x0 + eeg_len + 0.1, ys[0] + 0.12, 'EEG', ha='left', va='center', fontsize=6, color=EEG)
    ax.text(bar_x0 + ctrl_len + 0.1, ys[0] - 0.12, 'control', ha='left', va='center', fontsize=6, color=RED)

    # The delta axis: the rule is that every whisker lies wholly right of zero.
    ax.plot([zero_x, zero_x], [ys[-1] - 0.35, ys[0] + 0.35], color=INK_2, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.text(zero_x, ys[-1] - 0.5, '0', ha='center', va='center', fontsize=6, color=INK_2)
    ax.text(dot_x, ys[0] + 0.5, 'Δ = EEG − control', ha='center', va='center', fontsize=6.5, color=INK_2)
    bracket(ax, dot_x + ci + 0.35, ys[-1] - 0.15, ys[0] + 0.15, 'CI > 0\nall 8', size=6.0)

    # The remaining clauses of the verdict, ANDed with the ladder.
    ax.text(
        5.0,
        ys[-1] - 0.95,
        r'$\wedge\;\; p_{\mathrm{perm}} < 0.05 \qquad \wedge\;\; \mathrm{KL}(\mathrm{prefix}) \geq \mathrm{floor}$',
        ha='center',
        va='center',
        fontsize=7,
        color=INK_2,
    )
    arrow(ax, 5.0, ys[-1] - 1.2, 5.0, ys[-1] - 1.55, color=OBJECTIVE)
    ax.text(
        5.0,
        ys[-1] - 1.78,
        'generation_above_controls',
        ha='center',
        va='center',
        fontsize=6.5,
        color=OBJECTIVE,
        family='monospace',
    )

    return fig


SCHEMATICS = {
    'decoder_bridge': decoder_bridge,
    'decoder_bridge_compact': decoder_bridge_compact,
    'decoder_controls_ladder': decoder_controls_ladder,
}
"""This family's data-free schematics, by name."""
