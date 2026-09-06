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
    MIN_FONT_PT,
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

# Enough hollow positions to read as a sequence the prefix leads; the ellipsis says the model's own tokens follow.
LM_SLOTS_DRAWN: Final[int] = 12
"""Hollow sequence positions drawn inside the frozen model before the ellipsis."""

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


def _lead(fig: Figure, ax: Axes, span: float) -> float:
    """The smallest written line height in data units once `span` units fill the axes width: the layout's module."""
    return MIN_FONT_PT * span / (ax.get_position().width * fig.get_figwidth() * 72.0)


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


def _span(ax: Axes, x0: float, x1: float, y: float, text: str, *, lead: float, size: float = 5.5) -> None:
    """A horizontal dimension line with its label set clear beneath it, for a size along a bottom edge."""
    dim_label(ax, x0, y, x1, y, '')
    ax.text((x0 + x1) / 2, y - 0.8 * lead, text, ha='center', va='center', fontsize=size, color=INK_2)


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
    lead: float,
    size: float,
    film_label: bool,
) -> tuple[float, float, float]:
    """The trainable card: five bridge rows over the eight prefix bars; returns the top row's y, the FiLM row's y
    and the card top."""
    cx = (x0 + x1) / 2
    row_h = 1.05 * lead
    row_pitch = row_h + 0.22 * lead
    first_row = bars_y + bar_h + 0.35 * lead + row_h / 2
    top_row = first_row + (len(_BRIDGE_ROWS) - 1) * row_pitch
    film_row = first_row + _BRIDGE_ROWS.index('') * row_pitch
    count_y = top_row + row_pitch + 0.05 * lead
    title_y = count_y + 1.4 * lead
    y1 = title_y + 0.85 * lead
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
            square = row_h - 0.2 * lead
            for bx in centres:
                ax.add_patch(
                    Rectangle(
                        (bx - bar_w / 2, y - square / 2), bar_w, square, facecolor=EEG, edgecolor='none', zorder=3
                    )
                )
            if film_label:
                ax.text(x0 - 0.08, y, 'FiLM', ha='right', va='center', fontsize=size, color=INK_2)
        y += row_pitch
    ax.text(cx, count_y, BRIDGE_PARAMS, ha='center', va='center', fontsize=size + 0.5, color=INK_2)
    ax.text(cx, title_y, 'bridge', ha='center', va='center', fontsize=size + 1.5, color=INK)

    return top_row, film_row, y1


def _frozen_lm(
    ax: Axes,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    interior_x0: float,
    stack_x1: float,
    bars_y: float,
    bar_h: float,
    pitch: float,
    bar_w: float,
    stack_y: tuple[float, float],
    size: float,
) -> None:
    """The grey frozen box: hollow slots then an ellipsis along its input, the layer stack above, name and count on
    top."""
    box(ax, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, fill=FILL, edge=FROZEN, dashed=True)
    cx = interior_x0 + bar_w / 2
    for _ in range(LM_SLOTS_DRAWN):
        _slot(ax, cx, bars_y, bar_w, bar_h, filled=False)
        cx += pitch
    ax.text(cx + 0.1, bars_y + bar_h / 2, '⋯', ha='left', va='center', fontsize=size, color=FROZEN)
    _layer_stack(ax, interior_x0, stack_x1, *stack_y)
    bracket(ax, stack_x1 + 0.12, stack_y[0], stack_y[1], f'× {LM_LAYERS}', size=size)
    name_y = (stack_y[1] + y1) / 2
    ax.text(interior_x0, name_y, LM_NAME, ha='left', va='center', fontsize=size, color=INK_2)
    ax.text(stack_x1, name_y, LM_PARAMS, ha='right', va='center', fontsize=size, color=INK_2)
    snowflake(ax, x1 - 0.38, name_y, size=0.15)


def _readout(
    ax: Axes,
    lm_x1: float,
    out_y: float,
    loop_y: float,
    *,
    reach: float,
    lead: float,
    size: float,
    across: bool,
) -> float:
    """Greedy decoding out of the LM's right end, one chip per token in a row or a stack, each fed back into the
    sequence; returns the readout's right edge."""
    arrow(ax, lm_x1, out_y, lm_x1 + reach, out_y, color=TEXT)
    ax.text(lm_x1 + reach / 2, out_y + 1.2 * lead, 'greedy', ha='center', va='center', fontsize=size, color=INK_2)
    chip_h = 1.25 * lead
    natural = [0.56 * lead * len(word) + 1.1 * lead for word in OUTPUT_WORDS]
    widths = natural if across else [max(natural)] * len(natural)
    x, y = lm_x1 + reach + 0.05, out_y
    right = x
    for word, w in zip(OUTPUT_WORDS, widths, strict=True):
        _chip(ax, x + w / 2, y, w, chip_h, word, size=size + 1.0)
        right = max(right, x + w)
        if across:
            x += w + 0.1
        else:
            y -= chip_h + 0.3 * lead
    # No teacher forcing: the token just read out is what enters the next position.
    drop_x = right - widths[-1] / 2
    drop_y = out_y - chip_h / 2 if across else y + 0.3 * lead + chip_h / 2
    ax.plot([drop_x, drop_x], [drop_y, loop_y], color=TEXT, linewidth=0.7)
    arrow(ax, drop_x, loop_y, lm_x1, loop_y, color=TEXT)

    return right


# ---- Builders ---- #


def decoder_bridge() -> Figure:
    """The prefix decoder: a frozen ZTE encoder feeds a small trainable bridge whose eight slots enter a frozen LM.

    Returns:
        Figure: A double-column figure whose trainable-to-frozen area is the argument, with the eight controls beneath.
    """
    fig, ax = figure(DOUBLE_COLUMN_IN, 1.75)
    width = 18.7
    lead = _lead(fig, ax, width)
    bars_y, bar_h, pitch, bar_w = 0.65, 0.45, 0.26, 0.18
    card_x0, card_x1 = 5.4, 7.7
    lm_x0, lm_x1, lm_y0 = 5.95, 13.05, 0.45
    z_y, film_y, card_top = _card(
        ax,
        card_x0,
        card_x1,
        0.3,
        bars_y=bars_y,
        bar_h=bar_h,
        pitch=pitch,
        bar_w=bar_w,
        lead=lead,
        size=5.5,
        film_label=True,
    )
    _frozen_lm(
        ax,
        lm_x0,
        lm_x1,
        lm_y0,
        card_top - 0.6 * lead,
        interior_x0=card_x1 + 0.35,
        stack_x1=lm_x1 - 1.4,
        bars_y=bars_y,
        bar_h=bar_h,
        pitch=pitch,
        bar_w=bar_w,
        stack_y=(bars_y + bar_h + 0.9 * lead, z_y - 0.55 * lead),
        size=6.0,
    )
    _span(ax, card_x1 - 0.2, card_x0 + 0.2, 0.17, f'{SLOTS} × 896', lead=lead)

    # The encoder is frozen for the staged run drawn here; it hands over only z. Its label sits left of centre so
    # the snowflake in the corner has the room it needs.
    enc_cx, enc_w, enc_h = 2.71, 2.95, 0.95
    enc_x1 = enc_cx + enc_w / 2
    box(ax, enc_cx, z_y, enc_w, enc_h, fill=FILL, edge=FROZEN, dashed=True)
    ax.text(enc_cx - 0.14, z_y, 'ZTE encoder', ha='center', va='center', fontsize=6.5, color=INK)
    snowflake(ax, enc_x1 - 0.24, z_y + enc_h / 2 - 0.22, size=0.13)
    arrow(ax, enc_x1, z_y, card_x0, z_y, color=EEG)
    z_x = (enc_x1 + card_x0) / 2
    ax.text(z_x, z_y + 0.8 * lead, 'z', ha='center', va='center', fontsize=8, color=EEG, style='italic')
    ax.text(z_x + 0.1, z_y - 0.8 * lead, '768', ha='center', va='center', fontsize=5.5, color=INK_2, family='monospace')

    # The controls replace z, or the prefix itself, and go through the identical path; the grid stops short of the
    # FiLM label so the two never meet.
    tag_w, tag_h, tag_pitch = 2.5, 1.2 * lead, 1.5 * lead
    tag_top = film_y - 1.2 * lead
    for k, name in enumerate(CONTROLS):
        _tag(ax, 1.4 + 2.62 * (k % 2), tag_top - tag_h / 2 - tag_pitch * (k // 2), tag_w, tag_h, name, size=5.5)
    arrow(ax, enc_x1 + 0.16, tag_top, enc_x1 + 0.16, z_y - 0.06, color=RED)

    _readout(ax, lm_x1, z_y, bars_y + bar_h / 2, reach=1.3, lead=lead, size=5.5, across=True)
    blank(ax, (0, width), (0.17 - 1.4 * lead, card_top + 0.1))

    return fig


def decoder_bridge_compact() -> Figure:
    """The prefix decoder at single-column width: z arrives from above, the controls sit beneath the LM.

    Returns:
        Figure: A single-column figure with the same geometry and counts as `decoder_bridge`.
    """
    fig, ax = figure(SINGLE_COLUMN_IN, 2.35)
    width = 14.4
    lead = _lead(fig, ax, width)
    # Three columns: a single column cannot hold four monospace tags abreast at a legible size.
    tag_w, tag_h, tag_pitch = 3.6, 1.2 * lead, 1.5 * lead
    for k, name in enumerate(CONTROLS):
        _tag(ax, 1.95 + 3.7 * (k % 3), tag_h / 2 + tag_pitch * (2 - k // 3), tag_w, tag_h, name, size=5.5)
    tags_top = tag_h + 2 * tag_pitch

    dims_y = tags_top + 1.7 * lead
    card_y0 = dims_y + 0.12
    bars_y, bar_h, pitch, bar_w = card_y0 + 0.35, 0.45, 0.33, 0.2
    card_x0, card_x1 = 0.3, 3.15
    lm_x0, lm_x1 = 1.0, 10.3
    z_y, _, card_top = _card(
        ax,
        card_x0,
        card_x1,
        card_y0,
        bars_y=bars_y,
        bar_h=bar_h,
        pitch=pitch,
        bar_w=bar_w,
        lead=lead,
        size=5.5,
        film_label=False,
    )
    _frozen_lm(
        ax,
        lm_x0,
        lm_x1,
        card_y0 + 0.15,
        card_top - 0.6 * lead,
        interior_x0=card_x1 + 0.3,
        stack_x1=lm_x1 - 1.7,
        bars_y=bars_y,
        bar_h=bar_h,
        pitch=pitch,
        bar_w=bar_w,
        stack_y=(bars_y + bar_h + 0.9 * lead, z_y - 0.55 * lead),
        size=6.0,
    )
    card_cx = (card_x0 + card_x1) / 2
    _span(ax, card_x1 - 0.2, card_x0 + 0.2, dims_y, f'{SLOTS} × 896', lead=lead)
    arrow(ax, card_cx, card_top + 1.8 * lead, card_cx, card_top, color=EEG)
    z_label_y = card_top + 1.0 * lead
    ax.text(card_cx + 0.15, z_label_y, 'z', ha='left', va='center', fontsize=8, color=EEG, style='italic')
    ax.text(card_cx - 0.15, z_label_y, '768', ha='right', va='center', fontsize=5.5, color=INK_2, family='monospace')

    _readout(ax, lm_x1, z_y, bars_y + bar_h / 2, reach=1.7, lead=lead, size=5.5, across=False)
    blank(ax, (0, width), (0.0, card_top + 1.9 * lead))

    return fig


def decoder_controls_ladder() -> Figure:
    """The verdict gate as a ladder: every control is a paired comparison whose interval must clear zero.

    Note:
        Every rung is drawn with the same schematic geometry on purpose. Nothing here is a measurement; the figure
        states the rule, and `zte-decode` reports the values.

    Returns:
        Figure: A single-column figure of the eight rungs, the zero line, and the two remaining clauses beneath.
    """
    fig, ax = figure(SINGLE_COLUMN_IN, 2.7)
    blank(ax, (0, 10.15), (-0.5, 7.15))
    top, step = 6.3, 0.66
    bar_x0, eeg_len, ctrl_len, bar_h = 3.1, 2.0, 1.5, 0.22
    zero_x, dot_x, ci = 6.6, 7.8, 0.6
    ys = [top - step * k for k in range(len(CONTROLS))]
    for y, name in zip(ys, CONTROLS, strict=True):
        _tag(ax, 1.5, y, 2.7, 0.4, name, size=5.5)
        ax.add_patch(Rectangle((bar_x0, y + 0.03), eeg_len, bar_h, facecolor=EEG, edgecolor='none'))
        ax.add_patch(Rectangle((bar_x0, y - 0.03 - bar_h), ctrl_len, bar_h, facecolor=RED, edgecolor='none'))
        ax.plot([dot_x - ci, dot_x + ci], [y, y], color=INK_2, linewidth=0.7)
        for cap in (dot_x - ci, dot_x + ci):
            ax.plot([cap, cap], [y - 0.09, y + 0.09], color=INK_2, linewidth=0.7)
        ax.add_patch(Circle((dot_x, y), 0.09, facecolor=EEG, edgecolor='none', zorder=3))
    # Direct labels on the first rung stand in for a legend.
    ax.text(bar_x0 + eeg_len + 0.1, ys[0] + 0.19, 'EEG', ha='left', va='center', fontsize=6, color=EEG)
    ax.text(bar_x0 + ctrl_len + 0.1, ys[0] - 0.19, 'control', ha='left', va='center', fontsize=6, color=RED)

    # The delta axis: the rule is that every whisker lies wholly right of zero.
    ax.plot([zero_x, zero_x], [ys[-1] - 0.3, ys[0] + 0.35], color=INK_2, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.text(zero_x, ys[-1] - 0.5, '0', ha='center', va='center', fontsize=6, color=INK_2)
    ax.text(dot_x, ys[0] + 0.6, 'Δ = EEG − control', ha='center', va='center', fontsize=6.5, color=INK_2)
    bracket(ax, dot_x + ci + 0.3, ys[-1] - 0.15, ys[0] + 0.15, 'CI > 0\nall 8', size=6.0)

    # The remaining clauses of the verdict, ANDed with the ladder.
    ax.text(
        5.0,
        ys[-1] - 1.02,
        r'$\wedge\;\; p_{\mathrm{perm}} < 0.05 \qquad \wedge\;\; \mathrm{KL}(\mathrm{prefix}) \geq \mathrm{floor}$',
        ha='center',
        va='center',
        fontsize=7,
        color=INK_2,
    )
    arrow(ax, 5.0, ys[-1] - 1.3, 5.0, ys[-1] - 1.6, color=OBJECTIVE)
    ax.text(
        5.0,
        ys[-1] - 1.84,
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
