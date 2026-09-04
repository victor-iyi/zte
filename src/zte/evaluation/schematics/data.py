"""The data schematics: the presence mask, one fixated word window, and the eye-tracking segmentation path."""

import math
from typing import TYPE_CHECKING, Final

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from zte.evaluation.schematics._style import (
    DOUBLE_COLUMN_IN,
    EEG,
    FILL,
    FROZEN,
    INK,
    INK_2,
    MUTED,
    RED,
    SINGLE_COLUMN_IN,
    YELLOW,
    Axes,
    arrow,
    blank,
    box,
    dim_label,
    figure,
    head,
    montage,
    tensor_slab,
    traces,
)
from zte.lens.montage import azimuthal_xy

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SENTENCE: Final[tuple[str, ...]] = (
    'He',
    'was',
    'elected',
    'to',
    'the',
    'Senate',
    'in',
    '1990',
    'and',
    'served',
    'two',
    'terms',
)
"""The sample sentence every data schematic reads, the one the encoder schematics already use."""

# Readers skip short function words; four of twelve is ZuCo's overall omission rate of about 0.3.
SKIPPED: Final[frozenset[int]] = frozenset({3, 4, 6, 8})
"""Word slots the eye never fixated, so they have no window."""

# Dwell per fixated word in ms, longer on the content words as reading-time models predict; schematic values.
DWELL_MS: Final[dict[int, int]] = {0: 180, 1: 210, 2: 290, 5: 310, 7: 260, 9: 270, 10: 200, 11: 250}
"""Fixation duration of each fixated word slot."""

SACCADE_MS: Final[int] = 30
"""The gap between one fixation's offset and the next one's onset."""

SAMPLE_HZ: Final[int] = 500
"""ZuCo's EEG sampling rate."""

WINDOW: Final[int] = 350
"""Samples per word window, 700 ms at 500 Hz."""

N400_MS: Final[tuple[int, int]] = (300, 500)
"""The band a semantic-integration peak would fall in, relative to word onset."""

DISCARDED_CHANNELS: Final[int] = 23
"""Electrodes of the 128-channel cap that ZuCo drops, leaving 105."""

EEG_TINT: Final[str] = '#e6f0fb'
"""Fill of a trainable EEG-side shape."""

# The dashed-and-hatched idiom for a slot that holds nothing: the same hairline as every other stroke here.
_HATCH_LW: Final[float] = 0.4


# ---- Glyphs ---- #


def _eeg_like(rng: np.random.Generator, n: int) -> np.ndarray:
    """A plausible single-channel trace at unit scale: a slow wave, an alpha burst and leaky-integrated noise."""
    t = np.arange(n) / SAMPLE_HZ
    slow = np.sin(2 * math.pi * rng.uniform(2.0, 4.0) * t + rng.uniform(0, 2 * math.pi))
    burst = np.exp(-((t - rng.uniform(0.05, 0.25)) ** 2) / 0.008)
    alpha = 0.6 * burst * np.sin(2 * math.pi * 10.0 * t + rng.uniform(0, 2 * math.pi))
    kernel = 0.85 ** np.arange(60)
    noise = np.convolve(rng.standard_normal(n + 60), kernel, mode='full')[60 : 60 + n]
    noise /= max(float(noise.std()), 1e-9)

    return 0.55 * slow + alpha + 0.3 * noise


def _empty_slot(ax: Axes, x0: float, y0: float, w: float, h: float) -> None:
    """A hatched outline where a skipped word's window would be: no card, no data."""
    patch = Rectangle((x0, y0), w, h, facecolor='none', edgecolor=MUTED, linewidth=0.6, hatch='////')
    patch.set_hatch_linewidth(_HATCH_LW)
    ax.add_patch(patch)


def _card(ax: Axes, cx: float, cy: float, w: float, h: float, seed: int) -> None:
    """One fixated word's window as a small trace card."""
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor='white', edgecolor=EEG, linewidth=0.6))
    traces(ax, cx, cy, 0.88 * w, 0.8 * h, n=4, seed=seed)


def _scanpath(
    ax: Axes,
    discs: list[tuple[float, float, int]],
    *,
    scale: float,
    path: list[tuple[float, float]] | None = None,
) -> None:
    """The gaze as a polyline through the fixations, a dwell disc at each whose radius grows with duration."""
    route = path if path is not None else [(x, y) for x, y, _ in discs]
    ax.plot([p[0] for p in route], [p[1] for p in route], color=INK_2, linewidth=0.6, zorder=1)
    for x, y, dwell in discs:
        ax.add_patch(Circle((x, y), 0.05 + scale * dwell, facecolor=EEG_TINT, edgecolor=EEG, linewidth=0.6, zorder=2))


def _cap(ax: Axes, cx: float, cy: float, r: float) -> None:
    """The head from above with the real 105 retained electrodes filled and the discarded outer ring hollow."""
    head(ax, cx, cy, r)
    xyz, _, _ = montage()
    xy = azimuthal_xy(xyz)
    xy = xy / max(float(np.abs(xy).max()), 1e-9) * 0.92 * r
    ax.scatter(cx + xy[:, 0], cy + xy[:, 1], s=3.5, color=EEG, linewidths=0, zorder=3)
    # The dropped electrodes are the cap's outer ring, below the head circle in this projection; their exact
    # positions are not carried by the packaged montage, so the ring is evenly spaced: a schematic, not a measurement.
    angles = np.linspace(0, 2 * math.pi, DISCARDED_CHANNELS, endpoint=False) + math.pi / 2
    ax.scatter(
        cx + 1.12 * r * np.cos(angles),
        cy + 1.12 * r * np.sin(angles),
        s=7,
        facecolors='none',
        edgecolors=MUTED,
        linewidths=0.5,
        zorder=3,
    )


def _word_positions(x_left: float, indices: range) -> list[tuple[int, float]]:
    """Left-to-right centres for one line of words, spaced by an estimate of their set width."""
    out: list[tuple[int, float]] = []
    x = x_left
    for i in indices:
        w = 0.135 * len(SENTENCE[i]) + 0.05
        out.append((i, x + w / 2))
        x += w + 0.22

    return out


# ---- Builders ---- #


def presence_mask() -> Figure:
    """A sentence in gaze order: fixated words are trace cards, skipped words are hatched, only cards reach the pool.

    Returns:
        Figure: A double-column figure with the presence row beneath the words and the masked pool taking the ones.
    """
    fig, ax = figure(DOUBLE_COLUMN_IN, 1.55)
    blank(ax, (0, 20), (0.0, 4.25))
    pitch, card_w, card_h = 1.55, 1.3, 0.8
    centres = [2.05 + pitch * i for i in range(len(SENTENCE))]
    mid = (centres[0] + centres[-1]) / 2
    y_gaze, y_word, y_card, y_cell, cell_h = 3.9, 3.5, 2.65, 1.6, 0.36

    fixated = [(centres[i], y_gaze, DWELL_MS[i]) for i in range(len(SENTENCE)) if i not in SKIPPED]
    _scanpath(ax, fixated, scale=0.0003)
    pool_cx, pool_cy, pool_w, pool_h = mid, 0.32, 2.4, 0.56
    for i, (word, cx) in enumerate(zip(SENTENCE, centres, strict=True)):
        present = i not in SKIPPED
        ax.text(cx, y_word, word, ha='center', va='center', fontsize=7, color=INK if present else MUTED, zorder=3)
        if present:
            _card(ax, cx, y_card, card_w, card_h, seed=i)
        else:
            _empty_slot(ax, cx - card_w / 2, y_card - card_h / 2, card_w, card_h)
        ax.add_patch(
            Rectangle(
                (cx - 0.25, y_cell),
                0.5,
                cell_h,
                facecolor=EEG_TINT if present else 'white',
                edgecolor=EEG if present else MUTED,
                linewidth=0.6,
            )
        )
        ax.text(
            cx,
            y_cell + cell_h / 2,
            '1' if present else '0',
            ha='center',
            va='center',
            fontsize=6.5,
            color=EEG if present else MUTED,
            family='monospace',
        )
        # Only a present slot has a line into the pool; a zero contributes nothing and is never imputed.
        if present:
            ax.plot([cx, pool_cx + 0.13 * (cx - mid)], [y_cell, pool_cy + pool_h / 2], color=EEG, linewidth=0.55)
    box(ax, pool_cx, pool_cy, pool_w, pool_h, 'pool', fill='#dbe9fa', edge=EEG)
    arrow(ax, pool_cx + pool_w / 2, pool_cy, pool_cx + pool_w / 2 + 1.1, pool_cy, color=EEG)
    ax.text(
        pool_cx + pool_w / 2 + 1.25, pool_cy, '256', ha='left', va='center', fontsize=6, color=INK_2, family='monospace'
    )

    label_x = centres[0] - card_w / 2 - 0.2
    ax.text(label_x, y_cell + cell_h / 2, 'presence', ha='right', va='center', fontsize=6.5, color=INK_2)
    ax.text(label_x, y_card, '105 × 350', ha='right', va='center', fontsize=5.5, color=INK_2, family='monospace')

    return fig


def word_window() -> Figure:
    """One fixated word's 105 x 350 window: signal to the fixation offset, zeros after, the neighbour's window behind.

    Returns:
        Figure: A single-column figure with the offset in red, the padded tail shaded, the 300-500 ms band tinted,
            the axis in ms, and the next word's window as an offset outline that begins inside this one.
    """
    fig, ax = figure(SINGLE_COLUMN_IN, 1.55)
    blank(ax, (0, 9.35), (0.45, 4.4))
    x0, per_ms = 0.55, 0.0085
    window_ms = 1000 * WINDOW // SAMPLE_HZ
    fixation_ms = DWELL_MS[2]

    def tx(ms: float) -> float:
        return x0 + per_ms * ms

    y0, h = 1.05, 2.8
    win_w = per_ms * window_ms
    tensor_slab(ax, x0, y0, win_w, h, 0.0, fill='white', edge=INK_2, dims=('', '105', ''))
    ax.add_patch(
        Rectangle((tx(fixation_ms), y0), tx(window_ms) - tx(fixation_ms), h, facecolor=FILL, edgecolor='none', zorder=2)
    )
    ax.add_patch(
        Rectangle(
            (tx(N400_MS[0]), y0),
            tx(N400_MS[1]) - tx(N400_MS[0]),
            h,
            facecolor=YELLOW,
            alpha=0.15,
            edgecolor='none',
            zorder=2,
        )
    )
    rng = np.random.default_rng(6)
    n_signal = fixation_ms * SAMPLE_HZ // 1000
    t = x0 + per_ms * np.arange(WINDOW) * 1000 / SAMPLE_HZ
    for k in range(6):
        base = y0 + 0.35 + 0.42 * k
        signal = 0.16 * _eeg_like(rng, WINDOW)
        signal[n_signal:] = 0.0
        ax.plot(t, base + signal, color=EEG, linewidth=0.55, zorder=3)
    ax.plot([tx(fixation_ms)] * 2, [y0, y0 + h], color=RED, linewidth=0.8, zorder=4)

    ax.text(x0, y0 + h + 0.08, SENTENCE[2], ha='left', va='bottom', fontsize=7, color=INK)
    ax.text(tx(fixation_ms) - 0.06, y0 + h + 0.08, 'offset', ha='right', va='bottom', fontsize=6, color=RED)
    ax.text(tx(sum(N400_MS) / 2), y0 + h - 0.12, 'N400', ha='center', va='top', fontsize=5.5, color=INK_2, zorder=5)
    ax.text(tx(600), y0 + 0.12, 'zero-padded', ha='center', va='center', fontsize=5.5, color=INK_2, zorder=5)

    # The next word's window opens one saccade after this fixation ends, so the two overlap for most of their span
    # and this word's 300-500 ms lies inside the neighbour's fixation: drawn over everything, so the overlap is seen.
    next_x0, shift = tx(fixation_ms + SACCADE_MS), 0.35
    ax.add_patch(
        Rectangle(
            (next_x0, y0 + shift),
            win_w,
            h,
            facecolor='none',
            edgecolor=FROZEN,
            linewidth=0.7,
            linestyle=(0, (3, 2)),
            zorder=6,
        )
    )
    ax.text(
        next_x0 + win_w - 0.1,
        y0 + shift + h - 0.1,
        SENTENCE[5],
        ha='right',
        va='top',
        fontsize=6.5,
        color=INK_2,
        zorder=6,
    )

    axis_y = y0 - 0.22
    ax.plot([x0, tx(window_ms)], [axis_y, axis_y], color=INK_2, linewidth=0.6)
    for ms, label in ((0, '0'), (N400_MS[0], '300'), (N400_MS[1], '500'), (window_ms, f'{window_ms} ms')):
        ax.plot([tx(ms)] * 2, [axis_y, axis_y - 0.08], color=INK_2, linewidth=0.6)
        ax.text(tx(ms), axis_y - 0.16, label, ha='center', va='top', fontsize=6, color=INK_2)

    return fig


def eye_segmentation() -> Figure:
    """From the screen to one window per fixated word: the gaze cuts the continuous EEG of the 105-channel cap.

    Returns:
        Figure: A double-column figure: the sentence with its scanpath, the cap, the cut EEG strip, the windows.
    """
    fig, ax = figure(DOUBLE_COLUMN_IN, 1.8)
    blank(ax, (0, 20), (0.15, 5.0))

    # The screen: the sentence on two lines, a dwell disc over each fixated word, the scanpath through the discs.
    sx0, sy0, sw, sh = 0.3, 2.65, 5.3, 2.3
    ax.add_patch(
        FancyBboxPatch(
            (sx0, sy0),
            sw,
            sh,
            boxstyle='round,pad=0.0,rounding_size=0.12',
            facecolor='white',
            edgecolor=INK_2,
            linewidth=0.7,
        )
    )
    lines = (_word_positions(sx0 + 0.42, range(0, 6)), _word_positions(sx0 + 0.42, range(6, 12)))
    line_ys = (sy0 + sh - 0.62, sy0 + 0.55)
    disc_lift, sweep_y = 0.34, (line_ys[0] + line_ys[1]) / 2 + 0.19
    fixations: list[tuple[float, float, int]] = []
    route: list[tuple[float, float]] = []
    for line, (positions, y) in enumerate(zip(lines, line_ys, strict=True)):
        for i, cx in positions:
            ax.text(cx, y, SENTENCE[i], ha='center', va='center', fontsize=7, color=INK, zorder=3)
            if i in SKIPPED:
                continue

            # The return sweep to the second line is routed through the gap between the lines, not across the words.
            if line and route and route[-1][1] == line_ys[0] + disc_lift:
                route.extend([(route[-1][0], sweep_y), (cx, sweep_y)])
            fixations.append((cx, y + disc_lift, DWELL_MS[i]))
            route.append((cx, y + disc_lift))
    _scanpath(ax, fixations, scale=0.00045, path=route)

    # The cap: real coordinates for the 105 retained electrodes, a hollow ring for the 23 discarded ones.
    hx, hy, hr = 2.95, 1.45, 0.85
    _cap(ax, hx, hy, hr)
    ax.text(hx, hy - 1.42 * hr, '128 → 105', ha='center', va='center', fontsize=6, color=INK_2)

    # The continuous EEG, cut at the fixation onsets the eye-tracker supplies; the tinted spans are what is kept.
    tx0, ty0, tw, th = 7.0, 1.3, 6.2, 2.6
    ax.add_patch(Rectangle((tx0, ty0), tw, th, facecolor='white', edgecolor=INK_2, linewidth=0.6))
    ordered = [i for i in range(len(SENTENCE)) if i not in SKIPPED]
    total_ms = sum(DWELL_MS[i] for i in ordered) + SACCADE_MS * (len(ordered) - 1) + 120
    scale = tw / total_ms
    t_ms = 60
    marks: list[tuple[float, float, int]] = []
    for i in ordered:
        dwell = DWELL_MS[i]
        ax.add_patch(
            Rectangle((tx0 + scale * t_ms, ty0), scale * dwell, th, facecolor=EEG_TINT, edgecolor='none', zorder=1)
        )
        ax.plot([tx0 + scale * t_ms] * 2, [ty0, ty0 + th + 0.18], color=INK_2, linewidth=0.5, linestyle=(0, (2, 2)))
        marks.append((tx0 + scale * (t_ms + dwell / 2), ty0 + th + 0.4, dwell))
        t_ms += dwell + SACCADE_MS
    _scanpath(ax, marks, scale=0.0003)
    rng = np.random.default_rng(11)
    n = total_ms * SAMPLE_HZ // 1000
    tt = tx0 + scale * np.arange(n) * 1000 / SAMPLE_HZ
    for k in range(6):
        ax.plot(tt, ty0 + 0.3 + 0.4 * k + 0.11 * _eeg_like(rng, n), color=EEG, linewidth=0.45, zorder=2)

    # One window per fixated word: the first three, an ellipsis, the last.
    card_w, card_h, card_y = 1.15, 0.85, 2.62
    shown = (ordered[0], ordered[1], ordered[2], None, ordered[-1])
    cx = 14.55
    last_x1 = cx
    for slot in shown:
        if slot is None:
            ax.text(cx - 0.4, card_y, '⋯', ha='center', va='center', fontsize=9, color=INK_2)
            cx += 0.5
            continue
        _card(ax, cx, card_y, card_w, card_h, seed=slot)
        ax.text(cx, card_y + card_h / 2 + 0.2, SENTENCE[slot], ha='center', va='center', fontsize=6, color=INK)
        last_x1 = cx + card_w / 2
        cx += card_w + 0.15
    first_x0 = 14.55 - card_w / 2
    ax.text(
        first_x0,
        card_y - card_h / 2 - 0.2,
        '105 × 350',
        ha='left',
        va='center',
        fontsize=5.5,
        color=INK_2,
        family='monospace',
    )
    dim_label(ax, last_x1, card_y - card_h / 2 - 0.5, first_x0, card_y - card_h / 2 - 0.5, f'× {len(ordered)}')

    arrow(ax, sx0 + sw, ty0 + th + 0.4, tx0 - 0.08, ty0 + th + 0.4, color=INK_2)
    arrow(ax, hx + 1.15 * hr, hy, tx0, hy, color=EEG)
    arrow(ax, tx0 + tw, card_y, first_x0 - 0.1, card_y, color=EEG)

    return fig


SCHEMATICS = {
    'presence_mask': presence_mask,
    'word_window': word_window,
    'eye_segmentation': eye_segmentation,
}
"""This family's data-free schematics, by name."""
