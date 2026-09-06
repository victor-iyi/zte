"""Figures drawn from artifacts: the attention scalp map and curve, and the cross-task transfer heatmap."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from zte.evaluation.schematics._style import (
    DIVERGING_CMAP,
    DOUBLE_COLUMN_IN,
    EEG,
    INK,
    INK_2,
    ORANGE,
    RC,
    SEQUENTIAL_CMAP,
    SINGLE_COLUMN_IN,
    YELLOW,
    topomap,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def attention_topomap_figure(
    attention_json: str | Path, groups: Sequence[str] = ('all', 'correct', 'incorrect')
) -> Figure:
    """Attention received per electrode on the head the checkpoint was trained on, one panel per group.

    Refuses an artifact whose montage was not verified against the checkpoint basis: a map on unverified
    coordinates is the artifact this figure exists to replace.

    Args:
        attention_json (str | Path): An `attention.json` written by `zte-lens attention`.
        groups (Sequence[str], optional): Reading groups to draw, in order. Defaults to all, correct, incorrect.

    Returns:
        Figure: One scalp map per available group, plus the correct-minus-incorrect difference when both exist.

    Raises:
        ValueError: If the artifact has no scalp block, or its coordinates were not verified.
    """
    report = json.loads(Path(attention_json).read_text(encoding='utf-8'))
    spatial = report.get('spatial') or {}
    if not spatial or spatial.get('xyz') is None:
        raise ValueError(f'{attention_json} carries no scalp block with coordinates.')
    if spatial.get('approximate_geometry') or not spatial.get('montage_verified'):
        raise ValueError(f'{attention_json}: montage not verified ({spatial.get("montage_reason")}); refusing to draw.')

    xyz = np.asarray(spatial['xyz'], dtype=np.float64)
    labels = list(spatial['labels'])
    uniform = 1.0 / int(spatial['n_channels'])
    blocks = spatial['groups']
    panels: list[tuple[str, np.ndarray, bool]] = [
        (name, np.asarray(blocks[name]['mean']) / uniform - 1.0, False) for name in groups if name in blocks
    ]
    if 'correct' in blocks and 'incorrect' in blocks:
        diff = (np.asarray(blocks['correct']['mean']) - np.asarray(blocks['incorrect']['mean'])) / uniform
        panels.append(('correct − incorrect', diff, True))

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE_COLUMN_IN, 2.1))
        fig.subplots_adjust(left=0.01, right=0.93, top=0.9, bottom=0.02, wspace=0.05)
        limit = max(float(np.abs(v).max()) for _, v, _ in panels) or 1.0
        image = None
        for ax, (title, values, diverging) in zip(np.atleast_1d(axes), panels, strict=True):
            image = topomap(ax, values, xyz, cmap=DIVERGING_CMAP, vlim=(-limit, limit), labels=labels, contours=4)
            ax.set_title(title, pad=2)
            ax.axis('off')
        if image is not None:
            bar = fig.colorbar(image, ax=list(np.atleast_1d(axes)), fraction=0.02, pad=0.01)
            bar.set_ticks([-limit, 0.0, limit])
            bar.set_ticklabels([f'{-limit:+.2f}', '0', f'{limit:+.2f}'])
            bar.set_label('deviation from uniform', fontsize=6.5)
            bar.ax.tick_params(labelsize=6)

    return fig


def attention_temporal_figure(attention_json: str | Path, group: str = 'all') -> Figure:
    """Attention received per sample over the word window, the uniform line, and the a-priori N400 band.

    Args:
        attention_json (str | Path): An `attention.json` written by `zte-lens attention`.
        group (str, optional): The reading group to draw. Defaults to `'all'`.

    Returns:
        Figure: The last-layer curve with its bootstrap band.

    Raises:
        ValueError: If the artifact carries no temporal block for `group`.
    """
    report = json.loads(Path(attention_json).read_text(encoding='utf-8'))
    temporal = report.get('temporal') or {}
    block = (temporal.get('groups') or {}).get(group)
    if not block:
        raise ValueError(f'{attention_json} carries no temporal block for group {group!r}.')

    times = np.asarray(temporal['times_ms'])
    layer = block['layers'][int(temporal.get('headline_layer', temporal['n_layers'] - 1))]
    lo, hi = temporal['n400_window_ms']
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 1.9))
        ax.axvspan(lo, hi, color=YELLOW, alpha=0.12, linewidth=0)
        ax.fill_between(times, layer['ci_low'], layer['ci_high'], color=EEG, alpha=0.18, linewidth=0)
        ax.plot(times, layer['mean'], color=EEG, linewidth=1.0)
        ax.axhline(temporal['uniform'], color=INK_2, linestyle=':', linewidth=0.7)
        ax.set_xlim(float(times[0]), float(times[-1]))
        ax.set_xlabel('ms after fixation onset')
        ax.set_ylabel('attention received')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        fig.tight_layout(pad=0.3)

    return fig


def transfer_heatmap_figure(parallax_json: str | Path) -> Figure:
    """The cross-task transfer matrix as a heatmap: rank percentile per cell, novel-stimulus cells outlined.

    Args:
        parallax_json (str | Path): A `PARALLAX.json` written by `zte-parallax report`.

    Returns:
        Figure: A task-by-task heatmap, trained-on down the side and evaluated-on along the top.

    Raises:
        ValueError: If the artifact holds no transfer cells.
    """
    report = json.loads(Path(parallax_json).read_text(encoding='utf-8'))
    cells = report.get('cells') or {}
    tasks = [t for t in report.get('tasks') or [] if t in cells] or sorted(cells)
    if not tasks:
        raise ValueError(f'{parallax_json} holds no transfer cells.')

    matrix = np.full((len(tasks), len(tasks)), np.nan)
    novel = np.zeros((len(tasks), len(tasks)), dtype=bool)
    for i, train in enumerate(tasks):
        for j, eval_task in enumerate(tasks):
            summaries = (cells.get(train) or {}).get(eval_task) or []
            values = [s['rank_percentile'] for s in summaries if s.get('rank_percentile') is not None]
            if values:
                matrix[i, j] = float(np.mean(values))
                novel[i, j] = all(bool(s.get('novel_stimuli')) for s in summaries)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.9))
        finite = matrix[np.isfinite(matrix)]
        low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.5, 1.0)
        pad = 0.25 * max(high - low, 1e-3)
        # Every cell carries its value in black type, so the ramp stops short of the steps black cannot be read on.
        light = LinearSegmentedColormap.from_list(
            'zte_sequential_light', [SEQUENTIAL_CMAP(x) for x in np.linspace(0.0, 0.55, 8)]
        )
        image = ax.imshow(matrix, cmap=light, vmin=low - pad, vmax=high + pad)
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i, j]:.4f}', ha='center', va='center', fontsize=7, color=INK)
                if novel[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='none', edgecolor=ORANGE, linewidth=1.4))
                if i == j:
                    ax.add_patch(
                        Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, facecolor='none', edgecolor=INK_2, linewidth=0.5, linestyle=':'
                        )
                    )
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(tasks)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks)
        ax.xaxis.tick_top()
        ax.set_xlabel('evaluated on')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('trained on')
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label('rank percentile', fontsize=6.5)
        bar.ax.tick_params(labelsize=6)
        fig.tight_layout(pad=0.3)

    return fig
