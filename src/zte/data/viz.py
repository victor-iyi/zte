# pylint: disable=wrong-import-position
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use('Agg')  # headless-safe; callers may override before importing
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from zte.data.schema import ET_MEASURES  # noqa: E402
from zte.logging_utils import get_logger  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zte.data.dataset import ZuCoDataset

_LOG = get_logger('data.viz')


def plot_missingness(ds: ZuCoDataset) -> Figure:
    """Bars of missing-rate per eye-tracking measure, split by task.

    Args:
        ds (ZuCoDataset): A built dataset.

    Returns:
        Figure: The created `Figure`.

    """
    w = ds.words
    measures = [m for m in ET_MEASURES if m in w]
    tasks = sorted(w['task'].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(len(tasks), 1)
    x = np.arange(len(measures))
    # Plot the missing-rate per eye-tracking measure, split by task.
    for i, task in enumerate(tasks):
        sub = w[w['task'] == task]
        rates = [float(sub[m].isna().mean()) for m in measures]
        ax.bar(x + i * width, rates, width, label=task)

    # Set the x-ticks and labels.
    ax.set_xticks(x + width * (len(tasks) - 1) / 2)
    ax.set_xticklabels(measures)
    ax.set_ylabel('missing rate')

    ax.set_title('Word-level missingness by eye-tracking measure')
    ax.legend(title='task')

    fig.tight_layout()
    return fig


def plot_et_distributions(ds: ZuCoDataset) -> Figure:
    """Histograms of the present eye-tracking durations.

    Args:
        ds (ZuCoDataset): A built dataset.

    Returns:
        Figure: The created `Figure`.

    """
    w = ds.words
    measures = [m for m in ET_MEASURES if m in w]
    fig, axes = plt.subplots(1, len(measures), figsize=(3.2 * len(measures), 3.4))
    axes = np.atleast_1d(axes)
    for ax, measure in zip(axes, measures, strict=False):
        vals = w[measure].dropna()
        ax.hist(vals, bins=40, edgecolor='black')
        median = float(vals.median()) if len(vals) else float('nan')
        ax.set_title(f'{measure} (median={median:.0f} ms)')
        ax.set_xlabel('ms')
    fig.suptitle('Word-level eye-tracking durations')
    fig.tight_layout()
    return fig


def plot_correlations(ds: ZuCoDataset) -> Figure:
    """Correlation matrix of eye-tracking measures (present values).

    Args:
        ds (ZuCoDataset): A built dataset.

    Returns:
        Figure: The created `Figure`.

    """
    w = ds.words
    measures = [m for m in ET_MEASURES if m in w]
    corr = w[measures].astype(float).corr().to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(measures)))
    ax.set_xticklabels(measures, rotation=45, ha='right')
    ax.set_yticks(range(len(measures)))
    ax.set_yticklabels(measures)
    for i in range(len(measures)):
        for j in range(len(measures)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_title('Eye-tracking measure correlations')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_omission_by_length(ds: ZuCoDataset) -> Figure:
    """Omission probability and mean TRT as a function of word length.

    Args:
        ds (ZuCoDataset): A built dataset.

    Returns:
        Figure: The created `Figure`.
    """
    w = ds.words
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    by_len = w.groupby('word_len')['is_omitted'].mean()
    ax1.plot(by_len.index, by_len.to_numpy(), marker='o', color='crimson')
    ax1.set(xlabel='word length (chars)', ylabel='omission rate', title='Omission vs length')
    if 'TRT' in w:
        trt = w.dropna(subset=['TRT']).groupby('word_len')['TRT'].mean()
        ax2.plot(trt.index, trt.to_numpy(), marker='o')
        ax2.set(
            xlabel='word length (chars)', ylabel='mean TRT (ms)', title='Reading time vs length'
        )
    fig.tight_layout()
    return fig


def plot_eeg_availability(ds: ZuCoDataset, max_words: int = 600) -> Figure:
    """Heatmap of per-word EEG presence across subjects (first `max_words`).

    Args:
        ds (ZuCoDataset): A built dataset.
        max_words (int): Cap on the number of flattened word columns shown.

    Returns:
        Figure: The created `Figure`.
    """
    w = ds.words.copy()
    if ds.presence is not None:
        w = w.assign(_present=ds.presence.astype(float))
    else:
        w = w.assign(_present=w['has_word_eeg'].astype(float))
    pivot = w.pivot_table(
        index='subject', columns=['sentence_idx', 'word_idx'], values='_present', aggfunc='first'
    ).fillna(0.0)
    data = pivot.to_numpy()[:, :max_words]
    fig, ax = plt.subplots(figsize=(12, 0.5 * len(pivot) + 1.5))
    im = ax.imshow(data, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('flattened sentence/word index')
    ax.set_title('Word-level EEG availability')
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='present')
    fig.tight_layout()
    return fig


def plot_channel_importance(scores: np.ndarray, title: str = 'Channel importance') -> Figure:
    """Plots a per-channel importance curve (e.g. from feature selection).

    Args:
        scores (np.ndarray): 1-D importance scores; reshaped to a rough grid when length 105.
        title (str): Figure title.

    Returns:
        Figure: The created `Figure`.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(scores, color='purple', lw=1.3)
    ax.set(xlabel='channel index', ylabel='importance', title=title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_training_curves(history: dict[str, list[float]]) -> Figure:
    """Plots loss/metric curves recorded during training.

    Args:
        history (dict[str, list[float]]): Mapping of metric name to a per-step/epoch list of values.

    Returns:
        Figure: The created `Figure`.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, values in history.items():
        if values:
            ax.plot(range(1, len(values) + 1), values, marker='o', ms=3, label=name)
    ax.set(xlabel='epoch', ylabel='value', title='ZTE training history')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_overview(ds: ZuCoDataset, out_dir: str | Path) -> list[Path]:
    """Renders the standard analysis figures and writes them as PNGs.

    Args:
        ds (ZuCoDataset): A built dataset.
        out_dir (str | Path): Destination directory (created if needed).

    Returns:
        list[Path]: Paths of the written `Figure`s.

    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    builders = {
        'missingness.png': plot_missingness,
        'et_distributions.png': plot_et_distributions,
        'et_correlations.png': plot_correlations,
        'omission_by_length.png': plot_omission_by_length,
        'eeg_availability.png': plot_eeg_availability,
    }
    written: list[Path] = []
    for filename, builder in builders.items():
        try:
            fig = builder(ds)
        except (ValueError, KeyError, IndexError) as exc:  # robust to tiny datasets
            _LOG.warning('Skipped %s: %r', filename, exc)
            continue
        path = out / filename
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        written.append(path)
    _LOG.info('Wrote %d overview figures to %s', len(written), out)
    return written
