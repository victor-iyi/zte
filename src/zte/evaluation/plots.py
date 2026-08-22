"""Figures for the ZTE evaluation suite.

Every function returns a Matplotlib `Figure` (Agg backend, headless-safe) so the report layer can save them.
2-D projections use a plain NumPy PCA to avoid extra dependencies.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any, Final

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# One colour per conditioning arm, shared with the Plotly dashboard so an arm reads the same in the report and
# on the page: warm is the EEG-driven decode, earth and slate are the controls it has to beat.
CAPACITY_ARM_COLOURS: Final[dict[str, str]] = {
    'model': '#e4572e',
    'length_only': '#b08968',
    'shuffled_eeg': '#5c677d',
    'mismatch': '#4a5759',
    'null_prefix': '#8896ab',
}
"""Colour per capacity arm."""

# Draw order as well as legend order: the model first because it is the claim, then the controls in the order
# the certification argues them.
CAPACITY_ARM_LABELS: Final[dict[str, str]] = {
    'model': 'model (EEG prefix)',
    'length_only': 'length-only prefix',
    'shuffled_eeg': 'shuffled EEG (derangement)',
    'mismatch': 'mismatched stimulus',
    'null_prefix': 'no prefix',
}
"""Legend label per capacity arm, in the order the figures draw them."""

# The two arms whose interval is drawn: the claim and the control that decides it. Four ribbons on one panel
# hide the very gap the panel exists to show.
CAPACITY_RIBBON_ARMS: Final[tuple[str, str]] = ('model', 'length_only')
"""Arms whose bootstrap CI is drawn as a ribbon."""


def _pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Projects embeddings to 2-D with PCA (centred SVD)."""
    x = np.asarray(embeddings, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return (x @ vt[:2].T).astype(np.float32)


def scatter_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    categorical: bool,
    label_name: str = 'label',
) -> Figure:
    """2-D PCA scatter of embeddings coloured by a label.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
        labels (np.ndarray): Per-point labels `(n_samples,)`.
        title (str): Figure title.
        categorical (bool): Treat labels as categories (legend) vs continuous (bar).
        label_name (str): Legend/colourbar caption.

    Returns:
        Figure: The created figure.
    """
    xy = _pca_2d(embeddings)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    labels = np.asarray(labels)
    if categorical:
        for value in np.unique(labels):
            m = labels == value
            ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.6, label=str(value))
        ax.legend(title=label_name, fontsize=8, markerscale=1.5)
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels.astype(float), s=10, alpha=0.7, cmap='viridis')
        fig.colorbar(sc, ax=ax, label=label_name)
    ax.set(xlabel='PC1', ylabel='PC2', title=title)
    fig.tight_layout()
    return fig


def bar_probe_comparison(rows: list[dict[str, Any]], metric: str = 'linear_score', title: str | None = None) -> Figure:
    """Grouped bar chart of probe scores per target across representations.

    Args:
        rows (list[dict[str, Any]]): Output of `zte.evaluation.metrics.representation_comparison`.
        metric (str): `linear_score` or `knn_score`.
        title (str | None): Optional title.

    Returns:
        Figure: The created figure.
    """
    targets = sorted({r['target'] for r in rows})
    reps = sorted({r['representation'] for r in rows})
    lookup = {(r['target'], r['representation']): r for r in rows}

    x = np.arange(len(targets))
    width = 0.8 / max(len(reps), 1)
    fig, ax = plt.subplots(figsize=(1.8 * len(targets) + 3, 4.5))
    for i, rep in enumerate(reps):
        scores = [lookup.get((t, rep), {}).get(metric, np.nan) for t in targets]
        ax.bar(x + i * width, scores, width, label=rep)
    # Baseline markers (same per target across reps).
    baselines = [lookup.get((t, reps[0]), {}).get('baseline', np.nan) for t in targets]
    for xi, base in zip(x, baselines, strict=True):
        ax.hlines(
            base,
            xi - 0.4,
            xi + 0.4 + width * (len(reps) - 1),
            colors='black',
            linestyles='dashed',
            linewidth=1,
        )
    ax.set_xticks(x + width * (len(reps) - 1) / 2)
    ax.set_xticklabels(targets, rotation=20, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(title or f'Probe comparison ({metric}); dashed = baseline')
    ax.legend(title='representation', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def retrieval_curve(topk: dict[int, float], chance: float, title: str) -> Figure:
    """Top-K retrieval accuracy curve with a chance reference line.

    Args:
        topk (dict[int, float]): Map of K -> accuracy.
        chance (float): Random-chance Top-1 reference.
        title (str): Figure title.

    Returns:
        Figure: The created figure.
    """
    ks = sorted(topk)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ks, [topk[k] for k in ks], marker='o', label='ZTE retrieval')
    ax.axhline(chance, color='crimson', linestyle='dashed', linewidth=1, label='chance (Top-1)')
    ax.set(xlabel='K', ylabel='Top-K accuracy', title=title, ylim=(0, 1.02))
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def similarity_distribution(
    embeddings: np.ndarray, group_ids: np.ndarray, title: str, max_pairs: int = 20000
) -> Figure:
    """Histogram of cosine similarity for same-content vs different-content pairs.

    A clear rightward shift of the same-content distribution is evidence the
    embedding encodes stimulus content rather than noise.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
        group_ids (np.ndarray): Content/group id per row.
        title (str): Figure title.
        max_pairs (int): Cap on sampled pairs per category.

    Returns:
        Figure: The created figure.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    group_ids = np.asarray(group_ids)
    rng = np.random.default_rng(0)
    n = len(x)
    a = rng.integers(0, n, size=max_pairs)
    b = rng.integers(0, n, size=max_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    sims = np.sum(x[a] * x[b], axis=1)
    same = group_ids[a] == group_ids[b]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if same.any():
        ax.hist(sims[same], bins=40, alpha=0.6, density=True, label='same content')
    if (~same).any():
        ax.hist(sims[~same], bins=40, alpha=0.6, density=True, label='different content')
    ax.set(xlabel='cosine similarity', ylabel='density', title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def embedding_health_plot(embeddings: np.ndarray, title: str = 'Embedding health') -> Figure:
    """Two-panel collapse check: per-dim std and cumulative PCA variance.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
        title (str): Figure suptitle.

    Returns:
        Figure: The created figure.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    std = np.sort(x.std(axis=0))[::-1]
    sv = np.linalg.svd(x, compute_uv=False)
    var = sv**2
    cum = np.cumsum(var) / var.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(std, color='teal')
    ax1.set(xlabel='dimension (sorted)', ylabel='std', title='Per-dimension std (collapse check)')
    ax1.grid(alpha=0.3)
    ax2.plot(np.arange(1, len(cum) + 1), cum, color='purple')
    ax2.axhline(0.9, color='gray', linestyle='dashed', linewidth=1)
    ax2.set(xlabel='# components', ylabel='cumulative variance', title='PCA spectrum')
    ax2.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def region_importance_heatmap(
    rows: list[dict[str, Any]], title: str = 'Scalp-region importance by attribute'
) -> Figure:
    """Heatmap of normalised region importance (regions x targets).

    Args:
        rows (list[dict[str, Any]]): Output of `zte.data.montage.regions.region_importance`.
        title (str): Figure title.

    Returns:
        Figure: The created heatmap.
    """
    frame = _pivot(rows, index='region', column='target', value='importance')
    regions, targets = list(frame.index), list(frame.columns)
    mat = frame.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.4 * len(targets) + 3, 0.5 * len(regions) + 2))
    im = ax.imshow(mat, aspect='auto', cmap='magma')
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=20, ha='right')
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions)
    for i in range(len(regions)):
        for j in range(len(targets)):
            ax.text(
                j,
                i,
                f'{mat[i, j]:.2f}',
                ha='center',
                va='center',
                color='white' if mat[i, j] < mat.max() * 0.6 else 'black',
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, label='share of decodable info')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def breakdown_bars(rows: list[dict[str, Any]], metric: str, group: str, title: str | None = None) -> Figure:
    """Bar chart of one metric across the values of one stratification column.

    Args:
        rows (list[dict[str, Any]]): Output of `zte.evaluation.breakdown.stratified_report`.
        metric (str): Metric key to plot (e.g. `retrieval_top1`).
        group (str): Which `group` column's strata to show (e.g. `subject`).
        title (str | None): Optional title.

    Returns:
        Figure: The created bar chart.
    """
    sel = [r for r in rows if r.get('group') == group and metric in r]
    sel.sort(key=lambda r: r['value'])
    labels = [r['value'] for r in sel]
    values = [r[metric] for r in sel]
    fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 3, 4.2))
    ax.bar(labels, values, color='#4C78A8')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set(ylabel=metric, title=title or f'{metric} by {group}')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def analogy_bars(report: dict[str, Any], title: str = 'Vector-arithmetic transfer') -> Figure:
    """Grouped bars: transfer-analogy Top-1 vs chance (and a raw-feature control).

    Args:
        report (dict[str, Any]): Output of `zte.evaluation.analogy.analogy_report`.
        title (str): Figure title.

    Returns:
        Figure: The created bar chart.
    """
    blocks = [
        ('subject\n(ZTE)', report.get('subject_transfer', {})),
        ('subject\n(raw)', report.get('subject_transfer_raw', {})),
        ('task\n(ZTE)', report.get('task_transfer', {})),
    ]
    blocks = [(name, b) for name, b in blocks if b]
    labels = [n for n, _ in blocks]
    top1 = [b.get('top1', np.nan) for _, b in blocks]
    chance = [b.get('chance_top1', np.nan) for _, b in blocks]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 4.5))
    ax.bar(x - 0.2, top1, 0.4, label='Top-1', color='#54A24B')
    ax.bar(x + 0.2, chance, 0.4, label='chance', color='#BAB0AC')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set(ylabel='retrieval accuracy', title=title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def _pivot(rows: list[dict[str, Any]], index: str, column: str, value: str) -> Any:
    """Pivots tidy rows into an index x column matrix of `value`."""
    import pandas as pd

    return pd.DataFrame(rows).pivot(index=index, columns=column, values=value).fillna(0.0)


def probe_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> Figure:
    """Predicted-vs-true scatter for a regression probe, with an identity line.

    Args:
        y_true (np.ndarray): Ground-truth values `(n_samples,)`.
        y_pred (np.ndarray): Predicted values `(n_samples,)`.
        title (str): Figure title.

    Returns:
        Figure: The created figure.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, s=10, alpha=0.4)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color='crimson', linestyle='dashed', linewidth=1)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float('nan')
    ax.set(xlabel='true', ylabel='predicted (kNN)', title=f'{title}  (r={corr:.2f})')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def retrieval_rank_distribution(
    ranks: np.ndarray,
    n_bank: int,
    chance_top1: float,
    title: str = 'Retrieval rank distribution',
) -> Figure:
    """Headline honesty figure: rank histogram and rank-percentile CDF.

    A healthy retrieval space piles probability mass at rank 1 (left panel) and
    its rank-percentile CDF bows above the chance diagonal (right panel).

    Args:
        ranks (np.ndarray): 1-based rank of the correct match per query `(n_queries,)`.
        n_bank (int): Size of the retrieval bank (max possible rank).
        chance_top1 (float): Random-chance Top-1 accuracy reference.
        title (str): Figure suptitle.

    Returns:
        Figure: The created figure.
    """
    ranks = np.asarray(ranks, dtype=float).ravel()
    ranks = ranks[np.isfinite(ranks)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    if ranks.size == 0 or n_bank <= 0:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, 'no ranks', ha='center', va='center', transform=ax.transAxes)
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    n_bins = int(min(40, max(1, np.ceil(ranks.max()))))
    ax1.hist(ranks, bins=n_bins, color='#4C78A8', alpha=0.85)
    ax1.axvline(1.0, color='crimson', linestyle='dashed', linewidth=1, label='rank 1')
    ax1.set(xlabel='rank of correct match', ylabel='queries', title='Rank histogram')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    pct = np.sort(ranks / float(n_bank))
    cdf = np.arange(1, pct.size + 1) / pct.size
    ax2.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color='gray',
        linestyle='dashed',
        linewidth=1,
        label='chance diagonal',
    )
    ax2.plot(pct, cdf, color='#54A24B', linewidth=1.8, label='ZTE retrieval')
    ax2.axhline(
        chance_top1,
        color='crimson',
        linestyle='dotted',
        linewidth=1,
        label=f'chance Top-1 ({chance_top1:.2%})',
    )
    ax2.set(
        xlabel='rank percentile',
        ylabel='cumulative fraction',
        title='Rank-percentile CDF',
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def variance_budget_pie(summary: dict[str, Any], title: str = 'Variance budget: who vs what') -> Figure:
    """Pie of the variance budget (which attribute each dimension serves).

    Args:
        summary (dict[str, Any]): The `neurons` summary block with keys `variance_budget`
            (attr -> share in [0, 1]), `who_variance` and `what_variance`.
        title (str): Figure title (annotated with who/what shares).

    Returns:
        Figure: The created figure.
    """
    budget = dict(summary.get('variance_budget') or {})
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if not budget:
        ax.text(0.5, 0.5, 'no variance budget', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    palette = {
        'subject': '#eda100',
        'word_len': '#2a78d6',
        'log_freq': '#1baf7a',
        'category': '#4a3aa7',
        'task': '#008300',
        'none': '#b8b6ad',
    }
    fallback = '#8c8c8c'
    attrs = list(budget)
    shares = [float(budget[a]) for a in attrs]
    colors = [palette.get(a, fallback) for a in attrs]
    ax.pie(shares, labels=attrs, colors=colors, autopct='%1.0f%%', startangle=90)
    ax.set_aspect('equal')
    who = float(summary.get('who_variance', np.nan))
    what = float(summary.get('what_variance', np.nan))
    ax.set_title(f'{title}\nwho={who:.0%} what={what:.0%}')
    fig.tight_layout()
    return fig


def variance_budget_bars(rows: list[dict[str, Any]], title: str = 'Who vs what by run') -> Figure:
    """Grouped bars of who-variance vs what-variance across runs.

    Args:
        rows (list[dict[str, Any]]): List of `{'run': str, 'who': float, 'what': float}`.
        title (str): Figure title.

    Returns:
        Figure: The created bar chart.
    """
    fig, ax = plt.subplots(figsize=(1.4 * max(len(rows), 1) + 3, 4.5))
    if not rows:
        ax.text(0.5, 0.5, 'no runs', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    labels = [str(r.get('run', i)) for i, r in enumerate(rows)]
    who = [float(r.get('who', np.nan)) for r in rows]
    what = [float(r.get('what', np.nan)) for r in rows]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, who, 0.4, label='who (identity)', color='#eda100')
    ax.bar(x + 0.2, what, 0.4, label='what (content)', color='#1baf7a')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set(ylabel='share of variance', title=title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def geometry_before_after(
    before: np.ndarray,
    after: np.ndarray,
    group_ids: np.ndarray,
    label_before: str = 'raw',
    label_after: str = 'whiten+ABTT',
    title: str = 'Geometry before vs after',
) -> Figure:
    """Same-vs-different cosine histograms before and after the geometry fix.

    Visualises the anti-cone / anti-hubness fix honestly: each panel is annotated
    with its anisotropy so a shrinking cone is legible.

    Args:
        before (np.ndarray): Raw embeddings `(n_samples, embed_dim)`.
        after (np.ndarray): Post-fix embeddings `(n_samples, embed_dim)`.
        group_ids (np.ndarray): Content/group id per row `(n_samples,)`.
        label_before (str): Panel title for the `before` embeddings.
        label_after (str): Panel title for the `after` embeddings.
        title (str): Figure suptitle.

    Returns:
        Figure: The created figure.
    """
    from zte.evaluation import metrics as M

    group_ids = np.asarray(group_ids)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    def _panel(ax: Any, emb: np.ndarray, label: str) -> None:
        x = np.asarray(emb, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] < 2:
            ax.text(0.5, 0.5, 'too few points', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            return
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        n = len(x)
        a = rng.integers(0, n, size=20000)
        b = rng.integers(0, n, size=20000)
        keep = a != b
        a, b = a[keep], b[keep]
        sims = np.sum(x[a] * x[b], axis=1)
        same = group_ids[a] == group_ids[b]
        if same.any():
            ax.hist(sims[same], bins=40, alpha=0.6, density=True, label='same content', color='#4C78A8')
        if (~same).any():
            ax.hist(
                sims[~same],
                bins=40,
                alpha=0.6,
                density=True,
                label='different content',
                color='#E45756',
            )
        ax.set(
            xlabel='cosine similarity',
            ylabel='density',
            title=f'{label}  (anisotropy={M.anisotropy(x):.2f})',
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    _panel(axes[0], before, label_before)
    _panel(axes[1], after, label_after)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def subject_similarity_heatmap(
    emb: np.ndarray, subjects: np.ndarray, title: str = 'Cross-subject centroid cosine'
) -> Figure:
    """Subject x subject cosine matrix of per-subject centroids.

    A hubness / identity diagnostic: strong off-diagonal structure means the
    space encodes who rather than what.

    Args:
        emb (np.ndarray): Embeddings `(n_samples, embed_dim)`.
        subjects (np.ndarray): Subject code per row `(n_samples,)`.
        title (str): Figure title.

    Returns:
        Figure: The created heatmap.
    """
    subjects = np.asarray(subjects)
    codes = list(np.unique(subjects))
    fig, ax = plt.subplots(figsize=(0.6 * max(len(codes), 1) + 3, 0.6 * max(len(codes), 1) + 2))
    if len(codes) == 0 or np.asarray(emb).ndim != 2:
        ax.text(0.5, 0.5, 'no subjects', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    x = np.asarray(emb, dtype=np.float64)
    centroids = np.stack([x[subjects == c].mean(axis=0) for c in codes])
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    mat = centroids @ centroids.T

    im = ax.imshow(mat, cmap='magma', vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels([str(c) for c in codes], rotation=20, ha='right')
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([str(c) for c in codes])
    for i in range(len(codes)):
        for j in range(len(codes)):
            ax.text(
                j,
                i,
                f'{mat[i, j]:.2f}',
                ha='center',
                va='center',
                color='white' if mat[i, j] < mat.max() * 0.6 else 'black',
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, label='cosine')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def scalp_topomap(
    values: np.ndarray,
    coords_2d: np.ndarray,
    title: str = 'Scalp map',
    label: str = 'importance',
) -> Figure:
    """Interpolated scalp topography of per-channel values.

    Args:
        values (np.ndarray): Per-channel scalar `(n_channels,)`.
        coords_2d (np.ndarray): Channel positions `(n_channels, 2)` in `[0, 1]^2`
            (azimuthal-equidistant projection; +y is front).
        title (str): Figure title.
        label (str): Colourbar caption.

    Returns:
        Figure: The created figure.
    """
    values = np.asarray(values, dtype=float).ravel()
    coords = np.asarray(coords_2d, dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)

    finite = (
        np.isfinite(values) & np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
        if coords.ndim == 2 and coords.shape[1] == 2 and len(coords) == len(values)
        else np.zeros(len(values), dtype=bool)
    )
    xs, ys, vs = coords[finite, 0], coords[finite, 1], values[finite]

    # Head outline: unit circle centred at (0.5, 0.5) radius 0.5 with a nose triangle.
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.5 + 0.5 * np.cos(theta), 0.5 + 0.5 * np.sin(theta), color='#888888', linewidth=1)
    ax.plot([0.45, 0.5, 0.55], [0.98, 1.06, 0.98], color='#888888', linewidth=1)

    if finite.sum() < 4:
        if vs.size:
            sc = ax.scatter(xs, ys, c=vs, cmap='magma', s=40, edgecolors='k', linewidths=0.4)
            fig.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'no channels', ha='center', va='center', transform=ax.transAxes)
        fig.tight_layout()
        return fig

    tcf = ax.tricontourf(xs, ys, vs, levels=14, cmap='magma')
    ax.scatter(xs, ys, c='k', s=8, alpha=0.6)
    fig.colorbar(tcf, ax=ax, label=label, fraction=0.046, pad=0.04)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.15)
    fig.tight_layout()
    return fig


def neuron_selectivity_heatmap(
    top_neurons: list[dict[str, Any]], title: str = 'Neuron selectivity (top dimensions)'
) -> Figure:
    """Heatmap of per-dimension selectivity across attributes.

    Args:
        top_neurons (list[dict[str, Any]]): List of dicts each with `dim` (int) and
            `selectivity` (attr -> score in `[0, 1]`).
        title (str): Figure title.

    Returns:
        Figure: The created heatmap.
    """
    fig, ax = plt.subplots(figsize=(1.2 * max(len(top_neurons), 1) + 3, 0.5 * max(len(top_neurons), 1) + 2))
    if not top_neurons:
        ax.text(0.5, 0.5, 'no neurons', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    attrs: list[str] = []
    for n in top_neurons:
        for a in n.get('selectivity') or {}:
            if a not in attrs:
                attrs.append(a)
    if not attrs:
        ax.text(0.5, 0.5, 'no selectivity', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    dims = [int(n.get('dim', i)) for i, n in enumerate(top_neurons)]
    mat = np.array(
        [[float((n.get('selectivity') or {}).get(a, 0.0)) for a in attrs] for n in top_neurons],
        dtype=float,
    )
    im = ax.imshow(mat, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(attrs)))
    ax.set_xticklabels(attrs, rotation=20, ha='right')
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([f'#{d}' for d in dims])
    for i in range(len(dims)):
        for j in range(len(attrs)):
            ax.text(
                j,
                i,
                f'{mat[i, j]:.2f}',
                ha='center',
                va='center',
                color='white' if mat[i, j] < mat.max() * 0.6 else 'black',
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, label='variance explained')
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---- Decoder menu capacity ---- #


def _finite(value: Any) -> float:
    """A metric as a float, with anything missing, non-numeric or non-finite as NaN."""
    try:
        out = float(value)
    except TypeError, ValueError:
        return float('nan')

    return out if np.isfinite(out) else float('nan')


def _capacity_block(capacity: dict[str, Any]) -> dict[str, Any]:
    """The headline score-family and pool-flavor block of a capacity report."""
    headline = capacity.get('headline') or {}
    families = capacity.get('scores') or {}

    return ((families.get(headline.get('score')) or {}).get(headline.get('flavor'))) or {}


def _capacity_placeholder(fig: Figure, ax: Any, message: str, title: str) -> Figure:
    """Draws a centred message so a missing capacity report reads as absent rather than as a zero."""
    ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes, fontsize=11, color='#5c5c5c')
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()

    return fig


def _capacity_failures(capacity: dict[str, Any]) -> list[str]:
    """The clauses standing between this report and a certified menu size."""
    clauses = (capacity.get('verdict') or {}).get('capacity_clauses') or {}
    if clauses:
        return [name for name, passed in clauses.items() if not passed]

    per_k = _capacity_block(capacity).get('per_k') or {}
    first = per_k.get(str(min((int(k) for k in per_k), default=0)))

    return list((first or {}).get('failed_clauses') or [])


def _capacity_label(capacity: dict[str, Any], *, named: int = 2) -> str:
    """The certified menu size, or an em dash naming the clauses that failed -- never a blank and never a zero."""
    certified = capacity.get('certified_k')
    if certified is not None:
        return f'certified K = {certified}'

    failed = _capacity_failures(capacity)
    if not failed:
        return 'certified K = — (no cell scored)'

    rest = len(failed) - named

    return f'certified K = — (failed: {", ".join(failed[:named])}' + (f' +{rest} more)' if rest > 0 else ')')


def _capacity_subtitle(capacity: dict[str, Any]) -> str:
    """The provenance line every capacity panel carries: which cell, which pool, how many queries."""
    headline = capacity.get('headline') or {}
    line = (
        f'{headline.get("score", "?")} / {headline.get("flavor", "?")} · holdout {capacity.get("holdout", "?")} · '
        f'{capacity.get("n_queries", 0)} queries over a {capacity.get("n_gallery", 0)}-sentence gallery · '
        f'{capacity.get("tie_policy", "ties lose")}'
    )

    # The verdict keeps its own line so no wrap can split `certified K = —` from the clause that failed.
    return f'{textwrap.fill(line, width=104)}\n{_capacity_label(capacity)}'


def _capacity_arm_series(cells: list[dict[str, Any]], arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An arm's accuracy and CI bounds across menu sizes, NaN wherever that size held no scoreable pool."""
    accuracy, lo, hi = [], [], []
    for cell in cells:
        block = (cell.get('arms') or {}).get(arm) or {}
        interval = block.get('ci')
        accuracy.append(_finite(block.get('accuracy')))
        lo.append(_finite(interval[1]) if isinstance(interval, (list, tuple)) and len(interval) >= 3 else float('nan'))
        hi.append(_finite(interval[2]) if isinstance(interval, (list, tuple)) and len(interval) >= 3 else float('nan'))

    return np.asarray(accuracy), np.asarray(lo), np.asarray(hi)


def _capacity_run_label(capacity: dict[str, Any], index: int) -> str:
    """A per-run tick label: the held-out subject and the seed behind it."""
    seed = (capacity.get('provenance') or {}).get('seed')
    holdout = capacity.get('holdout') or f'run {index + 1}'

    return f'{holdout}\ns{seed}' if seed is not None else str(holdout)


def capacity_curve(
    capacity: dict[str, Any], title: str = 'Decoder menu capacity — accuracy against menu size'
) -> Figure:
    """Menu accuracy against menu size for every arm, with the model-over-length gap drawn as the result.

    Args:
        capacity (dict[str, Any]): A `zte.evaluation.audit.capacity.capacity_report` block.
        title (str, optional): Figure title. Defaults to 'Decoder menu capacity — accuracy against menu size'.

    Returns:
        Figure: The created figure.

    Note:
        The height of the model line is not the finding. A length-only prefix already scores well above chance
        inside a pool it shares a word count with, so what certifies is the vertical gap to the length-only
        trace -- which is why that gap is shaded and annotated with its paired sign-test p. Chance is a curve
        rather than a line because it is exactly `1/K` and moves with every menu size, and a size no pool could
        fill is greyed out so "unreachable" never reads as "the model failed here".
    """
    block = _capacity_block(capacity)
    per_k = block.get('per_k') or {}
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    ks = sorted(int(k) for k in per_k)
    if not ks:
        return _capacity_placeholder(fig, ax, 'no capacity report', title)

    cells = [per_k[str(k)] for k in ks]
    grid = np.asarray(ks, dtype=np.float64)
    chance = np.asarray([_finite(cell.get('chance')) for cell in cells])
    chance = np.where(np.isfinite(chance), chance, 1.0 / grid)

    unreachable = sorted({int(k) for k in (block.get('ks_unreachable') or [])})
    for k in unreachable:
        ax.axvspan(
            k / 1.32,
            k * 1.32,
            color='#ebe8e3',
            zorder=0,
            label='no pool could fill this menu' if k == unreachable[0] else None,
        )

    series = {arm: _capacity_arm_series(cells, arm) for arm in CAPACITY_ARM_LABELS}
    drawn = {arm: value for arm, value in series.items() if np.isfinite(value[0]).any()}
    if not drawn:
        return _capacity_placeholder(fig, ax, 'no arm scored at any menu size', title)

    # The gap is the claim, so it is filled before the lines and named in the legend as such.
    model, length = drawn.get('model'), drawn.get('length_only')
    if model is not None and length is not None:
        both = np.isfinite(model[0]) & np.isfinite(length[0])
        ax.fill_between(
            grid,
            np.where(both, length[0], np.nan),
            np.where(both, model[0], np.nan),
            where=both & (model[0] > length[0]),
            color=CAPACITY_ARM_COLOURS['model'],
            alpha=0.13,
            interpolate=True,
            zorder=1,
            label='model over length-only — the certifying gap',
        )

    ax.plot(
        grid,
        chance,
        color='#5c5c5c',
        linestyle=(0, (5, 3)),
        linewidth=1.4,
        zorder=2,
        label='chance = 1/K (moves with K)',
    )
    for arm, (accuracy, lo, hi) in drawn.items():
        colour = CAPACITY_ARM_COLOURS.get(arm, '#8c8c8c')
        headline_arm = arm == 'model'
        if arm in CAPACITY_RIBBON_ARMS and np.isfinite(lo).any():
            ax.fill_between(grid, lo, hi, color=colour, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(
            grid,
            accuracy,
            marker='o' if headline_arm else 's',
            markersize=7 if headline_arm else 4.5,
            linewidth=2.6 if headline_arm else 1.5,
            color=colour,
            zorder=4 if headline_arm else 3,
            label=CAPACITY_ARM_LABELS[arm],
        )

    stack = np.concatenate([np.concatenate(value) for value in drawn.values()] + [chance])
    finite = stack[np.isfinite(stack)]
    top = min(1.05, max(0.14, float(finite.max()) * 1.35)) if finite.size else 1.05

    certified = capacity.get('certified_k')
    if certified is not None:
        ax.axvline(float(certified), color='#1baf7a', linestyle='dotted', linewidth=1.8, zorder=2)
        ax.text(
            float(certified),
            top * 0.5,
            f'certified K = {certified}',
            color='#1baf7a',
            fontsize=9,
            rotation=90,
            ha='right',
            va='center',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 1.5},
        )

    # The paired delta is annotated where the claim is made: at the certified size, or at the largest size a
    # pool could actually fill when nothing certified.
    feasible = [int(k) for k in (block.get('ks_feasible') or ks)]
    focus = int(certified) if certified is not None else max(feasible, default=ks[0])
    paired = ((per_k.get(str(focus)) or {}).get('paired') or {}).get('length_only')
    if paired is not None and model is not None and length is not None:
        index = ks.index(focus)
        low, high = length[0][index], model[0][index]
        if np.isfinite(low) and np.isfinite(high):
            ax.annotate('', xy=(focus, high), xytext=(focus, low), arrowprops={'arrowstyle': '<->', 'color': '#e4572e'})
            ax.text(
                focus * 1.06,
                (low + high) / 2.0,
                f'{paired["delta"]:+.4f} over length-only\nsign-test p = {paired["sign_test_p"]:.2e}',
                fontsize=8,
                color='#e4572e',
                va='center',
            )

    ax.set_xscale('log', base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([f'{k}\nn={cell.get("n_queries", 0)}' for k, cell in zip(ks, cells, strict=True)])
    ax.minorticks_off()
    ax.set_ylim(0.0, top)
    ax.set_xlabel('menu size K (log2 axis) — n is the queries whose pool could fill that menu')
    ax.set_ylabel('accuracy: the read sentence scored above every distractor')
    ax.set_title(f'{title}\n{_capacity_subtitle(capacity)}', fontsize=10.5)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.92)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig


def capacity_bits_ledger(
    capacity: dict[str, Any], title: str = 'The bits ledger — free, certified, and still unrecovered'
) -> Figure:
    """One stacked bar of stimulus identity: what word count gives away, what certified, what is still missing.

    Args:
        capacity (dict[str, Any]): A `zte.evaluation.audit.capacity.capacity_report` block.
        title (str, optional): Figure title. Defaults to 'The bits ledger — free, certified, and still unrecovered'.

    Returns:
        Figure: The created figure.

    Note:
        The denominator is drawn as a bracket rather than left in the caption, because a certified bit count is
        credited against the identity that survives knowing word count, never against the full identity of the
        gallery. Nothing certified renders as a hatched remainder and an em dash, never as a zero-height bar.
    """
    bits = capacity.get('bits') or {}
    total, residual, free = (
        _finite(bits.get('entropy_identity')),
        _finite(bits.get('entropy_identity_given_length')),
        _finite(bits.get('bits_from_length')),
    )
    fig, ax = plt.subplots(figsize=(10, 4.0))
    if not (np.isfinite(total) and np.isfinite(residual) and np.isfinite(free)) or total <= 0.0:
        return _capacity_placeholder(fig, ax, 'no bits ledger', title)

    certified = _finite(bits.get('bits_certified'))
    earned = certified if np.isfinite(certified) else 0.0
    unrecovered = max(total - free - earned, 0.0)

    ax.barh(0.0, free, height=0.42, color='#b08968', label=f'word count, free — {free:.4f} bits')
    if earned > 0.0:
        ax.barh(
            0.0,
            earned,
            left=free,
            height=0.42,
            color='#e4572e',
            label=f'decoder certified — {earned:.4f} bits ({capacity.get("readout", "menu selection")})',
        )
    ax.barh(
        0.0,
        unrecovered,
        left=free + earned,
        height=0.42,
        color='#e9e9e9',
        edgecolor='#b5b5b5',
        hatch=None if earned > 0.0 else '//',
        label=(
            f'unrecovered — {unrecovered:.4f} bits'
            if earned > 0.0
            else f'unrecovered — {unrecovered:.4f} bits (nothing certified)'
        ),
    )

    # Staggered heights: the two references sit close enough on a 9.45-bit axis that one row would collide.
    for value, height, colour, note in (
        (total, 0.52, '#c1121f', f'{total:.4f} — full stimulus identity'),
        (free, 0.30, '#b08968', f'{free:.4f} — word count alone'),
    ):
        ax.axvline(value, color=colour, linestyle='dotted', linewidth=1.4)
        ax.text(value, height, note, ha='right', va='bottom', fontsize=8, color=colour)

    ax.annotate(
        '',
        xy=(free, -0.36),
        xytext=(total, -0.36),
        arrowprops={'arrowstyle': '|-|,widthA=0.4,widthB=0.4', 'color': '#4a5759', 'linewidth': 1.3},
    )
    ax.text(
        (free + total) / 2.0,
        -0.44,
        f'the honest denominator: {residual:.4f} bits of identity left once word count is known',
        ha='center',
        va='top',
        fontsize=9,
        color='#4a5759',
    )

    fraction = _finite(bits.get('fraction_of_residual'))
    share = '—' if not np.isfinite(fraction) else f'{fraction:.1%}'
    recovered = (
        f'recovered {certified:.4f} bits = {share} of the residual'
        if np.isfinite(certified)
        else f'recovered nothing of the {residual:.4f}-bit residual'
    )
    failed = _capacity_failures(capacity)
    footer = f'{_capacity_label(capacity)} · estimator {bits.get("estimator", "log2(certified K)")} · {recovered}'
    if not np.isfinite(certified) and failed:
        footer += '\nevery failing clause: ' + textwrap.fill(', '.join(failed), width=118, subsequent_indent='  ')
    ax.text(0.0, -0.72, footer, ha='left', va='top', fontsize=9, color='#333333')

    ax.set_xlim(0.0, total * 1.02)
    ax.set_ylim(-1.0, 0.95)
    ax.set_yticks([])
    ax.set_xlabel('bits of stimulus identity')
    ax.set_title(f'{title}\n{_capacity_subtitle(capacity)}', fontsize=10.5)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.92)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()

    return fig


def capacity_seed_strip(
    reports: list[dict[str, Any]],
    *,
    k: int = 2,
    pooled: dict[str, Any] | None = None,
    title: str = 'Every run as its own point — the 2-way menu across seeds',
) -> Figure:
    """Per-run menu accuracy at one size as individual points, over the interval the runs jointly support.

    Args:
        reports (list[dict[str, Any]]): One `capacity_report` block per seed or per held-out subject.
        k (int, optional): Menu size to read. Defaults to 2.
        pooled (dict[str, Any] | None, optional): A `pooled_capacity` block, whose verdict is printed beneath
            the points. Defaults to None.
        title (str, optional): Figure title. Defaults to 'Every run as its own point — the 2-way menu across seeds'.

    Returns:
        Figure: The created figure.

    Note:
        Run-to-run drift on this project has been the size of the effect, so a bar over the seeds would hide
        exactly what a reader needs. Each run keeps its own interval, and a single run is labelled as a
        measurement rather than a result.
    """
    points: list[dict[str, Any]] = []
    for index, report in enumerate(reports or []):
        cell = ((_capacity_block(report).get('per_k') or {}).get(str(k))) or {}
        score = _finite(cell.get('accuracy'))
        if not np.isfinite(score):
            continue

        reported = cell.get('ci')
        interval = list(reported) if isinstance(reported, (list, tuple)) and len(reported) >= 3 else [score] * 3
        points.append(
            {
                'label': _capacity_run_label(report, index),
                'accuracy': score,
                'lo': _finite(interval[1]),
                'hi': _finite(interval[2]),
                'chance': _finite(cell.get('chance')) if cell.get('chance') else 1.0 / k,
                'certified': report.get('certified_k') is not None,
                'n': int(cell.get('n_queries') or 0),
            }
        )

    fig, ax = plt.subplots(figsize=(max(6.0, 1.5 * len(points) + 3.5), 5.0))
    if not points:
        return _capacity_placeholder(fig, ax, f'no run scored a {k}-way menu', title)

    x = np.arange(len(points), dtype=np.float64)
    accuracy = np.asarray([p['accuracy'] for p in points])
    lo = np.asarray([p['lo'] if np.isfinite(p['lo']) else p['accuracy'] for p in points])
    hi = np.asarray([p['hi'] if np.isfinite(p['hi']) else p['accuracy'] for p in points])
    chance = float(np.nanmean([p['chance'] for p in points]))

    ax.axhspan(
        float(lo.min()),
        float(hi.max()),
        color='#dfe4ea',
        alpha=0.75,
        zorder=0,
        label='interval every run supports (union of per-run CIs)',
    )
    ax.axhline(
        float(accuracy.mean()),
        color='#5c677d',
        linestyle='dashed',
        linewidth=1.3,
        zorder=1,
        label=f'mean over runs = {accuracy.mean():.4f}',
    )
    ax.axhline(chance, color='#5c5c5c', linestyle=(0, (5, 3)), linewidth=1.4, zorder=1, label=f'chance = 1/{k}')
    ax.errorbar(
        x, accuracy, yerr=[accuracy - lo, hi - accuracy], fmt='none', ecolor='#8c8c8c', capsize=4, linewidth=1.2
    )
    for certified, colour, label in (
        (True, CAPACITY_ARM_COLOURS['model'], 'run certified a menu size'),
        (False, '#8896ab', 'run certified nothing'),
    ):
        picked = [i for i, p in enumerate(points) if p['certified'] is certified]
        if picked:
            ax.scatter(
                x[picked],
                accuracy[picked],
                s=80,
                color=colour,
                edgecolors='white',
                linewidths=0.9,
                zorder=4,
                label=label,
            )

    if len(points) == 1:
        ax.text(
            0.5,
            0.04,
            'one seed is a measurement, not yet a result',
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=10,
            color='#c1121f',
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f'{p["label"]}\nn={p["n"]}' for p in points], fontsize=8)
    ax.set_xlim(-0.6, len(points) - 0.4)
    ax.set_ylim(0.0, min(1.05, max(float(hi.max()) * 1.3, chance * 1.6)))
    ax.set_ylabel(f'{k}-way menu accuracy')
    subtitle = (pooled or {}).get('reason') or f'{len(points)} run(s) at K = {k}'
    ax.set_title(f'{title}\n{subtitle}', fontsize=10.5)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, framealpha=0.92)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    return fig


def capacity_vs_length_oracle(
    capacity: dict[str, Any],
    arm: str = 'length_only',
    title: str = 'Paired against the control — who wins each query, not who wins on average',
) -> Figure:
    """Per-query wins, losses and ties against one control arm, per menu size, with the exact sign-test p.

    Args:
        capacity (dict[str, Any]): A `zte.evaluation.audit.capacity.capacity_report` block.
        arm (str, optional): Control arm to compare against. Defaults to 'length_only'.
        title (str, optional): Figure title. Defaults to a paired-comparison caption.

    Returns:
        Figure: The created figure.

    Note:
        Every comparison in the certification is paired on identical query indices, and this is the panel that
        shows it: a mean difference can be carried by a handful of queries, a sign test over wins and losses
        cannot. Ties are drawn neutral but count as losses in the accuracy, which is why they are labelled.
    """
    block = _capacity_block(capacity)
    per_k = block.get('per_k') or {}
    alpha = _finite((capacity.get('headline') or {}).get('alpha'))
    alpha = alpha if np.isfinite(alpha) else 0.05

    rows = [
        (int(key), cell['paired'][arm])
        for key, cell in sorted(per_k.items(), key=lambda item: int(item[0]))
        if arm in (cell.get('paired') or {})
    ]
    fig, ax = plt.subplots(figsize=(10, 0.85 * max(len(rows), 1) + 3.2))
    if not rows:
        return _capacity_placeholder(fig, ax, f'no paired comparison against {arm}', title)

    y = np.arange(len(rows), dtype=np.float64)
    wins = np.asarray([float(cell['model_wins']) for _, cell in rows])
    losses = np.asarray([float(cell['control_wins']) for _, cell in rows])
    ties = np.asarray([float(cell['ties']) for _, cell in rows])

    ax.barh(y, ties, left=-ties / 2.0, height=0.5, color='#e9e9e9', edgecolor='#b5b5b5', label='ties (count as losses)')
    ax.barh(y, wins, height=0.5, color=CAPACITY_ARM_COLOURS['model'], label='model wins the query')
    ax.barh(
        y,
        -losses,
        height=0.5,
        color=CAPACITY_ARM_COLOURS.get(arm, '#8c8c8c'),
        label=f'{CAPACITY_ARM_LABELS.get(arm, arm)} wins the query',
    )
    ax.axvline(0.0, color='#333333', linewidth=1.1)

    span = float(max(np.max(wins), np.max(losses), np.max(ties) / 2.0, 1.0))
    for index, (k, cell) in enumerate(rows):
        passes = cell['ci'][1] > 0.0 and cell['sign_test_p'] < alpha
        ax.text(
            span * 1.06,
            float(index),
            f'Δ {cell["delta"]:+.4f} [{cell["ci"][1]:+.4f}, {cell["ci"][2]:+.4f}]   '
            f'p = {cell["sign_test_p"]:.2e}   {"clause holds" if passes else "clause fails"}',
            va='center',
            fontsize=8,
            color='#333333' if passes else '#c1121f',
        )

    certified = capacity.get('certified_k')

    def _tick(k: int, cell: dict[str, Any]) -> str:
        """One row label: the menu size, whether it certified, and how many pairs stand behind it."""
        mark = ' (certified)' if certified is not None and k <= int(certified) else ''

        return f'K = {k}{mark}\n{cell["n_pairs"]} pairs'

    ax.set_yticks(y)
    ax.set_yticklabels([_tick(k, cell) for k, cell in rows], fontsize=9)
    ticks = np.linspace(-span, span, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{abs(t):.0f}' for t in ticks])
    ax.set_xlim(-span * 1.15, span * 3.4)
    ax.invert_yaxis()
    ax.set_xlabel(f'queries, paired one-for-one — {CAPACITY_ARM_LABELS.get(arm, arm)} on the left, model on the right')
    ax.set_title(f'{title}\n{_capacity_subtitle(capacity)}', fontsize=10.5)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, framealpha=0.92)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()

    return fig
