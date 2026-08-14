"""Figures for the ZTE evaluation suite.

Every function returns a Matplotlib `Figure` (Agg backend, headless-safe) so the report layer can save them.
2-D projections use a plain NumPy PCA to avoid extra dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


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
