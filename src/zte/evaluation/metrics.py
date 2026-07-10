"""Quantitative evaluation of ZTE embeddings as a re-purposable representation.

The goal is evidence -- without a trained decoder -- that the encoder turns EEG into a useful, well-structured space. Three families of metric are provided:

- **Geometry / health** (label-free): effective rank, uniformity, alignment, anisotropy,
  dead dimensions -- detects representation collapse and measures how well the space is used.
- **Transfer probes** (supervised): linear and kNN probes that predict known attributes (word length, frequency, subject, task)
  from *frozen* embeddings, compared against the raw band-power features and a noise-matched control.
- **Content retrieval**: do embeddings of the same stimulus (across subjects) retrieve each other better than chance (Top-K, MRR)?
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from zte.inference.retrieval import NearestNeighborIndex
from zte.training.metrics import linear_probe

type ProbeTask = Literal['classification', 'regression']


# --------------------------------------------------------------------------- #
# Geometry / health (label-free)
# --------------------------------------------------------------------------- #


def effective_rank(embeddings: np.ndarray) -> float:
    """Computes the effective rank (Roy & Vetterli) of an embedding matrix.

    The effective rank is `exp(H)` where `H` is the Shannon entropy of the normalised singular-value spectrum. A value near 1
    indicates collapse onto a single direction; a value near `embed_dim` indicates a well-spread space.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.

    Returns:
        float: The effective rank.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(x, compute_uv=False)
    sv = sv[sv > 1e-12]
    if sv.size == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def uniformity(embeddings: np.ndarray, t: float = 2.0, max_samples: int = 2000) -> float:
    """Computes the uniformity metric (Wang & Isola, 2020).

    Lower (more negative) means embeddings are spread more uniformly on the unit hypersphere -- a sign the contrastive objective avoided collapse.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
        t (float): RBF temperature.
        max_samples (int): Subsample cap for the pairwise computation.

    Returns:
        float: `log E[exp(-t * ||x - y||^2)]` over distinct normalised pairs.
    """
    x = _normalize(_subsample(embeddings, max_samples))
    sq = np.maximum(0.0, 2.0 - 2.0 * (x @ x.T))
    iu = np.triu_indices(len(x), k=1)
    if iu[0].size == 0:
        return float('nan')
    return float(np.log(np.mean(np.exp(-t * sq[iu]))))


def alignment(embeddings: np.ndarray, pairs: np.ndarray, alpha: float = 2.0) -> float:
    """Computes the alignment metric over positive pairs (Wang & Isola, 2020).

    Lower means positive pairs (e.g. neighbouring words) sit closer together.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
        pairs (np.ndarray): Integer index pairs `(n_pairs, 2)` of positives.
        alpha (float): Distance exponent.

    Returns:
        float: Mean `||x_i - x_j||^alpha` over the pairs (`nan` if none).
    """
    if len(pairs) == 0:
        return float('nan')
    x = _normalize(embeddings)
    diff = x[pairs[:, 0]] - x[pairs[:, 1]]
    return float(np.mean(np.linalg.norm(diff, axis=1) ** alpha))


def anisotropy(embeddings: np.ndarray, max_samples: int = 2000) -> float:
    """Mean cosine similarity between random pairs (high = anisotropic/degenerate).

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
        max_samples (int): Subsample cap.

    Returns:
        float: Mean off-diagonal cosine similarity.
    """
    x = _normalize(_subsample(embeddings, max_samples))
    sims = x @ x.T
    iu = np.triu_indices(len(x), k=1)
    return float(np.mean(sims[iu])) if iu[0].size else float('nan')


def embedding_health(embeddings: np.ndarray, pairs: np.ndarray | None = None) -> dict[str, float]:
    """Bundles the label-free geometry metrics into one report.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
        pairs (np.ndarray | None): Optional positive pairs for alignment.

    Returns:
        dict[str, float]: `effective_rank`, `effective_rank_ratio`, `uniformity`, `anisotropy`, `mean_norm`, `dead_dim_fraction` and
            `alignment` (if `pairs` are provided).

    """
    x = np.asarray(embeddings, dtype=np.float32)
    dim = x.shape[1]
    per_dim_std = x.std(axis=0)
    eff = effective_rank(x)
    report = {
        'n': int(len(x)),
        'embed_dim': int(dim),
        'effective_rank': eff,
        'effective_rank_ratio': float(eff / dim) if dim else float('nan'),
        'uniformity': uniformity(x),
        'anisotropy': anisotropy(x),
        'mean_norm': float(np.linalg.norm(x, axis=1).mean()),
        'dead_dim_fraction': float(np.mean(per_dim_std < 1e-4)),
    }
    if pairs is not None:
        report['alignment'] = alignment(x, pairs)
    return report


# --------------------------------------------------------------------------- #
# Transfer probes
# --------------------------------------------------------------------------- #


def knn_probe(
    embeddings: np.ndarray,
    targets: np.ndarray,
    task: ProbeTask,
    k: int = 10,
    n_splits: int = 3,
    seed: int = 0,
) -> dict[str, float | list[float]]:
    """Cross-validated kNN (cosine) probe of an embedding's predictiveness.

    Folds are shuffled with a fixed seed (`KFold`/`StratifiedKFold`) so the split is
    reproducible and independent of row order; the cosine neighbour metric is kept.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
        targets (np.ndarray): Target labels `(n_samples,)`.
        task (ProbeTask): `classification` (accuracy) or `regression` (R^2).
        k (int): Neighbour count.
        n_splits (int): Cross-validation folds.
        seed (int): Seed for the shuffled fold splitter.

    Returns:
        dict[str, float | list[float]]: `score`, chance/mean `baseline`, the per-fold
            `scores` list and `score_std`.

    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    targets = np.asarray(targets)
    nan_out: dict[str, float | list[float]] = {
        'score': float('nan'),
        'baseline': float('nan'),
        'scores': [],
        'score_std': float('nan'),
    }
    if len(embeddings) < n_splits * 2:
        return nan_out
    try:
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    except ImportError:  # pragma: no cover
        return nan_out

    k = min(k, len(embeddings) - 1)
    if task == 'classification':
        if len(np.unique(targets)) < 2:
            return {**nan_out, 'baseline': 1.0}
        min_class = int(np.unique(targets, return_counts=True)[1].min())
        n_eff = min(n_splits, min_class)
        if n_eff < 2:
            return {**nan_out, 'baseline': 1.0}
        splitter: Any = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=seed)
        model: Any = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        scores = cross_val_score(model, embeddings, targets, cv=splitter, scoring='accuracy')
        baseline = float(max(np.mean(targets == c) for c in np.unique(targets)))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        model = KNeighborsRegressor(n_neighbors=k, metric='cosine')
        scores = cross_val_score(model, embeddings, targets, cv=splitter, scoring='r2')
        baseline = 0.0
    return {
        'score': float(np.mean(scores)),
        'baseline': baseline,
        'scores': [float(s) for s in scores],
        'score_std': float(np.std(scores)),
    }


def representation_comparison(
    representations: dict[str, np.ndarray],
    targets: dict[str, tuple[np.ndarray, ProbeTask]],
    knn_k: int = 10,
) -> list[dict[str, Any]]:
    """Probes several representations against several targets (tidy long table).

    For every (representation, target) pair this runs a linear probe and a kNN probe, so you can see that ZTE embeddings
    carry the attribute, ideally beating a noise-matched control and rivalling the raw features in far fewer dims.

    Args:
        representations (dict[str, np.ndarray]): Name -> `(n_samples, n_features)`
            matrix. All must share the same `n_samples` and row order (same tokens).
        targets (dict[str, tuple[np.ndarray, ProbeTask]]): Target name -> (values, task).
        knn_k (int): Neighbour count for the kNN probe.

    Returns:
        list[dict[str, Any]]: One row per (target, representation) with
        `linear_score`, `knn_score`, `baseline`, `metric` and `linear_scores` (the
        per-fold linear-probe scores, used by the bootstrap effect-size verdict).
    """
    rows: list[dict[str, Any]] = []
    for target_name, (values, task) in targets.items():
        metric = 'accuracy' if task == 'classification' else 'R2'
        for rep_name, matrix in representations.items():
            lin = linear_probe(matrix, values, task=task)
            knn = knn_probe(matrix, values, task=task, k=knn_k)
            rows.append(
                {
                    'target': target_name,
                    'representation': rep_name,
                    'metric': metric,
                    'linear_score': round(float(lin['score']), 4),
                    'knn_score': round(float(knn['score']), 4),
                    'baseline': round(float(lin['baseline']), 4),
                    'linear_scores': [float(s) for s in lin.get('scores', [])],  # type: ignore[union-attr]
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Content retrieval
# --------------------------------------------------------------------------- #


def content_retrieval(
    embeddings: np.ndarray,
    group_ids: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
    return_hits: bool = False,
) -> dict[str, float]:
    """Leave-one-out retrieval: do same-content items retrieve each other?

    Each item queries the bank (excluding itself); a hit is a neighbour sharing its `group_id` (e.g. the
    same stimulus sentence read by another subject).  Reported against a random-chance Top-1 baseline.

    The Top-1 chance baseline is **query-weighted**: a query in a group of size `g`
    has probability `(g - 1) / (n - 1)` of drawing a same-group neighbour uniformly at
    random, so averaging over *queries* (each group contributes `g` of them) gives
    `sum_g g*(g - 1) / (sum_g g) / (n - 1)`. This matches the hit rate, which is also a
    per-query (per-occurrence) mean -- unlike the older type-weighted average over
    distinct groups, which under-counts large groups and inflates the "x chance" lift.
    The legacy value is retained as `chance_top1_typeweighted`.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
        group_ids (np.ndarray): Integer content/group id per row `(n_samples,)`.
        ks (tuple[int, ...]): Top-K cut-offs.
        return_hits (bool): When `True`, also return `top1_hits`, the per-query Top-1
            hit vector (0/1 floats) used for bootstrap confidence intervals.

    Returns:
        dict[str, float]: `top{k}` for each `k`, `mrr`, `n_queries`, `chance_top1`
            (query-weighted) and `chance_top1_typeweighted`.

    """
    group_ids = np.asarray(group_ids)
    metadata = _ids_frame(group_ids)
    index = NearestNeighborIndex(embeddings, metadata)
    n = len(embeddings)

    # Only items whose group has another member can possibly hit.
    _, counts = np.unique(group_ids, return_counts=True)
    count_by_id = dict(zip(*np.unique(group_ids, return_counts=True), strict=True))
    valid = np.array([count_by_id[g] > 1 for g in group_ids])
    if not valid.any():
        empty = {f'top{k}': float('nan') for k in ks} | {'mrr': float('nan'), 'n_queries': 0.0}
        return {**empty, 'top1_hits': []} if return_hits else empty  # type: ignore[dict-item]

    max_k = min(max(ks), n - 1)
    idx, _ = index.query(embeddings[valid], k=max_k, self_indices=np.where(valid)[0])
    neighbour_groups = group_ids[idx]  # (n_queries, max_k)
    query_groups = group_ids[valid][:, None]
    hits = neighbour_groups == query_groups  # (n_queries, max_k)

    out: dict[str, float] = {}
    for k in ks:
        out[f'top{k}'] = float(np.mean(hits[:, :k].any(axis=1)))
    first_hit = np.argmax(hits, axis=1)
    has_hit = hits.any(axis=1)
    ranks = np.where(has_hit, 1.0 / (first_hit + 1), 0.0)
    out['mrr'] = float(np.mean(ranks))
    out['n_queries'] = float(valid.sum())
    # Query-weighted chance (matches the per-occurrence hit rate); large groups
    # contribute proportionally to the queries they generate.
    multi = counts[counts > 1]
    numer = float(np.sum(multi * (multi - 1)))
    denom = float(np.sum(multi)) * (n - 1)
    out['chance_top1'] = numer / denom if denom > 0 else float('nan')
    # Legacy type-weighted average over distinct groups (kept for comparison).
    out['chance_top1_typeweighted'] = float(np.mean((counts - 1) / (n - 1)))
    if return_hits:
        out['top1_hits'] = hits[:, 0].astype(float).tolist()  # type: ignore[assignment]
    return out


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #


def bootstrap_ci(
    values: np.ndarray,
    statistic: Any = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for a statistic of `values`.

    Resamples `values` with replacement `n_boot` times, applies `statistic` to each
    resample, and returns the point estimate together with the central
    `(1 - alpha)` percentile interval. Used to replace sign-only ("beat the baseline
    by an epsilon") verdicts with an effect-size interval that honestly reflects the
    sampling noise of small evaluation sets.

    Args:
        values (np.ndarray): 1-D sample (e.g. per-query hits or per-fold score diffs).
        statistic (Any): Callable reducing a 1-D array to a scalar (default mean).
        n_boot (int): Number of bootstrap resamples.
        alpha (float): Two-sided miscoverage (0.05 -> a 95% interval).
        seed (int): RNG seed for reproducibility.

    Returns:
        tuple[float, float, float]: `(point, lo, hi)`. All `nan` when `values` is empty.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return (float('nan'), float('nan'), float('nan'))
    point = float(statistic(values))
    if values.size == 1:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = np.array([statistic(values[row]) for row in idx], dtype=np.float64)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return (point, lo, hi)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalises rows of `x`."""
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _subsample(x: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Randomly subsamples at most `n` rows of `x` for O(n^2) metrics."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(len(x), size=n, replace=False)]


def _ids_frame(group_ids: np.ndarray) -> Any:
    """Wraps group ids in a one-column DataFrame for :class:`NearestNeighborIndex`."""
    import pandas as pd

    return pd.DataFrame({'group_id': group_ids})
