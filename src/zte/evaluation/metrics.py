"""Decoder-free evaluation of ZTE embeddings: label-free geometry, transfer probes and content retrieval."""

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


def whiten_features(
    emb: np.ndarray, fit_on: np.ndarray | None = None, eps: float = 1e-3, shrink: float = 0.0
) -> np.ndarray:
    """ZCA-whitens embeddings: centre, then decorrelate + equalise variance across dimensions.

    A label-free fix for the "cone" (anisotropy near 1) and dimensional collapse. ZCA rather than PCA
    keeps the axes aligned with the originals, so the transform is a minimal, content-preserving
    normalisation of the geometry.

    Args:
        emb (np.ndarray): Embeddings to transform `(n, d)`.
        fit_on (np.ndarray | None): Optional separate set to estimate the mean/covariance from (e.g. the
            training split, to avoid fitting the whitening on the same data it is scored on). Defaults to
            `emb` itself.
        eps (float): Floor added to eigenvalues for numerical stability.
        shrink (float): Optional shrinkage in `[0, 1]` blending the covariance toward the identity
            (Ledoit-Wolf style), for stability when `n` is not much larger than `d`.

    Returns:
        np.ndarray: The whitened embeddings `(n, d)` (float32).
    """
    x = np.asarray(emb, dtype=np.float64)
    base = x if fit_on is None else np.asarray(fit_on, dtype=np.float64)
    if len(base) < 2:
        return x.astype(np.float32)
    mean = base.mean(axis=0, keepdims=True)
    bc = base - mean
    cov = (bc.T @ bc) / (len(bc) - 1)
    if shrink > 0.0:
        d = cov.shape[0]
        cov = (1.0 - shrink) * cov + shrink * np.trace(cov) / d * np.eye(d)
    vals, vecs = np.linalg.eigh(cov)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.clip(vals, eps, None))) @ vecs.T
    return ((x - mean) @ inv_sqrt).astype(np.float32)


def all_but_the_top(emb: np.ndarray, n_components: int) -> np.ndarray:
    """Removes the top-`n_components` principal directions from embeddings (all-but-the-top).

    Label-free anti-hubness post-processing: the dominant shared PCA directions carry the "hub" axis
    that makes a few points everyone's nearest neighbour, a common cause of below-chance retrieval on
    an otherwise healthy space. Complementary to `whiten_features`; apply whiten then ABTT.

    Args:
        emb (np.ndarray): Embeddings `(n, d)`.
        n_components (int): Number of leading directions to remove (`<= 0` is a no-op).

    Returns:
        np.ndarray: Post-processed embeddings `(n, d)` (float32).
    """
    x = np.asarray(emb, dtype=np.float64)
    if n_components <= 0 or len(x) < 2:
        return x.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    u = vt[: min(n_components, vt.shape[0])]  # (n_components, d) leading directions
    return (x - (x @ u.T) @ u).astype(np.float32)


def effective_rank(embeddings: np.ndarray) -> float:
    """Computes the effective rank of an embedding matrix.

    The effective rank is `exp(H)` where `H` is the Shannon entropy of the normalised singular-value
    spectrum: near 1 means collapse onto a single direction, near `embed_dim` a well-spread space.

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
    """Computes the uniformity metric.

    Lower (more negative) means embeddings are spread more uniformly on the unit hypersphere -- a sign
    the contrastive objective avoided collapse.

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
    """Computes the alignment metric over positive pairs.

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
        dict[str, float]: `effective_rank`, `effective_rank_ratio`, `uniformity`, `anisotropy`,
            `mean_norm`, `dead_dim_fraction` and `alignment` (if `pairs` are provided).
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
    max_n: int = 20000,
) -> dict[str, float | list[float]]:
    """Cross-validated kNN (cosine geometry) probe of an embedding's predictiveness.

    Folds are shuffled with a fixed seed so the split is reproducible and independent of row order.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
        targets (np.ndarray): Target labels `(n_samples,)`.
        task (ProbeTask): `classification` (accuracy) or `regression` (R^2).
        k (int): Neighbour count.
        n_splits (int): Cross-validation folds.
        seed (int): Seed for the shuffled fold splitter (and the subsample).
        max_n (int): Row cap; above it a seeded subsample is drawn (stratified for classification).
            kNN is quadratic, so the full ZuCo word set costs ~30 min per probe block for a point
            estimate the cap already resolves to ~0.003.

    Returns:
        dict[str, float | list[float]]: `score`, chance/mean `baseline`, the per-fold `scores` list,
            `score_std` and the `n_used` row count the score was computed on.
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

    # kNN is quadratic in n, so cap the rows; classification is stratified to keep the rarest class.
    if len(embeddings) > max_n:
        rng = np.random.default_rng(seed)
        if task == 'classification':
            keep: list[np.ndarray] = []
            share = max_n / len(embeddings)
            for cls in np.unique(targets):
                rows = np.flatnonzero(targets == cls)
                take = max(1, int(round(len(rows) * share)))
                keep.append(rng.choice(rows, size=min(take, len(rows)), replace=False))
            idx = np.sort(np.concatenate(keep))
        else:
            idx = np.sort(rng.choice(len(embeddings), size=max_n, replace=False))
        embeddings, targets = embeddings[idx], targets[idx]

    # L2-normalised rows make squared euclidean rank-equivalent to cosine, on sklearn's fast ArgKmin path.
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)

    k = min(k, len(embeddings) - 1)
    if task == 'classification':
        if len(np.unique(targets)) < 2:
            return {**nan_out, 'baseline': 1.0}
        min_class = int(np.unique(targets, return_counts=True)[1].min())
        n_eff = min(n_splits, min_class)
        if n_eff < 2:
            return {**nan_out, 'baseline': 1.0}
        splitter: Any = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=seed)
        model: Any = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        scores = cross_val_score(model, embeddings, targets, cv=splitter, scoring='accuracy')
        baseline = float(max(np.mean(targets == c) for c in np.unique(targets)))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        model = KNeighborsRegressor(n_neighbors=k, metric='euclidean')
        scores = cross_val_score(model, embeddings, targets, cv=splitter, scoring='r2')
        baseline = 0.0
    return {
        'score': float(np.mean(scores)),
        'baseline': baseline,
        'scores': [float(s) for s in scores],
        'score_std': float(np.std(scores)),
        # A kNN score rises with gallery density, so scores compare across runs only at equal `n_used`.
        'n_used': float(len(embeddings)),
    }


def representation_comparison(
    representations: dict[str, np.ndarray],
    targets: dict[str, tuple[np.ndarray, ProbeTask]],
    knn_k: int = 10,
) -> list[dict[str, Any]]:
    """Probes several representations against several targets (tidy long table).

    Each (representation, target) pair gets a linear probe and a kNN probe, so ZTE can be read against
    a noise-matched control and against the raw features it should rival in far fewer dims.

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
    *,
    csls: bool = False,
    csls_k: int = 10,
    return_ranks: bool = False,
) -> dict[str, float]:
    """Leave-one-out retrieval: do same-content items retrieve each other?

    Each item queries the bank (excluding itself); a hit is a neighbour sharing its `group_id` (e.g. the
    same stimulus sentence read by another subject).

    The Top-1 chance baseline is query-weighted -- `sum_g g*(g - 1) / (sum_g g) / (n - 1)` -- to match
    the hit rate, which is also a per-occurrence mean. A type-weighted average over distinct groups
    under-counts large groups and inflates the "x chance" lift; it is kept as
    `chance_top1_typeweighted` for comparison.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
        group_ids (np.ndarray): Integer content/group id per row `(n_samples,)`.
        ks (tuple[int, ...]): Top-K cut-offs.
        return_hits (bool): When `True`, also return `top1_hits`, the per-query Top-1
            hit vector (0/1 floats) used for bootstrap confidence intervals.
        csls (bool): Apply CSLS hubness correction inside the nearest-neighbour index;
            default `False` keeps plain cosine.
        csls_k (int): CSLS neighbourhood size.
        return_ranks (bool): When `True`, also return the whole-distribution rank summary
            (`median_rank`, `rank_percentile`, `ranks`) that a Top-1 scalar hides. Computed on the
            plain-cosine geometry over a subsampled set of queries; see `_group_ranks`.

    Returns:
        dict[str, float]: `top{k}` for each `k`, `mrr`, `n_queries`, `chance_top1`
            (query-weighted) and `chance_top1_typeweighted`.
    """
    group_ids = np.asarray(group_ids)
    metadata = _ids_frame(group_ids)
    index = NearestNeighborIndex(embeddings, metadata, csls=csls, csls_k=csls_k)
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

    # Query-weighted chance: large groups contribute proportionally to the queries they generate.
    multi = counts[counts > 1]
    numer = float(np.sum(multi * (multi - 1)))
    denom = float(np.sum(multi)) * (n - 1)
    out['chance_top1'] = numer / denom if denom > 0 else float('nan')
    out['chance_top1_typeweighted'] = float(np.mean((counts - 1) / (n - 1)))

    if return_ranks:
        exact = _group_ranks(index.bank, group_ids, valid)
        if exact.size:
            out['median_rank'] = float(np.median(exact))
            out['mean_rank'] = float(np.mean(exact))
            # 1.0 = correct match ranked first; unlike Top-1 it degrades gracefully with gallery size.
            out['rank_percentile'] = float(np.mean(1.0 - (exact - 1.0) / max(n - 1, 1)))
            out['ranks'] = exact.tolist()  # type: ignore[assignment]
    if return_hits:
        out['top1_hits'] = hits[:, 0].astype(float).tolist()  # type: ignore[assignment]
    return out


def _group_ranks(
    bank: np.ndarray,
    group_ids: np.ndarray,
    valid: np.ndarray,
    *,
    max_queries: int = 4000,
    seed: int = 0,
) -> np.ndarray:
    """Exact 1-based rank of each query's best same-group neighbour over the full gallery.

    A query's rank is `1 + #{items strictly more similar than its best same-group item}`, computed on
    plain cosine and block-tiled so the full `(n x n)` similarity is never materialised.

    Args:
        bank (np.ndarray): L2-normalised embeddings `(n, d)` (the index's `bank`).
        group_ids (np.ndarray): Content/group id per row `(n,)`.
        valid (np.ndarray): Boolean `(n,)` marking queries whose group has >= 2 members.
        max_queries (int): Cap on the number of queries scored (random subsample above it).
        seed (int): Subsample seed.

    Returns:
        np.ndarray: 1-based ranks `(n_scored,)` (empty when no valid query).
    """
    n = len(bank)
    q_idx = np.where(valid)[0]
    if q_idx.size == 0:
        return np.empty(0, dtype=np.float64)
    if q_idx.size > max_queries:
        q_idx = np.sort(np.random.default_rng(seed).choice(q_idx, size=max_queries, replace=False))
    block = max(1, min(2048, (128 * 1024 * 1024) // (n * 4)))
    ranks = np.empty(q_idx.size, dtype=np.float64)
    for start in range(0, q_idx.size, block):
        rows = q_idx[start : start + block]
        sims = bank[rows] @ bank.T  # (b, n)
        br = np.arange(len(rows))
        sims[br, rows] = -np.inf  # exclude self
        same = group_ids[None, :] == group_ids[rows][:, None]
        same[br, rows] = False
        best_same = np.where(same, sims, -np.inf).max(axis=1)  # (b,)
        ranks[start : start + block] = 1.0 + (sims > best_same[:, None]).sum(axis=1)
    return ranks


def matched_content_retrieval(
    embeddings: np.ndarray,
    group_ids: np.ndarray,
    bins: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
    *,
    csls: bool = False,
    csls_k: int = 10,
) -> dict[str, float]:
    """Content retrieval with each query's distractor bank restricted to its own nuisance bin.

    Retrieval runs independently within every value of `bins`, so a hit can no longer be a rare word
    standing out among common ones. Per-bin results are pooled query-weighted, and chance is recomputed
    within each bin (a smaller gallery, so a higher chance) -- never against the global chance.

    Args:
        embeddings (np.ndarray): Array `(n, d)`.
        group_ids (np.ndarray): Content/group id per row `(n,)`.
        bins (np.ndarray): Nuisance-bin id per row `(n,)`; the bank is restricted to the same bin.
        ks (tuple[int, ...]): Top-K cut-offs.
        csls (bool): Apply CSLS correction within each bin.
        csls_k (int): CSLS neighbourhood size.

    Returns:
        dict[str, float]: Query-weighted `top{k}`, `mrr`, `n_queries`, `chance_top1`, `n_bins`.
    """
    bins = np.asarray(bins)
    finite = ~pd_isna(bins)
    keys = (*(f'top{k}' for k in ks), 'mrr', 'chance_top1')
    acc: dict[str, float] = {key: 0.0 for key in keys}
    total = 0.0
    n_bins = 0
    for value in np.unique(bins[finite]):
        mask = (bins == value) & finite
        if int(mask.sum()) < 4:
            continue
        res = content_retrieval(embeddings[mask], group_ids[mask], ks=ks, csls=csls, csls_k=csls_k)
        nq = float(res.get('n_queries', 0.0))
        if not nq:
            continue
        for key in keys:
            acc[key] += float(res.get(key, 0.0) or 0.0) * nq
        total += nq
        n_bins += 1
    if total <= 0:
        return {key: float('nan') for key in keys} | {'n_queries': 0.0, 'n_bins': 0.0}
    out = {key: acc[key] / total for key in keys}
    out['n_queries'] = total
    out['n_bins'] = float(n_bins)
    return out


def pd_isna(values: np.ndarray) -> np.ndarray:
    """NaN mask that works for float bins and is `False` for non-float (categorical) bins."""
    values = np.asarray(values)
    return np.isnan(values) if values.dtype.kind == 'f' else np.zeros(len(values), dtype=bool)


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

    Returns the point estimate with the central `(1 - alpha)` percentile interval, so verdicts rest on
    an effect size rather than a sign test that any positive fluctuation would pass.

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

    # Each resample is drawn as it is consumed; a whole (n_boot, n) index would cost tens of GB.
    n = values.size
    boot = np.array(
        [statistic(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)], dtype=np.float64
    )
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
    """Wraps group ids in a one-column DataFrame for `NearestNeighborIndex`."""
    import pandas as pd

    return pd.DataFrame({'group_id': group_ids})
