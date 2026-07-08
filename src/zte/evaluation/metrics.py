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
) -> dict[str, float]:
    """Cross-validated kNN probe of an embedding's predictiveness for a target.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
        targets (np.ndarray): Target labels `(n_samples,)`.
        task (ProbeTask): `classification` (accuracy) or `regression` (R^2).
        k (int): Neighbour count.
        n_splits (int): Cross-validation folds.

    Returns:
        dict[str, float]: `score` and chance/mean `baseline`.

    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    targets = np.asarray(targets)
    if len(embeddings) < n_splits * 2:
        return {'score': float('nan'), 'baseline': float('nan')}
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    except ImportError:  # pragma: no cover
        return {'score': float('nan'), 'baseline': float('nan')}

    k = min(k, len(embeddings) - 1)
    if task == 'classification':
        if len(np.unique(targets)) < 2:
            return {'score': float('nan'), 'baseline': 1.0}
        model: Any = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        scores = cross_val_score(model, embeddings, targets, cv=n_splits, scoring='accuracy')
        baseline = float(max(np.mean(targets == c) for c in np.unique(targets)))
    else:
        model = KNeighborsRegressor(n_neighbors=k, metric='cosine')
        scores = cross_val_score(model, embeddings, targets, cv=n_splits, scoring='r2')
        baseline = 0.0
    return {'score': float(np.mean(scores)), 'baseline': baseline}


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
        `linear_score`, `knn_score`, `baseline` and `metric`.
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
) -> dict[str, float]:
    """Leave-one-out retrieval: do same-content items retrieve each other?

    Each item queries the bank (excluding itself); a hit is a neighbour sharing its `group_id` (e.g. the
    same stimulus sentence read by another subject).  Reported against a random-chance Top-1 baseline.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
        group_ids (np.ndarray): Integer content/group id per row `(n_samples,)`.
        ks (tuple[int, ...]): Top-K cut-offs.

    Returns:
        dict[str, float]: `top{k}` for each `k`, `mrr`, `n_queries` and `chance_top1`.

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
        return {f'top{k}': float('nan') for k in ks} | {'mrr': float('nan'), 'n_queries': 0.0}

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
    out['chance_top1'] = float(np.mean((counts - 1) / (n - 1)))
    return out


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
