"""Evaluation metrics for self-supervised embeddings.

Two complementary, leakage-resistant probes are provided:

- `linear_probe` -- freezes the embeddings and fits a cheap linear model to predict a held-out attribute (word length, frequency, omission).  A score
well above chance means the embedding captured linguistic structure *without* the probe being able to memorise, since it is linear and cross-validated.
- `retrieval_metrics` -- Top-K / MRR for matching one set of embeddings to another (used downstream for EEG<->text retrieval; here for neighbour probes).

`noise_matched` builds the Gaussian control the project mandates: inputs with the genuine data's per-feature mean/variance but no real structure.
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import warnings

import numpy as np


def linear_probe(
    embeddings: np.ndarray,
    targets: np.ndarray,
    task: str = 'auto',
    n_splits: int = 3,
    seed: int = 0,
) -> dict[str, float | str | list[float]]:
    """Cross-validated linear probe of an embedding's information content.

    The estimator is a `StandardScaler` -> `Ridge`/`LogisticRegression` pipeline so
    every fold is scaled from *its own training rows only* (no test leakage), and the
    folds are shuffled with a fixed seed (`KFold`/`StratifiedKFold`) so the split is
    reproducible and independent of row order. The same `seed`+`n_splits` yield the
    *same* fold assignment for any matrix of equal length, so per-fold scores from two
    representations of one target are paired (used for bootstrap effect-size CIs).

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)` of frozen embeddings.
        targets (np.ndarray): Array `(n_samples,)` of attributes to predict.
        task (str): `regression`, `classification` or `auto` (decided from the number of unique targets).
        n_splits (int): Cross-validation folds.
        seed (int): Seed for the shuffled fold splitter.

    Returns:
        dict[str, float | str | list[float]]: A dict with `score` (mean R^2 for
            regression, mean accuracy for classification), `baseline` (predict-the-mean
            / majority), `task`, `scores` (the per-fold score list) and `score_std`.

    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    targets = np.asarray(targets)
    if task == 'auto':
        task = 'classification' if len(np.unique(targets)) <= 2 else 'regression'
    nan_out: dict[str, float | str | list[float]] = {
        'score': float('nan'),
        'baseline': float('nan'),
        'task': task,
        'scores': [],
        'score_std': float('nan'),
    }

    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:  # pragma: no cover
        return nan_out
    try:
        from scipy.linalg import LinAlgWarning
    except ImportError:  # pragma: no cover
        LinAlgWarning = RuntimeWarning  # type: ignore[assignment,misc]

    if len(embeddings) < n_splits * 2:
        return nan_out
    if task == 'classification' and len(np.unique(targets)) < 2:
        # Degenerate (single-class) target -- nothing to discriminate.
        return {**nan_out, 'baseline': 1.0}

    with warnings.catch_warnings():
        # Raw band-power / noise controls (and any collapsed embedding) yield an
        # ill-conditioned Gram matrix, so Ridge/LogisticRegression warn once per
        # fold. That is expected here and separately quantified by embedding_health's
        # effective-rank ratio -- suppress the per-fold spam rather than flood stderr.
        warnings.simplefilter('ignore', LinAlgWarning)
        warnings.simplefilter('ignore', ConvergenceWarning)
        if task == 'classification':
            # Stratified folds need at least `n_splits` members in the rarest class.
            min_class = int(np.unique(targets, return_counts=True)[1].min())
            n_eff = min(n_splits, min_class)
            if n_eff < 2:
                return {**nan_out, 'baseline': 1.0}
            splitter = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=seed)
            model: object = Pipeline(
                [('scale', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))]
            )
            scores = cross_val_score(model, embeddings, targets, cv=splitter, scoring='accuracy')
            baseline = float(max(np.mean(targets == c) for c in np.unique(targets)))
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            model = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
            scores = cross_val_score(model, embeddings, targets, cv=splitter, scoring='r2')
            baseline = 0.0
    return {
        'score': float(np.mean(scores)),
        'baseline': baseline,
        'task': task,
        'scores': [float(s) for s in scores],
        'score_std': float(np.std(scores)),
    }


def retrieval_metrics(
    query: np.ndarray, key: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """Computes Top-K accuracy and MRR for paired query/key embeddings.

    Row `i` of `query` is assumed to match row `i` of `key`. Similarity is cosine; the diagonal rank determines the metrics.

    Args:
        query (np.ndarray): Array `(n_samples, embed_dim)`.
        key (np.ndarray): Array `(n_samples, embed_dim)` aligned with `query`.
        ks (tuple[int, ...]): Cut-offs for Top-K accuracy.

    Returns:
        dict[str, float]: A dict with `top{k}` for each `k` and `mrr`.

    """
    q = _l2(np.asarray(query, dtype=np.float32))
    k = _l2(np.asarray(key, dtype=np.float32))
    sim = q @ k.T
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    ranks = np.empty(n, dtype=np.int64)
    for i in range(n):
        ranks[i] = int(np.where(order[i] == i)[0][0]) + 1
    out = {f'top{kk}': float(np.mean(ranks <= kk)) for kk in ks}
    out['mrr'] = float(np.mean(1.0 / ranks))
    return out


def noise_matched(x: np.ndarray, seed: int = 0, chunk: int = 2048) -> np.ndarray:
    """Returns Gaussian noise matched to `x`'s per-feature mean and variance.

    This is the empirical-floor control: a genuine encoder must beat embeddings learned from this noise to claim it decodes real neural structure.

    Computed in row blocks and entirely in float32. The obvious one-liner is not viable on a raw-signal
    matrix: `rng.standard_normal(x.shape)` draws in **float64** (twice the output), `np.nanmean`/`np.nanstd`
    each materialise a full-size mask *and* a full-size copy, and `* std + mean` adds more temporaries --
    together several times the input, which MemoryErrors on a multi-GB raw baseline. Moments accumulate in
    (n_features,)-sized float64 registers, so the reduction stays exact while peak memory is the output
    array plus one block.

    Args:
        x (np.ndarray): Real feature matrix `(n_samples, n_features)`; may be a read-only memmap.
        seed (int): RNG seed.
        chunk (int): Rows processed per block (bounds peak temporary memory).

    Returns:
        np.ndarray: A noise matrix of the same shape with matched first/second moments (float32).

    """
    rng = np.random.default_rng(seed)
    src = np.asarray(x)
    n, d = src.shape[0], int(np.prod(src.shape[1:]))
    count = np.zeros(d, dtype=np.int64)
    total = np.zeros(d, dtype=np.float64)
    for start in range(0, n, chunk):  # pass 1: nan-aware mean
        block = np.asarray(src[start : start + chunk], dtype=np.float32).reshape(-1, d)
        valid = ~np.isnan(block)
        count += valid.sum(axis=0)
        total += np.where(valid, block, np.float32(0.0)).sum(axis=0, dtype=np.float64)
    safe = np.maximum(count, 1)
    mean = (total / safe).astype(np.float32)
    resid = np.zeros(d, dtype=np.float64)
    for start in range(0, n, chunk):  # pass 2: nan-aware variance about that mean
        block = np.asarray(src[start : start + chunk], dtype=np.float32).reshape(-1, d)
        valid = ~np.isnan(block)
        delta = np.where(valid, block - mean, np.float32(0.0))
        resid += np.square(delta).sum(axis=0, dtype=np.float64)
    std = np.sqrt(resid / safe).astype(np.float32)
    out = np.empty((n, d), dtype=np.float32)
    for start in range(0, n, chunk):  # draw in float32 -- the default float64 draw doubles this
        stop = min(start + chunk, n)
        block = rng.standard_normal((stop - start, d), dtype=np.float32)
        block *= std
        block += mean
        out[start:stop] = block
    return out.reshape(src.shape)


def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalises rows of `x`.

    Args:
        x (np.ndarray): Array `(n_samples, embed_dim)`.
        eps (float): Numerical floor.

    Returns:
        np.ndarray: Row-normalised array.

    """
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)
