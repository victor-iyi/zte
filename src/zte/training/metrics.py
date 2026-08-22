"""Leakage-resistant probes for self-supervised embeddings, plus the matched-noise control."""

from __future__ import annotations

import warnings

import numpy as np

# Ridge penalties the inner search picks from. A fixed alpha=1.0 is effectively unregularised on a wide, correlated
# design, so a target with no signal scores -p/n rather than 0 -- which reads as "the representation is worse than
# useless" when the truth is "the probe cannot fit here". Searching the grid puts the no-signal floor back at 0.
_RIDGE_ALPHAS: tuple[float, ...] = tuple(float(a) for a in np.logspace(-2.0, 6.0, 17))


def residualise(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Removes each group's own mean from `values`, leaving only within-group variation.

    Note:
        On ZuCo, subject identity is linearly readable from raw band power at 0.81 accuracy while word length is not
        readable at all. A ridge fitted across subjects therefore spends its capacity on who is reading, and a weak
        lexical effect never surfaces. Centring per subject asks the question that was meant: within one person's
        recordings, does the signal move with the word?

    Args:
        values (np.ndarray): `(n,)` or `(n, d)` values.
        groups (np.ndarray): `(n,)` group label per row, typically the subject code.

    Returns:
        np.ndarray: The same shape, with every group centred on zero.
    """
    x = np.asarray(values, dtype=np.float64)
    keys = np.asarray(groups)
    out = x.copy()
    for key in np.unique(keys):
        rows = keys == key
        out[rows] -= out[rows].mean(axis=0, keepdims=x.ndim > 1)
    return out


def linear_probe(
    embeddings: np.ndarray,
    targets: np.ndarray,
    task: str = 'auto',
    n_splits: int = 3,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> dict[str, float | str | list[float]]:
    """Cross-validated linear probe of an embedding's information content.

    The estimator is a `StandardScaler` -> `RidgeCV`/`LogisticRegression` pipeline, so every fold is scaled and
    penalised from its own training rows only. The same `seed` and `n_splits` give the same fold assignment for any
    matrix of equal length, which pairs per-fold scores from two representations of one target for bootstrap
    effect-size CIs.

    Note:
        The ridge penalty is searched rather than fixed. At a fixed `alpha=1.0` a standardised design of `p` features
        is barely regularised, so a target the representation genuinely does not carry returns an out-of-sample
        `R2` of about `-p/n` -- the -0.005 that made ZuCo's raw band-power positive control look broken on 108k rows
        of 525 features. The searched version returns 0 there instead, so a negative score now means the probe
        actively mis-predicts rather than merely overfits.

    Args:
        embeddings (np.ndarray): Array `(n_samples, embed_dim)` of frozen embeddings.
        targets (np.ndarray): Array `(n_samples,)` of attributes to predict.
        task (str): `regression`, `classification` or `auto` (decided from the number of unique targets).
        n_splits (int): Cross-validation folds.
        seed (int): Seed for the shuffled fold splitter.
        groups (np.ndarray | None): Group label per row, e.g. the subject. When given, folds are grouped so no
            group spans the train/test boundary and the score is a generalisation number rather than an
            interpolation one.

    Returns:
        dict[str, float | str | list[float]]: A dict with `score` (mean R^2 for
            regression, mean accuracy for classification), `baseline` (predict-the-mean
            / majority), `task`, `scores` (the per-fold score list), `score_std` and `alpha`
            (the mean penalty the inner search chose, `nan` for classification).
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
        'alpha': float('nan'),
    }

    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression, RidgeCV
        from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_score, cross_validate
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

    alpha = float('nan')
    with warnings.catch_warnings():
        # Noise controls and collapsed embeddings give an ill-conditioned Gram matrix, warning per fold.
        warnings.simplefilter('ignore', LinAlgWarning)
        warnings.simplefilter('ignore', ConvergenceWarning)
        if task == 'classification':
            # Stratified folds need at least `n_splits` members in the rarest class.
            min_class = int(np.unique(targets, return_counts=True)[1].min())
            n_eff = min(n_splits, min_class)
            if n_eff < 2:
                return {**nan_out, 'baseline': 1.0}
            splitter: object = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=seed)
            model: object = Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))])
            scores = cross_val_score(model, embeddings, targets, cv=splitter, scoring='accuracy', groups=groups)
            baseline = float(max(np.mean(targets == c) for c in np.unique(targets)))
        else:
            n_eff = min(n_splits, len(np.unique(groups))) if groups is not None else n_splits
            if n_eff < 2:
                return nan_out
            splitter = (
                GroupKFold(n_splits=n_eff)
                if groups is not None
                else KFold(n_splits=n_eff, shuffle=True, random_state=seed)
            )
            model = Pipeline([('scale', StandardScaler()), ('ridge', RidgeCV(alphas=_RIDGE_ALPHAS))])
            fitted = cross_validate(
                model, embeddings, targets, cv=splitter, scoring='r2', groups=groups, return_estimator=True
            )
            scores = fitted['test_score']
            alpha = float(np.mean([float(e.named_steps['ridge'].alpha_) for e in fitted['estimator']]))
            baseline = 0.0
    return {
        'score': float(np.mean(scores)),
        'baseline': baseline,
        'task': task,
        'scores': [float(s) for s in scores],
        'score_std': float(np.std(scores)),
        'alpha': alpha,
    }


def retrieval_metrics(query: np.ndarray, key: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    """Computes Top-K accuracy and MRR for paired query/key embeddings.

    Row `i` of `query` is assumed to match row `i` of `key`. Similarity is cosine; the diagonal rank determines the
    metrics.

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

    This is the empirical-floor control: a genuine encoder must beat embeddings learned from this noise to claim it
    decodes real neural structure. Computed in row blocks and entirely in float32, since the naive whole-array form
    MemoryErrors on a multi-GB raw baseline; moments accumulate in `(n_features,)` float64 registers to stay exact.

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

    # Pass 1: nan-aware per-feature mean.
    count = np.zeros(d, dtype=np.int64)
    total = np.zeros(d, dtype=np.float64)
    for start in range(0, n, chunk):
        block = np.asarray(src[start : start + chunk], dtype=np.float32).reshape(-1, d)
        valid = ~np.isnan(block)
        count += valid.sum(axis=0)
        total += np.where(valid, block, np.float32(0.0)).sum(axis=0, dtype=np.float64)
    safe = np.maximum(count, 1)
    mean = (total / safe).astype(np.float32)

    # Pass 2: nan-aware variance about that mean.
    resid = np.zeros(d, dtype=np.float64)
    for start in range(0, n, chunk):
        block = np.asarray(src[start : start + chunk], dtype=np.float32).reshape(-1, d)
        valid = ~np.isnan(block)
        delta = np.where(valid, block - mean, np.float32(0.0))
        resid += np.square(delta).sum(axis=0, dtype=np.float64)
    std = np.sqrt(resid / safe).astype(np.float32)

    # Draw in float32 -- the default float64 draw doubles peak memory.
    out = np.empty((n, d), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = rng.standard_normal((stop - start, d), dtype=np.float32)
        block *= std
        block += mean
        out[start:stop] = block
    return out.reshape(src.shape)


def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalises the rows of `x`."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)
