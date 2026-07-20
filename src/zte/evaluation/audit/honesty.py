"""Honesty add-ons: permutation nulls, held-out cross-subject decoding, and anchor calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _l2norm(x: np.ndarray) -> np.ndarray:
    """Row-normalises to unit L2 length (zero rows are left at zero)."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def retrieval_permutation_test(
    emb: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_perm: int = 500,
    seed: int = 0,
    max_n: int = 6000,
) -> dict[str, Any]:
    """Empirical permutation p-value for same-group top-1 nearest-neighbour retrieval.

    The observed statistic is the fraction of items whose nearest neighbour shares their group id.
    The null shuffles the group ids against the same fixed neighbour structure, so the geometry is
    held constant and only the labels move.

    Args:
        emb (np.ndarray): Embeddings `(n, d)` (e.g. sentence embeddings).
        group_ids (np.ndarray): Content group id per item (e.g. stimulus id shared across subjects).
        n_perm (int): Number of label permutations.
        seed (int): RNG seed.
        max_n (int): If `n > max_n` a random subsample is used (kept deterministic by `seed`).

    Returns:
        dict: `{'applicable', 'observed_top1', 'null_mean', 'null_std', 'p_value', 'n_perm', 'n_queries',
        'above_chance'}`. `p_value` is `(1 + #{null >= observed}) / (n_perm + 1)`.
    """
    emb = np.asarray(emb, dtype=np.float32)
    group_ids = np.asarray(group_ids)
    n = len(emb)
    if n < 4:
        return {'applicable': False, 'reason': 'need >= 4 items'}
    rng = np.random.default_rng(seed)

    # The pairwise similarity is quadratic in n, so cap the item count.
    if n > max_n:
        keep = np.sort(rng.choice(n, size=max_n, replace=False))
        emb, group_ids = emb[keep], group_ids[keep]
        n = max_n
    _, counts = np.unique(group_ids, return_counts=True)
    if counts.max() < 2:
        return {'applicable': False, 'reason': 'no group has >= 2 members'}

    # Neighbours are found once, so the permutation moves labels over fixed geometry.
    x = _l2norm(emb)
    sims = x @ x.T
    np.fill_diagonal(sims, -np.inf)
    nn = np.argmax(sims, axis=1)
    nn_group = group_ids[nn]
    observed = float(np.mean(nn_group == group_ids))

    # Empirical null: reshuffle the group ids and re-score against the same neighbours.
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(group_ids)
        null[i] = np.mean(perm[nn] == perm)

    p = (1.0 + int(np.sum(null >= observed))) / (n_perm + 1.0)

    return {
        'applicable': True,
        'observed_top1': observed,
        'null_mean': float(null.mean()),
        'null_std': float(null.std()),
        'p_value': float(p),
        'n_perm': int(n_perm),
        'n_queries': int(n),
        'above_chance': bool(p < 0.05),
    }


def _fit_score(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray, task: str
) -> float | None:
    """Fits a standardised linear probe on `(xtr, ytr)` and scores it on the held-out fold."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    try:
        from scipy.linalg import LinAlgWarning
    except ImportError:  # pragma: no cover
        LinAlgWarning = RuntimeWarning

    if len(xtr) < 8 or len(xte) < 2:
        return None
    scaler = StandardScaler().fit(xtr)
    xtr_s, xte_s = scaler.transform(xtr), scaler.transform(xte)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', LinAlgWarning)
        if task == 'regression':
            model: object = Ridge(alpha=1.0).fit(xtr_s, ytr)
            # R^2 on the held-out subject.
            pred = model.predict(xte_s)  # type: ignore[attr-defined]
            ss_res = float(np.sum((yte - pred) ** 2))
            ss_tot = float(np.sum((yte - yte.mean()) ** 2)) or 1.0
            return 1.0 - ss_res / ss_tot

        if len(np.unique(ytr)) < 2:
            return None

        model = LogisticRegression(max_iter=2000, C=1.0).fit(xtr_s, ytr)
        return float(model.score(xte_s, yte))  # type: ignore[attr-defined]


def cross_subject_decode(
    word_emb: np.ndarray,
    word_meta: pd.DataFrame,
    *,
    targets: tuple[str, ...] = ('category', 'length_band', 'word_len', 'log_freq'),
    seed: int = 0,
    min_subjects: int = 3,
    max_train: int = 6000,
) -> dict[str, Any]:
    """Leave-one-subject-out linear decoding of content from the frozen embeddings.

    For every subject in turn, a probe is trained on all other subjects and scored on that held-out
    subject, against an honest chance baseline (test-set majority class for classification, 0 for
    R^2). A target "generalises" only if the bootstrap CI lower bound clears chance.

    Args:
        word_emb (np.ndarray): Word embeddings `(n, d)`.
        word_meta (pd.DataFrame): Aligned metadata; must have `subject` and the target columns.
        targets (tuple[str, ...]): Columns to decode. `word_len` / `log_freq` are regressions.
        seed (int): RNG seed (train subsampling + bootstrap).
        min_subjects (int): Minimum distinct subjects required to run.
        max_train (int): Cap on training rows per fold (subsampled) for speed.

    Returns:
        dict: `{'applicable', 'n_subjects', 'targets': {name: {task, mean, ci, chance, n_folds,
        above_chance}}}` or `{'applicable': False, 'reason'}`.
    """
    from zte.evaluation.metrics import bootstrap_ci

    if 'subject' not in word_meta.columns:
        return {'applicable': False, 'reason': 'no subject column'}

    subjects = word_meta['subject'].astype(str).to_numpy()
    uniq = np.unique(subjects)
    if len(uniq) < min_subjects:
        return {'applicable': False, 'reason': f'need >= {min_subjects} subjects'}

    rng = np.random.default_rng(seed)
    out_targets: dict[str, Any] = {}
    for tgt in targets:
        if tgt not in word_meta.columns:
            continue

        # One leave-one-subject-out fold per subject.
        task = 'regression' if tgt in ('word_len', 'log_freq') else 'classification'
        y_all = (
            pd.to_numeric(word_meta[tgt], errors='coerce').to_numpy(dtype=float)
            if task == 'regression'
            else word_meta[tgt].astype(str).to_numpy()
        )
        scores: list[float] = []
        chances: list[float] = []
        for s in uniq:
            tr = subjects != s
            te = subjects == s
            xtr, ytr = word_emb[tr], y_all[tr]
            xte, yte = word_emb[te], y_all[te]
            if task == 'regression':
                valid = np.isfinite(ytr)
                xtr, ytr = xtr[valid], ytr[valid]
                vte = np.isfinite(yte)
                xte, yte = xte[vte], yte[vte]
            if len(xtr) > max_train:
                idx = rng.choice(len(xtr), size=max_train, replace=False)
                xtr, ytr = xtr[idx], ytr[idx]
            score = _fit_score(xtr, ytr, xte, yte, task)
            if score is None:
                continue
            scores.append(score)
            if task == 'classification' and len(yte):
                _, c = np.unique(yte, return_counts=True)
                chances.append(float(c.max() / len(yte)))
            else:
                chances.append(0.0)

        if len(scores) < 2:
            continue

        # Aggregate the folds into a point estimate with a bootstrap CI over chance.
        arr = np.asarray(scores, dtype=float)
        point, lo, hi = bootstrap_ci(arr, seed=seed)
        chance = float(np.mean(chances)) if chances else 0.0

        out_targets[tgt] = {
            'task': task,
            'mean': float(point),
            'ci': [float(point), float(lo), float(hi)],
            'chance': chance,
            'n_folds': len(scores),
            'above_chance': bool(lo > chance),
        }

    return {'applicable': True, 'n_subjects': int(len(uniq)), 'targets': out_targets}


def _procrustes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthogonal Procrustes with translation: returns `(R, mean_a, mean_b)` mapping a -> b.

    Solves `min_R ||(A - mean_a) R - (B - mean_b)||` over orthogonal `R`. A point `x` in A's frame is
    mapped to B's frame by `(x - mean_a) @ R + mean_b`.
    """
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    ac, bc = a - ma, b - mb
    u, _, vt = np.linalg.svd(ac.T @ bc, full_matrices=False)
    r = u @ vt
    return r, ma, mb


def _centroids(emb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean embedding over the masked rows."""
    return emb[mask].mean(axis=0)


def _cohesion(h_vecs: list[np.ndarray], o_vecs: list[np.ndarray]) -> float:
    """Mean cosine between paired held-out and pooled-other centroids."""
    if not h_vecs:
        return float('nan')
    h = _l2norm(np.asarray(h_vecs, dtype=np.float32))
    o = _l2norm(np.asarray(o_vecs, dtype=np.float32))
    return float(np.mean(np.sum(h * o, axis=1)))


def _calibrate_one(
    emb: np.ndarray,
    subjects: np.ndarray,
    words: np.ndarray,
    holdout: str,
    n_anchors: int,
    min_shared: int,
) -> dict[str, Any] | None:
    """Calibrates one held-out subject from anchor words; returns before/after cohesion."""
    is_h = subjects == holdout
    if not is_h.any() or is_h.all():
        return None
    # Words this subject shares with at least one other subject.
    h_words = {w: np.where(is_h & (words == w))[0] for w in np.unique(words[is_h]) if w}
    shared = []
    for w, h_idx in h_words.items():
        o_idx = np.where((~is_h) & (words == w))[0]
        if len(h_idx) and len(o_idx):
            shared.append((w, h_idx, o_idx, min(len(h_idx), len(o_idx))))
    if len(shared) < max(min_shared, 4):
        return None

    # The most-shared words become anchors; the rest are the held-out test words.
    shared.sort(key=lambda t: t[3], reverse=True)
    k = max(3, min(n_anchors, len(shared) - 1))
    anchors, tests = shared[:k], shared[k:]
    if not tests:
        tests = anchors  # tiny data: fall back to in-sample (flagged by n_test == n_anchors)

    # Fit the alignment on the anchors only.
    a = np.asarray([emb[h].mean(axis=0) for _w, h, _o, _c in anchors], dtype=np.float32)
    b = np.asarray([emb[o].mean(axis=0) for _w, _h, o, _c in anchors], dtype=np.float32)
    r, ma, mb = _procrustes(a, b)

    # Cohesion on the test words, before vs after the map.
    h_before, h_after, o_c = [], [], []
    for _w, h, o, _c in tests:
        hc = emb[h].mean(axis=0)
        h_before.append(hc)
        h_after.append((hc - ma) @ r + mb)
        o_c.append(emb[o].mean(axis=0))
    before = _cohesion(h_before, o_c)
    after = _cohesion(h_after, o_c)
    return {
        'holdout': str(holdout),
        'n_anchors': int(k),
        'n_test_words': int(len(tests)),
        'cohesion_before': before,
        'cohesion_after': after,
        'lift': (after - before) if (np.isfinite(after) and np.isfinite(before)) else float('nan'),
    }


def anchor_calibration_lift(
    word_emb: np.ndarray,
    word_meta: pd.DataFrame,
    *,
    holdout: str | None = None,
    n_anchors: int = 12,
    min_shared: int = 6,
    seed: int = 0,
) -> dict[str, Any]:
    """Does aligning a held-out subject from a few anchor words improve cross-subject cohesion?

    An orthogonal Procrustes map is fit from the `n_anchors` most-shared words and applied to the
    held-out subject; cohesion (mean cosine of same-word cross-subject centroids) is measured on the
    remaining held-out words. A positive lift means a stranger's brain can be pulled toward the
    shared frame without retraining.

    Args:
        word_emb (np.ndarray): Word embeddings `(n, d)`.
        word_meta (pd.DataFrame): Metadata with `subject` and `word` columns.
        holdout (str | None): A specific held-out subject (e.g. the LOSO subject). `None` averages
            over every subject in turn.
        n_anchors (int): Number of anchor words used to fit the alignment.
        min_shared (int): Minimum shared words required to calibrate a subject.
        seed (int): Unused placeholder for API symmetry / future stochastic variants.

    Returns:
        dict: `{applicable, holdout, n_anchors, per_subject: [...], mean_cohesion_before,
            mean_cohesion_after, mean_lift, helps}` or `{applicable: False, reason}`.
    """
    if 'subject' not in word_meta.columns or 'word' not in word_meta.columns:
        return {'applicable': False, 'reason': 'need subject and word columns'}
    subjects = word_meta['subject'].astype(str).to_numpy()
    words = word_meta['word'].astype(str).to_numpy()
    uniq = np.unique(subjects)
    if len(uniq) < 2:
        return {'applicable': False, 'reason': 'need >= 2 subjects'}
    emb = np.asarray(word_emb, dtype=np.float32)

    targets = [holdout] if (holdout is not None and holdout in uniq) else list(uniq)
    per = []
    for h in targets:
        res = _calibrate_one(emb, subjects, words, str(h), n_anchors, min_shared)
        if res is not None:
            per.append(res)
    if not per:
        return {'applicable': False, 'reason': 'too few shared words to calibrate'}

    before = float(np.nanmean([p['cohesion_before'] for p in per]))
    after = float(np.nanmean([p['cohesion_after'] for p in per]))
    return {
        'applicable': True,
        'holdout': holdout,
        'n_anchors': int(n_anchors),
        'per_subject': per,
        'mean_cohesion_before': before,
        'mean_cohesion_after': after,
        'mean_lift': after - before,
        'helps': bool(after > before),
    }
