"""The honest scoreboard: the held-out subject's numbers, each stated as a lift over the raw band-power control."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from zte.evaluation import metrics as M
from zte.evaluation.audit.probe import detectability_curve, detectability_verdict
from zte.evaluation.neurons import neuron_report
from zte.training.metrics import residualise

if TYPE_CHECKING:
    import pandas as pd

# Probe targets a meaning code should carry, and those a thought code should not; mirrors zte.evaluation.neurons.
_CONTENT_TARGETS: tuple[str, ...] = ('word_len', 'log_freq', 'category')
_IDENTITY_TARGETS: tuple[str, ...] = ('subject',)

# Positive-control floor: below this linear R2 on raw band power the content probe is not trustworthy.
_CONTENT_PROBE_FLOOR: float = 0.02
# Splits that hold one subject out entirely, so a held-out block is meaningful.
_HOLDOUT_SPLITS: frozenset[str] = frozenset({'by_subject_loso', 'by_subject_and_stimulus'})
# Word-count columns a sentence metadata frame may carry, for the length-stratified gallery.
_LENGTH_COLUMNS: tuple[str, ...] = ('n_words', 'length', 'sentence_length')


def holdout_subject(config: Any | None) -> str | None:
    """Returns the held-out subject, or `None` when the run's split does not hold a subject out."""
    train = getattr(config, 'train', None)
    if train is None or getattr(train, 'split', None) not in _HOLDOUT_SPLITS:
        return None
    return getattr(train, 'loso_holdout_subject', None)


def lift_over_raw(comparison: list[dict[str, Any]]) -> dict[str, Any]:
    """Turns the probe-comparison rows into per-target `ZTE - raw` lifts.

    Args:
        comparison (list[dict]): Rows from `metrics.representation_comparison`, each with
            `target`, `representation`, `metric`, `linear_score`, `knn_score`.

    Returns:
        dict: `{target: {metric, zte, raw, noise, lift_linear, lift_knn}}` plus a
        `content_probe` positive-control block.
    """
    by_target: dict[str, dict[str, Any]] = {}
    for row in comparison:
        t = row['target']
        by_target.setdefault(t, {'metric': row['metric']})
        by_target[t][row['representation']] = {
            'linear': row['linear_score'],
            'knn': row['knn_score'],
        }

    lifts: dict[str, Any] = {}
    for t, reps in by_target.items():
        zte = reps.get('ZTE', {})
        raw = reps.get('raw band-power', {})
        noise = reps.get('noise (matched)', {})
        lifts[t] = {
            'metric': reps['metric'],
            'zte_linear': zte.get('linear'),
            'raw_linear': raw.get('linear'),
            'noise_linear': noise.get('linear'),
            'lift_linear': _sub(zte.get('linear'), raw.get('linear')),
            'lift_knn': _sub(zte.get('knn'), raw.get('knn')),
            'is_content': t in _CONTENT_TARGETS,
            'is_identity': t in _IDENTITY_TARGETS,
        }

    # Fallback only -- `build_scoreboard` fills this from genuinely raw band power. The probe-row value
    # is unreliable under whitening normalisers that strip amplitude, so never gate "content 0%" on it.
    raw_content = [
        by_target[t]['raw band-power']['linear']
        for t in ('word_len', 'log_freq')
        if t in by_target and 'raw band-power' in by_target[t]
    ]
    best = max(raw_content) if raw_content else float('nan')
    lifts['content_probe'] = {
        'raw_content_r2_best': best,
        'floor': _CONTENT_PROBE_FLOOR,
        'passes': bool(np.isfinite(best) and best >= _CONTENT_PROBE_FLOOR),
        'source': 'normalised-features (fallback)',
    }
    return lifts


# Eye-tracking columns that are a lexical-difficulty proxy by construction: the confound audit puts their
# correlation ratio with word length at 0.24-0.39, so a probe that cannot read word length from these is broken.
_MACHINERY_FEATURES: tuple[str, ...] = ('TRT', 'GD', 'FFD', 'GPT', 'n_fixations', 'regression_time')

# The empirical zero needs enough draws for its own spread to be readable; 20 also puts the smallest p at 1/21.
_N_PERMUTATIONS: Final[int] = 20
"""Target permutations behind the probe's empirical zero."""

# Every probe here is O(n p^2) per fold and runs inside every evaluation, so the rows it sees are bounded.
_MAX_PROBE_ROWS: Final[int] = 20_000
"""Rows the raw-content probe fits on, deterministically subsampled when the run has more."""

# One stimulus is read by every subject, so ungrouped folds train and score on the same sentence, best column first.
_PROBE_GROUP_COLUMNS: Final[tuple[str, ...]] = ('stimulus_key', 'sentence_uid', 'subject')
"""Columns the content probe groups its folds on."""


def raw_content_positive_control(
    word_band_power: np.ndarray | None,
    word_meta: 'pd.DataFrame',
    *,
    n_perm: int = _N_PERMUTATIONS,
    max_rows: int = _MAX_PROBE_ROWS,
    calibrate: bool = True,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Probes genuinely-raw band power for lexical content -- the honest positive control.

    Note:
        Three questions are asked, and conflating them is what made "the probe is broken" and "band power carries no
        lexical content" indistinguishable. `machinery` probes word length from eye-tracking features that carry it
        by construction, so a failure there is a fault in the probe. `detectability` plants a known signal in this
        run's own band-power matrix and reports the R2 at which the probe starts to see it, which is what separates
        "absent" from "below the floor". `per_target_r2` and `within_subject_r2` probe band power itself, and a
        failure there with the other two passing is a result about the signal, not a broken probe. Read the
        within-subject number: subject identity is linearly readable from raw band power at 0.81 while word length is
        not, so a probe fitted across subjects spends its capacity on who is reading. Every fold is grouped on a
        stimulus column, because one sentence is read by every subject and ungrouped folds score what they trained on.

    Args:
        word_band_power (np.ndarray | None): Raw band power `(n, bands, channels)`, or `None` for a
            raw-signal frontend (the control is then not applicable).
        word_meta (pd.DataFrame): Per-word metadata carrying `word_len`, `log_freq`, `subject`, a grouping column
            and, when eye tracking is present, the fixation columns the machinery check reads.
        n_perm (int, optional): Target permutations behind the empirical zero. Defaults to 20.
        max_rows (int, optional): Cap on the rows every probe here fits on. Defaults to 20000.
        calibrate (bool, optional): Whether to run the detectability calibration. Defaults to True.
        seed (int, optional): Seed for the subsample, the permutations and the planted signals. Defaults to 0.

    Returns:
        dict | None: The positive-control block, or `None` when there is nothing to probe.
    """
    groups, group_column = _probe_groups(word_meta)
    machinery = _probe_machinery_control(word_meta, groups, group_column, max_rows=max_rows, seed=seed)

    pooled: dict[str, float] = {}
    within: dict[str, float] = {}
    nulls: dict[str, dict[str, Any]] = {}
    detectability: dict[str, Any] | None = None
    n_probed: int | None = None

    if word_band_power is not None:
        flat = _flat_band_power(word_band_power)
        subjects = word_meta['subject'].to_numpy() if 'subject' in word_meta.columns else None

        for target in ('word_len', 'log_freq'):
            if target not in word_meta.columns:
                continue

            y = np.asarray(word_meta[target].to_numpy(), dtype=np.float64)
            rows = _probe_rows(np.isfinite(y), max_rows, seed)
            if rows is None:
                continue

            x, y_keep = flat[rows], y[rows]
            group = groups[rows] if groups is not None else None
            # Reported from the first target, which is also the matrix the detectability floor is calibrated on.
            n_probed = int(rows.size) if n_probed is None else n_probed

            pooled[target] = _r2(M.linear_probe(x, y_keep, task='regression', groups=group))
            if subjects is not None:
                # Centring both sides is only half of the within-subject question: the folds have to stay grouped too.
                reader = np.asarray(subjects)[rows]
                within[target] = _r2(
                    M.linear_probe(residualise(x, reader), residualise(y_keep, reader), task='regression', groups=group)
                )
            nulls[target] = _permutation_null(x, y_keep, group, n_perm=n_perm, observed=pooled[target], seed=seed)

            # One calibration per run: the two targets share a matrix, so the floor they are read against is one number.
            if calibrate and detectability is None:
                detectability = detectability_curve(x, groups=group, seed=seed)

    if not pooled and machinery is None:
        return None

    pooled_best = max(pooled.values()) if pooled else float('nan')
    within_best = max(within.values()) if within else float('nan')
    # A pooled probe can score on who is reading rather than what, so the honest within-subject number is the headline.
    best = within_best if np.isfinite(within_best) else pooled_best
    calibration = detectability_verdict(float(best) if np.isfinite(best) else None, detectability)

    clauses = {
        'band_power_probed': bool(pooled),
        'machinery_reads_word_length': machinery['passes'] if machinery else None,
        'estimator_recovers_planted_signal': detectability['established'] if detectability else None,
        'band_power_above_floor': bool(np.isfinite(best) and best >= _CONTENT_PROBE_FLOOR),
    }
    evidence = [
        name
        for name, holds in (
            ('eye-tracking machinery check', clauses['machinery_reads_word_length']),
            ('planted-signal detectability', clauses['estimator_recovers_planted_signal']),
            ('raw band power above floor', clauses['band_power_above_floor']),
        )
        if holds
    ]
    # Neither clause decides alone: an estimator that reads content where it exists, and band power actually probed.
    passes = bool(clauses['band_power_probed'] and evidence)

    return {
        'raw_content_r2_best': round(float(best), 4) if np.isfinite(best) else None,
        'raw_content_r2_pooled_best': round(float(pooled_best), 4) if np.isfinite(pooled_best) else None,
        'raw_content_r2_within_best': round(float(within_best), 4) if np.isfinite(within_best) else None,
        'best_is': 'within-subject' if np.isfinite(within_best) else ('pooled' if np.isfinite(pooled_best) else None),
        'per_target_r2': {k: round(v, 4) for k, v in pooled.items()},
        'within_subject_r2': {k: round(v, 4) for k, v in within.items()},
        'shuffled_target_r2': {k: v['mean'] for k, v in nulls.items()},
        'permutation_null': nulls,
        'detectability': detectability,
        'detectability_verdict': calibration,
        'machinery': machinery,
        'band_power_applicable': word_band_power is not None,
        'n_probed': n_probed,
        'cv_groups': group_column,
        'within_subject_centring': 'transductive (subject means over every row)' if within else None,
        'floor': _CONTENT_PROBE_FLOOR,
        'passes': passes,
        'passes_clauses': clauses,
        'decided_by': ' + '.join(evidence) if evidence else 'nothing established the estimator',
        'finding': _content_finding(machinery, pooled_best, within_best, best, calibration, bool(pooled)),
        'source': 'raw band-power' if word_band_power is not None else 'raw signal (no band power to probe)',
    }


def _probe_groups(word_meta: 'pd.DataFrame') -> tuple[np.ndarray | None, str | None]:
    """The first grouping column with two or more levels, so no stimulus is both trained on and scored."""
    for column in _PROBE_GROUP_COLUMNS:
        if column not in word_meta.columns:
            continue

        values = word_meta[column].astype(str).to_numpy()
        if len(np.unique(values)) >= 2:
            return values, column

    return None, None


def _probe_rows(keep: np.ndarray, max_rows: int, seed: int) -> np.ndarray | None:
    """Rows the probe fits on: the finite ones, capped by a deterministic subsample, or `None` when too few."""
    rows = np.flatnonzero(keep)
    if rows.size < 32:
        return None

    if rows.size > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, size=max_rows, replace=False))

    return rows


def _flat_band_power(word_band_power: np.ndarray) -> np.ndarray:
    """Flattens band power to `(n, bands * channels)`, imputing an omitted word to the column mean."""
    flat = np.asarray(word_band_power, dtype=np.float32).reshape(len(word_band_power), -1)
    col_mean = np.nanmean(np.where(np.isfinite(flat), flat, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)

    return np.where(np.isfinite(flat), flat, col_mean)


def _permutation_null(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray | None, *, n_perm: int, observed: float, seed: int
) -> dict[str, Any]:
    """The empirical zero: the identical estimator on `n_perm` permuted copies of the same target."""
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [_r2(M.linear_probe(x, rng.permutation(y), task='regression', groups=groups)) for _ in range(max(1, n_perm))],
        dtype=np.float64,
    )
    beaten = int(np.sum(draws >= observed)) if np.isfinite(observed) else draws.size

    return {
        'mean': round(float(np.mean(draws)), 4),
        'std': round(float(np.std(draws)), 4),
        'interval': [round(float(np.quantile(draws, 0.025)), 4), round(float(np.quantile(draws, 0.975)), 4)],
        'p_value': round((1.0 + beaten) / (draws.size + 1.0), 4),
        'p_floor': round(1.0 / (draws.size + 1.0), 4),
        'n_perm': int(draws.size),
        'scores': [round(float(s), 4) for s in draws],
    }


def _content_finding(
    machinery: dict[str, Any] | None,
    pooled_best: float,
    within_best: float,
    best: float,
    calibration: dict[str, Any],
    probed: bool,
) -> str:
    """One honest sentence per clause: what the estimator can read, what it recovers, and what band power gave."""
    if not probed:
        return (
            'No band power was probed, so nothing here establishes the content probe on the features the model reads.'
        )

    parts: list[str] = []
    if machinery:
        verb = 'reads' if machinery['passes'] else 'cannot read'
        parts.append(
            f'The estimator {verb} word length off {len(machinery["features"])} eye-tracking features at '
            f'R2={machinery["word_len_r2"]:.4f}.'
        )
    parts.append(str(calibration['statement']))

    if np.isfinite(pooled_best) and np.isfinite(within_best):
        parts.append(
            f'Pooled over subjects band power scores R2={pooled_best:.4f} and within a subject R2={within_best:.4f}; '
            'the within-subject number is the one quoted, because the pooled probe can score on who is reading.'
        )

    established = bool((machinery and machinery['passes']) or calibration['established'])
    if not established:
        parts.append('Nothing establishes the estimator here, so no content number in this report is readable yet.')
    elif np.isfinite(best) and best < _CONTENT_PROBE_FLOOR:
        parts.append(
            'Raw band power carries no word length this probe can decode linearly. With the estimator established, '
            'that is a result about the signal rather than a broken probe.'
        )
    else:
        parts.append('Raw band power carries lexical content, so a content budget measured against it is readable.')

    return ' '.join(parts)


def _probe_machinery_control(
    word_meta: 'pd.DataFrame',
    groups: np.ndarray | None,
    group_column: str | None,
    *,
    max_rows: int = _MAX_PROBE_ROWS,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Probes word length from eye-tracking features, which is the check that the estimator itself works."""
    columns = [c for c in _MACHINERY_FEATURES if c in word_meta.columns]
    if not columns or 'word_len' not in word_meta.columns:
        return None

    x = np.asarray(word_meta[columns].to_numpy(), dtype=np.float64)
    y = np.asarray(word_meta['word_len'].to_numpy(), dtype=np.float64)
    # Capped and grouped exactly like the band-power probe, or the two numbers answer questions of different difficulty.
    rows = _probe_rows(np.isfinite(y) & np.isfinite(x).all(axis=1), max_rows, seed)
    if rows is None:
        return None

    group = groups[rows] if groups is not None else None
    score = _r2(M.linear_probe(x[rows], y[rows], task='regression', groups=group))

    return {
        'features': columns,
        'word_len_r2': round(score, 4),
        'floor': _CONTENT_PROBE_FLOOR,
        'passes': bool(np.isfinite(score) and score >= _CONTENT_PROBE_FLOOR),
        'n': int(rows.size),
        'cv_groups': group_column,
    }


def _r2(result: dict[str, Any]) -> float:
    """Pulls the score out of a probe result, as a plain float."""
    return float(result.get('score', float('nan')))


def _sub(a: float | None, b: float | None) -> float | None:
    """`a - b`, tolerating missing operands."""
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 4)


def held_out_geometry(word_emb: np.ndarray, word_meta: 'pd.DataFrame', holdout: str) -> dict[str, Any] | None:
    """Geometry + variance budget over the held-out subject's rows only, undiluted by the training subjects."""
    if 'subject' not in word_meta.columns:
        return None
    mask = (word_meta['subject'] == holdout).to_numpy()
    if mask.sum() < 8:  # too few to characterise geometry
        return None
    sub_emb = np.asarray(word_emb, dtype=np.float64)[mask]
    sub_meta = word_meta.loc[mask].reset_index(drop=True)
    d = sub_emb.shape[1]
    budget = neuron_report(sub_emb, sub_meta).get('summary', {})
    return {
        'subject': holdout,
        'n_words': int(mask.sum()),
        'effective_rank': float(M.effective_rank(sub_emb)),
        'effective_rank_ratio': float(M.effective_rank(sub_emb) / d),
        'anisotropy': float(M.anisotropy(sub_emb)),
        'subject_variance': None,  # single subject -> identity budget is not meaningful here
        'content_variance': budget.get('what_variance'),
        'task_variance': budget.get('variance_budget', {}).get('task'),
    }


def cross_subject_holdout_retrieval(
    sent_emb: np.ndarray,
    content_ids: np.ndarray,
    subjects: np.ndarray,
    holdout: str,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any] | None:
    """Held-out queries vs a cross-subject gallery: find a stranger's thought in others.

    Args:
        sent_emb (np.ndarray): Sentence embeddings `(n, d)`.
        content_ids (np.ndarray): Stimulus id per sentence `(n,)`.
        subjects (np.ndarray): Subject code per sentence `(n,)`.
        holdout (str): The held-out subject code.
        ks (tuple[int, ...]): Top-K cut-offs.

    Returns:
        dict | None: `top{k}`, `mrr`, `chance_top1`, `n_queries`, `excluded_no_positive`,
            `lift_top1` (top1 - chance) and a per-query `top1_hits` vector for the verdict's CI,
            or `None` when the run has too few subjects/queries to be meaningful.
    """
    subjects = np.asarray(subjects)
    content_ids = np.asarray(content_ids)
    if len(np.unique(subjects)) < 2:
        return None
    q_mask = subjects == holdout
    if q_mask.sum() < 2:
        return None

    # Cosine similarity over the whole sentence set; the gallery is masked per query.
    emb = np.asarray(sent_emb, dtype=np.float64)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = emb @ emb.T  # cosine, (n, n)

    q_idx = np.where(q_mask)[0]
    out: dict[str, Any] = {f'top{k}': 0.0 for k in ks}
    rr = 0.0
    chances = []
    percentiles: list[float] = []  # rank-percentile per query (1.0 = correct match ranked first)
    top1_hits: list[float] = []  # per-query Top-1 hits, feeding the verdict's held-out CI
    excluded = 0

    for i in q_idx:
        cross = subjects != subjects[i]  # gallery: other people only
        if not cross.any():
            excluded += 1
            continue

        cand = np.where(cross)[0]
        order = cand[np.argsort(-sims[i, cand])]
        same = content_ids[order] == content_ids[i]
        # Truth absent from the cross-subject gallery: excluded and counted, never zero-scored into an artifact.
        if not same.any():
            excluded += 1
            continue

        for k in ks:
            out[f'top{k}'] += float(same[:k].any())

        top1_hits.append(float(same[:1].any()))
        rr += 1.0 / (np.argmax(same) + 1)
        rank = int(np.argmax(same)) + 1
        percentiles.append(1.0 - (rank - 1) / max(len(order) - 1, 1))
        chances.append(float((content_ids[cand] == content_ids[i]).mean()))

    n_scored = len(percentiles)
    if n_scored == 0:
        return None

    # Pool the per-query counters into rates.
    for k in ks:
        out[f'top{k}'] /= n_scored
    out['mrr'] = rr / n_scored
    out['chance_top1'] = float(np.mean(chances)) if chances else float('nan')
    out['rank_percentile'] = float(np.mean(percentiles)) if percentiles else float('nan')
    out['n_queries'] = int(n_scored)
    out['excluded_no_positive'] = int(excluded)
    out['top1_hits'] = top1_hits
    out['lift_top1'] = _sub(out['top1'], out['chance_top1'])

    # Top-1 on ~700 queries at 1/700 chance expects ONE hit, so rates there are unreadable: report an exact tail
    # probability, plus a CI on the one statistic that uses every query rather than only the winners.
    for k in ks:
        out[f'top{k}_p'] = _binom_tail_p(round(out[f'top{k}'] * n_scored), n_scored, out['chance_top1'] * k)
    out['rank_percentile_ci'] = _bootstrap_ci(np.asarray(percentiles, dtype=np.float64))
    out['headline_metric'] = 'rank_percentile'

    return out


def decoder_rescoring_retrieval(
    scores: np.ndarray,
    query_content_ids: np.ndarray,
    gallery_content_ids: np.ndarray,
    *,
    subjects: np.ndarray | None = None,
    holdout: str | None = None,
    query_n_words: np.ndarray | None = None,
    gallery_n_words: np.ndarray | None = None,
    length_tol: int = 1,
    ks: tuple[int, ...] = (1, 5, 10),
    seed: int = 0,
) -> dict[str, Any] | None:
    """Retrieval by decoder sequence likelihood over the sentence gallery -- the powered readout.

    Each query scores every gallery sentence by length-normalised `log p(text | z)`. This is forced
    choice over a known candidate set, so it is retrieval and is named as such: it is directly
    comparable to `held_out_retrieval` and is never reported as generation. It is also the statistically
    powered one -- 700 queries at ~9.5 bits, against a generation delta at n = 105.

    Args:
        scores (np.ndarray): Query-by-gallery scores `(n_queries, n_gallery)`, higher is better.
        query_content_ids (np.ndarray): Stimulus id of each query `(n_queries,)`.
        gallery_content_ids (np.ndarray): Stimulus id of each gallery sentence `(n_gallery,)`.
        subjects (np.ndarray | None, optional): Subject code per query, for filtering to `holdout`.
        holdout (str | None, optional): Keep only this subject's queries. Requires `subjects`.
        query_n_words (np.ndarray | None, optional): Word count per query reading, used only for a query whose
            stimulus is absent from the gallery; queries otherwise stratify on their stimulus's gallery length.
        gallery_n_words (np.ndarray | None, optional): Word count per gallery sentence (stimulus-level medians).
        length_tol (int, optional): Word-count tolerance for the stratified gallery. Defaults to 1.
        ks (tuple[int, ...], optional): Top-K cut-offs. Defaults to (1, 5, 10).
        seed (int, optional): Bootstrap seed. Defaults to 0.

    Returns:
        dict | None: `top{k}`, `top{k}_p`, `mrr`, `rank_percentile`, `rank_percentile_ci`, `mean_rank`,
            `chance_top1`, `n_queries`, `excluded_no_positive`, `n_gallery`, `headline_metric`, and a
            `length_stratified` sub-block when both word-count arrays are given; `None` when nothing is scoreable.
    """
    mat = np.asarray(scores, dtype=np.float64)
    q_ids = np.asarray(query_content_ids)
    g_ids = np.asarray(gallery_content_ids)
    if mat.ndim != 2 or mat.shape[0] != len(q_ids) or mat.shape[1] != len(g_ids):
        return None
    if subjects is not None and holdout is not None:
        keep = np.asarray(subjects) == holdout
        if keep.sum() < 2:
            return None
        mat, q_ids = mat[keep], q_ids[keep]
        if query_n_words is not None:
            query_n_words = np.asarray(query_n_words)[keep]
    if len(mat) < 2 or len(g_ids) < 2:
        return None

    out = _rescore_cell(mat, q_ids, g_ids, None, None, length_tol, ks, seed)
    if out is None:
        return None
    if query_n_words is not None and gallery_n_words is not None:
        g_words = np.asarray(gallery_n_words, dtype=np.float64)
        # One unit on both sides: queries stratify on their stimulus's gallery median, not the reading's own count.
        q_words = _lengths_in_gallery_units(q_ids, g_ids, g_words, np.asarray(query_n_words, dtype=np.float64))
        out['length_stratified'] = _rescore_cell(mat, q_ids, g_ids, q_words, g_words, length_tol, ks, seed)
    return out


def _lengths_in_gallery_units(
    q_ids: np.ndarray, g_ids: np.ndarray, g_words: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    """Per-query lengths in the gallery's unit: each query's own stimulus entry, else the reading's count."""
    lookup = {g: float(w) for g, w in zip(g_ids.tolist(), g_words.tolist())}

    return np.array([lookup.get(q, float(f)) for q, f in zip(q_ids.tolist(), fallback.tolist())], dtype=np.float64)


def _rescore_cell(
    scores: np.ndarray,
    q_ids: np.ndarray,
    g_ids: np.ndarray,
    q_words: np.ndarray | None,
    g_words: np.ndarray | None,
    length_tol: int,
    ks: tuple[int, ...],
    seed: int,
) -> dict[str, Any] | None:
    """Ranks one query-by-gallery score matrix, optionally inside a matched word-count gallery."""
    hits = {k: 0.0 for k in ks}
    reciprocal = 0.0
    percentiles: list[float] = []
    ranks: list[float] = []
    chances: list[float] = []
    galleries: list[float] = []
    excluded = 0

    for i in range(len(scores)):
        cand = np.arange(len(g_ids))
        if q_words is not None and g_words is not None:
            cand = cand[np.abs(g_words - q_words[i]) <= length_tol]
        # An unanswerable query (truth absent, or no distractor) is excluded and counted, never zero-scored.
        if cand.size < 2 or not (g_ids[cand] == q_ids[i]).any():
            excluded += 1
            continue
        galleries.append(float(cand.size))
        chances.append(float((g_ids[cand] == q_ids[i]).mean()))
        order = cand[np.argsort(-scores[i, cand])]
        same = g_ids[order] == q_ids[i]
        for k in ks:
            hits[k] += float(same[:k].any())
        rank = int(np.argmax(same)) + 1
        reciprocal += 1.0 / rank
        ranks.append(float(rank))
        percentiles.append(1.0 - (rank - 1) / max(cand.size - 1, 1))

    n_scored = len(percentiles)
    if n_scored == 0:
        return None

    out: dict[str, Any] = {f'top{k}': hits[k] / n_scored for k in ks}
    out['mrr'] = reciprocal / n_scored
    out['chance_top1'] = float(np.mean(chances))
    out['rank_percentile'] = float(np.mean(percentiles))
    out['mean_rank'] = float(np.mean(ranks))
    out['median_rank'] = float(np.median(ranks))
    out['mean_gallery'] = float(np.mean(galleries))
    out['n_queries'] = int(n_scored)
    out['excluded_no_positive'] = int(excluded)
    out['n_gallery'] = int(len(g_ids))
    out['length_tol'] = None if q_words is None else int(length_tol)
    for k in ks:
        out[f'top{k}_p'] = _binom_tail_p(round(out[f'top{k}'] * n_scored), n_scored, out['chance_top1'] * k)
    out['rank_percentile_ci'] = _bootstrap_ci(np.asarray(percentiles, dtype=np.float64), seed=seed)
    out['headline_metric'] = 'rank_percentile'
    out['readout'] = 'retrieval'
    return out


def within_task_retrieval(
    scores: np.ndarray,
    query_meta: 'pd.DataFrame',
    *,
    gallery_tasks: np.ndarray,
    gallery_n_words: np.ndarray | None = None,
    pools: Sequence[str] = ('SR', 'NR'),
    length_tol: int = 1,
    ks: tuple[int, ...] = (1, 5, 10),
    seed: int = 0,
) -> dict[str, Any]:
    """Re-ranks each query inside its own reading task, so passage identity cannot be doing the work.

    On ZuCo no stimulus appears under more than one task -- the confound audit puts Cramer's V(task, stimulus) at
    0.998 -- so the full 700-sentence gallery lets a model score by telling SR sentences from NR ones, which is a
    passage-set property and not a reading of the brain. Inside a single task the passage set is fixed, so a lift that
    survives here is a lift on sentence content. It is the pool a sceptical reader asks for, and it is smaller, so
    both its chance level and its confidence interval are reported alongside.

    Args:
        scores (np.ndarray): Query-by-gallery scores `(n_queries, n_gallery)`, higher is better.
        query_meta (pd.DataFrame): Per-query metadata carrying `task`, `text_id` and optionally `n_words`.
        gallery_tasks (np.ndarray): Task of each gallery sentence `(n_gallery,)`.
        gallery_n_words (np.ndarray | None, optional): Word count per gallery sentence, for the stratified cell.
        pools (Sequence[str], optional): Tasks to report. Defaults to ('SR', 'NR').
        length_tol (int, optional): Word-count tolerance inside a pool. Defaults to 1.
        ks (tuple[int, ...], optional): Top-K cut-offs. Defaults to (1, 5, 10).
        seed (int, optional): Bootstrap seed. Defaults to 0.

    Returns:
        dict[str, Any]: One block per task that had at least two queries and two candidates.
    """
    mat = np.asarray(scores, dtype=np.float64)
    tasks = np.asarray(gallery_tasks).astype(str)
    if mat.ndim != 2 or 'task' not in query_meta or 'text_id' not in query_meta:
        return {}

    q_tasks = query_meta['task'].astype(str).to_numpy()
    q_ids = query_meta['text_id'].to_numpy()
    q_words = query_meta['n_words'].to_numpy() if 'n_words' in query_meta else None

    out: dict[str, Any] = {}
    for task in pools:
        rows = np.flatnonzero(q_tasks == str(task))
        cols = np.flatnonzero(tasks == str(task))
        if rows.size < 2 or cols.size < 2:
            continue
        # Gallery column `j` is the sentence whose `text_id` is `j`, so the selected column indices *are* the ids.
        cell = _rescore_cell(mat[np.ix_(rows, cols)], q_ids[rows], cols, None, None, length_tol, ks, seed)
        if cell is None:
            continue
        cell['task'] = str(task)
        cell['n_candidates'] = int(cols.size)
        cell['pool'] = 'within_task'
        if q_words is not None and gallery_n_words is not None:
            pool_g_words = np.asarray(gallery_n_words, dtype=np.float64)[cols]
            # One unit on both sides: queries stratify on their stimulus's gallery median, not the reading's count.
            pool_q_words = _lengths_in_gallery_units(
                q_ids[rows], cols, pool_g_words, np.asarray(q_words, dtype=np.float64)[rows]
            )
            cell['length_stratified'] = _rescore_cell(
                mat[np.ix_(rows, cols)],
                q_ids[rows],
                cols,
                pool_q_words,
                pool_g_words,
                length_tol,
                ks,
                seed,
            )
        out[str(task)] = cell
    return out


def held_out_generation(block: dict[str, Any] | None) -> dict[str, Any] | None:
    """Condenses a `generation.generation_report` into the scoreboard's held-out generation row.

    Only the paired deltas, the permutation null and the prefix-influence KL survive; every absolute
    score travels with the control that makes it readable, and every `*_DIAGNOSTIC` / `*_RETRIEVAL` key
    is stripped so it cannot reach a verdict. A control recorded unavailable or skipped produced no
    delta at all, so it is named in `controls_absent` and denies `beats_all_controls`.

    Args:
        block (dict[str, Any] | None): The generation report, or `None`.

    Returns:
        dict | None: `{'applicable', 'n', 'split', 'split_strategy', 'primary_metric', 'hypothesis',
            'controls', 'oracle', 'deltas', 'worst_control', 'worst_control_ci', 'beats_all_controls',
            'controls_requested', 'controls_absent', 'permutation_p', 'prefix_influence_kl',
            'n_candidate_sentences', 'quarantined'}`.
    """
    if not block:
        return None
    from zte.evaluation.generation import quarantined_keys, strip_quarantined

    quarantined = quarantined_keys(block)
    clean = strip_quarantined(block)
    if not clean.get('applicable'):
        return {'applicable': False, 'reason': clean.get('reason', 'not applicable')}

    metric = clean.get('primary_metric', 'content_f1')
    absolute = clean.get('absolute') or {}
    perm = clean.get('permutation') or {}
    deltas = clean.get('deltas') or {}
    absent = (clean.get('controls_unavailable') or {}) | (clean.get('controls_skipped') or {})
    return {
        'applicable': True,
        'n': clean.get('n'),
        'split': clean.get('split'),
        'split_strategy': clean.get('split_strategy'),
        'primary_metric': metric,
        'headline_metric': f'{metric}_delta',
        'hypothesis': absolute.get('hypothesis'),
        'controls': absolute.get('controls'),
        'oracle': absolute.get('oracle'),
        'deltas': {name: d.get(metric) for name, d in deltas.items()},
        'worst_control': clean.get('worst_control'),
        'worst_control_ci': clean.get('worst_control_ci'),
        'beats_all_controls': bool(clean.get('beats_all_controls')) and not absent,
        'controls_requested': list(clean.get('controls_requested') or deltas),
        'controls_absent': absent,
        'permutation_p': perm.get('p_value') if perm.get('applicable') else None,
        'prefix_influence_kl': clean.get('prefix_influence_kl'),
        'n_candidate_sentences': clean.get('n_candidate_sentences'),
        'quarantined': quarantined,
        'readout': 'generation',
    }


def stimulus_median_lengths(n_words: np.ndarray, content_ids: np.ndarray) -> np.ndarray:
    """Replaces each reading's word count with the median over every reading of its stimulus.

    Eye-tracking segmentation lets two readings of one stimulus differ in word count, so a stratum keyed on a
    reading's own count can drop the truth that a median-keyed gallery retains; one unit on both sides removes that.

    Args:
        n_words (np.ndarray): Word count per reading `(n,)`.
        content_ids (np.ndarray): Stimulus id per reading `(n,)`.

    Returns:
        np.ndarray: Stimulus-level median word count per reading `(n,)`.
    """
    lengths = np.asarray(n_words, dtype=np.float64).ravel()
    ids = np.asarray(content_ids).ravel()

    out = lengths.copy()
    for value in np.unique(ids):
        mask = ids == value
        out[mask] = float(np.median(lengths[mask]))
    return out


def embedding_checksum(emb: np.ndarray) -> str:
    """Short sha256 of an embedding matrix, so two arms measuring byte-identical embeddings are visibly the same.

    Args:
        emb (np.ndarray): Embedding matrix `(n, d)`.

    Returns:
        str: First 16 hex digits of the sha256 over the float32 bytes and the shape.
    """
    arr = np.ascontiguousarray(np.asarray(emb, dtype=np.float32))
    digest = hashlib.sha256(repr(arr.shape).encode())
    digest.update(arr.tobytes())

    return digest.hexdigest()[:16]


def _binom_tail_p(hits: int, n: int, p_chance: float) -> float:
    """Exact probability of `hits` or more successes in `n` Bernoulli trials at rate `p_chance`."""
    if n <= 0 or not np.isfinite(p_chance) or p_chance <= 0.0 or hits <= 0:
        return 1.0

    from math import exp, lgamma, log

    p = min(float(p_chance), 1.0)
    if p >= 1.0:
        return 1.0

    # Summed through the log pmf: pooling twelve folds of 700 queries overflows the binomial coefficient alone.
    log_p, log_q, log_n = log(p), log(1.0 - p), lgamma(n + 1)
    below = sum(
        exp(log_n - lgamma(i + 1) - lgamma(n - i + 1) + i * log_p + (n - i) * log_q) for i in range(min(hits, n + 1))
    )
    return float(np.clip(1.0 - below, 0.0, 1.0))


def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap `(mean, lo, hi)` of a per-query statistic."""
    if values.size == 0:
        return (float('nan'), float('nan'), float('nan'))

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def build_scoreboard(
    word_emb: np.ndarray,
    word_meta: 'pd.DataFrame',
    comparison: list[dict[str, Any]],
    sent_emb: np.ndarray,
    sent_content_ids: np.ndarray,
    sent_meta: 'pd.DataFrame | None',
    config: Any | None,
    word_band_power: np.ndarray | None = None,
    *,
    sent_n_words: np.ndarray | None = None,
    phase_sent_emb: np.ndarray | None = None,
    generation: dict[str, Any] | None = None,
    rescoring: dict[str, Any] | None = None,
    length_tol: int = 1,
    postprocess_fit: str | None = None,
) -> dict[str, Any]:
    """Assembles the honest scoreboard from already-computed evaluation artefacts.

    Args:
        word_emb (np.ndarray): Word embeddings `(n_words, d)`.
        word_meta (pd.DataFrame): Aligned word metadata.
        comparison (list[dict[str, Any]]): Probe-comparison rows.
        sent_emb (np.ndarray): Sentence embeddings `(n_sentences, d)`.
        sent_content_ids (np.ndarray): Stimulus id per sentence `(n_sentences,)`.
        sent_meta (pd.DataFrame | None): Aligned sentence metadata (needs `subject`).
        config (Any | None): The run config, consulted for the split and the held-out subject.
        word_band_power (np.ndarray | None, optional): Raw band power for the positive control.
        sent_n_words (np.ndarray | None, optional): Word count per sentence, enabling the
            length-stratified gallery. Falls back to a word-count column on `sent_meta`.
        phase_sent_emb (np.ndarray | None, optional): Sentence embeddings of phase-scrambled EEG through
            the identical encoder -- the same-path control for held-out retrieval.
        generation (dict[str, Any] | None, optional): A `generation.generation_report` block.
        rescoring (dict[str, Any] | None, optional): A `decoder_rescoring_retrieval` block.
        length_tol (int, optional): Word-count tolerance for the stratified gallery. Defaults to 1.
        postprocess_fit (str | None, optional): What the retrieval geometry was fitted on (`none` /
            `train split` / `transductive`), stamped inside `held_out_retrieval` so it travels with the number.

    Returns:
        dict[str, Any]: The scoreboard, with a `held_out_*` block per readout that was supplied.
    """
    holdout = holdout_subject(config)

    # A factored model's thought code is the content subspace, so the headline is judged on those dims.
    model_cfg = getattr(config, 'model', None)
    if model_cfg is not None and getattr(model_cfg, 'factored', False):
        cdim = int(getattr(model_cfg, 'content_dim', word_emb.shape[1]))
        word_emb = np.asarray(word_emb)[:, :cdim]
        sent_emb = np.asarray(sent_emb)[:, :cdim]

    board: dict[str, Any] = {
        'is_loso': holdout is not None,
        'holdout_subject': holdout,
        'factored': bool(model_cfg is not None and getattr(model_cfg, 'factored', False)),
        'lift_over_raw': lift_over_raw(comparison),
    }

    # Replace the whitening-corrupted fallback control with the genuinely-raw one when band power exists.
    control = raw_content_positive_control(word_band_power, word_meta)
    if control is not None:
        board['lift_over_raw']['content_probe'] = control

    # The LOSO block: geometry and cross-subject retrieval for the stranger alone.
    if holdout is not None:
        board['held_out_geometry'] = held_out_geometry(word_emb, word_meta, holdout)
        subjects = sent_meta['subject'].to_numpy() if sent_meta is not None and 'subject' in sent_meta.columns else None
        board['held_out_retrieval'] = (
            cross_subject_holdout_retrieval(sent_emb, sent_content_ids, subjects, holdout)
            if subjects is not None
            else None
        )
        n_words = _sentence_lengths(sent_n_words, sent_meta)
        if subjects is not None and board['held_out_retrieval'] is not None and n_words is not None:
            # A hit inside a matched-length gallery cannot be a sentence-length shortcut.
            from zte.evaluation.audit.rebaseline import stratified_retrieval

            # One unit on both sides: every reading stratifies on its stimulus's median word count.
            n_words = stimulus_median_lengths(n_words, np.asarray(sent_content_ids))
            board['held_out_retrieval']['length_stratified'] = stratified_retrieval(
                sent_emb, sent_content_ids, subjects, holdout, n_words, length_tol=length_tol
            )
        if board['held_out_retrieval'] is not None:
            # Provenance travels inside the block: what was fitted on what, and exactly which matrix was measured.
            board['held_out_retrieval']['postprocess_fit'] = postprocess_fit
            board['held_out_retrieval']['alignment_fit'] = getattr(
                getattr(config, 'dataset', None), 'raw_align_fit', None
            )
            board['held_out_retrieval']['embedding_checksum'] = embedding_checksum(sent_emb)
        if phase_sent_emb is not None and subjects is not None:
            # The control travels the identical retrieval path, so the comparison is not rigged.
            board['phase_control_retrieval'] = cross_subject_holdout_retrieval(
                phase_sent_emb, sent_content_ids, subjects, holdout
            )

    board['held_out_generation'] = held_out_generation(generation)
    board['decoder_rescoring_retrieval'] = rescoring
    return board


def _sentence_lengths(sent_n_words: np.ndarray | None, sent_meta: 'pd.DataFrame | None') -> np.ndarray | None:
    """Word count per sentence, from the explicit array or the first matching metadata column."""
    if sent_n_words is not None:
        return np.asarray(sent_n_words, dtype=np.float64).ravel()
    if sent_meta is None:
        return None
    for column in _LENGTH_COLUMNS:
        if column in sent_meta.columns:
            return np.asarray(sent_meta[column].to_numpy(), dtype=np.float64)
    return None


def render_markdown(board: dict[str, Any]) -> str:
    """Renders the scoreboard as a compact Markdown block for the top of the report."""
    lines = ['## Scoreboard — the honest headline', '']
    lines += _content_probe_lines(board['lift_over_raw'].get('content_probe', {}))

    if board['is_loso']:
        g = board.get('held_out_geometry') or {}
        r = board.get('held_out_retrieval') or {}
        lines.append(f'**Held-out subject:** `{board["holdout_subject"]}` (the away game).')
        lines.append('')
        if g:
            lines += [
                f'- Effective-rank ratio (held-out only): **{g.get("effective_rank_ratio", float("nan")):.3f}** '
                '(want high — not collapsed)',
                f'- Anisotropy (held-out only): **{g.get("anisotropy", float("nan")):.3f}** (want low — not a cone)',
                f'- Content variance budget (held-out only): **{_pct(g.get("content_variance"))}**',
            ]
        if r:
            n_q = r.get('n_queries') or 0
            ci = r.get('rank_percentile_ci') or (float('nan'),) * 3
            hits1, hits5 = round((r.get('top1') or 0) * n_q), round((r.get('top5') or 0) * n_q)

            # Rank percentile leads because every query contributes to it; Top-K goes out as raw hit counts.
            lines += [
                f'- **Rank percentile: {ci[0]:.4f}** (95% CI {ci[1]:.4f}–{ci[2]:.4f}, chance 0.5) '
                f'— the headline: it uses all {n_q} queries, not just the ones that landed.',
                f'- Top-1: **{hits1} hits / {n_q}** vs {n_q * (r.get("chance_top1") or 0):.1f} expected '
                f'by chance (p={r.get("top1_p", float("nan")):.1e})',
                f'- Top-5: **{hits5} hits / {n_q}** vs {n_q * 5 * (r.get("chance_top1") or 0):.1f} expected '
                f'by chance (p={r.get("top5_p", float("nan")):.1e})',
            ]
            if r.get('excluded_no_positive'):
                lines.append(
                    f'- Excluded: **{r["excluded_no_positive"]}** queries whose truth is absent from the '
                    'cross-subject gallery — dropped from every statistic above, never zero-scored.'
                )
            lines += _length_stratified_lines(r.get('length_stratified'))
            lines += _phase_control_lines(board.get('phase_control_retrieval'), r)
        lines.append('')

    decoding = _rescoring_lines(board.get('decoder_rescoring_retrieval')) + _generation_lines(
        board.get('held_out_generation')
    )
    if decoding:
        lines += decoding + ['']

    lines += [
        '**Lift over raw band-power** (ZTE - raw; positive = the encoder earns its place):',
        '',
    ]
    lines.append('| target | kind | ZTE | raw | lift | ')
    lines.append('| --- | --- | --- | --- | --- |')
    for t, v in board['lift_over_raw'].items():
        if t == 'content_probe':
            continue
        kind = 'content ▲' if v.get('is_content') else ('identity ▼' if v.get('is_identity') else '—')
        lines.append(
            f'| {t} | {kind} | {_num(v.get("zte_linear"))} | {_num(v.get("raw_linear"))} '
            f'| {_signed(v.get("lift_linear"))} |'
        )
    lines.append('')
    return '\n'.join(lines)


def _content_probe_lines(cp: dict[str, Any]) -> list[str]:
    """Markdown for the positive control: both clauses of `passes`, the empirical zero and the detectability floor."""
    ok = '✅ PASS' if cp.get('passes') else '❌ FAIL'
    best = cp.get('raw_content_r2_best')
    reading = (
        f'the {cp.get("best_is", "raw")} band-power probe scores R²={best:.4f}'
        if best is not None
        else str(cp.get('source'))
    )
    lines = [f'**Content-probe positive control:** {ok} ({reading}, floor {cp.get("floor")}).']

    clauses = cp.get('passes_clauses') or {}
    if clauses:
        stated = ', '.join(f'`{name}` {_clause(value)}' for name, value in clauses.items())
        lines += [
            '',
            f'Clauses (a pass needs a probed measurement *and* an established estimator): {stated} — '
            f'established by {cp.get("decided_by")}.',
        ]

    machinery = cp.get('machinery') or {}
    if machinery:
        works = '✅ works' if machinery.get('passes') else '❌ BROKEN'
        lines += [
            '',
            f'**Probe machinery check:** {works} — word length off {len(machinery.get("features", []))} '
            f'eye-tracking features at R²={machinery.get("word_len_r2", float("nan")):.3f} '
            f'(n={machinery.get("n")}, folds grouped on `{machinery.get("cv_groups")}`).',
        ]

    within = cp.get('within_subject_r2') or {}
    nulls = cp.get('permutation_null') or {}
    if within or nulls or cp.get('per_target_r2'):
        n_perm = next((block['n_perm'] for block in nulls.values()), 0)
        lines += [
            '',
            f'Folds grouped on `{cp.get("cv_groups")}` over {cp.get("n_probed")} rows, so no stimulus is both '
            'trained on and scored.',
            '',
            '| band-power probe | word_len | log_freq |',
            '| --- | --- | --- |',
        ]
        for label, cell in (
            ('pooled over subjects', cp.get('per_target_r2') or {}),
            ('within subject (the honest one)', within),
            (f'permuted target (empirical zero, {n_perm} draws)', cp.get('shuffled_target_r2') or {}),
        ):
            lines.append(
                f'| {label} | {cell.get("word_len", float("nan")):.4f} | {cell.get("log_freq", float("nan")):.4f} |'
            )
        if nulls:
            lines.append(
                f'| empirical zero, 95% interval | {_interval(nulls, "word_len")} | {_interval(nulls, "log_freq")} |'
            )

    verdict = cp.get('detectability_verdict') or {}
    if verdict:
        detect = cp.get('detectability') or {}
        floor_r2 = detect.get('floor_r2')
        planted = (
            f'R²={floor_r2:.4f}, reached by a signal planted at SNR {detect.get("floor_snr")}'
            if floor_r2 is not None
            else 'not established'
        )
        lines += [
            '',
            f'**Detectability floor:** {planted} — calibrated on the same {detect.get("n_features")} features over '
            f'{detect.get("n")} rows with the same folds. {verdict.get("statement")}',
        ]

    if cp.get('finding'):
        lines += ['', f'**Reading:** {cp["finding"]}']

    lines.append('')
    return lines


def _clause(value: bool | None) -> str:
    """A `passes` clause as a mark, with `None` reading as `not applicable` rather than as a failure."""
    return '—' if value is None else ('✅' if value else '❌')


def _interval(nulls: dict[str, Any], target: str) -> str:
    """The permutation null's 95% interval for one target, or a dash when it was not probed."""
    block = nulls.get(target)
    if not block:
        return '—'

    lo, hi = block['interval']
    return f'[{lo:.4f}, {hi:.4f}]'


def _length_stratified_lines(block: dict[str, Any] | None) -> list[str]:
    """Markdown for the matched-length gallery, where a hit cannot be a sentence-length shortcut."""
    if not block:
        return []
    ci = block.get('rank_percentile_ci') or (float('nan'),) * 3
    excluded = int(block.get('excluded_no_positive') or 0)
    suffix = f' ({excluded} unanswerable queries excluded, never zero-scored.)' if excluded else ''
    return [
        f'- Length-stratified (|Δn_words| ≤ {block.get("length_tol")}, mean gallery '
        f'{block.get("mean_gallery", float("nan")):.1f}, chance '
        f'{block.get("chance_top1", float("nan")):.4f}): Top-1 '
        f'**{block.get("top1", float("nan")):.4f}**, rank percentile {ci[0]:.4f} '
        f'(95% CI {ci[1]:.4f}–{ci[2]:.4f}) — sentence length alone must not explain the hit.' + suffix
    ]


def _phase_control_lines(block: dict[str, Any] | None, real: dict[str, Any]) -> list[str]:
    """Markdown for the phase-scrambled control run through the identical retrieval path."""
    if not block:
        return []
    delta = _sub(real.get('rank_percentile'), block.get('rank_percentile'))
    return [
        f'- Phase-scrambled control (same encoder, same gallery): rank percentile '
        f'{block.get("rank_percentile", float("nan")):.4f}, Top-1 '
        f'{block.get("top1", float("nan")):.4f} — real minus control {_signed(delta)}.'
    ]


def _rescoring_lines(block: dict[str, Any] | None) -> list[str]:
    """Markdown for decoder-rescoring retrieval, labelled retrieval rather than generation."""
    if not block:
        return []
    ci = block.get('rank_percentile_ci') or (float('nan'),) * 3
    lines = [
        '',
        f'**Decoder-rescoring retrieval** ({block.get("n_queries", 0)} queries over a '
        f'{block.get("n_gallery", 0)}-sentence gallery, chance '
        f'{block.get("chance_top1", float("nan")):.4f}). Forced choice over a known candidate set: '
        'this is retrieval, directly comparable to the number above, and is never a generation claim.',
        '',
        f'- Top-1 **{block.get("top1", float("nan")):.4f}** '
        f'(p={block.get("top1_p", float("nan")):.1e}), rank percentile {ci[0]:.4f} '
        f'(95% CI {ci[1]:.4f}–{ci[2]:.4f})',
    ]
    lines += _length_stratified_lines(block.get('length_stratified'))
    return lines


def _generation_lines(block: dict[str, Any] | None) -> list[str]:
    """Markdown for free-running generation: deltas against every control, never an absolute score."""
    if not block or not block.get('applicable'):
        return []
    metric = block.get('primary_metric', 'content_f1')
    hypothesis = block.get('hypothesis') or {}
    free = block.get('n_candidate_sentences') is None
    lines = [
        '',
        f'**Free-running generation** ({block.get("n", 0)} held-out readings, `{block.get("split")}` '
        f'cell of `{block.get("split_strategy")}`, '
        f'{"no candidate set" if free else "CONSTRAINED DECODE"}). '
        f'Absolute {metric} is {hypothesis.get(metric, float("nan")):.4f} and means nothing on its '
        'own -- a decoder reciting the corpus scores the same. Only the paired deltas below are readable.',
        '',
        '| control | delta | 95% CI | beats |',
        '| --- | --- | --- | --- |',
    ]
    for name, delta in (block.get('deltas') or {}).items():
        if not delta:
            continue
        lines.append(
            f'| {name} | {delta.get("point", float("nan")):+.4f} '
            f'| [{delta.get("lo", float("nan")):+.4f}, {delta.get("hi", float("nan")):+.4f}] '
            f'| {"✓" if delta.get("beats") else "·"} |'
        )
    for name, reason in sorted((block.get('controls_absent') or {}).items()):
        lines.append(f'| {name} | NEVER RAN ({reason}) | -- | · |')
    perm_p = block.get('permutation_p')
    kl = block.get('prefix_influence_kl')
    lines += [
        '',
        f'- Beats every control: **{block.get("beats_all_controls")}** '
        f'(worst: `{block.get("worst_control")}`, requested: '
        f'{", ".join(f"`{c}`" for c in block.get("controls_requested") or []) or "none"})',
        f'- Permutation p (pairing shuffled): {"n/a" if perm_p is None else f"{perm_p:.4f}"} '
        f'| prefix-influence KL: {"n/a" if kl is None else f"{kl:.4f}"} nats',
    ]
    quarantined = block.get('quarantined') or []
    if quarantined:
        lines.append(
            f'- Quarantined diagnostics (computed, never read by the verdict): '
            f'{", ".join(f"`{k}`" for k in quarantined)}'
        )
    return lines


def _num(v: float | None) -> str:
    """Formats a score, or a dash when missing."""
    return '—' if v is None else f'{v:.3f}'


def _signed(v: float | None) -> str:
    """Formats a delta with an explicit sign, or a dash when missing."""
    return '—' if v is None else f'{v:+.3f}'


def _pct(v: float | None) -> str:
    """Formats a fraction as a percentage, or a dash when missing."""
    return '—' if v is None else f'{100 * v:.1f}%'


def _signed_pct(v: float | None) -> str:
    """Formats a fraction as signed percentage points, or a dash when missing."""
    return '—' if v is None else f'{100 * v:+.2f}pp'
