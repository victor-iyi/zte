"""The anchor-calibration curve: what N labelled readings buy a brand-new reader, with no retraining."""

from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from zte.evaluation.audit.honesty import _procrustes
from zte.evaluation.audit.rebaseline import GALLERY_CONDITIONS, fit_postprocess, stratified_retrieval
from zte.evaluation.audit.scoreboard import _binom_tail_p, _bootstrap_ci, cross_subject_holdout_retrieval
from zte.logging_utils import get_logger

_LOG = get_logger('evaluation.audit.calibration')

type MapFamily = Literal['procrustes', 'ridge']
"""Which family a new reader's calibration map is solved in."""

# 0 is the control a clinic starts from and 200 is about the longest enrollment a person will sit through.
DEFAULT_ANCHOR_COUNTS: Final[tuple[int, ...]] = (0, 10, 25, 50, 100, 200)
"""Anchor counts the curve is swept over."""

# Centring spends one pair and a rotation needs a spread to turn, so below this the fitted map is noise.
MIN_ANCHORS: Final[int] = 3
"""Fewest anchor pairs a calibration may be fitted from."""

# `stratified_retrieval` refuses fewer than two queries, and a curve point with one query says nothing anyway.
MIN_QUERY_STIMULI: Final[int] = 2
"""Non-anchor stimuli the held-out reader must retain for a reduced gallery to be scoreable."""

ARMS: Final[tuple[str, ...]] = ('uncalibrated', 'calibrated', 'shuffled')
"""The three arms scored on one identical gallery at every anchor count."""

CURVE_METRICS: Final[tuple[str, ...]] = ('top1', 'top5', 'top10', 'mrr', 'rank_percentile', 'mean_rank')
"""Metrics averaged over anchor draws at every point on the curve."""

# `cross_subject_holdout_retrieval` materialises a float64 n x n similarity; 12k readings is already ~1.1 GB.
MAX_SCOREBOARD_ROWS: Final[int] = 12_000
"""Readings above which the scoreboard cross-check is skipped rather than risking the machine."""

# Draws share their queries, so the same three hits reappear in each: pooling them would multiply the evidence.
_P_BASIS: Final[str] = 'one draw of queries -- the draws share their queries, so pooling the hits would double count'
"""What the exact binomial tails at each curve point are computed over."""


# --------------------------------------------------------------------------- #
# The fitted map
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True, kw_only=True)
class SubjectCalibration:
    """A new reader's map into the cohort frame, with its parameters frozen at fit time.

    Note:
        Both families are affine in the same shape -- centre on the anchor mean, apply the matrix, land on the
        cohort mean -- and differ only in how the matrix is solved, which is what makes them a bracket: an
        orthogonal rotation cannot inflate a score by rescaling, and a ridge map is free to.
    """

    family: MapFamily
    matrix: np.ndarray
    query_mean: np.ndarray
    reference_mean: np.ndarray
    n_anchors: int
    underdetermined: bool

    def __call__(self, emb: np.ndarray) -> np.ndarray:
        """Maps `(n, d)` held-out rows into the cohort frame, returning float32."""
        x = np.asarray(emb, dtype=np.float64) - self.query_mean

        return (x @ self.matrix + self.reference_mean).astype(np.float32)


def fit_calibration(
    anchor_query: np.ndarray,
    anchor_reference: np.ndarray,
    *,
    family: MapFamily = 'procrustes',
    ridge_alpha: float = 1.0,
) -> SubjectCalibration | None:
    """Fits one new reader's map from their anchor readings onto the cohort prototypes of the same sentences.

    Note:
        `None` means the reader stays uncalibrated, and it is logged rather than swallowed: a thin fit that
        quietly degrades into an identity map would report the uncalibrated number as a calibration result.

    Args:
        anchor_query (np.ndarray): The new reader's anchor embeddings `(k, d)`.
        anchor_reference (np.ndarray): Cross-subject prototype of the same `k` stimuli `(k, d)`.
        family (MapFamily, optional): `'procrustes'` for the rotation-only map, `'ridge'` for the regularised
            affine one. Defaults to 'procrustes'.
        ridge_alpha (float, optional): Ridge penalty, ignored by the Procrustes family. Defaults to 1.0.

    Returns:
        SubjectCalibration | None: The frozen map, or `None` when there is too little to fit one from.

    Raises:
        ValueError: If `family` is neither `'procrustes'` nor `'ridge'`.
    """
    q = np.asarray(anchor_query, dtype=np.float64)
    r = np.asarray(anchor_reference, dtype=np.float64)
    if q.ndim != 2 or r.ndim != 2 or q.shape != r.shape:
        _LOG.warning('Anchor pairs are shaped %s against %s; leaving the reader uncalibrated.', q.shape, r.shape)
        return None

    if len(q) < MIN_ANCHORS:
        _LOG.warning('Only %d anchor pairs, below the %d needed; leaving the reader uncalibrated.', len(q), MIN_ANCHORS)
        return None

    try:
        match family:
            case 'procrustes':
                matrix, q_mean, r_mean = _procrustes(q, r)
            case 'ridge':
                # The intercept is the centring itself, so both families share one `__call__`.
                q_mean, r_mean = q.mean(axis=0), r.mean(axis=0)
                qc, rc = q - q_mean, r - r_mean
                gram = qc.T @ qc + float(ridge_alpha) * np.eye(q.shape[1])
                matrix = np.linalg.solve(gram, qc.T @ rc)
            case unknown:
                raise ValueError(f'Unknown map family {unknown!r}; expected procrustes or ridge.')
    except np.linalg.LinAlgError as exc:
        _LOG.warning('Calibration fit failed (%s: %s); leaving the reader uncalibrated.', type(exc).__name__, exc)
        return None

    return SubjectCalibration(
        family=family,
        matrix=np.ascontiguousarray(matrix, dtype=np.float64),
        query_mean=np.ascontiguousarray(q_mean, dtype=np.float64),
        reference_mean=np.ascontiguousarray(r_mean, dtype=np.float64),
        n_anchors=int(len(q)),
        underdetermined=bool(len(q) <= q.shape[1]),
    )


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #


def _stimulus_pairs(
    emb: np.ndarray, content_ids: np.ndarray, hold_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per stimulus the new reader shares with the cohort: their own reading, and the others' prototype."""
    ids: list[Any] = []
    query: list[np.ndarray] = []
    reference: list[np.ndarray] = []
    for sid in np.unique(content_ids[hold_mask]):
        here = hold_mask & (content_ids == sid)
        others = (~hold_mask) & (content_ids == sid)
        if not others.any():
            continue
        ids.append(sid)
        query.append(emb[here].mean(axis=0))
        reference.append(emb[others].mean(axis=0))

    if not ids:
        empty = np.zeros((0, emb.shape[1]), dtype=np.float64)
        return np.asarray([]), empty, empty

    return (
        np.asarray(ids),
        np.asarray(query, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )


def _derangement(k: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of `k` indices with no fixed point, so no shuffled anchor keeps its true partner."""
    if k < 2:
        return np.arange(k)

    for _ in range(8):
        perm = rng.permutation(k)
        if not np.any(perm == np.arange(k)):
            return perm

    return (np.arange(k) + 1) % k


def _score_arm(
    emb: np.ndarray,
    content_ids: np.ndarray,
    subjects: np.ndarray,
    holdout: str,
    lengths: np.ndarray | None,
    keep: np.ndarray,
    hold_mask: np.ndarray,
    calibration: SubjectCalibration | None,
    *,
    length_tol: int,
    ks: tuple[int, ...],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Scores one arm on the reduced gallery, mapping the new reader's rows and leaving the cohort's alone."""
    mapped = emb
    if calibration is not None:
        mapped = emb.copy()
        mapped[hold_mask] = calibration(emb[hold_mask])

    n_gallery = float(np.sum(keep & ~hold_mask))
    out: dict[str, Any] = dict.fromkeys(GALLERY_CONDITIONS)
    for gallery in GALLERY_CONDITIONS:
        words: np.ndarray | None = None
        if gallery == 'length_stratified':
            if lengths is None:
                continue
            words = lengths[keep]

        cell = stratified_retrieval(
            mapped[keep],
            content_ids[keep],
            subjects[keep],
            holdout,
            words,
            length_tol=length_tol,
            ks=ks,
            n_boot=n_boot,
            seed=seed,
        )
        if cell is not None:
            cell['n_gallery'] = n_gallery
        out[gallery] = cell

    return out


def _aggregate(cells: list[dict[str, Any] | None], *, ks: tuple[int, ...], n_boot: int, seed: int) -> dict[str, Any]:
    """Mean over anchor draws with a bootstrap CI, because which sentences the reader got is real variance."""
    valid = [c for c in cells if c]
    if not valid:
        return {}

    out: dict[str, Any] = {'n_draws': len(valid)}
    for metric in CURVE_METRICS:
        values = np.asarray([float(c[metric]) for c in valid if metric in c], dtype=np.float64)
        if not values.size:
            continue
        point, lo, hi = _bootstrap_ci(values, n_boot=n_boot, seed=seed)
        out[metric] = point
        out[f'{metric}_ci'] = [point, lo, hi]

    for field in ('chance_top1', 'n_queries', 'n_gallery', 'mean_gallery', 'excluded_no_positive'):
        values = np.asarray([float(c[field]) for c in valid if field in c], dtype=np.float64)
        if values.size:
            out[field] = float(values.mean())

    n_queries = int(round(out.get('n_queries', 0.0)))
    chance = float(out.get('chance_top1', float('nan')))
    for k in ks:
        hits = round(float(out.get(f'top{k}', 0.0)) * n_queries)
        out[f'top{k}_p'] = _binom_tail_p(hits, n_queries, chance * k)
    out['p_basis'] = _P_BASIS
    out['per_draw_top1'] = [float(c['top1']) for c in valid if 'top1' in c]
    out['per_draw_rank_percentile'] = [float(c['rank_percentile']) for c in valid if 'rank_percentile' in c]

    return out


def _paired_delta(
    left: list[dict[str, Any] | None],
    right: list[dict[str, Any] | None],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, list[float]]:
    """Per-draw difference between two arms scored on the identical gallery, with a bootstrap CI over draws."""
    out: dict[str, list[float]] = {}
    for metric in CURVE_METRICS:
        deltas = [
            float(a[metric]) - float(b[metric])
            for a, b in zip(left, right, strict=True)
            if a and b and metric in a and metric in b
        ]
        if not deltas:
            continue
        point, lo, hi = _bootstrap_ci(np.asarray(deltas, dtype=np.float64), n_boot=n_boot, seed=seed)
        out[metric] = [point, lo, hi]

    return out


def calibration_curve(
    sent_emb: np.ndarray,
    content_ids: np.ndarray,
    subjects: np.ndarray,
    holdout: str,
    n_words: np.ndarray | None = None,
    *,
    anchor_counts: tuple[int, ...] = DEFAULT_ANCHOR_COUNTS,
    families: tuple[MapFamily, ...] = ('procrustes', 'ridge'),
    draws: int = 5,
    ridge_alpha: float = 1.0,
    postprocess: bool = False,
    whiten: bool = True,
    n_top: int = 1,
    length_tol: int = 1,
    ks: tuple[int, ...] = (1, 5, 10),
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Held-out retrieval against the number of sentences the new reader was labelled on, with no retraining.

    The reader supplies `n` anchor readings whose text is known. A map is fitted from those readings alone onto
    the cross-subject prototype of the same sentences -- the target a deployed system can actually build, since
    it needs no labels from the new reader beyond the anchors themselves -- and applied to their remaining
    readings. Every anchor stimulus is dropped from the queries **and** from the gallery: an anchor left in the
    gallery is a labelled answer sitting in the answer key, and it alone would manufacture the whole curve.

    Note:
        Three arms are scored on one identical reduced gallery per draw. `calibrated` is the fitted map,
        `uncalibrated` the same rows untouched -- the gallery-matched control, and the only honest comparator,
        because the raw `n = 0` point is a larger gallery and therefore a different problem. `shuffled` fits the
        same map on anchor readings paired with the wrong sentences: if it lifts as much, the lift is the
        transform's capacity rather than calibration, and the curve is not a result.

    This is a diagnostic and never raises: an uncomputable cell is `None` with its reason in `errors`.

    Args:
        sent_emb (np.ndarray): Sentence embeddings `(n, d)`, one row per reading.
        content_ids (np.ndarray): Stimulus id per reading `(n,)`.
        subjects (np.ndarray): Subject code per reading `(n,)`.
        holdout (str): The new reader -- the held-out subject whose readings are the queries.
        n_words (np.ndarray | None, optional): Word count per reading `(n,)`, enabling the length-stratified
            gallery. Defaults to None, which reports the full gallery alone.
        anchor_counts (tuple[int, ...], optional): The sweep. Defaults to `DEFAULT_ANCHOR_COUNTS`.
        families (tuple[MapFamily, ...], optional): Map families to fit. Defaults to both.
        draws (int, optional): Seeded anchor draws per count. Defaults to 5.
        ridge_alpha (float, optional): Ridge penalty. Defaults to 1.0.
        postprocess (bool, optional): Fit whitening and all-but-the-top on the cohort rows first, making the
            numbers comparable to the train-fitted rebaseline cell. Defaults to False.
        whiten (bool, optional): Whether that post-processing whitens. Defaults to True.
        n_top (int, optional): Leading directions all-but-the-top removes. Defaults to 1.
        length_tol (int, optional): Word-count tolerance of the stratified gallery. Defaults to 1.
        ks (tuple[int, ...], optional): Top-K cut-offs. Defaults to (1, 5, 10).
        n_boot (int, optional): Bootstrap resamples behind every interval. Defaults to 2000.
        seed (int, optional): Seed for the anchor draws and the bootstrap. Defaults to 0.

    Returns:
        dict[str, Any]: `{'applicable', 'holdout', 'n_readings', 'n_stimuli', 'n_anchor_candidates',
            'postprocess_fit', 'headline_metric', 'headline_gallery', 'curve', 'shuffled_control', 'detail',
            'series', 'scoreboard_baseline', 'verdict', 'errors'}`. `curve` is the flat plottable list, one
            record per family and anchor count on the headline gallery; `detail` carries both galleries, all
            three arms and the paired deltas.
    """
    emb = np.asarray(sent_emb, dtype=np.float32)
    content_ids = np.asarray(content_ids)
    subjects = np.asarray(subjects)
    lengths = None if n_words is None else np.asarray(n_words, dtype=np.float64).ravel()
    errors: dict[str, str] = {}

    hold_mask = subjects == str(holdout)
    header: dict[str, Any] = {
        'holdout': str(holdout),
        'n_readings': int(len(emb)),
        'n_stimuli': int(np.unique(content_ids).size),
        'dim': int(emb.shape[1]) if emb.ndim == 2 else 0,
        'draws': int(draws),
        'families': [str(f) for f in families],
        'anchor_counts': [int(c) for c in anchor_counts],
        'length_tol': int(length_tol),
        'ridge_alpha': float(ridge_alpha),
        'seed': int(seed),
    }
    if np.unique(subjects).size < 2 or int(hold_mask.sum()) < MIN_QUERY_STIMULI:
        return header | {
            'applicable': False,
            'postprocess_fit': 'none',
            'curve': [],
            'shuffled_control': {},
            'detail': {},
            'series': {},
            'scoreboard_baseline': None,
            'verdict': {},
            'errors': {'setup': f'need >= 2 subjects and >= {MIN_QUERY_STIMULI} readings from {holdout!r}'},
        }

    # Fitted on the cohort alone, so the transform a curve point is measured under never saw the new reader.
    postprocess_fit = 'none'
    if postprocess:
        try:
            emb = fit_postprocess(emb[~hold_mask], whiten=whiten, n_top=n_top)(emb)
            postprocess_fit = 'train split'
        except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover - defensive
            errors['postprocess'] = f'{type(exc).__name__}: {exc}'

    ids, anchor_query, anchor_reference = _stimulus_pairs(emb, content_ids, hold_mask)
    counts = sorted({int(c) for c in anchor_counts if int(c) >= 0})

    detail: dict[str, Any] = {}
    for n in counts:
        point = _curve_point(
            emb,
            content_ids,
            subjects,
            str(holdout),
            lengths,
            hold_mask,
            ids,
            anchor_query,
            anchor_reference,
            n=n,
            families=families,
            draws=draws,
            ridge_alpha=ridge_alpha,
            length_tol=length_tol,
            ks=ks,
            n_boot=n_boot,
            seed=seed,
            errors=errors,
        )
        if point is not None:
            detail[str(n)] = point

    baseline: dict[str, Any] | None = None
    if len(emb) > MAX_SCOREBOARD_ROWS:
        errors['scoreboard_baseline'] = f'{len(emb)} readings above the {MAX_SCOREBOARD_ROWS} n x n cross-check cap'
    else:
        try:
            baseline = cross_subject_holdout_retrieval(emb, content_ids, subjects, str(holdout), ks=ks)
        except (ValueError, IndexError, MemoryError) as exc:  # pragma: no cover - defensive
            errors['scoreboard_baseline'] = f'{type(exc).__name__}: {exc}'

    headline = 'full' if lengths is None else 'length_stratified'
    flat = _flat_curve(detail, families, headline)

    return header | {
        'applicable': bool(detail),
        'n_anchor_candidates': int(ids.size),
        'postprocess_fit': postprocess_fit,
        'headline_metric': 'rank_percentile',
        'headline_gallery': headline,
        'curve': flat,
        'shuffled_control': _shuffled_control(flat),
        'detail': detail,
        'series': _series(detail, families),
        'scoreboard_baseline': baseline,
        'verdict': _verdict(detail, families, headline),
        'errors': errors,
    }


def _curve_point(
    emb: np.ndarray,
    content_ids: np.ndarray,
    subjects: np.ndarray,
    holdout: str,
    lengths: np.ndarray | None,
    hold_mask: np.ndarray,
    ids: np.ndarray,
    anchor_query: np.ndarray,
    anchor_reference: np.ndarray,
    *,
    n: int,
    families: tuple[MapFamily, ...],
    draws: int,
    ridge_alpha: float,
    length_tol: int,
    ks: tuple[int, ...],
    n_boot: int,
    seed: int,
    errors: dict[str, str],
) -> dict[str, Any] | None:
    """One anchor count: every draw's three arms on its own reduced gallery, then aggregated across draws."""
    used = min(int(n), int(ids.size))
    if used and ids.size - used < MIN_QUERY_STIMULI:
        errors[f'n={n}'] = f'{ids.size} shared stimuli leave fewer than {MIN_QUERY_STIMULI} queries after anchoring'
        return None

    # With no anchors there is nothing to draw, so one pass is the whole story.
    n_draws = max(1, int(draws)) if used else 1
    control_cells: dict[str, list[dict[str, Any] | None]] = {gallery: [] for gallery in GALLERY_CONDITIONS}
    scored_cells: dict[str, dict[str, dict[str, list[dict[str, Any] | None]]]] = {
        str(family): {arm: {gallery: [] for gallery in GALLERY_CONDITIONS} for arm in ARMS[1:]} for family in families
    }

    underdetermined = False
    degraded = 0
    for draw in range(n_draws):
        rng = np.random.default_rng([int(seed), int(n), int(draw)])
        chosen = rng.choice(ids.size, size=used, replace=False) if used else np.zeros(0, dtype=int)
        keep = np.ones(len(emb), dtype=bool) if not used else ~np.isin(content_ids, ids[chosen])

        try:
            scored = _score_arm(
                emb,
                content_ids,
                subjects,
                holdout,
                lengths,
                keep,
                hold_mask,
                None,
                length_tol=length_tol,
                ks=ks,
                n_boot=n_boot,
                seed=seed,
            )
        except (ValueError, IndexError, MemoryError) as exc:  # pragma: no cover - defensive
            errors[f'n={n}/draw={draw}/uncalibrated'] = f'{type(exc).__name__}: {exc}'
            continue

        for gallery in GALLERY_CONDITIONS:
            control_cells[gallery].append(scored[gallery])

        q, r = anchor_query[chosen], anchor_reference[chosen]
        shuffled = _derangement(used, rng)
        for family in families:
            for arm, reference in (('calibrated', r), ('shuffled', r[shuffled] if used else r)):
                try:
                    calibration = (
                        fit_calibration(q, reference, family=family, ridge_alpha=ridge_alpha) if used else None
                    )
                    if used and calibration is None:
                        degraded += 1
                    underdetermined = underdetermined or bool(calibration is not None and calibration.underdetermined)
                    cells = _score_arm(
                        emb,
                        content_ids,
                        subjects,
                        holdout,
                        lengths,
                        keep,
                        hold_mask,
                        calibration,
                        length_tol=length_tol,
                        ks=ks,
                        n_boot=n_boot,
                        seed=seed,
                    )
                except (ValueError, IndexError, MemoryError, np.linalg.LinAlgError) as exc:
                    errors[f'n={n}/draw={draw}/{family}/{arm}'] = f'{type(exc).__name__}: {exc}'
                    cells = dict.fromkeys(GALLERY_CONDITIONS)

                # Every arm records one entry per draw, failures included, so the paired deltas stay aligned.
                for gallery in GALLERY_CONDITIONS:
                    scored_cells[str(family)][arm][gallery].append(cells[gallery])

    control = {g: _aggregate(control_cells[g], ks=ks, n_boot=n_boot, seed=seed) for g in GALLERY_CONDITIONS}
    out: dict[str, Any] = {
        'n_anchors_requested': int(n),
        'n_anchors_used': int(used),
        'n_draws': int(n_draws),
        'saturated': bool(used < int(n)),
        'degraded_fits': int(degraded),
        'underdetermined': bool(underdetermined),
        'uncalibrated': control,
        'control_note': 'The gallery-matched control: the same rows, the same reduced gallery, no map applied.',
        'families': {},
    }
    for family in families:
        arms = scored_cells[str(family)]
        block: dict[str, Any] = {
            arm: {g: _aggregate(arms[arm][g], ks=ks, n_boot=n_boot, seed=seed) for g in GALLERY_CONDITIONS}
            for arm in ARMS[1:]
        }
        for label, arm in (('lift', 'calibrated'), ('shuffled_lift', 'shuffled')):
            block[label] = {
                g: _paired_delta(arms[arm][g], control_cells[g], n_boot=n_boot, seed=seed) for g in GALLERY_CONDITIONS
            }
        block['calibrated_minus_shuffled'] = {
            g: _paired_delta(arms['calibrated'][g], arms['shuffled'][g], n_boot=n_boot, seed=seed)
            for g in GALLERY_CONDITIONS
        }
        out['families'][str(family)] = block

    return out


def _flat_curve(detail: dict[str, Any], families: tuple[MapFamily, ...], gallery: str) -> list[dict[str, Any]]:
    """One record per family and anchor count on the headline gallery -- the curve a plot or a claim row reads."""
    out: list[dict[str, Any]] = []
    for family in families:
        for count in sorted(detail, key=int):
            point = detail[count]
            block = (point.get('families') or {}).get(str(family), {})
            cal = (block.get('calibrated') or {}).get(gallery) or {}
            control = (point.get('uncalibrated') or {}).get(gallery) or {}
            shuffled = (block.get('shuffled') or {}).get(gallery) or {}
            lift = (block.get('lift') or {}).get(gallery, {})
            shuffled_lift = (block.get('shuffled_lift') or {}).get(gallery, {})
            margin = (block.get('calibrated_minus_shuffled') or {}).get(gallery, {})
            record: dict[str, Any] = {
                'family': str(family),
                'gallery': gallery,
                'n_anchors': point.get('n_anchors_used'),
                'n_anchors_requested': point.get('n_anchors_requested'),
                'n_draws': point.get('n_draws'),
                'saturated': point.get('saturated'),
                'underdetermined': point.get('underdetermined'),
                'degraded_fits': point.get('degraded_fits'),
                'uncalibrated_rank_percentile': control.get('rank_percentile'),
                'uncalibrated_top1': control.get('top1'),
                'shuffled_rank_percentile': shuffled.get('rank_percentile'),
                'shuffled_top1': shuffled.get('top1'),
                'lift': lift.get('rank_percentile', [None])[0],
                'lift_ci': lift.get('rank_percentile'),
                'shuffled_lift': shuffled_lift.get('rank_percentile', [None])[0],
                'shuffled_lift_ci': shuffled_lift.get('rank_percentile'),
                'margin_over_shuffled_ci': margin.get('rank_percentile'),
            }
            for field in ('rank_percentile', 'rank_percentile_ci', 'top1', 'top1_ci', 'top1_p', 'top5', 'top10'):
                record[field] = cal.get(field)
            for field in ('mrr', 'mean_rank', 'chance_top1', 'n_queries', 'n_gallery', 'mean_gallery'):
                record[field] = cal.get(field)
            out.append(record)

    return out


def _shuffled_control(curve: list[dict[str, Any]]) -> dict[str, Any]:
    """The best a shuffled-anchor map reached -- the floor the true map has to clear to mean anything."""
    scored = [r for r in curve if r.get('n_anchors') and r.get('shuffled_rank_percentile') is not None]
    if not scored:
        return {'measured': False, 'note': 'no shuffled-anchor arm could be scored'}

    best = max(scored, key=lambda r: float(r['shuffled_rank_percentile']))

    return {
        'measured': True,
        'best_rank_percentile': float(best['shuffled_rank_percentile']),
        'best_n_anchors': best.get('n_anchors'),
        'family': best.get('family'),
        'gallery': best.get('gallery'),
        'note': 'Anchor readings paired with the wrong sentences; whatever this lifts is transform capacity.',
    }


def _series(detail: dict[str, Any], families: tuple[MapFamily, ...]) -> dict[str, Any]:
    """The curve as plot-ready parallel arrays, one block per family and gallery."""
    counts = sorted(detail, key=int)
    out: dict[str, Any] = {}
    for family in families:
        out[str(family)] = {}
        for gallery in GALLERY_CONDITIONS:
            block: dict[str, Any] = {'anchor_counts': [int(c) for c in counts]}
            for metric in ('rank_percentile', 'top1'):
                block[f'uncalibrated_{metric}'] = [
                    ((detail[c].get('uncalibrated') or {}).get(gallery) or {}).get(metric) for c in counts
                ]
                for arm in ARMS[1:]:
                    block[f'{arm}_{metric}'] = [
                        (
                            ((detail[c].get('families') or {}).get(str(family), {}).get(arm) or {}).get(gallery) or {}
                        ).get(metric)
                        for c in counts
                    ]
            out[str(family)][gallery] = block

    return out


def _verdict(detail: dict[str, Any], families: tuple[MapFamily, ...], gallery: str) -> dict[str, Any]:
    """The honest reading per family: the best anchor count, its paired lift, and whether it clears the shuffle."""
    out: dict[str, Any] = {}
    for family in families:
        best_n: int | None = None
        best_lift = -np.inf
        for count in sorted(detail, key=int):
            if int(count) == 0:
                continue
            lift = ((detail[count].get('families') or {}).get(str(family), {}).get('lift') or {}).get(gallery, {})
            value = float(lift.get('rank_percentile', [float('nan')])[0])
            if np.isfinite(value) and value > best_lift:
                best_lift, best_n = value, int(count)

        if best_n is None:
            out[str(family)] = {'metric': 'rank_percentile', 'gallery': gallery, 'measured': False}
            continue

        block = (detail[str(best_n)].get('families') or {}).get(str(family), {})
        lift_ci = (block.get('lift') or {}).get(gallery, {}).get('rank_percentile', [float('nan')] * 3)
        shuffled_ci = (block.get('shuffled_lift') or {}).get(gallery, {}).get('rank_percentile', [float('nan')] * 3)
        margin_ci = (
            (block.get('calibrated_minus_shuffled') or {}).get(gallery, {}).get('rank_percentile', [float('nan')] * 3)
        )
        helps = bool(np.isfinite(lift_ci[1]) and lift_ci[1] > 0.0)
        beats = bool(np.isfinite(margin_ci[1]) and margin_ci[1] > 0.0)
        out[str(family)] = {
            'metric': 'rank_percentile',
            'gallery': gallery,
            'measured': True,
            'best_n_anchors': best_n,
            'lift': lift_ci[0],
            'lift_ci': list(lift_ci),
            'shuffled_lift': shuffled_ci[0],
            'shuffled_lift_ci': list(shuffled_ci),
            'margin_over_shuffled_ci': list(margin_ci),
            'helps': helps,
            'beats_shuffled': beats,
            'verdict': _verdict_line(helps, beats, best_n),
        }

    return out


def _verdict_line(helps: bool, beats_shuffled: bool, best_n: int) -> str:
    """One sentence stating what the curve supports, including when it supports nothing."""
    if not helps:
        return f'no measurable lift from calibration at any anchor count (best was {best_n})'

    if not beats_shuffled:
        return (
            f'lift at {best_n} anchors is matched by the shuffled-anchor control -- the transform capacity, '
            'not calibration'
        )

    return f'lift at {best_n} anchors clears the gallery-matched control and the shuffled-anchor control'


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def render_markdown(report: dict[str, Any]) -> str:
    """Renders the curve as the Markdown block that ships beside `calibration.json`.

    Args:
        report (dict[str, Any]): The dict from `calibration_curve`.

    Returns:
        str: A Markdown document with the per-family curve, the controls and the verdict.
    """
    gallery = str(report.get('headline_gallery', 'length_stratified'))
    lines = [
        f'# Anchor calibration -- new reader `{report.get("holdout")}`',
        '',
        f'{report.get("n_readings", 0)} readings over {report.get("n_stimuli", 0)} distinct sentences; '
        f'{report.get("n_anchor_candidates", 0)} of them are shared with the cohort and can serve as anchors. '
        f'Post-processing fit: `{report.get("postprocess_fit", "none")}`.',
        '',
        'Every anchor stimulus is excluded from the queries and from the gallery, so no curve point can be read '
        'off a labelled sentence still sitting in the answer key. `uncalibrated` is the gallery-matched control '
        '-- the same rows and the same reduced gallery with no map applied, which is the only comparator that '
        'holds the problem fixed; `shuffled` fits the same map on anchor readings paired with the wrong '
        'sentences.',
        '',
        f'Headline metric: **rank percentile** on the `{gallery}` gallery.',
        '',
    ]

    detail = report.get('detail') or {}
    counts = sorted(detail, key=int)
    for family in report.get('families') or []:
        lines += [
            f'## `{family}` map',
            '',
            '| anchors | uncalibrated | calibrated | shuffled | lift (95% CI) | Top-1 | n queries | n gallery |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |',
        ]
        for count in counts:
            point = detail[count]
            block = (point.get('families') or {}).get(family, {})
            control = (point.get('uncalibrated') or {}).get(gallery) or {}
            cal = (block.get('calibrated') or {}).get(gallery) or {}
            shuf = (block.get('shuffled') or {}).get(gallery) or {}
            lift = (block.get('lift') or {}).get(gallery, {}).get('rank_percentile', [float('nan')] * 3)
            lines.append(
                f'| {point.get("n_anchors_used", 0)} | {_fmt(control.get("rank_percentile"))} '
                f'| {_fmt(cal.get("rank_percentile"))} | {_fmt(shuf.get("rank_percentile"))} '
                f'| {_fmt(lift[0])} ({_fmt(lift[1])}–{_fmt(lift[2])}) | {_fmt(cal.get("top1"))} '
                f'| {_fmt(cal.get("n_queries"), 1)} | {_fmt(cal.get("n_gallery"), 1)} |'
            )
        verdict = (report.get('verdict') or {}).get(family, {})
        lines += ['', f'**Verdict:** {verdict.get("verdict", "not measured")}', '']

    baseline = report.get('scoreboard_baseline') or {}
    if baseline:
        lines += [
            '## Scoreboard cross-check',
            '',
            f'Uncalibrated held-out retrieval over the unreduced gallery -- the number `scoreboard.'
            f'held_out_retrieval` reports: Top-1 {_fmt(baseline.get("top1"))} against chance '
            f'{_fmt(baseline.get("chance_top1"))}, rank percentile {_fmt(baseline.get("rank_percentile"))} '
            f'over {baseline.get("n_queries", 0)} queries.',
            '',
        ]

    errors = report.get('errors') or {}
    if errors:
        lines += ['## Cells that could not be computed', '']
        lines += [f'- `{k}`: {v}' for k, v in errors.items()]
        lines.append('')

    return '\n'.join(lines)


def _fmt(value: Any, digits: int = 4) -> str:
    """Formats a score, or a dash when missing."""
    if value is None or not np.isfinite(float(value)):
        return '—'

    return f'{float(value):.{digits}f}'
