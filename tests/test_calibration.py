"""Tests for the anchor-calibration curve: the fitted map, the anchor exclusion and the shuffled-anchor control."""

import numpy as np
import pytest

from zte.evaluation.audit.calibration import (
    DEFAULT_ANCHOR_COUNTS,
    MIN_ANCHORS,
    SubjectCalibration,
    calibration_curve,
    fit_calibration,
    render_markdown,
)

_COHORT: tuple[str, ...] = ('ZAB', 'ZDM', 'ZKB', 'ZMG')
_HOLDOUT: str = 'ZNEW'


def _rotation(dim: int, seed: int) -> np.ndarray:
    """A random orthogonal matrix, so a known rotation can be planted and then recovered."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))

    return q


def _rotated_cohort(
    n_stimuli: int = 40,
    dim: int = 8,
    noise: float = 0.02,
    seed: int = 0,
    rotate_holdout: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A cohort reading fixed stimulus directions, plus one new reader whose frame is rotated away from theirs."""
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((n_stimuli, dim))
    lengths = rng.integers(5, 25, size=n_stimuli).astype(np.float64)
    rotation = _rotation(dim, seed + 1) if rotate_holdout else np.eye(dim)

    emb, content, subjects, words = [], [], [], []
    for code in (*_COHORT, _HOLDOUT):
        frame = directions @ rotation if code == _HOLDOUT else directions
        emb.append(frame + noise * rng.standard_normal(frame.shape))
        content.append(np.arange(n_stimuli))
        subjects += [code] * n_stimuli
        words.append(lengths)

    return (
        np.concatenate(emb).astype(np.float32),
        np.concatenate(content),
        np.array(subjects),
        np.concatenate(words),
        rotation,
    )


# --------------------------------------------------------------------------- #
# the fitted map
# --------------------------------------------------------------------------- #
def test_procrustes_recovers_a_planted_rotation() -> None:
    """Fitted on anchor pairs alone, the orthogonal map inverts the rotation that separated the new reader."""
    emb, content, subjects, _, rotation = _rotated_cohort(noise=0.0)
    hold = subjects == _HOLDOUT
    anchors = np.arange(12)

    query = np.stack([emb[hold & (content == s)].mean(axis=0) for s in anchors])
    reference = np.stack([emb[~hold & (content == s)].mean(axis=0) for s in anchors])
    calibration = fit_calibration(query, reference, family='procrustes')

    assert calibration is not None
    # The planted map sent cohort -> reader, so recovering it means the fit approximates its transpose.
    assert np.allclose(calibration.matrix, rotation.T, atol=1e-4)
    assert np.allclose(np.eye(emb.shape[1]), calibration.matrix @ calibration.matrix.T, atol=1e-6)

    held = emb[hold]
    assert np.linalg.norm(calibration(held) - emb[~hold][: len(held)]) < np.linalg.norm(held - emb[~hold][: len(held)])


def test_ridge_map_is_affine_and_more_expressive_than_the_rotation() -> None:
    """The ridge family fits a scaled frame the orthogonal family cannot, which is why the pair brackets calibration."""
    rng = np.random.default_rng(3)
    dim = 6
    reference = rng.standard_normal((30, dim))
    # A pure rescaling: an orthogonal map cannot express it, a ridge map can.
    query = reference * 4.0

    rotation = fit_calibration(query, reference, family='procrustes')
    ridge = fit_calibration(query, reference, family='ridge', ridge_alpha=1e-6)

    assert rotation is not None and ridge is not None
    assert ridge.family == 'ridge'
    assert np.linalg.norm(ridge(query) - reference) < np.linalg.norm(rotation(query) - reference)


def test_fit_returns_none_and_warns_rather_than_degrading_to_an_identity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Too few anchors leave the reader uncalibrated out loud -- a silent no-op would be read as a result."""
    rng = np.random.default_rng(0)
    pairs = rng.standard_normal((MIN_ANCHORS - 1, 5))

    with caplog.at_level('WARNING'):
        assert fit_calibration(pairs, pairs) is None
    assert 'uncalibrated' in caplog.text

    assert fit_calibration(pairs, rng.standard_normal((MIN_ANCHORS + 1, 5))) is None


def test_unknown_map_family_is_rejected() -> None:
    """A misspelled family raises rather than silently picking one."""
    pairs = np.random.default_rng(0).standard_normal((8, 4))
    with pytest.raises(ValueError, match='Unknown map family'):
        fit_calibration(pairs, pairs, family='affine')  # type: ignore[arg-type]


def test_calibration_is_frozen_and_reapplies_identically() -> None:
    """The map holds its parameters, so a row scored later is a function of the anchors alone."""
    rng = np.random.default_rng(1)
    query, reference = rng.standard_normal((20, 5)), rng.standard_normal((20, 5))
    calibration = fit_calibration(query, reference)

    assert isinstance(calibration, SubjectCalibration)
    with pytest.raises(AttributeError):
        calibration.n_anchors = 99  # type: ignore[misc]

    fresh = rng.standard_normal((7, 5))
    assert np.array_equal(calibration(fresh), calibration(fresh))
    assert calibration(fresh).dtype == np.float32


# --------------------------------------------------------------------------- #
# the honesty gate: anchors leave the gallery
# --------------------------------------------------------------------------- #
def test_anchor_stimuli_leave_both_the_query_set_and_the_gallery() -> None:
    """Every anchor is dropped from the queries and from the gallery, by exactly its own count."""
    emb, content, subjects, words, _ = _rotated_cohort(n_stimuli=30)
    report = calibration_curve(
        emb,
        content,
        subjects,
        _HOLDOUT,
        words,
        anchor_counts=(0, 10),
        families=('procrustes',),
        draws=2,
        n_boot=64,
    )

    baseline = next(r for r in report['curve'] if r['n_anchors'] == 0)
    anchored = next(r for r in report['curve'] if r['n_anchors'] == 10)

    assert baseline['n_queries'] == pytest.approx(30.0)
    assert anchored['n_queries'] == pytest.approx(20.0)
    # The cohort loses the same ten stimuli, once per member.
    assert baseline['n_gallery'] == pytest.approx(30.0 * len(_COHORT))
    assert anchored['n_gallery'] == pytest.approx(20.0 * len(_COHORT))


def test_uncalibrated_control_is_scored_on_the_same_reduced_gallery() -> None:
    """The control at each anchor count shares the calibrated arm's gallery, or the curve compares two problems."""
    emb, content, subjects, words, _ = _rotated_cohort(n_stimuli=30)
    report = calibration_curve(
        emb,
        content,
        subjects,
        _HOLDOUT,
        words,
        anchor_counts=(0, 12),
        families=('procrustes',),
        draws=2,
        n_boot=64,
    )

    for count in ('0', '12'):
        point = report['detail'][count]
        control = point['uncalibrated']['length_stratified']
        calibrated = point['families']['procrustes']['calibrated']['length_stratified']
        assert control['n_queries'] == calibrated['n_queries']
        assert control['n_gallery'] == calibrated['n_gallery']

    assert report['detail']['0']['uncalibrated']['full']['n_queries'] == pytest.approx(30.0)
    assert report['detail']['12']['uncalibrated']['full']['n_queries'] == pytest.approx(18.0)


# --------------------------------------------------------------------------- #
# the mutation test: a true map lifts, a shuffled one must not
# --------------------------------------------------------------------------- #
def test_calibration_lifts_retrieval_and_the_shuffled_control_does_not() -> None:
    """A recoverable rotation lifts held-out retrieval; the same map on wrong pairings does not."""
    emb, content, subjects, words, _ = _rotated_cohort(n_stimuli=48, dim=8, noise=0.05, seed=7)
    report = calibration_curve(
        emb,
        content,
        subjects,
        _HOLDOUT,
        words,
        anchor_counts=(0, 16),
        families=('procrustes',),
        draws=4,
        n_boot=256,
    )

    point = report['detail']['16']['families']['procrustes']
    lift = point['lift']['length_stratified']['rank_percentile']
    shuffled = point['shuffled_lift']['length_stratified']['rank_percentile']
    margin = point['calibrated_minus_shuffled']['length_stratified']['rank_percentile']

    assert lift[1] > 0.0, 'a recoverable rotation must lift the honest metric'
    assert shuffled[0] < lift[0], 'the shuffled-anchor map must not match the true one'
    assert margin[1] > 0.0, 'the true map must clear the shuffled control, not merely clear zero'

    verdict = report['verdict']['procrustes']
    assert verdict['helps'] and verdict['beats_shuffled']
    assert 'clears' in verdict['verdict']


def test_no_lift_when_the_new_reader_already_shares_the_cohort_frame() -> None:
    """With nothing to correct, calibration reports a null -- not a win manufactured by the transform."""
    emb, content, subjects, words, _ = _rotated_cohort(n_stimuli=40, noise=0.05, seed=11, rotate_holdout=False)
    report = calibration_curve(
        emb,
        content,
        subjects,
        _HOLDOUT,
        words,
        anchor_counts=(0, 12),
        families=('procrustes',),
        draws=4,
        n_boot=256,
    )

    lift = report['detail']['12']['families']['procrustes']['lift']['length_stratified']['rank_percentile']
    assert lift[1] <= 0.0, 'an unrotated reader has no calibration lift to find'


def test_shuffled_control_is_reported_at_the_top_level() -> None:
    """The permutation control travels with the payload, so a lift can never be read without its floor."""
    emb, content, subjects, words, _ = _rotated_cohort(n_stimuli=30)
    report = calibration_curve(
        emb,
        content,
        subjects,
        _HOLDOUT,
        words,
        anchor_counts=(0, 10),
        families=('ridge',),
        draws=2,
        n_boot=64,
    )

    control = report['shuffled_control']
    assert control['measured'] is True
    assert control['best_n_anchors'] == 10
    assert 0.0 <= control['best_rank_percentile'] <= 1.0


# --------------------------------------------------------------------------- #
# the payload
# --------------------------------------------------------------------------- #
def test_report_carries_the_postprocess_fit_and_the_headline_it_is_read_on() -> None:
    """Every retrieval number ships with how post-processing was fitted and which gallery it is quoted on."""
    emb, content, subjects, words, _ = _rotated_cohort(n_stimuli=24)
    plain = calibration_curve(
        emb, content, subjects, _HOLDOUT, words, anchor_counts=(0, 8), families=('ridge',), draws=1, n_boot=32
    )
    fitted = calibration_curve(
        emb,
        content,
        subjects,
        _HOLDOUT,
        words,
        anchor_counts=(0, 8),
        families=('ridge',),
        draws=1,
        n_boot=32,
        postprocess=True,
    )

    assert plain['postprocess_fit'] == 'none'
    assert fitted['postprocess_fit'] == 'train split'
    assert plain['headline_metric'] == 'rank_percentile'
    assert plain['headline_gallery'] == 'length_stratified'


def test_full_gallery_only_when_no_word_counts_are_supplied() -> None:
    """Without word counts the length-stratified column is absent rather than silently duplicating the full one."""
    emb, content, subjects, _, _ = _rotated_cohort(n_stimuli=24)
    report = calibration_curve(
        emb, content, subjects, _HOLDOUT, None, anchor_counts=(0, 8), families=('procrustes',), draws=1, n_boot=32
    )

    assert report['headline_gallery'] == 'full'
    assert report['detail']['8']['uncalibrated']['length_stratified'] == {}
    assert report['detail']['8']['uncalibrated']['full']['n_queries'] == pytest.approx(16.0)


def test_a_single_subject_is_reported_as_inapplicable_rather_than_raising() -> None:
    """An uncomputable sweep returns its reason; this is a diagnostic and never raises."""
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((12, 4)).astype(np.float32)
    report = calibration_curve(emb, np.arange(12), np.array(['ZAB'] * 12), 'ZAB', np.full(12, 7.0))

    assert report['applicable'] is False
    assert 'setup' in report['errors']
    assert report['curve'] == []


def test_saturated_and_impossible_anchor_counts_are_flagged_not_silently_shrunk() -> None:
    """Asking for more anchors than exist is recorded, and asking for all of them leaves no curve point."""
    emb, content, subjects, words, _ = _rotated_cohort(n_stimuli=12)
    report = calibration_curve(
        emb,
        content,
        subjects,
        _HOLDOUT,
        words,
        anchor_counts=(0, 8, 200),
        families=('procrustes',),
        draws=1,
        n_boot=32,
    )

    assert report['detail']['8']['saturated'] is False
    assert '200' not in report['detail']
    assert any('n=200' in key for key in report['errors'])


def test_markdown_names_every_arm_and_the_exclusion() -> None:
    """The rendered report states the anchor exclusion, both controls and the verdict."""
    emb, content, subjects, words, _ = _rotated_cohort(n_stimuli=30)
    report = calibration_curve(emb, content, subjects, _HOLDOUT, words, anchor_counts=(0, 10), draws=2, n_boot=64)
    text = render_markdown(report)

    assert 'Anchor calibration' in text and _HOLDOUT in text
    assert 'excluded from the queries and from the gallery' in text
    assert '`procrustes` map' in text and '`ridge` map' in text
    assert 'Verdict:' in text
    assert 'rank percentile' in text


def test_default_sweep_is_the_published_anchor_ladder() -> None:
    """The deliverable's x-axis is fixed, so two runs of the curve are comparable."""
    assert DEFAULT_ANCHOR_COUNTS == (0, 10, 25, 50, 100, 200)
