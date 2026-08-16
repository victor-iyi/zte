"""Tests for the length-confound audit: the length-only oracle, the bit budget and the non-transductive fit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from zte.evaluation.audit.rebaseline import (
    GALLERY_CONDITIONS,
    POSTPROCESS_CONDITIONS,
    bit_budget,
    fit_postprocess,
    length_oracle,
    rebaseline_report,
    render_markdown,
    stratified_retrieval,
)
from zte.evaluation.report import _postprocess

# A distribution whose strata are 3 / 2 / 1, so every oracle score has a closed form.
_LENGTHS: np.ndarray = np.array([1, 1, 1, 2, 2, 3], dtype=np.int64)

_SUBJECTS: tuple[str, ...] = ('ZAB', 'ZDM', 'ZKB')


def _cohort(
    n_stimuli: int = 30,
    dim: int = 16,
    noise: float = 0.0,
    seed: int = 0,
    codes: tuple[str, ...] = _SUBJECTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A cohort of subjects reading the same stimuli, each stimulus a fixed direction plus per-subject noise."""
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((n_stimuli, dim)).astype(np.float32)
    lengths = rng.integers(4, 30, size=n_stimuli).astype(np.float64)

    emb, content, subjects, words = [], [], [], []
    for code in codes:
        emb.append(directions + noise * rng.standard_normal(directions.shape).astype(np.float32))
        content.append(np.arange(n_stimuli))
        subjects += [code] * n_stimuli
        words.append(lengths)
    return (
        np.concatenate(emb),
        np.concatenate(content),
        np.array(subjects),
        np.concatenate(words),
    )


# --------------------------------------------------------------------------- #
# the floor
# --------------------------------------------------------------------------- #
def test_length_oracle_reproduces_its_closed_form() -> None:
    """The oracle's scores are exact expectations over a uniform ordering inside the stratum, not a simulation."""
    out = length_oracle(_LENGTHS, tol=0)
    assert out['top1'] == pytest.approx(0.5)
    assert out['top5'] == pytest.approx(1.0)
    assert out['top10'] == pytest.approx(1.0)
    assert out['mrr'] == pytest.approx(13 / 18)
    assert out['rank_percentile'] == pytest.approx(5.2 / 6)
    assert out['mean_rank'] == pytest.approx(10 / 6)
    assert out['chance_top1'] == pytest.approx(1 / 6)
    assert out['mean_stratum'] == pytest.approx(14 / 6)
    assert (out['min_stratum'], out['max_stratum'], out['n']) == (1.0, 3.0, 6.0)


def test_a_looser_oracle_knows_less() -> None:
    """Widening the tolerance widens every stratum, so Top-1 and the rank percentile can only fall."""
    tight = length_oracle(_LENGTHS, tol=0)
    loose = length_oracle(_LENGTHS, tol=1)
    assert loose['mean_stratum'] > tight['mean_stratum']
    assert loose['top1'] < tight['top1']
    assert loose['rank_percentile'] < tight['rank_percentile']


def test_length_oracle_survives_an_empty_gallery() -> None:
    """A diagnostic never raises; an empty gallery reports `nan` scores instead."""
    out = length_oracle(np.array([], dtype=np.int64))
    assert out['n'] == 0.0
    assert math.isnan(out['top1'])


def test_bit_budget_is_the_entropy_arithmetic() -> None:
    """`bits_from_length` is `H(identity) - H(identity | n_words)` and `bits_from_eeg` is `log2(n / mean_rank)`."""
    out = bit_budget(_LENGTHS, mean_rank=2.0, n_gallery=6)
    conditional = 0.5 * math.log2(3) + (1 / 3) * math.log2(2)
    assert out['bits_needed'] == pytest.approx(math.log2(6))
    assert out['entropy_identity_given_length'] == pytest.approx(conditional)
    assert out['bits_from_length'] == pytest.approx(math.log2(6) - conditional)
    assert out['bits_from_eeg'] == pytest.approx(math.log2(3))
    assert out['ratio'] == pytest.approx(math.log2(3) / math.log2(6))


def test_bit_budget_reports_nothing_when_the_encoder_has_no_rank() -> None:
    """Without a mean rank there is no encoder term, and the ratio must be `nan` rather than zero."""
    out = bit_budget(_LENGTHS, n_gallery=6)
    assert math.isnan(out['bits_from_eeg']) and math.isnan(out['ratio'])


# --------------------------------------------------------------------------- #
# retrieval inside a length stratum
# --------------------------------------------------------------------------- #
def test_a_perfect_embedding_ranks_first_in_both_galleries() -> None:
    """A noiseless cohort puts every query's own stimulus at rank 1, whether or not the gallery is length-matched."""
    emb, content, subjects, words = _cohort(noise=0.0)
    for lengths in (None, words):
        out = stratified_retrieval(emb, content, subjects, 'ZAB', lengths, n_boot=200)
        assert out is not None
        assert out['top1'] == pytest.approx(1.0)
        assert out['rank_percentile'] == pytest.approx(1.0)
        assert out['mrr'] == pytest.approx(1.0)


def test_a_random_embedding_sits_at_the_middle_of_the_ranking() -> None:
    """A null cohort's rank percentile must bracket 0.5, or the headline metric is not calibrated.

    Two subjects, so each query has exactly one cross-subject positive; a third subject would give the query two
    chances at rank 1 and pull the null above 0.5 for a reason that has nothing to do with the embedding.
    """
    rng = np.random.default_rng(0)
    _, content, subjects, words = _cohort(n_stimuli=60, codes=('ZAB', 'ZDM'))
    emb = rng.standard_normal((len(subjects), 16)).astype(np.float32)
    out = stratified_retrieval(emb, content, subjects, 'ZAB', words, n_boot=500)
    assert out is not None
    _, lo, hi = out['rank_percentile_ci']
    assert lo < 0.5 < hi, out['rank_percentile_ci']
    assert out['top1_p'] > 0.05


def test_the_stratified_gallery_is_smaller_and_length_matched() -> None:
    """Restricting to matched word counts is what stops a hit being a sentence-length shortcut."""
    emb, content, subjects, words = _cohort(n_stimuli=60, noise=0.5)
    full = stratified_retrieval(emb, content, subjects, 'ZAB', None, n_boot=200)
    strat = stratified_retrieval(emb, content, subjects, 'ZAB', words, length_tol=1, n_boot=200)
    assert full is not None and strat is not None
    assert strat['mean_gallery'] < full['mean_gallery']
    assert full['length_tol'] is None and strat['length_tol'] == 1
    assert strat['chance_top1'] > full['chance_top1']


def test_retrieval_declines_a_single_subject_cohort() -> None:
    """With one subject there is no cross-subject query set, and the audit reports `None` rather than inventing one."""
    emb, content, _, words = _cohort(n_stimuli=10)
    subjects = np.array(['ZAB'] * len(emb))
    assert stratified_retrieval(emb, content, subjects, 'ZAB', words) is None


def test_a_query_with_no_reachable_positive_is_excluded_and_counted() -> None:
    """An unanswerable query is dropped from the statistic and reported, never scored as a zero-percentile miss."""
    emb, content, subjects, _ = _cohort(n_stimuli=10, codes=('ZAB', 'ZDM'))
    keep = ~((subjects == 'ZDM') & (content == 0))

    out = stratified_retrieval(emb[keep], content[keep], subjects[keep], 'ZAB', None, n_boot=100)
    assert out is not None
    assert out['excluded_no_positive'] == 1
    assert out['n_queries'] == 9
    assert out['rank_percentile'] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# the non-transductive fit
# --------------------------------------------------------------------------- #
def test_the_fitted_transform_does_not_move_when_a_holdout_row_does() -> None:
    """The proof that the fit is not transductive: perturbing held-out data cannot change a train-fitted transform."""
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((120, 8)).astype(np.float32)
    train, held = rows[:90], rows[90:].copy()

    fitted = fit_postprocess(train, whiten=True, n_top=1)
    before = fitted(np.concatenate([train, held]))

    held[0] += 50.0
    after = fitted(np.concatenate([train, held]))

    assert fitted.n_train == 90
    assert np.array_equal(before[:90], after[:90])
    assert np.array_equal(fitted.mean, fit_postprocess(train, whiten=True, n_top=1).mean)


def test_the_transductive_path_does_move(monkeypatch: pytest.MonkeyPatch) -> None:
    """The published post-processing is fitted on every subject, so one held-out row shifts every other row."""
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((120, 8)).astype(np.float32)
    train, held = rows[:90], rows[90:].copy()

    before = _postprocess(np.concatenate([train, held]), None, True, 1)
    held[0] += 50.0
    after = _postprocess(np.concatenate([train, held]), None, True, 1)
    assert not np.allclose(before[:90], after[:90])

    # The same call with a train split named is the non-transductive path and is immovable.
    fixed_before = _postprocess(np.concatenate([train, held]), train, True, 1)
    held[1] += 50.0
    fixed_after = _postprocess(np.concatenate([train, held]), train, True, 1)
    assert np.array_equal(fixed_before[:90], fixed_after[:90])


def test_an_unfittable_split_degrades_to_the_identity_shift() -> None:
    """Fewer than two training rows cannot define a covariance, and the transform must pass data through."""
    fitted = fit_postprocess(np.zeros((1, 4), dtype=np.float32))
    assert fitted.inv_sqrt is None and fitted.directions is None
    assert np.array_equal(fitted(np.ones((3, 4), dtype=np.float32)), np.ones((3, 4)))


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def test_rebaseline_report_fills_every_cell_of_the_grid() -> None:
    """All three post-processing conditions x both galleries, plus the oracle floor and the bit budget."""
    emb, content, subjects, words = _cohort(n_stimuli=40, noise=0.4)
    report = rebaseline_report(emb, content, subjects, 'ZAB', words, oracle_tols=(0, 1, 2, 4), n_boot=200)

    assert set(report['grid']) == set(POSTPROCESS_CONDITIONS)
    for cond in POSTPROCESS_CONDITIONS:
        assert set(report['grid'][cond]) == set(GALLERY_CONDITIONS)
        for gallery in GALLERY_CONDITIONS:
            assert report['grid'][cond][gallery] is not None, (cond, gallery)
    assert report['errors'] == {}
    assert report['n_readings'] == len(emb)
    assert report['n_stimuli'] == 40
    assert report['n_train_rows'] == int((subjects != 'ZAB').sum())
    assert sorted(report['length_oracle']) == ['0', '1', '2', '4']
    assert report['bit_budget']['bits_needed'] == pytest.approx(math.log2(40))


def test_the_train_fitted_and_transductive_rows_are_genuinely_different_fits() -> None:
    """The contrast the whole grid exists to draw: the two rows must be fitted on different data.

    Collapsing `train_fitted` onto the scored rows makes it a second copy of `transductive`, which would hide the very
    leak the audit reports. The rows are compared numerically, so an audit that reported one fit twice would fail here
    rather than read as agreement between two conditions.
    """
    # Noisy enough that neither condition saturates at a rank percentile of 1.0, where any two fits agree.
    emb, content, subjects, words = _cohort(n_stimuli=40, noise=1.2)
    report = rebaseline_report(emb, content, subjects, 'ZAB', words, n_boot=200)

    for gallery in GALLERY_CONDITIONS:
        fitted = report['grid']['train_fitted'][gallery]
        transductive = report['grid']['transductive'][gallery]
        assert fitted is not None and transductive is not None
        assert fitted['rank_percentile'] != transductive['rank_percentile'], gallery


def test_the_floor_comparison_is_recorded_and_gates_nothing() -> None:
    """Gate G0 is a diagnostic here: the comparison is reported as a number and never raises or blocks."""
    emb, content, subjects, words = _cohort(n_stimuli=40, noise=0.4)
    floor = rebaseline_report(emb, content, subjects, 'ZAB', words, n_boot=200)['floor_comparison']
    assert floor['condition'] == 'train_fitted'
    assert floor['gallery'] == 'length_stratified'
    assert isinstance(floor['clears_floor'], bool)
    assert 'gates nothing' in floor['note']


def test_rebaseline_report_never_raises_on_a_cohort_it_cannot_score() -> None:
    """A single-subject cohort yields empty cells with their reason, because an audit that crashes audits nothing."""
    emb, content, _, words = _cohort(n_stimuli=8)
    subjects = np.array(['ZAB'] * len(emb))
    report = rebaseline_report(emb, content, subjects, 'ZAB', words, n_boot=100)
    assert all(
        report['grid'][cond][gallery] is None for cond in POSTPROCESS_CONDITIONS for gallery in GALLERY_CONDITIONS
    )
    assert math.isnan(report['bit_budget']['bits_from_eeg'])


def test_the_markdown_carries_the_grid_the_floor_and_the_budget() -> None:
    """`rebaseline.md` ships beside the JSON, so the numbers are readable without loading the artifact."""
    emb, content, subjects, words = _cohort(n_stimuli=40, noise=0.4)
    text = render_markdown(rebaseline_report(emb, content, subjects, 'ZAB', words, n_boot=200))
    assert '# Length-confound audit -- held-out subject `ZAB`' in text
    for cond in POSTPROCESS_CONDITIONS:
        assert f'| {cond} | full |' in text
        assert f'| {cond} | length_stratified |' in text
    assert '## Length-only oracle -- the floor' in text
    assert '## Bit budget' in text
