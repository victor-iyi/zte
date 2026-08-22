"""The decoder capacity certification: exact menu accuracies, paired controls and the bits ledger."""

from typing import Any

import numpy as np
import pytest

from zte.evaluation.audit.capacity import (
    CLAUSE_NAMES,
    ENTROPY_IDENTITY_GIVEN_LENGTH,
    _contiguous_k,
    _mi_bits,
    _sign_test_p,
    capacity_markdown_lines,
    capacity_report,
    pooled_capacity,
)

# Enough queries that an exact sign test over unanimous wins clears any sane alpha, and enough gallery
# that a pool still holds 8 candidates.
GALLERY: int = 16
"""Distinct stimuli in every hand-built gallery below."""


def _arrays(n_gallery: int = GALLERY) -> dict[str, np.ndarray]:
    """One training reading and one holdout reading of each stimulus, all the same length and task."""
    content = np.concatenate([np.arange(n_gallery), np.arange(n_gallery)])
    subjects = np.array(['A'] * n_gallery + ['B'] * n_gallery)
    n_words = np.full(2 * n_gallery, 9.0)
    tasks = np.array(['NR'] * (2 * n_gallery))

    return {'content_ids': content, 'subjects': subjects, 'n_words': n_words, 'tasks': tasks}


def _run(arms: dict[str, np.ndarray], **overrides: Any) -> dict[str, Any]:
    """Runs `capacity_report` on the standard synthetic arrays with an honest split."""
    kwargs: dict[str, Any] = {
        'ks': (2, 4, 8),
        'n_perm': 200,
        'n_boot': 200,
        'honest_split': True,
        'split_strategy': 'by_subject_and_stimulus',
        'split_cell': 'test',
    }
    kwargs.update(overrides)
    arrays = _arrays(next(iter(arms.values())).shape[1])
    report = capacity_report(
        arms,
        arrays['content_ids'],
        arrays['subjects'],
        'B',
        arrays['n_words'],
        tasks=arrays['tasks'],
        **kwargs,
    )
    assert report is not None

    return report


def _perfect(n: int = GALLERY) -> np.ndarray:
    """A score matrix where every query's true sentence strictly beats the whole gallery."""
    return np.eye(n, dtype=np.float64)


def _strong_arms() -> dict[str, np.ndarray]:
    """A model that always wins against three controls that never do -- the certifying case."""
    return {
        'model': _perfect(),
        'length_only': np.zeros((GALLERY, GALLERY)),
        'shuffled_eeg': np.zeros((GALLERY, GALLERY)),
        'mismatch': np.zeros((GALLERY, GALLERY)),
    }


def _with_beaten(beaten: list[int]) -> np.ndarray:
    """A score matrix whose query `i` has its true sentence strictly beat exactly `beaten[i]` distractors."""
    n = len(beaten)
    scores = np.zeros((n, n))
    for row, wins in enumerate(beaten):
        others = [column for column in range(n) if column != row]
        scores[row, others] = np.arange(n - 1, dtype=np.float64)
        scores[row, row] = wins - 0.5

    return scores


def test_accuracy_reproduces_the_closed_form() -> None:
    """Menu accuracy is the exact expectation over uniform distractors, with no distractor sampling."""
    # Three queries whose true sentence beats 2, 1 and 0 of the other two candidates.
    scores = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    arms = {'model': scores}

    report = _run(arms, ks=(2, 3), n_perm=0)
    per_k = report['scores']['pmi']['length_task_matched']['per_k']

    assert per_k['2']['accuracy'] == pytest.approx((1.0 + 0.5 + 0.0) / 3.0)
    assert per_k['3']['accuracy'] == pytest.approx(1.0 / 3.0)

    # No seed reaches the accuracy itself; only the CI around it is resampled.
    reseeded = _run(arms, ks=(2, 3), n_perm=0, seed=987654)['scores']['pmi']['length_task_matched']['per_k']
    assert reseeded['2']['accuracy'] == per_k['2']['accuracy']
    assert reseeded['3']['accuracy'] == per_k['3']['accuracy']


def test_a_constant_score_matrix_scores_zero_not_chance() -> None:
    """Ties lose, so a decoder that separates nothing scores zero rather than chance."""
    report = _run({'model': np.zeros((GALLERY, GALLERY))})
    per_k = report['scores']['pmi']['length_task_matched']['per_k']

    assert [per_k[str(k)]['accuracy'] for k in (2, 4, 8)] == [0.0, 0.0, 0.0]


def test_chance_is_exactly_one_over_k() -> None:
    """Every cell states chance as exactly 1/K, the exact expectation over uniform distractors."""
    report = _run(_strong_arms())
    per_k = report['scores']['pmi']['length_task_matched']['per_k']

    assert [per_k[str(k)]['chance'] for k in (2, 4, 8)] == [0.5, 0.25, 0.125]


def test_certified_k_is_the_largest_contiguous_pass_not_the_last() -> None:
    """A capacity holds only if every smaller menu size also holds, so a gap ends the sweep."""
    per_k = {'2': {'certified': True}, '4': {'certified': False}, '8': {'certified': True}}

    assert _contiguous_k(per_k, (2, 4, 8)) == 2


def test_a_point_estimate_above_chance_does_not_certify_without_the_ci() -> None:
    """Certification reads the CI lower bound, so a point estimate over 1/K on a handful of queries is not enough."""
    # Nine of sixteen queries win outright: 0.5625 sits above chance, its interval does not.
    beaten = [GALLERY - 1] * 9 + [0] * (GALLERY - 9)
    arms = dict(_strong_arms(), model=_with_beaten(beaten))

    cell = _run(arms, ks=(2,))['scores']['pmi']['length_task_matched']['per_k']['2']

    assert cell['accuracy'] == pytest.approx(9.0 / 16.0)
    assert cell['accuracy'] > cell['chance']
    assert cell['ci'][1] < cell['chance']
    assert 'above_chance' in cell['failed_clauses']


def test_a_control_is_beaten_per_query_not_on_average() -> None:
    """The control comparison is paired, so a steady per-query margin certifies where two marginals would not."""
    # The model beats its control by exactly two ranks on every single query, but both arms range so widely
    # that comparing their marginal accuracies could not resolve the gap.
    model_beaten = [8 + (i % 8) for i in range(GALLERY)]
    arms = {
        'model': _with_beaten(model_beaten),
        'length_only': _with_beaten([b - 2 for b in model_beaten]),
        'shuffled_eeg': _with_beaten([b - 2 for b in model_beaten]),
        'mismatch': _with_beaten([b - 2 for b in model_beaten]),
    }
    report = _run(arms, ks=(2,))
    block = report['scores']['pmi']['length_task_matched']
    paired = block['per_k']['2']['paired']['length_only']

    assert paired['model_wins'] == GALLERY and paired['control_wins'] == 0
    assert paired['delta'] == pytest.approx(2.0 / 15.0)
    assert paired['ci'][1] == pytest.approx(2.0 / 15.0)
    assert block['certified_k'] == 2


def test_a_perfect_but_length_only_matrix_certifies_nothing() -> None:
    """Perfect accuracy that a length-matched prefix reproduces exactly is not evidence of decoding."""
    perfect = _perfect()
    arms = {
        'model': perfect,
        'length_only': perfect.copy(),
        'shuffled_eeg': np.zeros((GALLERY, GALLERY)),
        'mismatch': np.zeros((GALLERY, GALLERY)),
    }
    report = _run(arms)
    block = report['scores']['pmi']['length_task_matched']
    cell = block['per_k']['2']

    # It fails for the right reason: the accuracy is perfect and the other two controls are beaten.
    assert cell['accuracy'] == pytest.approx(1.0)
    assert 'beats_shuffled_paired' not in cell['failed_clauses']
    assert 'beats_mismatch_paired' not in cell['failed_clauses']

    paired = cell['paired']['length_only']
    assert (paired['model_wins'], paired['control_wins']) == (0, 0)
    assert paired['ties'] == GALLERY
    assert paired['sign_test_p'] == 1.0
    assert paired['ci'][1] == 0.0

    assert 'beats_length_only_paired' in cell['failed_clauses']
    assert block['certified_k'] is None
    assert report['certified_k'] is None
    assert report['verdict']['capacity_certified'] is False


def test_a_dishonest_split_cannot_certify() -> None:
    """A split that is not by_subject_and_stimulus/test forecloses certification whatever the numbers say."""
    report = _run(_strong_arms(), split_cell='val')
    block = report['scores']['pmi']['length_task_matched']

    assert block['certifiable'] is False
    assert 'honest_split' in block['per_k']['2']['failed_clauses']
    assert block['certified_k'] is None
    assert report['certified_k'] is None


def test_open_flavor_is_never_certifiable() -> None:
    """The open pool lets word count legitimately help, so it may be reported but never certified."""
    report = _run(_strong_arms())
    open_block = report['scores']['pmi']['open']

    assert open_block['certifiable'] is False
    assert open_block['certified_k'] is None
    assert 'flavor_certifiable' in open_block['per_k']['2']['failed_clauses']

    # It is still scored, so the diagnostic remains readable beside the certified pool.
    assert open_block['per_k']['2']['accuracy'] == pytest.approx(1.0)


def test_sign_test_matches_the_hand_computed_binomial() -> None:
    """The exact sign test is the two-sided binomial tail over discordant pairs."""
    assert _sign_test_p(7, 8) == 2 * 9 / 256
    assert _sign_test_p(0, 0) == 1.0
    assert _sign_test_p(4, 8) == 1.0


def test_perm_p_floor_travels_with_every_cell() -> None:
    """Every cell carries the attainable p-floor, and the Markdown renders a floored p as a bound."""
    report = _run(_strong_arms(), n_perm=200)
    block = report['scores']['pmi']['length_task_matched']

    for per_k in (block['per_k'], block['common_subset']):
        for cell in per_k.values():
            assert cell['perm_p_floor'] == pytest.approx(1.0 / 201.0)

    assert block['per_k']['2']['perm_p'] == pytest.approx(1.0 / 201.0)
    assert any('<4.98e-03' in line for line in capacity_markdown_lines(report))


def test_mi_bits_at_perfect_accuracy_is_log2_k() -> None:
    """A channel that never errs carries the whole menu, log2 K bits."""
    assert _mi_bits(1.0, 8) == pytest.approx(3.0)
    assert _mi_bits(1.0, 64) == pytest.approx(6.0)


def test_mi_bits_at_chance_is_zero() -> None:
    """A channel at 1/K carries nothing, whatever the menu size."""
    assert _mi_bits(0.25, 4) == pytest.approx(0.0)
    assert _mi_bits(1.0 / 64.0, 64) == pytest.approx(0.0)
    assert _mi_bits(0.5, 2) == pytest.approx(0.0)


def test_ledger_denominator_is_the_residual_after_length() -> None:
    """Bits are credited against the identity left once word count is known, never the full 9.4512."""
    report = _run(_strong_arms())
    bits = report['bits']

    assert report['certified_k'] == 8
    assert bits['estimator'] == 'log2(certified K)'
    assert bits['bits_certified'] == pytest.approx(3.0)
    assert bits['entropy_identity_given_length'] == ENTROPY_IDENTITY_GIVEN_LENGTH == 4.3090
    assert bits['bits_from_length'] == 5.1422
    assert bits['fraction_of_residual'] == pytest.approx(3.0 / 4.3090)
    assert bits['bits_unrecovered'] == pytest.approx(4.3090 - 3.0)
    assert bits['bits_mi_confusion'] == pytest.approx(3.0)
    assert report['verdict']['capacity_clauses'] == dict.fromkeys(CLAUSE_NAMES, True)


def test_pooled_capacity_takes_the_min_over_seeds_and_none_if_any_fails() -> None:
    """A pooled capacity is what every run keeps, so the smallest wins and one failure sinks it."""
    strong = _run(_strong_arms())
    weaker = dict(strong, certified_k=4, holdout='C')

    pooled = pooled_capacity([strong, weaker])
    assert pooled['certified_k'] == 4
    assert pooled['bits_certified'] == pytest.approx(2.0)
    assert pooled['capacity_certified'] is True

    failing = dict(strong, certified_k=None, holdout='D')
    assert pooled_capacity([strong, weaker, failing])['certified_k'] is None
    assert pooled_capacity([])['certified_k'] is None


def test_the_tol_zero_length_oracle_is_identically_zero_and_gates_nothing() -> None:
    """On an exactly length-matched pool the distance oracle ties everywhere, so it can never be a clause."""
    report = _run(_strong_arms())
    block = report['scores']['pmi']['length_task_matched']

    assert block['length_oracle_2way_distance'] == 0.0
    assert block['gamed'] is False
    assert 'length_oracle' not in ' '.join(CLAUSE_NAMES)


def test_the_common_subset_curve_is_recomputed_not_copied() -> None:
    """Queries that drop out at a larger menu size cannot inflate a smaller one's certified capacity."""
    # A gallery of two length strata: 12 stimuli at 9 words, 4 at 3 words. Only the 12-strong stratum can
    # ever be scored at K = 8, so the common subset must be that stratum alone.
    n = GALLERY
    content = np.concatenate([np.arange(n), np.arange(n)])
    subjects = np.array(['A'] * n + ['B'] * n)
    n_words = np.concatenate([np.where(np.arange(n) < 12, 9.0, 3.0)] * 2)
    tasks = np.array(['NR'] * (2 * n))

    report = capacity_report(
        _strong_arms(),
        content,
        subjects,
        'B',
        n_words,
        tasks=tasks,
        ks=(2, 4, 8),
        n_perm=200,
        n_boot=200,
        honest_split=True,
        split_strategy='by_subject_and_stimulus',
        split_cell='test',
    )
    assert report is not None
    block = report['scores']['pmi']['length_task_matched']

    assert block['per_k']['2']['n_queries'] == 16
    assert block['per_k']['8']['n_queries'] == 12
    assert block['common_subset']['2']['n_queries'] == 12


def test_a_missing_control_arm_fails_its_clause() -> None:
    """An unavailable control is a failed clause, never a waived one."""
    arms = {k: v for k, v in _strong_arms().items() if k != 'mismatch'}
    report = _run(arms)
    cell = report['scores']['pmi']['length_task_matched']['per_k']['2']

    assert 'mismatch' not in cell['paired']
    assert 'beats_mismatch_paired' in cell['failed_clauses']
    assert report['certified_k'] is None


def test_the_report_names_menu_selection_and_carries_its_provenance() -> None:
    """The readout is menu selection, and the settings behind every number travel with it."""
    report = _run(_strong_arms(), seed=3)

    assert report['readout'] == 'menu selection'
    assert report['tie_policy'] == 'ties lose'
    assert report['headline'] == {'score': 'pmi', 'flavor': 'length_task_matched', 'alpha': 0.05}
    assert report['provenance']['arms_present'] == ['length_only', 'mismatch', 'model', 'shuffled_eeg']
    assert report['provenance']['ks'] == [2, 4, 8]
    assert report['provenance']['seed'] == 3
    assert report['n_queries'] == GALLERY and report['n_gallery'] == GALLERY


def test_a_raw_family_is_recognised_by_its_null_prefix_arm() -> None:
    """PMI cancels the null prefix identically, so an arm set carrying one can only be the raw family."""
    arms = dict(_strong_arms(), null_prefix=np.zeros((GALLERY, GALLERY)))
    report = _run(arms)

    assert set(report['scores']) == {'raw'}
    assert report['headline']['score'] == 'raw'
    assert 'null_prefix' in report['scores']['raw']['length_task_matched']['per_k']['2']['arms']


def test_a_family_without_a_model_arm_is_refused() -> None:
    """There is nothing to certify without the model arm, so the report refuses rather than guessing."""
    with pytest.raises(ValueError, match='no `model` arm'):
        _run({'shuffled_eeg': np.zeros((GALLERY, GALLERY))})


def test_markdown_reports_a_failed_certification_without_claiming_bits() -> None:
    """A run that certifies nothing renders an em dash, never a number the evidence does not support."""
    lines = capacity_markdown_lines(_run({'model': np.zeros((GALLERY, GALLERY))}))
    text = '\n'.join(lines)

    assert 'Certified menu size: **K = —**' in text
    assert 'menu selection' in text
    assert 'Fraction of that residual recovered: —' in text


def test_a_menu_size_no_pool_can_fill_is_unreachable_not_failed() -> None:
    """The default sweep reaches K=64, which no exact-length ZuCo pool fills, and that must not sink certification."""
    report = _run(_strong_arms(), ks=(2, 4, 8, 16, 32, 64))
    block = report['scores']['pmi']['length_task_matched']

    # A 16-stimulus gallery leaves 15 distractors, so K=32 and K=64 have no query to score at all.
    assert block['ks_feasible'] == [2, 4, 8, 16]
    assert block['ks_unreachable'] == [32, 64]
    assert block['per_k']['32']['n_queries'] == 0

    # Bounding the common subset by the unreachable K=64 would empty it and certify nothing.
    assert block['common_subset']['2']['n_queries'] > 0
    assert report['certified_k'] == 16
