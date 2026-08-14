"""Tests for the generation metrics, the permutation null and the control-gated generation verdict."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from zte.config import DecoderConfig
from zte.evaluation.audit.honesty import generation_permutation_test
from zte.evaluation.audit.scoreboard import held_out_generation
from zte.evaluation.generation import (
    bleu,
    content_word_f1,
    corpus_scores,
    generation_report,
    paired_delta,
    pairwise_metric,
    per_sentence_scores,
    quarantined_keys,
    rouge,
    sentence_bleu,
    strip_quarantined,
    tokenise,
    wer,
)
from zte.evaluation.report import _generation_section, _verdict

# Three fixed pairs: an exact match, an empty hypothesis and a partial overlap.
_HYPS: list[str] = ['The quick brown fox', '', 'the fox jumped high']
_REFS: list[str] = ['the quick brown fox!', 'a sentence with words', 'The quick brown fox']

# The corpus the cheating decoder recites, and the sentences it ignores.
_FREQUENT: str = 'the market closed higher on friday'
_OTHERS: list[str] = [
    'she walked to the station in the rain',
    'he opened the letter and started to read',
    'they built a bridge across the river',
]


def _cheating_corpus() -> tuple[list[str], list[str]]:
    """Returns 24 references, half of them the most frequent sentence, and the decode that always emits it."""
    references = [_FREQUENT] * 12
    for text in _OTHERS:
        references += [text] * 4
    return [_FREQUENT] * len(references), references


# Six distinct sentences, so a shuffled hypothesis-to-reference pairing rarely lands on a match.
_DISTINCT: list[str] = [
    'she walked to the station in the rain',
    'he opened the letter and started to read',
    'they built a bridge across the river',
    'the market closed higher on friday',
    'a small dog barked at the postman',
    'winter arrived early in the northern valleys',
]


def _generation_keys(verdict: dict[str, Any]) -> dict[str, Any]:
    """Returns the generation half of a verdict, which is the part a quarantined key could reach."""
    return {k: v for k, v in verdict.items() if k.startswith('generation')}


def _clauses(block: dict[str, Any], min_prefix_kl: float = 0.05) -> dict[str, Any]:
    """Returns the generation clauses a block produces, which is what the gate ANDs over."""
    verdict = _verdict([], {}, {}, generation=block, min_prefix_kl=min_prefix_kl)
    clauses: dict[str, Any] = verdict['generation_clauses']
    return clauses


def _generation_block(**overrides: Any) -> dict[str, Any]:
    """Builds the generation report of a decoder that ignores its conditioning vector entirely.

    Shaped exactly as `cli.decode.decode_evaluation` writes it: `split` is the cell, the strategy it was
    cut with is beside it, and the controls the run pre-registered are named whether or not they ran.
    """
    hypotheses, references = _cheating_corpus()
    names = ('mean_prefix', 'null_prefix', 'mismatch')
    controls = {name: list(hypotheses) for name in names}
    block = generation_report(
        hypotheses,
        references,
        controls,
        prefix_kl=1.0,
        n_candidate_sentences=None,
        split='test',
        n_boot=500,
        n_perm=200,
        seed=0,
    )
    block['split_strategy'] = 'by_subject_and_stimulus'
    block['controls_requested'] = list(names)
    block['controls_unavailable'] = {}
    block.update(overrides)
    return block


def _working_block(**overrides: Any) -> dict[str, Any]:
    """Builds the production-shaped report of a decoder that does read the brain and clears every clause."""
    references = _DISTINCT * 2
    names = ('mean_prefix', 'null_prefix', 'mismatch')
    controls = {name: [_FREQUENT] * len(references) for name in names}
    block = generation_report(
        list(references),
        references,
        controls,
        prefix_kl=1.0,
        n_candidate_sentences=None,
        split='test',
        n_boot=500,
        n_perm=200,
        seed=0,
    )
    block['split_strategy'] = 'by_subject_and_stimulus'
    block['controls_requested'] = list(names)
    block['controls_unavailable'] = {}
    block.update(overrides)
    return block


# --------------------------------------------------------------------------- #
# metrics against hand-computed values
# --------------------------------------------------------------------------- #
def test_tokenise_lowercases_and_drops_punctuation() -> None:
    """Case and punctuation are normalised away, so a decoder is not scored on typography."""
    assert tokenise("The Fox's tail, again!") == ['the', "fox's", 'tail', 'again']
    assert tokenise('!!!') == []


def test_corpus_bleu_matches_the_hand_computation() -> None:
    """Clipped modified precisions, the brevity penalty and the geometric mean all reproduce by hand."""
    out = bleu(_HYPS, _REFS)
    assert out['hyp_len'] == 8.0
    assert out['ref_len'] == 12.0
    penalty = math.exp(1.0 - 12.0 / 8.0)
    assert out['brevity_penalty'] == pytest.approx(penalty)
    assert out['precision1'] == pytest.approx(6 / 8)
    assert out['precision2'] == pytest.approx(3 / 6)
    assert out['precision3'] == pytest.approx(2 / 4)
    assert out['precision4'] == pytest.approx(1 / 2)
    assert out['bleu1'] == pytest.approx(penalty * 0.75)
    assert out['bleu4'] == pytest.approx(penalty * (0.75 * 0.5 * 0.5 * 0.5) ** 0.25)


def test_a_zero_precision_zeroes_the_higher_orders() -> None:
    """No 4-gram match means BLEU-4 is 0, which is the expected outcome here and must not be smoothed away."""
    out = bleu(['the fox'], ['the quick brown fox'])
    assert out['bleu1'] > 0.0
    assert out['bleu4'] == 0.0
    assert sentence_bleu('the fox', 'the quick brown fox') > 0.0  # add-1 smoothed


def test_rouge_matches_the_hand_computation() -> None:
    """ROUGE is the mean per-sentence F1, so the empty hypothesis contributes a zero rather than being dropped."""
    out = rouge(_HYPS, _REFS)
    assert out['rouge1'] == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert out['rouge2'] == pytest.approx((1.0 + 0.0 + 0.0) / 3)
    assert out['rougeL'] == pytest.approx((1.0 + 0.0 + 0.5) / 3)


def test_wer_matches_the_hand_computation() -> None:
    """Corpus WER pools edits over reference words: 0 for the exact pair, 4 for the empty one, 3 for the partial."""
    assert wer(_HYPS, _REFS) == pytest.approx(7 / 12)
    assert wer(['a b c'], ['a b c']) == 0.0


def test_content_word_f1_ignores_function_words() -> None:
    """Only content-word types count, so a fluent LM earns nothing from `the` and `on`."""
    scores = content_word_f1(_HYPS, _REFS)
    assert scores.tolist() == pytest.approx([1.0, 0.0, 1 / 3])


def test_per_sentence_scores_expose_every_paired_metric() -> None:
    """The paired bootstrap needs one array per metric, all the same length as the corpus."""
    per = per_sentence_scores(_HYPS, _REFS)
    assert set(per) == {
        'bleu1',
        'sentence_bleu4',
        'rouge1',
        'rouge2',
        'rougeL',
        'wer',
        'content_f1',
    }
    assert all(v.shape == (3,) for v in per.values())
    assert per['bleu1'].tolist() == pytest.approx([1.0, 0.0, 0.5])


def test_pairwise_metric_diagonal_is_the_per_sentence_score() -> None:
    """The permutation null and the observed statistic must be the same definition, not two implementations."""
    matrix = pairwise_metric(_HYPS, _REFS, metric='content_f1')
    assert matrix.shape == (3, 3)
    assert np.diag(matrix).tolist() == pytest.approx(content_word_f1(_HYPS, _REFS).tolist())
    with pytest.raises(ValueError, match='unknown per-sentence metric'):
        pairwise_metric(_HYPS, _REFS, metric='perplexity')


def test_a_positive_delta_whose_interval_straddles_zero_does_not_beat_its_control() -> None:
    """`beats` reads the CI lower bound: a positive mean over 24 sentences is not evidence on its own."""
    delta = np.zeros(24)
    delta[:5] = 1.0
    delta[5:9] = -1.0
    out = paired_delta(delta, np.zeros(24), metric='content_f1', n_boot=2000, seed=0)
    assert out['point'] > 0.0
    assert out['lo'] < 0.0 < out['hi']
    assert out['beats'] is False


def test_a_delta_clear_of_zero_beats_its_control() -> None:
    """The criterion has to be reachable, or every control would be un-beatable by construction."""
    out = paired_delta(np.full(24, 0.4), np.zeros(24), metric='content_f1', n_boot=2000, seed=0)
    assert out['lo'] > 0.0
    assert out['beats'] is True


def test_paired_delta_inverts_the_sign_for_word_error_rate() -> None:
    """Every delta reads "higher is better", so a lower WER than the control must come out positive."""
    hyp = np.array([0.2, 0.3, 0.1])
    ctrl = np.array([0.5, 0.6, 0.4])
    assert paired_delta(hyp, ctrl, metric='wer', n_boot=200)['point'] == pytest.approx(0.3)
    assert paired_delta(hyp, ctrl, metric='content_f1', n_boot=200)['point'] == pytest.approx(-0.3)


# --------------------------------------------------------------------------- #
# the permutation null
# --------------------------------------------------------------------------- #
def test_permutation_test_refuses_a_corpus_it_cannot_score() -> None:
    """Fewer than four sentences and a length mismatch both report why rather than returning a p-value."""
    short = generation_permutation_test(['a', 'b'], ['a', 'b'])
    assert short['applicable'] is False and 'reason' in short
    mismatched = generation_permutation_test(['a'] * 5, ['a'] * 4)
    assert mismatched['applicable'] is False and 'reason' in mismatched


def test_permutation_p_value_is_exactly_the_rank_formula() -> None:
    """`p = (1 + #{null >= observed}) / (n_perm + 1)`, checked at both ends of its range."""
    perfect = [f'sentence number {i} about topic {i}' for i in range(12)]
    out = generation_permutation_test(perfect, list(perfect), n_perm=200, seed=0)
    assert out['observed'] == pytest.approx(1.0)
    assert out['p_value'] == pytest.approx(1 / 201)
    assert out['above_chance'] is True

    # Every pairing scores identically, so every permutation ties and the null can never be beaten.
    flat = generation_permutation_test(['alpha beta'] * 8, ['alpha beta'] * 8, n_perm=200, seed=0)
    assert flat['p_value'] == pytest.approx(201 / 201)
    assert flat['above_chance'] is False


# --------------------------------------------------------------------------- #
# the load-bearing controls
# --------------------------------------------------------------------------- #
def test_cheating_decoder_is_rejected() -> None:
    """A decoder that recites the corpus scores well in absolute terms and must still be rejected.

    This is the test the whole control stack exists for: without it, a headline BLEU that comes entirely from the
    sentence-length and word-frequency statistics of ZuCo would read as a result.
    """
    hypotheses, references = _cheating_corpus()
    absolute = corpus_scores(hypotheses, references)
    assert absolute['bleu1'] > 0.2, absolute['bleu1']
    assert absolute['content_f1'] == pytest.approx(0.5)

    block = _generation_block()
    assert block['applicable'] is True
    assert block['absolute']['controls']['mean_prefix']['bleu1'] == pytest.approx(absolute['bleu1'])

    delta = block['deltas']['mean_prefix']['content_f1']
    assert abs(delta['point']) < 1e-9
    assert delta['lo'] <= 0.0 <= delta['hi']
    assert delta['beats'] is False
    assert block['controls_beaten'] == []
    assert block['beats_all_controls'] is False
    assert block['permutation']['p_value'] > 0.5

    verdict = _verdict([], {}, {}, generation=block, min_prefix_kl=0.05)
    assert verdict['generation_above_controls'] is False
    clauses = verdict['generation_clauses']
    assert clauses['honest_split'] is True
    assert clauses['prefix_influences_output'] is True
    assert clauses['beats_every_control'] is False
    assert clauses['permutation_significant'] is False


def test_a_dishonest_split_alone_demotes_the_verdict() -> None:
    """Every clause is an AND, so a headline on a stimulus-sharing split is refused whatever the deltas say."""
    block = _generation_block(split='val', split_strategy='by_subject_loso')
    verdict = _verdict([], {}, {}, generation=block)
    assert verdict['generation_clauses']['honest_split'] is False
    assert verdict['generation_above_controls'] is False


def test_the_honest_split_clause_reads_the_strategy_and_the_cell() -> None:
    """Only the cell that is held out from BOTH the subject and the stimulus passes; its name alone cannot."""
    assert _clauses(_generation_block())['honest_split'] is True

    # A cell name every strategy produces, so `test` alone says nothing about what it generalises over.
    for strategy in ('by_subject_loso', 'by_sentence', 'random', None):
        block = _generation_block(split='test', split_strategy=strategy)
        assert _clauses(block)['honest_split'] is False, strategy

    # The honest strategy's other cells share a subject or a stimulus with training.
    for cell in ('val', 'test_seen_stim', 'train'):
        block = _generation_block(split=cell, split_strategy='by_subject_and_stimulus')
        assert _clauses(block)['honest_split'] is False, cell


def test_a_split_recorded_the_old_way_cannot_pass() -> None:
    """A block naming only the strategy has no cell to check, so it is refused rather than assumed honest."""
    block = _generation_block(split='by_subject_and_stimulus')
    del block['split_strategy']
    assert _clauses(block)['honest_split'] is False


def test_a_production_shaped_honest_run_can_reach_the_headline() -> None:
    """The gate must be reachable: a decoder that beats every registered control on the honest cell passes."""
    verdict = _verdict([], {}, {}, generation=_working_block(), min_prefix_kl=0.05)
    assert verdict['generation_clauses'] == {
        'honest_split': True,
        'no_candidate_set': True,
        'beats_every_control': True,
        'permutation_significant': True,
        'prefix_influences_output': True,
    }
    assert verdict['generation_above_controls'] is True
    assert verdict['generation_controls_missing'] == []
    assert (verdict['generation_split_strategy'], verdict['generation_split_cell']) == (
        'by_subject_and_stimulus',
        'test',
    )


def test_a_control_that_never_ran_fails_its_clause() -> None:
    """A pre-registered control the decode could not build is a failed clause, not one absent from the AND."""
    block = _working_block(
        controls_requested=['mean_prefix', 'null_prefix', 'mismatch', 'phase', 'noise'],
        controls_unavailable={'phase': 'the encoder consumes no raw signal to destroy'},
    )
    verdict = _verdict([], {}, {}, generation=block, min_prefix_kl=0.05)
    assert verdict['generation_controls_missing'] == ['noise', 'phase']
    assert verdict['generation_clauses']['beats_every_control'] is False
    assert verdict['generation_above_controls'] is False


def test_a_lone_surviving_control_cannot_carry_the_verdict() -> None:
    """One control of five, four recorded unavailable: the run the review built must not read as a pass."""
    registered = list(DecoderConfig().generation_controls)
    working = _working_block()
    block = _working_block(
        controls_requested=registered,
        deltas={'mean_prefix': working['deltas']['mean_prefix']},
        controls_unavailable={c: 'unavailable' for c in registered if c != 'mean_prefix'},
    )
    verdict = _verdict([], {}, {}, generation=block, min_prefix_kl=0.05)
    assert verdict['generation_clauses']['beats_every_control'] is False
    assert verdict['generation_above_controls'] is False


def test_a_skipped_control_fails_its_clause() -> None:
    """A misaligned control produces no delta either, so the AND refuses it exactly as it refuses an absent one."""
    references = _DISTINCT * 2
    block = generation_report(
        list(references),
        references,
        {'mean_prefix': [_FREQUENT] * len(references), 'mismatch': [_FREQUENT] * 5},
        prefix_kl=1.0,
        split='test',
        n_boot=200,
        n_perm=50,
    )
    block['split_strategy'] = 'by_subject_and_stimulus'
    block['controls_requested'] = ['mean_prefix', 'mismatch']
    assert block['controls_skipped'] == {'mismatch': f'length 5 != {len(references)}'}
    assert _clauses(block)['beats_every_control'] is False


def test_the_report_names_a_control_that_never_ran() -> None:
    """A reader of `report.md` must see the pre-registered control that never executed, not just its absence."""
    block = _working_block(
        controls_requested=['mean_prefix', 'null_prefix', 'mismatch', 'phase'],
        controls_unavailable={'phase': 'the encoder consumes no raw signal to destroy'},
    )
    metrics = {'generation': block, 'verdict': _verdict([], {}, {}, generation=block)}
    text = '\n'.join(_generation_section(metrics))
    assert '| phase | NEVER RAN (the encoder consumes no raw signal to destroy) | -- | · |' in text
    assert 'Pre-registered controls not beaten: 1 of 4 -- `phase`' in text
    assert '`test` cell of `by_subject_and_stimulus`' in text


def test_the_report_says_which_cell_the_headline_is_reserved_for() -> None:
    """A run on the wrong cell must read as refused in `report.md`, not merely as a False in a clause list."""
    block = _generation_block(split='val', split_strategy='by_subject_loso')
    metrics = {'generation': block, 'verdict': _verdict([], {}, {}, generation=block)}
    text = '\n'.join(_generation_section(metrics))
    assert '`val` cell of `by_subject_loso`' in text
    assert 'The headline is reserved for the `test` cell of `by_subject_and_stimulus`' in text


def test_the_scoreboard_row_denies_a_pass_to_an_incomplete_control_stack() -> None:
    """The board reports the same completeness the gate demands, so the two can never disagree."""
    row = held_out_generation(
        _working_block(controls_unavailable={'phase': 'the encoder consumes no raw signal to destroy'})
    )
    complete = held_out_generation(_working_block())
    assert row is not None and complete is not None
    assert row['beats_all_controls'] is False
    assert row['controls_absent'] == {'phase': 'the encoder consumes no raw signal to destroy'}
    assert complete['beats_all_controls'] is True


def test_a_candidate_set_is_not_generation() -> None:
    """A decode restricted to a gallery is retrieval, and the verdict says so instead of scoring it as writing."""
    block = _generation_block(n_candidate_sentences=700)
    verdict = _verdict([], {}, {}, generation=block)
    assert verdict['generation_clauses']['no_candidate_set'] is False
    assert verdict['generation_above_controls'] is False


def test_a_collapsed_bridge_is_refused_before_its_scores_are_read() -> None:
    """Below `min_prefix_kl` the prompt does not depend on the brain, so no delta could mean anything."""
    block = _generation_block(prefix_influence_kl=0.0)
    verdict = _verdict([], {}, {}, generation=block, min_prefix_kl=0.05)
    assert verdict['generation_clauses']['prefix_influences_output'] is False
    assert verdict['generation_prefix_kl'] == 0.0


def test_verdict_ignores_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    """No `*_DIAGNOSTIC` or `*_RETRIEVAL` key reaches a clause, and a perfect one cannot flip the verdict."""
    from zte.evaluation import report as R

    clean = _generation_block()
    dirty = _generation_block(
        teacher_forced_ppl_DIAGNOSTIC=1.0,
        forced_choice_RETRIEVAL={'two_way': 1.0, 'four_way': 1.0},
    )
    dirty['absolute']['hypothesis']['teacher_forced_ppl_DIAGNOSTIC'] = 1.0
    assert quarantined_keys(dirty) == [
        'forced_choice_RETRIEVAL',
        'teacher_forced_ppl_DIAGNOSTIC',
    ]
    assert quarantined_keys(strip_quarantined(dirty)) == []

    seen: list[dict[str, Any]] = []
    real = R._generation_verdict  # noqa: SLF001

    def spy(generation: dict[str, Any], min_prefix_kl: float) -> dict[str, Any]:
        seen.append(generation)
        return real(generation, min_prefix_kl)

    monkeypatch.setattr(R, '_generation_verdict', spy)
    verdict = _verdict([], {}, {}, generation=dirty)

    assert len(seen) == 1
    assert quarantined_keys(seen[0]) == []
    assert verdict['generation_above_controls'] is False
    assert _generation_keys(verdict) == _generation_keys(_verdict([], {}, {}, generation=clean))


def test_the_generation_page_refuses_what_the_verdict_refuses() -> None:
    """The offline page reads the same ledger as the gate, so it cannot advertise a pass the gate denies."""
    from zte.evaluation.interactive.generation import _build_payload

    payload = _build_payload(_working_block(), 'run')
    assert payload['honest_split'] is True
    assert payload['verdict']['beats_all_controls'] is True

    crippled = _build_payload(
        _working_block(split='val', controls_unavailable={'phase': 'no raw signal to destroy'}),
        'run',
    )
    assert crippled['honest_split'] is False
    assert crippled['verdict']['beats_all_controls'] is False
    assert crippled['verdict']['controls_absent'] == ['phase']


def test_the_scoreboard_row_reports_what_it_stripped() -> None:
    """A quarantined key is removed from the board and named, so its absence is visible rather than silent."""
    row = held_out_generation(_generation_block(teacher_forced_ppl_DIAGNOSTIC=1.0))
    assert row is not None
    assert row['quarantined'] == ['teacher_forced_ppl_DIAGNOSTIC']
    assert quarantined_keys(row) == []
    assert row['headline_metric'] == 'content_f1_delta'
    assert row['beats_all_controls'] is False


# --------------------------------------------------------------------------- #
# report plumbing
# --------------------------------------------------------------------------- #
def test_generation_report_refuses_an_unscoreable_corpus() -> None:
    """Too few sentences or a length mismatch is reported as a reason, never as a near-zero delta."""
    assert generation_report(['a'] * 3, ['a'] * 3, {})['applicable'] is False
    assert generation_report(['a'] * 5, ['a'] * 4, {})['applicable'] is False


def test_a_control_of_the_wrong_length_is_skipped_not_silently_dropped() -> None:
    """A misaligned control would make the pairing meaningless, so it is named in `controls_skipped`."""
    hypotheses, references = _cheating_corpus()
    block = generation_report(
        hypotheses,
        references,
        {'mean_prefix': hypotheses[:5]},
        split='by_subject_and_stimulus',
        n_boot=200,
        n_perm=50,
    )
    assert block['controls_skipped'] == {'mean_prefix': f'length 5 != {len(hypotheses)}'}
    assert block['beats_all_controls'] is False


def test_beating_one_control_of_two_is_not_beating_them_all() -> None:
    """The discriminating case for the ALL that `beats_all_controls` asserts: some beaten, not every one.

    A block where no control is beaten and one where all are already agree under either reading, so the partial case
    is the only one that separates the pre-registered floor from "cleared the easiest control".
    """
    references = _DISTINCT * 3
    block = generation_report(
        list(references),
        references,
        {'easy': [_FREQUENT] * len(references), 'hard': list(references)},
        split='test',
        n_boot=500,
        n_perm=200,
        seed=0,
    )
    assert block['controls_beaten'] == ['easy']
    assert set(block['deltas']) == {'easy', 'hard'}
    assert block['beats_all_controls'] is False


def test_the_oracle_is_reported_as_a_bound_never_as_a_delta() -> None:
    """The true-text oracle bounds the achievable score; it is an absolute number and takes no control comparison."""
    hypotheses, references = _cheating_corpus()
    block = generation_report(
        hypotheses,
        references,
        {'mean_prefix': hypotheses},
        oracle=list(references),
        split='by_subject_and_stimulus',
        n_boot=200,
        n_perm=50,
    )
    assert block['absolute']['oracle']['content_f1'] == pytest.approx(1.0)
    assert 'oracle' not in block['deltas']


def test_report_rows_carry_the_side_by_side_the_page_renders() -> None:
    """The interactive page reads the block alone, so each row must carry its reference, controls and scores."""
    hypotheses, references = _cheating_corpus()
    block = generation_report(
        hypotheses,
        references,
        {'mean_prefix': hypotheses},
        oracle=list(references),
        split='by_subject_and_stimulus',
        n_boot=200,
        n_perm=50,
        max_rows=4,
    )
    rows = block['rows']
    assert len(rows) == 4
    assert rows[0]['reference'] == _FREQUENT
    assert rows[0]['controls']['mean_prefix']['text'] == _FREQUENT
    assert set(rows[0]['scores']) == set(per_sentence_scores(['a'], ['a']))
