"""Tests that the evidence board refuses to call an unfloored or underpowered number a result."""

from pathlib import Path
from typing import Any

import pytest

from zte.cli.evidence import collect_artifacts
from zte.evaluation.audit.evidence import (
    MIN_RESOLVABLE_HITS,
    Claim,
    Verdict,
    board_to_dict,
    calibration_claim,
    decoder_control_claim,
    evidence_report,
    granularity_claims,
    piece_oracle_claim,
    render_markdown,
    transfer_claim,
)


@pytest.fixture
def levels_payload() -> dict[str, Any]:
    """The three alignment levels in the shape `zte-levels` writes, at their measured twelve-fold values."""

    def level(name: str, rank: float, sd: float, lo: float, hi: float, hits: int) -> dict[str, Any]:
        return {
            'level': name,
            'n_folds': 12,
            'postprocess_fit': 'train split',
            'length_stratified': {
                'rank_percentile': rank,
                'rank_percentile_sd': sd,
                'rank_percentile_ci': [lo, hi],
                'hits_top1': hits,
                'n_queries': 8199,
                'top1_p': 0.0,
            },
            'length_floor': {'rank_percentile': 0.9525, 'tol': 1},
            'oracle_floor': {'ceiling_top1': 0.9443, 'signature': 'total'},
            'clears_floor': False,
            'missing': [],
        }

    return {
        'levels': [
            level('sentence', 0.9238, 0.0079, 0.9195, 0.9281, 335),
            level('token', 0.9286, 0.0063, 0.9251, 0.9318, 390),
        ]
    }


def test_a_level_below_its_length_oracle_is_verdicted_below_the_floor(levels_payload: dict[str, Any]) -> None:
    """Both measured levels sit under the word-count oracle, and neither may read as a win."""
    claims = granularity_claims(levels_payload)

    assert len(claims) == 2
    assert all(c.verdict is Verdict.BELOW_FLOOR for c in claims)
    assert not any(c.headline_safe() for c in claims)


def test_the_nominally_higher_level_is_still_below_the_floor(levels_payload: dict[str, Any]) -> None:
    """Token scores above sentence and below the oracle, so ranking the levels cannot promote it."""
    by_key = {c.key: c for c in granularity_claims(levels_payload)}
    token, sentence = by_key['granularity.token'], by_key['granularity.sentence']

    assert token.value is not None and sentence.value is not None
    assert token.value > sentence.value
    assert token.verdict is Verdict.BELOW_FLOOR


def test_a_claim_with_no_floor_is_never_a_result() -> None:
    """An unfloored number is `not measured`, which is what stops a bare rate becoming a headline."""
    unfloored = {
        'levels': [
            {
                'level': 'sentence',
                'length_stratified': {'rank_percentile': 0.99, 'rank_percentile_ci': [0.98, 0.999], 'n_queries': 700},
                'length_floor': {},
            }
        ]
    }
    claim = granularity_claims(unfloored)[0]

    assert claim.verdict is Verdict.NOT_MEASURED
    assert not claim.headline_safe()


def test_a_point_estimate_above_the_floor_with_a_straddling_interval_does_not_clear() -> None:
    """The interval has to clear the floor; a point estimate alone is the shape of every retraction here."""

    def word(rank: float, lo: float, hi: float) -> dict[str, Any]:
        return {
            'levels': [
                {
                    'level': 'word',
                    'length_stratified': {'rank_percentile': rank, 'rank_percentile_ci': [lo, hi], 'n_queries': 700},
                    'length_floor': {'rank_percentile': 0.9525, 'tol': 1},
                }
            ]
        }

    straddling = word(0.96, 0.94, 0.98)
    clear = word(0.98, 0.97, 0.99)

    assert granularity_claims(straddling)[0].verdict is Verdict.BELOW_FLOOR
    assert granularity_claims(clear)[0].verdict is Verdict.CLEARS


def _nested_matrix() -> dict[str, Any]:
    """The parallax matrix as `PARALLAX.json` writes it: {train_task: {eval_task: [one record per seed]}}."""
    return {
        'cells': {
            'NR': {
                'NR': [{'seed': 42, 'novel_stimuli': False, 'rank_percentile': 0.9488, 'n_queries': 300}],
                'SR': [
                    {
                        'seed': 42,
                        'novel_stimuli': True,
                        'rank_percentile': 0.9595,
                        'rank_percentile_ci': [0.9595, 0.9542, 0.9643],
                        'top1_hits': 5,
                        'n_queries': 400,
                        'top1_p': 0.0036,
                        'postprocess_fit': 'train split',
                    }
                ],
            },
            'TSR': {'SR': [{'seed': 42, 'novel_stimuli': True, 'rank_percentile': 0.9243, 'n_queries': 400}]},
        }
    }


def test_the_nested_parallax_matrix_is_flattened_with_its_task_pair_restored() -> None:
    """`PARALLAX.json` carries the task pair as nesting keys, so a flat reader would see every cell as unlabelled."""
    claim = transfer_claim(_nested_matrix())

    assert claim is not None
    assert claim.value == pytest.approx(0.9595)
    assert any('train NR -> eval SR' in c for c in claim.caveats)


def test_the_best_novel_cell_is_chosen_over_a_higher_within_task_one() -> None:
    """The claim is about never-seen stimuli, so a diagonal cell may never stand in for an off-diagonal one."""
    matrix = _nested_matrix()
    matrix['cells']['NR']['NR'][0]['rank_percentile'] = 0.999

    claim = transfer_claim(matrix)

    assert claim is not None
    assert claim.value == pytest.approx(0.9595)


def test_a_single_cell_transfer_json_is_read_from_its_held_out_block() -> None:
    """One cell's artifact keeps its metrics under `held_out`, not at the top level like the aggregated matrix."""
    claim = transfer_claim(
        {
            'train_task': 'NR',
            'eval_task': 'SR',
            'novel_stimuli': True,
            'postprocess_fit': 'train split',
            'held_out': {
                'rank_percentile': 0.9595,
                'rank_percentile_ci': [0.9595, 0.9542, 0.9643],
                'top1': 0.0125,
                'n_queries': 400,
                'top1_p': 0.0036,
            },
        }
    )

    assert claim is not None
    assert claim.value == pytest.approx(0.9595)
    assert claim.hits == 5
    assert claim.n_queries == 400


def test_a_cell_sharing_stimuli_with_training_cannot_carry_the_claim() -> None:
    """`novel_stimuli` is measured, not assumed; a cell with overlap says nothing about passage memorisation."""
    matrix = _nested_matrix()
    for columns in matrix['cells'].values():
        for records in columns.values():
            for record in records:
                record['novel_stimuli'] = False

    assert transfer_claim(matrix) is None


def test_a_thin_hit_count_is_flagged_even_when_the_interval_metric_clears() -> None:
    """Rank percentile uses every query, so it stays powered -- but the Top-k beside it must admit it is not."""
    thin = {
        'cells': [
            {
                'train_task': 'NR',
                'eval_task': 'SR',
                'novel_stimuli': True,
                'rank_percentile': 0.9595,
                'rank_percentile_ci': [0.9595, 0.9542, 0.9643],
                'top1_hits': MIN_RESOLVABLE_HITS - 5,
                'n_queries': 400,
                'top1_p': 0.0036,
            },
            {'train_task': 'NR', 'eval_task': 'NR', 'novel_stimuli': False, 'rank_percentile': 0.9488},
        ]
    }
    claim = transfer_claim(thin)

    assert claim is not None
    assert claim.verdict is Verdict.CLEARS
    assert any('below the' in c and 'resolve' in c for c in claim.caveats)


def test_a_hit_count_with_no_interval_behind_it_is_underpowered() -> None:
    """When the hit count is the only evidence, too few hits cannot settle the claim either way."""
    from zte.evaluation.audit.evidence import _verdict_for

    assert _verdict_for(0.02, 0.01, hits=MIN_RESOLVABLE_HITS - 1) is Verdict.UNDERPOWERED
    assert _verdict_for(0.02, 0.01, hits=MIN_RESOLVABLE_HITS + 1) is Verdict.CLEARS


def test_the_piece_oracle_row_names_the_gate_it_failed() -> None:
    """The measured ZuCo gate is the total sub-token count, and the observed Top-1 is far below it."""
    claim = piece_oracle_claim(
        {
            'piece_oracle': {
                'observed_top1': 0.0214,
                'gate_top1': 0.1014,
                'gate_signature': 'total',
                'ceiling_top1': 0.9443,
                'tokenizer': 'Qwen/Qwen2.5-0.5B',
                'alignment_coverage': 0.9999,
            }
        }
    )

    assert claim is not None
    assert claim.verdict is Verdict.BELOW_FLOOR
    assert claim.floor_name is not None and 'total' in claim.floor_name
    assert any('0.9443' in c for c in claim.caveats)


def test_a_low_alignment_coverage_is_carried_as_a_caveat() -> None:
    """Below 0.99 the piece counts are partly wrong, so the bits are not trustworthy and must say so."""
    claim = piece_oracle_claim(
        {
            'piece_oracle': {
                'observed_top1': 0.02,
                'gate_top1': 0.10,
                'gate_signature': 'total',
                'alignment_coverage': 0.91,
            }
        }
    )

    assert claim is not None
    assert any('0.99' in c for c in claim.caveats)


def test_calibration_is_read_against_its_shuffled_anchor_control() -> None:
    """A map fitted on wrong pairings that lifts as much means the lift was the transform, not calibration."""
    payload = {
        'curve': [
            {'n_anchors': 0, 'rank_percentile': 0.9130, 'n_queries': 700, 'family': 'procrustes'},
            {
                'n_anchors': 50,
                'rank_percentile': 0.9310,
                'rank_percentile_ci': [0.9310, 0.9260, 0.9360],
                'n_queries': 650,
                'family': 'procrustes',
            },
        ],
        'shuffled_control': {'best_rank_percentile': 0.9400},
        'postprocess_fit': 'train split',
    }
    claim = calibration_claim(payload)

    assert claim is not None
    assert claim.floor_name == 'shuffled-anchor control'
    assert claim.verdict is Verdict.BELOW_FLOOR


def test_calibration_without_a_shuffled_control_says_the_lift_is_not_separable() -> None:
    """No control means the number cannot be attributed to calibration, and the row has to admit it."""
    payload = {
        'curve': [
            {'n_anchors': 0, 'rank_percentile': 0.9130, 'n_queries': 700},
            {'n_anchors': 50, 'rank_percentile': 0.9500, 'rank_percentile_ci': [0.95, 0.945, 0.955], 'n_queries': 650},
        ]
    }
    claim = calibration_claim(payload)

    assert claim is not None
    assert any('not separable' in c for c in claim.caveats)


def test_the_decoder_row_reads_the_worst_control_not_the_kindest() -> None:
    """A verdict that ANDs over controls is set by the weakest margin, so the board must quote that one."""
    claim = decoder_control_claim(
        {
            'deltas': {
                'null_prefix': {'delta': 0.12, 'ci': [0.12, 0.08, 0.16]},
                'noise_prefix': {'delta': 0.002, 'ci': [0.002, -0.01, 0.015]},
            },
            'verdict': {'permutation_p': 0.3},
            'controls_absent': ['length_only'],
        }
    )

    assert claim is not None
    assert claim.value == pytest.approx(0.002)
    assert claim.verdict is Verdict.BELOW_FLOOR
    assert any('length_only' in c for c in claim.caveats)


def test_a_missing_artifact_is_named_rather_than_dropped() -> None:
    """A silently absent row reads as a claim nobody made, which is how a gap becomes an implied pass."""
    board = evidence_report()

    assert board.claims == ()
    assert set(board.missing) == {'granularity', 'resolution_limit', 'deployment', 'confound', 'decoder'}
    assert all(why for why in board.missing.values())


def test_the_rendered_board_says_nothing_is_quotable_when_nothing_clears(levels_payload: dict[str, Any]) -> None:
    """The document's job is to stop a floor-failing row being quoted without its floor."""
    board = evidence_report(levels=levels_payload, sources={'levels': 'res/levels/levels.json'})
    text = render_markdown(board)

    assert 'Nothing on this board clears its floor' in text
    assert 'below a brain-free floor' in text
    assert 'not measured' in text.lower()


def test_top_k_renders_as_a_hit_count_never_a_bare_rate(levels_payload: dict[str, Any]) -> None:
    """A rate with no denominator and no binomial tail is exactly what this project forbids quoting."""
    board = evidence_report(levels=levels_payload)
    text = render_markdown(board)

    assert '335/8199' in text
    assert '390/8199' in text


def test_the_payload_counts_how_many_rows_may_be_quoted_alone(levels_payload: dict[str, Any]) -> None:
    """The notebook renders this number, so it has to be derived rather than asserted."""
    payload = board_to_dict(evidence_report(levels=levels_payload))

    assert payload['n_headline_safe'] == 0
    assert all(row['headline_safe'] is False for row in payload['claims'])


def test_a_clearing_row_becomes_quotable_only_with_a_floor() -> None:
    """The one path to a headline is an interval above a named brain-free floor."""
    floored = Claim(
        key='x',
        question='q',
        metric='m',
        value=0.99,
        ci=(0.98, 0.995),
        floor=0.9525,
        floor_name='length oracle (tol=1)',
        verdict=Verdict.CLEARS,
    )
    unfloored = Claim(key='y', question='q', metric='m', value=0.99, verdict=Verdict.CLEARS)

    assert floored.headline_safe()
    assert not unfloored.headline_safe()


def test_an_artifact_nested_under_a_root_is_found_at_any_depth(tmp_path: Path) -> None:
    """A session lays artifacts out as `<root>/<run>/<audit>/x.json`, so an exact-depth glob would miss them."""
    nested = tmp_path / 'run_a' / 'calibration'
    nested.mkdir(parents=True)
    (nested / 'calibration.json').write_text('{"curve": []}', encoding='utf-8')

    payloads, where = collect_artifacts([tmp_path], depth=3)

    assert 'deployment' in payloads
    assert where['deployment'].endswith('calibration/calibration.json')


def test_a_shallower_artifact_wins_over_a_deeper_one(tmp_path: Path) -> None:
    """A run's own audit outranks one buried in an archived session under the same root."""
    (tmp_path / 'levels').mkdir()
    (tmp_path / 'levels' / 'levels.json').write_text('{"levels": []}', encoding='utf-8')
    deep = tmp_path / 'archive' / 'old' / 'levels'
    deep.mkdir(parents=True)
    (deep / 'levels.json').write_text('{"levels": []}', encoding='utf-8')

    _, where = collect_artifacts([tmp_path], depth=4)

    assert where['levels'] == str(tmp_path / 'levels' / 'levels.json')
