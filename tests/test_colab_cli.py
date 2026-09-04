"""Tests for `zte-colab`, the JSON-on-stdout bridge the notebook kernel reads instead of importing ZTE."""

import argparse
import json
import platform
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from zte.cli.colab import (
    ARM_TIERS,
    CAPACITY_ARTIFACT,
    DEFAULT_MIN_PREFIX_KL,
    _arms,
    _evenly_spaced,
    _mirror,
    _panels,
    main,
    parse_arguments,
    read_capacity_artifact,
    read_decode_artifacts,
)
from zte.cli.decode import _provenance
from zte.data.schema import SUBJECTS_V1
from zte.device import device_plan
from zte.evaluation.analysis import Study, panel_builders
from zte.evaluation.audit.capacity import CLAUSE_NAMES, capacity_report
from zte.logging_utils import configure_logging
from zte.utils.env import env_defaults, machine_resources, set_env
from zte.utils.session import WRITE_MODES


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """`main()` reroutes the `zte` logger to stderr, which would otherwise outlive the test that asked for it."""
    yield
    configure_logging()


def _record(index: int) -> dict[str, Any]:
    """One `generation.jsonl` reading, scored so every number in it is predictable from its index."""
    return {
        'index': index,
        'subject': 'ZAB',
        'task': 'NR',
        'n_words': 10 + index,
        'reference': f'reference {index}',
        # The row lists its controls before the hypothesis, which is the ordering the payload must not inherit.
        'controls': {
            'length_only': {'text': f'length {index}', 'scores': {'content_f1': 0.01, 'wer': 1.2, 'rouge1': 0.1}},
            'null_prefix': {'text': f'null {index}', 'scores': {'content_f1': 0.02, 'wer': 1.3, 'rouge1': 0.2}},
        },
        'hypothesis': f'hypothesis {index}',
        'scores': {'content_f1': index / 100, 'wer': 1.0, 'rouge1': 0.5},
        'prefix_influence_kl': 0.4,
        'oracle': {'text': f'oracle {index}', 'scores': {'content_f1': 0.9, 'wer': 0.1, 'rouge1': 0.9}},
    }


def _block(*, n: int, unavailable: bool) -> dict[str, Any]:
    """A generation block that clears all five verdict clauses, bar one control the decode could not run."""
    beaten = {'metric': 'content_f1', 'point': 0.04, 'lo': 0.01, 'hi': 0.07, 'n': n, 'beats': True}
    block: dict[str, Any] = {
        'applicable': True,
        'n': n,
        'split': 'test',
        'split_strategy': 'by_subject_and_stimulus',
        'primary_metric': 'content_f1',
        'n_candidate_sentences': None,
        'absolute': {
            'hypothesis': {'content_f1': 0.31, 'wer': 0.9},
            'controls': {'length_only': {'content_f1': 0.27}, 'null_prefix': {'content_f1': 0.02}},
        },
        'deltas': {'length_only': {'content_f1': beaten}, 'null_prefix': {'content_f1': dict(beaten)}},
        'controls_requested': ['length_only', 'null_prefix'],
        'beats_all_controls': True,
        'worst_control': 'length_only',
        'worst_control_ci': beaten,
        'permutation': {'applicable': True, 'p_value': 0.004},
        'prefix_influence_kl': 0.42,
        'rows': [],
    }
    if unavailable:
        block['controls_unavailable'] = {'phase_shuffled': 'no phase model on this VM'}

    return block


def _decode_dir(
    directory: Path,
    *,
    n: int = 20,
    unavailable: bool = True,
    min_prefix_kl: float | None = None,
    applicable: bool = True,
) -> Path:
    """Fabricates the two artifacts `zte-decode --out` leaves behind, and nothing else."""
    directory.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, Any] = {'run_name': 'exp15_decode_v2', 'seed': 42}
    if min_prefix_kl is not None:
        provenance['min_prefix_kl'] = min_prefix_kl

    block = (
        _block(n=n, unavailable=unavailable)
        if applicable
        else {'applicable': False, 'reason': 'fewer than 4 held-out sentences'}
    )
    report = {
        'generation': block,
        'rescoring': {},
        'bit_budget': None,
        'provenance': provenance,
    }
    (directory / 'generation.json').write_text(json.dumps(report), encoding='utf-8')
    (directory / 'generation.jsonl').write_text(
        ''.join(json.dumps(_record(i)) + '\n' for i in range(n)), encoding='utf-8'
    )

    return directory


def _experiments(root: Path) -> Path:
    """Fabricates the tiered `experiments/` tree, `archive/` included, so the reader has something to skip."""
    for tier in (*ARM_TIERS, 'archive'):
        (root / tier).mkdir(parents=True)

    (root / 'flagship' / 'decode_v2.yaml').write_text(
        '# decode_v2 -- the decoder, on a metered leash.\nrun_name: exp15_decode_v2\ntrain:\n  mode: decoder\n'
        '  split: by_subject_and_stimulus\n  loso_holdout_subject: ZAB\n',
        encoding='utf-8',
    )
    (root / 'decoder' / 'decode_joint.yaml').write_text(
        'dataset:\n  tasks: [SR]\nrun_name: exp13_decode_joint\nobjective:\n  name: decode\ntrain:\n  mode: joint\n',
        encoding='utf-8',
    )
    (root / 'ablation' / 'raw_vs_band.yaml').write_text(
        '# raw_vs_band -- one knob moved: the frontend.\nrun_name: exp16_raw_vs_band\nobjective:\n  name: clip\n',
        encoding='utf-8',
    )
    (root / 'benchmark' / 'static_glove.yaml').write_text(
        '# static_glove -- the control a flagship must beat.\nrun_name: bench_static_glove\n', encoding='utf-8'
    )
    (root / 'archive' / 'exp8_dead.yaml').write_text(
        '# exp8_dead -- superseded, kept for the record.\nrun_name: exp8_dead\ntrain:\n  mode: decoder\n',
        encoding='utf-8',
    )

    return root


def _arm_args(root: Path, *, kind: str = 'any') -> argparse.Namespace:
    """The parsed arguments `zte-colab arms` hands its reader."""
    return argparse.Namespace(experiments=root, tiers=list(ARM_TIERS), kind=kind)


# ---- Reading a decode directory ---- #


def test_a_reading_carries_the_scores_of_the_row_it_names(tmp_path: Path) -> None:
    """Every returned reading is the `generation.jsonl` row at its own index, target and per-condition scores alike."""
    payload = read_decode_artifacts(_decode_dir(tmp_path / 'decode'), rows=12)

    twelfth = next(r for r in payload['readings'] if r['index'] == 12)

    assert twelfth['target'] == 'reference 12'
    assert twelfth['subject'] == 'ZAB'
    assert twelfth['task'] == 'NR'
    assert twelfth['n_words'] == 22
    assert twelfth['prefix_influence_kl'] == 0.4

    # `rouge1` is scored in the row and absent here, which is the default metric filter doing its job.
    assert twelfth['conditions'][0] == {
        'name': 'hypothesis',
        'text': 'hypothesis 12',
        'scores': {'content_f1': 0.12, 'wer': 1.0},
    }
    assert twelfth['conditions'][1] == {
        'name': 'length_only',
        'text': 'length 12',
        'scores': {'content_f1': 0.01, 'wer': 1.2},
    }
    assert twelfth['conditions'][3]['scores'] == {'content_f1': 0.9, 'wer': 0.1}


def test_the_hypothesis_always_leads_the_conditions(tmp_path: Path) -> None:
    """A control is only readable beside the hypothesis, so the hypothesis is first however the row was written."""
    payload = read_decode_artifacts(_decode_dir(tmp_path / 'decode'), rows=4)

    assert payload['conditions'] == ['hypothesis', 'length_only', 'null_prefix']
    for reading in payload['readings']:
        assert [c['name'] for c in reading['conditions']] == ['hypothesis', 'length_only', 'null_prefix', 'oracle']


def test_rows_samples_across_the_split_rather_than_its_head(tmp_path: Path) -> None:
    """Twelve rows out of twenty reach the last reading, so the sample is never the first twelve subjects seen."""
    payload = read_decode_artifacts(_decode_dir(tmp_path / 'decode'), rows=12)

    assert payload['source']['n_total'] == 20
    assert payload['source']['n_shown'] == 12
    assert [r['index'] for r in payload['readings']] == [0, 2, 3, 5, 7, 9, 10, 12, 14, 16, 17, 19]


def test_pick_overrides_rows_and_drops_indices_off_the_end(tmp_path: Path) -> None:
    """Named indices replace the sample entirely, and one that does not exist is dropped rather than faked."""
    payload = read_decode_artifacts(_decode_dir(tmp_path / 'decode'), rows=12, pick=[-1, 3, 7, 20, 99])

    assert [r['index'] for r in payload['readings']] == [3, 7]
    assert payload['source']['n_shown'] == 2
    assert payload['source']['n_total'] == 20


def test_metrics_filters_the_scores_of_every_condition(tmp_path: Path) -> None:
    """Only the asked-for metrics survive, and one the decode never scored is dropped rather than reported as null."""
    payload = read_decode_artifacts(_decode_dir(tmp_path / 'decode'), rows=3, metrics=('wer', 'sentence_bleu4'))

    assert payload['readings']
    for reading in payload['readings']:
        for condition in reading['conditions']:
            assert set(condition['scores']) == {'wer'}


def test_the_source_block_names_the_run_and_the_files_it_was_read_from(tmp_path: Path) -> None:
    """The payload carries its own provenance, so a rendered table can never be traced to the wrong decode."""
    out = _decode_dir(tmp_path / 'decode')

    payload = read_decode_artifacts(out, rows=2)

    assert payload['source']['generation_json'] == str(out / 'generation.json')
    assert payload['source']['generation_jsonl'] == str(out / 'generation.jsonl')
    assert payload['source']['run_name'] == 'exp15_decode_v2'
    assert payload['source']['split'] == 'test'
    assert payload['source']['split_strategy'] == 'by_subject_and_stimulus'
    assert payload['primary_metric'] == 'content_f1'
    assert payload['provenance'] == {'run_name': 'exp15_decode_v2', 'seed': 42}


@pytest.mark.parametrize('missing', ['generation.json', 'generation.jsonl'])
def test_half_a_decode_directory_raises_instead_of_reporting_an_empty_split(missing: str, tmp_path: Path) -> None:
    """A decode that never finished writing is an error, not a run with no readings in it."""
    out = _decode_dir(tmp_path / 'decode')
    (out / missing).unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        read_decode_artifacts(out)

    assert str(out / missing) in str(excinfo.value)
    assert 'zte-decode' in str(excinfo.value)


# ---- The pre-registered generation gate ---- #


def test_a_control_that_could_not_run_fails_the_verdict(tmp_path: Path) -> None:
    """An unavailable control is a control that was not beaten, and it demotes the whole five-clause gate."""
    payload = read_decode_artifacts(_decode_dir(tmp_path / 'decode', unavailable=True), rows=2)
    verdict = payload['verdict']

    assert verdict['clauses']['beats_every_control'] is False
    assert verdict['above_controls'] is False
    assert verdict['controls_absent'] == ['phase_shuffled']
    assert 'phase_shuffled' in verdict['controls_missing']
    assert verdict['beats_all_controls'] is False

    # Nothing else is what failed: every other clause of the AND is satisfied by this block.
    assert verdict['clauses']['honest_split'] is True
    assert verdict['clauses']['no_candidate_set'] is True
    assert verdict['clauses']['permutation_significant'] is True
    assert verdict['clauses']['prefix_influences_output'] is True


def test_the_same_block_passes_once_every_control_has_actually_run(tmp_path: Path) -> None:
    """Dropping the unavailable control flips the clause and the verdict, which is the whole cost of a missing one."""
    payload = read_decode_artifacts(_decode_dir(tmp_path / 'decode', unavailable=False), rows=2)
    verdict = payload['verdict']

    assert verdict['clauses']['beats_every_control'] is True
    assert verdict['above_controls'] is True
    assert verdict['controls_absent'] == []
    assert verdict['controls_missing'] == []
    assert verdict['beats_all_controls'] is True


def test_the_verdict_reports_the_numbers_behind_its_clauses(tmp_path: Path) -> None:
    """A failed clause still shows its number, so the reader can see how far short the run fell."""
    verdict = read_decode_artifacts(_decode_dir(tmp_path / 'decode'), rows=2)['verdict']

    assert verdict['permutation_p'] == 0.004
    assert verdict['prefix_kl'] == 0.42
    assert verdict['min_prefix_kl'] == 0.05
    assert verdict['worst_control'] == 'length_only'
    assert verdict['worst_ci']['lo'] == 0.01


def test_the_gate_is_read_against_the_floor_the_run_configured(tmp_path: Path) -> None:
    """A run that raised `min_prefix_kl` above its measured KL fails the clause, whatever the packaged default is."""
    out = _decode_dir(tmp_path / 'strict', unavailable=False, min_prefix_kl=0.9)

    verdict = read_decode_artifacts(out, rows=2)['verdict']

    # The block's KL is 0.42: it clears the packaged 0.05 and falls well short of this run's own floor.
    assert verdict['min_prefix_kl'] == 0.9
    assert verdict['prefix_kl'] == 0.42
    assert verdict['clauses']['prefix_influences_output'] is False
    assert verdict['above_controls'] is False


def test_a_floor_a_decode_never_recorded_falls_back_rather_than_dropping_the_clause(tmp_path: Path) -> None:
    """An artifact written before the floor travelled still gets the clause evaluated, at the registered default."""
    verdict = read_decode_artifacts(_decode_dir(tmp_path / 'old', unavailable=False), rows=2)['verdict']

    assert verdict['min_prefix_kl'] == DEFAULT_MIN_PREFIX_KL
    assert verdict['clauses']['prefix_influences_output'] is True


def test_zte_decode_records_the_verdict_floor_beside_the_numbers_it_gates() -> None:
    """The floor is unrecoverable from anywhere else in the artifacts, so the decode has to write it down."""
    decoder_config = SimpleNamespace(
        beams=1,
        max_new_tokens=32,
        cfg_weight=0.0,
        conditioning='prefix',
        rate_ladder=None,
        evidence_schedule=None,
        gap_correction=False,
        min_prefix_kl=0.05,
    )
    decoder = SimpleNamespace(
        lm=SimpleNamespace(provenance=None),
        device=SimpleNamespace(name='cpu', kind='cpu'),
        decoder_config=decoder_config,
        uses_evidence=False,
        gap=SimpleNamespace(fitted=False, n_fit=0),
        normalizer=None,
        aligner=None,
    )
    config = SimpleNamespace(
        run_name='exp15_decode_v2',
        train=SimpleNamespace(split='by_subject_and_stimulus'),
        objective=SimpleNamespace(text_source='e5'),
        decoder=SimpleNamespace(min_prefix_kl=0.37),
    )
    opts = SimpleNamespace(
        controls=['null_prefix'],
        beams=None,
        max_new_tokens=None,
        seed=42,
        seeds=[42],
        n_perm=1000,
        n_boot=1000,
        length_tol=2,
    )

    record = _provenance(decoder, config, 'test', opts, 8)  # type: ignore[arg-type]

    # The run's own floor, not the decoder's packaged one, is what the gate has to be re-read against.
    assert record['min_prefix_kl'] == 0.37


def test_a_block_that_could_not_be_scored_fails_the_gate_instead_of_losing_it(tmp_path: Path) -> None:
    """A clause that cannot be evaluated must read as failed; an absent verdict beside live scores reads as passed."""
    payload = read_decode_artifacts(_decode_dir(tmp_path / 'short', applicable=False, n=2), rows=2)

    assert payload['applicable'] is False
    assert payload['reason'] == 'fewer than 4 held-out sentences'
    assert payload['verdict']['above_controls'] is False
    assert payload['verdict']['clauses'] == {'applicable': False}
    assert payload['verdict']['reason'] == 'fewer than 4 held-out sentences'

    # The readings still render, which is exactly why the verdict may not quietly vanish from beside them.
    assert payload['readings']
    assert payload['source']['n_scored'] is None


# ---- Sampling ---- #


@pytest.mark.parametrize(('total', 'count'), [(20, 12), (700, 12), (13, 13), (5, 2), (9, 8)])
def test_a_sample_spans_the_whole_split_without_repeating(total: int, count: int) -> None:
    """The sample reaches the last reading, so a decode is never judged on the head of its split alone."""
    picked = _evenly_spaced(total, count)

    assert picked[0] == 0
    assert picked[-1] == total - 1
    assert picked == sorted(picked)
    assert len(set(picked)) == len(picked) == count


def test_a_degenerate_sample_is_still_a_valid_one() -> None:
    """One row is the first row, and asking for more rows than exist returns every one of them, once."""
    assert _evenly_spaced(20, 1) == [0]
    assert _evenly_spaced(3, 9) == [0, 1, 2]


# ---- Trainable arms ---- #


def test_an_arm_is_labelled_by_its_own_header_comment(tmp_path: Path) -> None:
    """A config describes itself, so promoting one needs no notebook edit to make it readable."""
    arms = _arms(_arm_args(_experiments(tmp_path / 'experiments')))['arms']

    by_stem = {arm['stem']: arm for arm in arms}

    assert by_stem['decode_v2']['label'] == 'decode_v2 -- the decoder, on a metered leash.'
    assert by_stem['raw_vs_band']['label'] == 'raw_vs_band -- one knob moved: the frontend.'
    assert by_stem['decode_v2']['tier'] == 'flagship'
    assert by_stem['decode_v2']['run_name'] == 'exp15_decode_v2'


def test_a_config_without_a_header_comment_falls_back_to_its_run_name(tmp_path: Path) -> None:
    """A config opening on `dataset:` is named by the run it produces, never by its first line of YAML."""
    arms = _arms(_arm_args(_experiments(tmp_path / 'experiments')))['arms']

    joint = next(arm for arm in arms if arm['stem'] == 'decode_joint')

    assert joint['label'] == 'exp13_decode_joint'


def test_a_decoder_arm_is_recognised_by_either_the_mode_or_the_objective(tmp_path: Path) -> None:
    """`train.mode: decoder` and `objective.name: decode` both mean decoder; everything else is an encoder."""
    arms = _arms(_arm_args(_experiments(tmp_path / 'experiments')))['arms']

    assert {arm['stem']: arm['kind'] for arm in arms} == {
        'decode_v2': 'decoder',
        'decode_joint': 'decoder',
        'raw_vs_band': 'encoder',
        'static_glove': 'encoder',
    }


@pytest.mark.parametrize(
    ('kind', 'expected'), [('decoder', ['decode_v2', 'decode_joint']), ('encoder', ['raw_vs_band', 'static_glove'])]
)
def test_kind_filters_the_offered_arms(kind: str, expected: list[str], tmp_path: Path) -> None:
    """Asking for one kind of arm returns only that kind, in tier order."""
    arms = _arms(_arm_args(_experiments(tmp_path / 'experiments'), kind=kind))['arms']

    assert [arm['stem'] for arm in arms] == expected


def test_a_decoder_arm_carries_the_split_and_held_out_subject_its_config_names(tmp_path: Path) -> None:
    """A decoder run's holdout lives in the config, not on a flag, so the notebook names its run directory from it."""
    payload = _arms(_arm_args(_experiments(tmp_path), kind='decoder'))
    arms = {arm['stem']: arm for arm in payload['arms']}

    assert (arms['decode_v2']['split'], arms['decode_v2']['holdout']) == ('by_subject_and_stimulus', 'ZAB')
    # A config that names no holdout reports None rather than a subject it never asked for.
    assert arms['decode_joint']['holdout'] is None


def test_the_archive_is_never_offered_as_a_trainable_arm(tmp_path: Path) -> None:
    """`experiments/archive/` is the record of what failed, so no notebook can start a run from it."""
    payload = _arms(_arm_args(_experiments(tmp_path / 'experiments')))

    assert 'archive' not in ARM_TIERS
    assert [arm['stem'] for arm in payload['arms']] == ['decode_v2', 'decode_joint', 'raw_vs_band', 'static_glove']
    assert all(Path(arm['path']).parent.name != 'archive' for arm in payload['arms'])
    assert payload['holdouts'] == list(SUBJECTS_V1)


# ---- The certified menu capacity ---- #

CAPACITY_GALLERY: int = 12
"""Stimuli in the synthetic capacity gallery -- large enough that K = 8 fills and K = 16 cannot."""

CAPACITY_KS: tuple[int, ...] = (2, 4, 8, 16)
"""Menu sizes the fixtures sweep, the last of which no pool on this gallery can fill."""


def _capacity_dir(directory: Path, *, wins: bool, tasks: bool = True) -> Path:
    """Writes a real `capacity_report` artifact, so the reader is exercised on the shapes `zte-decode` emits."""
    n = CAPACITY_GALLERY
    model = np.eye(n) if wins else np.zeros((n, n))
    arms = {
        'model': model,
        'length_only': np.zeros((n, n)),
        'shuffled_eeg': np.zeros((n, n)),
        'mismatch': np.zeros((n, n)),
    }
    report = capacity_report(
        arms,
        np.concatenate([np.arange(n), np.arange(n)]),
        np.array(['A'] * n + ['B'] * n),
        'B',
        np.full(2 * n, 9.0),
        tasks=np.array(['NR'] * (2 * n)) if tasks else None,
        ks=CAPACITY_KS,
        n_perm=200,
        n_boot=200,
        honest_split=True,
        split_strategy='by_subject_and_stimulus',
        split_cell='test',
    )
    directory.mkdir(parents=True, exist_ok=True)
    payload = {'capacity': report, 'provenance': {'run_name': directory.name, 'seed': 42}}
    (directory / CAPACITY_ARTIFACT).write_text(json.dumps(payload, default=str), encoding='utf-8')

    return directory


def test_a_capacity_that_certified_nothing_comes_back_as_none_with_its_failing_clauses(tmp_path: Path) -> None:
    """A dead decoder is the expected first real result; rendered as a blank or a zero it would read as a number."""
    payload = read_capacity_artifact(_capacity_dir(tmp_path / 'dead', wins=False))

    assert payload['certified_k'] is None
    assert payload['bits']['bits_certified'] is None
    assert set(payload['clauses']) == set(CLAUSE_NAMES)
    assert payload['clauses']['above_chance'] is False
    assert 'above_chance' in payload['per_k'][0]['failed_clauses']


def test_a_certified_capacity_carries_the_size_the_run_verdict_recorded(tmp_path: Path) -> None:
    """The payload restates `zte-decode`'s certification rather than re-running one the verdict never saw."""
    payload = read_capacity_artifact(_capacity_dir(tmp_path / 'strong', wins=True))

    assert payload['certified_k'] == payload['verdict']['capacity_k'] == 8
    assert payload['bits']['bits_certified'] == payload['verdict']['capacity_bits']
    assert all(payload['clauses'].values())
    assert payload['readout'] == 'menu selection'
    assert payload['tie_policy'] == 'ties lose'


def test_a_menu_size_no_pool_can_fill_is_named_unreachable_rather_than_failed(tmp_path: Path) -> None:
    """Exact word-count pools hold ~8 candidates, so K = 16 is a pool that cannot be built, not a decoder that lost."""
    payload = read_capacity_artifact(_capacity_dir(tmp_path / 'strong', wins=True))

    assert payload['ks']['swept'] == list(CAPACITY_KS)
    assert payload['ks']['unreachable'] == [16]
    assert [row['reachable'] for row in payload['per_k']] == [True, True, True, False]
    assert next(row for row in payload['per_k'] if row['k'] == 16)['n_queries'] == 0


def test_every_menu_size_reports_the_queries_it_was_scored_on(tmp_path: Path) -> None:
    """Pools shrink as K grows, so an accuracy without its query count is unreadable."""
    payload = read_capacity_artifact(_capacity_dir(tmp_path / 'strong', wins=True))

    assert [row['n_queries'] for row in payload['per_k'] if row['reachable']] == [12, 12, 12]
    assert all('n_queries' in row for row in payload['common_subset'])
    assert all(row['chance'] == 1.0 / row['k'] for row in payload['per_k'])


def test_every_control_comparison_travels_paired_with_both_win_counts(tmp_path: Path) -> None:
    """A delta without its win counts hides whether the model beat the control or merely averaged above it."""
    payload = read_capacity_artifact(_capacity_dir(tmp_path / 'strong', wins=True))

    rows = [row for row in payload['paired'] if row['k'] == 2]

    assert {row['control'] for row in rows} == {'length_only', 'shuffled_eeg', 'mismatch'}
    assert all(row['n_pairs'] == row['model_wins'] + row['control_wins'] + row['ties'] for row in rows)
    assert all(row['ci_lo'] is not None and row['sign_test_p'] is not None for row in rows)

    # Each control's own accuracy travels too, so the gap is plotted against a visible floor, not an implied one.
    arms = {row['arm'] for row in payload['arms'] if row['k'] == 2}
    assert arms == {'model', 'length_only', 'shuffled_eeg', 'mismatch'}
    assert next(row for row in payload['arms'] if row['k'] == 2 and row['arm'] == 'length_only')['accuracy'] == 0.0


def test_a_pool_the_run_never_swept_is_substituted_out_loud(tmp_path: Path) -> None:
    """An `open` pool standing in silently for a length-matched one would read as a certification it is not."""
    payload = read_capacity_artifact(_capacity_dir(tmp_path / 'no_tasks', wins=True, tasks=False))

    assert payload['selected']['requested_flavor'] == 'length_task_matched'
    assert payload['selected']['flavor'] == 'length_matched'
    assert payload['selected']['substituted'] is True
    assert payload['selected']['is_headline'] is True


def test_pooling_seeds_takes_the_smallest_and_sinks_on_a_seed_that_certified_nothing(tmp_path: Path) -> None:
    """A pooled capacity is a promise every seed keeps, so one failing run must sink it rather than be averaged away."""
    strong = _capacity_dir(tmp_path / 's42', wins=True)
    dead = _capacity_dir(tmp_path / 's43', wins=False)

    both = read_capacity_artifact(strong, seeds=[dead])
    alone = read_capacity_artifact(strong, seeds=[strong])

    assert both['pooled']['n_reports'] == 2
    assert both['pooled']['certified_k'] is None
    assert both['pooled']['capacity_certified'] is False
    # The primary directory is one of the seeds, so naming it twice must not count it twice.
    assert alone['pooled']['n_reports'] == 1
    assert alone['pooled']['certified_k'] == 8


def test_a_decode_that_never_ran_the_audit_says_so_instead_of_reporting_no_capacity(tmp_path: Path) -> None:
    """`capacity.json` is absent when `--capacity` was never passed, which is not the same as certifying nothing."""
    (tmp_path / 'decode').mkdir()

    with pytest.raises(FileNotFoundError, match='--capacity'):
        read_capacity_artifact(tmp_path / 'decode')


# ---- Mirroring ---- #


def test_a_mirror_onto_itself_says_so_rather_than_reporting_a_backup(tmp_path: Path) -> None:
    """Writing runs straight to Drive makes the mirror a copy onto itself, which must never read as a backup."""
    drive = tmp_path / 'drive'
    remote = drive / '2026-08-13' / 'experiments'
    remote.mkdir(parents=True)
    (remote / 'run.txt').write_text('trained', encoding='utf-8')
    args = argparse.Namespace(
        drive=drive, date='2026-08-13', write_mode='drive', sub='experiments', local=str(remote), direction='up'
    )

    payload = _mirror(args)

    assert payload['skipped_reason'] == 'source and destination are the same directory'
    assert (payload['copied'], payload['failed']) == (0, 0)
    assert payload['exclude_files'] == ['ckpt_epoch*.pt']
    assert sorted(p.name for p in remote.iterdir()) == ['run.txt']


@pytest.mark.parametrize('mode', WRITE_MODES)
def test_every_mode_a_session_opens_with_is_a_mode_the_mirror_accepts(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The notebook hands the mirror the mode it opened the session with, so refusing one of them ends the run."""
    for command in ('session', 'mirror'):
        monkeypatch.setattr('sys.argv', ['zte-colab', command, '--write-mode', mode])

        assert parse_arguments().write_mode == mode


def test_a_session_that_wrote_straight_to_drive_has_nothing_to_mirror(tmp_path: Path) -> None:
    """`auto` resolves to Drive when Drive is mounted, so the runs are already there and the mirror says so."""
    drive = tmp_path / 'drive'
    remote = drive / '2026-08-13' / 'experiments'
    remote.mkdir(parents=True)
    (remote / 'run.txt').write_text('trained', encoding='utf-8')
    args = argparse.Namespace(
        drive=drive, date='2026-08-13', write_mode='auto', sub='experiments', local=None, direction='up'
    )

    payload = _mirror(args)

    assert payload['skipped_reason'] == 'source and destination are the same directory'
    assert payload['src'] == payload['dst'] == str(remote)
    assert (payload['copied'], payload['failed']) == (0, 0)


def test_a_mirror_from_a_source_that_is_not_there_names_the_source(tmp_path: Path) -> None:
    """A local run directory the VM never made is reported, not silently counted as zero files copied."""
    drive = tmp_path / 'drive'
    gone = tmp_path / 'gone'
    args = argparse.Namespace(
        drive=drive, date='2026-08-13', write_mode='local+mirror', sub='experiments', local=str(gone), direction='up'
    )

    payload = _mirror(args)

    assert payload['skipped_reason'] == f'{gone} does not exist'
    assert (payload['copied'], payload['failed']) == (0, 0)
    assert not (drive / '2026-08-13').exists()


# ---- The panels the kernel renders ---- #


def _panel_args(roots: list[Path], out: Path, *, only: str | None = None) -> argparse.Namespace:
    """The parsed arguments `zte-colab panels` hands its reader."""
    return argparse.Namespace(experiments=roots, out=out, only=only, montage=None)


def test_every_panel_the_page_draws_is_offered_to_the_notebook_under_a_stable_name() -> None:
    """One list, drawn twice: a name that shifted would silently render a different chart under the same caption."""
    panels = panel_builders(Study())

    names = [panel.name for panel in panels]

    assert len(names) == len(set(names)), 'two panels share a name, so one would overwrite the other on disk'
    assert all(panel.caption and panel.section for panel in panels)
    assert {'metric_explorer', 'control_ladder', 'length_confound_scatter', 'mechanism_curves'} <= set(names)


def test_a_panel_with_no_data_is_named_rather_than_dropped(tmp_path: Path) -> None:
    """A chart that silently vanishes reads as a chart that had nothing to say, which is a different claim."""
    empty = tmp_path / 'experiments'
    empty.mkdir()

    payload = _panels(_panel_args([empty], tmp_path / 'panels'))

    assert payload['study'] == {'runs': 0, 'folds': 0, 'generations': 0, 'synthetic_runs': 0}
    assert 'control_ladder' in payload['empty']
    assert [p['name'] for p in payload['panels']] == [p['name'] for p in payload['panels'] if Path(p['path']).is_file()]
    assert set(payload['empty']).isdisjoint({p['name'] for p in payload['panels']})


def test_only_draws_the_named_panels_and_nothing_else(tmp_path: Path) -> None:
    """Asking for three panels must not pay for twenty-seven, and must not quietly draw a fourth."""
    empty = tmp_path / 'experiments'
    empty.mkdir()

    payload = _panels(_panel_args([empty], tmp_path / 'panels', only='scalp_3d,control_ladder'))

    assert {p['name'] for p in payload['panels']} | set(payload['empty']) == {'scalp_3d', 'control_ladder'}


def test_a_drawn_panel_lands_on_disk_as_figure_json(tmp_path: Path) -> None:
    """The kernel renders these with its own plotly, so what is written has to be a figure it can read back."""
    empty = tmp_path / 'experiments'
    empty.mkdir()

    payload = _panels(_panel_args([empty], tmp_path / 'panels', only='scalp_3d'))

    assert [p['name'] for p in payload['panels']] == ['scalp_3d']
    figure = json.loads(Path(payload['panels'][0]['path']).read_text(encoding='utf-8'))
    assert set(figure) >= {'data', 'layout'}


def test_a_synthetic_run_is_counted_so_it_can_never_be_read_as_a_result(tmp_path: Path) -> None:
    """`--synthetic` proves the plumbing works; a panel drawn over it is a wiring check and has to say so."""
    roots = tmp_path / 'experiments'
    run = roots / 'smoke'
    (run / 'evaluation').mkdir(parents=True)
    (run / 'config.yaml').write_text('run_name: smoke\n', encoding='utf-8')
    (run / 'manifest.json').write_text(json.dumps({'run_name': 'smoke', 'synthetic': True}), encoding='utf-8')
    (run / 'evaluation' / 'metrics.json').write_text(json.dumps({'scoreboard': {}}), encoding='utf-8')

    payload = _panels(_panel_args([roots], tmp_path / 'panels', only='scalp_3d'))

    assert payload['study']['runs'] == 1
    assert payload['study']['synthetic_runs'] == 1


# ---- What the env payload actually reports ---- #


def test_the_environment_is_returned_as_data_so_a_kernel_can_apply_it_itself() -> None:
    """A notebook's `!` subprocesses inherit the kernel's environment, so the kernel is where these have to land."""
    defaults = env_defaults('/tmp/zte-root')

    assert defaults['MPLBACKEND'] == 'Agg'
    assert defaults['PYTHONUNBUFFERED'] == '1'
    assert defaults['MPLCONFIGDIR'] == '/tmp/zte-root/res/.cache/matplotlib'
    assert all(isinstance(value, str) for value in defaults.values())


def test_set_env_applies_exactly_what_env_defaults_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """One source for the environment: a var added to the report but not applied in-process is a silent divergence."""
    for key in env_defaults():
        monkeypatch.delenv(key, raising=False)

    assert set(set_env()) == set(env_defaults())


def test_the_machine_report_flags_a_vm_the_raw_arms_will_not_fit_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raw-EEG bundle is ~24 GB materialised, so a small VM is an out-of-memory kill that should be predicted."""
    resources = machine_resources()

    assert resources['cpu_count'] >= 1
    assert resources['free_disk_gb'] > 0
    assert resources['low_ram'] is (resources['ram_gb'] is not None and resources['ram_gb'] < 20.0)

    monkeypatch.setattr('os.sysconf', lambda name: 1 if name == 'SC_PAGE_SIZE' else 1)

    assert machine_resources()['low_ram'] is True


def test_the_device_plan_names_what_a_batch_will_actually_do() -> None:
    """The four settings below decide whether a run fits in memory, so the notebook prints them before it starts."""
    plan = device_plan('cpu')

    assert plan['backend'] == 'cpu'
    assert plan['autocast_dtype'] == 'fp32'
    assert plan['mixed_precision'] is False
    assert plan['pin_memory'] is False
    assert plan['dataloader_workers_auto'] == 0
    assert plan['static_shapes'] is False


# ---- One JSON object on stdout ---- #


@pytest.mark.parametrize('command', ['env', 'session', 'runs', 'arms', 'readings', 'capacity', 'panels', 'mirror'])
def test_every_subcommand_prints_one_json_object_even_at_debug(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Logs are routed to stderr, so the kernel parses a whole subcommand's stdout with the standard library."""
    drive, local = tmp_path / 'drive', tmp_path / 'runs'
    run = local / 'exp16_lo_ZAB_s42'
    run.mkdir(parents=True)
    (run / 'config.yaml').write_text('run_name: exp16_lo_ZAB_s42\n', encoding='utf-8')
    _decode_dir(tmp_path / 'decode')
    _capacity_dir(tmp_path / 'decode', wins=True)
    _experiments(tmp_path / 'experiments')

    # `mirror` is given a source with a file in it, because a mirror that copies is the one that logs.
    tail = {
        'env': [],
        'session': ['--drive', str(drive)],
        'runs': ['--drive', str(drive), '--experiments', str(local)],
        'arms': ['--experiments', str(tmp_path / 'experiments')],
        'readings': ['--from', str(tmp_path / 'decode')],
        'capacity': ['--from', str(tmp_path / 'decode')],
        'panels': ['--experiments', str(local), '--out', str(tmp_path / 'panels'), '--only', 'scalp_3d'],
        'mirror': ['--drive', str(drive), '--local', str(local)],
    }[command]
    monkeypatch.setattr('sys.argv', ['zte-colab', command, *tail, '--log-level', 'DEBUG'])

    main()

    captured = capsys.readouterr()

    assert isinstance(json.loads(captured.out), dict)


def test_a_mirror_run_through_main_logs_to_stderr_and_still_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one subcommand that logs on its happy path proves where the log went: stderr, beside a clean payload."""
    drive, local = tmp_path / 'drive', tmp_path / 'runs'
    run = local / 'exp16_lo_ZAB_s42'
    run.mkdir(parents=True)
    (run / 'config.yaml').write_text('run_name: exp16_lo_ZAB_s42\n', encoding='utf-8')
    argv = ['zte-colab', 'mirror', '--drive', str(drive), '--local', str(local), '--log-level', 'DEBUG']
    monkeypatch.setattr('sys.argv', argv)

    main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload['copied'] == 1
    assert payload['skipped_reason'] is None
    assert 'Mirrored' in captured.err
    assert 'Mirrored' not in captured.out


def test_env_reports_the_interpreter_the_venv_actually_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The notebook kernel is an older Python than this, so the payload states which one the commands run on."""
    monkeypatch.setattr('sys.argv', ['zte-colab', 'env'])

    main()

    payload = json.loads(capsys.readouterr().out)

    assert set(payload) == {'root', 'env', 'accelerator', 'plan', 'resources', 'venv'}
    assert payload['venv']['python'] == platform.python_version()


def test_geometry_reads_a_checkpoint_and_says_whether_a_scalp_map_can_be_drawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`geometry` reports the checkpoint's own flag and whether a montage on this machine reproduces its basis."""
    import torch

    from zte.config import ModelConfig, ZTEConfig
    from zte.data.montage.montage import packaged_montage_csv
    from zte.models.embedding import build_model

    monkeypatch.delenv('ZTE_CACHE_REMOTE', raising=False)
    config = ZTEConfig(
        run_name='geometry_test',
        model=ModelConfig(
            frontend='raw_conformer',
            embed_dim=16,
            hidden_dim=16,
            n_layers=1,
            n_heads=2,
            conformer_filters=8,
            factored=False,
            subject_adapter=False,
            spatial_encoding='spherical_harmonics',
            spatial_harmonic_degree=2,
        ),
    )
    named = tmp_path / 'res' / 'montage_gsn105.csv'
    named.parent.mkdir()
    named.write_bytes(packaged_montage_csv().read_bytes())
    torch.manual_seed(0)
    model = build_model(config.model, raw_shape=(105, 32), n_channels=105, montage_csv=str(named))
    ckpt = tmp_path / 'best.pt'
    torch.save({'config': config.to_dict(), 'model': model.state_dict(), 'extra': {'montage_csv': str(named)}}, ckpt)
    named.unlink()
    monkeypatch.setattr('sys.argv', ['zte-colab', 'geometry', '--ckpt', str(ckpt)])

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload['run_name'] == 'geometry_test' and payload['has_harmonic_basis'] is True
    assert payload['approximate_geometry'] is False and payload['n_channels'] == 105 and payload['l_max'] == 2
    assert payload['montage_source'] == 'packaged' and payload['topomap_readable'] is True
    assert payload['montage_csv'] == str(named) and payload['reason'] is None
