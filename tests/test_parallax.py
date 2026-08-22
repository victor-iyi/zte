"""Tests for the parallax study: config derivation, transfer-cell honesty, CKA, and report aggregation."""

import json
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pytest

import zte.parallax.transfer as transfer_mod
from zte.cli.parallax import parse_arguments
from zte.config import ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.parallax.report import build_report, load_cells
from zte.parallax.study import (
    PARALLAX_TASKS,
    arm_run_name,
    cell_name,
    derive_eval_config,
    parse_cell_name,
    resolve_transfer_holdout,
    run_dir_name,
)
from zte.parallax.transfer import POSTPROCESS_FIT, linear_cka, transfer_report, write_cell

_SUBJECTS: Final[tuple[str, ...]] = ('ZAB', 'ZDM', 'ZKB')


def _cohort(
    n_stimuli: int = 60,
    dim: int = 16,
    seed: int = 0,
    task: str = 'SR',
    codes: tuple[str, ...] = _SUBJECTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One task's readings: unique random vectors per reading over shared stimuli, lengths and texts."""
    rng = np.random.default_rng(seed)
    lengths = np.asarray([5, 10], dtype=np.float64)[rng.integers(0, 2, size=n_stimuli)]

    emb, content, subjects, words, texts = [], [], [], [], []
    for code in codes:
        emb.append(rng.standard_normal((n_stimuli, dim)).astype(np.float32))
        content.append(np.arange(n_stimuli))
        subjects += [code] * n_stimuli
        words.append(lengths)
        texts.append(np.array([f'{task} sentence {i}' for i in range(n_stimuli)]))

    return (
        np.concatenate(emb),
        np.concatenate(content),
        np.array(subjects),
        np.concatenate(words),
        np.concatenate(texts),
    )


def _assert_overlap_flagged(report: dict[str, object]) -> None:
    """The honest reading of a contaminated cross-task cell: overlap recorded, novelty denied."""
    assert report['stimulus_overlap'], 'the construction shares stimuli, so the overlap must be non-zero'
    assert report['novel_stimuli'] is False


# --------------------------------------------------------------------------- #
# study: the triad, the names, the derived config
# --------------------------------------------------------------------------- #
def test_the_task_triad_and_names_are_pinned() -> None:
    """The task order and naming scheme are quoted by configs, the notebook and the runner suffixes."""
    assert PARALLAX_TASKS == ('NR', 'SR', 'TSR')
    assert arm_run_name('NR') == 'parallax_nr'
    assert run_dir_name('SR', 'ZAB', 1) == 'parallax_sr_loZAB_s1'


def test_cell_names_round_trip_and_reject_strangers() -> None:
    """Cell directories are the report's only index, so the name must parse back exactly or not at all."""
    assert cell_name('NR', 'SR', 3) == 'NR_to_SR_s3'
    assert parse_cell_name('NR_to_SR_s3') == ('NR', 'SR', 3)
    assert parse_cell_name('TSR_to_TSR_s12') == ('TSR', 'TSR', 12)
    for stranger in ('NR_to_SR', 'XX_to_SR_s0', 'NR_to_SR_s0_extra', 'evaluation'):
        assert parse_cell_name(stranger) is None


def test_derive_eval_config_swaps_tasks_and_nothing_else() -> None:
    """The eval clone changes `dataset.tasks` alone, so its cache key moves only through the task set."""
    config = ZTEConfig.from_dict(
        {
            'run_name': 'parallax_nr',
            'dataset': {'tasks': ['NR'], 'raw_window': 64, 'time_bins': 2},
            'train': {'seed': 7},
        }
    )
    derived = derive_eval_config(config, 'TSR')

    expected = config.to_dict()
    expected['dataset']['tasks'] = ('TSR',)
    assert derived.to_dict() == expected
    assert config.dataset.tasks == ('NR',), 'the source config must be left untouched'

    with pytest.raises(ValueError, match='eval_task'):
        derive_eval_config(config, 'XR')  # type: ignore[arg-type]


def test_a_derived_config_builds_a_single_task_dataset(synthetic_dir: Path, tmp_path: Path) -> None:
    """Deriving to one task and building yields readings of that task only."""
    config = ZTEConfig.from_dict(
        {'dataset': {'root': str(synthetic_dir), 'tasks': ['SR', 'NR'], 'cache_dir': str(tmp_path / 'cache')}}
    )
    derived = derive_eval_config(config, 'NR')
    dataset = ZuCoDataset(derived.dataset).build(show_progress=False)

    assert set(dataset.sentences['task']) == {'NR'}


# --------------------------------------------------------------------------- #
# CKA
# --------------------------------------------------------------------------- #
def test_cka_is_exactly_one_against_itself_and_rotation_invariant() -> None:
    """A representation carries the same similarity structure as itself, rotated or not."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 8))
    assert linear_cka(x, x) == 1.0

    rotation, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    assert linear_cka(x, x @ rotation) == pytest.approx(1.0, abs=1e-12)
    assert linear_cka(x, 3.0 * x) == pytest.approx(1.0, abs=1e-12)


def test_cka_pins_a_hand_computed_value() -> None:
    """The 3-reading case works out to 5 / (2 * sqrt(10)) by hand; the code must reproduce it exactly."""
    x = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    y = np.array([[1.0], [0.0], [0.0]])

    assert linear_cka(x, y) == pytest.approx(5.0 / (2.0 * np.sqrt(10.0)))
    assert linear_cka(x, y) == pytest.approx(linear_cka(y, x))
    assert np.isnan(linear_cka(x, np.ones((3, 2))))

    with pytest.raises(ValueError, match='same readings'):
        linear_cka(x, np.zeros((4, 2)))


# --------------------------------------------------------------------------- #
# transfer cells: the canary, the leak, the novelty guard
# --------------------------------------------------------------------------- #
def test_a_random_cohort_brackets_chance_and_carries_the_contract_fields() -> None:
    """No-signal embeddings must sit at rank percentile 0.5 -- and this is also the holdout-leak canary.

    Every reading is a unique random vector. If the holdout's readings leaked into the post-processing
    fit or the gallery-side references, the queries would meet their own vectors and inflate above chance.
    One gallery subject, so each query has exactly one positive and chance rank percentile is exactly 0.5.
    """
    emb, content, subjects, words, texts = _cohort(codes=('ZAB', 'ZDM'))
    report = transfer_report(
        emb,
        content,
        subjects,
        words,
        texts,
        train_task='NR',
        eval_task='SR',
        holdout='ZAB',
        train_stimulus_texts={f'NR sentence {i}' for i in range(60)},
        n_boot=500,
    )

    assert set(report) == {
        'train_task',
        'eval_task',
        'seed',
        'holdout',
        'novel_stimuli',
        'stimulus_overlap',
        'stimulus_overlap_normalized',
        'n_queries',
        'held_out',
        'held_out_length_stratified',
        'menu',
        'postprocess_fit',
        'provenance',
    }
    assert report['postprocess_fit'] == POSTPROCESS_FIT
    assert report['novel_stimuli'] is True and report['stimulus_overlap'] == 0
    assert report['n_queries'] == 60

    held = report['held_out']
    assert held is not None
    _, lo, hi = held['rank_percentile_ci']
    assert lo < 0.5 < hi, (lo, hi)

    stratified = report['held_out_length_stratified']
    assert stratified is not None and stratified['length_tol'] == 1

    # The same canary for the menu: prototypes are training-subject centroids, so 2-way sits at chance.
    menu = report['menu']
    assert menu is not None
    _, menu_lo, menu_hi = menu['flavors']['length_matched']['per_k']['2']['ci']
    assert menu_lo < 0.5 < menu_hi, (menu_lo, menu_hi)


def test_enrolling_the_holdouts_own_readings_inflates_the_canary() -> None:
    """MUTATION stand-in for a holdout leak: copy the queries into the gallery and watch chance vanish.

    This is what the canary above protects against -- were the holdout's rows ever enrolled on the
    gallery side, every query would meet its own vector and the CI would leave 0.5 far below.
    """
    emb, content, subjects, words, texts = _cohort()
    leak = subjects == 'ZAB'
    emb2 = np.concatenate([emb, emb[leak]])
    content2 = np.concatenate([content, content[leak]])
    subjects2 = np.concatenate([subjects, np.array(['LEAK'] * int(leak.sum()))])
    words2 = np.concatenate([words, words[leak]])
    texts2 = np.concatenate([texts, texts[leak]])

    report = transfer_report(
        emb2,
        content2,
        subjects2,
        words2,
        texts2,
        train_task='NR',
        eval_task='SR',
        holdout='ZAB',
        train_stimulus_texts=set(),
        n_boot=200,
    )
    held = report['held_out']
    assert held is not None
    _, lo, _ = held['rank_percentile_ci']
    assert held['rank_percentile'] > 0.9 and lo > 0.5, 'the leaked gallery must blow the canary, not pass it'


def test_a_cross_task_overlap_is_flagged_and_shouted(caplog: pytest.LogCaptureFixture) -> None:
    """A cross-task cell sharing stimuli loses its novelty claim out loud, never silently."""
    emb, content, subjects, words, texts = _cohort(n_stimuli=20)
    with caplog.at_level(logging.WARNING, logger='zte.parallax.transfer'):
        report = transfer_report(
            emb,
            content,
            subjects,
            words,
            texts,
            train_task='NR',
            eval_task='SR',
            holdout='ZAB',
            train_stimulus_texts={f'SR sentence {i}' for i in range(10)},
            n_boot=100,
        )

    assert report['stimulus_overlap'] == 10
    _assert_overlap_flagged(report)
    assert any('NOT never-seen-stimuli' in record.message for record in caplog.records)


def test_a_same_task_cell_is_never_novel(caplog: pytest.LogCaptureFixture) -> None:
    """The diagonal shares its stimuli by construction: not novel, and not worth a warning either."""
    emb, content, subjects, words, texts = _cohort(n_stimuli=20)
    with caplog.at_level(logging.WARNING, logger='zte.parallax.transfer'):
        report = transfer_report(
            emb,
            content,
            subjects,
            words,
            texts,
            train_task='SR',
            eval_task='SR',
            holdout='ZAB',
            train_stimulus_texts={f'SR sentence {i}' for i in range(20)},
            n_boot=100,
        )

    assert report['novel_stimuli'] is False
    assert report['stimulus_overlap'] == 20
    assert not caplog.records, 'full overlap on the diagonal is expected, not a contamination'


def test_mutation_disabling_the_novelty_guard_turns_the_flag_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: with `stimulus_novelty` forced to True, the honest overlap check must go red."""
    monkeypatch.setattr(transfer_mod, 'stimulus_novelty', lambda train, eval_task, overlap: True)

    emb, content, subjects, words, texts = _cohort(n_stimuli=20)
    mutated = transfer_report(
        emb,
        content,
        subjects,
        words,
        texts,
        train_task='NR',
        eval_task='SR',
        holdout='ZAB',
        train_stimulus_texts={f'SR sentence {i}' for i in range(10)},
        n_boot=100,
    )

    assert mutated['novel_stimuli'] is True, 'the lie the guard exists to prevent'
    with pytest.raises(AssertionError):
        _assert_overlap_flagged(mutated)


def test_write_cell_persists_report_and_embeddings(tmp_path: Path) -> None:
    """A cell on disk is transfer.json plus the raw float32 embeddings the report stage rebuilds from."""
    emb, content, subjects, words, texts = _cohort(n_stimuli=10)
    report = transfer_report(
        emb,
        content,
        subjects,
        words,
        texts,
        train_task='NR',
        eval_task='SR',
        holdout='ZAB',
        train_stimulus_texts=set(),
        n_boot=100,
    )
    cell_dir = write_cell(
        tmp_path / cell_name('NR', 'SR', 0),
        report,
        sent_emb=emb,
        content_ids=content,
        subjects=subjects,
        n_words=words,
        texts=texts,
    )

    parsed = json.loads((cell_dir / 'transfer.json').read_text(encoding='utf-8'))
    assert parsed['train_task'] == 'NR' and parsed['eval_task'] == 'SR'
    with np.load(cell_dir / 'embeddings.npz', allow_pickle=False) as data:
        assert set(data.files) == {'sent_emb', 'content_ids', 'subjects', 'n_words', 'texts'}
        assert data['sent_emb'].dtype == np.float32
        assert len(data['sent_emb']) == len(subjects)


# --------------------------------------------------------------------------- #
# report aggregation
# --------------------------------------------------------------------------- #
def test_report_aggregation_round_trip(tmp_path: Path) -> None:
    """Two models x two eval tasks aggregate into the matrix, the markdown and the chamber data."""
    rng = np.random.default_rng(5)
    n_stimuli, dim = 30, 8
    cells_dir, out_dir = tmp_path / 'cells', tmp_path / 'report'

    cohorts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for task in ('NR', 'SR'):
        _, content, subjects, words, texts = _cohort(n_stimuli=n_stimuli, dim=dim, task=task, seed=7)
        shared = rng.standard_normal((len(subjects), dim)).astype(np.float32)
        cohorts[task] = (content, subjects, words, texts, shared)

    for train in ('NR', 'SR'):
        for eval_task in ('NR', 'SR'):
            content, subjects, words, texts, shared = cohorts[eval_task]
            emb = shared + 0.5 * rng.standard_normal(shared.shape).astype(np.float32)
            report = transfer_report(
                emb,
                content,
                subjects,
                words,
                texts,
                train_task=train,
                eval_task=eval_task,
                holdout='ZAB',
                train_stimulus_texts=set(cohorts[train][3].tolist()),
                n_boot=100,
                provenance={'run_name': f'parallax_{train.lower()}'},
            )
            write_cell(
                cells_dir / cell_name(train, eval_task, 0),
                report,
                sent_emb=emb,
                content_ids=content,
                subjects=subjects,
                n_words=words,
                texts=texts,
            )
    (cells_dir / 'notacell').mkdir()

    assert len(load_cells(cells_dir)) == 4, 'the stranger directory is skipped, never mistaken for a cell'
    parallax = build_report(cells_dir, out_dir)

    # PARALLAX.json: the matrix with per-seed summaries.
    assert parallax['study'] == 'parallax' and parallax['holdout'] == 'ZAB'
    assert parallax['tasks'] == ['NR', 'SR', 'TSR'] and parallax['seeds'] == [0]
    off_diagonal = parallax['cells']['NR']['SR'][0]
    assert off_diagonal['novel_stimuli'] is True and off_diagonal['stimulus_overlap'] == 0
    assert isinstance(off_diagonal['rank_percentile'], float)
    assert parallax['cells']['NR']['NR'][0]['novel_stimuli'] is False
    assert off_diagonal['run_name'] == 'parallax_nr'

    # CKA: one pair, one seed, a similarity in [0, 1].
    pair = parallax['cka']['NR|SR']
    assert pair['eval_task'] in ('NR', 'SR') and len(pair['per_seed']) == 1
    assert 0.0 <= pair['per_seed'][0] <= 1.0

    # PARALLAX.md: matrix, novel markers, and the honest-reading section.
    markdown = (out_dir / 'PARALLAX.md').read_text(encoding='utf-8')
    assert 'train \\ eval' in markdown and '*novel*' in markdown
    assert 'Reading this honestly' in markdown and 'null' in markdown
    assert 'Free generation is not a parallax deliverable' in markdown

    # CHAMBER_DATA.json: pre-reduced geometry per the contract.
    chamber = json.loads((out_dir / 'CHAMBER_DATA.json').read_text(encoding='utf-8'))
    assert set(chamber) == {'holdout', 'tasks', 'points', 'transfer', 'capacity', 'cka'}
    for eval_task in ('NR', 'SR'):
        points = chamber['points'][eval_task]
        assert len(points) == n_stimuli
        point = points[0]
        assert set(point) == {'text', 'cluster', 'n_words', 'views', 'rank_percentile'}
        assert set(point['views']) == {'NR', 'SR'}
        assert len(point['views']['NR']) == 3 and all(isinstance(v, float) for v in point['views']['NR'])
        assert point['text'].startswith(eval_task)
        assert isinstance(point['cluster'], int) and isinstance(point['n_words'], int)
    assert chamber['transfer']['NR']['SR']['n_seeds'] == 1
    assert set(chamber['capacity']) == {'NR', 'SR'}
    assert isinstance(chamber['cka']['NR|SR'], float)


def test_report_refuses_an_empty_directory(tmp_path: Path) -> None:
    """No cells means no report: a clear refusal, never an empty file pretending to be evidence."""
    with pytest.raises(ValueError, match='No transfer cells'):
        build_report(tmp_path, tmp_path / 'out')


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #
def test_cli_surface_parses_the_contract_flags() -> None:
    """The three subcommands accept exactly the contract flags, with the contract defaults."""
    args = parse_arguments(['transfer', '--ckpt', 'best.pt', '--eval-task', 'SR', '--out', 'cells', '--synthetic'])
    assert args.command == 'transfer' and args.eval_task == 'SR'
    # No holdout default: the checkpoint's own training holdout is the only subject a cell may query.
    assert args.holdout is None and args.n_boot == 2000 and args.seed == 0 and args.device == 'auto'
    assert args.synthetic is True and args.out == Path('cells')

    report = parse_arguments(['report', '--transfers', 'cells', '--out', 'rep'])
    assert report.command == 'report' and report.transfers == Path('cells') and report.out == Path('rep')

    chamber = parse_arguments(['chamber', '--report-dir', 'rep', '--out', 'chamber.html'])
    assert chamber.command == 'chamber' and chamber.report_dir == Path('rep')
    assert chamber.out == Path('chamber.html')


def test_cli_rejects_a_missing_subcommand_and_a_stranger_task() -> None:
    """The CLI refuses to guess: no subcommand and no off-triad eval task."""
    with pytest.raises(SystemExit):
        parse_arguments([])
    with pytest.raises(SystemExit):
        parse_arguments(['transfer', '--ckpt', 'x', '--eval-task', 'XR', '--out', 'o', '--synthetic'])


def test_the_transfer_holdout_is_the_checkpoints_own_and_nothing_else() -> None:
    """A cell may only query the subject the checkpoint held out; anything else is a training brain."""
    config = ZTEConfig.from_dict({'train': {'split': 'by_subject_loso', 'loso_holdout_subject': 'ZAB'}})
    assert resolve_transfer_holdout(config, None) == 'ZAB'
    assert resolve_transfer_holdout(config, 'ZAB') == 'ZAB'

    with pytest.raises(ValueError, match='held out of training'):
        resolve_transfer_holdout(config, 'ZDM')


def test_a_checkpoint_without_a_loso_holdout_has_no_honest_cell() -> None:
    """A non-LOSO checkpoint holds nobody out, so no subject can be labelled held-out for it."""
    config = ZTEConfig.from_dict({'train': {'split': 'by_subject_and_stimulus'}})
    with pytest.raises(ValueError, match='names no LOSO holdout'):
        resolve_transfer_holdout(config, 'ZAB')


def test_a_duplicate_hiding_behind_case_or_whitespace_still_fails_the_novelty_claim() -> None:
    """The disjointness verdict runs on a canonical text form, so normalisation can only tighten it."""
    emb, content, subjects, words, texts = _cohort(n_stimuli=6)
    disguised = {'  ' + t.upper() + ' ' for t in np.unique(texts.astype(str))}

    report = transfer_report(
        emb,
        content,
        subjects,
        words,
        texts,
        train_task='NR',
        eval_task='SR',
        holdout='ZAB',
        train_stimulus_texts=disguised,
        n_boot=100,
    )

    assert report['stimulus_overlap'] == 0
    assert report['stimulus_overlap_normalized'] == 6
    assert report['novel_stimuli'] is False


def test_menu_decomposition_recovers_a_clean_signal_under_every_rule(tmp_path: Path) -> None:
    """The diagnostic grid sits high under every scoring rule when same-sentence readings genuinely cluster."""
    rng = np.random.default_rng(5)
    n_stimuli = 24
    directions = rng.standard_normal((n_stimuli, 16)).astype(np.float32)
    lengths = np.asarray([5.0, 10.0])[rng.integers(0, 2, size=n_stimuli)]

    emb, content, subjects, words, texts = [], [], [], [], []
    for code in _SUBJECTS:
        emb.append(directions + 0.05 * rng.standard_normal(directions.shape).astype(np.float32))
        content.append(np.arange(n_stimuli))
        subjects += [code] * n_stimuli
        words.append(lengths)
        texts.append(np.array([f'NR sentence {i}' for i in range(n_stimuli)]))

    report = transfer_report(
        np.concatenate(emb),
        np.concatenate(content),
        np.array(subjects),
        np.concatenate(words),
        np.concatenate(texts),
        train_task='NR',
        eval_task='NR',
        holdout='ZAB',
        train_stimulus_texts={f'NR sentence {i}' for i in range(n_stimuli)},
        n_boot=100,
    )
    write_cell(
        tmp_path / 'cells' / cell_name('NR', 'NR', 42),
        report,
        sent_emb=np.concatenate(emb),
        content_ids=np.concatenate(content),
        subjects=np.array(subjects),
        n_words=np.concatenate(words),
        texts=np.concatenate(texts),
    )

    out = build_report(tmp_path / 'cells', tmp_path / 'rep')

    grid = out['menu_decomposition']['NR']
    for name in ('prototype_tol0', 'prototype_tol1', 'best_reading_tol0', 'best_reading_tol1'):
        assert grid[name] > 0.9, (name, grid[name])
    text = (tmp_path / 'rep' / 'PARALLAX.md').read_text(encoding='utf-8')
    assert 'decomposition (diagnostic)' in text


def test_the_enrolled_capacity_travels_through_the_report_round_trip(tmp_path: Path) -> None:
    """The enrolled menu numbers land in PARALLAX.json, CHAMBER_DATA.json and the markdown, beside the prototype's."""
    rng = np.random.default_rng(9)
    n_stimuli = 24
    directions = rng.standard_normal((n_stimuli, 16)).astype(np.float32)
    lengths = np.asarray([5.0, 10.0])[rng.integers(0, 2, size=n_stimuli)]

    emb, content, subjects, words, texts = [], [], [], [], []
    for code in _SUBJECTS:
        emb.append(directions + 0.05 * rng.standard_normal(directions.shape).astype(np.float32))
        content.append(np.arange(n_stimuli))
        subjects += [code] * n_stimuli
        words.append(lengths)
        texts.append(np.array([f'NR sentence {i}' for i in range(n_stimuli)]))

    report = transfer_report(
        np.concatenate(emb),
        np.concatenate(content),
        np.array(subjects),
        np.concatenate(words),
        np.concatenate(texts),
        train_task='NR',
        eval_task='NR',
        holdout='ZAB',
        train_stimulus_texts={f'NR sentence {i}' for i in range(n_stimuli)},
        n_boot=100,
    )
    write_cell(
        tmp_path / 'cells' / cell_name('NR', 'NR', 42),
        report,
        sent_emb=np.concatenate(emb),
        content_ids=np.concatenate(content),
        subjects=np.array(subjects),
        n_words=np.concatenate(words),
        texts=np.concatenate(texts),
    )

    build_report(tmp_path / 'cells', tmp_path / 'rep')

    # PARALLAX.json: the per-seed summary carries the enrolled capacity beside the prototype's.
    parallax = json.loads((tmp_path / 'rep' / 'PARALLAX.json').read_text(encoding='utf-8'))
    summary = parallax['cells']['NR']['NR'][0]
    assert summary['menu_capacity_enrolled'] is not None
    assert summary['menu_k2_enrolled'] is not None and summary['menu_k2_enrolled'] > 0.9

    # CHAMBER_DATA.json: the pooled capacity block gains the enrolled keys, prototype keys untouched.
    chamber = json.loads((tmp_path / 'rep' / 'CHAMBER_DATA.json').read_text(encoding='utf-8'))
    block = chamber['capacity']['NR']
    assert set(block) == {'k_at_target', 'k2_accuracy', 'gamed', 'enrolled_k_at_target', 'enrolled_k2_accuracy', 'open'}
    assert block['enrolled_k_at_target'] is not None
    assert block['enrolled_k2_accuracy'] is not None and block['enrolled_k2_accuracy'] > 0.9

    # PARALLAX.md: both lines per arm, each naming its scoring rule.
    markdown = (tmp_path / 'rep' / 'PARALLAX.md').read_text(encoding='utf-8')
    assert '## Menu capacity (in-task diagonal)' in markdown
    assert '- `NR` prototype: certified capacity K =' in markdown
    assert '- `NR` enrolled (best cross-subject reading): certified capacity K =' in markdown


def test_the_capacity_block_carries_the_gamed_verdicts(tmp_path: Path) -> None:
    """The pooled capacity block carries gamed and the open diagnostic, so the chamber's badge is reachable."""
    emb, content, subjects, words, texts = _cohort(n_stimuli=20)
    report = transfer_report(
        emb,
        content,
        subjects,
        words,
        texts,
        train_task='NR',
        eval_task='NR',
        holdout='ZAB',
        train_stimulus_texts={f'SR sentence {i}' for i in range(20)},
        n_boot=100,
    )
    write_cell(
        tmp_path / 'cells' / cell_name('NR', 'NR', 42),
        report,
        sent_emb=emb,
        content_ids=content,
        subjects=subjects,
        n_words=words,
        texts=texts,
    )

    build_report(tmp_path / 'cells', tmp_path / 'rep')

    chamber = json.loads((tmp_path / 'rep' / 'CHAMBER_DATA.json').read_text(encoding='utf-8'))
    block = chamber['capacity']['NR']
    assert set(block) >= {'k_at_target', 'k2_accuracy', 'gamed', 'enrolled_k_at_target', 'open'}
    assert isinstance(block['gamed'], bool)
    assert set(block['open']) == {'k2_accuracy', 'gamed'}
