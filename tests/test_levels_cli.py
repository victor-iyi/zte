"""`zte-levels`: the granularity ablation read off disk -- fold aggregation, the floors, and what is not measured."""

import json
import statistics
from pathlib import Path
from typing import Any

import pytest
import yaml

from zte.cli import levels

# The three folds every fixture level is measured over; small enough that the sample sd is checkable by hand.
FOLDS: tuple[str, ...] = ('ZAB', 'ZDM', 'ZDN')

# Levers the resolved config carries, per level -- what `level_of` reads to decide which rung a run aligned at.
LEVERS: dict[str, dict[str, float]] = {
    'sentence': {'token_weight': 0.0, 'lexical_weight': 0.0},
    'word': {'token_weight': 0.0, 'lexical_weight': 1.0},
    'token': {'token_weight': 1.0, 'lexical_weight': 0.0},
}


def write_run(
    root: Path,
    level: str,
    fold: str,
    *,
    rank_percentile: float,
    top1: float,
    n_queries: int = 700,
    chance: float = 0.0285,
    audit: dict[str, Any] | None = None,
    run_name: str | None = None,
    stratified: bool = True,
) -> Path:
    """Writes one evaluated run's artifacts -- metrics, config and optionally its rebaseline audit."""
    run_dir = root / (run_name or f'align_{level}_combined_lo{fold}_s42')
    held: dict[str, Any] = {
        'top1': top1,
        'rank_percentile': rank_percentile,
        'chance_top1': 1 / 700,
        'n_queries': n_queries,
        'postprocess_fit': 'train split',
    }
    if stratified:
        held['length_stratified'] = {
            'top1': top1,
            'rank_percentile': rank_percentile,
            'chance_top1': chance,
            'n_queries': n_queries,
        }

    metrics = {
        'honesty': {'loso_holdout': fold},
        'embedding_health': {'effective_rank_ratio': 0.16},
        'scoreboard': {'holdout_subject': fold, 'held_out_retrieval': held},
    }
    (run_dir / 'evaluation').mkdir(parents=True, exist_ok=True)
    (run_dir / 'evaluation' / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    (run_dir / 'config.yaml').write_text(
        yaml.safe_dump({'objective': LEVERS[level], 'train': {'loso_holdout_subject': fold}}), encoding='utf-8'
    )

    if audit is not None:
        (run_dir / 'rebaseline').mkdir(parents=True, exist_ok=True)
        (run_dir / 'rebaseline' / 'rebaseline.json').write_text(json.dumps(audit), encoding='utf-8')

    return run_dir


def length_audit(*, rank_percentile: float, top1: float = 0.0214, tol: int = 2) -> dict[str, Any]:
    """A `rebaseline.json` carrying only the word-count oracle."""
    return {
        'floor_comparison': {'oracle_tol': tol, 'clears_floor': False},
        'length_oracle': {str(tol): {'rank_percentile': rank_percentile, 'top1': top1, 'tol': float(tol)}},
    }


def piece_audit(*, rank_percentile: float, gate_top1: float = 0.0331) -> dict[str, Any]:
    """The same audit with the sub-word piece oracle a token-level number has to clear."""
    audit = length_audit(rank_percentile=rank_percentile)
    audit['piece_oracle'] = {
        'gate_signature': 'total',
        'gate_top1': gate_top1,
        'ceiling_signature': 'profile',
        'ceiling_top1': 0.996,
    }

    return audit


@pytest.fixture
def sentence_tree(tmp_path: Path) -> Path:
    """Three sentence-level folds whose rank percentiles sit below a measured word-count floor."""
    root = tmp_path / 'experiments'
    for fold, rank, top1 in zip(FOLDS, (0.9150, 0.9238, 0.9326), (0.03, 0.04, 0.05), strict=True):
        write_run(root, 'sentence', fold, rank_percentile=rank, top1=top1, audit=length_audit(rank_percentile=0.9525))

    return root


def table_for(root: Path) -> dict[str, Any]:
    """Builds the cross-level payload from a fixture tree, with a small bootstrap so the test stays fast."""
    return levels.build_table(levels.discover_runs(None, root, '*'), n_boot=200, seed=0)


def block_for(table: dict[str, Any], level: str) -> dict[str, Any]:
    """The one block for a level, so a test never silently reads the wrong row."""
    (block,) = [entry for entry in table['levels'] if entry['level'] == level]

    return block


# --------------------------------------------------------------------------- #
# Aggregation across folds
# --------------------------------------------------------------------------- #


def test_the_mean_and_spread_are_taken_over_folds_with_a_sample_standard_deviation(sentence_tree: Path) -> None:
    """A twelve-fold sweep is a sample of subjects, so the spread is the n-1 sd, not the population one."""
    ranks = [0.9150, 0.9238, 0.9326]
    block = block_for(table_for(sentence_tree), 'sentence')

    assert block['n_folds'] == 3
    assert block['folds'] == list(FOLDS)
    assert block['rank_percentile'] == pytest.approx(statistics.mean(ranks))
    assert block['rank_percentile_sd'] == pytest.approx(statistics.stdev(ranks))

    # The population sd would under-report the spread; at n=3 the two differ by more than a rounding error.
    assert block['rank_percentile_sd'] != pytest.approx(statistics.pstdev(ranks), abs=1e-6)


def test_top1_is_a_hit_count_out_of_the_queries_scored_with_its_exact_binomial_tail(sentence_tree: Path) -> None:
    """A rate against a gallery of hundreds hides that chance alone expects a handful of hits, so counts are printed."""
    block = block_for(table_for(sentence_tree), 'sentence')

    # 700 queries per fold at Top-1 rates of 0.03, 0.04 and 0.05.
    assert block['n_queries'] == 2100
    assert block['hits_top1'] == 21 + 28 + 35
    assert block['expected_hits_top1'] == pytest.approx(2100 / 700)
    assert block['top1'] == pytest.approx(84 / 2100)
    assert 0.0 <= block['top1_p'] < 1e-6


def test_a_fold_measured_twice_enters_the_mean_once(sentence_tree: Path) -> None:
    """A re-run of one held-out subject must not weight that subject twice in the level's mean."""
    write_run(
        sentence_tree,
        'sentence',
        'ZAB',
        rank_percentile=0.5,
        top1=0.5,
        audit=length_audit(rank_percentile=0.9525),
        run_name='align_sentence_combined_loZAB_s7',
    )
    block = block_for(table_for(sentence_tree), 'sentence')

    assert block['n_folds'] == 3
    assert block['rank_percentile'] == pytest.approx(statistics.mean([0.9150, 0.9238, 0.9326]))


def test_the_level_is_read_from_the_config_not_the_directory_name(tmp_path: Path) -> None:
    """A run directory can be renamed; the lever the run actually trained with cannot."""
    root = tmp_path / 'experiments'
    for fold in FOLDS:
        write_run(
            root,
            'word',
            fold,
            rank_percentile=0.92,
            top1=0.04,
            audit=length_audit(rank_percentile=0.9525),
            run_name=f'align_token_mislabelled_lo{fold}',
        )
    table = table_for(root)

    assert [block['level'] for block in table['levels']] == ['word']


# --------------------------------------------------------------------------- #
# The floors
# --------------------------------------------------------------------------- #


def test_a_level_below_its_length_floor_is_verdicted_below_it(sentence_tree: Path) -> None:
    """The whole point of the table: an encoder under a brain-free floor reads as under it, never as a winner."""
    table = table_for(sentence_tree)
    block = block_for(table, 'sentence')

    assert block['length_floor']['rank_percentile'] == pytest.approx(0.9525)
    assert block['clears_length_floor'] is False
    assert block['clears_floor'] is False
    assert table['verdict']['any_clears_floor'] is False

    markdown = levels.render_markdown(table)
    assert 'NO -- below a brain-free floor' in markdown
    assert 'No level clears the brain-free floor' in table['verdict']['reading']


def test_a_level_above_every_measured_floor_says_so(tmp_path: Path) -> None:
    """The comparison is a measurement, not a verdict of failure -- an encoder above its floors reads as above them."""
    root = tmp_path / 'experiments'
    for fold in FOLDS:
        write_run(root, 'sentence', fold, rank_percentile=0.99, top1=0.4, audit=length_audit(rank_percentile=0.50))
    table = table_for(root)
    block = block_for(table, 'sentence')

    assert block['clears_length_floor'] is True
    assert block['beats_oracle_floor'] is True
    assert block['clears_floor'] is True
    assert table['verdict']['levels_clearing_floor'] == ['sentence']
    assert 'NO -- below a brain-free floor' not in levels.render_markdown(table)


def test_a_level_with_no_rebaseline_reads_as_floor_not_measured_and_names_what_is_missing(tmp_path: Path) -> None:
    """A `--` cell must never be readable as a pass: the row says which artifact was absent."""
    root = tmp_path / 'experiments'
    for fold in FOLDS:
        write_run(root, 'sentence', fold, rank_percentile=0.9238, top1=0.04)
    table = table_for(root)
    block = block_for(table, 'sentence')

    assert block['clears_floor'] is None
    assert 'clears_length_floor' not in block
    assert any('rebaseline/rebaseline.json' in name for name in block['missing'])

    markdown = levels.render_markdown(table)
    assert 'floor not measured' in markdown
    assert 'rebaseline/rebaseline.json' in markdown
    assert 'Floor not measured for sentence' in table['verdict']['reading']


def test_a_floor_measured_on_only_some_folds_names_the_folds_that_lack_it(tmp_path: Path) -> None:
    """A mean over the folds that have a floor would otherwise read as a floor measured on every fold."""
    root = tmp_path / 'experiments'
    for index, fold in enumerate(FOLDS):
        write_run(
            root,
            'sentence',
            fold,
            rank_percentile=0.92,
            top1=0.04,
            audit=None if index == 0 else length_audit(rank_percentile=0.9525),
        )
    block = block_for(table_for(root), 'sentence')

    assert block['length_floor']['n_folds'] == 2
    assert block['clears_length_floor'] is False
    assert any(name == 'rebaseline/rebaseline.json:length_oracle (ZAB)' for name in block['missing'])
    assert 'ZAB' in levels.render_markdown(table_for(root))


def test_a_token_level_without_its_piece_oracle_is_reported_rather_than_dropped_or_passed(tmp_path: Path) -> None:
    """A token number quoted without its sub-word floor is not evidence, so the number is withheld -- not the row."""
    root = tmp_path / 'experiments'
    for fold in FOLDS:
        write_run(root, 'token', fold, rank_percentile=0.9286, top1=0.0475, audit=length_audit(rank_percentile=0.9525))
    table = table_for(root)
    block = block_for(table, 'token')

    assert block['unmeasured'] is True
    assert block['n_folds'] == 3
    assert block['clears_floor'] is None
    assert 'rank_percentile' not in block
    assert any('piece_oracle' in name for name in block['missing'])

    markdown = levels.render_markdown(table)
    assert '| token |' in markdown
    assert 'piece_oracle' in markdown


def test_the_token_level_is_scored_against_its_piece_oracle_when_it_is_there(tmp_path: Path) -> None:
    """With the sub-word floor measured, the token row carries its number and both floors beside it."""
    root = tmp_path / 'experiments'
    for fold in FOLDS:
        write_run(root, 'token', fold, rank_percentile=0.9286, top1=0.0475, audit=piece_audit(rank_percentile=0.9525))
    block = block_for(table_for(root), 'token')

    assert block['oracle_floor']['top1'] == pytest.approx(0.0331)
    assert block['oracle_floor']['ceiling_top1'] == pytest.approx(0.996)
    assert block['beats_oracle_floor'] is True
    assert block['clears_length_floor'] is False

    # Above the piece gate but under the length floor is still under a brain-free floor.
    assert block['clears_floor'] is False


def test_the_nominally_highest_level_is_not_presented_as_the_winner(tmp_path: Path) -> None:
    """Token highest while no level clears its floor is the confound signature, and the reading has to say so."""
    root = tmp_path / 'experiments'
    for fold, rank in zip(FOLDS, (0.9150, 0.9238, 0.9326), strict=True):
        write_run(root, 'sentence', fold, rank_percentile=rank, top1=0.04, audit=length_audit(rank_percentile=0.9525))
        write_run(
            root, 'word', fold, rank_percentile=rank - 0.005, top1=0.04, audit=length_audit(rank_percentile=0.9525)
        )
        write_run(
            root, 'token', fold, rank_percentile=rank + 0.005, top1=0.0475, audit=piece_audit(rank_percentile=0.9525)
        )
    table = table_for(root)
    verdict = table['verdict']

    assert [block['level'] for block in table['levels']] == ['token', 'word', 'sentence']
    assert verdict['nominal_best'] == 'token'
    assert verdict['nominal_best_clears_floor'] is False
    assert verdict['headline_gallery'] == 'length_stratified'
    assert 'confound signature' in verdict['reading']


# --------------------------------------------------------------------------- #
# The length-stratified twin
# --------------------------------------------------------------------------- #


def test_the_length_stratified_twin_carries_its_own_gallery_and_chance(sentence_tree: Path) -> None:
    """A matched-length gallery is a different gallery, so its chance rate and hit count are reported apart."""
    block = block_for(table_for(sentence_tree), 'sentence')
    cell = block['length_stratified']

    assert cell['chance_top1'] == pytest.approx(0.0285)
    assert cell['gallery_size'] == round(1 / 0.0285)
    assert cell['expected_hits_top1'] == pytest.approx(0.0285 * 2100)
    assert block['length_floor_gallery'] == 'length_stratified'


def test_a_level_missing_the_stratified_twin_says_so_instead_of_mixing_galleries(tmp_path: Path) -> None:
    """Folds scored on different galleries are not comparable, so a partial twin is dropped whole and named."""
    root = tmp_path / 'experiments'
    for index, fold in enumerate(FOLDS):
        write_run(
            root,
            'sentence',
            fold,
            rank_percentile=0.92,
            top1=0.04,
            audit=length_audit(rank_percentile=0.9525),
            stratified=index > 0,
        )
    block = block_for(table_for(root), 'sentence')

    assert block['length_stratified'] is None
    assert any('length_stratified' in name and 'ZAB' in name for name in block['missing'])
    assert block['length_floor_gallery'] == 'full'


# --------------------------------------------------------------------------- #
# The driver itself
# --------------------------------------------------------------------------- #


def test_both_artifacts_are_written_and_carry_their_provenance(sentence_tree: Path, tmp_path: Path) -> None:
    """The table is a deliverable: JSON for a notebook, Markdown for a report, and what it was built from."""
    out = tmp_path / 'levels'
    json_path, md_path = levels.write_table(table_for(sentence_tree), out)

    payload = json.loads(json_path.read_text(encoding='utf-8'))
    assert payload['provenance']['tool'] == 'zte-levels'
    assert payload['provenance']['n_runs'] == 3
    assert {entry['fold'] for entry in payload['runs']} == set(FOLDS)
    assert 'Runs read' in md_path.read_text(encoding='utf-8')


def test_a_second_run_over_unchanged_artifacts_rebuilds_nothing(
    sentence_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard exists so a re-run costs a hash rather than a rebuild; it has to actually skip the work."""
    out = tmp_path / 'levels'
    argv = ['zte-levels', '--root', str(sentence_tree), '--pattern', '*', '--out', str(out)]
    monkeypatch.setattr('sys.argv', argv)
    levels.main()

    def refuse(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError('The table was rebuilt even though nothing it reads changed.')

    monkeypatch.setattr(levels, 'build_table', refuse)
    levels.main()

    assert (out / 'levels.json').is_file()


def test_a_changed_artifact_rebuilds_the_table(
    sentence_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale table read as a fresh result is the failure the guard exists to prevent, so it must notice a change."""
    out = tmp_path / 'levels'
    argv = ['zte-levels', '--root', str(sentence_tree), '--pattern', '*', '--out', str(out)]
    monkeypatch.setattr('sys.argv', argv)
    levels.main()

    write_run(
        sentence_tree,
        'sentence',
        'ZGW',
        rank_percentile=0.93,
        top1=0.06,
        audit=length_audit(rank_percentile=0.9525),
    )
    levels.main()

    payload = json.loads((out / 'levels.json').read_text(encoding='utf-8'))
    assert block_for(payload, 'sentence')['n_folds'] == 4


def test_a_run_that_names_no_level_is_skipped_rather_than_guessed(tmp_path: Path) -> None:
    """Guessing a level would put a run in the wrong row, which is worse than leaving it out and saying so."""
    root = tmp_path / 'experiments'
    run = write_run(root, 'sentence', 'ZAB', rank_percentile=0.92, top1=0.04, run_name='mystery_run')
    (run / 'config.yaml').unlink()

    assert levels.read_fold(run) is None


def test_nothing_to_read_is_an_error_not_an_empty_table(tmp_path: Path) -> None:
    """An empty table would be indistinguishable from a measured null, so the driver refuses to write one."""
    with pytest.raises(SystemExit):
        levels.discover_runs(None, None, '*')

    with pytest.raises(SystemExit, match='named an alignment level'):
        levels.build_table([tmp_path])
