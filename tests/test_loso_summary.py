"""Tests for the honest LOSO sweep aggregator."""

from __future__ import annotations

import json
from pathlib import Path

from zte.cli.loso_summary import collect_folds, fold_metrics, summarise


def _fold(directory: Path, holdout: str, pooled: float, held_top1: float, cat_above: bool) -> None:
    """Writes a minimal LOSO metrics.json for one fold."""
    d = directory / f'exp8_clip_e5_lo{holdout}' / 'evaluation'
    d.mkdir(parents=True, exist_ok=True)
    metrics = {
        'sentence_retrieval': {'top1': pooled},
        'neurons': {'who_variance': 0.1},
        'emergence': {'cross_subject': {'same_word': {'gap': 0.02}}},
        'honesty': {
            'loso_holdout': holdout,
            'cross_subject_decode': {
                'targets': {'category': {'mean': 0.66, 'chance': 0.54, 'above_chance': cat_above}}
            },
            'calibration': {'mean_lift': 0.08},
        },
        'scoreboard': {
            'held_out_geometry': {'n_words': 9000},
            'held_out_retrieval': {
                'top1': held_top1,
                'chance_top1': 0.00143,
                'lift_top1': round(held_top1 - 0.00143, 4),
                'rank_percentile': 0.91,
            },
            'content_probe': {'passes': False},
        },
    }
    (d / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')


def test_collect_and_summarise_reports_the_honest_trend(tmp_path: Path) -> None:
    """The aggregator separates converged/collapsed folds and averages the held-out lift."""
    _fold(tmp_path, 'ZDN', pooled=0.131, held_top1=0.0, cat_above=True)  # converged, held-out at chance
    _fold(tmp_path, 'ZGW', pooled=0.008, held_top1=0.0086, cat_above=True)  # collapsed pooled
    _fold(tmp_path, 'ZJM', pooled=0.0015, held_top1=0.0, cat_above=False)  # collapsed + degenerate
    _fold(tmp_path, 'ZKB', pooled=0.02, held_top1=0.0071, cat_above=True)  # partial

    rows = collect_folds(tmp_path)
    assert [r['holdout'] for r in rows] == ['ZDN', 'ZGW', 'ZJM', 'ZKB']  # sorted

    summary = summarise(rows)
    assert summary['n_folds'] == 4
    assert summary['converged_folds'] == 1  # only ZDN >= 0.10
    assert summary['collapsed_folds'] == 2  # ZGW and ZJM < 0.01
    assert summary['category_above_chance_folds'] == 3
    # Held-out above chance: ZGW (0.0086) and ZKB (0.0071) only.
    assert summary['held_out_above_chance_folds'] == 2
    assert summary['content_probe_pass_folds'] == 0


def test_non_loso_runs_are_ignored(tmp_path: Path) -> None:
    """A non-LOSO run (no held-out subject) is not counted as a fold."""
    d = tmp_path / 'some_run' / 'evaluation'
    d.mkdir(parents=True)
    (d / 'metrics.json').write_text(
        json.dumps({'sentence_retrieval': {'top1': 0.1}, 'honesty': {'loso_holdout': None}}),
        encoding='utf-8',
    )
    assert collect_folds(tmp_path) == []


def _arm(root: Path, arm: str, holdout: str) -> Path:
    """Writes one fold of a named arm and returns its run directory."""
    run = root / f'{arm}_lo{holdout}_s42'
    _fold(root, holdout, 0.05, 0.01, True)
    (root / f'exp8_clip_e5_lo{holdout}').rename(run)

    return run


def test_naming_run_directories_summarises_one_arm_without_pooling_its_siblings(tmp_path: Path) -> None:
    """Folds are keyed on the holdout alone, so a shared sweep root would average three levels into one trend."""
    sentence = [_arm(tmp_path, 'align_sentence_combined', h) for h in ('ZAB', 'ZDM')]
    _arm(tmp_path, 'align_token_combined', 'ZGW')

    assert len(collect_folds(sentence)) == 2
    assert len(collect_folds(tmp_path)) == 3


def test_a_run_directory_is_read_directly_rather_than_globbed_under(tmp_path: Path) -> None:
    """A sweep root holds run folders; a run folder holds its own metrics, and both must be accepted."""
    run = _arm(tmp_path, 'align_word_combined', 'ZAB')

    assert fold_metrics(run) == [(run / 'evaluation' / 'metrics.json').resolve()]
    assert fold_metrics(tmp_path) == [(run / 'evaluation' / 'metrics.json').resolve()]


def test_the_same_run_named_twice_is_counted_once(tmp_path: Path) -> None:
    """A root and one of its run folders may both be passed; the fold behind them is still one fold."""
    run = _arm(tmp_path, 'align_sentence_combined', 'ZAB')

    assert len(collect_folds([tmp_path, run])) == 1
