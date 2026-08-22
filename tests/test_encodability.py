"""Tests for the per-subject encodability analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from zte.cli.encodability import _spearman, collect_subjects, correlate


def _fold(directory: Path, holdout: str, rank_pct: float, aniso: float, calib: float) -> None:
    """Writes a minimal LOSO fold (metrics + manifest) for one held-out subject."""
    run = directory / f'exp8_clip_e5_lo{holdout}'
    (run / 'evaluation').mkdir(parents=True, exist_ok=True)
    metrics = {
        'honesty': {
            'loso_holdout': holdout,
            'calibration': {'mean_lift': calib},
            'cross_subject_decode': {'targets': {'category': {'mean': 0.66, 'chance': 0.54, 'above_chance': True}}},
        },
        'neurons': {'who_variance': aniso},  # correlated with anisotropy, as in the real sweep
        'scoreboard': {
            'held_out_geometry': {'n_words': 9000, 'anisotropy': aniso, 'task_variance': 0.4},
            'held_out_retrieval': {'rank_percentile': rank_pct, 'lift_top1': 0.0},
        },
    }
    (run / 'evaluation' / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    (run / 'manifest.json').write_text(
        json.dumps({'dataset': {'omission_rate_by_subject': {holdout: 0.3}}}), encoding='utf-8'
    )


def test_spearman_matches_known_values() -> None:
    """Test the dependency-free Spearman against monotone and tied cases."""
    assert _spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert _spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Monotone non-decreasing with ties: still strongly positive, but ties pull it below 1.
    assert 0.85 < _spearman([1, 2, 3, 4], [1, 1, 2, 2]) < 0.95
    assert math.isnan(_spearman([1.0, float('nan')], [1.0, 2.0]))  # too few finite pairs


def test_calibration_rescues_the_most_collapsed_brains(tmp_path: Path) -> None:
    """The analysis recovers "calibration helps most where anisotropy is worst".

    This is the real 2026-07-24 finding (rho +0.84): a held-out brain whose embeddings collapse into a
    cone benefits most from anchor calibration.
    """
    # anisotropy and calibration lift rise together across subjects.
    for i, subj in enumerate(['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM']):
        _fold(tmp_path, subj, rank_pct=0.90, aniso=0.05 + 0.04 * i, calib=0.03 + 0.03 * i)

    rows = collect_subjects(tmp_path)
    assert len(rows) == 5
    table = correlate(rows)
    assert table['calibration_lift']['held_anisotropy'] > 0.9  # strong positive


def test_seeds_are_averaged_per_subject(tmp_path: Path) -> None:
    """Multiple seeds of one held-out subject collapse to a single averaged row."""
    _fold(tmp_path / 's42', 'ZAB', rank_pct=0.90, aniso=0.10, calib=0.05)
    _fold(tmp_path / 's43', 'ZAB', rank_pct=0.80, aniso=0.20, calib=0.15)
    # Same subject across two seed dirs -> collect from the common parent.
    for seed in ('s42', 's43'):
        for child in (tmp_path / seed).iterdir():
            child.rename(tmp_path / f'{child.name}_{seed}')

    rows = collect_subjects(tmp_path)
    assert len(rows) == 1
    assert rows[0]['n_seeds'] == 2
    assert rows[0]['held_out_rank_pct'] == pytest.approx(0.85)  # (0.90 + 0.80) / 2
