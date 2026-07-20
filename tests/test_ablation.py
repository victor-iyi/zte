"""Tests for the single-variable ablation harness (Gate 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zte.config import ZTEConfig
from zte.evaluation.ablation import (
    _set_dotted,
    diff_scoreboards,
    grid_configs,
    single_variable_configs,
)


def test_set_dotted_changes_exactly_one_field() -> None:
    """Test that set dotted changes exactly one field."""
    base = ZTEConfig()
    changed = _set_dotted(base, 'objective.subject_adversary_weight', 1.0)
    assert changed.objective.subject_adversary_weight == 1.0
    # Nothing else moved.
    assert changed.model == base.model
    assert changed.dataset == base.dataset
    assert base.objective.subject_adversary_weight == 0.0  # base untouched


def test_set_dotted_rejects_bad_paths() -> None:
    """Test that set dotted rejects bad paths."""
    base = ZTEConfig()
    for bad in ('nope', 'objective.not_a_field', 'ghost.field'):
        with pytest.raises(ValueError):
            _set_dotted(base, bad, 1)


def test_single_variable_configs_sweep_and_coerce() -> None:
    """Test that single variable configs sweep and coerce."""
    base = ZTEConfig()
    base.run_name = 'sota'
    pairs = single_variable_configs(base, 'model.factored', ['false', 'true'])
    assert [t for t, _ in pairs] == ['model_factored=false', 'model_factored=true']
    assert pairs[0][1].model.factored is False and pairs[1][1].model.factored is True
    assert pairs[1][1].run_name == 'sota__model_factored=true'
    # numeric coercion
    p = single_variable_configs(base, 'objective.meaning_distill_weight', ['0', '0.5'])
    assert p[1][1].objective.meaning_distill_weight == 0.5


def test_grid_configs_takes_cartesian_product() -> None:
    """A multi-knob grid emits one config per value combination, differing only in the swept fields."""
    base = ZTEConfig()
    base.run_name = 'sota'
    pairs = grid_configs(
        base,
        [
            ('model.spatial_encoding', ['none', 'spherical_harmonics']),
            ('objective.meaning_distill_weight', ['0', '0.5']),
        ],
    )
    assert len(pairs) == 4  # 2 x 2
    tags = [t for t, _ in pairs]
    assert tags[0] == 'model_spatial_encoding=none__objective_meaning_distill_weight=0'
    # Each arm sets exactly its combination; nothing else moved off the base.
    by_tag = {t: c for t, c in pairs}
    hi = by_tag['model_spatial_encoding=spherical_harmonics__objective_meaning_distill_weight=0.5']
    assert hi.model.spatial_encoding == 'spherical_harmonics'
    assert hi.objective.meaning_distill_weight == 0.5
    assert hi.run_name == (
        'sota__model_spatial_encoding=spherical_harmonics__objective_meaning_distill_weight=0.5'
    )
    assert hi.train == base.train  # untouched sections identical
    # The base is never mutated by the sweep.
    assert base.model.spatial_encoding == 'none'


def test_grid_configs_single_knob_matches_single_variable() -> None:
    """One (knob, values) spec is exactly the single-variable sweep."""
    base = ZTEConfig()
    base.run_name = 'sota'
    grid = grid_configs(base, [('model.factored', ['false', 'true'])])
    single = single_variable_configs(base, 'model.factored', ['false', 'true'])
    assert [t for t, _ in grid] == [t for t, _ in single]
    assert [c.run_name for _, c in grid] == [c.run_name for _, c in single]


def _write_metrics(path: Path, retr_top1: float, lift_word_len: float, effrank: float) -> None:
    """Write metrics to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                'scoreboard': {
                    'is_loso': True,
                    'lift_over_raw': {
                        'word_len': {'lift_linear': lift_word_len, 'is_content': True},
                        'content_probe': {'passes': True},
                    },
                    'held_out_geometry': {
                        'effective_rank_ratio': effrank,
                        'anisotropy': 0.1,
                        'content_variance': 0.2,
                    },
                    'held_out_retrieval': {'top1': retr_top1, 'lift_top1': retr_top1 - 0.1},
                }
            }
        ),
        encoding='utf-8',
    )


def test_diff_scoreboards_isolates_delta(tmp_path: Path) -> None:
    """Test that diff scoreboards isolates delta."""
    a = tmp_path / 'base' / 'evaluation' / 'metrics.json'
    b = tmp_path / 'var' / 'evaluation' / 'metrics.json'
    _write_metrics(a, retr_top1=0.10, lift_word_len=-0.5, effrank=0.10)
    _write_metrics(b, retr_top1=0.18, lift_word_len=-0.2, effrank=0.15)
    diff = diff_scoreboards(a, b)
    assert diff['held_out_retrieval_top1_delta'] == pytest.approx(0.08, abs=1e-6)
    assert diff['lift_over_raw_delta']['word_len'] == pytest.approx(0.3, abs=1e-6)
    assert diff['held_out_effrank_delta'] == pytest.approx(0.05, abs=1e-6)
