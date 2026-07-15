"""Tests for the confound audit (Gate 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from zte.evaluation.confound import (
    abs_spearman,
    association,
    confound_report,
    correlation_ratio,
    cramers_v,
    task_stimulus_overlap,
)


def test_cramers_v_identical_and_independent() -> None:
    """Test that Cramér's V is 1 for identical variables and 0 for independent variables."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, size=2000)
    assert cramers_v(a, a) > 0.99  # a variable is perfectly associated with itself
    b = rng.integers(0, 4, size=2000)  # independent
    assert cramers_v(a, b) < 0.1  # bias-corrected -> near zero for independence


def test_correlation_ratio_separation() -> None:
    """Test that the correlation ratio is 1 for perfectly separated groups and 0 for constant groups."""
    groups = np.array(['x'] * 100 + ['y'] * 100)
    perfectly = np.array([0.0] * 100 + [5.0] * 100)
    assert correlation_ratio(groups, perfectly) > 0.99
    constant = np.ones(200)
    assert correlation_ratio(groups, constant) == 0.0


def test_correlation_ratio_ignores_nan() -> None:
    """Test that the correlation ratio ignores NaN values."""
    groups = np.array(['x', 'x', 'y', 'y'])
    values = np.array([1.0, np.nan, 5.0, 5.0])  # nan dropped pairwise
    assert correlation_ratio(groups, values) > 0.9


def test_abs_spearman_monotonic() -> None:
    """Test that the absolute Spearman correlation is 1 for monotonic non-linear relationships and 0 for non-monotonic relationships."""
    x = np.arange(200, dtype=float)
    assert abs_spearman(x, x**2) > 0.99  # monotonic non-linear -> Spearman ~1
    assert abs_spearman(x, -(x**2)) > 0.99  # sign-invariant
    rng = np.random.default_rng(1)
    assert abs_spearman(x, rng.normal(size=200)) < 0.2


def test_association_dispatch() -> None:
    """Test that the association function dispatches to the correct measure based on the variable types."""
    df = pd.DataFrame({'cat': ['a', 'b', 'a', 'b'], 'num': [1.0, 2.0, 1.0, 2.0]})
    assert association(df, 'cat', 'cat')[1] == 'cramers_v'
    assert association(df, 'cat', 'num')[1] == 'eta'
    assert association(df, 'num', 'num')[1] == 'spearman'


def test_task_stimulus_confound_detected(small_dataset) -> None:  # noqa: ANN001
    """Test that the task/stimulus confound is detected."""
    ov = task_stimulus_overlap(small_dataset.words)
    assert ov['available']
    assert ov['fully_confounded']  # SR and NR draw disjoint corpora
    assert ov['n_shared_across_tasks'] == 0
    assert ov['cramers_v_task_stimulus'] > 0.9


def test_confound_report_shape(small_dataset) -> None:  # noqa: ANN001
    """Test that the confound report has the correct shape."""
    rep = confound_report(small_dataset.words)
    for key in (
        'n_words',
        'task_stimulus',
        'nuisance_vs_content',
        'behaviour_vs_lexical',
        'association_matrix',
    ):
        assert key in rep
    am = rep['association_matrix']
    n = len(am['factors'])
    assert np.allclose(np.array(am['matrix']), np.array(am['matrix']).T)  # symmetric
    assert all(am['matrix'][i][i] == 1.0 for i in range(n))  # unit diagonal
