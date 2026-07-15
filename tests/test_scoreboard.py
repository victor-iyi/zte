"""Tests for the honest scoreboard (Gate 1)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from zte.evaluation.scoreboard import (
    build_scoreboard,
    cross_subject_holdout_retrieval,
    held_out_geometry,
    holdout_subject,
    lift_over_raw,
)


def _loso_config(holdout: str = 'ZAB') -> SimpleNamespace:
    """Test that the holdout subject is detected."""
    return SimpleNamespace(
        train=SimpleNamespace(split='by_subject_loso', loso_holdout_subject=holdout)
    )


def test_holdout_subject_detection() -> None:
    """Test that the holdout subject is detected."""
    assert holdout_subject(_loso_config('ZAB')) == 'ZAB'
    assert holdout_subject(SimpleNamespace(train=SimpleNamespace(split='by_sentence'))) is None
    assert holdout_subject(None) is None


def test_lift_over_raw_and_positive_control() -> None:
    """Test that the lift over raw and positive control are computed correctly."""
    comparison = [
        {
            'target': 'word_len',
            'representation': 'ZTE',
            'metric': 'R2',
            'linear_score': 0.30,
            'knn_score': 0.2,
        },
        {
            'target': 'word_len',
            'representation': 'raw band-power',
            'metric': 'R2',
            'linear_score': 0.10,
            'knn_score': 0.1,
        },
        {
            'target': 'word_len',
            'representation': 'noise (matched)',
            'metric': 'R2',
            'linear_score': 0.0,
            'knn_score': 0.0,
        },
        {
            'target': 'subject',
            'representation': 'ZTE',
            'metric': 'accuracy',
            'linear_score': 0.9,
            'knn_score': 0.9,
        },
        {
            'target': 'subject',
            'representation': 'raw band-power',
            'metric': 'accuracy',
            'linear_score': 0.95,
            'knn_score': 0.95,
        },
    ]
    lift = lift_over_raw(comparison)
    assert lift['word_len']['lift_linear'] == 0.2  # 0.30 - 0.10
    assert lift['word_len']['is_content'] and not lift['word_len']['is_identity']
    assert lift['subject']['lift_linear'] == -0.05
    assert lift['content_probe']['passes']  # raw reads word_len at 0.10 > floor


def test_positive_control_fails_when_raw_blind() -> None:
    """Test that the positive control fails when the raw is blind."""
    comparison = [
        {
            'target': 'word_len',
            'representation': 'raw band-power',
            'metric': 'R2',
            'linear_score': 0.001,
            'knn_score': 0.0,
        },
        {
            'target': 'log_freq',
            'representation': 'raw band-power',
            'metric': 'R2',
            'linear_score': 0.005,
            'knn_score': 0.0,
        },
    ]
    assert not lift_over_raw(comparison)['content_probe']['passes']


def test_cross_subject_holdout_retrieval_perfect_and_degenerate() -> None:
    """Test that the cross-subject holdout retrieval is perfect and degenerate."""
    # Two subjects, two stimuli; held-out ZAB's readings sit exactly on ZDM's same-stimulus readings.
    emb = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],  # ZDM: stim0, stim1
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )  # ZAB: stim0, stim1
    content = np.array([0, 1, 0, 1])
    subjects = np.array(['ZDM', 'ZDM', 'ZAB', 'ZAB'])
    r = cross_subject_holdout_retrieval(emb, content, subjects, 'ZAB')

    assert r is not None
    assert r['top1'] == 1.0 and r['n_queries'] == 2
    assert r['lift_top1'] is not None
    # Single subject -> not meaningful.
    assert (
        cross_subject_holdout_retrieval(emb[:2], content[:2], np.array(['ZAB', 'ZAB']), 'ZAB')
        is None
    )


def test_held_out_geometry_masks_to_subject() -> None:
    """Test that the held-out geometry masks to the subject."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 16))
    meta = pd.DataFrame(
        {
            'subject': ['ZAB'] * 30 + ['ZDM'] * 30,
            'word': [f'w{i}' for i in range(60)],
            'word_len': rng.integers(2, 9, size=60),
            'log_freq': rng.normal(size=60),
        }
    )
    g = held_out_geometry(emb, meta, 'ZAB')
    assert g is not None
    assert g['n_words'] == 30 and g['subject'] == 'ZAB'
    assert 0.0 <= g['anisotropy'] <= 1.0
    assert g['effective_rank_ratio'] > 0.0


def test_build_scoreboard_non_loso() -> None:
    """Test that the scoreboard is built correctly for a non-LOSO run."""
    board = build_scoreboard(
        np.zeros((4, 4)),
        pd.DataFrame({'subject': ['a'] * 4}),
        [],
        np.zeros((4, 4)),
        np.array([0, 0, 1, 1]),
        None,
        SimpleNamespace(train=SimpleNamespace(split='by_sentence')),
    )
    assert board['is_loso'] is False and 'lift_over_raw' in board
