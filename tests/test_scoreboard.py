"""Tests for the honest held-out scoreboard."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from zte.evaluation.audit.scoreboard import (
    build_scoreboard,
    cross_subject_holdout_retrieval,
    held_out_geometry,
    holdout_subject,
    lift_over_raw,
)


def _loso_config(holdout: str = 'ZAB') -> SimpleNamespace:
    """The holdout subject is detected."""
    return SimpleNamespace(train=SimpleNamespace(split='by_subject_loso', loso_holdout_subject=holdout))


def test_holdout_subject_detection() -> None:
    """The holdout subject is detected."""
    assert holdout_subject(_loso_config('ZAB')) == 'ZAB'
    assert holdout_subject(SimpleNamespace(train=SimpleNamespace(split='by_sentence'))) is None
    assert holdout_subject(None) is None


def test_lift_over_raw_and_positive_control() -> None:
    """The lift over raw and positive control are computed correctly."""
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
    """The positive control fails when the raw is blind."""
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
    """The cross-subject holdout retrieval is perfect and degenerate."""
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
    assert cross_subject_holdout_retrieval(emb[:2], content[:2], np.array(['ZAB', 'ZAB']), 'ZAB') is None


def test_held_out_geometry_masks_to_subject() -> None:
    """The held-out geometry masks to the subject."""
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
    """The scoreboard is built correctly for a non-LOSO run."""
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


def test_positive_control_uses_genuinely_raw_band_power() -> None:
    """The content control probes raw band power, immune to a whitening normaliser.

    The regression: the old control probed the model's *normalised* input, and a whitening normaliser
    (riemannian/zscore_subject) strips the amplitude that word_len rides on, so it read ~0 and branded
    the whole content probe broken even when raw EEG carries the signal.
    """
    from zte.evaluation.audit.scoreboard import raw_content_positive_control

    rng = np.random.default_rng(0)
    n = 1500
    word_len = rng.integers(1, 12, n).astype(float)
    # Band power whose amplitude scales with word length (the real ZuCo reading effect), plus omissions.
    bp = (word_len[:, None, None] * 0.3 + rng.normal(0, 1, (n, 8, 105))).astype(np.float32)
    bp[rng.random(n) < 0.1] = np.nan
    meta = pd.DataFrame({'word_len': word_len, 'log_freq': rng.normal(0, 1, n)})

    control = raw_content_positive_control(bp, meta)
    assert control is not None
    assert control['source'] == 'raw band-power'
    assert control['passes']  # the signal is present in raw band power
    assert control['per_target_r2']['word_len'] > 0.02

    # A whitening normaliser that centres each row destroys the amplitude signal -> the OLD false FAIL.
    from zte.training.metrics import linear_probe

    flat = np.nan_to_num(bp.reshape(n, -1))
    row_centred = flat - flat.mean(axis=1, keepdims=True)
    assert linear_probe(row_centred, word_len, task='regression')['score'] < 0.02

    # build_scoreboard must adopt the raw-band-power control over the normalised-features fallback.
    board = build_scoreboard(
        np.zeros((n, 4)),
        meta.assign(subject=['a'] * n),
        [],
        np.zeros((4, 4)),
        np.array([0, 0, 1, 1]),
        None,
        SimpleNamespace(train=SimpleNamespace(split='by_sentence')),
        word_band_power=bp,
    )
    assert board['lift_over_raw']['content_probe']['source'] == 'raw band-power'
    assert board['lift_over_raw']['content_probe']['passes']


def test_positive_control_not_applicable_without_band_power() -> None:
    """A raw-signal frontend (no band power) yields no positive control rather than crashing."""
    from zte.evaluation.audit.scoreboard import raw_content_positive_control

    assert raw_content_positive_control(None, pd.DataFrame({'word_len': [1, 2, 3]})) is None
