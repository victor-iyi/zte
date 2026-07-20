"""Tests for the honesty add-ons: permutation null, held-out decode, anchor calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from zte.evaluation.audit.honesty import (
    anchor_calibration_lift,
    cross_subject_decode,
    retrieval_permutation_test,
)


def _orthogonal(d: int, seed: int) -> np.ndarray:
    """A random orthogonal `(d, d)` matrix."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    return q


def test_retrieval_permutation_detects_structure_and_null() -> None:
    """Test that retrieval_permutation detects structure and null."""
    rng = np.random.default_rng(0)
    d, n_groups, per = 12, 20, 4
    centers = rng.normal(size=(n_groups, d))
    emb, groups = [], []
    for g in range(n_groups):
        for _ in range(per):
            emb.append(centers[g] + 0.05 * rng.normal(size=d))
            groups.append(g)
    emb = np.asarray(emb, dtype=np.float32)
    groups = np.asarray(groups)

    res = retrieval_permutation_test(emb, groups, n_perm=200, seed=1)
    assert res['applicable']
    assert res['observed_top1'] > res['null_mean']
    assert res['p_value'] < 0.05 and res['above_chance']

    # Shuffle the embeddings vs their groups -> structure destroyed -> not above chance.
    perm = rng.permutation(len(emb))
    null = retrieval_permutation_test(emb[perm], groups, n_perm=200, seed=1)
    assert null['p_value'] > 0.05


def test_anchor_calibration_recovers_cohesion_under_per_subject_rotation() -> None:
    """Test that anchor_calibration recovers cohesion under per-subject rotation."""
    rng = np.random.default_rng(3)
    d, n_words, reps = 16, 30, 2
    base = rng.normal(size=(n_words, d))
    q_h = _orthogonal(d, 7)  # held-out subject 'A' lives in a rotated frame

    rows, emb = [], []
    for subj in ('A', 'B', 'C'):
        rot = q_h if subj == 'A' else np.eye(d)
        for w in range(n_words):
            for _ in range(reps):
                emb.append(base[w] @ rot + 0.02 * rng.normal(size=d))
                rows.append({'subject': subj, 'word': f'w{w}'})
    emb = np.asarray(emb, dtype=np.float32)
    meta = pd.DataFrame(rows)

    cal = anchor_calibration_lift(emb, meta, holdout='A', n_anchors=12, min_shared=6)
    assert cal['applicable']
    # 'A' is rotated away from B/C; anchor calibration should pull its held-out words back.
    assert cal['mean_cohesion_after'] > cal['mean_cohesion_before']
    assert cal['mean_lift'] > 0.1 and cal['helps']


def test_cross_subject_decode_runs_and_reports_folds() -> None:
    """Test that cross_subject_decode runs and reports folds."""
    rng = np.random.default_rng(5)
    d, n = 10, 240
    subj = rng.choice(['A', 'B', 'C'], size=n)
    cat = rng.choice(['x', 'y'], size=n)
    emb = rng.normal(size=(n, d)).astype(np.float32)
    # give category a shared linear direction so it is at least decodable in-sample
    emb[cat == 'x'] += 1.5
    meta = pd.DataFrame(
        {
            'subject': subj,
            'category': cat,
            'length_band': rng.choice(['s', 'l'], size=n),
            'word_len': rng.integers(2, 10, size=n),
            'log_freq': rng.normal(size=n),
            'word': [f'w{i % 40}' for i in range(n)],
        }
    )
    res = cross_subject_decode(emb, meta, seed=0)
    assert res['applicable'] and res['n_subjects'] == 3
    assert 'category' in res['targets']
    assert res['targets']['category']['n_folds'] == 3


def test_honesty_functions_degrade_on_tiny_input() -> None:
    """Test that honesty functions degrade on tiny input."""
    emb = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    meta = pd.DataFrame({'subject': ['A', 'A', 'A'], 'word': ['a', 'b', 'c']})
    assert not anchor_calibration_lift(emb, meta)['applicable']
    assert not cross_subject_decode(emb, meta)['applicable']
    assert not retrieval_permutation_test(emb[:2], np.array([0, 1]))['applicable']
