"""Tests that the honest numbers stay legible: exclusion counts, length units, provenance and the verdict basis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from zte.config import ZTEConfig
from zte.evaluation.audit.scoreboard import (
    build_scoreboard,
    cross_subject_holdout_retrieval,
    decoder_rescoring_retrieval,
    embedding_checksum,
    stimulus_median_lengths,
    within_task_retrieval,
)
from zte.evaluation.report import evaluate_representation


# --------------------------------------------------------------------------- #
# Exclusion, never zero-scoring
# --------------------------------------------------------------------------- #
def test_unanswerable_holdout_query_is_excluded_and_counted_not_zero_scored() -> None:
    """A held-out reading of a stimulus nobody else read is dropped and counted, never scored as a zero percentile."""
    # ZAB reads stimuli 0, 1, 2 but the only other subject read 0 and 1: the stim-2 query has no possible positive.
    emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # ZDM: stim0, stim1
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],  # ZAB: stim0, stim1, stim2
        ]
    )
    content = np.array([0, 1, 0, 1, 2])
    subjects = np.array(['ZDM', 'ZDM', 'ZAB', 'ZAB', 'ZAB'])

    r = cross_subject_holdout_retrieval(emb, content, subjects, 'ZAB')

    assert r is not None
    assert r['n_queries'] == 2 and r['excluded_no_positive'] == 1
    # Zero-scoring the unanswerable query would drag this to 2/3; the answerable queries are perfect.
    assert r['rank_percentile'] == 1.0
    assert r['top1'] == 1.0


def test_rescoring_query_without_its_truth_in_the_gallery_is_excluded_and_counted() -> None:
    """A rescoring query whose stimulus is absent from the gallery is dropped and counted, never zero-scored."""
    g_ids = np.arange(4)
    q_ids = np.array([0, 1, 99])  # stimulus 99 is not in the gallery
    scores = np.array(
        [
            [9.0, 1.0, 1.0, 1.0],
            [1.0, 9.0, 1.0, 1.0],
            [1.0, 1.0, 9.0, 1.0],
        ]
    )

    block = decoder_rescoring_retrieval(scores, q_ids, g_ids)

    assert block is not None
    assert block['n_queries'] == 2 and block['excluded_no_positive'] == 1
    # Zero-scoring the unanswerable query would drag this to 2/3; the answerable queries are perfect.
    assert block['rank_percentile'] == 1.0
    assert block['top1'] == 1.0


def test_stratified_rescoring_counts_its_exclusions() -> None:
    """Inside the matched-length gallery an unanswerable query is counted in `excluded_no_positive`, not scored."""
    g_ids = np.arange(4)
    gallery_n_words = np.array([8.0, 8.0, 20.0, 20.0])
    q_ids = np.array([0, 2, 99])  # stimulus 99 has no gallery entry and its 100-word stratum is empty
    query_n_words = np.array([8.0, 20.0, 100.0])
    scores = np.array(
        [
            [9.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 9.0, 1.0],
            [1.0, 1.0, 1.0, 9.0],
        ]
    )

    block = decoder_rescoring_retrieval(
        scores, q_ids, g_ids, query_n_words=query_n_words, gallery_n_words=gallery_n_words, length_tol=1
    )

    assert block is not None
    strat = block['length_stratified']
    assert strat is not None
    assert strat['n_queries'] == 2 and strat['excluded_no_positive'] == 1
    assert strat['top1'] == 1.0


# --------------------------------------------------------------------------- #
# One length unit on both sides of a stratum
# --------------------------------------------------------------------------- #
def test_stimulus_median_lengths_unifies_readings_of_one_stimulus() -> None:
    """Every reading of a stimulus gets the same stratum key: the median word count across its readings."""
    lengths = stimulus_median_lengths(np.array([4.0, 6.0, 10.0]), np.array([7, 7, 8]))

    assert lengths.tolist() == [5.0, 5.0, 10.0]


def test_rescoring_strata_use_the_gallery_unit_for_queries() -> None:
    """A reading that skipped words still lands in its own stimulus's stratum, because both sides use the median."""
    g_ids = np.arange(4)
    gallery_n_words = np.array([8.0, 8.0, 20.0, 20.0])  # stimulus-level medians
    q_ids = np.array([0, 2])
    query_n_words = np.array([4.0, 20.0])  # reading of stimulus 0 skipped half its words
    scores = np.array(
        [
            [9.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 9.0, 1.0],
        ]
    )

    block = decoder_rescoring_retrieval(
        scores, q_ids, g_ids, query_n_words=query_n_words, gallery_n_words=gallery_n_words, length_tol=1
    )

    assert block is not None
    strat = block['length_stratified']
    assert strat is not None
    # Keyed on the reading's own count the first query's stratum would be empty and the query lost.
    assert strat['n_queries'] == 2 and strat['excluded_no_positive'] == 0
    assert strat['top1'] == 1.0


def test_within_task_strata_use_the_gallery_unit_for_queries() -> None:
    """The within-task stratified cell keys queries on their stimulus's gallery median, not the reading's count."""
    query_meta = pd.DataFrame({'task': ['SR', 'SR'], 'text_id': [0, 2], 'n_words': [4, 20]})
    gallery_tasks = np.array(['SR', 'SR', 'SR', 'SR'])
    gallery_n_words = np.array([8.0, 8.0, 20.0, 20.0])
    scores = np.array(
        [
            [9.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 9.0, 1.0],
        ]
    )

    out = within_task_retrieval(
        scores, query_meta, gallery_tasks=gallery_tasks, gallery_n_words=gallery_n_words, pools=('SR',), length_tol=1
    )

    strat = out['SR']['length_stratified']
    assert strat is not None
    assert strat['n_queries'] == 2 and strat['excluded_no_positive'] == 0
    assert strat['top1'] == 1.0


def test_scoreboard_length_strata_use_stimulus_median_on_both_sides() -> None:
    """Two readings of one stimulus with different word counts share a stratum in the held-out length audit."""
    # ZAB read stim0 in 4 words where ZDM took 6; on raw counts a +/-1 stratum would separate the two readings.
    sent_emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    content = np.array([0, 1, 0, 1])
    sent_meta = pd.DataFrame({'subject': ['ZDM', 'ZDM', 'ZAB', 'ZAB']})
    config = ZTEConfig()
    config.train.split = 'by_subject_loso'
    config.train.loso_holdout_subject = 'ZAB'

    board = build_scoreboard(
        np.zeros((4, 4)),
        pd.DataFrame({'subject': ['ZDM'] * 2 + ['ZAB'] * 2}),
        [],
        sent_emb,
        content,
        sent_meta,
        config,
        sent_n_words=np.array([6.0, 10.0, 4.0, 10.0]),
    )

    strat = (board['held_out_retrieval'] or {})['length_stratified']
    assert strat is not None
    assert strat['n_queries'] == 2 and strat['excluded_no_positive'] == 0
    assert strat['top1'] == 1.0


# --------------------------------------------------------------------------- #
# Provenance inside the held-out block
# --------------------------------------------------------------------------- #
def test_embedding_checksum_tracks_the_matrix() -> None:
    """The checksum is stable for identical embeddings and changes with any change to the matrix."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(12, 6)).astype(np.float32)

    assert embedding_checksum(emb) == embedding_checksum(emb.copy())
    assert re.fullmatch(r'[0-9a-f]{16}', embedding_checksum(emb))

    changed = emb.copy()
    changed[0, 0] += 1e-3
    assert embedding_checksum(emb) != embedding_checksum(changed)


def test_provenance_travels_inside_the_held_out_block() -> None:
    """`postprocess_fit`, `alignment_fit` and the embedding checksum sit inside `held_out_retrieval` itself."""
    sent_emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    content = np.array([0, 1, 0, 1])
    sent_meta = pd.DataFrame({'subject': ['ZDM', 'ZDM', 'ZAB', 'ZAB']})
    config = ZTEConfig()
    config.train.split = 'by_subject_loso'
    config.train.loso_holdout_subject = 'ZAB'
    config.dataset.raw_align_fit = 'train'

    board = build_scoreboard(
        np.zeros((4, 4)),
        pd.DataFrame({'subject': ['ZDM'] * 2 + ['ZAB'] * 2}),
        [],
        sent_emb,
        content,
        sent_meta,
        config,
        postprocess_fit='train split',
    )

    held = board['held_out_retrieval']
    assert held is not None
    assert held['postprocess_fit'] == 'train split'
    assert held['alignment_fit'] == 'train'
    assert held['embedding_checksum'] == embedding_checksum(sent_emb)


# --------------------------------------------------------------------------- #
# The verdict reads the held-out block
# --------------------------------------------------------------------------- #
@pytest.fixture(scope='module')
def loso_metrics(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A full evaluation where the training subjects retrieve perfectly and the held-out subject anti-retrieves."""
    rng = np.random.default_rng(0)
    subjects = ('ZDM', 'ZGW', 'ZAB')
    n_stimuli, words_per_sentence, dim = 6, 4, 8

    word_rows: list[dict[str, Any]] = []
    sent_rows: list[dict[str, Any]] = []
    sent_emb_rows: list[np.ndarray] = []
    for subject in subjects:
        for stim in range(n_stimuli):
            sent_rows.append({'subject': subject, 'task': 'SR', 'category': 'c', 'n_words': words_per_sentence})
            base = np.zeros(dim)
            if subject == 'ZAB':
                # The stranger ranks its own truth last, so the held-out block sits below chance deterministically.
                base[6] = 1.0
                base[stim] = -0.5
            else:
                base[stim] = 1.0
            sent_emb_rows.append(base + rng.normal(scale=0.01, size=dim))
            for w in range(words_per_sentence):
                word_rows.append(
                    {
                        'word': f'w{stim}{w}',
                        'word_len': 2 + (stim + w) % 5,
                        'log_freq': float((stim * w) % 7),
                        'subject': subject,
                        'task': 'SR',
                        'category': 'c',
                        'sentence_idx': stim,
                        'word_idx': w,
                    }
                )

    word_meta = pd.DataFrame(word_rows)
    word_emb = rng.normal(size=(len(word_meta), 16)).astype(np.float32)
    raw_feats = rng.normal(size=(len(word_meta), 12)).astype(np.float32)
    sent_emb = np.asarray(sent_emb_rows, dtype=np.float32)
    sent_ids = np.tile(np.arange(n_stimuli), len(subjects))
    sent_meta = pd.DataFrame(sent_rows)

    config = ZTEConfig()
    config.train.split = 'by_subject_loso'
    config.train.loso_holdout_subject = 'ZAB'

    out: Path = tmp_path_factory.mktemp('loso_eval')
    return evaluate_representation(
        word_emb,
        word_meta,
        raw_feats,
        sent_emb,
        sent_ids,
        out_dir=out,
        run_name='integrity',
        sent_meta=sent_meta,
        config=config,
        tensorboard=False,
        interactive=False,
    )


def test_verdict_retrieval_clause_reads_the_held_out_block_not_pooled(loso_metrics: dict[str, Any]) -> None:
    """The retrieval clause is judged on `held_out_retrieval`: a pooled pass cannot turn the verdict green.

    Pooled `sentence_retrieval` scores the training subjects' brains alongside the stranger's, so here it passes
    while the held-out subject is at chance -- and the verdict must read the held-out number and stay red.
    """
    verdict = loso_metrics['verdict']
    assert verdict['retrieval_basis'] == 'held_out_retrieval'

    # The pooled number is far above chance; only the held-out basis can keep the clause honest.
    pooled = loso_metrics['sentence_retrieval']
    assert pooled['top1'] - pooled['chance_top1'] > 0.2

    assert verdict['retrieval_above_chance'] is False
    assert verdict['retrieval_ci'][0] < 0.0


def test_held_out_block_carries_provenance_and_no_per_query_vectors(loso_metrics: dict[str, Any]) -> None:
    """The held-out block ships its provenance keys, and the per-query hit vector never reaches metrics.json."""
    held = loso_metrics['scoreboard']['held_out_retrieval']

    assert held['postprocess_fit'] == 'none'
    assert held['alignment_fit'] == 'all'
    assert re.fullmatch(r'[0-9a-f]{16}', held['embedding_checksum'])
    assert 'top1_hits' not in held
    assert 'excluded_no_positive' in held


def test_non_loso_verdict_names_its_pooled_basis(tmp_path: Path) -> None:
    """Without a held-out subject the verdict says plainly that its retrieval clause rests on the pooled number."""
    rng = np.random.default_rng(1)
    word_meta = pd.DataFrame(
        {
            'word': [f'w{i}' for i in range(24)],
            'word_len': rng.integers(2, 9, size=24),
            'log_freq': rng.normal(size=24),
            'subject': ['ZDM'] * 12 + ['ZGW'] * 12,
            'task': ['SR'] * 24,
            'category': ['c'] * 24,
            'sentence_idx': list(range(4)) * 6,
            'word_idx': [i % 3 for i in range(24)],
        }
    )
    metrics = evaluate_representation(
        rng.normal(size=(24, 8)).astype(np.float32),
        word_meta,
        rng.normal(size=(24, 6)).astype(np.float32),
        rng.normal(size=(8, 4)).astype(np.float32),
        np.array([0, 1, 2, 3, 0, 1, 2, 3]),
        out_dir=tmp_path,
        run_name='pooled',
        sent_meta=pd.DataFrame({'subject': ['ZDM'] * 4 + ['ZGW'] * 4, 'category': ['c'] * 8}),
        config=ZTEConfig(),
        tensorboard=False,
        interactive=False,
    )

    assert metrics['verdict']['retrieval_basis'] == 'sentence_retrieval (pooled; the split holds no subject out)'
