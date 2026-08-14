"""Stratified evaluation by subject, task and sentence category -- a global score hides where an encoder is weak."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from zte.evaluation.metrics import content_retrieval, embedding_health
from zte.training.metrics import linear_probe

_REGRESSION_TARGETS = ('word_len', 'corpus_log_freq', 'log_freq')


def _encode(values: np.ndarray) -> np.ndarray:
    """Integer-codes categorical values."""
    return pd.factorize(pd.Series(values))[0]


def stratum_metrics(emb: np.ndarray, meta: pd.DataFrame, min_probe: int = 30) -> dict[str, Any]:
    """Computes a compact metric block for one stratum of word embeddings.

    Args:
        emb (np.ndarray): Word embeddings for the stratum `(n_words, embed_dim)`.
        meta (pd.DataFrame): Aligned word metadata for the stratum.
        min_probe (int): Minimum rows before linear probes are attempted.

    Returns:
        A dict with `n`, per-target probe R^2, within-stratum same-word retrieval Top-1 (and chance),
            and two geometry markers (effective-rank ratio, anisotropy).
    """
    out: dict[str, Any] = {'n': int(len(emb))}
    if len(emb) >= min_probe:
        for tgt in _REGRESSION_TARGETS:
            if tgt in meta and np.isfinite(meta[tgt].to_numpy()).sum() >= min_probe:
                score = linear_probe(emb, meta[tgt].to_numpy(), task='regression')['score']
                out[f'r2_{tgt}'] = round(float(score), 4)
    if 'word' in meta and len(emb) >= 10:
        ret = content_retrieval(emb, _encode(meta['word'].to_numpy()))
        out['retrieval_top1'] = round(float(ret.get('top1', float('nan'))), 4)
        out['retrieval_chance'] = round(float(ret.get('chance_top1', float('nan'))), 4)
    if len(emb) >= 10:
        health = embedding_health(emb)
        out['eff_rank_ratio'] = round(float(health['effective_rank_ratio']), 4)
        out['anisotropy'] = round(float(health['anisotropy']), 4)
    return out


def stratified_report(
    word_emb: np.ndarray,
    word_meta: pd.DataFrame,
    group_cols: tuple[str, ...] = ('subject', 'task', 'category', 'length_band'),
    min_stratum: int = 20,
) -> list[dict[str, Any]]:
    """Metric blocks for the whole set and for every value of each group column.

    Args:
        word_emb (np.ndarray): Word embeddings `(n_words, embed_dim)`.
        word_meta (pd.DataFrame): Aligned word metadata.
        group_cols (tuple[str, ...]): Metadata columns to stratify by (missing ones are skipped).
        min_stratum (int): Skip strata smaller than this.

    Returns:
        Tidy rows, each with `group` (`'ALL'` or the column name), `value` and the `stratum_metrics` fields.
    """
    word_meta = word_meta.reset_index(drop=True)
    rows: list[dict[str, Any]] = [{'group': 'ALL', 'value': 'all', **stratum_metrics(word_emb, word_meta)}]
    for col in group_cols:
        if col not in word_meta:
            continue
        for value, block in word_meta.groupby(col):
            idx = block.index.to_numpy()
            if len(idx) < min_stratum:
                continue
            rows.append(
                {
                    'group': col,
                    'value': str(value),
                    **stratum_metrics(word_emb[idx], block.reset_index(drop=True)),
                }
            )
    return rows


def stratified_retrieval(
    sent_emb: np.ndarray,
    sent_meta: pd.DataFrame,
    content_ids: np.ndarray,
    group_col: str = 'category',
    min_stratum: int = 6,
) -> list[dict[str, Any]]:
    """Cross-subject sentence retrieval computed *within* each group value.

    Answers "does the same sentence read by different people retrieve itself equally well across sentence categories /
    tasks?".

    Args:
        sent_emb (np.ndarray): Sentence embeddings `(n_sentences, embed_dim)`.
        sent_meta (pd.DataFrame): Aligned sentence metadata containing `group_col`.
        content_ids (np.ndarray): Stimulus/content id per sentence `(n_sentences,)`.
        group_col (str): Column to stratify by (e.g. `'category'` or `'task'`).
        min_stratum (int): Skip strata smaller than this.

    Returns:
        One row per group value with retrieval `top1`/`top5`/`mrr` and chance.
    """
    sent_meta = sent_meta.reset_index(drop=True)
    content_ids = np.asarray(content_ids)
    rows: list[dict[str, Any]] = []
    if group_col not in sent_meta:
        return rows
    for value, block in sent_meta.groupby(group_col):
        idx = block.index.to_numpy()
        if len(idx) < min_stratum:
            continue
        ret = content_retrieval(sent_emb[idx], content_ids[idx])
        rows.append(
            {
                'group': group_col,
                'value': str(value),
                'n': int(len(idx)),
                'top1': round(float(ret.get('top1', float('nan'))), 4),
                'top5': round(float(ret.get('top5', float('nan'))), 4),
                'mrr': round(float(ret.get('mrr', float('nan'))), 4),
                'chance_top1': round(float(ret.get('chance_top1', float('nan'))), 4),
            }
        )
    return rows
