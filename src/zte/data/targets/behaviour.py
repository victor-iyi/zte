"""Per-word eye-tracking as a target matrix, so an auxiliary head can predict reading difficulty from the embedding.

Covers the duration measures, derived `regression_time` (`GPT - GD`), `n_fixations`/`mean_pupil` and binary
`is_omitted`. Duration and count targets are log1p-then-z-scored; missing-by-design cells stay `NaN` and are masked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Targets that are binary (classification) rather than regressed.
_BINARY = frozenset({'is_omitted'})
# Targets we synthesise rather than read directly.
_DERIVED = frozenset({'regression_time'})


def _column(words: 'pd.DataFrame', name: str) -> np.ndarray:
    """Returns a float column, synthesising derived targets; NaN where unavailable."""
    if name == 'regression_time':
        if {'GPT', 'GD'}.issubset(words.columns):
            return (words['GPT'].astype('float64') - words['GD'].astype('float64')).clip(lower=0.0).to_numpy()
        return np.full(len(words), np.nan)
    if name in words.columns:
        return words[name].astype('float64').to_numpy()
    return np.full(len(words), np.nan)


def build_behaviour_matrix(words: 'pd.DataFrame', targets: tuple[str, ...]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Builds a `(n_words, n_targets)` behaviour target matrix aligned to `words` rows.

    Args:
        words (pd.DataFrame): The word-level metadata table (`ZuCoDataset.words`).
        targets (tuple[str, ...]): Which behaviour signals to include.

    Returns:
        tuple[np.ndarray, list[str], np.ndarray]: `(matrix, names, is_binary)` -- a `(n_words, n_targets)` float32
            matrix (NaN where missing), the resolved names with unavailable targets dropped, and a bool array
            marking the classification targets.
    """
    cols: list[np.ndarray] = []
    names: list[str] = []
    binary: list[bool] = []
    for t in targets:
        raw = _column(words, t)
        if not np.isfinite(raw).any():
            continue  # target unavailable in this dataset
        if t in _BINARY:
            col = raw.astype(np.float32)
        else:
            # log1p compresses the right skew; the z-score uses the finite (present) rows only.
            finite = np.isfinite(raw)
            logv = np.where(finite, np.log1p(np.clip(raw, 0.0, None)), np.nan)
            mu = np.nanmean(logv)
            sd = np.nanstd(logv)
            col = ((logv - mu) / (sd if sd > 1e-8 else 1.0)).astype(np.float32)
        cols.append(col)
        names.append(t)
        binary.append(t in _BINARY)
    if not cols:
        return np.zeros((len(words), 0), dtype=np.float32), [], np.zeros(0, dtype=bool)
    return np.stack(cols, axis=1).astype(np.float32), names, np.asarray(binary, dtype=bool)
