"""Sentence-category labels and corpus word frequency for stratified evaluation.

Real ZuCo stimuli carry natural *sentence categories* that let us ask "does the thought embedding work equally well across sentence types?":

- **SR** (sentiment reading) sentences are POSITIVE / NEGATIVE / NEUTRAL.
- **TSR** (task-specific reading) sentences belong to a semantic *relation* type (e.g. `AWARD`, `EMPLOYER`, `WIFE`).
- **NR** (normal reading) has no intrinsic label; sentences fall back to a length band so every sentence still has a reproducible category.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from zte.logging_utils import get_logger

_LOG = get_logger('data.categories')

#: Canonical ZuCo sentiment labels (task SR / task 1).
SENTIMENT_LABELS: tuple[str, ...] = ('POSITIVE', 'NEGATIVE', 'NEUTRAL')

#: The nine semantic relations annotated in ZuCo task-specific reading (TSR).
RELATION_TYPES: tuple[str, ...] = (
    'AWARD',
    'EDUCATION',
    'EMPLOYER',
    'FOUNDER',
    'JOB_TITLE',
    'NATIONALITY',
    'POLITICAL_AFFILIATION',
    'VISITED',
    'WIFE',
)

_LABEL_TOKENS: set[str] = {s.upper() for s in SENTIMENT_LABELS} | set(RELATION_TYPES)
_WORD_RE: re.Pattern[str] = re.compile(r"[a-z0-9']+")


def normalise_text(text: str) -> str:
    """Lower-cases and collapses whitespace/punctuation for robust text joins.

    Args:
        text (str): A raw sentence string.

    Returns:
        A normalised key (lower-case, single-spaced, punctuation-stripped).
    """
    return ' '.join(_WORD_RE.findall(str(text).lower()))


def length_band(n_words: int) -> str:
    """Maps a sentence word count to a fixed, reproducible length band.

    Args:
        n_words (int): Number of words in the sentence.

    Returns:
        `short` (<=7), `medium` (8-15) or `long` (>15).
    """
    if n_words <= 7:
        return 'short'
    if n_words <= 15:
        return 'medium'
    return 'long'


def corpus_frequencies(words: pd.Series) -> pd.Series:
    """Computes normalised term frequencies over the loaded corpus.

    Frequency is `count(word) / count(most_common_word)` on lower-cased, punctuation-stripped tokens,
    so the commonest token scores `1.0` and rarer tokens tail toward `0` -- a real, reproducible replacement
    for the length proxy when building from the actual corpus.

    Args:
        words (pd.Series): Series of surface word forms (one per row).

    Returns:
        A float Series aligned to `words` in `(0, 1]`.
    """
    keys = words.astype(str).str.lower().map(lambda w: ''.join(_WORD_RE.findall(w)))
    counts = Counter(k for k in keys if k)
    if not counts:
        return pd.Series(np.full(len(words), np.nan), index=words.index)
    top = counts.most_common(1)[0][1]
    return keys.map(lambda k: counts.get(k, 0) / top).astype(float).clip(lower=1.0 / top)


def _scan_label_files(root: Path) -> dict[str, str]:
    """Best-effort scan of ZuCo label CSVs -> `{normalised_text: LABEL}`.

    Tolerant of ZuCo's irregular delimiters: every CSV under `root` is sniffed; any cell that is a known
    sentiment/relation token labels the longest text cell in its row. Absent or unparseable files simply contribute nothing.

    Args:
        root (Path): Dataset root to search recursively for `*.csv` files.

    Returns:
        A mapping from normalised sentence text to an upper-case label.
    """
    mapping: dict[str, str] = {}
    for csv_path in sorted(root.rglob('*.csv')):
        try:
            frame = pd.read_csv(csv_path, sep=None, engine='python', dtype=str, on_bad_lines='skip')
        except ValueError, OSError, pd.errors.ParserError:
            continue
        for _, row in frame.iterrows():
            cells = [str(v) for v in row.to_numpy() if isinstance(v, str) or not pd.isna(v)]
            label = next(
                (c.strip().upper() for c in cells if c.strip().upper() in _LABEL_TOKENS), None
            )
            if label is None:
                continue
            text_cells = [c for c in cells if len(_WORD_RE.findall(c)) >= 3]
            if not text_cells:
                continue
            key = normalise_text(max(text_cells, key=len))
            mapping.setdefault(key, label)
    if mapping:
        _LOG.info('Loaded %d sentence labels from CSVs under %s', len(mapping), root)
    return mapping


def sentence_categories(sentences: pd.DataFrame, root: str | Path | None = None) -> pd.DataFrame:
    """Adds `category` and `length_band` columns to a sentence table.

    `category` is, in priority order: a real sentiment/relation label joined from the corpus label files
    (when `root` is given and they parse), otherwise the task code (`SR`/`NR`/`TSR`). `length_band` is always derived from
    `n_words`. The function never raises on malformed label files; it degrades to the task-level category.

    Args:
        sentences (pd.DataFrame): Sentence metadata with `task`, `text` and `n_words`.
        root (str | Path | None): Optional dataset root to search for label CSVs.

    Returns:
        The same frame with `category`, `category_scheme` and `length_band` columns added.

    """
    out = sentences.copy()
    n_words = out['n_words'] if 'n_words' in out else out.get('text', '').str.split().str.len()
    out['length_band'] = n_words.fillna(0).astype(int).map(length_band)

    labels: dict[str, str] = {}
    if root is not None:
        try:
            labels = _scan_label_files(Path(root))
        except OSError:
            labels = {}

    keys = out['text'].map(normalise_text) if 'text' in out else pd.Series([''] * len(out))
    joined = keys.map(labels.get)
    out['category'] = joined.where(joined.notna(), out['task'].astype(str))
    out['category_scheme'] = np.where(joined.notna(), 'sentiment/relation', 'task')
    n_labeled = int(joined.notna().sum())
    if n_labeled:
        _LOG.info('Matched %d/%d sentences to real category labels.', n_labeled, len(out))
    return out
