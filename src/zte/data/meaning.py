"""Frozen word-meaning vectors: the Lexical Meaning Distillation target (MOSAIC §2).

Builds a word→vector matrix aligned to the training vocabulary, from either a real embedding
file (GloVe/fastText `word v1 v2 …` text, or `.npy` + `.vocab`) or a deterministic per-word
hash (mechanism verification only, no semantics). Rows are L2-normalised for cosine
distillation; out-of-vocabulary words fall back to the mean. See `docs/METHODS.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zte.logging_utils import get_logger

_LOG = get_logger('data.meaning')


def _hash_vector(word: str, dim: int) -> np.ndarray:
    """A deterministic, seeded pseudo-embedding for a word (verification only)."""
    seed = abs(hash(('zte-meaning', word))) % (2**32)
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _load_vectors_file(path: Path) -> tuple[dict[str, int], np.ndarray]:
    """Loads a `word v1 v2 …` text file or an `.npy` matrix + `.vocab` sidecar."""
    if path.suffix == '.npy':
        mat = np.load(path).astype(np.float32)
        vocab_path = path.with_suffix('.vocab')
        words = [
            w.strip() for w in vocab_path.read_text(encoding='utf-8').splitlines() if w.strip()
        ]
        if len(words) != len(mat):
            raise ValueError(f'{vocab_path} has {len(words)} words but {path} has {len(mat)} rows.')
        return {w: i for i, w in enumerate(words)}, mat
    vocab: dict[str, int] = {}
    rows: list[np.ndarray] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.rstrip().split(' ')
        if len(parts) < 3:  # skip a possible "n dim" header
            continue
        word, vec = parts[0], np.asarray(parts[1:], dtype=np.float32)
        if word not in vocab:
            vocab[word] = len(rows)
            rows.append(vec)
    if not rows:
        raise ValueError(f'No vectors parsed from {path}.')
    return vocab, np.vstack(rows)


def build_meaning_matrix(
    vocab: dict[str, int], source: str | None = None, dim: int = 64
) -> np.ndarray:
    """Builds a `(len(vocab), dim)` frozen, L2-normalised meaning matrix aligned to `vocab`.

    Args:
        vocab (dict[str, int]): Word→row-id map (e.g. the training word vocabulary). Row `i`
            of the returned matrix is the vector for the word whose id is `i`.
        source (str | None): Path to a vectors file, or `None`/`'hash'` for the deterministic
            hash embedding.
        dim (int): Vector dimensionality (ignored when a file sets its own width).

    Returns:
        np.ndarray: `(vocab_size, dim)` float32, rows L2-normalised. OOV words get the mean.
    """
    size = len(vocab)
    use_file = source is not None and source != 'hash'
    if use_file and not Path(source).exists():  # type: ignore[arg-type]
        _LOG.warning(
            'meaning_source %r does not exist; falling back to the hash embedding (no semantics). '
            'Provision real vectors with scripts/build_meaning_vectors.py to make this meaningful.',
            source,
        )
        use_file = False
    if use_file:
        src_vocab, src_mat = _load_vectors_file(Path(source))  # type: ignore[arg-type]
        dim = src_mat.shape[1]
        mean_vec = src_mat.mean(axis=0)
        mat = np.zeros((size, dim), dtype=np.float32)
        n_oov = 0
        for word, idx in vocab.items():
            src = src_vocab.get(word) if word else None
            if src is None:  # try lowercase, then fall back to the corpus mean
                src = src_vocab.get(word.lower()) if word else None
            mat[idx] = src_mat[src] if src is not None else mean_vec
            n_oov += int(src is None)
        _LOG.info(
            'Meaning matrix: %d words from %s (dim %d), %d OOV → mean vector.',
            size,
            source,
            dim,
            n_oov,
        )
    else:
        mat = np.zeros((size, dim), dtype=np.float32)
        for word, idx in vocab.items():
            mat[idx] = _hash_vector(word, dim)
        _LOG.info(
            'Meaning matrix: %d words via deterministic hash (dim %d, no semantics).', size, dim
        )
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.clip(norms, 1e-8, None)).astype(np.float32)
