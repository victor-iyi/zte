"""Frozen word-meaning vectors: the distillation target, word-type-keyed or per-occurrence contextual.

Rows are L2-normalised for cosine distillation; out-of-vocabulary words fall back to the corpus mean.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from zte.data.cache import fetch_artifact, publish_artifact
from zte.logging_utils import get_logger

if TYPE_CHECKING:
    import pandas as pd

_LOG = get_logger('data.meaning')


def _hash_vector(word: str, dim: int) -> np.ndarray:
    """A deterministic, seeded pseudo-embedding for a word (verification only)."""
    seed = abs(hash(('zte-meaning', word))) % (2**32)
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _load_vectors_file(path: Path) -> tuple[dict[str, int], np.ndarray]:
    """Loads a `word v1 v2 ...` text file or an `.npy` matrix plus its `.vocab` sidecar."""
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
        vocab (dict[str, int]): Word to row-id map; row `i` holds the vector for the word whose id is `i`.
        source (str | None): Path to a vectors file, or `None`/`'hash'` for the deterministic hash embedding.
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


def _hf_cache_path(
    model_name: str,
    layer: int,
    skey: np.ndarray,
    widx: np.ndarray,
    warr: np.ndarray,
    cache_dir: str,
) -> Path:
    """Deterministic cache file for a contextual meaning matrix, keyed by (model, layer, occurrences).

    Hashing the per-occurrence identity keeps the key stable across LOSO subjects and ablation arms, which all pass the
    same full `dataset.words`, while changing the moment the corpus or the model/layer does.
    """
    h = hashlib.sha1()
    h.update(f'{model_name}|{layer}|{len(warr)}'.encode())
    for s, wi, w in zip(skey, widx, warr):
        h.update(f'{s}\x1f{wi}\x1f{w}\x00'.encode('utf-8', 'ignore'))
    return Path(cache_dir) / f'meaning_hf_{h.hexdigest()[:16]}.npy'


def build_meaning_matrix_hf(
    words: pd.DataFrame,
    model_name: str = 'bert-base-uncased',
    *,
    layer: int = -1,
    device: str = 'cpu',
    max_length: int = 256,
    cache_dir: str = 'res/cache/meaning',
) -> tuple[np.ndarray | None, int]:
    """Builds a per-occurrence contextual meaning target aligned row-for-row with `words`.

    Where `build_meaning_matrix` gives one static vector per word type, each row here is the word's contextual hidden
    state from a frozen encoder run on its whole sentence, sub-word pieces mean-pooled, which disambiguates the
    polysemy the static target collapses. Uncovered rows (truncation, empty tokens) stay `NaN` so the loss masks them.

    A word occurrence's linguistic content is subject-independent, so the encoder runs once per unique sentence text
    and broadcasts to every subject's reading of that word -- ~12x cheaper, and the subject-invariant target LOSO wants.

    Args:
        words (pd.DataFrame): The word-level table (`ZuCoDataset.words`) with `word` and `word_idx`, and
            ideally `stimulus_key` (falls back to a `task|sentence_idx` key). Row order is positional.
        model_name (str): HuggingFace model id, e.g. `'bert-base-uncased'`.
        layer (int): Hidden-state layer index (`-1` = last; a middle layer ~7-9 aligns best with brain data).
        device (str): Torch device for the encoder pass.
        max_length (int): Tokeniser truncation length.
        cache_dir (str): Directory for the on-disk matrix cache. The frozen encoder pass is the expensive step, so a
            matching cache is reused across LOSO subjects and ablation arms, even without `transformers` installed.

    Returns:
        tuple[np.ndarray | None, int]: `((n_words, hidden) float32 with NaN for uncovered rows, hidden)`,
            or `(None, 0)` when `transformers` is unavailable (the caller falls back to the static path).
    """
    n = len(words)

    # The occurrence identity is both encoder input and cache key, so build it before any heavy import.
    if 'stimulus_key' in words.columns:
        skey = words['stimulus_key'].fillna('').astype(str).to_numpy()
    else:
        skey = (words['task'].astype(str) + '|' + words['sentence_idx'].astype(str)).to_numpy()
    widx = (words['word_idx'].to_numpy() if 'word_idx' in words.columns else np.arange(n)).astype(
        int
    )
    warr = words['word'].fillna('').astype(str).to_numpy() if 'word' in words.columns else None
    if warr is None:
        _LOG.warning('words table has no `word` column; cannot build a contextual target.')
        return None, 0

    # Layered like the dataset bundles: a Colab runtime wipes the local copy, the persistent store keeps it.
    cache = _hf_cache_path(model_name, layer, skey, widx, warr, cache_dir)
    fetch_artifact(cache)
    if cache.is_file():
        mat = np.load(cache).astype(np.float32)
        if len(mat) == n:
            _LOG.info(
                'Loaded cached contextual meaning %s (%s layer %d, %d word rows, dim %d).',
                cache.name,
                model_name,
                layer,
                n,
                mat.shape[1],
            )
            return mat, int(mat.shape[1])

    try:
        from collections import defaultdict

        import torch
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]
    except ImportError:
        _LOG.warning(
            'meaning_contextual=%r needs the optional `transformers` dependency, which is not '
            'installed; falling back to the static (word-type) meaning target.',
            model_name,
        )
        return None, 0

    # Loading the weights is part of the same optional path: unreachable weights (offline, cold cache,
    # wrong id) take the documented fallback instead of aborting the run.
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        enc = AutoModel.from_pretrained(model_name, output_hidden_states=True).eval().to(device)
    except OSError as exc:
        _LOG.warning(
            'meaning_contextual=%r could not be loaded (%r); falling back to the static (word-type) '
            'meaning target.',
            model_name,
            exc,
        )
        return None, 0
    hidden = int(enc.config.hidden_size)
    out = np.full((n, hidden), np.nan, dtype=np.float32)

    key_to_rows: dict[str, list[int]] = defaultdict(list)
    for i in range(n):
        key_to_rows[skey[i]].append(i)

    # One encoder pass per unique sentence, broadcast back to every row that read it.
    n_sent = 0
    for row_group in key_to_rows.values():
        rows = sorted(row_group, key=lambda i: widx[i])
        pos_word: dict[int, str] = {}
        for i in rows:
            pos_word.setdefault(int(widx[i]), warr[i])
        positions = sorted(pos_word)
        toks = [pos_word[p] or '[UNK]' for p in positions]
        if not toks:
            continue
        enc_in = tok(
            toks,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            hs = enc(**enc_in).hidden_states[layer][0]  # (n_subword, hidden)
        pooled: dict[int, list[torch.Tensor]] = defaultdict(list)
        for sub, wp in enumerate(enc_in.word_ids(0)):
            if wp is not None and wp < len(positions):
                pooled[wp].append(hs[sub])
        # numpy has no bfloat16 dtype, which is how some encoders run on GPU.
        vec_by_pos = {
            positions[wp]: torch.stack(vecs).mean(0).float().cpu().numpy()
            for wp, vecs in pooled.items()
        }
        for i in rows:
            v = vec_by_pos.get(int(widx[i]))
            if v is not None:
                out[i] = v
        n_sent += 1

    # L2-normalise the covered rows, restoring NaN on the rest.
    seen = np.isfinite(out).all(axis=1)
    norms = np.linalg.norm(np.nan_to_num(out), axis=1, keepdims=True)
    out = out / np.clip(norms, 1e-8, None)
    out[~seen] = np.nan
    out = out.astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    publish_artifact(cache)
    _LOG.info(
        'Contextual meaning target: %s layer %d over %d unique sentences -> %d/%d word rows covered '
        '(dim %d) -> cached %s.',
        model_name,
        layer,
        n_sent,
        int(seen.sum()),
        n,
        hidden,
        cache.name,
    )
    return out, hidden
