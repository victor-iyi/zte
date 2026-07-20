"""Frozen sentence-text embeddings: the CLIP alignment target (`objective.name='clip'`).

Each unique ZuCo sentence is embedded once with a *frozen* text encoder, and the EEG encoder is trained to align its sentence vector
to this target via a symmetric InfoNCE loss (Radford et al., 2021, CLIP;
Défossez et al., 2023 for non-invasive brain signals). Two backends are supported so the text encoder can be A/B'd:

- **sentence-transformers** — purpose-built sentence embeddings (E5, BGE, MiniLM, …).
  The right granularity and strongest semantics for a sentence-level target.
- **hf mean-pool** — any raw HuggingFace model, mean-pooled over the attention mask.
  This is how a decoder LLM (e.g. Qwen) is turned into a sentence embedder for fast local iteration.

Embeddings are L2-normalised and cached to disk keyed by (model, backend, prefix, corpus), so a rerun never re-encodes. When neither optional
dependency is available the builder returns `None` and the caller falls back to a deterministic hash target (mechanism verification only, no semantics).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from zte.logging_utils import get_logger

_LOG = get_logger('data.text')

# Substrings that mark a decoder / generative LLM -> use the hf mean-pool backend, not sentence-transformers.
_DECODER_HINTS: tuple[str, ...] = (
    'qwen',
    'llama',
    'mistral',
    'gpt',
    'phi',
    'gemma',
    'falcon',
    'opt-',
)


def _resolve_backend(source: str, backend: str) -> str:
    """Chooses `sentence-transformers` vs `hf` for `source` when `backend='auto'`."""
    if backend != 'auto':
        return backend
    if any(h in source.lower() for h in _DECODER_HINTS):
        return 'hf'
    try:
        import sentence_transformers  # type: ignore[import-untyped]  # noqa: F401

        return 'sentence-transformers'
    except ImportError:
        return 'hf'


def _cache_path(texts: list[str], source: str, backend: str, prefix: str, cache_dir: str) -> Path:
    """Deterministic cache file for a (corpus, model, backend, prefix) text-embedding matrix."""
    h = hashlib.sha1()
    h.update(f'{source}|{backend}|{prefix}|{len(texts)}'.encode())
    for t in texts:
        h.update(t.encode('utf-8', 'ignore'))
        h.update(b'\x00')
    return Path(cache_dir) / f'text_{h.hexdigest()[:16]}.npy'


def _l2norm(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (float32)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.clip(norms, 1e-8, None)).astype(np.float32)


def _encode_sentence_transformers(
    texts: list[str], source: str, prefix: str, device: str
) -> np.ndarray:
    """Encodes with a sentence-transformers model (E5/BGE/…)."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    model = SentenceTransformer(source, device=device)
    inputs = [prefix + t for t in texts] if prefix else texts
    emb = model.encode(
        inputs,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)


def _encode_hf_meanpool(texts: list[str], source: str, prefix: str, device: str) -> np.ndarray:
    """Encodes with a raw HuggingFace model, mean-pooling the last hidden state over the attention mask.

    This is the path for decoder LLMs (Qwen etc.): a generative model has no sentence head, so its
    contextual token states are mean-pooled into a sentence vector.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

    tok = AutoTokenizer.from_pretrained(source)
    if tok.pad_token is None and tok.eos_token is not None:  # decoder LLMs often lack a pad token
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(source).eval().to(device)
    hidden = int(model.config.hidden_size)
    out = np.zeros((len(texts), hidden), dtype=np.float32)
    inputs = [prefix + t for t in texts] if prefix else texts
    with torch.no_grad():
        for start in range(0, len(inputs), 32):
            chunk = inputs[start : start + 32]
            enc = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors='pt').to(
                device
            )
            hs = model(**enc).last_hidden_state  # (b, seq, hidden)
            mask = enc['attention_mask'].unsqueeze(-1).to(hs.dtype)  # (b, seq, 1)
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            # .float(): decoder LLMs (Qwen etc.) load in bfloat16 on a GPU, and numpy has no bfloat16 --
            # cast the pooled sentence vectors to float32 before leaving the device.
            out[start : start + len(chunk)] = pooled.float().cpu().numpy()
    return out


def build_sentence_text_matrix(
    texts: list[str],
    source: str | None,
    *,
    backend: str = 'auto',
    prefix: str = '',
    device: str = 'cpu',
    cache_dir: str = 'res/cache/text',
) -> tuple[np.ndarray | None, int]:
    """Builds an L2-normalised `(len(texts), dim)` frozen text-embedding matrix, aligned to `texts` order.

    Args:
        texts (list[str]): Unique sentence strings; row `i` of the result is the embedding of `texts[i]`.
        source (str | None): Frozen text-encoder model id, or `None` for the deterministic hash fallback.
        backend (str): `'sentence-transformers'`, `'hf'`, or `'auto'`.
        prefix (str): Optional instruction prefix (e.g. E5's `'query: '`).
        device (str): Torch device for the encoder pass.
        cache_dir (str): Directory for the on-disk embedding cache.

    Returns:
        tuple[np.ndarray | None, int]: `((n_texts, dim) float32 L2-normalised, dim)`, or `(None, 0)` when
            `source` is unset / the encoder cannot be loaded (the caller then uses a hash target).
    """
    if not source or source == 'hash':
        return None, 0
    resolved = _resolve_backend(source, backend)
    cache = _cache_path(texts, source, resolved, prefix, cache_dir)
    if cache.is_file():
        mat = np.load(cache).astype(np.float32)
        if len(mat) == len(texts):
            _LOG.info(
                'Loaded cached text embeddings %s (%d sentences, dim %d).',
                cache.name,
                len(mat),
                mat.shape[1],
            )
            return _l2norm(mat), int(mat.shape[1])
    try:
        if resolved == 'sentence-transformers':
            raw = _encode_sentence_transformers(texts, source, prefix, device)
        else:
            raw = _encode_hf_meanpool(texts, source, prefix, device)
    except ImportError as exc:
        _LOG.warning(
            'CLIP text target %r (%s backend) needs an optional dependency (%s); falling back to the '
            'hash target (no semantics). Install the `meaning` group to enable it.',
            source,
            resolved,
            exc.name,
        )
        return None, 0
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, raw)
    _LOG.info(
        'Built text embeddings for %d sentences with %s (%s backend, dim %d) -> cached %s.',
        len(texts),
        source,
        resolved,
        raw.shape[1],
        cache.name,
    )
    return _l2norm(raw), int(raw.shape[1])


def mine_hard_negatives(texts: list[str], text_matrix: np.ndarray, k: int = 8) -> np.ndarray:
    """Mines *semantically-hard* negatives per sentence: surface-similar but semantically distinct.

    For each sentence it ranks the others by ``surface_overlap - semantic_cosine`` -- high word-token
    Jaccard overlap (they *look* alike) but low frozen-text-embedding cosine (they *mean* different
    things). Co-locating these in a CLIP batch forces the encoder to represent meaning rather than
    surface form, which is the novelty lever of the recipe (a distractor set no random sampler would
    produce). Runs on the small ZuCo sentence set (~hundreds of unique sentences), so the O(n^2) scan
    is cheap; it is computed once and cached with the text matrix.

    Args:
        texts (list[str]): Unique sentence strings (row `i` aligns with `text_matrix[i]`).
        text_matrix (np.ndarray): L2-normalised sentence embeddings `(n, dim)`.
        k (int): Number of hard negatives per sentence.

    Returns:
        np.ndarray: `(n, k)` int array of hard-negative sentence ids (`-1` padding when fewer exist).
    """
    n = len(texts)
    tokens = [set(t.lower().split()) for t in texts]
    sem = np.asarray(text_matrix, dtype=np.float32) @ np.asarray(text_matrix, dtype=np.float32).T
    out = np.full((n, k), -1, dtype=np.int64)
    for i in range(n):
        ti = tokens[i]
        if not ti:
            continue
        jac = np.fromiter(
            (len(ti & tj) / max(len(ti | tj), 1) for tj in tokens), dtype=np.float32, count=n
        )
        score = jac - sem[i]  # surface-similar (high jac) AND semantically distinct (low sem)
        score[i] = -np.inf
        kk = min(k, n - 1)
        if kk <= 0:
            continue
        top = np.argpartition(-score, kk - 1)[:kk]
        out[i, :kk] = top[np.argsort(-score[top])]
    return out
