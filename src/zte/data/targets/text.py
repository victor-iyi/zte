"""Frozen sentence-text embeddings: the CLIP alignment target for `objective.name='clip'`.

Two backends let the text encoder be A/B'd: sentence-transformers for purpose-built sentence embeddings, and hf
mean-pool to turn a decoder LLM into a sentence embedder. Embeddings are L2-normalised and cached on disk keyed by
(model, backend, prefix, corpus). With neither optional dependency the builder returns `None` and the caller falls
back to a deterministic hash target that carries no semantics.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from zte.data.cache import fetch_artifact, publish_artifact
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
    """Encodes with a sentence-transformers model (E5/BGE/...)."""
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

    The path for decoder LLMs, which have no sentence head, so their contextual token states are pooled instead.
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

    # Chunked to bound peak memory.
    with torch.no_grad():
        for start in range(0, len(inputs), 32):
            chunk = inputs[start : start + 32]
            enc = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors='pt').to(
                device
            )
            hs = model(**enc).last_hidden_state  # (b, seq, hidden)
            mask = enc['attention_mask'].unsqueeze(-1).to(hs.dtype)  # (b, seq, 1)
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            # numpy has no bfloat16, which is how decoder LLMs load on a GPU.
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

    # Reuse the cached matrix whenever it still matches the corpus length. Layered like the dataset
    # bundles: a Colab runtime wipes the local copy, the persistent store keeps it.
    resolved = _resolve_backend(source, backend)
    cache = _cache_path(texts, source, resolved, prefix, cache_dir)
    fetch_artifact(cache)
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
    except OSError as exc:
        # The package is installed but the weights are not reachable (offline, cold HF cache, bad id).
        # Same outcome as a missing package, so take the same documented fallback rather than killing
        # the run -- with a loud warning, because a hash target makes the result meaningless.
        _LOG.warning(
            'CLIP text target %r could not be loaded (%r); falling back to the hash target (NO '
            'semantics -- results are not meaningful). Pre-download the model or fix connectivity.',
            source,
            exc,
        )
        return None, 0

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, raw)
    publish_artifact(cache)
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
    """Mines semantically-hard negatives per sentence: surface-similar but semantically distinct.

    Sentences are ranked by `surface_overlap - semantic_cosine`, so they look alike but mean different things.
    Co-locating them in a CLIP batch forces the encoder to represent meaning rather than surface form. The O(n^2) scan
    is cheap on ZuCo's few hundred unique sentences and is cached with the text matrix.

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

        # Word-token Jaccard overlap against every other sentence.
        jac = np.fromiter(
            (len(ti & tj) / max(len(ti | tj), 1) for tj in tokens), dtype=np.float32, count=n
        )

        # High overlap minus high cosine: surface-similar and semantically distinct.
        score = jac - sem[i]
        score[i] = -np.inf
        kk = min(k, n - 1)
        if kk <= 0:
            continue
        top = np.argpartition(-score, kk - 1)[:kk]
        out[i, :kk] = top[np.argsort(-score[top])]
    return out
