"""Text encoders that define the language side of the EEG-OT-CLIP shared space.

Provides a small abstract interface (:class:`TextEncoder`) plus two backends:

- :class:`HashTextEncoder` — deterministic, dependency-free Gaussian embeddings
  (used in CI / when ``transformers`` is unavailable).
- :class:`TransformersTextEncoder` — HuggingFace ``AutoModel`` + tokenizer with
  mean / CLS / pooler pooling (lazy import).

:func:`build_text_encoder` selects the backend from :class:`TextEncoderConfig`.
:class:`TextEmbeddingCache` memoises embeddings on disk as ``.npz``.
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from zte.decode.config import TextEncoderConfig
from zte.device import resolve_device
from zte.logging_utils import get_logger

_LOG = get_logger('decode.text_encoder')


def _transformers_available() -> bool:
    """Returns whether the ``transformers`` package can be imported."""
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


class TextEncoder(nn.Module, ABC):
    """Abstract interface for mapping strings to fixed-size embeddings."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output embedding dimensionality."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> Tensor:
        """Embeds a batch of free-form strings.

        Args:
            texts: Input strings (sentences or phrases).

        Returns:
            Tensor of shape ``(N, D)``.
        """

    def embed_tokens(self, tokens: list[str]) -> Tensor:
        """Embeds word-level tokens (default: same path as :meth:`embed_texts`).

        Args:
            tokens: Surface word forms.

        Returns:
            Tensor of shape ``(N, D)``.
        """
        return self.embed_texts(tokens)

    def forward(self, texts: list[str]) -> Tensor:
        """Alias for :meth:`embed_texts` so the module is callable."""
        return self.embed_texts(texts)


class HashTextEncoder(TextEncoder):
    """Deterministic hash→Gaussian text encoder (no external dependencies).

    Each string is hashed to a seed; a fixed Gaussian vector of ``embed_dim`` is
    drawn and optionally L2-normalised. Identical strings always map to the same
    vector, which is enough for unit tests and CI without HuggingFace weights.
    """

    def __init__(self, config: TextEncoderConfig) -> None:
        """Initialises the hash encoder.

        Args:
            config: Text-encoder configuration (uses ``embed_dim`` / ``normalize``).
        """
        super().__init__()
        self.config = config
        self._dim = int(config.embed_dim)
        # Register a dummy buffer so ``.to(device)`` / ``.parameters()`` behave.
        self.register_buffer('_device_marker', torch.zeros(1), persistent=False)

    @property
    def dim(self) -> int:
        """Output embedding dimensionality."""
        return self._dim

    def embed_texts(self, texts: list[str]) -> Tensor:
        """Embeds strings via seeded Gaussian vectors.

        Args:
            texts: Input strings.

        Returns:
            Tensor ``(N, D)`` on the module's device.
        """
        device = self._device_marker.device
        if not texts:
            return torch.empty(0, self._dim, device=device)
        rows = [_hash_vector(t, self._dim) for t in texts]
        emb = torch.from_numpy(np.stack(rows, axis=0)).to(device=device, dtype=torch.float32)
        if self.config.normalize:
            emb = F.normalize(emb, dim=-1)
        return emb


class TransformersTextEncoder(TextEncoder):
    """HuggingFace ``AutoModel`` text encoder with configurable pooling."""

    def __init__(self, config: TextEncoderConfig) -> None:
        """Loads tokenizer + model (lazy ``transformers`` import).

        Args:
            config: Text-encoder configuration.

        Raises:
            ImportError: If ``transformers`` is not installed.
        """
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        device = resolve_device(config.device)
        self._device = device.device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        hidden = int(getattr(self.model.config, 'hidden_size', config.embed_dim))
        self._dim = hidden
        if config.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self.model.to(self._device)
        _LOG.info(
            'Loaded TransformersTextEncoder %s (dim=%d, freeze=%s)',
            config.model_name,
            self._dim,
            config.freeze,
        )

    @property
    def dim(self) -> int:
        """Output embedding dimensionality."""
        return self._dim

    def embed_texts(self, texts: list[str]) -> Tensor:
        """Tokenises and embeds a batch of strings.

        Args:
            texts: Input strings.

        Returns:
            Tensor ``(N, D)`` on the encoder device.
        """
        if not texts:
            return torch.empty(0, self._dim, device=self._device)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt',
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        ctx = torch.no_grad() if self.config.freeze else torch.enable_grad()
        with ctx:
            outputs = self.model(**encoded)
        emb = _pool_hidden(
            outputs,
            attention_mask=encoded['attention_mask'],
            pooling=self.config.pooling,
        )
        if self.config.normalize:
            emb = F.normalize(emb, dim=-1)
        return emb

    def train(self, mode: bool = True) -> TransformersTextEncoder:
        """Keeps the backbone in eval when frozen."""
        super().train(mode)
        if self.config.freeze:
            self.model.eval()
        return self


def build_text_encoder(config: TextEncoderConfig | None = None) -> TextEncoder:
    """Constructs a :class:`TextEncoder` from config / availability.

    Args:
        config: Text-encoder configuration (defaults to :class:`TextEncoderConfig`).

    Returns:
        A ready encoder (hash or transformers).

    Raises:
        ImportError: If ``backend='transformers'`` but the package is missing.
        ValueError: If ``backend`` is unknown.
    """
    config = config or TextEncoderConfig()
    backend = config.backend
    force_hash = config.model_name.strip().lower() == 'hash'
    if backend == 'hash' or force_hash:
        return HashTextEncoder(config)
    if backend == 'transformers':
        if not _transformers_available():
            raise ImportError(
                "TextEncoderConfig.backend='transformers' but the transformers "
                'package is not installed.'
            )
        return TransformersTextEncoder(config)
    if backend == 'auto':
        if _transformers_available() and not force_hash:
            try:
                return TransformersTextEncoder(config)
            except Exception as exc:  # noqa: BLE001 — fall back for CI / missing weights
                _LOG.warning('Transformers text encoder failed (%s); using hash backend.', exc)
        return HashTextEncoder(config)
    raise ValueError(f'Unknown text-encoder backend: {backend!r}')


class TextEmbeddingCache:
    """Disk cache of text embeddings keyed by model name + content hash.

    Attributes:
        cache_dir: Root directory for ``.npz`` shards.
        model_name: Encoder identity included in the cache key.
    """

    def __init__(self, cache_dir: str | Path, model_name: str) -> None:
        """Creates (or reuses) a cache directory.

        Args:
            cache_dir: Root directory for cached embeddings.
            model_name: Model id used as part of the cache key.
        """
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key_for(self, texts: list[str]) -> str:
        """Stable content hash for a list of texts.

        Args:
            texts: Strings to hash (order-sensitive).

        Returns:
            Hex digest string.
        """
        payload = json.dumps({'model': self.model_name, 'texts': texts}, ensure_ascii=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]

    def path_for(self, texts: list[str]) -> Path:
        """Returns the on-disk path for ``texts``.

        Args:
            texts: Strings whose embeddings would be cached.

        Returns:
            Path to the ``.npz`` shard.
        """
        return self.cache_dir / f'{self.model_name.replace("/", "__")}__{self.key_for(texts)}.npz'

    def load(self, texts: list[str]) -> np.ndarray | None:
        """Loads cached embeddings if present.

        Args:
            texts: Strings previously passed to :meth:`save`.

        Returns:
            ``(N, D)`` float32 array, or ``None`` on miss.
        """
        path = self.path_for(texts)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data['embeddings'], dtype=np.float32)

    def save(self, texts: list[str], embeddings: np.ndarray) -> Path:
        """Writes embeddings for ``texts`` to disk.

        Args:
            texts: Strings aligned with ``embeddings`` rows.
            embeddings: Array ``(N, D)``.

        Returns:
            Path written.
        """
        path = self.path_for(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, embeddings=np.asarray(embeddings, dtype=np.float32))
        return path

    def get_or_compute(self, texts: list[str], encoder: TextEncoder) -> np.ndarray:
        """Returns cached embeddings or computes and stores them.

        Args:
            texts: Strings to embed.
            encoder: Encoder used on cache miss.

        Returns:
            ``(N, D)`` float32 array.
        """
        cached = self.load(texts)
        if cached is not None and cached.shape[0] == len(texts):
            return cached
        with torch.no_grad():
            emb = encoder.embed_texts(texts).detach().cpu().numpy().astype(np.float32)
        self.save(texts, emb)
        return emb


def _hash_vector(text: str, dim: int) -> np.ndarray:
    """Maps one string to a deterministic unit-scale Gaussian vector.

    Args:
        text: Input string.
        dim: Embedding dimensionality.

    Returns:
        Float32 vector of length ``dim``.
    """
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    seed = int.from_bytes(digest[:8], byteorder='little', signed=False)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim, dtype=np.float32)


def _pool_hidden(outputs: Any, attention_mask: Tensor, pooling: str) -> Tensor:
    """Pools last-hidden states according to ``pooling``.

    Args:
        outputs: HuggingFace model outputs (needs ``last_hidden_state`` / ``pooler_output``).
        attention_mask: Tokenizer attention mask ``(N, T)``.
        pooling: ``'mean'``, ``'cls'`` or ``'pooler'``.

    Returns:
        Pooled tensor ``(N, D)``.
    """
    hidden = outputs.last_hidden_state  # (N, T, D)
    if pooling == 'cls':
        return hidden[:, 0]
    if pooling == 'pooler':
        pooler = getattr(outputs, 'pooler_output', None)
        if pooler is not None:
            return pooler
        return hidden[:, 0]
    # mean over non-padding tokens
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts
