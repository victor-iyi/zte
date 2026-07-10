"""Build EEG↔text paired datasets from ZuCo + ZTE embeddings.

Sentence-level pairs join :meth:`ZTEEmbedder.embed` (``level='sentence'``) with
text embeddings of ``dataset.sentences['text']``. Word-level pairs use the
``word`` column from embedder metadata. Precomputed arrays feed
:class:`PairedEmbeddingDataset` / :func:`make_paired_loader` for alignment
training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from zte.data.dataset import ZuCoDataset
from zte.decode.text_encoder import TextEmbeddingCache, TextEncoder
from zte.logging_utils import get_logger

_LOG = get_logger('decode.pairing')


@dataclass
class PairedBatch:
    """One collated batch of EEG / text pairs.

    Attributes:
        eeg: EEG embeddings ``(B, D)``.
        text_emb: Text embeddings ``(B, D)``.
        texts: Surface strings aligned with the rows.
        meta: Metadata rows (subject, task, …) of length ``B``.
    """

    eeg: torch.Tensor
    text_emb: torch.Tensor
    texts: list[str]
    meta: pd.DataFrame


def build_sentence_pairs(
    dataset: ZuCoDataset,
    embedder: Any,
    text_encoder: TextEncoder,
    indices: np.ndarray | None = None,
    cache: TextEmbeddingCache | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Embeds sentences with ZTE and a text encoder, aligned by sentence identity.

    Args:
        dataset: Built :class:`~zte.data.dataset.ZuCoDataset`.
        embedder: A :class:`~zte.inference.embed.ZTEEmbedder` (or duck-typed equivalent
            exposing ``embed(dataset, level=..., indices=..., batch_size=...)``).
        text_encoder: Language-side encoder.
        indices: Optional word-row indices restricting which sentences are embedded
            (same semantics as :meth:`ZTEEmbedder.embed`).
        cache: Optional disk cache for text embeddings.
        batch_size: Sentences per ZTE forward pass.

    Returns:
        ``(eeg_emb, text_emb, texts, meta)`` where embeddings are ``(N, D)`` float32,
        ``texts`` has length ``N``, and ``meta`` is a DataFrame of length ``N``.
    """
    eeg_emb, meta = embedder.embed(
        dataset, level='sentence', indices=indices, batch_size=batch_size
    )
    if len(meta) == 0:
        dim = int(getattr(text_encoder, 'dim', eeg_emb.shape[1] if eeg_emb.size else 768))
        return (
            np.empty((0, eeg_emb.shape[1] if eeg_emb.ndim == 2 else dim), np.float32),
            np.empty((0, dim), np.float32),
            [],
            meta,
        )

    texts = _lookup_sentence_texts(dataset, meta)
    text_emb = _embed_texts(texts, text_encoder, cache)
    _LOG.info(
        'Built %d sentence pairs (eeg_dim=%d, text_dim=%d)',
        len(texts),
        eeg_emb.shape[1],
        text_emb.shape[1],
    )
    return eeg_emb, text_emb, texts, meta


def build_word_pairs(
    dataset: ZuCoDataset,
    embedder: Any,
    text_encoder: TextEncoder,
    indices: np.ndarray | None = None,
    present_only: bool = True,
    cache: TextEmbeddingCache | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Embeds present words with ZTE and a text encoder on surface forms.

    Args:
        dataset: Built dataset.
        embedder: ZTE embedder.
        text_encoder: Language-side encoder.
        indices: Optional word-row indices.
        present_only: Drop omitted words (recommended).
        cache: Optional disk cache for text embeddings.
        batch_size: Sentences per ZTE forward pass.

    Returns:
        ``(eeg_emb, text_emb, texts, meta)`` aligned at word level.
    """
    eeg_emb, meta = embedder.embed(
        dataset,
        level='word',
        indices=indices,
        batch_size=batch_size,
        present_only=present_only,
    )
    if len(meta) == 0:
        dim = int(getattr(text_encoder, 'dim', 768))
        return (
            np.empty((0, eeg_emb.shape[1] if eeg_emb.ndim == 2 else dim), np.float32),
            np.empty((0, dim), np.float32),
            [],
            meta,
        )
    if 'word' in meta.columns:
        texts = [str(w) for w in meta['word'].tolist()]
    else:
        texts = [''] * len(meta)
    text_emb = _embed_texts(texts, text_encoder, cache)
    _LOG.info(
        'Built %d word pairs (eeg_dim=%d, text_dim=%d)',
        len(texts),
        eeg_emb.shape[1],
        text_emb.shape[1],
    )
    return eeg_emb, text_emb, texts, meta


class PairedEmbeddingDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Yields ``(eeg_i, text_emb_i, text_i)`` from precomputed arrays.

    Attributes:
        eeg: Float32 array ``(N, D_eeg)``.
        text_emb: Float32 array ``(N, D_text)``.
        texts: Surface strings of length ``N``.
    """

    def __init__(
        self,
        eeg: np.ndarray,
        text_emb: np.ndarray,
        texts: list[str],
    ) -> None:
        """Validates alignment and stores arrays.

        Args:
            eeg: EEG embeddings ``(N, D_eeg)``.
            text_emb: Text embeddings ``(N, D_text)``.
            texts: Strings of length ``N``.

        Raises:
            ValueError: If lengths disagree.
        """
        self.eeg = np.asarray(eeg, dtype=np.float32)
        self.text_emb = np.asarray(text_emb, dtype=np.float32)
        self.texts = list(texts)
        n = len(self.texts)
        if len(self.eeg) != n or len(self.text_emb) != n:
            raise ValueError(
                f'eeg ({len(self.eeg)}), text_emb ({len(self.text_emb)}) and '
                f'texts ({n}) must align.'
            )

    def __len__(self) -> int:
        """Number of pairs."""
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Returns one pair.

        Args:
            index: Row index.

        Returns:
            ``(eeg, text_emb, text)`` tensors / string.
        """
        return (
            torch.from_numpy(self.eeg[index]),
            torch.from_numpy(self.text_emb[index]),
            self.texts[index],
        )


def make_paired_loader(
    eeg: np.ndarray,
    text_emb: np.ndarray,
    texts: list[str],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader[Any]:
    """Builds a :class:`~torch.utils.data.DataLoader` over paired embeddings.

    Args:
        eeg: EEG embeddings ``(N, D_eeg)``.
        text_emb: Text embeddings ``(N, D_text)``.
        texts: Surface strings.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: DataLoader workers.

    Returns:
        A DataLoader yielding ``(eeg_batch, text_batch, text_list)``.
    """
    ds = PairedEmbeddingDataset(eeg, text_emb, texts)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_pairs,
        drop_last=False,
    )


def _collate_pairs(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collates paired samples into stacked tensors + a text list."""
    eeg = torch.stack([b[0] for b in batch], dim=0)
    text = torch.stack([b[1] for b in batch], dim=0)
    texts = [b[2] for b in batch]
    return eeg, text, texts


def _lookup_sentence_texts(dataset: ZuCoDataset, meta: pd.DataFrame) -> list[str]:
    """Resolves sentence text strings for embedder metadata rows.

    Args:
        dataset: Source dataset with ``sentences`` table.
        meta: Sentence-level metadata from the embedder (subject / task / sentence_idx).

    Returns:
        List of text strings aligned with ``meta`` rows.
    """
    sent = dataset.sentences
    if sent is None or len(sent) == 0 or 'text' not in sent.columns:
        return [''] * len(meta)
    key_cols = ['subject', 'task', 'sentence_idx']
    lookup = sent.set_index(key_cols)['text']
    texts: list[str] = []
    for _, row in meta.iterrows():
        key = (row['subject'], row['task'], int(row['sentence_idx']))
        try:
            texts.append(str(lookup.loc[key]))
        except KeyError:
            texts.append('')
    return texts


def _embed_texts(
    texts: list[str],
    encoder: TextEncoder,
    cache: TextEmbeddingCache | None,
) -> np.ndarray:
    """Embeds texts, optionally via disk cache.

    Args:
        texts: Strings to embed.
        encoder: Text encoder.
        cache: Optional cache.

    Returns:
        ``(N, D)`` float32 array.
    """
    if cache is not None:
        return cache.get_or_compute(texts, encoder)
    with torch.no_grad():
        return encoder.embed_texts(texts).detach().cpu().numpy().astype(np.float32)
