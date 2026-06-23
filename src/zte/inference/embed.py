"""Inference: turn a trained ZTE checkpoint into thought embeddings.

:class:`ZTEEmbedder` rebuilds the encoder from a checkpoint's embedded config, restores its weights, and produces word- or sentence-level embeddings for a
:class:`~zte.data.dataset.ZuCoDataset`. Outputs can be exported to `.npz` (with aligned metadata) and uploaded to Google Drive, and a nearest-neighbour helper
supports qualitative probing of the learned space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from zte.config import ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.torch_dataset import (
    ZuCoTorchDataset,
    build_subject_vocab,
    make_dataloader,
)
from zte.device import DeviceSpec, resolve_device
from zte.logging_utils import get_logger, progress
from zte.models.embedding import ZTEModel, build_model
from zte.training.checkpoint import CheckpointManager

_LOG = get_logger('inference.embed')

type EmbeddingLevel = Literal['word', 'sentence']


class ZTEEmbedder:
    """Produces thought embeddings from a trained ZTE model.

    Attributes:
        config (ZTEConfig): The full run config recovered from the checkpoint.
        model (ZTEModel): The restored, eval-mode encoder.
        device (DeviceSpec): The device the model runs on.
    """

    def __init__(self, model: ZTEModel, config: ZTEConfig, device: DeviceSpec) -> None:
        """Wraps an already-built model (prefer :meth:`from_checkpoint`).

        Args:
            model (ZTEModel): A constructed and weight-loaded :class:`ZTEModel`.
            config (ZTEConfig): The run configuration.
            device (DeviceSpec): The resolved device spec.
        """
        self.config = config
        self.device = device
        self.model = model.to(device.device).eval()

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        dataset: ZuCoDataset,
        device: DeviceSpec | None = None,
    ) -> ZTEEmbedder:
        """Rebuilds the encoder from a checkpoint, sized to `dataset`.

        Args:
            ckpt_path (str | Path): Path to a `best.pt`/`last.pt` checkpoint.
            dataset (ZuCoDataset): The (built) dataset to be embedded; used to infer the
                frontend's input shapes.
            device (DeviceSpec | None): Optional device spec (auto-resolved when `None`).

        Returns:
            ZTEEmbedder: A ready :class:`ZTEEmbedder`.

        """
        device = device or resolve_device('auto')
        payload = CheckpointManager.load(ckpt_path, map_location=str(device.device))
        config = ZTEConfig.from_dict(payload['config'])
        in_dim, raw_shape = _input_shapes(dataset)
        model = build_model(config.model, in_dim=in_dim, raw_shape=raw_shape)
        model.load_state_dict(payload['model'])
        _LOG.info('Loaded ZTE checkpoint %s (epoch %s)', ckpt_path, payload.get('epoch'))
        return cls(model, config, device)

    @torch.no_grad()
    def embed(
        self,
        dataset: ZuCoDataset,
        level: EmbeddingLevel = 'word',
        indices: np.ndarray | None = None,
        batch_size: int = 64,
        present_only: bool = True,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Embeds a dataset at word or sentence level.

        Args:
            dataset (ZuCoDataset): A built dataset.
            level (EmbeddingLevel): `'word'` for one embedding per (present) word, or `'sentence'` for one pooled embedding per sentence.
            indices (np.ndarray | None): Optional word-row indices to restrict to (e.g. a split).
            batch_size (int): Sentences per forward pass.
            present_only (bool): For word level, keep only present (non-omitted) words.

        Returns:
            tuple[np.ndarray, pd.DataFrame]: `(embeddings (M, E), metadata DataFrame of length M)`.

        """
        vocab = build_subject_vocab(dataset)
        torch_ds = ZuCoTorchDataset(dataset, indices=indices, subject_vocab=vocab)
        loader = make_dataloader(torch_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        embeddings: list[np.ndarray] = []
        meta_rows: list[dict[str, Any]] = []
        seq_ptr = 0

        for batch in progress(loader, description=f'embedding ({level})'):
            dev_batch = {
                k: (v.to(self.device.device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            if level == 'sentence':
                emb = self.model.embed_sentence(dev_batch).cpu().numpy()
                for b in range(emb.shape[0]):
                    rows = torch_ds.sequences[seq_ptr]
                    embeddings.append(emb[b])
                    meta_rows.append(self._sentence_meta(dataset, rows))
                    seq_ptr += 1
            else:
                token_emb = self.model(dev_batch, contextual=False).cpu().numpy()
                lengths = batch['lengths'].tolist()
                presence = batch['presence'].cpu().numpy()
                for b, length in enumerate(lengths):
                    rows = torch_ds.sequences[seq_ptr]
                    seq_ptr += 1
                    for j in range(length):
                        if present_only and not presence[b, j]:
                            continue
                        embeddings.append(token_emb[b, j])
                        meta_rows.append(self._word_meta(dataset, rows[j]))
        emb_array = (
            np.asarray(embeddings, dtype=np.float32)
            if embeddings
            else np.empty((0, self.model.embed_dim), np.float32)
        )
        return emb_array, pd.DataFrame(meta_rows)

    @staticmethod
    def _word_meta(dataset: ZuCoDataset, row: int) -> dict[str, Any]:
        """Builds the metadata record for one word row."""
        w = dataset.words.iloc[int(row)]
        return {
            'subject': w['subject'],
            'task': w['task'],
            'sentence_idx': int(w['sentence_idx']),
            'word_idx': int(w['word_idx']),
            'word': w['word'],
            'word_len': int(w.get('word_len', len(str(w['word'])))),
            'log_freq': float(w.get('log_freq', float('nan'))),
            'is_omitted': int(w.get('is_omitted', 0)),
        }

    @staticmethod
    def _sentence_meta(dataset: ZuCoDataset, rows: np.ndarray) -> dict[str, Any]:
        """Builds the metadata record for one sentence (from its first word row)."""
        first = dataset.words.iloc[int(rows[0])]
        return {
            'subject': first['subject'],
            'task': first['task'],
            'sentence_idx': int(first['sentence_idx']),
            'n_words': int(len(rows)),
        }

    def export(self, embeddings: np.ndarray, meta: pd.DataFrame, path: str | Path) -> Path:
        """Saves embeddings + metadata as a single `.npz` bundle.

        Args:
            embeddings (np.ndarray): Array `(M, E)`.
            meta (pd.DataFrame): Aligned metadata of length `M`.
            path (str | Path): Output path (`.npz` appended if absent).

        Returns:
            Path: The written path.
        """
        out = Path(path)
        if out.suffix != '.npz':
            out = out.with_suffix('.npz')
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            embeddings=embeddings,
            meta_columns=np.array(list(meta.columns)),
            **{f'meta__{c}': meta[c].to_numpy() for c in meta.columns},
        )
        _LOG.info('Exported %d embeddings to %s', len(embeddings), out)
        return out

    @staticmethod
    def nearest_neighbors(
        embeddings: np.ndarray, query_idx: int, k: int = 5
    ) -> list[tuple[int, float]]:
        """Returns the `k` nearest neighbours (cosine) of one embedding.

        Args:
            embeddings (np.ndarray): Array `(M, E)`.
            query_idx (int): Row index of the query embedding.
            k (int): Number of neighbours to return (excluding the query itself).

        Returns:
            list[tuple[int, float]]: A list of `(index, cosine_similarity)` pairs, most similar first.
        """
        normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        sims = normed @ normed[query_idx]
        order = np.argsort(-sims)
        order = order[order != query_idx][:k]
        return [(int(i), float(sims[i])) for i in order]


def _input_shapes(dataset: ZuCoDataset) -> tuple[int | None, tuple[int, int] | None]:
    """Infers `(in_dim, raw_shape)` for model construction from a dataset.

    Args:
        dataset (ZuCoDataset): A built dataset.

    Returns:
        tuple[int | None, tuple[int, int] | None]: `(in_dim, raw_shape)` where unused entries are `None`.

    """
    in_dim = None if dataset.features is None else int(dataset.features.shape[1])
    raw_shape = (
        None
        if dataset.raw_eeg is None
        else (int(dataset.raw_eeg.shape[1]), int(dataset.raw_eeg.shape[2]))
    )
    return in_dim, raw_shape
