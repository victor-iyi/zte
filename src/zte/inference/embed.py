"""Inference: turn a trained ZTE checkpoint into thought embeddings.

`ZTEEmbedder` rebuilds the encoder and its fitted normaliser from a checkpoint, then embeds either a built `ZuCoDataset`
with aligned metadata or brand-new in-memory EEG token arrays.
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
        """Wraps an already-built model (prefer `from_checkpoint`).

        Args:
            model (ZTEModel): A constructed and weight-loaded `ZTEModel`.
            config (ZTEConfig): The run configuration.
            device (DeviceSpec): The resolved device spec.
        """
        self.config = config
        self.device = device
        self.model = model.to(device.device).eval()

        # Populated by `from_checkpoint`, so new signals are normalised exactly as during training.
        self.normalizer: Any | None = None
        self.aligner: Any | None = None
        self.subject_vocab: dict[str, int] | None = None
        self.in_dim: int | None = None
        self.raw_shape: tuple[int, int] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        dataset: ZuCoDataset | None = None,
        device: DeviceSpec | None = None,
    ) -> ZTEEmbedder:
        """Rebuilds the encoder (and its normaliser) from a checkpoint.

        Input shapes and the fitted normaliser come from the checkpoint itself, so no dataset is needed to embed new
        signals; `dataset` is only a fallback for checkpoints that carry no shapes.

        Args:
            ckpt_path (str | Path): Path to a `best.pt`/`last.pt` checkpoint.
            dataset (ZuCoDataset | None): Built dataset, used only to infer frontend input shapes.
            device (DeviceSpec | None): Optional device spec (auto-resolved when `None`).

        Returns:
            ZTEEmbedder: A ready embedder.

        Raises:
            ValueError: If input shapes can be found neither in the checkpoint nor from a supplied dataset.
        """
        device = device or resolve_device('auto')
        payload = CheckpointManager.load(ckpt_path, map_location=str(device.device))
        config = ZTEConfig.from_dict(payload['config'])
        extra = payload.get('extra', {}) or {}

        in_dim = extra.get('in_dim')
        raw_shape = extra.get('raw_shape')
        raw_shape = tuple(raw_shape) if raw_shape is not None else None
        if in_dim is None and raw_shape is None and dataset is not None:
            in_dim, raw_shape = _input_shapes(dataset)
        if in_dim is None and raw_shape is None:
            raise ValueError('Checkpoint lacks input shapes; pass the dataset it was trained on.')

        model = build_model(
            config.model,
            in_dim=in_dim,
            raw_shape=raw_shape,
            n_channels=extra.get('n_channels'),
            bp_features_per_channel=extra.get('bp_features_per_channel'),
            montage_csv=extra.get('montage_csv'),
            signature_dim=int(extra.get('signature_dim') or 0),
        )
        model.load_state_dict(payload['model'])
        embedder = cls(model, config, device)
        embedder.in_dim = in_dim
        embedder.raw_shape = raw_shape

        if extra.get('normalizer'):
            from zte.data.features.transforms import FeatureNormalizer

            embedder.normalizer = FeatureNormalizer.from_state(extra['normalizer'])
        if extra.get('aligner'):
            from zte.data.features.alignment import RawSubjectAligner

            embedder.aligner = RawSubjectAligner.from_state(extra['aligner'])
        embedder.subject_vocab = extra.get('subject_vocab')
        _LOG.info('Loaded ZTE checkpoint %s (epoch %s)', ckpt_path, payload.get('epoch'))
        return embedder

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
            level (EmbeddingLevel): `word` for one embedding per present word, or `sentence` for one pooled per
                sentence.
            indices (np.ndarray | None): Optional word-row indices to restrict to (e.g. a split).
            batch_size (int): Sentences per forward pass.
            present_only (bool): For word level, keep only present (non-omitted) words.

        Returns:
            tuple[np.ndarray, pd.DataFrame]: `(n_samples, embed_dim)` embeddings and a metadata frame of the same
                length.
        """
        vocab = build_subject_vocab(dataset)
        torch_ds = ZuCoTorchDataset(dataset, indices=indices, subject_vocab=vocab)
        loader = make_dataloader(torch_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        embeddings: list[np.ndarray] = []
        meta_rows: list[dict[str, Any]] = []
        seq_ptr = 0

        for batch in progress(loader, description=f'embedding ({level})'):
            dev_batch = {k: (v.to(self.device.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            objective = self.config.objective.name
            if level == 'sentence':
                emb = self.model.embed_sentence(dev_batch, objective=objective).cpu().numpy()
                for b in range(emb.shape[0]):
                    rows = torch_ds.sequences[seq_ptr]
                    embeddings.append(emb[b])
                    meta_rows.append(self._sentence_meta(dataset, rows))
                    seq_ptr += 1
            else:
                # Word-level routing mirrors each objective's trained representation.
                contextual = objective in {'cpc', 'masked'}
                causal = objective == 'cpc'
                token_emb = self.model.forward(dev_batch, contextual=contextual, causal=causal).cpu().numpy()
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
            np.asarray(embeddings, dtype=np.float32) if embeddings else np.empty((0, self.model.embed_dim), np.float32)
        )
        return emb_array, pd.DataFrame(meta_rows)

    def calibrate_subject(
        self,
        baseline_band_power: np.ndarray | None = None,
        subject_code: str = '',
        baseline_raw: np.ndarray | None = None,
    ) -> None:
        """Calibrates a new subject into the shared space from an unlabelled baseline.

        The zero-shot new-brain path: a short recording of them reading anything yields their own whitening map and
        signature, so their words land on the cohort's scale and the subject adapter configures itself for a person
        the model never trained on. No labels, no retraining. Pass their code to `embed_signals` afterwards.

        Args:
            baseline_band_power (np.ndarray | None): `(n, n_features)` un-normalised baseline tokens.
            subject_code (str): The code to register the calibrated statistics under.
            baseline_raw (np.ndarray | None): `(n, n_channels, time_steps)` baseline raw windows, for the raw frontend.
        """
        if self.normalizer is not None and baseline_band_power is not None:
            self.normalizer.calibrate_subject(np.asarray(baseline_band_power, dtype=np.float32), subject_code)
        if self.aligner is not None and baseline_raw is not None:
            self.aligner.calibrate_subject(np.asarray(baseline_raw, dtype=np.float32), subject_code)

    @torch.no_grad()
    def embed_signals(
        self,
        band_power: np.ndarray | None = None,
        raw: np.ndarray | None = None,
        subjects: np.ndarray | None = None,
        subject_codes: np.ndarray | None = None,
        apply_normalizer: bool = True,
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embeds brand-new EEG token signals held in memory, with no `ZuCoDataset` needed.

        Each input row is one word's neural response and yields one embedding; whichever of `band_power`/`raw` matches
        the model's frontend is used. Tokens stream with no surrounding sentence context, so this always takes the
        non-contextual per-token path -- an approximation for `cpc`/`masked` checkpoints, whose trained word
        representation is contextual. Use `embed` on a dataset for those.

        Args:
            band_power (np.ndarray | None): Un-normalised `(n_tokens, n_features)` tokens; the last dim must equal the
                model's input size.
            raw (np.ndarray | None): `(n_tokens, n_channels, time_steps)` raw EEG windows for a raw model.
            subjects (np.ndarray | None): `(n_tokens,)` integer subject ids for a subject-conditioned model.
            subject_codes (np.ndarray | None): `(n_tokens,)` subject codes selecting per-subject normalisation
                statistics; unknown codes use the population fallback.
            apply_normalizer (bool): Apply the checkpoint's band-power normaliser (ignored for raw input).
            batch_size (int): Tokens per forward pass.
            show_progress (bool): Show a progress bar.

        Returns:
            np.ndarray: `(n_tokens, embed_dim)` float32 embeddings, one per input token.

        Raises:
            ValueError: If the input required by the model's frontend is missing or mis-shaped.
        """
        if self.model.uses_raw:
            if raw is None:
                raise ValueError('This checkpoint uses a raw frontend; pass raw=(n_tokens, n_channels, time_steps).')
            signals = np.asarray(raw, dtype=np.float32)
            if signals.ndim != 3:
                raise ValueError(f'raw must be (n_tokens, n_channels, time_steps); got shape {signals.shape}.')

            # Same per-subject whitening the model trained under; an uncalibrated code gets the cohort reference.
            if apply_normalizer and self.aligner is not None and subject_codes is not None:
                signals = self.aligner.transform(signals.copy(), np.asarray(subject_codes))
        else:
            if band_power is None:
                raise ValueError('This checkpoint uses a band-power frontend; pass band_power=(n_tokens, n_features).')
            signals = np.asarray(band_power, dtype=np.float32)
            if signals.ndim != 2:
                raise ValueError(f'band_power must be (n_tokens, n_features); got shape {signals.shape}.')
            if self.in_dim is not None and signals.shape[1] != self.in_dim:
                include_et = self.config.dataset.include_eye_tracking
                raise ValueError(
                    f'band_power has width {signals.shape[1]} but this checkpoint expects '
                    f'{self.in_dim} (it was trained with include_eye_tracking='
                    f'{include_et}). Append the eye-tracking scalars to match, or embed '
                    'with an EEG-only checkpoint (include_eye_tracking=False) when the new '
                    'signals have no eye tracking -- the imagined-thought path.'
                )
            if apply_normalizer and self.normalizer is not None:
                signals = self.normalizer.transform(signals, subjects=subject_codes)

        n = signals.shape[0]
        subj = np.zeros(n, dtype=np.int64) if subjects is None else np.asarray(subjects, dtype=np.int64)

        # The adapter reads the signature, so build one per token from whichever subject code it carries.
        sig = None
        if self.model.subject_adapter is not None and self.aligner is not None:
            codes = np.asarray(subject_codes) if subject_codes is not None else np.array([''] * n)
            sig = np.stack([self.aligner.signature_for(str(c)) for c in codes]).astype(np.float32)
        dev = self.device.device
        out: list[np.ndarray] = []
        for start in progress(range(0, n, batch_size), description='embedding signals', disable=not show_progress):
            end = min(start + batch_size, n)
            count = end - start
            chunk = torch.from_numpy(np.ascontiguousarray(signals[start:end])).to(dev)

            # Each token is a length-1 sequence, so the non-contextual path yields one embedding per signal.
            batch: dict[str, Any] = {
                'features': None,
                'raw': None,
                'pad_mask': torch.ones(count, 1, dtype=torch.bool, device=dev),
                'presence': torch.ones(count, 1, dtype=torch.bool, device=dev),
                'subject': torch.from_numpy(subj[start:end]).to(dev),
                'subject_signature': None if sig is None else torch.from_numpy(sig[start:end]).to(dev),
                'lengths': torch.ones(count, dtype=torch.long, device=dev),
            }
            batch['raw' if self.model.uses_raw else 'features'] = chunk.unsqueeze(1)
            token_emb = self.model(batch, contextual=False)  # (count, 1, embed_dim)
            out.append(token_emb[:, 0, :].cpu().numpy())
        if not out:
            return np.empty((0, self.model.embed_dim), dtype=np.float32)
        return np.concatenate(out, axis=0)

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
            embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
            meta (pd.DataFrame): Aligned metadata of length `n_samples`.
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
    def nearest_neighbors(embeddings: np.ndarray, query_idx: int, k: int = 5) -> list[tuple[int, float]]:
        """Returns the `k` nearest neighbours (cosine) of one embedding.

        Args:
            embeddings (np.ndarray): Array `(n_samples, embed_dim)`.
            query_idx (int): Row index of the query embedding.
            k (int): Number of neighbours to return (excluding the query itself).

        Returns:
            list[tuple[int, float]]: `(index, cosine_similarity)` pairs, most similar first.
        """
        normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        sims = normed @ normed[query_idx]
        order = np.argsort(-sims)
        order = order[order != query_idx][:k]
        return [(int(i), float(sims[i])) for i in order]


def _input_shapes(dataset: ZuCoDataset) -> tuple[int | None, tuple[int, int] | None]:
    """Infers `(in_dim, raw_shape)` for model construction from a dataset.

    Returns:
        tuple[int | None, tuple[int, int] | None]: `(in_dim, raw_shape)`, with unused entries `None`.
    """
    in_dim = None if dataset.features is None else int(dataset.features.shape[1])
    raw_shape = None if dataset.raw_eeg is None else (int(dataset.raw_eeg.shape[1]), int(dataset.raw_eeg.shape[2]))
    return in_dim, raw_shape
