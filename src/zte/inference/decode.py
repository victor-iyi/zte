"""Inference: drive a frozen LM from a trained prefix bridge -- free-running decode and decoder-rescoring retrieval."""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from zte.config import DecoderConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.features.alignment import RawSubjectAligner
from zte.data.features.transforms import FeatureNormalizer
from zte.data.targets.tokens import build_target_tokens
from zte.data.torch_dataset import ZuCoTorchDataset, build_subject_vocab, make_dataloader
from zte.device import DeviceSpec, resolve_device
from zte.logging_utils import get_logger, progress
from zte.models.decoder import FrozenLM, GapCorrector, PrefixBridge, build_bridge, build_lm
from zte.models.embedding import ZTEModel, build_model
from zte.training.checkpoint import CheckpointManager

_LOG = get_logger('inference.decode')


class ZTEDecoder:
    """Turns a decoder checkpoint into text, and into a ranking over a sentence gallery.

    Two readouts live here and they are not the same claim. `generate` is free-running: it is handed a conditioning
    vector, a beginning-of-sequence token and nothing else -- no reference, no length, no candidate set -- and every
    brain-independent control runs through the identical call, so a difference between them is the only evidence the
    method admits. `rescore` ranks a fixed gallery by sequence likelihood; that is retrieval, it is far better powered,
    and it is named that way everywhere it appears.

    Attributes:
        config (ZTEConfig): The source run's configuration.
        decoder_config (DecoderConfig): The decoder configuration the checkpoint was trained under.
        model (ZTEModel): The eval-mode encoder.
        bridge (PrefixBridge): The trained soft-prompt bridge, which conditions on the pooled sentence vector.
        gap (GapCorrector): The train-fitted EEG-to-text correction.
        lm (FrozenLM): The frozen causal LM.
        clip_head (nn.Linear | None): Projection into the frozen text space the bridge reads.
        device (DeviceSpec): The device everything runs on.
        normalizer (FeatureNormalizer | None): The source run's fitted band-power statistics, applied by
            `prepare_dataset` to every dataset this decoder reads.
        aligner (RawSubjectAligner | None): The source run's fitted raw whitening, applied by `prepare_dataset`.
        subject_vocab (dict[str, int] | None): The subject-code map the encoder was trained under.
    """

    def __init__(
        self,
        model: ZTEModel,
        config: ZTEConfig,
        decoder_config: DecoderConfig,
        bridge: PrefixBridge,
        lm: FrozenLM,
        gap: GapCorrector,
        clip_head: nn.Linear | None = None,
        device: DeviceSpec | None = None,
    ) -> None:
        """Wraps already-built parts (prefer `from_checkpoint`).

        Args:
            model (ZTEModel): A weight-loaded encoder.
            config (ZTEConfig): The source run's configuration.
            decoder_config (DecoderConfig): The decoder configuration.
            bridge (PrefixBridge): The trained bridge.
            lm (FrozenLM): The frozen LM.
            gap (GapCorrector): The fitted gap correction.
            clip_head (nn.Linear | None, optional): The projection into the text space. Defaults to None.
            device (DeviceSpec | None, optional): Device spec. Defaults to None, which resolves automatically.
        """
        self.config = config
        self.decoder_config = decoder_config
        self.device = device or resolve_device('auto')
        target = self.device.device
        self.model = model.to(target).eval()
        self.bridge = bridge.to(target).eval()
        self.lm = lm.to(target).eval()
        self.gap = gap.to(target).eval()
        self.clip_head = None if clip_head is None else clip_head.to(target).eval()

        self.normalizer: FeatureNormalizer | None = None
        self.aligner: RawSubjectAligner | None = None
        self.subject_vocab: dict[str, int] | None = None
        self._prepared: weakref.ref[ZuCoDataset] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        dataset: ZuCoDataset | None = None,
        device: DeviceSpec | None = None,
    ) -> ZTEDecoder:
        """Rebuilds the encoder, bridge, gap correction and frozen LM from a decoder run's checkpoint.

        Args:
            ckpt_path (str | Path): A `best.pt`/`last.pt` written by a `decoder` or `joint` run.
            dataset (ZuCoDataset | None, optional): Built dataset, used only when the checkpoint carries no input
                shapes. Defaults to None.
            device (DeviceSpec | None, optional): Device spec. Defaults to None.

        Returns:
            ZTEDecoder: A ready decoder.

        Raises:
            ValueError: If the checkpoint holds no `decoder_state` (an encoder-only run), or was trained with
                word-level conditioning, whose prefix this pooled-only path cannot reproduce.
        """
        device = device or resolve_device('auto')
        payload = CheckpointManager.load(ckpt_path, map_location=str(device.device))
        config = ZTEConfig.from_dict(payload['config'])
        extra: dict[str, Any] = payload.get('extra') or {}
        state: dict[str, torch.Tensor] = extra.get('decoder_state') or {}
        if not state:
            raise ValueError(
                f'{ckpt_path} carries no decoder_state; it is an encoder run. Train with train.mode=decoder first.'
            )

        decoder_config = _decoder_config(extra, config)
        model = _rebuild_encoder(config, extra, dataset)
        model.load_state_dict(payload['model'])

        if decoder_config.conditioning != 'pooled':
            raise ValueError(
                f'{ckpt_path} was trained with conditioning={decoder_config.conditioning!r}, whose prefix is half '
                'word slots. Decoding it through the pooled-only path here would silently drop them and produce a '
                'number for a prompt the run never used, so that arm is reported from its training metrics only.'
            )
        lm = build_lm(decoder_config)
        z_dim = int(state['bridge.to_bottleneck.weight'].shape[1])
        bridge, _ = build_bridge(
            decoder_config, z_dim, cast('int', model.hidden_dim), lm.hidden_dim
        )
        bridge.load_state_dict(_sub_state(state, 'bridge.'), strict=True)

        gap_state = extra.get('gap_correction')
        gap = (
            GapCorrector.from_state(gap_state)
            if gap_state
            else GapCorrector(z_dim, mode=decoder_config.gap_correction)
        )
        decoder = cls(
            model=model,
            config=config,
            decoder_config=decoder_config,
            bridge=bridge,
            lm=lm,
            gap=gap,
            clip_head=_rebuild_clip_head(state),
            device=device,
        )
        if extra.get('normalizer'):
            decoder.normalizer = FeatureNormalizer.from_state(extra['normalizer'])
        if extra.get('aligner'):
            decoder.aligner = RawSubjectAligner.from_state(extra['aligner'])
        decoder.subject_vocab = extra.get('subject_vocab')
        _LOG.info(
            'Loaded decoder checkpoint %s (epoch %s, z_dim %d, %d prefix slots, LM %s).',
            ckpt_path,
            payload.get('epoch'),
            z_dim,
            bridge.slots,
            decoder_config.lm_source,
        )
        return decoder

    # ---- Conditioning ---- #

    @property
    def z_dim(self) -> int:
        """Width of the conditioning vector the bridge reads."""
        return int(self.bridge.norm_in.normalized_shape[0])

    def prepare_dataset(self, dataset: ZuCoDataset) -> None:
        """Re-scales a dataset onto the fitted statistics the encoder was trained under.

        A freshly built dataset normalises itself, so its rows arrive on statistics this encoder has never seen. That
        does not fail -- the frozen encoder simply produces a worse conditioning vector -- so the checkpoint's own
        normaliser and raw aligner are installed here, exactly as a decoder training run installs them over its source
        encoder. Idempotent: a dataset already prepared by this decoder is left alone.

        Args:
            dataset (ZuCoDataset): The built dataset about to be embedded, transformed in place.
        """
        if self._prepared is not None and self._prepared() is dataset:
            return
        if self.normalizer is not None:
            dataset.set_normalizer_state(self.normalizer.state)
        if self.aligner is not None:
            dataset.set_aligner_state(self.aligner.state)
        self._prepared = weakref.ref(dataset)

    @torch.no_grad()
    def conditioning(
        self,
        dataset: ZuCoDataset,
        indices: np.ndarray | None = None,
        batch_size: int = 16,
        *,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Embeds every reading in a split into the bridge-ready conditioning vector.

        The returned vectors are already gap-corrected, so `prefix_from_z` applies the bridge and nothing else. A true
        text embedding fed to `prefix_from_z` is therefore the text oracle: the identical head, without the correction
        that exists only to move EEG vectors onto the text cloud.

        Args:
            dataset (ZuCoDataset): A built dataset, re-scaled in place by `prepare_dataset` before it is embedded.
            indices (np.ndarray | None, optional): Word-row indices selecting a split. Defaults to None.
            batch_size (int, optional): Sentences per forward pass. Defaults to 16.
            transform (Callable[[dict[str, Any]], dict[str, Any]] | None, optional): Applied to each device-resident
                batch before the encoder sees it, which is how the signal-destroying controls (phase-scrambled,
                noise-matched) run through this identical path rather than a parallel one. Defaults to None.

        Returns:
            tuple[np.ndarray, pd.DataFrame]: `(n_readings, z_dim)` vectors and a metadata frame of the same length,
                carrying `subject`, `task`, `sentence_idx`, `n_words`, `reading_id`, `text_id`, `stimulus_key` and the
                reference `text`.
        """
        self.prepare_dataset(dataset)
        vocab = self.subject_vocab or build_subject_vocab(dataset)
        torch_ds = ZuCoTorchDataset(dataset, indices=indices, subject_vocab=vocab)
        loader = make_dataloader(torch_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        vectors: list[np.ndarray] = []
        for batch in progress(loader, description='conditioning'):
            moved = {
                k: (v.to(self.device.device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            ready = moved if transform is None else transform(moved)
            vectors.append(self._sentence_z(ready).cpu().numpy())
        z = np.concatenate(vectors) if vectors else np.empty((0, self.z_dim), np.float32)
        return z.astype(np.float32), _sentence_meta(dataset, torch_ds)

    def prefix_from_z(self, z: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Maps bridge-ready conditioning vectors to soft prompts.

        Args:
            z (np.ndarray | torch.Tensor): `(n, z_dim)` vectors, already gap-corrected.

        Returns:
            torch.Tensor: `(n, slots, lm_dim)` prefix embeddings on the decoder's device.
        """
        return self.bridge(_as_tensor(z, self.device.device))

    def null_prefix(self, n: int) -> torch.Tensor:
        """Returns the learned unconditional prefix broadcast over `n` rows, which is the `null_prefix` control."""
        return self.bridge.null(n)

    def mean_prefix(self, z: np.ndarray | torch.Tensor, n: int) -> torch.Tensor:
        """Returns the prefix of the mean conditioning vector, repeated `n` times.

        This is the decisive control: it absorbs whatever text prior the bridge learned that does not depend on the
        reading, so a decoder reciting the corpus scores identically here and in `generate`.

        Args:
            z (np.ndarray | torch.Tensor): `(m, z_dim)` training-split vectors to average.
            n (int): Rows to produce.

        Returns:
            torch.Tensor: `(n, slots, lm_dim)` prefix embeddings.
        """
        mean = _as_tensor(z, self.device.device).mean(dim=0, keepdim=True)
        return self.bridge(mean).expand(n, -1, -1)

    # ---- Free-running decode ---- #

    @torch.no_grad()
    def generate(
        self,
        z: np.ndarray | torch.Tensor,
        *,
        max_new_tokens: int | None = None,
        beams: int | None = None,
        batch_size: int = 8,
    ) -> list[str]:
        """Decodes free-running text from conditioning vectors.

        Args:
            z (np.ndarray | torch.Tensor): `(n, z_dim)` bridge-ready vectors.
            max_new_tokens (int | None, optional): Decode cap. Defaults to None, which uses the configured value.
            beams (int | None, optional): Beam width. Defaults to None, which uses the configured value.
            batch_size (int, optional): Rows per decode call. Defaults to 8.

        Returns:
            list[str]: One hypothesis per row.
        """
        tensor = _as_tensor(z, self.device.device)
        out: list[str] = []
        for lo, hi in _spans(tensor.shape[0], batch_size):
            out.extend(self._decode(self.bridge(tensor[lo:hi]), max_new_tokens, beams))
        return out

    @torch.no_grad()
    def generate_from_prefix(
        self,
        prefix: torch.Tensor,
        *,
        max_new_tokens: int | None = None,
        beams: int | None = None,
        batch_size: int = 8,
    ) -> list[str]:
        """Decodes free-running text from ready-made prefixes, which is how the prefix-side controls run.

        Args:
            prefix (torch.Tensor): `(n, slots, lm_dim)` soft prompts.
            max_new_tokens (int | None, optional): Decode cap. Defaults to None.
            beams (int | None, optional): Beam width. Defaults to None.
            batch_size (int, optional): Rows per decode call. Defaults to 8.

        Returns:
            list[str]: One hypothesis per row.
        """
        out: list[str] = []
        for lo, hi in _spans(prefix.shape[0], batch_size):
            out.extend(self._decode(prefix[lo:hi], max_new_tokens, beams))
        return out

    # ---- Retrieval and diagnostics ---- #

    @torch.no_grad()
    def rescore(
        self,
        z: np.ndarray | torch.Tensor,
        candidate_texts: Sequence[str],
        *,
        length_normalise: bool = True,
        batch_size: int = 8,
        chunk: int = 64,
    ) -> np.ndarray:
        """Scores every gallery sentence under every conditioning vector -- this is RETRIEVAL, not generation.

        The decoder is handed the candidate set, so a high score here says the true sentence ranks well among 700
        alternatives, not that the model wrote it. It is reported as `decoder_rescoring_retrieval` and never as a
        generation result.

        Args:
            z (np.ndarray | torch.Tensor): `(n_query, z_dim)` bridge-ready vectors.
            candidate_texts (Sequence[str]): The gallery, in the order the returned columns follow.
            length_normalise (bool, optional): Divide by token count, so the ranking is not a length ranking.
                Defaults to True.
            batch_size (int, optional): Queries per pass. Defaults to 8.
            chunk (int, optional): Candidate rows per LM forward. Defaults to 64.

        Returns:
            np.ndarray: `(n_query, n_candidates)` length-normalised sequence log-probabilities.
        """
        ids, mask = self._tokenise(candidate_texts)
        tensor = _as_tensor(z, self.device.device)
        scores: list[np.ndarray] = []
        for lo, hi in progress(
            list(_spans(tensor.shape[0], batch_size)), description='rescoring gallery'
        ):
            prefix = self.bridge(tensor[lo:hi])
            block = self.lm.sequence_logprob(prefix, ids, mask, length_normalise, chunk)
            scores.append(block.float().cpu().numpy())
        if not scores:
            return np.empty((0, len(candidate_texts)), dtype=np.float32)
        return np.concatenate(scores).astype(np.float32)

    @torch.no_grad()
    def teacher_forced_nll(
        self, z: np.ndarray | torch.Tensor, texts: Sequence[str], batch_size: int = 8
    ) -> np.ndarray:
        """Returns the per-sentence teacher-forced negative log-likelihood -- a DIAGNOSTIC, never a headline.

        Teacher forcing hands the model every previous reference token, so this number measures the LM's fluency far
        more than it measures the brain. It is stored under a quarantined key and no verdict reads it.

        Args:
            z (np.ndarray | torch.Tensor): `(n, z_dim)` bridge-ready vectors.
            texts (Sequence[str]): One reference per row.
            batch_size (int, optional): Rows per forward pass. Defaults to 8.

        Returns:
            np.ndarray: `(n,)` token-mean negative log-likelihoods.
        """
        ids, mask = self._tokenise(texts)
        tensor = _as_tensor(z, self.device.device)
        out: list[np.ndarray] = []
        for lo, hi in _spans(tensor.shape[0], batch_size):
            prefix = self.bridge(tensor[lo:hi])
            logprob = self.lm.target_token_logprobs(prefix, ids[lo:hi], mask[lo:hi])
            tokens = mask[lo:hi].sum(dim=1).clamp_min(1).to(logprob.dtype)
            out.append((-logprob.sum(dim=1) / tokens).float().cpu().numpy())
        if not out:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def prefix_influence_kl(
        self, z: np.ndarray | torch.Tensor, batch_size: int = 8, seed: int = 0
    ) -> np.ndarray:
        """Returns the per-row KL in nats between a reading's own prefix and another reading's prefix.

        This is the bridge-collapse detector the generation verdict gates on, and it measures the one thing that
        matters: whether the prompt depends on the conditioning vector. A bridge whose output cannot vary with `z`
        hands every reading the same prompt, so both distributions are the same distribution and the divergence is
        exactly 0 -- which is why the verdict refuses to consider a run below `decoder.min_prefix_kl`. The partner is
        drawn without fixed points, so no row is ever compared against itself.

        Args:
            z (np.ndarray | torch.Tensor): `(n, z_dim)` bridge-ready vectors.
            batch_size (int, optional): Rows per forward pass. Defaults to 8.
            seed (int, optional): Seed for the pairing, so a reported number can be recomputed. Defaults to 0.

        Returns:
            np.ndarray: `(n,)` KL divergences, all-zero when fewer than two rows leave nothing to pair against.
        """
        tensor = _as_tensor(z, self.device.device)
        n = int(tensor.shape[0])
        if n < 2:
            return np.zeros(n, dtype=np.float32)
        partner = torch.from_numpy(_paired_shuffle(n, seed)).to(self.device.device)
        out: list[np.ndarray] = []
        for lo, hi in _spans(n, batch_size):
            own = self.bridge(tensor[lo:hi])
            other = self.bridge(tensor[partner[lo:hi]])
            out.append(self.lm.next_token_kl(own, other).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def null_prefix_kl(self, z: np.ndarray | torch.Tensor, batch_size: int = 8) -> np.ndarray:
        """Returns the per-row KL in nats between the conditional and the learned null prefix.

        How far the conditional prompt sits from the unconditional one, which is what the `null_prefix` control
        decodes from. It is not a collapse detector: the null prefix is a free parameter trained by null dropout, so a
        bridge emitting one constant prompt for every reading still scores above zero here and can clear a
        `min_prefix_kl` floor while ignoring the brain entirely.

        Args:
            z (np.ndarray | torch.Tensor): `(n, z_dim)` bridge-ready vectors.
            batch_size (int, optional): Rows per forward pass. Defaults to 8.

        Returns:
            np.ndarray: `(n,)` KL divergences.
        """
        tensor = _as_tensor(z, self.device.device)
        out: list[np.ndarray] = []
        for lo, hi in _spans(tensor.shape[0], batch_size):
            prefix = self.bridge(tensor[lo:hi])
            out.append(
                self.lm.next_token_kl(prefix, self.bridge.null(prefix.shape[0]))
                .float()
                .cpu()
                .numpy()
            )
        if not out:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    # ---- Internals ---- #

    def _decode(
        self, prefix: torch.Tensor, max_new_tokens: int | None, beams: int | None
    ) -> list[str]:
        """The single free-running decode path: the headline, all five controls and the oracle all land here.

        Raises:
            ValueError: If `decoder.cfg_weight` is not 1.0, which would make the headline and the `null_prefix`
                control different code paths and the comparison between them meaningless.
        """
        weight = self.decoder_config.cfg_weight
        if weight != 1.0:
            raise ValueError(
                f'decoder.cfg_weight={weight} is not supported: guidance would make the headline decode and the '
                'null_prefix control different code paths. Set it to 1.0.'
            )
        return self.lm.generate_from_prefix(
            prefix,
            max_new_tokens=max_new_tokens or self.decoder_config.max_new_tokens,
            beams=beams or self.decoder_config.beams,
        )

    def _sentence_z(self, batch: dict[str, Any]) -> torch.Tensor:
        """Runs the training-time conditioning recipe: pool, project, align to the text space, correct the gap."""
        emb = self.model.project(self.model.sentence_hidden(batch))
        z = F.normalize(self.clip_head(emb) if self.clip_head is not None else emb, dim=-1)
        return self.gap(z)

    def _tokenise(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenises reference or gallery sentences with the checkpoint's own tokeniser."""
        config = self.decoder_config
        targets = build_target_tokens(
            list(texts),
            config.tokenizer_source or config.lm_source,
            revision=config.lm_revision,
            max_length=config.max_target_tokens,
            model_cache_dir=config.lm_cache_dir,
        )
        device = self.device.device
        return (
            torch.from_numpy(targets.ids).to(device),
            torch.from_numpy(targets.mask).to(device),
        )


def _spans(n: int, size: int) -> Iterator[tuple[int, int]]:
    """Yields `(start, stop)` row spans of at most `size`."""
    step = max(int(size), 1)
    for start in range(0, n, step):
        yield start, min(start + step, n)


def _paired_shuffle(n: int, seed: int) -> np.ndarray:
    """Returns a permutation of `range(n)` with no fixed point, so no row is ever paired with itself."""
    perm = np.random.default_rng(seed).permutation(n)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    return perm.astype(np.int64)


def _as_tensor(z: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Returns `z` as a float32 tensor on `device`."""
    if torch.is_tensor(z):
        return z.to(device=device, dtype=torch.float32)
    return torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)).to(device)


def _sub_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Returns the entries of `state` under `prefix`, with the prefix removed."""
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


def _decoder_config(extra: dict[str, Any], config: ZTEConfig) -> DecoderConfig:
    """Returns the decoder configuration the checkpoint was trained under."""
    stored = extra.get('decoder_config')
    if not isinstance(stored, dict):
        return config.decoder
    fields = {f.name for f in DecoderConfig.__dataclass_fields__.values()}
    return DecoderConfig(**{k: v for k, v in stored.items() if k in fields})


def _rebuild_encoder(
    config: ZTEConfig, extra: dict[str, Any], dataset: ZuCoDataset | None
) -> ZTEModel:
    """Rebuilds the encoder at the shapes the checkpoint records.

    Raises:
        ValueError: If neither the checkpoint nor a supplied dataset yields input shapes.
    """
    in_dim = extra.get('in_dim')
    raw_shape = extra.get('raw_shape')
    raw_shape = tuple(raw_shape) if raw_shape is not None else None
    if in_dim is None and raw_shape is None and dataset is not None:
        in_dim = None if dataset.features is None else int(dataset.features.shape[1])
        raw_shape = (
            None
            if dataset.raw_eeg is None
            else (int(dataset.raw_eeg.shape[1]), int(dataset.raw_eeg.shape[2]))
        )
    if in_dim is None and raw_shape is None:
        raise ValueError('Checkpoint lacks input shapes; pass the dataset it was trained on.')
    return build_model(
        config.model,
        in_dim=in_dim,
        raw_shape=raw_shape,
        n_channels=extra.get('n_channels'),
        bp_features_per_channel=extra.get('bp_features_per_channel'),
        montage_csv=extra.get('montage_csv'),
        signature_dim=int(extra.get('signature_dim') or 0),
    )


def _rebuild_clip_head(state: dict[str, torch.Tensor]) -> nn.Linear | None:
    """Rebuilds the projection into the frozen text space from a decoder state dict."""
    weight = state.get('clip_head.weight')
    if weight is None:
        return None
    bias = state.get('clip_head.bias')
    head = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=bias is not None)
    with torch.no_grad():
        head.weight.copy_(weight)
        if bias is not None:
            head.bias.copy_(bias)
    return head


def _sentence_meta(dataset: ZuCoDataset, torch_ds: ZuCoTorchDataset) -> pd.DataFrame:
    """Builds the per-reading metadata frame that names each generation's subject, length and reference."""
    texts = torch_ds.stimulus_texts
    vocab = torch_ds.text_vocab
    rows: list[dict[str, Any]] = []
    for i, sequence in enumerate(torch_ds.sequences):
        first = dataset.words.iloc[int(sequence[0])]
        key = torch_ds.stimulus_keys[i]
        rows.append(
            {
                'subject': first['subject'],
                'task': first['task'],
                'sentence_idx': int(first['sentence_idx']),
                'n_words': int(len(sequence)),
                'reading_id': int(torch_ds.reading_ids[i]),
                'text_id': int(vocab.get(key, -1)),
                'stimulus_key': key,
                'text': texts.get(key, key),
            }
        )
    return pd.DataFrame(rows)
