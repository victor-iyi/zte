"""Inference: drive a frozen LM from a trained bridge -- free-running decode and decoder-rescoring retrieval."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
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
from zte.models.decoder import (
    EvidenceFn,
    FrozenLM,
    GapCorrector,
    PrefixBridge,
    SemanticRateLadder,
    WordEvidence,
    build_bridge,
    build_evidence,
    build_lm,
    build_rate_ladder,
)
from zte.models.embedding import ZTEModel, build_model
from zte.models.objectives.lexical import LexicalAligner
from zte.training.checkpoint import CheckpointManager

_LOG = get_logger('inference.decode')


@dataclass(slots=True)
class ReadingBatch:
    """One split's readings, embedded once and reused by every readout and every control.

    Attributes:
        z (np.ndarray): Bridge-ready conditioning vectors `(n, z_dim)`, gap-corrected and rate-quantised.
        meta (pd.DataFrame): Per-reading metadata: `subject`, `task`, `n_words`, `reading_id`, `text_id`,
            `stimulus_key` and the reference `text`.
        words (np.ndarray | None): Per-word text-space vectors `(n, max_words, text_dim)` for the evidence path.
        valid (np.ndarray | None): Boolean `(n, max_words)` marking readable word positions.
        durations (np.ndarray | None): Per-word read time `(n, max_words)` for the `fixation` pointer schedule.
        codes (np.ndarray | None): Rate-ladder codes `(n, n_stages)`, the measured conditioning channel.
    """

    z: np.ndarray
    meta: pd.DataFrame
    words: np.ndarray | None = None
    valid: np.ndarray | None = None
    durations: np.ndarray | None = None
    codes: np.ndarray | None = None

    def __len__(self) -> int:
        """Number of readings."""
        return int(len(self.meta))

    @classmethod
    def from_vectors(cls, z: np.ndarray | torch.Tensor) -> ReadingBatch:
        """Wraps bare conditioning vectors for a decode that needs no metadata and no evidence.

        Args:
            z (np.ndarray | torch.Tensor): `(n, z_dim)` bridge-ready vectors.

        Returns:
            ReadingBatch: A metadata-free batch; the pooled prefix path is complete without it.
        """
        rows = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
        return cls(z=rows.astype(np.float32), meta=pd.DataFrame(index=range(len(rows))))

    def take(self, rows: np.ndarray) -> ReadingBatch:
        """Returns the same batch re-ordered or subset by `rows`, keeping every per-word tensor aligned.

        Args:
            rows (np.ndarray): Row indices into this batch.

        Returns:
            ReadingBatch: The re-ordered view; the metadata frame keeps the *original* rows so a control decoded from
                a partner's brain is still scored against its own reference.
        """
        return ReadingBatch(
            z=self.z[rows],
            meta=self.meta,
            words=None if self.words is None else self.words[rows],
            valid=None if self.valid is None else self.valid[rows],
            durations=None if self.durations is None else self.durations[rows],
            codes=None if self.codes is None else self.codes[rows],
        )


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
        bridge (PrefixBridge): The trained soft-prompt bridge.
        ladder (SemanticRateLadder | None): The rate ladder, when the run used one.
        evidence (WordEvidence | None): The word-synchronous path, when the run used one.
        lexical (LexicalAligner | None): The per-word projection the evidence path reads.
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
        ladder: SemanticRateLadder | None = None,
        evidence: WordEvidence | None = None,
        lexical: LexicalAligner | None = None,
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
            ladder (SemanticRateLadder | None, optional): The rate ladder. Defaults to None.
            evidence (WordEvidence | None, optional): The word-synchronous path. Defaults to None.
            lexical (LexicalAligner | None, optional): The per-word text projection. Defaults to None.
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
        self.ladder = None if ladder is None else ladder.to(target).eval()
        self.evidence = None if evidence is None else evidence.to(target).eval()
        self.lexical = None if lexical is None else lexical.to(target).eval()

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
        """Rebuilds every trained part of a decoder run from its checkpoint.

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
        lm = build_lm(decoder_config, encoder=model)
        z_dim = int(state['bridge.to_bottleneck.weight'].shape[1])
        bridge, _ = build_bridge(decoder_config, z_dim, cast('int', model.hidden_dim), lm.hidden_dim)
        bridge.load_state_dict(_sub_state(state, 'bridge.'), strict=True)

        gap_state = extra.get('gap_correction')
        if not gap_state and decoder_config.gap_correction != 'none':
            raise ValueError(
                f'{ckpt_path} was trained with gap_correction={decoder_config.gap_correction!r} but carries no fitted '
                "statistics in extra['gap_correction']. An unfitted corrector passes vectors through, so every number "
                'decoded from it would be produced off the text manifold while provenance claimed a correction.'
            )
        gap = (
            GapCorrector.from_state(gap_state) if gap_state else GapCorrector(z_dim, mode=decoder_config.gap_correction)
        )
        decoder = cls(
            model=model,
            config=config,
            decoder_config=decoder_config,
            bridge=bridge,
            lm=lm,
            gap=gap,
            clip_head=_rebuild_clip_head(state),
            ladder=_rebuild_ladder(decoder_config, state, z_dim),
            evidence=_rebuild_evidence(decoder_config, state, z_dim, lm.hidden_dim),
            lexical=_rebuild_lexical(state),
            device=device,
        )
        if extra.get('normalizer'):
            decoder.normalizer = FeatureNormalizer.from_state(extra['normalizer'])
        if extra.get('aligner'):
            decoder.aligner = RawSubjectAligner.from_state(extra['aligner'])
        decoder.subject_vocab = extra.get('subject_vocab')
        _LOG.info(
            'Loaded decoder checkpoint %s (epoch %s, z_dim %d, %d prefix slots, LM %s, ladder %s, evidence %s).',
            ckpt_path,
            payload.get('epoch'),
            z_dim,
            bridge.slots,
            decoder_config.lm_source,
            'on' if decoder.ladder is not None else 'off',
            'on' if decoder.evidence is not None else 'off',
        )
        return decoder

    # ---- Conditioning ---- #

    @property
    def z_dim(self) -> int:
        """Width of the conditioning vector the bridge reads."""
        return int(self.bridge.norm_in.normalized_shape[0])

    @property
    def uses_evidence(self) -> bool:
        """Whether this checkpoint decodes with the word-synchronous evidence path."""
        return self.evidence is not None and self.lexical is not None

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
    ) -> ReadingBatch:
        """Embeds every reading in a split into everything the frozen LM will be handed.

        The returned vectors are already gap-corrected and rate-quantised, so `prefix_from_z` applies the bridge and
        nothing else. A true text embedding fed to `prefix_from_z` is therefore the text oracle: the identical head,
        without the correction that exists only to move EEG vectors onto the text cloud.

        Args:
            dataset (ZuCoDataset): A built dataset, re-scaled in place by `prepare_dataset` before it is embedded.
            indices (np.ndarray | None, optional): Word-row indices selecting a split. Defaults to None.
            batch_size (int, optional): Sentences per forward pass. Defaults to 16.
            transform (Callable[[dict[str, Any]], dict[str, Any]] | None, optional): Applied to each device-resident
                batch before the encoder sees it, which is how the signal-destroying controls (phase-scrambled,
                noise-matched) run through this identical path rather than a parallel one. Defaults to None.

        Returns:
            ReadingBatch: The conditioning vectors, per-word evidence tensors and the metadata frame.
        """
        self.prepare_dataset(dataset)
        vocab = self.subject_vocab or build_subject_vocab(dataset)
        torch_ds = ZuCoTorchDataset(dataset, indices=indices, subject_vocab=vocab)
        loader = make_dataloader(torch_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        vectors: list[np.ndarray] = []
        codes: list[np.ndarray] = []
        words: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        for batch in progress(loader, description='conditioning'):
            moved = {k: (v.to(self.device.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            ready = moved if transform is None else transform(moved)
            z, code, word, valid = self._sentence_z(ready)
            vectors.append(z.cpu().numpy())
            if code is not None:
                codes.append(code.cpu().numpy())
            if word is not None and valid is not None:
                words.append(word.float().cpu().numpy())
                valids.append(valid.cpu().numpy())

        z_all = np.concatenate(vectors) if vectors else np.empty((0, self.z_dim), np.float32)
        return ReadingBatch(
            z=z_all.astype(np.float32),
            meta=_sentence_meta(dataset, torch_ds),
            words=_stack_ragged(words) if words else None,
            valid=_stack_ragged(valids) if valids else None,
            codes=np.concatenate(codes) if codes else None,
        )

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

    def length_matched_z(
        self,
        train_z: np.ndarray,
        train_words: np.ndarray,
        query_words: np.ndarray,
        tol: int = 1,
    ) -> np.ndarray:
        """Returns, per query, the mean training vector of readings with a matching word count.

        Note:
            This is the `length_only` control, and it is the one the ZuCo arithmetic demands. Word count alone carries
            5.14 bits of sentence identity here, free, from eye-tracking segmentation -- so a decoder can look like it
            is reading the brain while only reading how long the sentence was. A plain mean prefix does not test that;
            a length-conditional mean prefix has exactly the length information and nothing else.
        """
        lengths = np.asarray(train_words, dtype=np.float64).ravel()
        queries = np.asarray(query_words, dtype=np.float64).ravel()
        out = np.zeros((queries.size, train_z.shape[1]), dtype=np.float32)
        overall = train_z.mean(axis=0) if len(train_z) else np.zeros(train_z.shape[1], np.float32)
        for i, want in enumerate(queries):
            rows = np.flatnonzero(np.abs(lengths - want) <= max(int(tol), 0))
            # A word count no training reading shares falls back to the widest available band rather than to a
            # different length, which would give the control information the headline never had.
            if rows.size == 0 and lengths.size:
                rows = np.flatnonzero(np.abs(lengths - want) == np.abs(lengths - want).min())
            out[i] = train_z[rows].mean(axis=0) if rows.size else overall
        return out

    # ---- Free-running decode ---- #

    @torch.no_grad()
    def generate(
        self,
        readings: ReadingBatch,
        *,
        max_new_tokens: int | None = None,
        beams: int | None = None,
        batch_size: int = 8,
        evidence_content: bool = True,
    ) -> list[str]:
        """Decodes strictly autoregressive text from a reading batch.

        Args:
            readings (ReadingBatch): The conditioning bundle.
            max_new_tokens (int | None, optional): Decode cap. Defaults to None, which uses the configured value.
            beams (int | None, optional): Beam width. Defaults to None, which uses the configured value.
            batch_size (int, optional): Rows per decode call. Defaults to 8.
            evidence_content (bool, optional): Keep the per-word lexical content. `False` keeps the pointer schedule
                -- and therefore the word count -- and destroys only what each word was, which is the `length_only`
                control. Defaults to True.

        Returns:
            list[str]: One hypothesis per row.
        """
        tensor = _as_tensor(readings.z, self.device.device)
        out: list[str] = []
        for lo, hi in _spans(tensor.shape[0], batch_size):
            evidence = self._evidence_fn(readings, lo, hi, content=evidence_content)
            out.extend(self._decode(self.bridge(tensor[lo:hi]), max_new_tokens, beams, evidence))
        return out

    @torch.no_grad()
    def generate_from_prefix(
        self,
        prefix: torch.Tensor,
        *,
        readings: ReadingBatch | None = None,
        max_new_tokens: int | None = None,
        beams: int | None = None,
        batch_size: int = 8,
        evidence_content: bool = True,
    ) -> list[str]:
        """Decodes from ready-made prefixes, which is how the prefix-side controls run.

        Args:
            prefix (torch.Tensor): `(n, slots, lm_dim)` soft prompts.
            readings (ReadingBatch | None, optional): Supplies the evidence path's word tensors and its pointer
                schedule. Defaults to None, which decodes from the prefix alone.
            max_new_tokens (int | None, optional): Decode cap. Defaults to None.
            beams (int | None, optional): Beam width. Defaults to None.
            batch_size (int, optional): Rows per decode call. Defaults to 8.
            evidence_content (bool, optional): Keep the per-word lexical content. Defaults to True.

        Returns:
            list[str]: One hypothesis per row.
        """
        out: list[str] = []
        for lo, hi in _spans(prefix.shape[0], batch_size):
            evidence = None if readings is None else self._evidence_fn(readings, lo, hi, content=evidence_content)
            out.extend(self._decode(prefix[lo:hi], max_new_tokens, beams, evidence))
        return out

    # ---- Retrieval and diagnostics ---- #

    @torch.no_grad()
    def rescore(
        self,
        readings: ReadingBatch,
        candidate_texts: Sequence[str],
        *,
        length_normalise: bool = True,
        batch_size: int = 8,
        chunk: int | None = None,
        pmi: bool | None = None,
    ) -> np.ndarray:
        """Scores every gallery sentence under every conditioning vector -- this is RETRIEVAL, not generation.

        The decoder is handed the candidate set, so a high score here says the true sentence ranks well among 700
        alternatives, not that the model wrote it. It is reported as `decoder_rescoring_retrieval` and never as a
        generation result.

        Args:
            readings (ReadingBatch): The conditioning bundle.
            candidate_texts (Sequence[str]): The gallery, in the order the returned columns follow.
            length_normalise (bool, optional): Divide by token count, so the ranking is not a length ranking.
                Defaults to True.
            batch_size (int, optional): Queries per pass. Defaults to 8.
            chunk (int | None, optional): Candidate rows per LM forward. Defaults to None, which uses the
                configured `decoder.rescore_chunk`.
            pmi (bool | None, optional): Subtract each candidate's null-prefix score, cancelling the candidate-side
                familiarity bias the train-fitted decoder gives train-cell texts. Defaults to None, which reads
                `decoder.rescore_pmi`.

        Returns:
            np.ndarray: `(n_query, n_candidates)` length-normalised sequence log-probabilities.
        """
        ids, mask = self._tokenise(candidate_texts)
        tensor = _as_tensor(readings.z, self.device.device)
        rows = chunk or self.decoder_config.rescore_chunk
        use_pmi = self.decoder_config.rescore_pmi if pmi is None else pmi
        scores: list[np.ndarray] = []
        for lo, hi in progress(list(_spans(tensor.shape[0], batch_size)), description='rescoring gallery'):
            prefix = self.bridge(tensor[lo:hi])
            evidence = self._evidence_fn(readings, lo, hi)
            block = self.lm.sequence_logprob(prefix, ids, mask, length_normalise, rows, evidence)
            scores.append(block.float().cpu().numpy())
        if not scores:
            return np.empty((0, len(candidate_texts)), dtype=np.float32)

        out = np.concatenate(scores).astype(np.float32)
        if use_pmi:
            # The null branch is query-independent, so one gallery pass broadcasts over every query.
            out = out - self._null_candidate_scores(ids, mask, length_normalise, rows)[None, :]
        return out

    @torch.no_grad()
    def null_rescore(
        self,
        candidate_texts: Sequence[str],
        *,
        length_normalise: bool = True,
        chunk: int | None = None,
    ) -> np.ndarray:
        """Scores every gallery sentence under the learned null prefix -- the unconditional half of the PMI score.

        Args:
            candidate_texts (Sequence[str]): The gallery, in the order the returned entries follow.
            length_normalise (bool, optional): Divide by token count. Defaults to True.
            chunk (int | None, optional): Candidate rows per LM forward. Defaults to None, which uses the
                configured `decoder.rescore_chunk`.

        Returns:
            np.ndarray: `(n_candidates,)` length-normalised sequence log-probabilities.
        """
        ids, mask = self._tokenise(candidate_texts)
        rows = chunk or self.decoder_config.rescore_chunk

        return self._null_candidate_scores(ids, mask, length_normalise, rows)

    def _null_candidate_scores(
        self, ids: torch.Tensor, mask: torch.Tensor, length_normalise: bool, rows: int
    ) -> np.ndarray:
        """Scores the tokenised gallery under the null prefix once, with no evidence, so no query enters the number."""
        block = self.lm.sequence_logprob(self.bridge.null(1), ids, mask, length_normalise, rows, None)

        return block[0].float().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def teacher_forced_nll(self, readings: ReadingBatch, texts: Sequence[str], batch_size: int = 8) -> np.ndarray:
        """Returns the per-sentence teacher-forced negative log-likelihood -- a DIAGNOSTIC, never a headline.

        Teacher forcing hands the model every previous reference token, so this number measures the LM's fluency far
        more than it measures the brain. It is stored under a quarantined key and no verdict reads it.

        Args:
            readings (ReadingBatch): The conditioning bundle.
            texts (Sequence[str]): One reference per row.
            batch_size (int, optional): Rows per forward pass. Defaults to 8.

        Returns:
            np.ndarray: `(n,)` token-mean negative log-likelihoods.
        """
        ids, mask = self._tokenise(texts)
        tensor = _as_tensor(readings.z, self.device.device)
        out: list[np.ndarray] = []
        for lo, hi in _spans(tensor.shape[0], batch_size):
            prefix = self.bridge(tensor[lo:hi])
            evidence = self._evidence_fn(readings, lo, hi)
            logprob = self.lm.target_token_logprobs(prefix, ids[lo:hi], mask[lo:hi], evidence=evidence)
            tokens = mask[lo:hi].sum(dim=1).clamp_min(1).to(logprob.dtype)
            out.append((-logprob.sum(dim=1) / tokens).float().cpu().numpy())
        if not out:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def prefix_influence_kl(self, z: np.ndarray | torch.Tensor, batch_size: int = 8, seed: int = 0) -> np.ndarray:
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
        partner = torch.from_numpy(paired_shuffle(n, seed)).to(self.device.device)
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
            out.append(self.lm.next_token_kl(prefix, self.bridge.null(prefix.shape[0])).float().cpu().numpy())
        if not out:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def bit_report(self, readings: ReadingBatch) -> dict[str, Any] | None:
        """Measures the bits the rate ladder actually delivered on this split, or `None` without a ladder.

        Args:
            readings (ReadingBatch): The conditioning bundle, whose `codes` were recorded during embedding.

        Returns:
            dict[str, Any] | None: The ladder's bit report against sentence identity.
        """
        if self.ladder is None or readings.codes is None:
            return None
        targets = readings.meta['text_id'].to_numpy() if 'text_id' in readings.meta else None
        return self.ladder.bit_report(readings.codes, targets)

    # ---- Internals ---- #

    def _evidence_fn(self, readings: ReadingBatch, lo: int, hi: int, *, content: bool = True) -> EvidenceFn | None:
        """Builds the per-step nudge closure for one row span, or `None` when the run has no evidence path."""
        if self.evidence is None or readings.words is None or readings.valid is None:
            return None

        device = self.device.device
        words = torch.from_numpy(readings.words[lo:hi]).to(device)
        valid = torch.from_numpy(readings.valid[lo:hi]).to(device)
        if not content:
            words = self.evidence.null(words)
        durations = None
        if readings.durations is not None:
            durations = torch.from_numpy(readings.durations[lo:hi]).to(device)
        return lambda steps: self.evidence.nudge(words, valid, steps, durations)  # type: ignore[union-attr]

    @torch.no_grad()
    def decode_trace(
        self,
        readings: ReadingBatch,
        *,
        max_new_tokens: int | None = None,
        batch_size: int = 4,
    ) -> list[dict[str, Any]]:
        """Decodes every reading and returns what the decoder did at each step, not only what it wrote.

        Note:
            The text this returns is produced by the same `generate_from_prefix` call the headline uses, with the
            trace sink as the only difference, so the studio can never show a decode the evaluation did not make.
            It is still free-running: no reference, no reference length, no candidate set.

        Args:
            readings (ReadingBatch): The conditioning bundle.
            max_new_tokens (int | None, optional): Decode cap. Defaults to None, which uses the configured value.
            batch_size (int, optional): Rows per decode call; the trace is memory-hungry. Defaults to 4.

        Returns:
            list[dict[str, Any]]: Per reading: the hypothesis, the per-step record, the pointer's walk over the
                reading's words, and the rate-ladder codes that carried the conditioning.
        """
        tensor = _as_tensor(readings.z, self.device.device)
        steps = max_new_tokens or self.decoder_config.max_new_tokens
        records: list[dict[str, Any]] = []
        for lo, hi in _spans(tensor.shape[0], batch_size):
            sink: list[list[dict[str, Any]]] = []
            evidence = self._evidence_fn(readings, lo, hi, content=True)
            texts = self.lm.generate_from_prefix(
                self.bridge(tensor[lo:hi]),
                max_new_tokens=steps,
                beams=self.decoder_config.beams,
                evidence=evidence,
                trace=sink,
            )
            pointer = self._pointer_walk(readings, lo, hi, steps)
            for row, text in enumerate(texts):
                records.append(
                    {
                        'row': lo + row,
                        'hypothesis': text,
                        'steps': sink[row] if row < len(sink) else [],
                        'pointer': None if pointer is None else pointer[row].tolist(),
                        'codes': None if readings.codes is None else readings.codes[lo + row].tolist(),
                    }
                )
        return records

    def _pointer_walk(self, readings: ReadingBatch, lo: int, hi: int, steps: int) -> np.ndarray | None:
        """Returns the `(rows, steps, words)` pointer weights, which are a function of the schedule alone."""
        if self.evidence is None or readings.valid is None:
            return None

        device = self.device.device
        valid = torch.from_numpy(readings.valid[lo:hi]).to(device)
        durations = None if readings.durations is None else torch.from_numpy(readings.durations[lo:hi]).to(device)
        index = torch.arange(steps, device=device)

        return self.evidence.pointer(index, valid, durations).float().cpu().numpy()

    def _decode(
        self,
        prefix: torch.Tensor,
        max_new_tokens: int | None,
        beams: int | None,
        evidence: EvidenceFn | None,
    ) -> list[str]:
        """The single free-running decode path: the headline, every control and the oracle all land here.

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
            evidence=evidence,
        )

    def _sentence_z(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Runs the training-time conditioning recipe: pool, project, align, correct the gap, quantise."""
        valid = self.model.pooling_mask(batch)
        words = None
        if self.uses_evidence:
            hidden = self.model.contextualize(self.model.token_hidden(batch), valid)
            pooled = self.model._pool_tokens(hidden, valid)  # noqa: SLF001 -- shared pooling
            words = self.evidence.word_vectors(self.lexical.project(hidden))  # type: ignore[union-attr]
        else:
            pooled = self.model.sentence_hidden(batch)

        emb = self.model.project(pooled)
        z = self.gap(F.normalize(self.clip_head(emb) if self.clip_head is not None else emb, dim=-1))
        codes = None
        if self.ladder is not None:
            out = self.ladder(z, valid.sum(dim=1).long())
            z, codes = out.z, out.codes
        return z, codes, words, (valid if words is not None else None)

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


def _stack_ragged(blocks: list[np.ndarray]) -> np.ndarray:
    """Concatenates per-batch tensors that were padded to their own batch's longest sentence."""
    width = max(block.shape[1] for block in blocks)
    padded = [
        block
        if block.shape[1] == width
        else np.pad(block, [(0, 0), (0, width - block.shape[1])] + [(0, 0)] * (block.ndim - 2))
        for block in blocks
    ]
    return np.concatenate(padded)


def paired_shuffle(n: int, seed: int) -> np.ndarray:
    """Returns a permutation of `range(n)` with no fixed point, so no row is ever paired with itself.

    Args:
        n (int): Number of rows.
        seed (int): Seed, so a reported pairing can be recomputed.

    Returns:
        np.ndarray: The derangement.
    """
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


def _rebuild_encoder(config: ZTEConfig, extra: dict[str, Any], dataset: ZuCoDataset | None) -> ZTEModel:
    """Rebuilds the encoder at the shapes the checkpoint records.

    Raises:
        ValueError: If neither the checkpoint nor a supplied dataset yields input shapes.
    """
    in_dim = extra.get('in_dim')
    raw_shape = extra.get('raw_shape')
    raw_shape = tuple(raw_shape) if raw_shape is not None else None
    if in_dim is None and raw_shape is None and dataset is not None:
        in_dim = None if dataset.features is None else int(dataset.features.shape[1])
        raw_shape = None if dataset.raw_eeg is None else (int(dataset.raw_eeg.shape[1]), int(dataset.raw_eeg.shape[2]))
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


def _rebuild_lexical(state: dict[str, torch.Tensor]) -> LexicalAligner | None:
    """Rebuilds the per-word text projection the evidence path reads, or `None` when the run had none."""
    weight = state.get('lexical.head.weight')
    if weight is None:
        return None
    aligner = LexicalAligner(int(weight.shape[1]), int(weight.shape[0]))
    aligner.load_state_dict(_sub_state(state, 'lexical.'), strict=False)
    return aligner


def _rebuild_ladder(config: DecoderConfig, state: dict[str, torch.Tensor], z_dim: int) -> SemanticRateLadder | None:
    """Rebuilds the rate ladder from a decoder state dict, or `None` when the run had none."""
    if 'ladder.codebook' not in state:
        return None
    ladder = build_rate_ladder(config, z_dim, max_words=config.max_target_tokens)
    if ladder is None:
        return None
    ladder.load_state_dict(_sub_state(state, 'ladder.'), strict=False)
    return ladder


def _rebuild_evidence(
    config: DecoderConfig, state: dict[str, torch.Tensor], text_dim: int, lm_dim: int
) -> WordEvidence | None:
    """Rebuilds the word-synchronous evidence path from a decoder state dict, or `None` when the run had none."""
    if 'evidence.down.weight' not in state:
        return None
    evidence = build_evidence(config, text_dim, lm_dim)
    if evidence is None:
        return None
    evidence.load_state_dict(_sub_state(state, 'evidence.'), strict=False)
    return evidence


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
