"""Prefix-decode objective: a frozen LM writing text from a soft prompt built out of one EEG sentence vector."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from zte.config import DecoderConfig, ObjectiveConfig, TrainMode
from zte.logging_utils import get_logger
from zte.models.decoder import GapCorrector, build_bridge, build_lm
from zte.models.embedding import ZTEModel
from zte.models.objectives.base import _ObjectiveBase, _usable_mask
from zte.models.objectives.clip import _clip_direction

_LOG = get_logger('models.objectives.decode')

# Grounding softmax temperature: low enough that a prefix scoring every candidate alike pays the full log(M+1).
_GROUND_TEMPERATURE: float = 0.1


@dataclass(slots=True)
class _Encoded:
    """One batch's encoder outputs, computed in a single forward pass.

    Attributes:
        z (torch.Tensor): L2-normalised sentence vectors `(batch_size, z_dim)`, before the gap correction.
        token_mask (torch.Tensor): Boolean `(batch_size, seq_len)` pooling/attention mask.
        hidden (torch.Tensor | None): Pre-contextual token hiddens for the adversary and VICReg, when they are needed.
        hidden_ctx (torch.Tensor | None): Contextual token hiddens for the word resampler, when it exists.
        cache_hits (int): Rows served from the frozen-encoder cache instead of a forward pass.
    """

    z: torch.Tensor
    token_mask: torch.Tensor
    hidden: torch.Tensor | None = None
    hidden_ctx: torch.Tensor | None = None
    cache_hits: int = 0


class PrefixDecodeObjective(_ObjectiveBase):
    """Trains a small bridge so a frozen LM writes the sentence a person was reading.

    Nothing in the LM moves, so the text it produces cannot be corpus recall stored in decoder weights: the entire
    trainable surface is the bridge, and the only thing that varies between one sentence and the next is the
    conditioning vector. The loss pairs teacher-forced cross-entropy with an in-batch grounding term, because
    cross-entropy alone is minimised perfectly well by a prefix that ignores the brain and recites the corpus prior --
    the failure mode `prefix_kl` exists to expose while training rather than after it, by comparing each row's prompt
    against another row's rather than against the unconditional one.

    Attributes:
        decoder_config (DecoderConfig): Bridge geometry, LM identity and generation controls.
        lm (FrozenLM): The frozen causal LM, excluded from every optimiser and every checkpoint.
        bridge (PrefixBridge): The soft-prompt bridge, which is almost all of the trainable surface.
        resampler (WordResampler | None): The word-level ablation arm, present only for
            `conditioning='pooled_plus_words'`.
        gap (GapCorrector): Train-fitted affine map from the EEG cloud onto the text cloud.
        clip_head (nn.Linear | None): Projection into the frozen text space the bridge reads.
        stage (str): The curriculum stage last announced by `set_stage`, recorded for logging.
    """

    text_matrix: torch.Tensor | None
    target_ids: torch.Tensor | None
    target_mask: torch.Tensor | None
    cache_z: torch.Tensor | None
    cache_hit: torch.Tensor | None

    def __init__(self, config: ObjectiveConfig, model: ZTEModel, decoder_config: DecoderConfig) -> None:
        """Builds the frozen LM and the trainable bridge over it.

        The bridge is sized to whatever space the conditioning vector lives in. That is the encoder's own embedding
        width until `attach_clip_head` or `attach_text` names the frozen text space, which is the space that makes
        text-only pretraining possible; either call re-sizes the bridge before the optimiser is ever built.

        Args:
            config (ObjectiveConfig): Objective configuration (uses `clip_temperature` for the joint-mode auxiliary).
            model (ZTEModel): The encoder producing the conditioning vector.
            decoder_config (DecoderConfig): Decoder configuration.
        """
        super().__init__(config, model)
        self.decoder_config = decoder_config
        self.lm = build_lm(decoder_config)
        self._embed_dim: int = int(model.embed_dim)
        self._token_dim: int = cast('int', model.hidden_dim)
        self.stage = 'a'

        self.clip_head: nn.Linear | None = None
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / max(config.clip_temperature, 1e-4))))
        for name in ('text_matrix', 'target_ids', 'target_mask', 'cache_z', 'cache_hit'):
            self.register_buffer(name, None, persistent=False)

        self.z_dim: int = int(model.embed_dim)
        self.bridge, self.resampler = build_bridge(decoder_config, self.z_dim, self._token_dim, self.lm.hidden_dim)
        self.gap = GapCorrector(self.z_dim, mode=decoder_config.gap_correction)

    # ---- Attachment ---- #

    def attach_clip_head(self, weight: torch.Tensor, bias: torch.Tensor | None = None, trainable: bool = False) -> None:
        """Installs the source run's projection into the frozen text space.

        Args:
            weight (torch.Tensor): `(text_dim, embed_dim)` weight from the source objective's `clip_head`.
            bias (torch.Tensor | None, optional): `(text_dim,)` bias. Defaults to None, which builds a bias-free head.
            trainable (bool, optional): Let the head receive gradients. Defaults to False, which keeps the
                conditioning vector a fixed function of the encoder.

        Raises:
            ValueError: If `weight` is not two-dimensional or its input width is not the encoder's embedding width.
        """
        if weight.ndim != 2:
            raise ValueError(f'clip_head weight must be (text_dim, embed_dim), got {tuple(weight.shape)}.')
        text_dim, embed_dim = int(weight.shape[0]), int(weight.shape[1])
        if embed_dim != self._embed_dim:
            raise ValueError(f'clip_head expects an encoder of width {embed_dim} but this one is {self._embed_dim}.')
        head = nn.Linear(embed_dim, text_dim, bias=bias is not None)
        with torch.no_grad():
            head.weight.copy_(weight)
            if bias is not None:
                head.bias.copy_(bias)
        head.requires_grad_(trainable)
        self.clip_head = head
        self._resize(text_dim)
        _LOG.info(
            'Attached clip_head %d -> %d (%s).',
            embed_dim,
            text_dim,
            'trainable' if trainable else 'frozen',
        )

    def attach_text(self, text_matrix: torch.Tensor) -> None:
        """Attaches the frozen `(n_sentences, text_dim)` L2-normalised text embeddings.

        They are the target cloud the gap correction maps onto, the anchor of the joint-mode CLIP auxiliary, and the
        input to text-only bridge pretraining.

        Args:
            text_matrix (torch.Tensor): Frozen sentence embeddings indexed by `batch['sentence_text_id']`.

        Raises:
            ValueError: If an attached `clip_head` projects into a different width.
        """
        text_dim = int(text_matrix.shape[1])
        if self.clip_head is not None and self.clip_head.out_features != text_dim:
            raise ValueError(
                f'Text matrix width {text_dim} does not match the attached clip_head ({self.clip_head.out_features}).'
            )
        self.text_matrix = text_matrix
        if self.clip_head is None:
            self.clip_head = nn.Linear(self._embed_dim, text_dim)
            self._resize(text_dim)
            _LOG.info(
                'No source projection was attached, so a %d -> %d text projection is learned by the decode loss.',
                self._embed_dim,
                text_dim,
            )

    def attach_tokens(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> None:
        """Attaches the tokenised reference sentences, indexed by `batch['sentence_text_id']`.

        Args:
            token_ids (torch.Tensor): `(n_sentences, n_target)` long target ids.
            token_mask (torch.Tensor): `(n_sentences, n_target)` bool, `True` at real target tokens.
        """
        self.target_ids = token_ids.long()
        self.target_mask = token_mask.bool()

    def attach_cache(self, n_readings: int, mode: TrainMode = 'decoder') -> None:
        """Allocates the frozen-encoder sentence-vector cache, keyed by `batch['reading_id']`.

        With the encoder frozen and in eval the conditioning vector is a pure function of the reading, so it is
        computed once and looked up thereafter, which is what makes a twelve-fold sweep affordable.

        Args:
            n_readings (int): Size of the `reading_id` space, from `ZuCoTorchDataset.n_readings`.
            mode (TrainMode, optional): This run's training mode. Defaults to 'decoder'.

        Raises:
            ValueError: If the cache is requested in `'joint'` mode, where the encoder moves and every cached vector
                is stale.
        """
        if not self.decoder_config.cache_embeddings:
            return
        if mode == 'joint':
            raise ValueError(
                "decoder.cache_embeddings cannot be used with train.mode='joint': the encoder moves every epoch, "
                'so a cached sentence vector is a stale conditioning signal rather than a speed-up. Set '
                'decoder.cache_embeddings=false for this run.'
            )
        if self.resampler is not None:
            _LOG.warning(
                "Embedding cache disabled: conditioning='pooled_plus_words' needs the word hiddens every step, "
                'which the cache does not hold.'
            )
            return
        if self._head_trains():
            _LOG.warning(
                'Embedding cache disabled: the text projection is still learning, so a sentence vector cached now '
                'would not be the one the bridge reads next epoch.'
            )
            return
        self.cache_z = torch.zeros(int(n_readings), self.z_dim, dtype=torch.float32)
        self.cache_hit = torch.zeros(int(n_readings), dtype=torch.bool)
        _LOG.info('Frozen-encoder cache: %d readings x %d dims.', n_readings, self.z_dim)

    def set_stage(self, stage: str) -> None:
        """Records the curriculum stage announced by the trainer.

        Args:
            stage (str): The stage name from `zte.training.stages`.
        """
        self.stage = stage

    # ---- Conditioning ---- #

    def conditioning(self, model: ZTEModel, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the gap-corrected sentence vector and the soft prompt it produces.

        Args:
            model (ZTEModel): The encoder.
            batch (dict[str, Any]): A collated batch dict.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(z (batch_size, z_dim), prefix (batch_size, slots, lm_dim))`.
        """
        encoded = self._encode(model, batch, grad=self._encoder_trains(model))
        z = self.gap(encoded.z)
        return z, self._prefix(model, z, encoded)

    @torch.no_grad()
    def fit_gap(self, model: ZTEModel, loader: Iterable[dict[str, Any]], device: torch.device) -> int:
        """Fits the gap correction on the training split, warming the frozen-encoder cache in the same pass.

        Args:
            model (ZTEModel): The encoder.
            loader (Iterable[dict[str, Any]]): The training loader; every batch it yields is a training row, which is
                what keeps the correction non-transductive.
            device (torch.device): Device to move each batch onto.

        Returns:
            int: Rows the correction was fitted on (`0` when there was nothing to do).
        """
        if self.gap.mode == 'none' and self.cache_z is None:
            return 0
        if self.text_matrix is None:
            _LOG.warning('No text matrix attached; the gap correction stays at the identity.')
            return 0
        model.eval()
        rows: list[torch.Tensor] = []
        seen: set[int] = set()
        for raw in loader:
            moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
            rows.append(self._encode(model, moved, grad=False).z.detach().float().cpu())
            text_id = moved.get('sentence_text_id')
            if text_id is not None:
                seen.update(int(i) for i in text_id.tolist() if i >= 0)
        if not rows:
            return 0
        eeg = torch.cat(rows)
        index = torch.as_tensor(sorted(seen), dtype=torch.long, device=self.text_matrix.device)
        txt = self.text_matrix[index] if index.numel() else self.text_matrix
        self.gap.fit(eeg.to(self.text_matrix.device), txt)
        return int(eeg.shape[0])

    def pretrain_text(
        self,
        text_ids: Sequence[int] | np.ndarray,
        *,
        holdout_text_ids: Collection[int] = (),
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
        seed: int = 0,
    ) -> dict[str, float]:
        """Pretrains the bridge on `(text embedding -> text)` pairs, with no EEG involved.

        Learning "a vector on the text manifold becomes this English sentence" is a problem where data is not the
        constraint, so doing it first leaves the EEG stage only the residual. Held-out stimuli are refused outright
        rather than filtered, because a leaked reference here would be memorised into the one component that
        generates text.

        Args:
            text_ids (Sequence[int] | np.ndarray): `sentence_text_id` values of the training stimuli.
            holdout_text_ids (Collection[int], optional): Ids that must not appear in `text_ids`. Defaults to ().
            epochs (int, optional): Passes over the training stimuli. Defaults to 20.
            lr (float, optional): Bridge learning rate. Defaults to 1e-3.
            batch_size (int, optional): Sentences per step. Defaults to 32.
            seed (int, optional): Shuffling seed. Defaults to 0.

        Returns:
            dict[str, float]: `stage0_loss`, `stage0_epochs` and `stage0_texts`.

        Raises:
            ValueError: If a held-out id is present, or if the text embeddings or target tokens are not attached.
        """
        ids = np.asarray(list(text_ids), dtype=np.int64)
        leaked = sorted(set(ids.tolist()) & {int(i) for i in holdout_text_ids})
        if leaked:
            raise ValueError(
                f'Text-only pretraining was given {len(leaked)} held-out stimulus id(s), first {leaked[0]}. '
                'Only train-split stimuli may reach the bridge before evaluation.'
            )
        if self.text_matrix is None or self.target_ids is None or self.target_mask is None:
            raise ValueError('Text-only pretraining needs both attach_text and attach_tokens first.')
        idle = {'stage0_loss': float('nan'), 'stage0_epochs': 0.0, 'stage0_texts': float(ids.size)}
        if epochs <= 0 or ids.size == 0:
            return idle
        params = [p for p in self.bridge.parameters() if p.requires_grad]
        if not params:
            return idle

        device = self.text_matrix.device
        index = torch.as_tensor(ids, dtype=torch.long, device=device)
        optimizer = torch.optim.AdamW(params, lr=lr)
        generator = torch.Generator().manual_seed(seed)
        self.bridge.train()
        loss_value = float('nan')
        for epoch in range(1, epochs + 1):
            order = torch.randperm(index.numel(), generator=generator)
            total, n_steps = 0.0, 0
            for start in range(0, order.numel(), max(batch_size, 1)):
                sel = index[order[start : start + max(batch_size, 1)].to(device)]
                prefix, _ = self.bridge.dropout_null(
                    self.bridge(self.text_matrix[sel]), self.decoder_config.null_prefix_prob
                )
                loss = self.lm.forward_with_prefix(prefix, self.target_ids[sel], self.target_mask[sel])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                n_steps += 1
            loss_value = total / max(n_steps, 1)
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                _LOG.info('Stage 0 epoch %d/%d: text-only CE %.4f.', epoch, epochs, loss_value)
        return {
            'stage0_loss': loss_value,
            'stage0_epochs': float(epochs),
            'stage0_texts': float(ids.size),
        }

    # ---- Loss ---- #

    def compute(self, model: ZTEModel, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the teacher-forced decode loss plus its grounding and joint-mode auxiliaries.

        Args:
            model (ZTEModel): The ZTE encoder.
            batch (dict[str, Any]): A collated batch dict (uses `sentence_text_id` and `reading_id`).

        Returns:
            tuple[torch.Tensor, dict[str, float]]: `(loss, metrics)`.

        Raises:
            RuntimeError: If the encoder is training while a frozen-encoder cache is installed, which would feed the
                bridge sentence vectors the encoder no longer produces.
            ValueError: If no target tokens have been attached.
        """
        joint = self._encoder_trains(model)
        if joint and self.cache_z is not None:
            raise RuntimeError(
                'The encoder is receiving gradients while an embedding cache is installed; the cached conditioning '
                'vectors are stale. Set decoder.cache_embeddings=false for this run.'
            )
        targets, target_mask = self.target_ids, self.target_mask
        if targets is None or target_mask is None:
            raise ValueError('PrefixDecodeObjective needs attach_tokens before it can compute a loss.')

        encoded = self._encode(model, batch, grad=joint)
        z = self.gap(encoded.z)
        prefix = self._prefix(model, z, encoded)

        text_id: torch.Tensor | None = batch.get('sentence_text_id')
        if text_id is None:
            text_id = torch.full((z.shape[0],), -1, dtype=torch.long, device=z.device)
        has_target = text_id >= 0
        ids = targets[text_id.clamp(min=0)]
        mask = target_mask[text_id.clamp(min=0)] & has_target[:, None]

        dropped, replaced = self.bridge.dropout_null(
            prefix, self.decoder_config.null_prefix_prob if self.training else 0.0
        )
        ce = self.lm.forward_with_prefix(dropped, ids, mask) if bool(has_target.any()) else prefix.sum() * 0.0
        ground = self._grounding(prefix, ids, mask, text_id, has_target, has_target & ~replaced)
        loss = ce + self.decoder_config.ground_weight * ground

        metrics: dict[str, float] = {
            'ce': float(ce.detach()),
            'ground': float(ground.detach()),
            'prefix_kl': self._prefix_kl(prefix),
            'null_kl': self._null_kl(prefix),
            'null_frac': float(replaced.float().mean()),
            'cache_hits': float(encoded.cache_hits),
            'n_valid': float(has_target.sum()),
        }
        if joint:
            aux_loss, aux_metrics = self._joint_auxiliaries(model, batch, encoded, text_id, has_target)
            loss = loss + aux_loss
            metrics.update(aux_metrics)
        metrics['loss'] = float(loss.detach())
        return loss, metrics

    # ---- Internals ---- #

    def _resize(self, z_dim: int) -> None:
        """Rebuilds the bridge and gap correction for a conditioning space of width `z_dim`."""
        if z_dim == self.z_dim:
            return
        self.z_dim = z_dim
        self.bridge, self.resampler = build_bridge(self.decoder_config, z_dim, self._token_dim, self.lm.hidden_dim)
        self.gap = GapCorrector(z_dim, mode=self.decoder_config.gap_correction)
        self.cache_z, self.cache_hit = None, None

    @staticmethod
    def _encoder_trains(model: ZTEModel) -> bool:
        """Returns whether any encoder parameter can currently receive a gradient."""
        return any(p.requires_grad for p in model.parameters())

    def _head_trains(self) -> bool:
        """Returns whether the text projection is learning, which makes every cached sentence vector stale."""
        return self.clip_head is not None and any(p.requires_grad for p in self.clip_head.parameters())

    def _cache_view(
        self, batch: dict[str, Any], want_tokens: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Returns `(vectors, filled, reading_id)` when this batch may use the frozen-encoder cache."""
        reading = batch.get('reading_id')
        if want_tokens or self._head_trains() or reading is None:
            return None
        if self.cache_z is None or self.cache_hit is None:
            return None
        return self.cache_z, self.cache_hit, reading

    def _encode(self, model: ZTEModel, batch: dict[str, Any], grad: bool) -> _Encoded:
        """Runs the encoder once, serving the sentence vector from the cache whenever it can."""
        token_mask = model.pooling_mask(batch)
        want_tokens = grad or self.resampler is not None
        cache = self._cache_view(batch, want_tokens)
        if cache is not None:
            vectors, filled, reading = cache
            if bool(filled[reading].all()):
                return _Encoded(
                    z=vectors[reading].clone(),
                    token_mask=token_mask,
                    cache_hits=int(reading.numel()),
                )

        with contextlib.nullcontext() if grad else torch.no_grad():
            hidden = hidden_ctx = None
            if want_tokens:
                hidden = model.token_hidden(batch)
                hidden_ctx = model.contextualize(hidden, token_mask)
                pooled = model._pool_tokens(hidden_ctx, token_mask)  # noqa: SLF001 -- shared pooling
            else:
                pooled = model.sentence_hidden(batch)
            emb = model.project(pooled)

        # The text projection sits outside the encoder's no-grad boundary, so a head learned here still gets gradient.
        emb = emb if grad else emb.detach()
        z = F.normalize(self.clip_head(emb) if self.clip_head is not None else emb, dim=-1)

        hits = 0
        if cache is not None:
            vectors, filled, reading = cache
            miss = ~filled[reading]
            hits = int(reading.numel()) - int(miss.sum())
            vectors[reading[miss]] = z[miss].detach().float()
            filled[reading[miss]] = True
        return _Encoded(z=z, token_mask=token_mask, hidden=hidden, hidden_ctx=hidden_ctx, cache_hits=hits)

    def _prefix(self, model: ZTEModel, z: torch.Tensor, encoded: _Encoded) -> torch.Tensor:
        """Builds the soft prompt, appending the resampled word slots for the `pooled_plus_words` arm."""
        prefix = self.bridge(z)
        if self.resampler is None:
            return prefix
        if encoded.hidden_ctx is None:
            raise RuntimeError("conditioning='pooled_plus_words' needs the contextual token hiddens.")
        words = self.resampler(encoded.hidden_ctx, encoded.token_mask)
        return torch.cat([prefix, words], dim=1)

    def _grounding(
        self,
        prefix: torch.Tensor,
        ids: torch.Tensor,
        mask: torch.Tensor,
        text_id: torch.Tensor,
        has_target: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Scores each item's own reference against in-batch negatives under its own prefix.

        Cross-entropy alone is happy with a prefix that ignores the brain, since the corpus prior explains most of the
        tokens. This term is not: a constant prefix gives every candidate the same score and pays the full `log(M+1)`.
        Rows whose prefix was replaced by the null one are not anchors, since asking the unconditional branch to prefer
        one particular reference is the opposite of what it is for.

        Args:
            prefix (torch.Tensor): Conditional soft prompts `(batch_size, slots, lm_dim)`.
            ids (torch.Tensor): Per-row target ids `(batch_size, n_target)`.
            mask (torch.Tensor): Per-row target mask `(batch_size, n_target)`.
            text_id (torch.Tensor): `(batch_size,)` stimulus ids, so a negative is never the same sentence.
            has_target (torch.Tensor): `(batch_size,)` rows usable as candidates.
            anchors (torch.Tensor): `(batch_size,)` rows usable as queries.

        Returns:
            torch.Tensor: Scalar softmax cross-entropy over `(1 + ground_negatives)` candidates.
        """
        n_neg = self.decoder_config.ground_negatives
        if n_neg <= 0 or prefix.shape[0] < 2:
            return prefix.sum() * 0.0
        allowed = (text_id[:, None] != text_id[None, :]) & has_target[None, :] & has_target[:, None]
        usable = anchors & (allowed.sum(dim=1) > 0)
        if not bool(usable.any()):
            return prefix.sum() * 0.0

        rows = usable.nonzero(as_tuple=False).squeeze(1)
        negatives = torch.multinomial(allowed[rows].float(), n_neg, replacement=True)
        candidates = torch.cat([rows[:, None], negatives], dim=1)
        scores = self.lm.candidate_logprobs(prefix[rows], ids[candidates], mask[candidates], length_normalise=True)
        target = torch.zeros(rows.numel(), dtype=torch.long, device=scores.device)
        return F.cross_entropy(scores / _GROUND_TEMPERATURE, target)

    def _prefix_kl(self, prefix: torch.Tensor) -> float:
        """Returns the mean KL in nats between each row's own prefix and the next row's, which collapse drives to 0."""
        if prefix.shape[0] < 2:
            return 0.0
        other = torch.roll(prefix.detach(), 1, dims=0)
        return float(self.lm.next_token_kl(prefix.detach(), other).mean())

    def _null_kl(self, prefix: torch.Tensor) -> float:
        """Returns the mean KL in nats between the conditional and null-prefix next-token distributions."""
        null = self.bridge.null(prefix.shape[0])
        if self.resampler is not None:
            null = torch.cat([null, prefix[:, self.bridge.slots :].detach()], dim=1)
        return float(self.lm.next_token_kl(prefix.detach(), null).mean())

    def _joint_auxiliaries(
        self,
        model: ZTEModel,
        batch: dict[str, Any],
        encoded: _Encoded,
        text_id: torch.Tensor | None,
        has_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Anchors an unfrozen encoder in the text space and re-applies the shared invariance regularisers."""
        loss = encoded.z.new_zeros(())
        metrics: dict[str, float] = {}
        if encoded.hidden is not None:
            reg_loss, reg_metrics = self.regularize(
                batch, encoded.hidden, model.project(encoded.hidden), _usable_mask(batch)
            )
            loss = loss + reg_loss
            metrics.update(reg_metrics)
        if self.text_matrix is None or text_id is None or not bool(has_target.any()):
            return loss, metrics

        z_txt = F.embedding(text_id.clamp(min=0), self.text_matrix)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = (encoded.z @ z_txt.t()) * scale
        positives = (text_id[:, None] == text_id[None, :]) & has_target[:, None] & has_target[None, :]
        clip_loss = 0.5 * (
            _clip_direction(logits, positives, has_target) + _clip_direction(logits.t(), positives, has_target)
        )
        loss = loss + self.decoder_config.clip_aux_weight * clip_loss
        metrics['clip_loss'] = float(clip_loss.detach())
        return loss, metrics
