"""Sentence-level CLIP objective: align EEG sentence vectors to frozen text embeddings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from zte.config import ObjectiveConfig
from zte.models.embedding import ZTEModel
from zte.models.encoder.gallery import GalleryContrast, build_gallery_contrast
from zte.models.objectives.base import _ObjectiveBase, _usable_mask


def _clip_direction(
    logits: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor, cand: torch.Tensor | None = None
) -> torch.Tensor:
    """One direction of a multi-positive InfoNCE over a `(B, B)` similarity matrix.

    Rows are anchors, columns candidates. Candidates without a text target are masked out of the denominator; an
    anchor's positives are the columns sharing its sentence text, so the same sentence read by different subjects
    counts as a positive. An optional pairwise `cand` mask narrows each anchor's denominator further (within-task
    negatives); positives share the anchor's text and therefore its task, so they always survive it.

    Returns:
        torch.Tensor: Scalar mean loss over valid anchors (0 when none).
    """
    neg_inf = torch.finfo(logits.dtype).min
    cols = valid[None, :] if cand is None else valid[None, :] & cand
    masked = logits.masked_fill(~cols, neg_inf)
    denom = torch.logsumexp(masked, dim=1)
    numer = torch.logsumexp(masked.masked_fill(~(pos & cols), neg_inf), dim=1)
    per = denom - numer
    return per[valid].mean() if bool(valid.any()) else logits.new_zeros(())


class SentenceClipObjective(_ObjectiveBase):
    """Symmetric sentence-level CLIP alignment between EEG and a frozen text encoder.

    Each sentence's word-EEG tokens are pooled into one vector, projected to the text space, and aligned to a frozen
    sentence embedding of the ground-truth text by a symmetric InfoNCE loss. The loss is multi-positive: every EEG
    reading of a text is a positive for that text, so subject identity is pushed out. VICReg and the adversaries stay on
    as auxiliaries via `_ObjectiveBase.regularize`.

    Attributes:
        clip_head (nn.Module | None): Projects the pooled sentence embedding to the text-embedding width.
        logit_scale (nn.Parameter): Learnable CLIP temperature (log scale), clamped in the forward pass.
    """

    def __init__(self, config: ObjectiveConfig, model: ZTEModel) -> None:
        """Initialises the CLIP objective.

        Args:
            config (ObjectiveConfig): Objective configuration (uses `clip_temperature`).
            model (ZTEModel): The encoder (its `embed_dim` sizes the CLIP projection head).
        """
        super().__init__(config, model)
        self._embed_dim = model.embed_dim
        self.clip_head: nn.Module | None = None
        self.register_buffer('text_matrix', None, persistent=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / max(config.clip_temperature, 1e-4))))
        self.gallery: GalleryContrast | None = None

    def attach_text(
        self,
        text_matrix: torch.Tensor,
        text_lengths: torch.Tensor | None = None,
        split_text_ids: Sequence[int] | None = None,
        text_tasks: torch.Tensor | None = None,
    ) -> None:
        """Attaches the frozen `(n_sentences, text_dim)` L2-normalised text-embedding matrix.

        Args:
            text_matrix (torch.Tensor): Frozen sentence embeddings indexed by `batch['sentence_text_id']`.
            text_lengths (torch.Tensor | None): Word count of each gallery text, needed only for the length-matched
                denominator of the full-gallery term.
            split_text_ids (Sequence[int] | None): The gallery rows the training split actually reads. The matrix is
                indexed by a whole-dataset id, so without this a stimulus-holding-out split would train the
                full-gallery term against its own held-out sentences as negatives.
            text_tasks (torch.Tensor | None): Long `(n_sentences,)` task per gallery text, needed only for the
                same-task denominator of `within_task_negatives`.
        """
        self.text_matrix = text_matrix  # buffer: moves with .to(device), never trained
        self.clip_head = nn.Linear(self._embed_dim, int(text_matrix.shape[1]))
        n_texts = int(text_matrix.shape[0])
        self.gallery = build_gallery_contrast(self.config, n_texts)
        if self.gallery is None:
            return

        if text_lengths is not None:
            self.gallery.attach_lengths(text_lengths)
        if split_text_ids is not None:
            self.gallery.restrict_to(split_text_ids, n_texts)
        if text_tasks is not None:
            self.gallery.attach_tasks(text_tasks)

    def _sentence_vectors(self, model: ZTEModel, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pools each sentence's word-EEG tokens into one contextual sentence embedding `(B, embed_dim)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(sentence_embeddings (B, embed_dim), token_hiddens (B, L, H))`.
        """
        valid = model.pooling_mask(batch)
        hidden = model.token_hidden(batch)  # (B, L, H)
        hidden_ctx = model.contextualize(hidden, valid)  # sentence-contextual (bidirectional)
        pooled = model._pool_tokens(hidden_ctx, valid)  # (B, H)  # noqa: SLF001 -- shared pooling
        return model.project(pooled), hidden  # (B, embed_dim), plus token hiddens for VICReg/adversary

    # Missing task ids are an error rather than a fallback: silently widening back to cross-task candidates would
    # reintroduce the exact shortcut the knob exists to remove, with nothing in the metrics to show it happened.
    def _task_candidates(self, batch: dict[str, Any]) -> torch.Tensor | None:
        """Returns the `(B, B)` same-task candidate mask, or `None` when `within_task_negatives` is off."""
        if not self.config.within_task_negatives:
            return None
        task_id = batch.get('task_id')
        if task_id is None:
            raise ValueError('within_task_negatives needs task_id in the batch; collate_sentences provides it.')

        return task_id[:, None] == task_id[None, :]

    def compute(self, model: ZTEModel, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the symmetric CLIP loss (+ VICReg/invariance auxiliaries) for a batch.

        Args:
            model (ZTEModel): The ZTE encoder.
            batch (dict[str, Any]): A collated batch dict (uses `sentence_text_id`).

        Returns:
            tuple[torch.Tensor, dict[str, float]]: `(loss, metrics)`.
        """
        usable = _usable_mask(batch)
        z_sent, hidden = self._sentence_vectors(model, batch)  # (B, embed_dim), (B, L, H)
        emb_tok = model.project(hidden)  # token embeddings for VICReg / adversary
        reg_loss, reg_metrics = self.regularize(batch, hidden, emb_tok, usable)

        # Twelve readings of this sentence exist; align against what they agreed on, not only against the text.
        cons_loss, cons_metrics = self.sentence_consensus(z_sent, batch)
        reg_loss, reg_metrics = reg_loss + cons_loss, {**reg_metrics, **cons_metrics}

        # Anti-collapse on the tensor retrieval actually scores: the token-level guard never sees the pooled vector.
        if self.config.sentence_variance_weight > 0.0 or self.config.sentence_covariance_weight > 0.0:
            sent_loss, sent_metrics = self.sentence_regularize(z_sent)
            reg_loss, reg_metrics = reg_loss + sent_loss, {**reg_metrics, **sent_metrics}

        text_id = batch.get('sentence_text_id')
        if self.clip_head is None or self.text_matrix is None or text_id is None:
            zero = z_sent.sum() * 0.0 + reg_loss
            return zero, {'loss': float(reg_loss.detach()), 'n_valid': 0.0, **reg_metrics}

        # Cosine logits between every EEG reading and every text in the batch.
        z_eeg = F.normalize(self.clip_head(z_sent), dim=-1)  # (B, text_dim)
        valid = text_id >= 0
        z_txt = F.embedding(text_id.clamp(min=0), self.text_matrix)  # (B, text_dim), already L2-normed
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = (z_eeg @ z_txt.t()) * scale  # (B, B): row=EEG reading, col=text
        pos = (text_id[:, None] == text_id[None, :]) & valid[:, None] & valid[None, :]
        cand = self._task_candidates(batch)

        if not bool(valid.any()):
            zero = logits.sum() * 0.0 + reg_loss
            return zero, {'loss': float(reg_loss.detach()), 'n_valid': 0.0, **reg_metrics}

        clip_loss = 0.5 * (
            _clip_direction(logits, pos, valid, cand)  # EEG -> text
            + _clip_direction(logits.t(), pos, valid, None if cand is None else cand.t())  # text -> EEG
        )
        loss = clip_loss + reg_loss

        # The batch denominator holds fifteen distractors; the evaluation holds 699, and the hard ones are the
        # same-length neighbours a batch almost never contains.
        if self.gallery is not None and bool(valid.any()):
            gal_loss, gal_metrics = self.gallery.compute(z_eeg[valid], self.text_matrix, text_id[valid], scale)
            loss = loss + self.config.gallery_weight * gal_loss
            reg_metrics.update(gal_metrics)
        with torch.no_grad():
            neg_inf = torch.finfo(logits.dtype).min
            pred = logits.masked_fill(~valid[None, :], neg_inf).argmax(dim=1)
            hit = pos[torch.arange(len(pred), device=pred.device), pred] & valid
            acc = float(hit.sum()) / max(int(valid.sum()), 1)
        return loss, {
            'loss': float(loss.detach()),
            'clip_loss': float(clip_loss.detach()),
            'clip_top1': acc,
            'logit_scale': float(scale.detach()),
            'n_valid': float(valid.sum()),
            **reg_metrics,
        }
