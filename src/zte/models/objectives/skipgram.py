"""Skip-gram objective: identify EEG-word neighbours via multi-positive InfoNCE."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from zte.config import ObjectiveConfig
from zte.models.embedding import ZTEModel
from zte.models.heads import ProjectionHead
from zte.models.objectives.base import _ObjectiveBase, _usable_mask
from zte.models.objectives.losses import alignment_penalty, debiased_infonce


class SkipGramObjective(_ObjectiveBase):
    """Multi-positive InfoNCE skip-gram over word-EEG token embeddings.

    Attributes:
        context_head (ProjectionHead): The word2vec "output" embedding projection.
    """

    def __init__(self, config: ObjectiveConfig, model: ZTEModel) -> None:
        """Initialises the skip-gram objective.

        Args:
            config (ObjectiveConfig): Objective configuration (uses `context_window`, `temperature`,
                `cross_subject_positives`).
            model (ZTEModel): The encoder, used to size the context head.
        """
        super().__init__(config, model)
        self.context_head = ProjectionHead(model.hidden_dim, model.config.projection_hidden, model.embed_dim)

    def compute(self, model: ZTEModel, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the skip-gram InfoNCE loss for a batch.

        Positives are either same-sentence neighbours (default) or the same stimulus read by another
        subject (`cross_subject_positives`), which makes subject identity a nuisance rather than a shortcut.

        Args:
            model (ZTEModel): The ZTE encoder.
            batch (dict[str, Any]): A collated batch dict.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: `(loss, metrics)`.
        """
        hidden = model.token_hidden(batch)  # (batch_size, seq_len, hidden_dim), non-contextual
        emb_raw = model.project(hidden)  # (batch_size, seq_len, embed_dim), for VICReg
        center = F.normalize(emb_raw, dim=-1)
        context = F.normalize(self.context_head(hidden), dim=-1)
        b, length, _ = center.shape

        center_flat = center.reshape(b * length, -1)
        context_flat = context.reshape(b * length, -1)
        usable = _usable_mask(batch)
        usable_flat = usable.reshape(-1)  # (n_tokens,)

        not_self = ~torch.eye(b * length, dtype=torch.bool, device=center.device)
        valid_pair = usable_flat[:, None] & usable_flat[None, :]

        # Positives are either the same stimulus read by another subject, or same-sentence neighbours.
        use_cross = self.config.cross_subject_positives and batch.get('content_id') is not None
        if use_cross:
            content_flat = batch['content_id'].reshape(-1)
            has_content = content_flat >= 0
            same_content = content_flat[:, None] == content_flat[None, :]
            pos_mask = same_content & has_content[None, :] & has_content[:, None]
            pos_mask = pos_mask & not_self & valid_pair
        else:
            sent_id = torch.arange(b, device=center.device).repeat_interleave(length)
            pos = torch.arange(length, device=center.device).repeat(b)
            same_sent = sent_id[:, None] == sent_id[None, :]
            within = (pos[:, None] - pos[None, :]).abs() <= self.config.context_window
            pos_mask = same_sent & within & not_self & valid_pair

        # Meaning positives: the same content word across sentences, so clusters follow meaning rather than passage.
        if self.config.meaning_positives and batch.get('word_id') is not None:
            wid = batch['word_id'].reshape(-1)
            has_w = wid >= 0
            same_word = (wid[:, None] == wid[None, :]) & has_w[None, :] & has_w[:, None]
            sent_ids = torch.arange(b, device=center.device).repeat_interleave(length)
            diff_sent = sent_ids[:, None] != sent_ids[None, :]
            pos_mask = pos_mask | (same_word & diff_sent & not_self & valid_pair)

        # Score every anchor against every candidate, then restrict the denominator.
        logits = center_flat @ context_flat.t() / self.config.temperature
        neg_inf = torch.finfo(logits.dtype).min
        cand_mask = usable_flat[None, :] & not_self
        # Hard negatives share the anchor's subject/task, so subject identity cannot win the softmax.
        if self.config.hard_negatives:
            match = torch.ones(b * length, b * length, dtype=torch.bool, device=center.device)
            for key in self.config.hard_negative_keys:
                per_sent = batch.get(key if key != 'task' else 'task_id')
                if per_sent is None:
                    continue
                tok = per_sent[:, None].expand(b, length).reshape(-1)
                match = match & (tok[:, None] == tok[None, :])
            cand_mask = cand_mask & (match | pos_mask)
        logits = logits.masked_fill(~cand_mask, neg_inf)

        has_pos = pos_mask.any(dim=1) & usable_flat
        reg_loss, reg_metrics = self.regularize(batch, hidden, emb_raw, usable)
        if not bool(has_pos.any()):
            zero = center_flat.sum() * 0.0 + reg_loss
            return zero, {'loss': float(reg_loss.detach()), 'n_anchors': 0.0, **reg_metrics}

        anchor_logits = logits[has_pos]
        anchor_pos = pos_mask[has_pos]
        pos_logits = anchor_logits.masked_fill(~anchor_pos, neg_inf)
        numer = torch.logsumexp(pos_logits, dim=1)
        if self.config.tau_plus > 0.0:
            # Debiased contrastive -- correct for same-word false negatives.
            info_loss = debiased_infonce(
                anchor_logits,
                anchor_pos,
                cand_mask[has_pos],
                self.config.temperature,
                self.config.tau_plus,
            )
        else:
            denom = torch.logsumexp(anchor_logits, dim=1)
            info_loss = (denom - numer).mean()
        loss = info_loss + reg_loss
        metrics = {
            'loss': 0.0,  # filled after alignment below
            'n_anchors': float(has_pos.sum()),
            'cross_subject': float(use_cross),
            **reg_metrics,
        }
        # Alignment term: pull positive pairs together, the other half of alignment + uniformity.
        if self.config.alignment_weight > 0.0:
            align_loss = alignment_penalty(center_flat, context_flat, pos_mask)
            loss = loss + self.config.alignment_weight * align_loss
            metrics['alignment_loss'] = float(align_loss.detach())
        metrics['loss'] = float(loss.detach())
        return loss, metrics
