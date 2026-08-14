"""CPC objective: autoregressively predict future word latents (wav2vec/BENDR)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from zte.config import ObjectiveConfig
from zte.models.embedding import ZTEModel
from zte.models.heads import ProjectionHead
from zte.models.objectives.base import _ObjectiveBase, _usable_mask


class CPCObjective(_ObjectiveBase):
    """Contrastive predictive coding: predict future word latents (wav2vec/BENDR).

    Attributes:
        target_head (ProjectionHead): Projects token hiddens to the target latent space.
        predictors (nn.ModuleList): One linear predictor per future step `k`.
    """

    def __init__(self, config: ObjectiveConfig, model: ZTEModel) -> None:
        """Initialises the CPC objective.

        Args:
            config (ObjectiveConfig): Objective configuration (uses `cpc_steps`, `temperature`).
            model (ZTEModel): The encoder, used to size the heads.
        """
        super().__init__(config, model)
        self.target_head = ProjectionHead(model.hidden_dim, model.config.projection_hidden, model.embed_dim)
        self.predictors = nn.ModuleList(nn.Linear(model.embed_dim, model.embed_dim) for _ in range(config.cpc_steps))

    def compute(self, model: ZTEModel, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the multi-step CPC InfoNCE loss for a batch.

        Args:
            model (ZTEModel): The ZTE encoder.
            batch (dict[str, Any]): A collated batch dict.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: `(loss, metrics)`.
        """
        # Targets are per-token; the context is causal, with omitted tokens out of its keys/values.
        hidden = model.token_hidden(batch)
        targets = F.normalize(self.target_head(hidden), dim=-1)  # (batch_size, seq_len, embed_dim)
        context_ctx = model.contextualize(hidden, model.pooling_mask(batch), causal=True)
        context_emb = model.project(context_ctx)  # (batch_size, seq_len, embed_dim), for VICReg
        context = F.normalize(context_emb, dim=-1)
        b, length, e = targets.shape
        usable = _usable_mask(batch)

        # Every usable token in the batch is a candidate for every prediction.
        pool = targets.reshape(b * length, e)
        pool_valid = usable.reshape(-1)
        pool = pool.masked_fill(~pool_valid.unsqueeze(-1), 0.0)

        # One InfoNCE term per future step k, averaged over the steps that had anchors.
        total = context.new_zeros(())
        n_terms = 0
        correct = 0.0
        n_anchor = 0.0
        for k in range(1, self.config.cpc_steps + 1):
            if k >= length:
                break
            anchor_valid = usable[:, :-k] & usable[:, k:]
            if not bool(anchor_valid.any()):
                continue
            pred = self.predictors[k - 1](context[:, :-k])  # (batch_size, seq_len - k, embed_dim)
            pred = F.normalize(pred, dim=-1)[anchor_valid]  # (n_anchors, embed_dim)
            tgt_index = (
                torch.arange(b, device=context.device).repeat_interleave(length - k) * length
                + (torch.arange(k, length, device=context.device).repeat(b))
            ).reshape(b, length - k)[anchor_valid]
            logits = pred @ pool.t() / self.config.temperature
            neg_inf = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~pool_valid.unsqueeze(0), neg_inf)
            loss_k = F.cross_entropy(logits, tgt_index)
            total = total + loss_k
            n_terms += 1
            correct += float((logits.argmax(dim=1) == tgt_index).float().mean())
            n_anchor += float(anchor_valid.sum())

        reg_loss, reg_metrics = self.regularize(batch, hidden, context_emb, usable)
        if n_terms == 0:
            zero = hidden.sum() * 0.0 + reg_loss
            return zero, {'loss': float(reg_loss.detach()), 'n_anchors': 0.0, **reg_metrics}
        loss = total / n_terms + reg_loss
        return loss, {
            'loss': float(loss.detach()),
            'top1': correct / n_terms,
            'n_anchors': n_anchor,
            **reg_metrics,
        }
