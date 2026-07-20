"""CBOW objective: predict a word embedding from its averaged neighbours."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from zte.config import ObjectiveConfig
from zte.models.embedding import ZTEModel
from zte.models.heads import ProjectionHead
from zte.models.objectives.base import _ObjectiveBase, _usable_mask


class CBOWObjective(_ObjectiveBase):
    """Continuous bag-of-words: predict a word from its neighbours' EEG.

    Attributes:
        context_head (ProjectionHead): Projection for the neighbour ("context") embeddings.
    """

    def __init__(self, config: ObjectiveConfig, model: ZTEModel) -> None:
        """Initialises the CBOW objective.

        Args:
            config (ObjectiveConfig): Objective configuration.
            model (ZTEModel): The encoder, used to size the context head.
        """
        super().__init__(config, model)
        self.context_head = ProjectionHead(
            model.hidden_dim, model.config.projection_hidden, model.embed_dim
        )

    def compute(
        self, model: ZTEModel, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the CBOW InfoNCE loss for a batch.

        Args:
            model (ZTEModel): The ZTE encoder.
            batch (dict[str, Any]): A collated batch dict.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: `(loss, metrics)`.
        """
        hidden = model.token_hidden(batch)
        emb_raw = model.project(hidden)
        center = F.normalize(emb_raw, dim=-1)
        context = self.context_head(hidden)
        b, length, _ = center.shape

        center_flat = center.reshape(b * length, -1)
        context_flat = context.reshape(b * length, -1)
        usable = _usable_mask(batch)
        usable_flat = usable.reshape(-1)

        # Neighbours are within-sentence, within-window, non-self and usable.
        sent_id = torch.arange(b, device=center.device).repeat_interleave(length)
        pos = torch.arange(length, device=center.device).repeat(b)
        same_sent = sent_id[:, None] == sent_id[None, :]
        within = (pos[:, None] - pos[None, :]).abs() <= self.config.context_window
        not_self = ~torch.eye(b * length, dtype=torch.bool, device=center.device)
        neigh = same_sent & within & not_self & usable_flat[None, :]

        counts = neigh.sum(dim=1, keepdim=True).clamp_min(1).float()
        ctx_repr = (neigh.float() @ context_flat) / counts  # (n_tokens, embed_dim)
        ctx_repr = F.normalize(ctx_repr, dim=-1)

        anchors = usable_flat & (neigh.sum(dim=1) > 0)
        reg_loss, reg_metrics = self.regularize(batch, hidden, emb_raw, usable)
        if not bool(anchors.any()):
            zero = center_flat.sum() * 0.0 + reg_loss
            return zero, {'loss': float(reg_loss.detach()), 'n_anchors': 0.0, **reg_metrics}

        # Each anchor's averaged context must pick its own centre embedding out of the batch.
        logits = ctx_repr[anchors] @ center_flat.t() / self.config.temperature
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~usable_flat[None, :], neg_inf)
        targets = torch.nonzero(anchors, as_tuple=False).squeeze(1)
        loss = F.cross_entropy(logits, targets) + reg_loss
        acc = (logits.argmax(dim=1) == targets).float().mean()
        return loss, {
            'loss': float(loss.detach()),
            'top1': float(acc),
            'n_anchors': float(anchors.sum()),
            **reg_metrics,
        }
