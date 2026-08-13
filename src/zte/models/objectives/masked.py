"""Masked objective: predict an EMA-teacher latent (data2vec) or raw features (MAEEG)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from zte.config import ObjectiveConfig
from zte.models.embedding import ZTEModel
from zte.models.heads import EMATeacher, Predictor
from zte.models.objectives.base import _ObjectiveBase, _usable_mask


class MaskedObjective(_ObjectiveBase):
    """Masked word-EEG modelling: data2vec latent prediction or reconstruction.

    Both variants predict through the exported projection head so it receives gradient. The data2vec teacher target is normalised across
    tokens with a variance floor, which stops teacher and student co-collapsing onto a constant, and the EMA decay is ramped.

    Attributes:
        teacher (EMATeacher | None): EMA teacher (latent target) or `None` (reconstruction).
    """

    needs_teacher = True

    def __init__(self, config: ObjectiveConfig, model: ZTEModel, feature_dim: int | None) -> None:
        """Initialises the masked objective and its prediction machinery.

        Args:
            config (ObjectiveConfig): Objective configuration (uses `mask_ratio`, `masked_target`, `ema_decay`,
                `ema_decay_end`, `teacher_variance_floor`).
            model (ZTEModel): The encoder (also cloned into the EMA teacher).
            feature_dim (int | None): Reconstruct-target dimension -- `n_features` (flattened band power) for the band-power frontend or
                `n_channels * time_steps` for the raw frontend. Only used when `masked_target='reconstruct'`.
        """
        super().__init__(config, model)
        self.mask_token = nn.Parameter(torch.zeros(model.hidden_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Prediction/reconstruction both operate on the exported embedding so `model.project` trains.
        if config.masked_target == 'latent':
            self.predictor = Predictor(model.embed_dim)
            self.teacher: EMATeacher | None = EMATeacher(model, decay=config.ema_decay)
            self.recon_head: nn.Module | None = None
        else:
            self.predictor = Predictor(model.embed_dim)
            self.teacher = None
            dim = feature_dim if feature_dim is not None else model.embed_dim
            self.recon_head = nn.Linear(model.embed_dim, dim)

    def compute(
        self, model: ZTEModel, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the masked-modelling loss for a batch.

        Args:
            model (ZTEModel): The ZTE encoder.
            batch (dict[str, Any]): A collated batch dict.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: `(loss, metrics)`.
        """
        # Draw the mask over usable tokens.
        usable = _usable_mask(batch)  # (batch_size, seq_len)
        hidden = model.token_hidden(batch)  # (batch_size, seq_len, hidden_dim)
        rand = torch.rand_like(usable, dtype=torch.float32)
        mask = usable & (rand < self.config.mask_ratio)
        if not bool(mask.any()):  # guarantee at least one masked token
            flat = usable.reshape(-1).nonzero(as_tuple=False)
            if flat.numel() == 0:
                return hidden.sum() * 0.0, {'loss': 0.0, 'n_masked': 0.0}
            mask = mask.clone()
            mask.view(-1)[flat[0]] = True

        # Encode the corrupted sequence through the exported projection.
        key_mask = model.pooling_mask(batch)  # exclude omitted tokens from attention
        student_in = torch.where(mask.unsqueeze(-1), self.mask_token.to(hidden.dtype), hidden)
        student_ctx = model.contextualize(student_in, key_mask)
        student_emb = model.project(
            student_ctx
        )  # (batch_size, seq_len, embed_dim) -- trains project

        # Score the masked positions against the teacher latent or the token's own input.
        if self.config.masked_target == 'latent' and self.teacher is not None:
            pred = self.predictor(student_emb)[mask]
            with torch.no_grad():
                t_hidden = self.teacher.module.token_hidden(batch)
                t_ctx = self.teacher.module.contextualize(t_hidden, key_mask)
                t_emb = self.teacher.module.project(t_ctx)[mask]
            target = self._normalize_across_tokens(t_emb)
            loss = F.smooth_l1_loss(pred, target)
        else:
            assert self.recon_head is not None
            recon = self.recon_head(student_emb)[mask]
            if not model.uses_raw and batch.get('features') is not None:
                target = batch['features'][mask]
            else:
                target = batch['raw'][mask].reshape(int(mask.sum()), -1)
            loss = F.mse_loss(recon, target)

        reg_loss, reg_metrics = self.regularize(batch, hidden, student_emb, usable)
        loss = loss + reg_loss
        return loss, {'loss': float(loss.detach()), 'n_masked': float(mask.sum()), **reg_metrics}

    def _normalize_across_tokens(self, target: torch.Tensor) -> torch.Tensor:
        """Normalises a data2vec target across the token batch, flooring the per-dim std.

        A per-token LayerNorm would leave between-token variance unconstrained, letting teacher and student co-collapse onto a constant.

        Args:
            target (torch.Tensor): Teacher latents `(n_masked, embed_dim)`.

        Returns:
            torch.Tensor: The across-token-normalised target (same shape).
        """
        if target.shape[0] < 2:
            return target
        mean = target.mean(dim=0, keepdim=True)
        std = target.std(dim=0, unbiased=False, keepdim=True).clamp_min(
            self.config.teacher_variance_floor
        )
        return (target - mean) / std

    def post_step(
        self, model: ZTEModel, step: int | None = None, total_steps: int | None = None
    ) -> None:
        """Updates the EMA teacher after each optimiser step (latent target only).

        The decay ramps linearly from `config.ema_decay` to `config.ema_decay_end`: a fast teacher early gives signal, a slow one stabilises late.

        Args:
            model (ZTEModel): The student encoder.
            step (int | None): Current global optimiser step (for the ramp).
            total_steps (int | None): Total optimiser steps (for the ramp).
        """
        if self.teacher is None:
            return
        decay: float | None = None
        if (
            step is not None
            and total_steps is not None
            and total_steps > 1
            and self.config.ema_decay_end != self.config.ema_decay
        ):
            frac = min(1.0, step / (total_steps - 1))
            decay = (
                self.config.ema_decay + (self.config.ema_decay_end - self.config.ema_decay) * frac
            )
        self.teacher.update(model, decay=decay)
