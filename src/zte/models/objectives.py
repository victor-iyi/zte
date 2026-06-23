"""Self-supervised training objectives for ZTE -- the EEG analogues of word2vec.

Four interchangeable objectives, all driven by :class:`~zte.config.ObjectiveConfig`:

* :class:`SkipGramObjective` -- given a word's EEG embedding, identify the EEG of its neighbours via multi-positive
    InfoNCE with in-batch negatives (word2vec skip-gram, but over continuous neural tokens).
* :class:`CBOWObjective` -- predict a word's embedding from the averaged embeddings of its neighbours.
* :class:`MaskedObjective` -- mask word tokens and predict either an EMA-teacher latent (data2vec) or the raw features (MAEEG-style reconstruction).
* :class:`CPCObjective` -- autoregressively predict future word latents (wav2vec/BENDR contrastive predictive coding).

**Every** objective treats omitted words (presence `False`) as non-tokens:
they are never anchors, positives or targets, so the zero/NaN vectors that plague word-level ZuCo modelling cannot leak into the loss.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from zte.config import ObjectiveConfig
from zte.models.embedding import ZTEModel
from zte.models.heads import EMATeacher, Predictor, ProjectionHead


def _usable_mask(batch: dict[str, Any]) -> torch.Tensor:
    """Returns `(B, L)` mask of tokens usable as anchors/positives/targets.

    A token is usable only if it is a real (non-padding) position *and* the word
    received a fixation (present). This is the anti-leakage gate.

    Args:
        batch (dict[str, Any]): A collated batch dict.

    Returns:
        Boolean tensor `(B, L)`.
    """
    return batch['pad_mask'] & batch['presence']


def _context_key_mask(batch: dict[str, Any]) -> torch.Tensor:
    """Returns the attention key mask for contextual objectives (`True` = attend).

    Omitted words carry no real signal, so they are excluded from the transformer's
    keys/values as well -- otherwise their zero-imputed states would leak into the
    contextual representation and the teacher targets. Sentences that happen to
    have *no* present tokens fall back to the padding mask so a row is never fully
    masked (which would produce NaNs).

    Args:
        batch (dict[str, Any]): A collated batch dict.

    Returns:
        Boolean tensor `(B, L)`; `True` at positions allowed as attention keys.
    """
    valid = batch['pad_mask'] & batch['presence']
    empty = ~valid.any(dim=1)
    if bool(empty.any()):
        valid = valid.clone()
        valid[empty] = batch['pad_mask'][empty]
    return valid


class SkipGramObjective(nn.Module):  # pylint: disable=abstract-method
    """Multi-positive InfoNCE skip-gram over word-EEG token embeddings.

    Attributes:
        config (ObjectiveConfig): The objective configuration.
        context_head (ProjectionHead): The word2vec "output" embedding projection.
    """

    needs_teacher = False

    def __init__(self, config: ObjectiveConfig, model: ZTEModel) -> None:
        """Initialises the skip-gram objective.

        Args:
            config (ObjectiveConfig): Objective configuration (uses `context_window`, `temperature`).
            model (ZTEModel): The encoder, used to size the context head.
        """
        super().__init__()
        self.config = config
        self.context_head = ProjectionHead(
            model.hidden_dim, model.config.projection_hidden, model.embed_dim
        )

    def compute(
        self, model: ZTEModel, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the skip-gram InfoNCE loss for a batch.

        Args:
            model (ZTEModel): The ZTE encoder.
            batch (dict[str, Any]): A collated batch dict.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: `(loss, metrics)` where metrics include `loss` and `n_anchors`.

        """
        hidden = model.token_hidden(batch)  # (B, L, Hd), non-contextual
        center = F.normalize(model.project(hidden), dim=-1)
        context = F.normalize(self.context_head(hidden), dim=-1)
        b, length, _ = center.shape

        center_flat = center.reshape(b * length, -1)
        context_flat = context.reshape(b * length, -1)
        usable = _usable_mask(batch).reshape(-1)  # (M,)

        sent_id = torch.arange(b, device=center.device).repeat_interleave(length)
        pos = torch.arange(length, device=center.device).repeat(b)
        same_sent = sent_id[:, None] == sent_id[None, :]
        within = (pos[:, None] - pos[None, :]).abs() <= self.config.context_window
        not_self = ~torch.eye(b * length, dtype=torch.bool, device=center.device)
        valid_pair = usable[:, None] & usable[None, :]
        pos_mask = same_sent & within & not_self & valid_pair

        logits = center_flat @ context_flat.t() / self.config.temperature
        neg_inf = torch.finfo(logits.dtype).min
        cand_mask = usable[None, :] & not_self
        logits = logits.masked_fill(~cand_mask, neg_inf)

        has_pos = pos_mask.any(dim=1) & usable
        if not bool(has_pos.any()):
            zero = center_flat.sum() * 0.0
            return zero, {'loss': 0.0, 'n_anchors': 0.0}

        denom = torch.logsumexp(logits[has_pos], dim=1)
        pos_logits = logits[has_pos].masked_fill(~pos_mask[has_pos], neg_inf)
        numer = torch.logsumexp(pos_logits, dim=1)
        loss = (denom - numer).mean()
        return loss, {'loss': float(loss.detach()), 'n_anchors': float(has_pos.sum())}


class CBOWObjective(nn.Module):  # pylint: disable=abstract-method
    """Continuous bag-of-words: predict a word from its neighbours' EEG.

    Attributes:
        config (ObjectiveConfig): The objective configuration.
        context_head (ProjectionHead): Projection for the neighbour ("context") embeddings.

    """

    needs_teacher = False

    def __init__(self, config: ObjectiveConfig, model: ZTEModel) -> None:
        """Initialises the CBOW objective.

        Args:
            config (ObjectiveConfig): Objective configuration.
            model (ZTEModel): The encoder, used to size the context head.

        """
        super().__init__()
        self.config = config
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
        center = F.normalize(model.project(hidden), dim=-1)
        context = self.context_head(hidden)
        b, length, _ = center.shape

        center_flat = center.reshape(b * length, -1)
        context_flat = context.reshape(b * length, -1)
        usable = _usable_mask(batch).reshape(-1)

        sent_id = torch.arange(b, device=center.device).repeat_interleave(length)
        pos = torch.arange(length, device=center.device).repeat(b)
        same_sent = sent_id[:, None] == sent_id[None, :]
        within = (pos[:, None] - pos[None, :]).abs() <= self.config.context_window
        not_self = ~torch.eye(b * length, dtype=torch.bool, device=center.device)
        neigh = same_sent & within & not_self & usable[None, :]

        counts = neigh.sum(dim=1, keepdim=True).clamp_min(1).float()
        ctx_repr = (neigh.float() @ context_flat) / counts  # (M, E)
        ctx_repr = F.normalize(ctx_repr, dim=-1)

        anchors = usable & (neigh.sum(dim=1) > 0)
        if not bool(anchors.any()):
            zero = center_flat.sum() * 0.0
            return zero, {'loss': 0.0, 'n_anchors': 0.0}

        logits = ctx_repr[anchors] @ center_flat.t() / self.config.temperature
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~usable[None, :], neg_inf)
        targets = torch.nonzero(anchors, as_tuple=False).squeeze(1)
        loss = F.cross_entropy(logits, targets)
        acc = (logits.argmax(dim=1) == targets).float().mean()
        return loss, {
            'loss': float(loss.detach()),
            'top1': float(acc),
            'n_anchors': float(anchors.sum()),
        }


class MaskedObjective(nn.Module):  # pylint: disable=abstract-method
    """Masked word-EEG modelling: data2vec latent prediction or reconstruction.

    Attributes:
        config (ObjectiveConfig): The objective configuration.
        teacher (EMATeacher | None): EMA teacher (latent target) or `None` (reconstruction).
    """

    needs_teacher = True

    def __init__(self, config: ObjectiveConfig, model: ZTEModel, feature_dim: int | None) -> None:  # pylint: disable=unused-argument
        """Initialises the masked objective and its prediction machinery.

        Args:
            config (ObjectiveConfig): Objective configuration (uses `mask_ratio`,
                `masked_target`, `ema_decay`).
            model (ZTEModel): The encoder (also cloned into the EMA teacher).
            feature_dim (int | None): Reconstruct-target dimension -- `F*C` for the
                band-power frontend or `C*T` for the raw frontend. Only used when
                `masked_target='reconstruct'`.
        """
        super().__init__()
        self.config = config
        self.mask_token = nn.Parameter(torch.zeros(model.hidden_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        if config.masked_target == 'latent':
            self.predictor = Predictor(model.hidden_dim)
            self.teacher: EMATeacher | None = EMATeacher(model, decay=config.ema_decay)
            self.recon_head: nn.Module | None = None
        else:
            self.predictor = Predictor(model.hidden_dim)
            self.teacher = None
            dim = feature_dim if feature_dim is not None else model.embed_dim
            self.recon_head = nn.Linear(model.hidden_dim, dim)

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
        usable = _usable_mask(batch)  # (B, L)
        hidden = model.token_hidden(batch)  # (B, L, Hd)
        rand = torch.rand_like(usable, dtype=torch.float32)
        mask = usable & (rand < self.config.mask_ratio)
        if not bool(mask.any()):  # guarantee at least one masked token
            flat = usable.reshape(-1).nonzero(as_tuple=False)
            if flat.numel() == 0:
                return hidden.sum() * 0.0, {'loss': 0.0, 'n_masked': 0.0}
            mask = mask.clone()
            mask.view(-1)[flat[0]] = True

        key_mask = _context_key_mask(batch)  # exclude omitted tokens from attention
        student_in = torch.where(mask.unsqueeze(-1), self.mask_token.to(hidden.dtype), hidden)
        student_ctx = model.contextualize(student_in, key_mask)
        pred = self.predictor(student_ctx)[mask]

        if self.config.masked_target == 'latent' and self.teacher is not None:
            with torch.no_grad():
                t_hidden = self.teacher.module.token_hidden(batch)
                target = self.teacher.module.contextualize(t_hidden, key_mask)[mask]
            target = F.layer_norm(target, (target.shape[-1],))
            loss = F.smooth_l1_loss(pred, target)
        else:
            assert self.recon_head is not None
            recon = self.recon_head(student_ctx)[mask]
            # Reconstruct the token's own input: band-power features for the MLP
            # frontend, or the flattened raw window for the Conformer frontend.
            if not model.uses_raw and batch.get('features') is not None:
                target = batch['features'][mask]
            else:
                target = batch['raw'][mask].reshape(int(mask.sum()), -1)
            loss = F.mse_loss(recon, target)
        return loss, {'loss': float(loss.detach()), 'n_masked': float(mask.sum())}

    def post_step(self, model: ZTEModel) -> None:
        """Updates the EMA teacher after each optimiser step (latent target only).

        Args:
            model (ZTEModel): The student encoder.
        """
        if self.teacher is not None:
            self.teacher.update(model)


class CPCObjective(nn.Module):  # pylint: disable=abstract-method
    """Contrastive predictive coding: predict future word latents (wav2vec/BENDR).

    Attributes:
        config (ObjectiveConfig): The objective configuration.
        target_head (ProjectionHead): Projects token hiddens to the target latent space.
        predictors (nn.ModuleList): One linear predictor per future step `k`.
    """

    needs_teacher = False

    def __init__(self, config: ObjectiveConfig, model: ZTEModel) -> None:
        """Initialises the CPC objective.

        Args:
            config (ObjectiveConfig): Objective configuration (uses `cpc_steps`, `temperature`).
            model (ZTEModel): The encoder, used to size the heads.
        """
        super().__init__()
        self.config = config
        self.target_head = ProjectionHead(
            model.hidden_dim, model.config.projection_hidden, model.embed_dim
        )
        self.predictors = nn.ModuleList(
            nn.Linear(model.embed_dim, model.embed_dim) for _ in range(config.cpc_steps)
        )

    def compute(
        self, model: ZTEModel, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the multi-step CPC InfoNCE loss for a batch.

        Args:
            model (ZTEModel): The ZTE encoder.
            batch (dict[str, Any]): A collated batch dict.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: `(loss, metrics)`.

        """
        hidden = model.token_hidden(batch)
        targets = F.normalize(self.target_head(hidden), dim=-1)  # (B, L, E)
        # Exclude omitted tokens from the causal context's keys/values.
        context = model.contextualize(hidden, _context_key_mask(batch), causal=True)
        context = F.normalize(model.project(context), dim=-1)  # (B, L, E)
        b, length, e = targets.shape
        usable = _usable_mask(batch)

        pool = targets.reshape(b * length, e)
        pool_valid = usable.reshape(-1)
        pool = pool.masked_fill(~pool_valid.unsqueeze(-1), 0.0)

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
            pred = self.predictors[k - 1](context[:, :-k])  # (B, L-k, E)
            pred = F.normalize(pred, dim=-1)[anchor_valid]  # (A, E)
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

        if n_terms == 0:
            return hidden.sum() * 0.0, {'loss': 0.0, 'n_anchors': 0.0}
        loss = total / n_terms
        return loss, {
            'loss': float(loss.detach()),
            'top1': correct / n_terms,
            'n_anchors': n_anchor,
        }


def build_objective(
    config: ObjectiveConfig, model: ZTEModel, feature_dim: int | None = None
) -> nn.Module:
    """Constructs the objective module selected by `config.name`.

    Args:
        config (ObjectiveConfig): Objective configuration.
        model (ZTEModel): The ZTE encoder the objective wraps.
        feature_dim (int | None): Band-power feature dimension (used by masked reconstruction).

    Returns:
        An objective `nn.Module` exposing `compute(model, batch)` and the `needs_teacher` flag (and `post_step` when applicable).

    Raises:
        ValueError: If `config.name` is unknown.

    """
    if config.name == 'skipgram':
        return SkipGramObjective(config, model)
    if config.name == 'cbow':
        return CBOWObjective(config, model)
    if config.name == 'masked':
        return MaskedObjective(config, model, feature_dim)
    if config.name == 'cpc':
        return CPCObjective(config, model)
    raise ValueError(f'Unknown objective: {config.name!r}')
