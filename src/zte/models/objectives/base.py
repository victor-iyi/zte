"""Shared objective base: VICReg anti-collapse, the adversaries and the auxiliary heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from zte.config import ObjectiveConfig
from zte.models.embedding import ZTEModel
from zte.models.heads import SubjectAdversary
from zte.models.objectives.losses import identity_orthogonality, vicreg_terms


def _usable_mask(batch: dict[str, Any]) -> torch.Tensor:
    """Returns the `(batch_size, seq_len)` mask of real, fixated tokens -- the anti-leakage gate."""
    return batch['pad_mask'] & batch['presence']


def _context_key_mask(batch: dict[str, Any]) -> torch.Tensor:
    """Returns the attention key mask for contextual objectives (`True` = attend).

    Omitted words are excluded from the keys/values so their zero-imputed states cannot leak into the contextual
    representation. A sentence with no present tokens falls back to the padding mask, since a fully-masked row yields NaNs.
    """
    valid = batch['pad_mask'] & batch['presence']
    empty = ~valid.any(dim=1)
    if bool(empty.any()):
        valid = valid.clone()
        valid[empty] = batch['pad_mask'][empty]
    return valid


class _ObjectiveBase(nn.Module):  # pylint: disable=abstract-method
    """Shared base wiring the anti-collapse (VICReg) and subject-adversary regularisers.

    Subclasses compute their main self-supervised loss and call `regularize` with the token hiddens (for the adversary)
    and the exported embeddings (for VICReg); the returned extra loss is added to the objective's loss and its metrics merged in.

    Attributes:
        config (ObjectiveConfig): The objective configuration.
        subject_adversary (SubjectAdversary | None): The gradient-reversal subject classifier, or `None`.
    """

    needs_teacher = False

    def __init__(self, config: ObjectiveConfig, model: ZTEModel) -> None:
        """Builds shared regularisers.

        Args:
            config (ObjectiveConfig): Objective configuration.
            model (ZTEModel): The encoder (used to size the adversary head and read `n_subjects`).
        """
        super().__init__()
        self.config = config
        self._n_subjects = model.config.n_subjects

        # When factored, the adversary acts on the content subspace, driving identity out of content.
        self._factored = bool(model.config.factored)
        self._content_dim = model.config.content_dim if self._factored else model.embed_dim
        adv_in = self._content_dim if self._factored else model.hidden_dim
        self.subject_adversary: SubjectAdversary | None = (
            SubjectAdversary(adv_in, model.config.n_subjects)
            if config.subject_adversary_weight > 0.0
            else None
        )
        # A second referee predicting which passage/task a token came from, blocking the stimulus shortcut.
        self._n_tasks = getattr(model.config, 'n_tasks', 3)
        self.stimulus_adversary: SubjectAdversary | None = (
            SubjectAdversary(model.hidden_dim, self._n_tasks)
            if config.stimulus_adversary_weight > 0.0
            else None
        )

        # Auxiliary heads, all sized in `attach_auxiliary` once the dataset targets are known.
        self.meaning_head: nn.Module | None = None
        self.register_buffer('meaning_matrix', None, persistent=False)
        self.behaviour_head: nn.Module | None = None
        self.register_buffer('_behaviour_binary', None, persistent=False)
        self.data2vec_head: nn.Module | None = None
        self.data2vec_proj: nn.Module | None = None

        # Nuisance subspace of a factored embedding; the behaviour / data2vec heads are routed here.
        self._nuisance_dim = (model.embed_dim - self._content_dim) if self._factored else 0

        # Width of a per-occurrence contextual meaning target (0 = the word-type-keyed matrix path).
        self._meaning_contextual_dim = 0

        # Training progress, set by the trainer, so the adversary's reversal strength can ramp 0 -> 1.
        self._cur_step: int | None = None
        self._cur_total: int | None = None

    def set_progress(self, step: int | None, total_steps: int | None) -> None:
        """Records the current optimiser step so the subject-adversary lambda can ramp.

        Called by the `Trainer` before each training `compute`; left at `None` during
        evaluation and for legacy callers, where the ramp is inert (`lambda_ = 1`).

        Args:
            step (int | None): Current global optimiser step.
            total_steps (int | None): Total optimiser steps in the run.
        """
        self._cur_step, self._cur_total = step, total_steps

    def _adv_lambda(self) -> float:
        """Ramped gradient-reversal strength for the subject adversary, in `[0, 1]`.

        Ramps linearly `0 -> 1` over `subject_adversary_warmup_ratio * total_steps`, then holds at 1; a cold adversary
        early lets the encoder learn content before invariance pressure is applied. Unset progress or a 0 ratio gives 1.0.
        """
        ratio = self.config.subject_adversary_warmup_ratio
        if self._cur_step is None or self._cur_total is None or ratio <= 0.0:
            return 1.0
        warmup = max(1.0, ratio * float(self._cur_total))
        return min(1.0, float(self._cur_step) / warmup)

    def attach_auxiliary(
        self,
        meaning_matrix: torch.Tensor | None = None,
        behaviour_binary: 'torch.Tensor | None' = None,
        feature_dim: int | None = None,
    ) -> None:
        """Attaches dataset-derived auxiliary targets (meaning, behaviour, and the collapse-proof regression auxiliary) after construction.

        Args:
            meaning_matrix (torch.Tensor | None): Frozen `(vocab, meaning_dim)` word vectors indexed by `batch['word_id']`;
                required when `meaning_distill_weight > 0` and the target is word-type-keyed (the per-occurrence contextual
                path carries its target in the batch instead -- see `meaning_contextual`).
            behaviour_binary (torch.Tensor | None): Bool `(n_behaviour,)` mask marking which behaviour targets are binary;
                its length sizes the behaviour head.
            feature_dim (int | None): Flattened band-power input width, used to build the frozen regression target for the data2vec
                collapse-insurance head. `None` (or a raw frontend) disables that head.
        """
        # Sized from the attached word-type matrix, else from the contextual target's width.
        if self.config.meaning_distill_weight > 0.0:
            if meaning_matrix is not None:
                self.meaning_matrix = meaning_matrix  # buffer: moves with .to(device), not trained
                self.meaning_head = nn.Linear(self._content_dim, int(meaning_matrix.shape[1]))
            elif self._meaning_contextual_dim > 0:
                self.meaning_head = nn.Linear(self._content_dim, self._meaning_contextual_dim)

        # Behaviour is meaning-adjacent, so it stays on the content subspace.
        if self.config.behaviour_weight > 0.0 and behaviour_binary is not None:
            n_beh = int(behaviour_binary.numel())
            self.behaviour_head = nn.Linear(self._content_dim, n_beh)
            self._behaviour_binary = behaviour_binary.bool()

        # The nuisance subspace regresses toward a frozen projection, which cannot co-collapse with it.
        if (
            self.config.data2vec_aux_weight > 0.0
            and self._factored
            and self._nuisance_dim > 0
            and feature_dim
        ):
            target_dim = min(64, self._nuisance_dim)
            self.data2vec_head = nn.Linear(self._nuisance_dim, target_dim)
            proj = nn.Linear(int(feature_dim), target_dim, bias=False)
            proj.requires_grad_(False)  # frozen random target projection (never trained)
            self.data2vec_proj = proj

    def _content_slice(self, emb: torch.Tensor) -> torch.Tensor:
        """Returns the content subspace of `emb` (all dims unless the model is factored)."""
        return emb[..., : self._content_dim]

    def _nuisance_slice(self, emb: torch.Tensor) -> torch.Tensor:
        """Returns the nuisance subspace `emb[..., content_dim:]` (empty when not factored)."""
        return emb[..., self._content_dim :]

    def regularize(
        self,
        batch: dict[str, Any],
        adv_hidden: torch.Tensor,
        emb: torch.Tensor,
        usable: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes VICReg + subject-adversary losses over the usable tokens.

        Args:
            batch (dict[str, Any]): The collated batch (for per-sentence `subject` ids).
            adv_hidden (torch.Tensor): Token hiddens `(batch_size, seq_len, hidden_dim)` fed to the adversary.
            emb (torch.Tensor): Exported embeddings `(batch_size, seq_len, embed_dim)` fed to VICReg.
            usable (torch.Tensor): Boolean `(batch_size, seq_len)` mask of real, present tokens.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: `(extra_loss, metrics)`.
        """
        loss = emb.new_zeros(())
        metrics: dict[str, float] = {}
        flat = usable.reshape(-1)
        if not bool(flat.any()):
            return loss, metrics
        emb_u = emb.reshape(-1, emb.shape[-1])[flat]
        vc_loss, vc_metrics = vicreg_terms(
            emb_u,
            self.config.variance_target,
            self.config.variance_weight,
            self.config.covariance_weight,
            self.config.anisotropy_weight,
        )
        loss = loss + vc_loss
        metrics.update(vc_metrics)

        need_hidden = (self.subject_adversary is not None and self._n_subjects > 1) or (
            self.stimulus_adversary is not None and batch.get('task_id') is not None
        )
        hid_u = adv_hidden.reshape(-1, adv_hidden.shape[-1])[flat] if need_hidden else None
        # Factored: the adversary sees the content subspace, else the shared token hidden.
        subj_adv_in = self._content_slice(emb_u) if self._factored else hid_u
        if self.subject_adversary is not None and self._n_subjects > 1:
            subj = batch['subject'][:, None].expand(usable.shape).reshape(-1)[flat]
            adv_lambda = self._adv_lambda()  # ramped gradient-reversal strength
            logits = self.subject_adversary(subj_adv_in, lambda_=adv_lambda)
            adv_loss = F.cross_entropy(logits, subj)
            loss = loss + self.config.subject_adversary_weight * adv_loss
            metrics['adv_loss'] = float(adv_loss.detach())
            metrics['adv_acc'] = float((logits.argmax(dim=-1) == subj).float().mean().detach())
            metrics['adv_lambda'] = adv_lambda
        # Rank-preserving identity removal: decorrelate content from the signature, don't just hide it.
        signature = batch.get('subject_signature')
        if self.config.identity_orthogonality_weight > 0.0 and signature is not None:
            sig_tok = signature[:, None, :].expand(*usable.shape, signature.shape[-1])
            sig_u = sig_tok.reshape(-1, signature.shape[-1])[flat]
            id_loss = identity_orthogonality(self._content_slice(emb_u), sig_u)
            loss = loss + self.config.identity_orthogonality_weight * id_loss
            metrics['identity_orth'] = float(id_loss.detach())

        if self.stimulus_adversary is not None and batch.get('task_id') is not None:
            task = batch['task_id'][:, None].expand(usable.shape).reshape(-1)[flat].clamp(min=0)
            t_logits = self.stimulus_adversary(hid_u, lambda_=1.0)
            stim_loss = F.cross_entropy(t_logits, task)
            loss = loss + self.config.stimulus_adversary_weight * stim_loss
            metrics['stim_adv_loss'] = float(stim_loss.detach())
            metrics['stim_adv_acc'] = float(
                (t_logits.argmax(dim=-1) == task).float().mean().detach()
            )

        # Meaning distillation: a per-occurrence contextual target is preferred (it disambiguates polysemy).
        if self.meaning_head is not None and batch.get('meaning_target') is not None:
            tgt = batch['meaning_target'].reshape(-1, batch['meaning_target'].shape[-1])[flat]
            ok = torch.isfinite(tgt).all(dim=-1)  # skip padding / uncovered rows (NaN)
            if bool(ok.any()):
                pred = self.meaning_head(self._content_slice(emb_u)[ok])
                m_loss = (
                    1.0 - F.cosine_similarity(F.normalize(pred, dim=-1), tgt[ok], dim=-1)
                ).mean()
                loss = loss + self.config.meaning_distill_weight * m_loss
                metrics['meaning_loss'] = float(m_loss.detach())
        elif (
            self.meaning_head is not None
            and self.meaning_matrix is not None
            and batch.get('word_id') is not None
        ):
            wid = batch['word_id'].reshape(-1)[flat]
            has_w = wid >= 0
            if bool(has_w.any()):
                pred = self.meaning_head(self._content_slice(emb_u)[has_w])
                target = F.embedding(wid[has_w], self.meaning_matrix)
                m_loss = (
                    1.0 - F.cosine_similarity(F.normalize(pred, dim=-1), target, dim=-1)
                ).mean()
                loss = loss + self.config.meaning_distill_weight * m_loss
                metrics['meaning_loss'] = float(m_loss.detach())

        # Reading-behaviour supervision: regress fixation difficulty from the content subspace.
        if self.behaviour_head is not None and batch.get('behaviour_target') is not None:
            beh = batch['behaviour_target'].reshape(-1, batch['behaviour_target'].shape[-1])[flat]
            pred = self.behaviour_head(self._content_slice(emb_u))
            finite = torch.isfinite(beh)
            if bool(finite.any()):
                binary = self._behaviour_binary
                reg_m = finite & ~binary[None, :] if binary is not None else finite
                bin_m = finite & binary[None, :] if binary is not None else torch.zeros_like(finite)
                b_loss = pred.new_zeros(())
                if bool(reg_m.any()):
                    b_loss = b_loss + F.mse_loss(pred[reg_m], beh[reg_m])
                if bool(bin_m.any()):
                    b_loss = b_loss + F.binary_cross_entropy_with_logits(pred[bin_m], beh[bin_m])
                loss = loss + self.config.behaviour_weight * b_loss
                metrics['behaviour_loss'] = float(b_loss.detach())

        # The nuisance subspace reconstructs the input against a frozen target that cannot co-collapse.
        if (
            self.data2vec_head is not None
            and self.data2vec_proj is not None
            and batch.get('features') is not None
        ):
            feats = batch['features'].reshape(-1, batch['features'].shape[-1])[flat]
            with torch.no_grad():
                target = F.normalize(self.data2vec_proj(feats), dim=-1)
            pred = self.data2vec_head(self._nuisance_slice(emb_u))
            d_loss = (1.0 - F.cosine_similarity(F.normalize(pred, dim=-1), target, dim=-1)).mean()
            loss = loss + self.config.data2vec_aux_weight * d_loss
            metrics['data2vec_loss'] = float(d_loss.detach())
        return loss, metrics
