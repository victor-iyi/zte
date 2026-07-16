"""Self-supervised training objectives for ZTE -- the EEG analogues of word2vec.

Four interchangeable objectives, all driven by `~zte.config.ObjectiveConfig`:

- `SkipGramObjective` -- given a word's EEG embedding, identify the EEG of its neighbours (or the same stimulus read by another subject)
    via multi-positive InfoNCE with in-batch negatives.
- `CBOWObjective` -- predict a word's embedding from the averaged embeddings of its neighbours.
- `MaskedObjective` -- mask word tokens and predict either an EMA-teacher latent (data2vec) or the raw features (MAEEG-style reconstruction).
- `CPCObjective` -- autoregressively predict future word latents (wav2vec/BENDR contrastive predictive coding).

**Every** objective treats omitted words (presence `False`) as non-tokens:
they are never anchors, positives or targets, so the zero/NaN vectors that plague word-level ZuCo modelling cannot leak into the loss.

Three cross-cutting improvements are wired through the shared `_ObjectiveBase`:

- **Anti-collapse (VICReg).** A variance-hinge + covariance penalty on the exported embeddings (`objective.variance_weight`, `objective.covariance_weight`)
    stops the InfoNCE / smooth-L1 losses from solving the task in a handful of the 768 dimensions.
- **Subject invariance (adversary).** An optional gradient-reversal subject classifier (`objective.subject_adversary_weight`)
    trains the encoder to *hide* subject identity.
- **Cross-subject positives.** Skip-gram can draw its positives from the *same stimulus read by different subjects* (`objective.cross_subject_positives`
    + a stimulus-grouped batch sampler), turning subject identity from a shortcut into a nuisance the loss must remove.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from zte.config import ObjectiveConfig
from zte.models.embedding import ZTEModel
from zte.models.heads import EMATeacher, Predictor, ProjectionHead, SubjectAdversary


def _usable_mask(batch: dict[str, Any]) -> torch.Tensor:
    """Returns `(batch_size, seq_len)` mask of tokens usable as anchors/positives/targets.

    A token is usable only if it is a real (non-padding) position *and* the word received a fixation (present). This is the anti-leakage gate.

    Args:
        batch (dict[str, Any]): A collated batch dict.

    Returns:
        Boolean tensor `(batch_size, seq_len)`.
    """
    return batch['pad_mask'] & batch['presence']


def _context_key_mask(batch: dict[str, Any]) -> torch.Tensor:
    """Returns the attention key mask for contextual objectives (`True` = attend).

    Omitted words carry no real signal, so they are excluded from the transformer's keys/values as well -- otherwise
    their zero-imputed states would leak into the contextual representation and the teacher targets. Sentences that happen to
    have *no* present tokens fall back to the padding mask so a row is never fully masked (which would produce NaNs).

    Args:
        batch (dict[str, Any]): A collated batch dict.

    Returns:
        Boolean tensor `(batch_size, seq_len)`; `True` at positions allowed as attention keys.
    """
    valid = batch['pad_mask'] & batch['presence']
    empty = ~valid.any(dim=1)
    if bool(empty.any()):
        valid = valid.clone()
        valid[empty] = batch['pad_mask'][empty]
    return valid


def vicreg_terms(
    emb: torch.Tensor,
    gamma: float,
    var_weight: float,
    cov_weight: float,
    aniso_weight: float = 0.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes the VICReg variance/covariance penalties plus an anti-cone anisotropy penalty.

    The variance term hinges each embedding dimension's batch std up toward `gamma`, so no dimension is allowed to go silent -- this
    is what prevents the ~15-of-768 dimensional collapse that otherwise appears. The covariance term drives off-diagonal feature
    covariances toward zero so the dimensions carry decorrelated information (raising effective rank). The **anisotropy** term
    penalises the squared norm of the mean L2-normalised embedding -- which equals the population mean off-diagonal cosine -- so the space
    cannot degenerate into a cone (every vector pointing the same way, the LOSO failure mode where rank looks high but no dimension separates content).

    Args:
        emb (torch.Tensor): Embeddings `(n_tokens, embed_dim)` (un-normalised).
        gamma (float): Target per-dimension standard deviation for the variance hinge.
        var_weight (float): Weight of the variance term (0 disables).
        cov_weight (float): Weight of the covariance term (0 disables).
        aniso_weight (float): Weight of the anti-cone anisotropy term (0 disables).
        eps (float): Numerical floor inside the std.

    Returns:
        tuple[torch.Tensor, dict[str, float]]: `(loss, metrics)`.
    """
    loss = emb.new_zeros(())
    metrics: dict[str, float] = {}
    n, e = emb.shape
    if n < 2 or (var_weight == 0.0 and cov_weight == 0.0 and aniso_weight == 0.0):
        return loss, metrics
    if var_weight > 0.0:
        std = torch.sqrt(emb.var(dim=0, unbiased=False) + eps)
        var_loss = torch.relu(gamma - std).mean()
        loss = loss + var_weight * var_loss
        metrics['vicreg_var'] = float(var_loss.detach())
        metrics['emb_std'] = float(std.mean().detach())
    if cov_weight > 0.0:
        z = emb - emb.mean(dim=0, keepdim=True)
        cov = (z.t() @ z) / (n - 1)
        off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        cov_loss = off_diag_sq / e
        loss = loss + cov_weight * cov_loss
        metrics['vicreg_cov'] = float(cov_loss.detach())
    if aniso_weight > 0.0 and n >= 2:
        # Anti-cone via a Wang & Isola uniformity term: spread the L2-normalised embeddings over the
        # sphere. Unlike a mean-direction penalty (which is a saddle at a perfect cone and cannot break
        # it), pairwise repulsion genuinely lowers anisotropy. Subsample for O(n^2) safety on big batches.
        unit = F.normalize(emb, dim=-1)
        if n > 1024:
            idx = torch.randperm(n, device=unit.device)[:1024]
            unit = unit[idx]
        # Pairwise squared distances from the Gram matrix (unit vectors: ||a-b||^2 = 2 - 2 a·b)
        # rather than torch.pdist, which is unimplemented on MPS (Apple Silicon) and XLA (TPU).
        # This is numerically identical to pdist(unit).pow(2) but runs on every backend.
        m = unit.shape[0]
        gram = unit @ unit.t()
        sq = (2.0 - 2.0 * gram).clamp_min(0.0)
        iu = torch.triu_indices(m, m, offset=1, device=unit.device)
        sq_dist = sq[iu[0], iu[1]]
        uniform_loss = sq_dist.mul(-2.0).exp().mean().clamp_min(1e-12).log()
        loss = loss + aniso_weight * uniform_loss
        metrics['uniformity_loss'] = float(uniform_loss.detach())
    return loss, metrics


def alignment_penalty(
    center: torch.Tensor, context: torch.Tensor, pos_mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared distance over positive pairs of L2-normalised embeddings.

    The *alignment* half of alignment + uniformity (Wang & Isola, 2020): for unit vectors, `||c_i - x_j||^2 = 2 - 2 c_i . x_j`,
    so pulling positives together directly tightens the same-word geometry that retrieval depends on. `anisotropy_weight`
    already supplies the uniformity half; this closes the theory's other side.

    Args:
        center (torch.Tensor): L2-normalised anchor embeddings `(n_tokens, d)`.
        context (torch.Tensor): L2-normalised context embeddings `(n_tokens, d)`.
        pos_mask (torch.Tensor): Boolean `(n_tokens, n_tokens)`; `True` at positive (i, j) pairs.

    Returns:
        torch.Tensor: Scalar alignment loss (0 when no positive pair exists).
    """
    if not bool(pos_mask.any()):
        return center.new_zeros(())
    sims = center @ context.t()
    return (2.0 - 2.0 * sims)[pos_mask].mean()


def debiased_infonce(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    cand_mask: torch.Tensor,
    temperature: float,
    tau_plus: float,
) -> torch.Tensor:
    """Debiased multi-positive InfoNCE (Chuang et al., 2020) -- stops punishing false negatives.

    In a word-level batch, another EEG trial of the same word sits among the "negatives" and plain InfoNCE shoves it away.
    The debiased estimator corrects the negative expectation with a class-prior `tau_plus`: `E_neg = (mean_neg - tau_plus * mean_pos) / (1 - tau_plus)`,
    floored at `exp(-1/temp)`, so the loss subtracts an estimate of the positive mass leaking into the negatives.
    Computed with a per-anchor max-shift for numerical stability (the shift cancels in the final log-ratio).

    Args:
        logits (torch.Tensor): Similarity logits `(n_anchor, n_items)` with non-candidates at `-inf`.
        pos_mask (torch.Tensor): Boolean `(n_anchor, n_items)` positive-pair mask.
        cand_mask (torch.Tensor): Boolean `(n_anchor, n_items)` candidate mask (positives + negatives).
        temperature (float): InfoNCE temperature (sets the `exp(-1/temp)` floor).
        tau_plus (float): Class-prior in `[0, 1)`.

    Returns:
        torch.Tensor: Scalar debiased loss (mean over anchors).
    """
    m = logits.max(dim=1, keepdim=True).values  # (n_anchor, 1) per-anchor max candidate logit
    shifted = torch.exp(logits - m)  # ~0 at the -inf (non-candidate) positions
    pos_m = pos_mask.to(shifted.dtype)
    neg_m = (cand_mask & ~pos_mask).to(shifted.dtype)
    pos_sum = (shifted * pos_m).sum(dim=1)
    neg_sum = (shifted * neg_m).sum(dim=1)
    pos_cnt = pos_m.sum(dim=1).clamp_min(1.0)
    neg_cnt = neg_m.sum(dim=1).clamp_min(1.0)
    e_neg = (neg_sum / neg_cnt - tau_plus * (pos_sum / pos_cnt)) / (1.0 - tau_plus)
    floor = torch.exp(logits.new_tensor(-1.0 / temperature) - m.squeeze(1))  # shifted exp(-1/temp)
    e_neg = torch.maximum(e_neg, floor)
    denom = pos_sum + neg_cnt * e_neg
    # log(denom) - log(pos_sum); the shift m cancels (both sides carry exp(-m)).
    return (torch.log(denom.clamp_min(1e-12)) - torch.log(pos_sum.clamp_min(1e-12))).mean()


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
        # When factored, the subject adversary acts on the *content subspace* of the
        # embedding (pushing identity OUT of content), not the shared hidden -- the crux of
        # disentanglement. Otherwise it acts on the token hidden, as before.
        self._factored = bool(model.config.factored)
        self._content_dim = model.config.content_dim if self._factored else model.embed_dim
        adv_in = self._content_dim if self._factored else model.hidden_dim
        self.subject_adversary: SubjectAdversary | None = (
            SubjectAdversary(adv_in, model.config.n_subjects)
            if config.subject_adversary_weight > 0.0
            else None
        )
        # A second gradient-reversal referee that predicts *which passage/task* a token came from,
        # so the encoder cannot lean on the "which of the sentence-sets" shortcut (the stimulus crutch).
        self._n_tasks = getattr(model.config, 'n_tasks', 3)
        self.stimulus_adversary: SubjectAdversary | None = (
            SubjectAdversary(model.hidden_dim, self._n_tasks)
            if config.stimulus_adversary_weight > 0.0
            else None
        )

        # Meaning-distillation head projects the (content) embedding to the frozen
        # word-vector space. It is sized in `attach_auxiliary` to the *attached matrix's* width
        # (a real LM file sets its own dim; the hash fallback uses config.meaning_dim), so the
        # projection can never mismatch the target.
        self.meaning_head: nn.Module | None = None
        self.register_buffer('meaning_matrix', None, persistent=False)
        # Reading-behaviour head (sized when the targets are attached).
        self.behaviour_head: nn.Module | None = None
        self.register_buffer('_behaviour_binary', None, persistent=False)
        # Nuisance subspace of a factored embedding (embed_dim - content_dim); 0 when not factored.
        # The behaviour / data2vec-aux heads are routed here so the otherwise-idle nuisance dims are
        # supervised instead of receiving gradient from nothing.
        self._nuisance_dim = (model.embed_dim - self._content_dim) if self._factored else 0
        # Collapse-proof regression auxiliary. A FIXED (non-trainable) random projection
        # of the token's own input features is the frozen regression target for the nuisance subspace;
        # because the target never moves it cannot co-collapse with the student (unlike a plain EMA
        # teacher), and it gives the nuisance room a job. Both sized in `attach_auxiliary`.
        self.data2vec_head: nn.Module | None = None
        self.data2vec_proj: nn.Module | None = None
        # Width of a per-occurrence contextual meaning target (0 = the word-type-keyed matrix path).
        # Set by the training pipeline before `attach_auxiliary` when `meaning_contextual` is used.
        self._meaning_contextual_dim = 0
        # Current training progress, set by the trainer before each `compute`, so the
        # subject-adversary gradient-reversal strength can ramp 0 -> 1. `None` (eval / legacy) -> full.
        self._cur_step: int | None = None
        self._cur_total: int | None = None

    def set_progress(self, step: int | None, total_steps: int | None) -> None:
        """Records the current optimiser step so the subject-adversary lambda can ramp.

        Called by the :class:`~zte.training.trainer.Trainer` before each training `compute`; left at `None` during
        evaluation and for legacy callers, where the ramp is inert (`lambda_ = 1`).

        Args:
            step (int | None): Current global optimiser step.
            total_steps (int | None): Total optimiser steps in the run.
        """
        self._cur_step, self._cur_total = step, total_steps

    def _adv_lambda(self) -> float:
        """Ramped gradient-reversal strength for the subject adversary, in `[0, 1]`.

        Ramps linearly `0 -> 1` across `subject_adversary_warmup_ratio * total_steps` optimiser steps (DANN / data2vec schedule),
        then holds at 1. Returns 1.0 when progress is unset or the ratio is 0, preserving the pre-ramp behaviour exactly.
        A cold adversary early lets the encoder learn content before invariance pressure is applied.

        Returns:
            float: Reversal strength in `[0, 1]`.
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
                path carries its target in the batch instead — see `meaning_contextual`).
            behaviour_binary (torch.Tensor | None): Bool `(n_behaviour,)` mask marking which behaviour targets are binary;
                its length sizes the behaviour head.
            feature_dim (int | None): Flattened band-power input width, used to build the frozen regression target for the data2vec
                collapse-insurance head. `None` (or a raw frontend) disables that head.
        """
        # Meaning head: sized from an attached word-type matrix, else (per-occurrence
        # contextual target carried in the batch) from the configured contextual model width.
        if self.config.meaning_distill_weight > 0.0:
            if meaning_matrix is not None:
                self.meaning_matrix = meaning_matrix  # buffer: moves with .to(device), not trained
                self.meaning_head = nn.Linear(self._content_dim, int(meaning_matrix.shape[1]))
            elif self._meaning_contextual_dim > 0:
                self.meaning_head = nn.Linear(self._content_dim, self._meaning_contextual_dim)
        if self.config.behaviour_weight > 0.0 and behaviour_binary is not None:
            # Behaviour (reading difficulty) is a meaning-adjacent, privileged target, so it stays on
            # the CONTENT subspace -- predicting processing load pulls surprisal/frequency in.
            n_beh = int(behaviour_binary.numel())
            self.behaviour_head = nn.Linear(self._content_dim, n_beh)
            self._behaviour_binary = behaviour_binary.bool()
        # Collapse-proof regression auxiliary on the NUISANCE subspace. The target is a
        # FIXED random projection of the token's own band-power input (frozen, so no teacher/student
        # co-collapse); regressing the otherwise-idle nuisance dims toward it gives them a real job.
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
        # The subject adversary sees the content subspace when factored (drive identity out
        # of content), else the shared token hidden.
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
        if self.stimulus_adversary is not None and batch.get('task_id') is not None:
            task = batch['task_id'][:, None].expand(usable.shape).reshape(-1)[flat].clamp(min=0)
            t_logits = self.stimulus_adversary(hid_u, lambda_=1.0)
            stim_loss = F.cross_entropy(t_logits, task)
            loss = loss + self.config.stimulus_adversary_weight * stim_loss
            metrics['stim_adv_loss'] = float(stim_loss.detach())
            metrics['stim_adv_acc'] = float(
                (t_logits.argmax(dim=-1) == task).float().mean().detach()
            )

        # Meaning distillation -- pull the content subspace toward the frozen word vector
        # (cosine). This is the explicit content target skip-gram never had. A per-occurrence
        # *contextual* target carried in the batch (`meaning_target`) is preferred when present, so the
        # same surface word gets a context-specific target (disambiguating polysemy); otherwise the
        # word-type-keyed static matrix is used.
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

        # Reading-behaviour supervision -- regress fixation difficulty from the content
        # subspace (privileged information); NaN cells (padding / skipped words) are masked.
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

        # Collapse-proof regression auxiliary. The nuisance subspace regresses (cosine)
        # toward a FROZEN random projection of the token's own band-power input. The target cannot move,
        # so there is no teacher/student co-collapse, and the otherwise-idle nuisance dims are put to
        # work reconstructing the input -- the factored "nuisance room" is now genuinely used.
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


class SkipGramObjective(_ObjectiveBase):
    """Multi-positive InfoNCE skip-gram over word-EEG token embeddings.

    Attributes:
        context_head (ProjectionHead): The word2vec "output" embedding projection.
    """

    def __init__(self, config: ObjectiveConfig, model: ZTEModel) -> None:
        """Initialises the skip-gram objective.

        Args:
            config (ObjectiveConfig): Objective configuration (uses `context_window`, `temperature`, `cross_subject_positives`).
            model (ZTEModel): The encoder, used to size the context head.
        """
        super().__init__(config, model)
        self.context_head = ProjectionHead(
            model.hidden_dim, model.config.projection_hidden, model.embed_dim
        )

    def compute(
        self, model: ZTEModel, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
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

        # Meaning positives: also pull together the *same content word in different sentences*
        # (subject-agnostic word identity), so the space can cluster by meaning rather than by which
        # passage a word came from -- the "chase meaning, not the stimulus shortcut" lever.
        if self.config.meaning_positives and batch.get('word_id') is not None:
            wid = batch['word_id'].reshape(-1)
            has_w = wid >= 0
            same_word = (wid[:, None] == wid[None, :]) & has_w[None, :] & has_w[:, None]
            sent_ids = torch.arange(b, device=center.device).repeat_interleave(length)
            diff_sent = sent_ids[:, None] != sent_ids[None, :]
            pos_mask = pos_mask | (same_word & diff_sent & not_self & valid_pair)

        logits = center_flat @ context_flat.t() / self.config.temperature
        neg_inf = torch.finfo(logits.dtype).min
        cand_mask = usable_flat[None, :] & not_self
        # Confound-matched hard negatives -- keep only negatives that SHARE the anchor's
        # confound (same subject & task), so the softmax cannot be won by reading identity/task off
        # the negatives; positives stay eligible regardless. De-confounds the contrastive objective.
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
        # Alignment term -- pull positive pairs together (the missing half of
        # alignment + uniformity). Uses the already-L2-normalised center/context and the final pos_mask.
        if self.config.alignment_weight > 0.0:
            align_loss = alignment_penalty(center_flat, context_flat, pos_mask)
            loss = loss + self.config.alignment_weight * align_loss
            metrics['alignment_loss'] = float(align_loss.detach())
        metrics['loss'] = float(loss.detach())
        return loss, metrics


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


class MaskedObjective(_ObjectiveBase):
    """Masked word-EEG modelling: data2vec latent prediction or reconstruction.

    Both variants predict/reconstruct *through the exported 768-d projection head*, so that head is
    actually trained (otherwise it receives no gradient at all). The data2vec teacher target is
    normalised **across tokens** with a variance floor, which is what stops the teacher/student
    co-collapsing onto a constant, and the teacher EMA decay is **ramped** across training.

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

        key_mask = _context_key_mask(batch)  # exclude omitted tokens from attention
        student_in = torch.where(mask.unsqueeze(-1), self.mask_token.to(hidden.dtype), hidden)
        student_ctx = model.contextualize(student_in, key_mask)
        student_emb = model.project(
            student_ctx
        )  # (batch_size, seq_len, embed_dim) -- trains project

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
            # Reconstruct the token's own input: band-power features for the MLP
            # frontend, or the flattened raw window for the Conformer frontend.
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

        A per-token LayerNorm (the previous behaviour) leaves *between-token* variance unconstrained,
        which is exactly what lets the teacher and student co-collapse onto a constant. Normalising
        across tokens with a variance floor keeps the target spread out.

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

        The decay is ramped linearly from `config.ema_decay` to `config.ema_decay_end` across training
        (data2vec schedule): a fast-moving teacher early gives more signal, a slow one late stabilises it.

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
        targets = F.normalize(self.target_head(hidden), dim=-1)  # (batch_size, seq_len, embed_dim)
        # Exclude omitted tokens from the causal context's keys/values.
        context_ctx = model.contextualize(hidden, _context_key_mask(batch), causal=True)
        context_emb = model.project(context_ctx)  # (batch_size, seq_len, embed_dim), for VICReg
        context = F.normalize(context_emb, dim=-1)
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


def _clip_direction(logits: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """One direction of a multi-positive InfoNCE over a `(B, B)` similarity matrix.

    Rows are anchors, columns candidates. Candidates whose item is invalid (no text target) are masked
    out of the denominator; an anchor's positives are the columns sharing its sentence text (so the same
    sentence read by different subjects counts as a positive, which drives subject-invariance for free).

    Args:
        logits (torch.Tensor): Scaled cosine logits `(B, B)`.
        pos (torch.Tensor): Boolean `(B, B)` positive mask.
        valid (torch.Tensor): Boolean `(B,)`; `True` for items with a text target.

    Returns:
        torch.Tensor: Scalar mean loss over valid anchors (0 when none).
    """
    neg_inf = torch.finfo(logits.dtype).min
    cand = valid[None, :]
    masked = logits.masked_fill(~cand, neg_inf)
    denom = torch.logsumexp(masked, dim=1)
    numer = torch.logsumexp(masked.masked_fill(~(pos & cand), neg_inf), dim=1)
    per = denom - numer
    return per[valid].mean() if bool(valid.any()) else logits.new_zeros(())


class SentenceClipObjective(_ObjectiveBase):
    """Symmetric sentence-level CLIP alignment between EEG and a frozen text encoder.

    The pivot from self-supervised structural mapping to *explicit semantic alignment*: each sentence's
    word-EEG tokens are pooled into one sentence vector (contextual attention pooling), projected to the
    text space, and aligned to a **frozen** sentence embedding of the ground-truth text via a symmetric
    InfoNCE loss (Radford et al., 2021, CLIP; Défossez et al., 2023 for non-invasive brain signals). The
    loss is multi-positive: because the same ZuCo sentence is read by several subjects, every EEG reading
    of a text is a positive for that text, so identity is pushed out for free.

    The anti-collapse (VICReg) and invariance (subject/stimulus adversary, cross-subject positives) levers
    are kept as *auxiliaries* through the shared :meth:`_ObjectiveBase.regularize` -- CLIP supplies the
    content, VICReg/invariance keep the geometry healthy and subject-agnostic.

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
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / max(config.clip_temperature, 1e-4)))
        )

    def attach_text(self, text_matrix: torch.Tensor) -> None:
        """Attaches the frozen `(n_sentences, text_dim)` L2-normalised text-embedding matrix.

        Args:
            text_matrix (torch.Tensor): Frozen sentence embeddings indexed by `batch['sentence_text_id']`.
        """
        self.text_matrix = text_matrix  # buffer: moves with .to(device), never trained
        self.clip_head = nn.Linear(self._embed_dim, int(text_matrix.shape[1]))

    def _sentence_vectors(
        self, model: ZTEModel, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pools each sentence's word-EEG tokens into one contextual sentence embedding `(B, embed_dim)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(sentence_embeddings (B, embed_dim), token_hiddens (B, L, H))`.
        """
        valid = batch['pad_mask'] & batch.get('presence', batch['pad_mask'])
        empty = ~valid.any(dim=1)
        if bool(empty.any()):
            valid = valid.clone()
            valid[empty] = batch['pad_mask'][empty]
        hidden = model.token_hidden(batch)  # (B, L, H)
        hidden_ctx = model.contextualize(hidden, valid)  # sentence-contextual (bidirectional)
        pooled = model._pool_tokens(hidden_ctx, valid)  # (B, H)  # noqa: SLF001 — shared pooling
        return model.project(
            pooled
        ), hidden  # (B, embed_dim), plus token hiddens for VICReg/adversary

    def compute(
        self, model: ZTEModel, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
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

        text_id = batch.get('sentence_text_id')
        if self.clip_head is None or self.text_matrix is None or text_id is None:
            zero = z_sent.sum() * 0.0 + reg_loss
            return zero, {'loss': float(reg_loss.detach()), 'n_valid': 0.0, **reg_metrics}

        z_eeg = F.normalize(self.clip_head(z_sent), dim=-1)  # (B, text_dim)
        valid = text_id >= 0
        z_txt = F.embedding(
            text_id.clamp(min=0), self.text_matrix
        )  # (B, text_dim), already L2-normed
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = (z_eeg @ z_txt.t()) * scale  # (B, B): row=EEG reading, col=text
        pos = (text_id[:, None] == text_id[None, :]) & valid[:, None] & valid[None, :]

        if not bool(valid.any()):
            zero = logits.sum() * 0.0 + reg_loss
            return zero, {'loss': float(reg_loss.detach()), 'n_valid': 0.0, **reg_metrics}

        clip_loss = 0.5 * (
            _clip_direction(logits, pos, valid)  # EEG -> text
            + _clip_direction(logits.t(), pos, valid)  # text -> EEG (pos is symmetric)
        )
        loss = clip_loss + reg_loss
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
    if config.name == 'clip':
        return SentenceClipObjective(config, model)
    raise ValueError(f'Unknown objective: {config.name!r}')
