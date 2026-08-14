"""Shared loss terms used across objectives: VICReg, alignment and debiased InfoNCE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def vicreg_terms(
    emb: torch.Tensor,
    gamma: float,
    var_weight: float,
    cov_weight: float,
    aniso_weight: float = 0.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes the VICReg variance/covariance penalties plus an anti-cone anisotropy penalty.

    The variance term hinges each dimension's batch std up toward `gamma` so no dimension goes silent; the covariance
    term decorrelates dimensions to raise effective rank; the anisotropy term spreads the L2-normalised embeddings over
    the sphere so the space cannot degenerate into a cone.

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
    # Variance: hinge every per-dimension std up to `gamma`.
    if var_weight > 0.0:
        std = torch.sqrt(emb.var(dim=0, unbiased=False) + eps)
        var_loss = torch.relu(gamma - std).mean()
        loss = loss + var_weight * var_loss
        metrics['vicreg_var'] = float(var_loss.detach())
        metrics['emb_std'] = float(std.mean().detach())

    # Covariance: push the off-diagonal feature covariances to zero.
    if cov_weight > 0.0:
        z = emb - emb.mean(dim=0, keepdim=True)
        cov = (z.t() @ z) / (n - 1)
        off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        cov_loss = off_diag_sq / e
        loss = loss + cov_weight * cov_loss
        metrics['vicreg_cov'] = float(cov_loss.detach())

    if aniso_weight > 0.0 and n >= 2:
        # Uniformity term: pairwise repulsion on the sphere, subsampled to bound the O(n^2) cost.
        unit = F.normalize(emb, dim=-1)
        if n > 1024:
            idx = torch.randperm(n, device=unit.device)[:1024]
            unit = unit[idx]

        # Squared distances from the Gram matrix (`||a-b||^2 = 2 - 2 a.b`); torch.pdist is unimplemented on MPS and XLA.
        m = unit.shape[0]
        gram = unit @ unit.t()
        sq = (2.0 - 2.0 * gram).clamp_min(0.0)
        iu = torch.triu_indices(m, m, offset=1, device=unit.device)
        sq_dist = sq[iu[0], iu[1]]
        uniform_loss = sq_dist.mul(-2.0).exp().mean().clamp_min(1e-12).log()
        loss = loss + aniso_weight * uniform_loss
        metrics['uniformity_loss'] = float(uniform_loss.detach())
    return loss, metrics


def identity_orthogonality(content: torch.Tensor, signature: torch.Tensor) -> torch.Tensor:
    """Linear dependence (CKA) between the content subspace and the subject signature, in `[0, 1]`.

    Asks only that content be uncorrelated with who produced it, so unlike a gradient-reversal adversary it cannot be
    satisfied by collapsing: a full-rank identity-free space scores zero.

    Args:
        content (torch.Tensor): Content-subspace embeddings `(n_tokens, content_dim)`.
        signature (torch.Tensor): Per-token subject signatures `(n_tokens, signature_dim)`.

    Returns:
        torch.Tensor: Scalar penalty (0 when either side is degenerate).
    """
    n = content.shape[0]
    if n < 2:
        return content.new_zeros(())

    zc = content - content.mean(dim=0, keepdim=True)
    zs = signature - signature.mean(dim=0, keepdim=True)

    # Normalising by each side's own self-covariance makes the term scale-free, so shrinking earns no credit.
    cross = (zc.t() @ zs).pow(2).sum()
    denom = (zc.t() @ zc).pow(2).sum().sqrt() * (zs.t() @ zs).pow(2).sum().sqrt()
    return cross / denom.clamp_min(1e-8)


def alignment_penalty(center: torch.Tensor, context: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared distance over positive pairs of L2-normalised embeddings.

    The alignment half of alignment + uniformity: for unit vectors `||c_i - x_j||^2 = 2 - 2 c_i . x_j`, so pulling
    positives together tightens the same-word geometry retrieval depends on. `anisotropy_weight` supplies the uniformity
    half.

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
    """Debiased multi-positive InfoNCE -- stops punishing false negatives.

    Another EEG trial of the same word sits among the "negatives", and plain InfoNCE shoves it away. The debiased
    estimator corrects the negative expectation with a class-prior: `E_neg = (mean_neg - tau_plus * mean_pos) / (1 -
    tau_plus)`, floored at `exp(-1/temp)`. A per-anchor max-shift keeps the exponentials in range and cancels in the
    final log-ratio.

    Args:
        logits (torch.Tensor): Similarity logits `(n_anchor, n_items)` with non-candidates at `-inf`.
        pos_mask (torch.Tensor): Boolean `(n_anchor, n_items)` positive-pair mask.
        cand_mask (torch.Tensor): Boolean `(n_anchor, n_items)` candidate mask (positives + negatives).
        temperature (float): InfoNCE temperature (sets the `exp(-1/temp)` floor).
        tau_plus (float): Class-prior in `[0, 1)`.

    Returns:
        torch.Tensor: Scalar debiased loss (mean over anchors).
    """
    # Exponentiate against the per-anchor max and split the mass into positives and negatives.
    m = logits.max(dim=1, keepdim=True).values  # (n_anchor, 1) per-anchor max candidate logit
    shifted = torch.exp(logits - m)  # ~0 at the -inf (non-candidate) positions
    pos_m = pos_mask.to(shifted.dtype)
    neg_m = (cand_mask & ~pos_mask).to(shifted.dtype)
    pos_sum = (shifted * pos_m).sum(dim=1)
    neg_sum = (shifted * neg_m).sum(dim=1)
    pos_cnt = pos_m.sum(dim=1).clamp_min(1.0)
    neg_cnt = neg_m.sum(dim=1).clamp_min(1.0)

    # Subtract the leaked positive mass from the negative mean, floored to stay positive.
    e_neg = (neg_sum / neg_cnt - tau_plus * (pos_sum / pos_cnt)) / (1.0 - tau_plus)
    floor = torch.exp(logits.new_tensor(-1.0 / temperature) - m.squeeze(1))  # shifted exp(-1/temp)
    e_neg = torch.maximum(e_neg, floor)
    denom = pos_sum + neg_cnt * e_neg
    return (torch.log(denom.clamp_min(1e-12)) - torch.log(pos_sum.clamp_min(1e-12))).mean()
