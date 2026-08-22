"""The semantic rate ladder: a text-anchored residual quantiser that caps, and measures, the decoder's bit budget."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from zte.config import DecoderConfig
from zte.logging_utils import get_logger

_LOG = get_logger('models.decoder.quantiser')

# Laplace smoothing on the EMA cluster sizes, so a stage with a briefly-idle code does not divide by zero.
_CLUSTER_EPS: float = 1e-5


@dataclass(slots=True)
class LadderOutput:
    """One forward pass of the rate ladder.

    Attributes:
        z (torch.Tensor): The quantised vector `(batch_size, dim)`, straight-through so gradients reach the encoder.
        codes (torch.Tensor): Chosen code index per stage `(batch_size, n_stages)`.
        commit (torch.Tensor): Scalar commitment loss pulling the continuous vector onto its codes.
        length_logits (torch.Tensor | None): Word-count logits read off the reserved stage, when one is reserved.
        residual_z (torch.Tensor): The quantised vector with the reserved length stage removed, which is the vector
            whose bits are attributable to the brain rather than to eye-tracking word segmentation.
        usage (torch.Tensor): Fraction of each stage's codebook used by this batch `(n_stages,)`.
    """

    z: torch.Tensor
    codes: torch.Tensor
    commit: torch.Tensor
    length_logits: torch.Tensor | None
    residual_z: torch.Tensor
    usage: torch.Tensor


class SemanticRateLadder(nn.Module):
    """A residual vector quantiser whose stages are a hard, auditable ceiling on the decoder's bit budget.

    The measured problem is arithmetic, not engineering: sentence identity over ZuCo's 700 stimuli needs 9.45 bits,
    word count alone supplies 5.14 of them free through eye-tracking segmentation, and the encoder has been measured
    at roughly 1.5. A continuous 768-d conditioning vector hides that -- it can in principle carry any number of bits,
    so "how much did the brain contribute" has to be argued after the fact from retrieval ranks. Sending the vector
    through `n_stages` codebooks of `n_codes` each replaces the argument with a constraint: the channel carries at
    most `n_stages * log2(n_codes)` bits, and `bit_report` measures how many of them the codes actually use.

    Reserving stage 0 for word count closes the other half. That stage is trained to predict the length the decoder
    would otherwise infer for free, and the remaining stages are penalised for correlating with it, so `residual_z`
    is the conditioning whose information is not the confound.

    Attributes:
        dim (int): Vector width.
        n_stages (int): Residual stages.
        n_codes (int): Codes per stage.
        length_stage (bool): Whether stage 0 is reserved for word count.
        capacity_bits (float): The architectural ceiling, `n_stages * log2(n_codes)`.
    """

    codebook: torch.Tensor
    cluster_size: torch.Tensor
    cluster_sum: torch.Tensor
    idle_steps: torch.Tensor

    def __init__(
        self,
        dim: int,
        n_stages: int = 4,
        n_codes: int = 256,
        *,
        decay: float = 0.99,
        commit_weight: float = 0.25,
        revive_after: int = 200,
        length_stage: bool = False,
        max_words: int = 96,
    ) -> None:
        """Builds an unfitted ladder whose codebooks are random until `anchor` seeds them from the text space.

        Args:
            dim (int): Vector width.
            n_stages (int, optional): Residual stages. Defaults to 4.
            n_codes (int, optional): Codes per stage. Defaults to 256.
            decay (float, optional): EMA decay for codebook updates. Defaults to 0.99.
            commit_weight (float, optional): Weight of the commitment loss. Defaults to 0.25.
            revive_after (int, optional): Steps an unused code survives before re-seeding. Defaults to 200.
            length_stage (bool, optional): Reserve stage 0 for word count. Defaults to False.
            max_words (int, optional): Word-count classes the reserved stage predicts. Defaults to 96.
        """
        super().__init__()
        self.dim = int(dim)
        self.n_stages = max(int(n_stages), 1)
        self.n_codes = max(int(n_codes), 2)
        self.decay = float(decay)
        self.commit_weight = float(commit_weight)
        self.revive_after = max(int(revive_after), 0)
        self.length_stage = bool(length_stage)
        self.capacity_bits = self.n_stages * math.log2(self.n_codes)

        book = torch.randn(self.n_stages, self.n_codes, self.dim) * 0.02
        self.register_buffer('codebook', book)
        self.register_buffer('cluster_size', torch.zeros(self.n_stages, self.n_codes))
        self.register_buffer('cluster_sum', book.clone())
        self.register_buffer('idle_steps', torch.zeros(self.n_stages, self.n_codes))

        self.length_head = nn.Linear(self.dim, max_words) if length_stage else None

    @torch.no_grad()
    def anchor(self, cloud: torch.Tensor, iters: int = 12, seed: int = 0) -> None:
        """Seeds every stage from a cloud of frozen text vectors, so a code means something linguistic from step 0.

        Args:
            cloud (torch.Tensor): Text-space vectors `(n, dim)` -- the training stimuli's frozen sentence embeddings.
            iters (int, optional): k-means refinement passes per stage. Defaults to 12.
            seed (int, optional): Seed for the initial code sample. Defaults to 0.

        Note:
            Each stage is fitted to the *residual* left by the stages above it, which is what makes the ladder
            coarse-to-fine: stage 0 splits the text manifold into broad neighbourhoods and later stages refine inside
            them, so truncating the ladder degrades gracefully instead of failing.
        """
        rows = cloud.detach().to(torch.float32)
        if rows.ndim != 2 or rows.shape[1] != self.dim or rows.shape[0] < 2:
            _LOG.warning('Rate ladder not anchored: need a (n>=2, %d) text cloud, got %s.', self.dim, tuple(rows.shape))
            return

        generator = torch.Generator(device='cpu').manual_seed(seed)
        residual = rows.cpu()
        for stage in range(self.n_stages):
            centres = _kmeans(residual, self.n_codes, iters=iters, generator=generator)
            self.codebook[stage].copy_(centres.to(self.codebook.device))
            self.cluster_sum[stage].copy_(centres.to(self.codebook.device))
            self.cluster_size[stage].fill_(1.0)
            residual = residual - centres[_nearest(residual, centres)]
        _LOG.info(
            'Rate ladder anchored on %d text vectors: %d stages x %d codes = %.1f bit ceiling.',
            int(rows.shape[0]),
            self.n_stages,
            self.n_codes,
            self.capacity_bits,
        )

    def forward(self, z: torch.Tensor, n_words: torch.Tensor | None = None) -> LadderOutput:
        """Quantises `z` stage by stage, returning the straight-through vector and its codes.

        Args:
            z (torch.Tensor): Continuous conditioning vectors `(batch_size, dim)`.
            n_words (torch.Tensor | None, optional): Word count per row, read by the reserved length stage.
                Defaults to None.

        Returns:
            LadderOutput: The quantised vector, its codes, the commitment loss and the length read-out.
        """
        residual = z
        quantised = torch.zeros_like(z)
        reserved = torch.zeros_like(z)
        codes: list[torch.Tensor] = []
        commit = z.new_zeros(())
        usage: list[torch.Tensor] = []

        for stage in range(self.n_stages):
            book = self.codebook[stage]
            index = _nearest(residual.detach(), book)
            picked = F.embedding(index, book)

            commit = commit + F.mse_loss(residual, picked.detach())
            if self.training:
                self._ema_update(stage, residual.detach(), index)

            quantised = quantised + picked
            if self.length_stage and stage == 0:
                reserved = picked
            residual = residual - picked
            codes.append(index)
            usage.append(index.unique().numel() / self.n_codes * torch.ones((), device=z.device))

        # Straight-through: the LM sees the quantised vector, the encoder sees the gradient of the continuous one.
        out = z + (quantised - z).detach()
        residual_out = out - reserved.detach() if self.length_stage else out
        length_logits = None
        if self.length_head is not None and n_words is not None:
            length_logits = self.length_head(reserved)

        return LadderOutput(
            z=out,
            codes=torch.stack(codes, dim=1),
            commit=self.commit_weight * commit / self.n_stages,
            length_logits=length_logits,
            residual_z=residual_out,
            usage=torch.stack(usage),
        )

    @torch.no_grad()
    def _ema_update(self, stage: int, residual: torch.Tensor, index: torch.Tensor) -> None:
        """Moves stage `stage`'s codes toward the batch mean of the vectors that chose them, and revives dead ones."""
        onehot = F.one_hot(index, self.n_codes).to(residual.dtype)  # (batch_size, n_codes)
        counts = onehot.sum(dim=0)
        sums = onehot.t() @ residual  # (n_codes, dim)

        self.cluster_size[stage].mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        self.cluster_sum[stage].mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
        total = self.cluster_size[stage].sum()
        smoothed = (self.cluster_size[stage] + _CLUSTER_EPS) / (total + self.n_codes * _CLUSTER_EPS) * total
        self.codebook[stage].copy_(self.cluster_sum[stage] / smoothed.clamp_min(_CLUSTER_EPS).unsqueeze(1))

        # A code nothing has chosen for `revive_after` steps is silently shrinking the ladder's real bit-rate below
        # the ceiling the report quotes, so it is re-seeded onto the batch's worst-fitting vector instead.
        self.idle_steps[stage].add_(1.0)
        self.idle_steps[stage][counts > 0] = 0.0
        if not self.revive_after:
            return
        dead = torch.nonzero(self.idle_steps[stage] > self.revive_after, as_tuple=False).squeeze(1)
        if dead.numel() == 0 or residual.shape[0] == 0:
            return
        error = (residual - F.embedding(index, self.codebook[stage])).pow(2).sum(dim=1)
        worst = torch.argsort(error, descending=True)[: dead.numel()]
        seeds = residual[worst % residual.shape[0]]
        self.codebook[stage][dead[: seeds.shape[0]]] = seeds
        self.cluster_sum[stage][dead[: seeds.shape[0]]] = seeds
        self.cluster_size[stage][dead[: seeds.shape[0]]] = 1.0
        self.idle_steps[stage][dead] = 0.0

    def length_orthogonality(self, codes: torch.Tensor, n_words: torch.Tensor) -> torch.Tensor:
        """Penalises the non-reserved stages for carrying word count -- the 5.14 bits ZuCo gives away free.

        Args:
            codes (torch.Tensor): Chosen codes `(batch_size, n_stages)`.
            n_words (torch.Tensor): Word count per row `(batch_size,)`.

        Returns:
            torch.Tensor: Scalar penalty, zero when no stage is reserved or the batch has one length.
        """
        if not self.length_stage or self.n_stages < 2 or codes.shape[0] < 2:
            return codes.new_zeros((), dtype=torch.float32)

        length = n_words.to(torch.float32)
        length = (length - length.mean()) / length.std().clamp_min(1e-6)
        penalty = codes.new_zeros((), dtype=torch.float32)

        # A code index is not differentiable, so the penalty acts on the codes' *vectors*: their normalised
        # cross-covariance with length, which is exactly what a length-predicting stage would maximise.
        for stage in range(1, self.n_stages):
            vectors = F.embedding(codes[:, stage], self.codebook[stage])
            centred = vectors - vectors.mean(dim=0, keepdim=True)
            scale = centred.std(dim=0).clamp_min(1e-6)
            penalty = penalty + ((centred / scale) * length[:, None]).mean(dim=0).pow(2).sum()

        return penalty / max(self.n_stages - 1, 1)

    @torch.no_grad()
    def bit_report(self, codes: np.ndarray, targets: np.ndarray | None = None) -> dict[str, Any]:
        """Measures the bits the ladder actually delivered, against the ceiling it was built with.

        Args:
            codes (np.ndarray): Chosen codes `(n, n_stages)` over an evaluation set.
            targets (np.ndarray | None, optional): Stimulus id per row, so the mutual information between the code
                and sentence identity can be estimated. Defaults to None.

        Returns:
            dict[str, Any]: `capacity_bits`, per-stage `entropy_bits` and `used_codes`, the joint `code_entropy_bits`
                and, when `targets` are given, `mutual_information_bits` against sentence identity.
        """
        rows = np.asarray(codes, dtype=np.int64).reshape(-1, self.n_stages)
        report: dict[str, Any] = {
            'capacity_bits': float(self.capacity_bits),
            'n_stages': int(self.n_stages),
            'n_codes': int(self.n_codes),
            'length_stage': bool(self.length_stage),
            'n': int(rows.shape[0]),
        }
        if rows.shape[0] < 2:
            return report

        report['entropy_bits'] = [float(_entropy_bits(rows[:, s])) for s in range(self.n_stages)]
        report['used_codes'] = [int(np.unique(rows[:, s]).size) for s in range(self.n_stages)]
        joint = _joint_index(rows)
        report['code_entropy_bits'] = float(_entropy_bits(joint))
        if targets is not None:
            ids = np.asarray(targets).reshape(-1)
            report['mutual_information_bits'] = float(_mutual_information_bits(joint, ids))
            if self.length_stage and self.n_stages > 1:
                residual = _joint_index(rows[:, 1:])
                report['residual_mutual_information_bits'] = float(_mutual_information_bits(residual, ids))
        return report


def _kmeans(rows: torch.Tensor, k: int, iters: int, generator: torch.Generator) -> torch.Tensor:
    """Returns `k` centres for `rows` by Lloyd's algorithm, seeded from a random sample without replacement."""
    n = rows.shape[0]
    take = min(k, n)
    start = torch.randperm(n, generator=generator)[:take]
    centres = rows[start].clone()
    if take < k:
        pad = (
            centres[torch.randint(take, (k - take,), generator=generator)]
            + torch.randn(k - take, rows.shape[1], generator=generator) * 0.01
        )
        centres = torch.cat([centres, pad])

    for _ in range(max(iters, 0)):
        assign = _nearest(rows, centres)
        counts = torch.bincount(assign, minlength=k).clamp_min(1).unsqueeze(1).to(rows.dtype)
        sums = torch.zeros_like(centres).index_add_(0, assign, rows)
        moved = sums / counts
        # An empty cluster would collapse to the origin, so it keeps its previous centre instead.
        empty = torch.bincount(assign, minlength=k) == 0
        moved[empty] = centres[empty]
        centres = moved
    return centres


def _nearest(rows: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """Returns the index of the nearest centre for each row, by squared euclidean distance."""
    # ||a - b||^2 expanded so the (n, k) distance matrix never materialises the (n, k, d) difference.
    distance = rows.pow(2).sum(dim=1, keepdim=True) - 2.0 * rows @ centres.t() + centres.pow(2).sum(dim=1)
    return distance.argmin(dim=1)


def _entropy_bits(values: np.ndarray) -> float:
    """Shannon entropy in bits of a discrete sample."""
    counts = np.unique(values, return_counts=True)[1].astype(np.float64)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def _joint_index(rows: np.ndarray) -> np.ndarray:
    """Collapses a multi-stage code into one integer per row, without assuming the codebook size."""
    if rows.ndim == 1 or rows.shape[1] == 1:
        return rows.reshape(-1)
    return np.unique(rows, axis=0, return_inverse=True)[1].reshape(-1)


def _mutual_information_bits(codes: np.ndarray, targets: np.ndarray) -> float:
    """Plug-in mutual information in bits between a discrete code and a discrete target.

    Note:
        The plug-in estimator is biased upward when the code space is large against `n`, which is exactly this
        regime -- 700 queries against a four-stage ladder. It is reported as an upper bound on what the channel
        delivered, never as the channel's rate.
    """
    joint, counts = np.unique(np.stack([codes, targets], axis=1), axis=0, return_counts=True)
    total = counts.sum()
    p_joint = counts / total
    p_code = np.bincount(np.unique(codes, return_inverse=True)[1]).astype(np.float64) / total
    p_target = np.bincount(np.unique(targets, return_inverse=True)[1]).astype(np.float64) / total
    code_index = np.unique(codes, return_inverse=True)[1][np.searchsorted(np.unique(codes), joint[:, 0])]
    target_index = np.unique(targets, return_inverse=True)[1][np.searchsorted(np.unique(targets), joint[:, 1])]
    marginal = p_code[code_index] * p_target[target_index]
    return float(np.sum(p_joint * np.log2(np.maximum(p_joint, 1e-12) / np.maximum(marginal, 1e-12))))


def build_rate_ladder(config: DecoderConfig, dim: int, max_words: int = 96) -> SemanticRateLadder | None:
    """Constructs the rate ladder a decoder configuration asks for, or `None` for the continuous path.

    Args:
        config (DecoderConfig): Decoder configuration (uses the `rate_*` fields).
        dim (int): Width of the conditioning vector.
        max_words (int, optional): Word-count classes the reserved length stage predicts. Defaults to 96.

    Returns:
        SemanticRateLadder | None: The ladder, or `None` when `rate_ladder='none'`.
    """
    if config.rate_ladder == 'none':
        return None
    return SemanticRateLadder(
        dim,
        n_stages=config.rate_stages,
        n_codes=config.rate_codes,
        decay=config.rate_decay,
        commit_weight=config.rate_commit_weight,
        revive_after=config.rate_revive_after,
        length_stage=config.rate_length_stage,
        max_words=max_words,
    )
