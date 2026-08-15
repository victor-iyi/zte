"""Cross-reader consensus: twelve brains read the same 700 sentences, so the content is the part they agree on."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from zte.logging_utils import get_logger

_LOG = get_logger('models.encoder.consensus')


class ConsensusBank(nn.Module):
    """An EMA bank of one consensus vector per stimulus, averaged over the readers who have read it.

    ZuCo's design gives every sentence to all twelve subjects, so each stimulus has twelve noisy measurements of one
    latent content vector. A single reading is `content + reader style + trial noise`; the mean over readers cancels
    the second and third at a rate of `1 / sqrt(n_readers)` and is therefore a strictly better content estimate than
    any row the encoder can see. This bank holds that estimate and hands it back as a teacher.

    Note:
        The bank is written only while training and is never consulted at inference, so a held-out subject's readings
        never enter it and never read from it. Its one approximation is self-inclusion: the anchor's own earlier
        passes sit in its prototype with weight bounded by `1 - decay`. That can weaken the teacher; it cannot
        manufacture a held-out result.

    Attributes:
        dim (int): Vector width.
        n_keys (int): Number of stimulus keys the bank is sized for.
        decay (float): EMA decay applied on every write.
        min_readers (int): Distinct subjects a key needs before its prototype is served.
    """

    prototypes: torch.Tensor
    readers: torch.Tensor
    writes: torch.Tensor

    def __init__(self, n_keys: int, dim: int, n_subjects: int = 12, decay: float = 0.99, min_readers: int = 2) -> None:
        """Builds an empty bank.

        Args:
            n_keys (int): Number of distinct stimulus keys.
            dim (int): Vector width.
            n_subjects (int, optional): Subject vocabulary size. Defaults to 12.
            min_readers (int, optional): Distinct subjects required before a prototype is served. Defaults to 2.
            decay (float, optional): EMA decay per write. Defaults to 0.99.
        """
        super().__init__()
        self.dim = int(dim)
        self.n_keys = int(n_keys)
        self.n_subjects = max(int(n_subjects), 1)
        self.decay = float(decay)
        self.min_readers = int(min_readers)
        self.register_buffer('prototypes', torch.zeros(self.n_keys, self.dim))

        # Who has read each stimulus, not how often: two passes by one reader must not look like two readers.
        self.register_buffer('readers', torch.zeros(self.n_keys, self.n_subjects, dtype=torch.bool))
        self.register_buffer('writes', torch.zeros(self.n_keys, dtype=torch.long))

    @torch.no_grad()
    def update(self, keys: torch.Tensor, vectors: torch.Tensor, subject: torch.Tensor) -> None:
        """Folds a batch of readings into their stimuli's prototypes.

        Args:
            keys (torch.Tensor): Long `(n,)` stimulus keys; negative entries are skipped.
            vectors (torch.Tensor): Float `(n, dim)` readings.
            subject (torch.Tensor): Long `(n,)` subject ids, used to count distinct readers per key.
        """
        valid = (keys >= 0) & (keys < self.n_keys)
        if not bool(valid.any()):
            return

        k = keys[valid]
        v = F.normalize(vectors[valid].detach().float(), dim=-1)

        # Several readings of one stimulus can land in the same batch; average them before the single EMA step so a
        # crowded batch does not decay the prototype harder than a sparse one.
        summed = torch.zeros_like(self.prototypes).index_add_(0, k, v)
        ones = torch.ones_like(k, dtype=v.dtype)
        counted = torch.zeros(self.n_keys, device=v.device, dtype=v.dtype).index_add_(0, k, ones)
        touched = counted > 0
        batch_mean = summed[touched] / counted[touched, None]

        cold = self.writes[touched] == 0
        blended = self.decay * self.prototypes[touched] + (1.0 - self.decay) * batch_mean
        self.prototypes[touched] = torch.where(cold[:, None], batch_mean, blended)
        self.writes[touched] += 1
        self.readers[k, subject[valid].clamp(0, self.n_subjects - 1)] = True

    def lookup(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(prototypes, ready)` for `keys`, where `ready` marks keys with enough distinct readers.

        Args:
            keys (torch.Tensor): Long `(n,)` stimulus keys.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: L2-normalised `(n, dim)` prototypes and a boolean `(n,)` readiness mask.
        """
        safe = keys.clamp(0, self.n_keys - 1)
        served = self.readers[safe].sum(dim=1) >= self.min_readers
        ready = (keys >= 0) & (keys < self.n_keys) & served
        return F.normalize(self.prototypes[safe], dim=-1), ready

    def ready_keys(self) -> torch.Tensor:
        """Returns the long indices of every key whose prototype has enough distinct readers to serve."""
        return torch.nonzero(self.readers.sum(dim=1) >= self.min_readers, as_tuple=False).squeeze(1)

    def coverage(self) -> dict[str, float]:
        """Returns how much of the stimulus set the bank currently covers, for the run's metrics."""
        counts = self.readers.sum(dim=1)
        return {
            'consensus_keys_ready': float(int((counts >= self.min_readers).sum())),
            'consensus_keys_total': float(self.n_keys),
            'consensus_mean_readers': float(counts.float().mean()),
        }


class ConsensusDistiller(nn.Module):
    """Trains each single reading against the cross-reader consensus for the stimulus it read.

    Two terms, and they ask different things. The *pull* term is a denoising target: move this reading toward what
    everyone who read this sentence agreed on. The *gallery* term is the evaluation, moved into the loss: pick your own
    stimulus out of every stimulus the bank knows, scored in EEG space rather than against text, so the modality gap
    cannot be the thing that separates them.

    Attributes:
        bank (ConsensusBank): The prototype store.
        logit_scale (nn.Parameter): Learnable inverse temperature for the gallery term, clamped in the forward pass.
    """

    def __init__(
        self,
        n_keys: int,
        dim: int,
        *,
        n_subjects: int = 12,
        decay: float = 0.99,
        min_readers: int = 2,
        temperature: float = 0.07,
        gallery_size: int = 1024,
    ) -> None:
        """Builds the distiller and its bank.

        Args:
            n_keys (int): Number of distinct stimulus keys.
            dim (int): Vector width of the readings being distilled.
            n_subjects (int, optional): Subject vocabulary size. Defaults to 12.
            decay (float, optional): Prototype EMA decay. Defaults to 0.99.
            min_readers (int, optional): Distinct readers a prototype needs before it is used. Defaults to 2.
            temperature (float, optional): Initial softmax temperature for the gallery term. Defaults to 0.07.
            gallery_size (int, optional): Cap on prototypes in the denominator per step. Defaults to 1024.
        """
        super().__init__()
        self.bank = ConsensusBank(n_keys, dim, n_subjects=n_subjects, decay=decay, min_readers=min_readers)
        self.logit_scale = nn.Parameter(torch.tensor(float(1.0 / max(temperature, 1e-4))).log())
        self.gallery_size = int(gallery_size)

    def compute(
        self,
        vectors: torch.Tensor,
        keys: torch.Tensor,
        subject: torch.Tensor,
        *,
        pull_weight: float,
        gallery_weight: float,
        prefix: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Scores readings against their own consensus and against the consensus of every other stimulus.

        Args:
            vectors (torch.Tensor): Readings `(n, dim)`.
            keys (torch.Tensor): Long `(n,)` stimulus keys; negatives are skipped.
            subject (torch.Tensor): Long `(n,)` subject ids.
            pull_weight (float): Weight of the denoising term.
            gallery_weight (float): Weight of the pick-your-own-stimulus term.
            prefix (str): Metric-name prefix, e.g. `'consensus_sentence'`.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: `(loss, metrics)`.
        """
        loss = vectors.new_zeros(())
        metrics: dict[str, float] = {}
        if vectors.numel() == 0 or (pull_weight <= 0.0 and gallery_weight <= 0.0):
            return loss, metrics

        z = F.normalize(vectors, dim=-1)
        protos, ready = self.bank.lookup(keys)
        protos = protos.to(z.dtype)

        if bool(ready.any()) and pull_weight > 0.0:
            pull = (1.0 - (z[ready] * protos[ready]).sum(-1)).mean()
            loss = loss + pull_weight * pull
            metrics[f'{prefix}_pull'] = float(pull.detach())

        if gallery_weight > 0.0:
            gal_loss, gal_metrics = self._gallery(z, keys, ready, prefix)
            loss = loss + gallery_weight * gal_loss
            metrics.update(gal_metrics)

        # Read then write: this step's teacher is built from earlier steps only.
        if self.training:
            self.bank.update(keys, vectors, subject)

        return loss, metrics

    def _gallery(
        self, z: torch.Tensor, keys: torch.Tensor, ready: torch.Tensor, prefix: str
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Cross-entropy over the bank's ready prototypes, with each anchor's own prototype as the answer."""
        served = self.bank.ready_keys()
        if served.numel() < 2 or not bool(ready.any()):
            return z.new_zeros(()), {}

        # The anchors' own keys must survive the cap, or the answer would not be in the gallery at all.
        anchor_keys = torch.unique(keys[ready])
        if served.numel() > self.gallery_size:
            others = served[~torch.isin(served, anchor_keys)]
            room = max(self.gallery_size - int(anchor_keys.numel()), 0)
            served = torch.cat([anchor_keys, others[:room]])

        gallery = F.normalize(self.bank.prototypes[served], dim=-1).to(z.dtype)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = (z[ready] @ gallery.t()) * scale

        # Position of each anchor's own key inside `served`, which is what cross-entropy needs as the label.
        position = torch.full((self.bank.n_keys,), -1, device=z.device, dtype=torch.long)
        position[served] = torch.arange(served.numel(), device=z.device)
        target = position[keys[ready]]
        hit = target >= 0
        if not bool(hit.any()):
            return z.new_zeros(()), {}

        loss = F.cross_entropy(logits[hit], target[hit])
        with torch.no_grad():
            top1 = float((logits[hit].argmax(dim=1) == target[hit]).float().mean())
        return loss, {
            f'{prefix}_gallery': float(loss.detach()),
            f'{prefix}_gallery_top1': top1,
            f'{prefix}_gallery_size': float(served.numel()),
        }


def build_consensus(
    config: Any,
    n_sentences: int,
    n_content: int,
    dim: int,
    n_subjects: int = 12,
) -> tuple[ConsensusDistiller | None, ConsensusDistiller | None]:
    """Constructs the sentence-level and word-level distillers an objective configuration asks for.

    Args:
        config (ObjectiveConfig): Objective configuration (reads the `consensus_*` fields).
        n_sentences (int): Number of distinct stimulus texts (sizes the sentence bank).
        n_content (int): Number of distinct word slots across the stimulus set (sizes the word bank).
        dim (int): Width of the vectors being distilled.
        n_subjects (int, optional): Subject vocabulary size, which is how "distinct readers" is counted.
            Defaults to 12.

    Returns:
        tuple[ConsensusDistiller | None, ConsensusDistiller | None]: `(sentence, word)`, either of which may be `None`.
    """
    sentence_on = config.consensus_weight > 0.0 or config.consensus_gallery_weight > 0.0
    word_on = config.consensus_word_weight > 0.0

    def _make(n_keys: int, kind: str) -> ConsensusDistiller | None:
        if n_keys < 2:
            _LOG.warning('Consensus %s distillation requested but only %d keys exist; disabling it.', kind, n_keys)
            return None

        _LOG.info(
            'Consensus %s distillation on: %d keys x %d dims, min %d of %d readers.',
            kind,
            n_keys,
            dim,
            config.consensus_min_readers,
            n_subjects,
        )
        return ConsensusDistiller(
            n_keys,
            dim,
            n_subjects=n_subjects,
            decay=config.consensus_decay,
            min_readers=config.consensus_min_readers,
            temperature=config.consensus_temperature,
            gallery_size=config.consensus_gallery_size,
        )

    return (_make(n_sentences, 'sentence') if sentence_on else None, _make(n_content, 'word') if word_on else None)
