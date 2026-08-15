"""Full-gallery contrastive scoring with length-matched negatives: train on the task the evaluation actually asks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from zte.logging_utils import get_logger

_LOG = get_logger('models.encoder.gallery')


class GalleryContrast(nn.Module):
    """Scores every EEG reading against the whole text gallery, optionally inside its own word-count band.

    An in-batch InfoNCE asks a reading to beat fifteen other sentences; the evaluation asks it to beat six hundred and
    ninety-nine, and the hardest of those -- same length, same passage, same register -- are almost never in a batch.
    Widening the denominator to the full gallery closes that gap for free, because the frozen text matrix is already
    resident.

    Restricting the denominator to texts of a similar length closes a second, worse gap. On ZuCo, word count carries
    5.14 bits of the 9.45 needed to name a sentence, and eye-tracking segmentation hands the model that count for
    nothing. A denominator of same-length texts makes counting words worth zero, so whatever the loss goes on to learn
    is not that.

    Attributes:
        band (int): Half-width in words of the length-matched denominator; 0 uses the whole gallery.
    """

    lengths: torch.Tensor | None
    admissible: torch.Tensor | None

    def __init__(self, band: int = 0, min_candidates: int = 32) -> None:
        """Builds the scorer.

        Args:
            band (int, optional): Half-width in words of the length-matched candidate set. Defaults to 0 (off).
            min_candidates (int, optional): Widen the band rather than score against fewer texts than this.
                Defaults to 32.
        """
        super().__init__()
        self.band = int(band)
        self.min_candidates = int(min_candidates)
        self.register_buffer('lengths', None, persistent=False)
        self.register_buffer('admissible', None, persistent=False)

    def attach_lengths(self, lengths: torch.Tensor) -> None:
        """Attaches the `(n_texts,)` word count of every gallery text.

        Args:
            lengths (torch.Tensor): Long word counts aligned with the gallery's row order.
        """
        self.lengths = lengths

    def restrict_to(self, text_ids: Sequence[int], n_texts: int) -> None:
        """Limits the denominator to the texts actually read in the training split.

        The frozen text matrix is indexed by a whole-dataset text id, so under a stimulus-holding-out split it
        contains rows for sentences the model must never be trained against -- not as a positive and not as a
        negative. Training on a held-out text as a negative still teaches the encoder where *not* to map, which
        shapes the evaluation geometry, so those rows are masked out of the denominator entirely.

        Args:
            text_ids (Sequence[int]): Gallery rows the training split actually read.
            n_texts (int): Gallery size.
        """
        mask = torch.zeros(n_texts, dtype=torch.bool)
        keep = torch.as_tensor([t for t in text_ids if 0 <= t < n_texts], dtype=torch.long)
        if keep.numel():
            mask[keep] = True
        self.admissible = mask
        if int(mask.sum()) < n_texts:
            _LOG.info(
                'Gallery denominator restricted to the %d of %d texts the training split reads; the rest are '
                'held-out stimuli and are not negatives either.',
                int(mask.sum()),
                n_texts,
            )

    def candidate_mask(self, text_id: torch.Tensor, n_texts: int) -> torch.Tensor:
        """Returns the boolean `(n_anchors, n_texts)` denominator mask for a batch of anchors.

        Args:
            text_id (torch.Tensor): Long `(n_anchors,)` gallery row of each anchor's own text.
            n_texts (int): Gallery size.

        Returns:
            torch.Tensor: `True` where a text may appear in that anchor's denominator; the anchor's own text is always
                included, because a softmax with no numerator is not a loss.
        """
        device = text_id.device
        allowed = (
            torch.ones(n_texts, dtype=torch.bool, device=device)
            if self.admissible is None
            else self.admissible.to(device)
        )
        if self.band <= 0 or self.lengths is None:
            return allowed[None, :].expand(text_id.shape[0], n_texts).clone()

        lengths = self.lengths.to(device)
        anchor_len = lengths[text_id.clamp(0, n_texts - 1)]
        mask = ((lengths[None, :] - anchor_len[:, None]).abs() <= self.band) & allowed[None, :]

        # A band that strands an anchor with almost no distractors would make its loss trivially small rather than
        # hard, so those rows fall back to the full gallery and the metric below says how often that happened.
        # Widening a stranded anchor reaches for the rest of the *admissible* gallery, never past it.
        sparse = mask.sum(dim=1) < self.min_candidates
        if bool(sparse.any()):
            mask[sparse] = allowed[None, :]

        # A zero-distance text is inside any band, so this is an invariant guard rather than a correction: a
        # cross-entropy whose target column is masked saturates at the float floor and stops reading the model.
        mask[torch.arange(text_id.shape[0], device=device), text_id.clamp(0, n_texts - 1)] = True
        return mask

    def compute(
        self, z_eeg: torch.Tensor, gallery: torch.Tensor, text_id: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Cross-entropy over the (optionally length-matched) gallery, with each anchor's own text as the answer.

        Args:
            z_eeg (torch.Tensor): L2-normalised EEG sentence vectors `(n_anchors, text_dim)`.
            gallery (torch.Tensor): L2-normalised frozen text matrix `(n_texts, text_dim)`.
            text_id (torch.Tensor): Long `(n_anchors,)` row of each anchor's own text.
            scale (torch.Tensor): Inverse temperature.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: `(loss, metrics)`.
        """
        n_texts = int(gallery.shape[0])
        if n_texts < 2 or z_eeg.shape[0] == 0:
            return z_eeg.new_zeros(()), {}

        logits = (z_eeg @ gallery.t()) * scale
        mask = self.candidate_mask(text_id, n_texts)
        neg_inf = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(~mask, neg_inf)
        target = text_id.clamp(0, n_texts - 1)
        loss = F.cross_entropy(masked, target)

        with torch.no_grad():
            top1 = float((masked.argmax(dim=1) == target).float().mean())
            candidates = float(mask.sum(dim=1).float().mean())
        return loss, {
            'gallery_loss': float(loss.detach()),
            'gallery_top1': top1,
            'gallery_candidates': candidates,
            'gallery_chance': 1.0 / max(candidates, 1.0),
        }


def text_word_counts(texts: list[str]) -> torch.Tensor:
    """Returns the whitespace word count of every gallery text, in gallery row order.

    Args:
        texts (list[str]): Gallery texts, ordered as the frozen text matrix's rows.

    Returns:
        torch.Tensor: Long `(n_texts,)` word counts.
    """
    return torch.tensor([len(t.split()) for t in texts], dtype=torch.long)


def build_gallery_contrast(config: object, n_texts: int) -> GalleryContrast | None:
    """Constructs the gallery scorer an objective configuration asks for, or `None` when it is off.

    Args:
        config (ObjectiveConfig): Objective configuration (reads the `gallery_*` fields).
        n_texts (int): Gallery size, used only for the log line.

    Returns:
        GalleryContrast | None: The scorer, or `None`.
    """
    weight = float(getattr(config, 'gallery_weight', 0.0))
    if weight <= 0.0:
        return None

    band = int(getattr(config, 'gallery_length_band', 0))
    _LOG.info(
        'Gallery contrast on: %d texts in the denominator, length band %s.',
        n_texts,
        f'+/-{band} words' if band > 0 else 'off (whole gallery)',
    )
    return GalleryContrast(band=band, min_candidates=int(getattr(config, 'gallery_min_candidates', 32)))
