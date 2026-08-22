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
        within_task (bool): Restrict every anchor's denominator to texts of its own task (needs `attach_tasks`).
    """

    lengths: torch.Tensor | None
    admissible: torch.Tensor | None
    tasks: torch.Tensor | None

    def __init__(self, band: int = 0, min_candidates: int = 32, within_task: bool = False) -> None:
        """Builds the scorer.

        Args:
            band (int, optional): Half-width in words of the length-matched candidate set. Defaults to 0 (off).
            min_candidates (int, optional): Widen the band rather than score against fewer texts than this.
                Defaults to 32.
            within_task (bool, optional): Score each anchor against same-task texts only; task and stimulus are fully
                confounded on ZuCo, so a cross-task distractor is separable by task alone. Defaults to False.
        """
        super().__init__()
        self.band = int(band)
        self.min_candidates = int(min_candidates)
        self.within_task = bool(within_task)
        self.register_buffer('lengths', None, persistent=False)
        self.register_buffer('admissible', None, persistent=False)
        self.register_buffer('tasks', None, persistent=False)

    def attach_lengths(self, lengths: torch.Tensor) -> None:
        """Attaches the `(n_texts,)` word count of every gallery text.

        Args:
            lengths (torch.Tensor): Long word counts aligned with the gallery's row order.
        """
        self.lengths = lengths

    def attach_tasks(self, tasks: torch.Tensor) -> None:
        """Attaches the `(n_texts,)` task id of every gallery text, for the `within_task` denominator.

        Args:
            tasks (torch.Tensor): Long task ids aligned with the gallery's row order (`-1` = unknown, which matches
                no anchor and so never enters a denominator).
        """
        self.tasks = tasks.long()

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
                included, because a softmax with no numerator is not a loss. Under `within_task` an anchor whose
                same-task band is too thin is dropped instead of widened, marked by an all-`False` row.
        """
        device = text_id.device
        own = text_id.clamp(0, n_texts - 1)
        allowed = (
            torch.ones(n_texts, dtype=torch.bool, device=device)
            if self.admissible is None
            else self.admissible.to(device)
        )
        same_task = self._same_task(own)

        if self.band <= 0 or self.lengths is None:
            mask = allowed[None, :].expand(text_id.shape[0], n_texts).clone()
            if same_task is None:
                return mask
            mask &= same_task
            mask[torch.arange(text_id.shape[0], device=device), own] = True
            return mask

        lengths = self.lengths.to(device)
        anchor_len = lengths[own]
        mask = ((lengths[None, :] - anchor_len[:, None]).abs() <= self.band) & allowed[None, :]
        if same_task is not None:
            mask &= same_task
        sparse = mask.sum(dim=1) < self.min_candidates

        if same_task is None:
            # A band that strands an anchor with almost no distractors would make its loss trivially small rather than
            # hard, so those rows fall back to the full gallery and the metric below says how often that happened.
            # Widening a stranded anchor reaches for the rest of the *admissible* gallery, never past it.
            if bool(sparse.any()):
                mask[sparse] = allowed[None, :]

            # A zero-distance text is inside any band, so this is an invariant guard rather than a correction: a
            # cross-entropy whose target column is masked saturates at the float floor and stops reading the model.
            mask[torch.arange(text_id.shape[0], device=device), own] = True
            return mask

        # Within-task the widening fallback would reach across tasks and hand back the shortcut this mode exists to
        # remove, so a stranded anchor is dropped -- an all-False row `compute` excludes from the loss.
        mask[torch.arange(text_id.shape[0], device=device), own] = True
        mask[sparse] = False
        return mask

    def _same_task(self, own: torch.Tensor) -> torch.Tensor | None:
        """Returns the `(n_anchors, n_texts)` same-task mask, or `None` when task matching is off or unattached."""
        if not self.within_task or self.tasks is None:
            return None

        tasks = self.tasks.to(own.device)

        return tasks[None, :] == tasks[own][:, None]

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
        target = text_id.clamp(0, n_texts - 1)
        extra: dict[str, float] = {}
        if self.within_task and self.tasks is not None:
            # A dropped anchor is an all-False row: it has no denominator, so it leaves the loss entirely.
            active = mask.any(dim=1)
            extra['gallery_dropped'] = float(int((~active).sum()))
            if not bool(active.any()):
                return z_eeg.new_zeros(()), extra
            mask, logits, target = mask[active], logits[active], target[active]

        neg_inf = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(~mask, neg_inf)
        loss = F.cross_entropy(masked, target)

        with torch.no_grad():
            top1 = float((masked.argmax(dim=1) == target).float().mean())
            candidates = float(mask.sum(dim=1).float().mean())
        return loss, {
            'gallery_loss': float(loss.detach()),
            'gallery_top1': top1,
            'gallery_candidates': candidates,
            'gallery_chance': 1.0 / max(candidates, 1.0),
            **extra,
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
    within_task = bool(getattr(config, 'within_task_negatives', False))
    _LOG.info(
        'Gallery contrast on: %d texts in the denominator, length band %s%s.',
        n_texts,
        f'+/-{band} words' if band > 0 else 'off (whole gallery)',
        ', same-task candidates only' if within_task else '',
    )
    return GalleryContrast(
        band=band,
        min_candidates=int(getattr(config, 'gallery_min_candidates', 32)),
        within_task=within_task,
    )
