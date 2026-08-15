"""Token-level lexical alignment: make each word's EEG say which word it was, and say it the same way in every head.

Sentence-level InfoNCE pulls only at the pooled vector, so nothing in the loss ever asks a single word's EEG to mean
that word -- and on ZuCo, measured, nothing does: cross-subject word retrieval sits at Top-1 0.004 against a chance of
0.003. Lexical structure is not going to emerge from a sentence-level gradient. This module demands it.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from zte.config import ObjectiveConfig
from zte.logging_utils import get_logger

_LOG = get_logger('models.objectives.lexical')


class LexicalAligner(nn.Module):
    """Projects per-word EEG tokens into the frozen text space and scores them against the word they read.

    Two directions, and they answer different questions. Against the frozen text embeddings it asks "is this the EEG
    of *this* word" -- absolute lexical identity, learnable from one reader. Against the same word read by *another
    person* it asks "is this the same word, whoever read it" -- which is the property a cross-subject decoder needs
    and the one a single-reader loss will happily skip.

    Attributes:
        head (nn.Linear): The projection into the frozen text space, shared with the decoder's evidence path.
        logit_scale (nn.Parameter): Learnable inverse temperature, clamped in the forward pass.
    """

    text_matrix: torch.Tensor | None

    def __init__(self, hidden_dim: int, text_dim: int, temperature: float = 0.07) -> None:
        """Builds the aligner.

        Args:
            hidden_dim (int): Width of the encoder's token hiddens.
            text_dim (int): Width of the frozen word-embedding space.
            temperature (float, optional): Initial softmax temperature. Defaults to 0.07.
        """
        super().__init__()
        self.head = nn.Linear(hidden_dim, text_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(1.0 / max(temperature, 1e-4))).log())
        self.register_buffer('text_matrix', None, persistent=False)

    def attach(self, matrix: torch.Tensor) -> None:
        """Attaches the frozen `(n_word_types, text_dim)` L2-normalised word-embedding matrix.

        Args:
            matrix (torch.Tensor): Word embeddings indexed by `batch['word_id']`.

        Raises:
            ValueError: If the matrix width disagrees with the projection this aligner was built for.
        """
        if int(matrix.shape[1]) != self.head.out_features:
            raise ValueError(
                f'Lexical target is {int(matrix.shape[1])}-wide against a {self.head.out_features}-wide head; '
                'objective.lexical_source and the head must name the same text space.'
            )
        self.text_matrix = matrix

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns the L2-normalised text-space vector of every token hidden.

        Args:
            hidden (torch.Tensor): Token hiddens `(..., hidden_dim)`.

        Returns:
            torch.Tensor: `(..., text_dim)` unit vectors.
        """
        return F.normalize(self.head(hidden), dim=-1)

    def compute(
        self,
        hidden: torch.Tensor,
        batch: dict[str, Any],
        usable: torch.Tensor,
        *,
        type_weight: float,
        reader_weight: float,
        max_tokens: int = 4096,
        same_subject_negatives: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Scores the batch's usable word tokens against their own word, and against other readers of it.

        Args:
            hidden (torch.Tensor): Token hiddens `(batch_size, seq_len, hidden_dim)`.
            batch (dict[str, Any]): The collated batch (uses `word_id`, `content_id`, `subject`).
            usable (torch.Tensor): Boolean `(batch_size, seq_len)` mask of real, fixated tokens.
            type_weight (float): Weight of the EEG-to-frozen-text-embedding direction.
            reader_weight (float): Weight of the same-word-different-reader direction.
            max_tokens (int, optional): Cap on tokens per step, since the reader direction is quadratic in them.
                Defaults to 4096.
            same_subject_negatives (bool, optional): Restrict the reader direction's negatives to the anchor's own
                subject, so telling anchor from negative cannot be done on subject identity. Defaults to True.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: `(loss, metrics)`.
        """
        device = hidden.device
        loss = hidden.new_zeros(())
        metrics: dict[str, float] = {}
        flat = usable.reshape(-1)
        if not bool(flat.any()) or (type_weight <= 0.0 and reader_weight <= 0.0):
            return loss, metrics

        rows = torch.nonzero(flat, as_tuple=False).squeeze(1)
        if rows.numel() > max_tokens:
            # Deterministic thinning by stride rather than by sample: the tokens kept are spread over every sentence
            # in the batch, so the cap never quietly turns the loss into a few-sentence loss.
            rows = rows[torch.linspace(0, rows.numel() - 1, max_tokens, device=device).long().unique()]

        vectors = self.project(hidden.reshape(-1, hidden.shape[-1])[rows])
        scale = self.logit_scale.exp().clamp(max=100.0)
        word_id = _flat_field(batch, 'word_id', usable)[rows]
        content_id = _flat_field(batch, 'content_id', usable)[rows]
        subject = batch['subject'][:, None].expand_as(usable).reshape(-1)[rows]

        if type_weight > 0.0 and self.text_matrix is not None:
            type_loss, type_metrics = self._type_direction(vectors, word_id, scale)
            loss = loss + type_weight * type_loss
            metrics.update(type_metrics)

        if reader_weight > 0.0:
            reader_loss, reader_metrics = self._reader_direction(
                vectors, content_id, subject, scale, same_subject_negatives
            )
            loss = loss + reader_weight * reader_loss
            metrics.update(reader_metrics)

        metrics['lexical_tokens'] = float(rows.numel())
        return loss, metrics

    def _type_direction(
        self, vectors: torch.Tensor, word_id: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Scores each token against the frozen embeddings of the distinct word types present in the batch."""
        known = word_id >= 0
        if not bool(known.any()) or self.text_matrix is None:
            return vectors.new_zeros(()), {}

        types, inverse = torch.unique(word_id[known], return_inverse=True)
        if types.numel() < 2:
            return vectors.new_zeros(()), {}

        gallery = F.normalize(F.embedding(types, self.text_matrix), dim=-1)  # (n_types, text_dim)
        logits = (vectors[known] @ gallery.t()) * scale
        loss = F.cross_entropy(logits, inverse)
        with torch.no_grad():
            top1 = float((logits.argmax(dim=1) == inverse).float().mean())
        return loss, {
            'lexical_type_loss': float(loss.detach()),
            'lexical_type_top1': top1,
            'lexical_types': float(types.numel()),
        }

    def _reader_direction(
        self,
        vectors: torch.Tensor,
        content_id: torch.Tensor,
        subject: torch.Tensor,
        scale: torch.Tensor,
        same_subject_negatives: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Scores each token against the same word position read by a different person."""
        known = content_id >= 0
        if int(known.sum()) < 2:
            return vectors.new_zeros(()), {}

        v = vectors[known]
        cid, subj = content_id[known], subject[known]
        same_word = cid[:, None] == cid[None, :]
        same_subject = subj[:, None] == subj[None, :]
        eye = torch.eye(v.shape[0], dtype=torch.bool, device=v.device)

        positive = same_word & ~same_subject
        anchors = positive.any(dim=1)
        if not bool(anchors.any()):
            return vectors.new_zeros(()), {}

        # A different reading of the *same* word is never a negative, whoever produced it; and when hard negatives are
        # on, the denominator holds only the anchor's own subject, so subject identity separates nothing.
        candidate = ~same_word & ~eye
        if same_subject_negatives:
            candidate = candidate & same_subject
        usable_rows = anchors & candidate.any(dim=1)
        if not bool(usable_rows.any()):
            return vectors.new_zeros(()), {}

        logits = (v @ v.t()) * scale
        neg_inf = torch.finfo(logits.dtype).min
        allowed = candidate | positive
        denom = torch.logsumexp(logits.masked_fill(~allowed, neg_inf), dim=1)
        numer = torch.logsumexp(logits.masked_fill(~positive, neg_inf), dim=1)
        loss = (denom - numer)[usable_rows].mean()

        with torch.no_grad():
            picked = logits.masked_fill(~allowed, neg_inf).argmax(dim=1)
            hit = positive[torch.arange(picked.numel(), device=picked.device), picked]
            top1 = float(hit[usable_rows].float().mean())
        return loss, {
            'lexical_reader_loss': float(loss.detach()),
            'lexical_reader_top1': top1,
            'lexical_anchors': float(int(usable_rows.sum())),
        }


def _flat_field(batch: dict[str, Any], key: str, usable: torch.Tensor) -> torch.Tensor:
    """Returns a per-token batch field flattened to `(batch_size * seq_len,)`, or all `-1` when it is absent."""
    value = batch.get(key)
    if value is None:
        return torch.full((usable.numel(),), -1, dtype=torch.long, device=usable.device)
    return value.reshape(-1)


def build_lexical_aligner(config: ObjectiveConfig, hidden_dim: int, text_dim: int) -> LexicalAligner | None:
    """Constructs the aligner an objective configuration asks for, or `None` when both directions are off.

    Args:
        config (ObjectiveConfig): Objective configuration (uses the `lexical_*` fields).
        hidden_dim (int): Width of the encoder's token hiddens.
        text_dim (int): Width of the frozen word-embedding space.

    Returns:
        LexicalAligner | None: The aligner, or `None` when it would contribute nothing.
    """
    if config.lexical_weight <= 0.0 and config.lexical_reader_weight <= 0.0:
        return None
    aligner = LexicalAligner(hidden_dim, text_dim, temperature=config.lexical_temperature)
    _LOG.info(
        'Lexical alignment on: %d -> %d projection, type weight %.3f, cross-reader weight %.3f.',
        hidden_dim,
        text_dim,
        config.lexical_weight,
        config.lexical_reader_weight,
    )
    return aligner
