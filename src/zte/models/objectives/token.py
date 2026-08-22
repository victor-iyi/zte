"""Sub-word alignment: make a slice of one word's EEG say which word-piece it was, and say it the same way in every head.

Sentence InfoNCE pulls at the pooled vector and the word-level term at whole words. Neither ever asks a slice of a
word's EEG to mean the piece it spells, so the lexical structure a decoder needs to emit one token at a time was
never requested. This module demands it -- while refusing to let the request become a channel of its own.
"""

from typing import Any, Final

import torch
import torch.nn.functional as F
from torch import nn

from zte.config import ObjectiveConfig
from zte.logging_utils import get_logger

_LOG = get_logger('models.objectives.token')

# A word's own slices are adjacent windows of one 350-sample fixation and are near-identical inputs. Left in the
# denominator they are hard negatives with different labels, and the only way to separate them is to encode position
# within the fixation -- a clock, which is the sub-word count arriving by the back door.
_EXCLUDE_SAME_WORD: Final[bool] = True
"""Whether every slice of one word is barred from being another slice's negative. Not a knob: the alternative is a
loss whose only solution reintroduces the leak the level exists to avoid."""


class TokenAligner(nn.Module):
    """Projects intra-word EEG sub-tokens into a frozen sub-word space and scores them against the piece they read.

    Two directions, answering different questions. Against the frozen sub-word embeddings it asks "is this the EEG of
    *this* word-piece" -- absolute identity, learnable from one reader. Against the same piece read by *another
    person* it asks "is this the same piece, whoever read it", which is the property a cross-subject decoder needs
    and the one a single-reader loss will happily skip.

    Attributes:
        head (nn.Linear): The projection into the frozen sub-word space.
        logit_scale (nn.Parameter): Learnable inverse temperature, clamped in the forward pass.
    """

    subword_matrix: torch.Tensor | None
    piece_target: torch.Tensor | None

    def __init__(self, hidden_dim: int, text_dim: int, n_sub: int, temperature: float = 0.07) -> None:
        """Builds the aligner.

        Args:
            hidden_dim (int): Width of the encoder's sub-token hiddens.
            text_dim (int): Width of the frozen sub-word space.
            n_sub (int): Sub-tokens the frontend emits per word, fixed for every word.
            temperature (float, optional): Initial softmax temperature. Defaults to 0.07.
        """
        super().__init__()
        self.head = nn.Linear(hidden_dim, text_dim)
        self.n_sub = int(n_sub)
        self.logit_scale = nn.Parameter(torch.tensor(float(1.0 / max(temperature, 1e-4))).log())
        self.register_buffer('subword_matrix', None, persistent=False)
        self.register_buffer('piece_target', None, persistent=False)

    def attach(self, subword_matrix: torch.Tensor, piece_target: torch.Tensor) -> None:
        """Attaches the frozen sub-word embeddings and the per-word piece targets the loss reads.

        Args:
            subword_matrix (torch.Tensor): `(n_types, text_dim)` L2-normalised frozen sub-word embeddings.
            piece_target (torch.Tensor): Long `(n_content, n_sub)`; entry `(c, k)` is the row of `subword_matrix`
                for piece `k` of the word `content_id == c`, and `-1` where that word has fewer than `k` pieces.

        Raises:
            ValueError: If the matrix width disagrees with the projection, or the target's sub-token axis
                disagrees with the count the frontend emits.
        """
        if int(subword_matrix.shape[1]) != self.head.out_features:
            raise ValueError(
                f'Sub-word target is {int(subword_matrix.shape[1])}-wide against a {self.head.out_features}-wide '
                'head; objective.token_source and the head must name the same space.'
            )
        if int(piece_target.shape[1]) != self.n_sub:
            raise ValueError(
                f'Piece target carries {int(piece_target.shape[1])} slots against {self.n_sub} sub-tokens per word; '
                'the target mask and the frontend must agree.'
            )
        self.subword_matrix = subword_matrix
        self.piece_target = piece_target

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns the L2-normalised sub-word-space vector of every sub-token hidden.

        Args:
            hidden (torch.Tensor): Sub-token hiddens `(..., hidden_dim)`.

        Returns:
            torch.Tensor: `(..., text_dim)` unit vectors.
        """
        return F.normalize(self.head(hidden), dim=-1)

    def compute(
        self,
        sub_hidden: torch.Tensor,
        batch: dict[str, Any],
        usable: torch.Tensor,
        *,
        type_weight: float,
        reader_weight: float,
        max_tokens: int = 4096,
        same_subject_negatives: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Scores the batch's usable sub-tokens against their own word-piece, and against other readers of it.

        Args:
            sub_hidden (torch.Tensor): Sub-token hiddens `(batch_size, seq_len, n_sub, hidden_dim)`.
            batch (dict[str, Any]): The collated batch (uses `content_id` and `subject`).
            usable (torch.Tensor): Boolean `(batch_size, seq_len)` mask of real, fixated words.
            type_weight (float): Weight of the EEG-to-frozen-sub-word direction.
            reader_weight (float): Weight of the same-piece-different-reader direction.
            max_tokens (int, optional): Cap on sub-tokens per step, since the reader direction is quadratic in
                them. Defaults to 4096.
            same_subject_negatives (bool, optional): Restrict the reader direction's negatives to the anchor's own
                subject. Defaults to True.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: `(loss, metrics)`.
        """
        loss = sub_hidden.new_zeros(())
        metrics: dict[str, float] = {}
        if self.piece_target is None or (type_weight <= 0.0 and reader_weight <= 0.0):
            return loss, metrics

        rows, content, slot, subject = self._select(sub_hidden, batch, usable, max_tokens)
        if rows.numel() == 0:
            return loss, metrics

        flat = sub_hidden.reshape(-1, sub_hidden.shape[-1])
        vectors = self.project(flat[rows])
        scale = self.logit_scale.exp().clamp(max=100.0)
        target = self.piece_target[content, slot]

        if type_weight > 0.0 and self.subword_matrix is not None:
            type_loss, type_metrics = self._type_direction(vectors, target, scale)
            loss = loss + type_weight * type_loss
            metrics.update(type_metrics)

        if reader_weight > 0.0:
            reader_loss, reader_metrics = self._reader_direction(
                vectors, content, slot, subject, scale, same_subject_negatives
            )
            loss = loss + reader_weight * reader_loss
            metrics.update(reader_metrics)

        metrics['token_sub_tokens_scored'] = float(rows.numel())
        return loss, metrics

    def _select(
        self, sub_hidden: torch.Tensor, batch: dict[str, Any], usable: torch.Tensor, max_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Picks the scorable sub-token rows and their `(content_id, slot, subject)` keys."""
        b, length, n_sub = sub_hidden.shape[0], sub_hidden.shape[1], sub_hidden.shape[2]
        device = sub_hidden.device

        content = batch['content_id'][:, :, None].expand(b, length, n_sub).reshape(-1)
        slot = torch.arange(n_sub, device=device)[None, None, :].expand(b, length, n_sub).reshape(-1)
        subject = batch['subject'][:, None, None].expand(b, length, n_sub).reshape(-1)
        live = usable[:, :, None].expand(b, length, n_sub).reshape(-1) & (content >= 0)

        # A slot past the word's own piece count has no target; masking it here is the only place the sub-word
        # count is allowed to act, and it acts on the loss rather than on anything the encoder computed.
        if self.piece_target is not None:
            live = live & (self.piece_target[content.clamp(min=0), slot] >= 0)

        rows = torch.nonzero(live, as_tuple=False).squeeze(1)
        if rows.numel() > max_tokens:
            # Deterministic thinning by stride rather than by sample, so the cap never quietly turns the loss into
            # a few-sentence loss.
            rows = rows[torch.linspace(0, rows.numel() - 1, max_tokens, device=device).long().unique()]

        return rows, content[rows], slot[rows], subject[rows]

    def _type_direction(
        self, vectors: torch.Tensor, target: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Scores each sub-token against the frozen embeddings of the distinct piece types present in the batch."""
        if self.subword_matrix is None:
            return vectors.new_zeros(()), {}

        types, inverse = torch.unique(target, return_inverse=True)
        if types.numel() < 2:
            return vectors.new_zeros(()), {}

        gallery = F.normalize(F.embedding(types, self.subword_matrix), dim=-1)
        logits = (vectors @ gallery.t()) * scale
        loss = F.cross_entropy(logits, inverse)
        with torch.no_grad():
            top1 = float((logits.argmax(dim=1) == inverse).float().mean())

        return loss, {
            'token_type_loss': float(loss.detach()),
            'token_type_top1': top1,
            'token_types': float(types.numel()),
        }

    def _reader_direction(
        self,
        vectors: torch.Tensor,
        content: torch.Tensor,
        slot: torch.Tensor,
        subject: torch.Tensor,
        scale: torch.Tensor,
        same_subject_negatives: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Scores each sub-token against the same word-piece of the same word read by a different person."""
        if vectors.shape[0] < 2:
            return vectors.new_zeros(()), {}

        same_word = content[:, None] == content[None, :]
        same_slot = slot[:, None] == slot[None, :]
        same_subject = subject[:, None] == subject[None, :]
        eye = torch.eye(vectors.shape[0], dtype=torch.bool, device=vectors.device)

        # The positive is keyed on (stimulus, word index, slot) -- which `content_id` already encodes -- and never on
        # a position in this reading's own sub-token sequence, because a third of ZuCo's words are unfixated and two
        # readers' k-th sub-tokens are then different pieces entirely.
        positive = same_word & same_slot & ~same_subject
        anchors = positive.any(dim=1)
        if not bool(anchors.any()):
            return vectors.new_zeros(()), {}

        candidate = ~same_word & ~eye if _EXCLUDE_SAME_WORD else ~(same_word & same_slot) & ~eye
        if same_subject_negatives:
            candidate = candidate & same_subject
        usable_rows = anchors & candidate.any(dim=1)
        if not bool(usable_rows.any()):
            return vectors.new_zeros(()), {}

        logits = (vectors @ vectors.t()) * scale
        neg_inf = torch.finfo(logits.dtype).min
        allowed = candidate | positive
        denom = torch.logsumexp(logits.masked_fill(~allowed, neg_inf), dim=1)
        numer = torch.logsumexp(logits.masked_fill(~positive, neg_inf), dim=1)
        loss = (denom - numer)[usable_rows].mean()

        with torch.no_grad():
            ranked = logits.masked_fill(~allowed, neg_inf).argmax(dim=1)
            hit = positive[torch.arange(len(ranked), device=ranked.device), ranked]
            top1 = float(hit[usable_rows].float().mean())

        return loss, {
            'token_reader_loss': float(loss.detach()),
            'token_reader_top1': top1,
            'token_reader_anchors': float(int(usable_rows.sum())),
        }


def build_token_aligner(config: ObjectiveConfig, hidden_dim: int, text_dim: int) -> TokenAligner | None:
    """Builds the sub-word aligner, or `None` when both its directions are switched off.

    Args:
        config (ObjectiveConfig): The objective configuration (uses the `token_*` knobs).
        hidden_dim (int): Width of the encoder's sub-token hiddens.
        text_dim (int): Width of the frozen sub-word space.

    Returns:
        TokenAligner | None: The aligner, or `None` when it would contribute nothing.
    """
    if config.token_weight <= 0.0 and config.token_reader_weight <= 0.0:
        return None

    aligner = TokenAligner(hidden_dim, text_dim, config.token_sub_tokens, temperature=config.token_temperature)
    _LOG.info(
        'Sub-word alignment on: %d sub-tokens/word, type weight %.3g, reader weight %.3g -> %d-d target.',
        config.token_sub_tokens,
        config.token_weight,
        config.token_reader_weight,
        text_dim,
    )
    return aligner
