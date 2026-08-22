"""Word-synchronous lexical evidence: the per-word EEG tokens steer the frozen LM at every decoding step."""

from __future__ import annotations

import torch
from torch import nn

from zte.config import DecoderConfig, EvidenceSchedule
from zte.logging_utils import get_logger

_LOG = get_logger('models.decoder.evidence')

# Below this the pointer window is narrower than one word and the walk stops overlapping anything.
_MIN_WIDTH: float = 0.25


class MonotonicPointer(nn.Module):
    """A soft, left-to-right window over a reading's words, advanced by the decoder's own step count.

    ZuCo's word boundaries come from eye tracking, so the decoder is told -- for free, for every reading, including
    a held-out one -- which stretch of EEG belongs to which word. A pooled prefix throws that away. This walks it: at
    generated token `t` the window sits over word `t * words_per_token`, so the evidence the LM sees at the start of
    a sentence comes from the EEG recorded while the first words were read.

    Note:
        The walk depends only on the step count and the reading's word count, never on the content, so every
        brain-independent control inherits the identical schedule. That is the whole point: word count is worth 5.14
        bits on ZuCo, and a schedule the controls did not get would hand the headline those bits for free.
    """

    rate: torch.Tensor

    def __init__(self, schedule: EvidenceSchedule = 'linear', width: float = 1.5, tokens_per_word: float = 1.4) -> None:
        """Builds the pointer.

        Args:
            schedule (EvidenceSchedule, optional): `'linear'` walks at a constant rate; `'fixation'` weights the walk
                by how long each word was read. Defaults to 'linear'.
            width (float, optional): Gaussian window standard deviation, in words. Defaults to 1.5.
            tokens_per_word (float, optional): LM tokens the walk spends per word. Defaults to 1.4.
        """
        super().__init__()
        self.schedule = schedule
        self.width = max(float(width), _MIN_WIDTH)
        # A persistent buffer, not an attribute: the rate is measured from the training corpus, so a checkpoint that
        # lost it would decode against a different alignment than the one it was trained under.
        self.register_buffer('rate', torch.tensor(max(float(tokens_per_word), 1e-3)))

    @property
    def tokens_per_word(self) -> float:
        """LM tokens the walk spends per word, measured once from the training corpus."""
        return float(self.rate)

    @tokens_per_word.setter
    def tokens_per_word(self, value: float) -> None:
        """Sets the walking rate, floored so the cursor always advances."""
        self.rate.fill_(max(float(value), 1e-3))

    def forward(
        self,
        steps: torch.Tensor,
        valid: torch.Tensor,
        durations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the pointer weights over words for a block of decoding steps.

        Args:
            steps (torch.Tensor): Zero-based generated-token indices `(n_steps,)`.
            valid (torch.Tensor): Boolean `(batch_size, seq_len)`; `True` at readable word positions.
            durations (torch.Tensor | None, optional): Per-word read time `(batch_size, seq_len)`, which makes a
                slowly-read word occupy more decoding steps. Defaults to None, which spaces words evenly.

        Returns:
            torch.Tensor: Weights `(batch_size, n_steps, seq_len)`, summing to 1 over the word axis.
        """
        mask = valid.to(torch.float32)
        n_words = mask.sum(dim=1).clamp_min(1.0)  # (batch_size,)

        # Each word's position on the walk: its index, or its share of the total read time.
        if self.schedule == 'fixation' and durations is not None:
            weight = durations.to(torch.float32).clamp_min(0.0) * mask
            total = weight.sum(dim=1, keepdim=True)
            spacing = torch.where(total > 0, weight / total.clamp_min(1e-6), mask / n_words[:, None])
            centre = torch.cumsum(spacing, dim=1) * n_words[:, None] - 0.5 * spacing * n_words[:, None]
        else:
            centre = torch.arange(mask.shape[1], device=mask.device, dtype=torch.float32).expand_as(mask)

        cursor = steps.to(torch.float32) / self.tokens_per_word  # (n_steps,)
        distance = centre[:, None, :] - cursor[None, :, None]  # (batch_size, n_steps, seq_len)
        logits = -0.5 * (distance / self.width) ** 2

        # A row with no readable word softmaxes an all -inf row to NaN, and `torch.where` would carry that NaN into
        # the backward pass, so the mask is neutralised before the softmax rather than patched after it.
        attendable = valid | ~valid.any(dim=1, keepdim=True)
        logits = logits.masked_fill(~attendable[:, None, :], float('-inf'))
        weights = torch.softmax(logits, dim=-1)

        return weights * valid.any(dim=1)[:, None, None].to(weights.dtype)


class WordEvidence(nn.Module):
    """Turns per-word text-space vectors into a word-synchronous nudge on the frozen LM's output state.

    The nudge is added to the LM's final hidden state, which the frozen output head then reads, so it is exactly a
    rank-limited additive bias on the token logits -- with no new vocabulary parameters, and no second decode path.
    The gate starts at zero, so a run begins as the pooled-prefix decoder and the evidence path only enters the
    output to the extent the loss pays for it.

    Attributes:
        gate (nn.Parameter): Scalar multiplier on the nudge, logged per epoch as `evidence_gate`.
        max_bias (float): Cap on the nudge's norm relative to the hidden state's, so it cannot saturate the softmax.
    """

    def __init__(
        self,
        text_dim: int,
        lm_dim: int,
        *,
        rank: int = 64,
        schedule: EvidenceSchedule = 'linear',
        width: float = 1.5,
        tokens_per_word: float = 1.4,
        gate_init: float = 0.0,
        max_bias: float = 4.0,
    ) -> None:
        """Builds the evidence path.

        Args:
            text_dim (int): Width of the per-word text-space vectors the encoder's lexical head produces.
            lm_dim (int): Frozen-LM hidden width.
            rank (int, optional): Rank of the text-to-LM map, which is the whole trainable width here. Defaults to 64.
            schedule (EvidenceSchedule, optional): Pointer schedule. Defaults to 'linear'.
            width (float, optional): Pointer window width in words. Defaults to 1.5.
            tokens_per_word (float, optional): Pointer advance rate. Defaults to 1.4.
            gate_init (float, optional): Initial gate value. Defaults to 0.0.
            max_bias (float, optional): Cap on the nudge's norm. Defaults to 4.0.
        """
        super().__init__()
        self.pointer = MonotonicPointer(schedule, width, tokens_per_word)
        self.down = nn.Linear(text_dim, rank, bias=False)
        self.up = nn.Linear(rank, lm_dim, bias=False)
        self.norm = nn.LayerNorm(lm_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.max_bias = float(max_bias)
        nn.init.trunc_normal_(self.down.weight, std=0.02)
        nn.init.trunc_normal_(self.up.weight, std=0.02)

    def word_vectors(self, lexical: torch.Tensor) -> torch.Tensor:
        """Maps per-word text-space vectors into the frozen LM's hidden space.

        Args:
            lexical (torch.Tensor): L2-normalised per-word text vectors `(batch_size, seq_len, text_dim)`.

        Returns:
            torch.Tensor: `(batch_size, seq_len, lm_dim)` nudge candidates.
        """
        return self.norm(self.up(self.down(lexical)))

    def nudge(
        self,
        words: torch.Tensor,
        valid: torch.Tensor,
        steps: torch.Tensor,
        durations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the per-step nudge to add to the LM's final hidden state.

        Args:
            words (torch.Tensor): Per-word LM-space vectors `(batch_size, seq_len, lm_dim)` from `word_vectors`.
            valid (torch.Tensor): Boolean `(batch_size, seq_len)`; `True` at readable word positions.
            steps (torch.Tensor): Zero-based generated-token indices `(n_steps,)`.
            durations (torch.Tensor | None, optional): Per-word read time `(batch_size, seq_len)`. Defaults to None.

        Returns:
            torch.Tensor: `(batch_size, n_steps, lm_dim)`, already gated and capped.
        """
        weights = self.pointer(steps, valid, durations)  # (batch_size, n_steps, seq_len)
        pooled = torch.einsum('bsl,bld->bsd', weights, words.to(weights.dtype))
        gated = self.gate * pooled
        scale = gated.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return gated * (scale.clamp(max=self.max_bias) / scale)

    def null(self, words: torch.Tensor) -> torch.Tensor:
        """Returns a content-free stand-in for `words`, keeping the shape and destroying the lexical identity.

        Note:
            This is what the `length_only` control decodes from. The pointer schedule survives -- so the control
            still receives the word count, the 5.14 bits ZuCo gives away -- and only what each word *was* is gone.
            A headline that beats this control beat it on lexical content and on nothing else.
        """
        return torch.zeros_like(words)


def measure_tokens_per_word(target_mask: torch.Tensor, n_words: torch.Tensor) -> float:
    """Returns the mean LM tokens per whitespace word over a corpus, which sets the pointer's walking rate.

    Note:
        Measured rather than configured. The rate is a property of the tokeniser and the corpus, and a hand-set value
        would silently desynchronise the pointer from the text whenever either changed.

    Args:
        target_mask (torch.Tensor): Boolean `(n_sentences, n_target)`; `True` at real target tokens.
        n_words (torch.Tensor): Whitespace word count per sentence `(n_sentences,)`.

    Returns:
        float: Tokens per word, floored at 1.0 -- a tokeniser never emits fewer than one token for a word.
    """
    tokens = float(target_mask.sum().item())
    words = float(n_words.clamp_min(1).sum().item())
    if words <= 0.0:
        return 1.0
    return max(tokens / words, 1.0)


def build_evidence(config: DecoderConfig, text_dim: int, lm_dim: int) -> WordEvidence | None:
    """Constructs the evidence path a decoder configuration asks for, or `None` for the pooled-only decoder.

    Args:
        config (DecoderConfig): Decoder configuration (uses the `evidence_*` fields).
        text_dim (int): Width of the per-word text-space vectors.
        lm_dim (int): Frozen-LM hidden width.

    Returns:
        WordEvidence | None: The evidence path, or `None` when `evidence_schedule='none'`.
    """
    if config.evidence_schedule == 'none':
        return None
    evidence = WordEvidence(
        text_dim,
        lm_dim,
        rank=config.evidence_rank,
        schedule=config.evidence_schedule,
        width=config.evidence_width,
        tokens_per_word=config.evidence_tokens_per_word or 1.4,
        gate_init=config.evidence_gate_init,
        max_bias=config.evidence_max_bias,
    )
    _LOG.info(
        'Word-synchronous evidence: %s schedule, rank %d, %d trainable parameters, gate init %.3f.',
        config.evidence_schedule,
        config.evidence_rank,
        sum(p.numel() for p in evidence.parameters()),
        config.evidence_gate_init,
    )
    return evidence
