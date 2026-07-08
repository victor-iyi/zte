"""A pre-norm Transformer encoder with pluggable positional encoding.

The stock :class:`torch.nn.TransformerEncoder` only supports *absolute* position
information added to the inputs. Modern sequence models favour **relative** schemes
applied inside attention -- rotary embeddings (RoPE) and ALiBi -- which generalise
to unseen sequence lengths and encode *distance* directly. Since ZuCo sentences vary
in length and the north-star goal is a length-/device-agnostic thought code, this
module implements a small, explicit encoder supporting all of:

* `rope` -- rotary position embedding rotated into the query/key subspaces.
* `alibi` -- linear, head-specific distance penalties added to attention logits.
* `sinusoidal` / `learned` -- classic absolute encodings (added to inputs by the
  caller); attention here is then position-agnostic.
* `none` -- no positional signal (ablation).

The encoder is pre-norm (stable training), GELU-activated, and honours both a
key-padding mask (variable-length sentences) and an optional causal mask (for CPC),
matching the interface the ZTE objectives already rely on.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

AttnPos = Literal['rope', 'alibi', 'none']


def sinusoidal_encoding(length: int, dim: int, device: torch.device | None = None) -> torch.Tensor:
    """Builds the classic fixed sinusoidal positional encoding.

    Args:
        length: Number of positions.
        dim: Model dimensionality (encoding width).
        device: Optional device for the returned tensor.

    Returns:
        A `(1, length, dim)` tensor to add to token embeddings, where `dim` is the
        model/encoding width.
    """
    pos = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    i = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    div = torch.exp(-math.log(10000.0) * i / dim)
    enc = torch.zeros(length, dim, device=device)
    enc[:, 0::2] = torch.sin(pos * div)
    enc[:, 1::2] = torch.cos(pos * div)
    return enc.unsqueeze(0)


def _alibi_slopes(n_heads: int) -> torch.Tensor:
    """Returns the geometric ALiBi slope per head (Press et al., 2022).

    Args:
        n_heads: Number of attention heads.

    Returns:
        A `(n_heads,)` tensor of positive slopes.
    """
    def pow2_slopes(n: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        return [start ** (i + 1) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = pow2_slopes(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = pow2_slopes(closest)
        extra = pow2_slopes(2 * closest)[0::2][: n_heads - closest]
        slopes = slopes + extra
    return torch.tensor(slopes, dtype=torch.float32)


def _rope_cos_sin(length: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputes RoPE cosine/sine tables for a sequence.

    Args:
        length: Sequence length.
        head_dim: Per-head dimensionality (rotation uses its largest even prefix).
        device: Device for the tables.

    Returns:
        `(cos, sin)` each shaped `(1, 1, seq_len, rot)` where `rot` is the even
        rotation width.
    """
    rot = head_dim - (head_dim % 2)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rot, 2, device=device).float() / rot))
    t = torch.arange(length, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, rot/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, rot)
    return emb.cos()[None, None], emb.sin()[None, None]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotates the even-dimensional prefix of `x` by the RoPE angles.

    Args:
        x: Query/key tensor `(batch_size, n_heads, seq_len, head_dim)`.
        cos: Cosine table `(1, 1, seq_len, rot)`.
        sin: Sine table `(1, 1, seq_len, rot)`.

    Returns:
        `x` with its first `rot` channels rotated; any odd trailing channel is
        passed through unchanged.
    """
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    half = rot // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    x_rot = x_rot * cos + rotated * sin
    return torch.cat([x_rot, x_pass], dim=-1)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE or ALiBi and masking.

    Attributes:
        n_heads: Number of attention heads.
        head_dim: Per-head dimensionality.
        pos: In-attention positional scheme (`'rope'`, `'alibi'` or `'none'`).
    """

    def __init__(self, dim: int, n_heads: int, dropout: float, pos: AttnPos) -> None:
        """Initialises the attention block.

        Args:
            dim: Model dimensionality.
            n_heads: Head count (must divide `dim`).
            dropout: Attention dropout probability.
            pos: In-attention positional scheme.
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.pos = pos
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout
        if pos == 'alibi':
            self.register_buffer('slopes', _alibi_slopes(n_heads), persistent=False)

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor, causal: bool
    ) -> torch.Tensor:
        """Runs masked self-attention over a token sequence.

        Args:
            x: Input `(batch_size, seq_len, hidden_dim)`.
            valid_mask: Boolean `(batch_size, seq_len)`; `True` marks attendable key positions.
            causal: If `True`, position `i` may only attend to `j <= i`.

        Returns:
            The attended output `(batch_size, seq_len, hidden_dim)`.
        """
        b, length, dim = x.shape
        qkv = self.qkv(x).reshape(b, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (batch_size, n_heads, seq_len, head_dim)

        if self.pos == 'rope':
            cos, sin = _rope_cos_sin(length, self.head_dim, x.device)
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        # scores: (batch_size, n_heads, seq_len, seq_len)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.pos == 'alibi':
            dist = torch.arange(length, device=x.device)
            rel = -(dist[None, :] - dist[:, None]).abs().float()  # (seq_len, seq_len), <= 0
            scores = scores + self.slopes[None, :, None, None] * rel[None, None]

        allow = valid_mask[:, None, None, :].expand(b, self.n_heads, length, length)
        if causal:
            causal_ok = torch.tril(torch.ones(length, length, dtype=torch.bool, device=x.device))
            allow = allow & causal_ok[None, None]
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allow, neg_inf)
        # A query row with no attendable key would softmax to NaN; make it uniform
        # (its output is masked out downstream anyway).
        dead = ~allow.any(dim=-1, keepdim=True)
        scores = torch.where(dead, torch.zeros_like(scores), scores)

        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = attn @ v  # (batch_size, n_heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(b, length, dim)
        return self.out(out)


class EncoderLayer(nn.Module):
    """A single pre-norm Transformer encoder block (attention + MLP)."""

    def __init__(self, dim: int, n_heads: int, dropout: float, pos: AttnPos) -> None:
        """Initialises the block.

        Args:
            dim: Model dimensionality.
            n_heads: Attention head count.
            dropout: Dropout probability (attention and residual paths).
            pos: In-attention positional scheme.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, dropout, pos)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, causal: bool) -> torch.Tensor:
        """Applies attention and the feed-forward block with residuals.

        Args:
            x: Input `(batch_size, seq_len, hidden_dim)`.
            valid_mask: Boolean `(batch_size, seq_len)` of attendable positions.
            causal: Whether to apply a causal mask.

        Returns:
            The updated sequence `(batch_size, seq_len, hidden_dim)`.
        """
        x = x + self.drop(self.attn(self.norm1(x), valid_mask, causal))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class ZTETransformerEncoder(nn.Module):
    """A stack of pre-norm encoder layers with a configurable positional scheme.

    Attributes:
        pos_mode: The in-attention positional scheme actually used (`'rope'`,
            `'alibi'` or `'none'` -- absolute schemes are added by the caller).
    """

    def __init__(
        self, dim: int, n_heads: int, n_layers: int, dropout: float, pos: AttnPos
    ) -> None:
        """Builds the encoder stack.

        Args:
            dim: Model dimensionality.
            n_heads: Attention head count per layer.
            n_layers: Number of stacked layers.
            dropout: Dropout probability.
            pos: In-attention positional scheme.
        """
        super().__init__()
        self.pos_mode = pos
        self.layers = nn.ModuleList(
            EncoderLayer(dim, n_heads, dropout, pos) for _ in range(max(1, n_layers))
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor, causal: bool = False
    ) -> torch.Tensor:
        """Encodes a padded token sequence.

        Args:
            x: Input `(batch_size, seq_len, hidden_dim)`.
            valid_mask: Boolean `(batch_size, seq_len)`; `True` marks valid (attendable) tokens.
            causal: If `True`, apply a causal attention mask.

        Returns:
            The contextualised sequence `(batch_size, seq_len, hidden_dim)`.
        """
        for layer in self.layers:
            x = layer(x, valid_mask, causal)
        return self.norm(x)
