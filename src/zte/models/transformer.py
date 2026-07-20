"""A pre-norm Transformer encoder with pluggable positional encoding.

`torch.nn.TransformerEncoder` only supports absolute positions added to the inputs; ZuCo sentences vary in length, so this encoder also
offers the relative in-attention schemes `rope` and `alibi` (plus `none` for ablation). The absolute `sinusoidal` / `learned` schemes are
added to the inputs by the caller, leaving attention here position-agnostic. Key-padding and causal (CPC) masks are both honoured.
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
        length (int): Number of positions.
        dim (int): Model dimensionality (encoding width).
        device (torch.device | None): Optional device for the returned tensor.

    Returns:
        torch.Tensor: A `(1, length, dim)` tensor to add to token embeddings.
    """
    pos = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    i = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    div = torch.exp(-math.log(10000.0) * i / dim)
    enc = torch.zeros(length, dim, device=device)
    enc[:, 0::2] = torch.sin(pos * div)
    enc[:, 1::2] = torch.cos(pos * div)
    return enc.unsqueeze(0)


def _alibi_slopes(n_heads: int) -> torch.Tensor:
    """Returns the geometric ALiBi slope per head.

    Args:
        n_heads (int): Number of attention heads.

    Returns:
        torch.Tensor: A `(n_heads,)` tensor of positive slopes.
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


def _rope_cos_sin(
    length: int, head_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputes RoPE cosine/sine tables for a sequence.

    Args:
        length (int): Sequence length.
        head_dim (int): Per-head dimensionality (rotation uses its largest even prefix).
        device (torch.device): Device for the tables.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: `(cos, sin)`, each `(1, 1, seq_len, rot)` for the even rotation width `rot`.
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
        x (torch.Tensor): Query/key tensor `(batch_size, n_heads, seq_len, head_dim)`.
        cos (torch.Tensor): Cosine table `(1, 1, seq_len, rot)`.
        sin (torch.Tensor): Sine table `(1, 1, seq_len, rot)`.

    Returns:
        torch.Tensor: `x` with its first `rot` channels rotated; any odd trailing channel passes through unchanged.
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
        n_heads (int): Number of attention heads.
        head_dim (int): Per-head dimensionality.
        pos (AttnPos): In-attention positional scheme (`'rope'`, `'alibi'` or `'none'`).
    """

    def __init__(self, dim: int, n_heads: int, dropout: float, pos: AttnPos) -> None:
        """Initialises the attention block.

        Args:
            dim (int): Model dimensionality.
            n_heads (int): Head count (must divide `dim`).
            dropout (float): Attention dropout probability.
            pos (AttnPos): In-attention positional scheme.
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

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, causal: bool) -> torch.Tensor:
        """Runs masked self-attention over a token sequence.

        Args:
            x (torch.Tensor): Input `(batch_size, seq_len, hidden_dim)`.
            valid_mask (torch.Tensor): Boolean `(batch_size, seq_len)`; `True` marks attendable key positions.
            causal (bool): If `True`, position `i` may only attend to `j <= i`.

        Returns:
            torch.Tensor: The attended output `(batch_size, seq_len, hidden_dim)`.
        """
        # Project to per-head queries, keys and values.
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
            scores = scores + self.slopes[None, :, None, None] * rel[None, None]  # type: ignore[index]

        # Mask non-attendable keys, then flatten fully-masked rows -- an all-`-inf` row softmaxes to NaN.
        allow = valid_mask[:, None, None, :].expand(b, self.n_heads, length, length)
        if causal:
            causal_ok = torch.tril(torch.ones(length, length, dtype=torch.bool, device=x.device))
            allow = allow & causal_ok[None, None]
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allow, neg_inf)
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
            dim (int): Model dimensionality.
            n_heads (int): Attention head count.
            dropout (float): Dropout probability (attention and residual paths).
            pos (AttnPos): In-attention positional scheme.
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
            x (torch.Tensor): Input `(batch_size, seq_len, hidden_dim)`.
            valid_mask (torch.Tensor): Boolean `(batch_size, seq_len)` of attendable positions.
            causal (bool): Whether to apply a causal mask.

        Returns:
            torch.Tensor: The updated sequence `(batch_size, seq_len, hidden_dim)`.
        """
        x = x + self.drop(self.attn(self.norm1(x), valid_mask, causal))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class ZTETransformerEncoder(nn.Module):
    """A stack of pre-norm encoder layers with a configurable positional scheme.

    Attributes:
        pos_mode (AttnPos): The in-attention positional scheme actually used; absolute schemes are added by the caller.
    """

    def __init__(self, dim: int, n_heads: int, n_layers: int, dropout: float, pos: AttnPos) -> None:
        """Builds the encoder stack.

        Args:
            dim (int): Model dimensionality.
            n_heads (int): Attention head count per layer.
            n_layers (int): Number of stacked layers.
            dropout (float): Dropout probability.
            pos (AttnPos): In-attention positional scheme.
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
            x (torch.Tensor): Input `(batch_size, seq_len, hidden_dim)`.
            valid_mask (torch.Tensor): Boolean `(batch_size, seq_len)`; `True` marks valid (attendable) tokens.
            causal (bool): If `True`, apply a causal attention mask.

        Returns:
            torch.Tensor: The contextualised sequence `(batch_size, seq_len, hidden_dim)`.
        """
        for layer in self.layers:
            x = layer(x, valid_mask, causal)
        return self.norm(x)
