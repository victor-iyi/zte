"""The ZTE encoder: tokens -> (optionally contextual) word/sentence embeddings.

:class:`ZTEModel` is the trainable backbone shared by every objective. It runs a per-token frontend, optionally adds a learned subject embedding, optionally
contextualises tokens with a transformer (bidirectional for masked modelling, causal for CPC), and projects to the `embed_dim` space (768 by default, to
stay plug-compatible with the frozen LLM space used downstream in EEG-OT-CLIP).

The non-contextual path (frontend -> projection) is the word2vec analogue used by the skip-gram/CBOW objectives: each word's embedding depends only on its own EEG,
exactly like a word-embedding lookup depends only on the word.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from zte.config import ModelConfig
from zte.models.frontends import _largest_divisor, build_frontend
from zte.models.heads import ProjectionHead


class AttentionPool(nn.Module):
    """Masked attention pooling from a token sequence to one vector.

    Attributes:
        score: Linear layer producing per-token attention logits.
    """

    def __init__(self, dim: int) -> None:
        """Initialises the pooler.

        Args:
            dim: Token feature dimensionality.
        """
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pools `x` over the sequence axis using masked softmax weights.

        Args:
            x (torch.Tensor): Tensor `(B, L, D)`.
            mask (torch.Tensor): Boolean `(B, L)`; `True` marks valid positions.

        Returns:
            torch.Tensor: Pooled tensor `(B, D)`.
        """
        logits = self.score(x).squeeze(-1)  # (B, L)
        logits = logits.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)  # (B, L, 1)
        return (weights * x).sum(dim=1)


class ZTEModel(nn.Module):
    """ZuCo Thought Embedding encoder.

    Attributes:
        config (ModelConfig): The model configuration.
        hidden_dim (int): Frontend hidden width.
        embed_dim (int): Output embedding dimensionality.
        uses_raw (bool): Whether the frontend consumes raw EEG (vs band power).
    """

    def __init__(
        self,
        config: ModelConfig,
        in_dim: int | None = None,
        raw_shape: tuple[int, int] | None = None,
    ) -> None:
        """Builds the encoder for the configured representation.

        Args:
            config (ModelConfig): Model configuration.
            in_dim (int | None): Flattened band-power size `F*C` (band-power frontend).
            raw_shape (tuple[int, int] | None): `(C, T)` raw window shape (raw frontend).
        """
        super().__init__()
        self.config = config
        self.uses_raw = config.frontend == 'raw_conformer'
        self.frontend = build_frontend(config, in_dim, raw_shape)
        self.hidden_dim = self.frontend.out_dim  # type: ignore[attr-defined]
        self.embed_dim = config.embed_dim

        self.subject_emb = (
            nn.Embedding(config.n_subjects, self.hidden_dim)
            if config.subject_conditioning
            else None
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, self.hidden_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=_largest_divisor(self.hidden_dim, config.n_heads),
            dim_feedforward=self.hidden_dim * 4,  # type: ignore[arg-type]
            dropout=config.dropout,
            batch_first=True,
            activation='gelu',
        )
        # enable_nested_tensor=False keeps the explicit padding/causal mask path
        # active (the nested-tensor fast path is a prototype and warns).
        self.context_encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.projection = ProjectionHead(
            self.hidden_dim, config.projection_hidden, config.embed_dim, config.dropout
        )
        self.pool = AttentionPool(self.hidden_dim) if config.pool == 'attention' else None

    def select_input(self, batch: dict[str, Any]) -> torch.Tensor:
        """Picks the frontend's input tensor from a collated batch.

        Args:
            batch: A batch dict from :func:`~zte.data.torch_dataset.collate_sentences`.

        Returns:
            torch.Tensor: `(B, L, C, T)` raw tensor or `(B, L, D)` band-power tensor.

        Raises:
            ValueError: If the required representation is missing from `batch`.
        """
        key = 'raw' if self.uses_raw else 'features'
        value = batch.get(key)
        if value is None:
            raise ValueError(
                f'Frontend {self.config.frontend!r} needs batch[{key!r}] but it is None. '
                f'Set dataset representation accordingly.'
            )
        return value

    def token_hidden(self, batch: dict[str, Any]) -> torch.Tensor:
        """Runs the frontend (and subject conditioning) to per-token hiddens.

        Args:
            batch(dict[str, Any]): A collated batch dict.

        Returns:
            Tensor (torch.Tensor): `(B, L, hidden_dim)`.

        """
        x = self.select_input(batch)
        hidden = self.frontend(x)  # (B, L, Hd)
        if self.subject_emb is not None:
            hidden = hidden + self.subject_emb(batch['subject']).unsqueeze(1)
        return hidden

    def contextualize(
        self, hidden: torch.Tensor, pad_mask: torch.Tensor, causal: bool = False
    ) -> torch.Tensor:
        """Applies the transformer over the token sequence.

        Args:
            hidden(torch.Tensor): Token hiddens `(B, L, hidden_dim)`.
            pad_mask(torch.Tensor): Boolean `(B, L)`; `True` at valid positions.
            causal(bool): If `True`, apply a causal mask (for CPC).

        Returns:
            Contextualised hiddens (torch.Tensor): `(B, L, hidden_dim)`.

        """
        length = hidden.shape[1]
        hidden = hidden + self.pos_emb[:, :length]
        attn_mask = None
        if causal:
            # Boolean mask (True = disallowed) matches the boolean padding mask
            # dtype, avoiding the "mismatched mask types" deprecation.
            attn_mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=hidden.device), diagonal=1
            )
        return self.context_encoder(hidden, mask=attn_mask, src_key_padding_mask=~pad_mask)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Projects hiddens to the embedding space.

        Args:
            hidden(torch.Tensor): Tensor `(..., hidden_dim)`.

        Returns:
            Tensor (torch.Tensor): `(..., embed_dim)`.
        """
        return self.projection(hidden)

    def forward(
        self, batch: dict[str, Any], contextual: bool = False, causal: bool = False
    ) -> torch.Tensor:
        """Computes token embeddings, optionally contextualised.

        Args:
            batch: A collated batch dict.
            contextual: Run the transformer (masked/CPC) vs per-token (skip-gram).
            causal: Use a causal mask when `contextual` (CPC).

        Returns:
            Token embeddings `(B, L, embed_dim)`.
        """
        hidden = self.token_hidden(batch)
        if contextual:
            hidden = self.contextualize(hidden, batch['pad_mask'], causal=causal)
        return self.project(hidden)

    @torch.no_grad()
    def embed_sentence(self, batch: dict[str, Any]) -> torch.Tensor:
        """Produces one pooled embedding per sentence (for retrieval/inference).

        Args:
            batch: A collated batch dict.

        Returns:
            Sentence embeddings `(B, embed_dim)`.
        """
        # Treat omitted words as non-tokens for both attention and pooling, with a
        # fallback so a sentence with no present words is never fully masked.
        valid = batch['pad_mask'] & batch.get('presence', batch['pad_mask'])
        empty = ~valid.any(dim=1)
        if bool(empty.any()):
            valid = valid.clone()
            valid[empty] = batch['pad_mask'][empty]
        hidden = self.token_hidden(batch)
        hidden = self.contextualize(hidden, valid)
        if self.pool is not None:
            pooled = self.pool(hidden, valid)
        else:
            mask = valid.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.project(pooled)


def build_model(
    config: ModelConfig,
    in_dim: int | None = None,
    raw_shape: tuple[int, int] | None = None,
) -> ZTEModel:
    """Factory that constructs a :class:`ZTEModel` for the given input shapes.

    Args:
        config (ModelConfig): Model configuration.
        in_dim (int | None): Flattened band-power size (band-power frontend).
        raw_shape (tuple[int, int] | None): `(C, T)` raw window shape (raw frontend).

    Returns:
        ZTEModel: An initialised :class:`ZTEModel`.

    """
    return ZTEModel(config, in_dim=in_dim, raw_shape=raw_shape)
