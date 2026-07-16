"""The ZTE encoder: tokens -> (optionally contextual) word/sentence embeddings.

`ZTEModel` is the trainable backbone shared by every objective. It runs a per-token frontend, optionally adds a learned subject embedding, optionally
contextualises tokens with a transformer (bidirectional for masked modelling, causal for CPC), and projects to the `embed_dim` space (768 by default, to
stay plug-compatible with the frozen LLM space used downstream in EEG-OT-CLIP).

The non-contextual path (frontend -> projection) is the word2vec analogue used by the skip-gram/CBOW objectives: each word's embedding depends only on its own EEG,
exactly like a word-embedding lookup depends only on the word.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from zte.config import ModelConfig, ObjectiveName
from zte.models.frontends import _largest_divisor, build_frontend
from zte.models.heads import ProjectionHead
from zte.models.transformer import ZTETransformerEncoder, sinusoidal_encoding


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
            x (torch.Tensor): Tensor `(batch_size, seq_len, hidden_dim)`.
            mask (torch.Tensor): Boolean `(batch_size, seq_len)`; `True` marks valid positions.

        Returns:
            torch.Tensor: Pooled tensor `(batch_size, hidden_dim)`.
        """
        logits = self.score(x).squeeze(-1)  # (batch_size, seq_len)
        logits = logits.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)
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
        n_channels: int | None = None,
        bp_features_per_channel: int | None = None,
        montage_csv: str | None = None,
    ) -> None:
        """Builds the encoder for the configured representation.

        Args:
            config (ModelConfig): Model configuration.
            in_dim (int | None): Flattened band-power size `n_features` (band-power frontend).
            raw_shape (tuple[int, int] | None): `(n_channels, time_steps)` raw window shape (raw frontend).
            n_channels (int | None): EEG channel count, used to build electrode geometry for `spatial_encoding`.
            bp_features_per_channel (int | None): Band-power features per channel (electrode-token width) for band-power spatial encoding.
            montage_csv (str | None): Optional electrode-coordinate CSV (`channel,x,y,z`) for exact scalp geometry.
        """
        super().__init__()
        self.config = config
        self.uses_raw = config.frontend == 'raw_conformer'
        self.frontend = build_frontend(
            config,
            in_dim,
            raw_shape,
            n_channels=n_channels,
            bp_features_per_channel=bp_features_per_channel,
            montage_csv=montage_csv,
        )
        self.hidden_dim = self.frontend.out_dim  # type: ignore[attr-defined]
        self.embed_dim = config.embed_dim

        self.subject_emb = (
            nn.Embedding(config.n_subjects, self.hidden_dim)
            if config.subject_conditioning
            else None
        )
        # Report B §3.1: FiLM per-subject conditioning -- a feature-wise affine (gamma, beta) applied
        # to the token hiddens. Zero-initialised, so it starts as the identity and, crucially, stays
        # the identity for any subject id never seen in training (the held-out LOSO subject gets a valid
        # vocab id but its row is never updated). This is the "condition on identity, don't only
        # adversarially remove it" lever (Defossez et al., 2023) made honest for the held-out north-star.
        self.subject_film = (
            nn.Embedding(config.n_subjects, 2 * self.hidden_dim) if config.subject_film else None
        )
        if self.subject_film is not None:
            nn.init.zeros_(self.subject_film.weight)
        # Absolute positional schemes are added to the inputs here; relative schemes
        # (RoPE / ALiBi) are applied inside the encoder's attention.
        self.pos_encoding = config.pos_encoding
        self.pos_emb: nn.Parameter | None = None
        if config.pos_encoding == 'learned':
            self.pos_emb = nn.Parameter(torch.zeros(1, config.max_positions, self.hidden_dim))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        elif config.pos_encoding == 'sinusoidal':
            self.register_buffer(
                'sinusoidal',
                sinusoidal_encoding(config.max_positions, self.hidden_dim),
                persistent=False,
            )
        attn_pos = config.pos_encoding if config.pos_encoding in {'rope', 'alibi'} else 'none'
        self.context_encoder = ZTETransformerEncoder(
            dim=self.hidden_dim,
            n_heads=_largest_divisor(self.hidden_dim, config.n_heads),
            n_layers=config.n_layers,
            dropout=config.dropout,
            pos=attn_pos,
        )
        self.projection = ProjectionHead(
            self.hidden_dim, config.projection_hidden, config.embed_dim, config.dropout
        )
        self.pool = AttentionPool(self.hidden_dim) if config.pool == 'attention' else None

    def select_input(self, batch: dict[str, Any]) -> torch.Tensor:
        """Picks the frontend's input tensor from a collated batch.

        Args:
            batch: A batch dict from `~zte.data.torch_dataset.collate_sentences`.

        Returns:
            torch.Tensor: `(batch_size, seq_len, n_channels, time_steps)` raw tensor or `(batch_size, seq_len, n_features)` band-power tensor.

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
            Tensor (torch.Tensor): `(batch_size, seq_len, hidden_dim)`.

        """
        x = self.select_input(batch)
        hidden = self.frontend(x)  # (batch_size, seq_len, hidden_dim)
        if self.subject_emb is not None:
            hidden = hidden + self.subject_emb(batch['subject']).unsqueeze(1)
        if self.subject_film is not None:
            # Per-subject feature-wise affine, broadcast over the token axis. Zero-init -> gamma=0,
            # beta=0 => identity, so an unseen (held-out) subject id is a no-op, not injected noise.
            gamma, beta = self.subject_film(batch['subject']).chunk(2, dim=-1)
            hidden = (1.0 + gamma).unsqueeze(1) * hidden + beta.unsqueeze(1)
        return hidden

    def contextualize(
        self, hidden: torch.Tensor, pad_mask: torch.Tensor, causal: bool = False
    ) -> torch.Tensor:
        """Applies the transformer over the token sequence.

        Args:
            hidden(torch.Tensor): Token hiddens `(batch_size, seq_len, hidden_dim)`.
            pad_mask(torch.Tensor): Boolean `(batch_size, seq_len)`; `True` at valid positions.
            causal(bool): If `True`, apply a causal mask (for CPC).

        Returns:
            Contextualised hiddens (torch.Tensor): `(batch_size, seq_len, hidden_dim)`.

        """
        length = hidden.shape[1]
        if self.pos_emb is not None:  # learned absolute
            hidden = hidden + self.pos_emb[:, :length]
        elif self.pos_encoding == 'sinusoidal':
            table = self.sinusoidal
            pe = (
                table[:, :length]  # type: ignore[index]
                if length <= table.shape[1]  # type: ignore[index]
                else sinusoidal_encoding(  # type: ignore[attr-defined]
                    length, self.hidden_dim, hidden.device
                )
            )
            hidden = hidden + pe.to(hidden.dtype)
        # RoPE / ALiBi (or none) are handled inside the encoder; pad_mask marks valid
        # (attendable) positions, and `causal` gates future tokens for CPC.
        return self.context_encoder(hidden, pad_mask, causal=causal)

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
            torch.Tensor: Token embeddings `(batch_size, seq_len, embed_dim)`.
        """
        hidden = self.token_hidden(batch)
        if contextual:
            hidden = self.contextualize(hidden, batch['pad_mask'], causal=causal)
        return self.project(hidden)

    def _pool_tokens(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Pools per-token hiddens to one vector per sequence over `valid` positions.

        Args:
            hidden (torch.Tensor): Token hiddens `(batch_size, seq_len, hidden_dim)`.
            valid (torch.Tensor): Boolean `(batch_size, seq_len)`; `True` at poolable positions.

        Returns:
            torch.Tensor: Pooled `(batch_size, hidden_dim)` via attention pooling when configured, else a masked mean.

        """
        if self.pool is not None:
            return self.pool(hidden, valid)
        mask = valid.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)

    @torch.no_grad()
    def embed_sentence(
        self, batch: dict[str, Any], objective: ObjectiveName | None = None
    ) -> torch.Tensor:
        """Produces one pooled embedding per sentence (for retrieval/inference).

        Routing is objective-aware so the exported sentence embedding matches the path the objective actually trained:

        - `skipgram`/`cbow`: per-token frontend -> pool -> project (no transformer, since their contextual path was never in the gradient path).
        - `cpc`: causal-contextual -> pool -> project (CPC trains with a causal mask).
        - `masked` or `None` (back-compat default): bidirectional-contextual -> pool -> project.

        Args:
            batch (dict[str, Any]): A collated batch dict.
            objective (str | None): Trained objective name (`skipgram`|`cbow`|`masked`|`cpc`); `None` keeps the legacy bidirectional-contextual behaviour.

        Returns:
            torch.Tensor: Sentence embeddings `(batch_size, embed_dim)`.

        """
        # Treat omitted words as non-tokens for both attention and pooling, with a
        # fallback so a sentence with no present words is never fully masked.
        valid = batch['pad_mask'] & batch.get('presence', batch['pad_mask'])
        empty = ~valid.any(dim=1)
        if bool(empty.any()):
            valid = valid.clone()
            valid[empty] = batch['pad_mask'][empty]
        hidden = self.token_hidden(batch)
        if objective in {'skipgram', 'cbow'}:
            # Non-contextual path: pool the per-token frontend hiddens directly.
            pooled = self._pool_tokens(hidden, valid)
        elif objective == 'cpc':
            hidden = self.contextualize(hidden, valid, causal=True)
            pooled = self._pool_tokens(hidden, valid)
        else:  # 'masked' or None -> bidirectional-contextual (back-compat default).
            hidden = self.contextualize(hidden, valid, causal=False)
            pooled = self._pool_tokens(hidden, valid)
        return self.project(pooled)


def build_model(
    config: ModelConfig,
    in_dim: int | None = None,
    raw_shape: tuple[int, int] | None = None,
    n_channels: int | None = None,
    bp_features_per_channel: int | None = None,
    montage_csv: str | None = None,
) -> ZTEModel:
    """Factory that constructs a `ZTEModel` for the given input shapes.

    Args:
        config (ModelConfig): Model configuration.
        in_dim (int | None): Flattened band-power size (band-power frontend).
        raw_shape (tuple[int, int] | None): `(n_channels, time_steps)` raw window shape (raw frontend).
        n_channels (int | None): EEG channel count for electrode `spatial_encoding` geometry.
        bp_features_per_channel (int | None): Band-power features per channel (band-power spatial encoding).
        montage_csv (str | None): Optional electrode-coordinate CSV for exact scalp geometry.

    Returns:
        ZTEModel: An initialised `ZTEModel`.

    """
    return ZTEModel(
        config,
        in_dim=in_dim,
        raw_shape=raw_shape,
        n_channels=n_channels,
        bp_features_per_channel=bp_features_per_channel,
        montage_csv=montage_csv,
    )
