"""The ZTE encoder: tokens -> (optionally contextual) word/sentence embeddings.

`ZTEModel` is the trainable backbone shared by every objective: a per-token frontend, optional subject conditioning, an
optional transformer (bidirectional for masked modelling, causal for CPC), and a projection to `embed_dim`. Skipping the
transformer gives the word2vec analogue used by skip-gram/CBOW, where a word's embedding depends only on its own EEG.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from zte.config import ModelConfig, ObjectiveName
from zte.models.frontends import _largest_divisor, build_frontend
from zte.models.heads import ProjectionHead
from zte.models.subject import SubjectAdapter
from zte.models.transformer import ZTETransformerEncoder, sinusoidal_encoding


class AttentionPool(nn.Module):
    """Masked attention pooling from a token sequence to one vector.

    Attributes:
        score (nn.Linear): Linear layer producing per-token attention logits.
    """

    def __init__(self, dim: int) -> None:
        """Initialises the pooler.

        Args:
            dim (int): Token feature dimensionality.
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
        signature_dim: int = 0,
    ) -> None:
        """Builds the encoder for the configured representation.

        Args:
            config (ModelConfig): Model configuration.
            in_dim (int | None): Flattened band-power size `n_features` (band-power frontend).
            raw_shape (tuple[int, int] | None): `(n_channels, time_steps)` raw window shape (raw frontend).
            n_channels (int | None): EEG channel count, used to build electrode geometry for `spatial_encoding`.
            bp_features_per_channel (int | None): Band-power features per channel (electrode-token width) for band-power
                spatial encoding.
            montage_csv (str | None): Optional electrode-coordinate CSV (`channel,x,y,z`) for exact scalp geometry.
            signature_dim (int): Width of the subject signature; 0 disables the subject adapter.
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

        self.subject_emb = nn.Embedding(config.n_subjects, self.hidden_dim) if config.subject_conditioning else None
        # Per-subject FiLM affine, zero-initialised so an unseen (held-out LOSO) subject id stays the identity.
        film_dim = 2 * self.hidden_dim  # type: ignore[operator]
        self.subject_film = nn.Embedding(config.n_subjects, film_dim) if config.subject_film else None
        if self.subject_film is not None:
            nn.init.zeros_(self.subject_film.weight)

        # Id-free replacement for the two tables above: an unseen subject is adapted from its own statistics.
        self.subject_adapter: SubjectAdapter | None = None
        if config.subject_adapter and signature_dim:
            self.subject_adapter = SubjectAdapter(
                signature_dim,
                self.hidden_dim,
                n_channels=n_channels if (self.uses_raw and config.subject_adapter_spatial) else None,
                width=config.subject_adapter_width,
                dropout=config.dropout,
            )

        # Absolute schemes are added to the inputs here; RoPE / ALiBi act inside the encoder's attention.
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
        self.projection = ProjectionHead(self.hidden_dim, config.projection_hidden, config.embed_dim, config.dropout)
        self.pool = AttentionPool(self.hidden_dim) if config.pool == 'attention' else None

    def select_input(self, batch: dict[str, Any]) -> torch.Tensor:
        """Picks the frontend's input tensor from a collated batch.

        Args:
            batch (dict[str, Any]): A batch dict from `collate_sentences`.

        Returns:
            torch.Tensor: `(batch_size, seq_len, n_channels, time_steps)` raw tensor or `(batch_size, seq_len,
                n_features)` band-power tensor.

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
            batch (dict[str, Any]): A collated batch dict.

        Returns:
            torch.Tensor: `(batch_size, seq_len, hidden_dim)`.
        """
        x = self.select_input(batch)

        # One adapter pass per batch: the spatial gain re-weights electrodes before the frontend, the FiLM after.
        signature = batch.get('subject_signature')
        adapter_params = None
        if self.subject_adapter is not None and signature is not None:
            adapter_params = self.subject_adapter(signature)
            if self.uses_raw:
                x = self.subject_adapter.apply_spatial(x, adapter_params[0])

        hidden = self.frontend(x)  # (batch_size, seq_len, hidden_dim)
        if adapter_params is not None:
            hidden = SubjectAdapter.apply_film(hidden, adapter_params[1], adapter_params[2])
        if self.subject_emb is not None:
            hidden = hidden + self.subject_emb(batch['subject']).unsqueeze(1)
        if self.subject_film is not None:
            # Feature-wise affine broadcast over the token axis; zero-init makes an unseen subject a no-op.
            gamma, beta = self.subject_film(batch['subject']).chunk(2, dim=-1)
            hidden = (1.0 + gamma).unsqueeze(1) * hidden + beta.unsqueeze(1)
        return hidden

    def contextualize(self, hidden: torch.Tensor, pad_mask: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """Applies the transformer over the token sequence.

        Args:
            hidden (torch.Tensor): Token hiddens `(batch_size, seq_len, hidden_dim)`.
            pad_mask (torch.Tensor): Boolean `(batch_size, seq_len)`; `True` at valid positions.
            causal (bool): If `True`, apply a causal mask (for CPC).

        Returns:
            torch.Tensor: Contextualised hiddens `(batch_size, seq_len, hidden_dim)`.
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
        return self.context_encoder(hidden, pad_mask, causal=causal)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Projects hiddens to the embedding space.

        Args:
            hidden (torch.Tensor): Tensor `(..., hidden_dim)`.

        Returns:
            torch.Tensor: Tensor `(..., embed_dim)`.
        """
        return self.projection(hidden)

    def forward(self, batch: dict[str, Any], contextual: bool = False, causal: bool = False) -> torch.Tensor:
        """Computes token embeddings, optionally contextualised.

        Args:
            batch (dict[str, Any]): A collated batch dict.
            contextual (bool): Run the transformer (masked/CPC) vs per-token (skip-gram).
            causal (bool): Use a causal mask when `contextual` (CPC).

        Returns:
            torch.Tensor: Token embeddings `(batch_size, seq_len, embed_dim)`.
        """
        hidden = self.token_hidden(batch)
        if contextual:
            hidden = self.contextualize(hidden, batch['pad_mask'], causal=causal)
        return self.project(hidden)

    def _pool_tokens(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Pools per-token hiddens to one vector per sequence over `valid` positions.

        Returns:
            torch.Tensor: Pooled `(batch_size, hidden_dim)` via attention pooling when configured, else a masked mean.
        """
        if self.pool is not None:
            return self.pool(hidden, valid)
        mask = valid.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)

    def pooling_mask(self, batch: dict[str, Any]) -> torch.Tensor:
        """Returns the `(batch_size, seq_len)` mask of positions a sentence may attend to and pool over.

        Omitted words are non-tokens for attention and pooling; a sentence with none left falls back to its padding
        mask, because `AttentionPool` returns NaN for a fully masked row.

        Args:
            batch (dict[str, Any]): A collated batch dict.

        Returns:
            torch.Tensor: Boolean `(batch_size, seq_len)`; `True` at usable positions.
        """
        valid = batch['pad_mask'] & batch.get('presence', batch['pad_mask'])
        empty = ~valid.any(dim=1)
        if bool(empty.any()):
            valid = valid.clone()
            valid[empty] = batch['pad_mask'][empty]
        return valid

    def sentence_hidden(self, batch: dict[str, Any], contextual: bool = True, causal: bool = False) -> torch.Tensor:
        """Pools a sentence's word-EEG tokens into one hidden vector, keeping the gradient.

        This is the differentiable half of `embed_sentence`, which a decoder loss needs and cannot get from that
        `@torch.no_grad()` method.

        Args:
            batch (dict[str, Any]): A collated batch dict.
            contextual (bool, optional): Run the transformer before pooling. Defaults to True.
            causal (bool, optional): Use a causal mask when `contextual`. Defaults to False.

        Returns:
            torch.Tensor: Pooled sentence hiddens `(batch_size, hidden_dim)`, before the projection head.
        """
        valid = self.pooling_mask(batch)
        hidden = self.token_hidden(batch)
        if contextual:
            hidden = self.contextualize(hidden, valid, causal=causal)
        return self._pool_tokens(hidden, valid)

    @torch.no_grad()
    def embed_sentence(self, batch: dict[str, Any], objective: ObjectiveName | None = None) -> torch.Tensor:
        """Produces one pooled embedding per sentence (for retrieval/inference).

        Routing is objective-aware so the exported sentence embedding matches the path the objective actually trained:

        - `skipgram`/`cbow`: per-token frontend -> pool -> project (their contextual path never sees a gradient).
        - `cpc`: causal-contextual -> pool -> project (CPC trains with a causal mask).
        - `masked` or `None`: bidirectional-contextual -> pool -> project.

        Args:
            batch (dict[str, Any]): A collated batch dict.
            objective (ObjectiveName | None): Trained objective name; `None` defaults to the bidirectional-contextual
                path.

        Returns:
            torch.Tensor: Sentence embeddings `(batch_size, embed_dim)`.
        """
        contextual = objective not in {'skipgram', 'cbow'}
        pooled = self.sentence_hidden(batch, contextual=contextual, causal=objective == 'cpc')
        return self.project(pooled)


def build_model(
    config: ModelConfig,
    in_dim: int | None = None,
    raw_shape: tuple[int, int] | None = None,
    n_channels: int | None = None,
    bp_features_per_channel: int | None = None,
    montage_csv: str | None = None,
    signature_dim: int = 0,
) -> ZTEModel:
    """Factory that constructs a `ZTEModel` for the given input shapes.

    Args:
        config (ModelConfig): Model configuration.
        in_dim (int | None): Flattened band-power size (band-power frontend).
        raw_shape (tuple[int, int] | None): `(n_channels, time_steps)` raw window shape (raw frontend).
        n_channels (int | None): EEG channel count for electrode `spatial_encoding` geometry.
        bp_features_per_channel (int | None): Band-power features per channel (band-power spatial encoding).
        montage_csv (str | None): Optional electrode-coordinate CSV for exact scalp geometry.
        signature_dim (int): Width of the subject signature; 0 disables the subject adapter.

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
        signature_dim=signature_dim,
    )
