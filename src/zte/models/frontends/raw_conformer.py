"""The raw-EEG Conformer frontend: temporal convolution, pointwise mixing, self-attention, then temporal pooling."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from zte.config import ModelConfig


class RawConformer(nn.Module):
    """Conformer-style token encoder for raw EEG windows `(..., n_channels, time_steps)`.

    Attributes:
        out_dim (int): The hidden dimensionality produced for each token.
    """

    def __init__(
        self,
        n_channels: int,
        time_steps: int,
        config: ModelConfig,
        spatial: nn.Module | None = None,
    ) -> None:
        """Initialises the raw Conformer frontend.

        Args:
            n_channels (int): EEG channel count.
            time_steps (int): Raw window length (time steps).
            config (ModelConfig): Model configuration (uses `conformer_filters`, `conformer_temporal_kernel`, `n_heads`,
                `n_layers`, `hidden_dim`, `dropout`).
            spatial (SpatialChannelMixer | None): Optional electrode spatial-encoding mixer, applied to `(...,
                n_channels, time_steps)` before the temporal convolution mixes channels.
        """
        super().__init__()
        self.spatial_mixer = spatial
        self.grad_checkpoint = bool(config.grad_checkpoint)
        filters = config.conformer_filters
        kernels = tuple(config.conformer_multiscale_kernels) or (config.conformer_temporal_kernel,)

        # Temporal convs act as trainable band-passes: one kernel by default, or a parallel multi-scale bank that
        # spans fast (gamma) to slow (theta) rhythms. A pointwise conv then fuses the scales and mixes the filters.
        self.temporal_scales = nn.ModuleList(
            nn.Conv1d(n_channels, filters, kernel_size=k, padding=k // 2) for k in kernels
        )
        self.fuse = nn.Conv1d(filters * len(kernels), filters, kernel_size=1) if len(kernels) > 1 else nn.Identity()
        self.spatial = nn.Conv1d(filters, filters, kernel_size=1)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(filters)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=filters,
            nhead=_largest_divisor(filters, config.n_heads),
            dim_feedforward=filters * 4,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=max(1, config.n_layers // 2), enable_nested_tensor=False
        )

        # Temporal pooling collapses the time axis: a flat mean, or a learned attentive weighting.
        self.temporal_pool = config.conformer_temporal_pool
        self.attn_pool = nn.Linear(filters, 1) if self.temporal_pool == 'attention' else None
        self.head = nn.Linear(filters, config.hidden_dim)
        self.out_dim = config.hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes raw EEG tokens.

        Args:
            x (torch.Tensor): Tensor `(..., n_channels, time_steps)` with arbitrary leading dims.

        Returns:
            torch.Tensor: `(..., hidden_dim)`.
        """
        if self.spatial_mixer is not None:
            x = self.spatial_mixer(x)
        lead = x.shape[:-2]
        c, t = x.shape[-2:]
        flat = x.reshape(-1, c, t)
        h = self.act(torch.cat([conv(flat) for conv in self.temporal_scales], dim=1))
        h = self.fuse(h)  # Identity for a single scale; fuses the bank otherwise -> (n_tokens, filters, time_steps)
        h = self.act(self.spatial(h))
        h = h.transpose(1, 2)  # (n_tokens, time_steps, filters)
        h = self.norm(h)

        # Self-attention over 350 time steps, repeated for every word of every sentence in the batch.
        # Recomputing it in the backward pass is exact and is what makes a raw batch fit on a 16 GB GPU.
        if self.grad_checkpoint and self.training and h.requires_grad:
            h = checkpoint(self.transformer, h, use_reentrant=False)
        else:
            h = self.transformer(h)  # (n_tokens, time_steps, filters)
        if self.attn_pool is not None:
            # Softmax in float32 so the attention weights stay stable under fp16/bf16 autocast.
            weights = torch.softmax(self.attn_pool(h).float(), dim=1).to(h.dtype)  # (n_tokens, time_steps, 1)
            pooled = (h * weights).sum(dim=1)  # attentive temporal pool -> (n_tokens, filters)
        else:
            pooled = h.mean(dim=1)  # temporal average pool -> (n_tokens, filters)
        out = self.head(pooled)  # (n_tokens, hidden_dim)
        return out.reshape(*lead, self.out_dim)


def _largest_divisor(value: int, target: int) -> int:
    """Returns the largest divisor of `value` that is `<= target` (min 1).

    Keeps `nhead` an exact divisor of the transformer `d_model` even when the configured head count is not.
    """
    for h in range(min(target, value), 0, -1):
        if value % h == 0:
            return h
    return 1
