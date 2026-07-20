"""The raw-EEG Conformer frontend: temporal convolution, pointwise mixing, self-attention over time, then temporal pooling."""

from __future__ import annotations

import torch
from torch import nn

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
    ) -> None:  # pylint: disable=unused-argument
        """Initialises the raw Conformer frontend.

        Args:
            n_channels (int): EEG channel count.
            time_steps (int): Raw window length (time steps).
            config (ModelConfig): Model configuration (uses `conformer_filters`, `conformer_temporal_kernel`, `n_heads`, `n_layers`, `hidden_dim`, `dropout`).
            spatial (SpatialChannelMixer | None): Optional electrode spatial-encoding mixer, applied to `(..., n_channels, time_steps)`
                before the temporal convolution mixes channels.
        """
        super().__init__()
        self.spatial_mixer = spatial
        filters = config.conformer_filters
        kernel = config.conformer_temporal_kernel

        # Temporal conv acts as a trainable band-pass, then a pointwise conv mixes the learned filters.
        self.temporal = nn.Conv1d(n_channels, filters, kernel_size=kernel, padding=kernel // 2)
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
        h = self.act(self.temporal(flat))
        h = self.act(self.spatial(h))  # (n_tokens, filters, time_steps)
        h = h.transpose(1, 2)  # (n_tokens, time_steps, filters)
        h = self.norm(h)
        h = self.transformer(h)  # (n_tokens, time_steps, filters)
        pooled = h.mean(dim=1)  # temporal average pool -> (n_tokens, filters)
        out = self.head(pooled)  # (n_tokens, hidden_dim)
        return out.reshape(*lead, self.out_dim)


def _largest_divisor(value: int, target: int) -> int:
    """Returns the largest divisor of `value` that is `<= target` (min 1).

    Keeps `nhead` an exact divisor of the transformer `d_model` even when the configured head count is not.

    Args:
        value (int): The model dimension to divide.
        target (int): The desired (maximum) head count.

    Returns:
        int: A valid head count dividing `value`.
    """
    for h in range(min(target, value), 0, -1):
        if value % h == 0:
            return h
    return 1
