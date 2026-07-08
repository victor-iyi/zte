"""Per-token EEG frontends: a band-power MLP and a raw-EEG Conformer.

A *token* is one word's neural response. A frontend maps that token to a hidden vector, independent of its neighbours, so the sequence/objective layers can treat
words like tokens in a language model. Both frontends accept an arbitrary number of leading (batch/sequence) dimensions and only touch the trailing feature axes.

The `RawConformer` follows the project's EEG-Conformer recipe: a temporal 1-D convolution (a learnable band-pass),
a spatial/pointwise mixer, a small self-attention stack over time, and temporal pooling to a fixed hidden vector.
"""

from __future__ import annotations

import torch
from torch import nn

from zte.config import ModelConfig


class BandPowerMLP(nn.Module):
    """MLP token encoder for flattened band-power vectors `(..., in_dim)`.

    Attributes:
        out_dim (int): The hidden dimensionality produced for each token.
    """

    def __init__(self, in_dim: int, config: ModelConfig) -> None:
        """Initialises the band-power MLP.

        Args:
            in_dim (int): Flattened band-power size `n_features` per token.
            config (ModelConfig): Model configuration (uses `hidden_dim`, `n_layers`, `dropout`).
        """
        super().__init__()
        self.out_dim = config.hidden_dim
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for _ in range(max(1, config.n_layers)):
            layers += [
                nn.Linear(prev, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ]
            prev = config.hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes band-power tokens.

        Args:
            x (torch.Tensor): Tensor `(..., in_dim)`.

        Returns:
            torch.Tensor: `(..., hidden_dim)`.
        """
        return self.net(x)


class RawConformer(nn.Module):
    """Conformer-style token encoder for raw EEG windows `(..., n_channels, time_steps)`.

    Attributes:
        out_dim (int): The hidden dimensionality produced for each token.
    """

    def __init__(self, n_channels: int, time_steps: int, config: ModelConfig) -> None:  # pylint: disable=unused-argument
        """Initialises the raw Conformer frontend.

        Args:
            n_channels (int): EEG channel count.
            time_steps (int): Raw window length (time steps).
            config (ModelConfig): Model configuration
                (uses `conformer_filters`, `conformer_temporal_kernel`, `n_heads`, `n_layers`, `hidden_dim`, `dropout`).
        """
        super().__init__()
        filters = config.conformer_filters
        kernel = config.conformer_temporal_kernel
        # Temporal conv acts as a trainable band-pass across the time axis.
        self.temporal = nn.Conv1d(n_channels, filters, kernel_size=kernel, padding=kernel // 2)
        # Pointwise/spatial mixing of the learned temporal filters.
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

    Ensures `nhead` evenly divides the transformer `d_model` even when the
    configured head count does not.

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


def build_frontend(
    config: ModelConfig, in_dim: int | None, raw_shape: tuple[int, int] | None
) -> nn.Module:
    """Constructs the frontend selected by `config.frontend`.

    Args:
        config (ModelConfig): Model configuration.
        in_dim (int | None): Flattened band-power size (required for `band_power_mlp`).
        raw_shape (tuple[int, int] | None): `(n_channels, time_steps)` raw window shape
            (required for `raw_conformer`).

    Returns:
        nn.Module: An initialised frontend module exposing `out_dim`.

    Raises:
        ValueError: If the dimensions required by the chosen frontend are absent.

    """
    if config.frontend == 'band_power_mlp':
        if in_dim is None:
            raise ValueError(
                'band_power_mlp frontend requires in_dim (n_bp_features * n_channels).'
            )
        return BandPowerMLP(in_dim, config)
    if raw_shape is None:
        raise ValueError('raw_conformer frontend requires raw_shape (n_channels, time_steps).')
    return RawConformer(raw_shape[0], raw_shape[1], config)
