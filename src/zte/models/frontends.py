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
from zte.logging_utils import get_logger
from zte.models.spatial import SpatialChannelMixer, resolve_geometry

_LOG = get_logger('models.frontends')


def build_spatial_mixer(
    config: ModelConfig,
    feat_dim: int,
    n_channels: int | None,
    montage_csv: str | None,
) -> SpatialChannelMixer | None:
    """Constructs the electrode spatial-encoding mixer if enabled and applicable.

    Args:
        config (ModelConfig): Model configuration (reads the `spatial_*` fields).
        feat_dim (int): Per-electrode feature width the mixer will operate on (raw `time_steps`, or band-power features per channel).
        n_channels (int | None): EEG channel count; `None` disables spatial encoding (geometry cannot be built).
        montage_csv (str | None): Optional electrode-coordinate CSV for exact geometry.

    Returns:
        SpatialChannelMixer | None: The mixer, or `None` when spatial encoding is off or the channel count is unknown.
    """
    if config.spatial_encoding != 'spherical_harmonics':
        return None
    if not n_channels or n_channels <= 0:
        _LOG.warning(
            'spatial_encoding=%r requested but the channel count is unknown; '
            'disabling electrode positional encoding.',
            config.spatial_encoding,
        )
        return None
    geometry = resolve_geometry(n_channels, montage_csv)
    return SpatialChannelMixer(
        feat_dim=feat_dim,
        geometry=geometry,
        l_max=config.spatial_harmonic_degree,
        n_heads=config.n_heads,
        dropout=config.dropout,
        learnable=config.spatial_encoding_learnable,
        mix=config.spatial_mix,
    )


class BandPowerMLP(nn.Module):
    """MLP token encoder for flattened band-power vectors `(..., in_dim)`.

    When `spatial` is supplied, the leading `n_bp_features * n_channels` band-power columns are reshaped back to electrode tokens
    `(..., n_channels, n_bp_features)`, given their scalp-position encoding and mixed across electrodes, then re-flattened -- so the MLP sees
    geometry-aware channel features. Any trailing (eye-tracking) columns are passed through untouched.

    Attributes:
        out_dim (int): The hidden dimensionality produced for each token.
    """

    def __init__(
        self,
        in_dim: int,
        config: ModelConfig,
        spatial: SpatialChannelMixer | None = None,
        n_channels: int | None = None,
        bp_features_per_channel: int | None = None,
    ) -> None:
        """Initialises the band-power MLP.

        Args:
            in_dim (int): Flattened band-power size `n_features` per token.
            config (ModelConfig): Model configuration (uses `hidden_dim`, `n_layers`, `dropout`).
            spatial (SpatialChannelMixer | None): Optional electrode spatial-encoding mixer.
            n_channels (int | None): EEG channel count (required with `spatial`).
            bp_features_per_channel (int | None): Number of band-power features per channel, i.e. the size of each electrode token (required with
                `spatial`).
        """
        super().__init__()
        self.out_dim = config.hidden_dim
        self.spatial = spatial
        self.n_channels = n_channels
        self.bp_features_per_channel = bp_features_per_channel
        # Number of leading columns that form the (n_bp_features, n_channels) band-power block;
        # any remainder (appended eye-tracking scalars) bypasses the spatial mixer.
        self._channel_block = (
            n_channels * bp_features_per_channel
            if spatial is not None and n_channels and bp_features_per_channel
            else 0
        )
        if spatial is not None and self._channel_block > in_dim:
            _LOG.warning(
                'Spatial band-power block (%d) exceeds in_dim (%d); disabling electrode '
                'positional encoding for this frontend.',
                self._channel_block,
                in_dim,
            )
            self.spatial = None
            self._channel_block = 0
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

    def _apply_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the electrode mixer to the band-power block of a flattened feature vector."""
        # A non-zero channel block guarantees both counts are set (see __init__).
        n_bp = int(self.bp_features_per_channel or 0)
        n_ch = int(self.n_channels or 0)
        block, extra = x[..., : self._channel_block], x[..., self._channel_block :]
        lead = block.shape[:-1]
        # (..., n_bp_features, n_channels) -> electrode tokens (..., n_channels, n_bp_features).
        tokens = block.reshape(*lead, n_bp, n_ch)
        tokens = tokens.transpose(-1, -2)
        tokens = self.spatial(tokens)  # type: ignore[misc]
        block = tokens.transpose(-1, -2).reshape(*lead, self._channel_block)
        return torch.cat([block, extra], dim=-1) if extra.shape[-1] else block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes band-power tokens.

        Args:
            x (torch.Tensor): Tensor `(..., in_dim)`.

        Returns:
            torch.Tensor: `(..., hidden_dim)`.
        """
        if self.spatial is not None:
            x = self._apply_spatial(x)
        return self.net(x)


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
        spatial: SpatialChannelMixer | None = None,
    ) -> None:  # pylint: disable=unused-argument
        """Initialises the raw Conformer frontend.

        Args:
            n_channels (int): EEG channel count.
            time_steps (int): Raw window length (time steps).
            config (ModelConfig): Model configuration
                (uses `conformer_filters`, `conformer_temporal_kernel`, `n_heads`, `n_layers`, `hidden_dim`, `dropout`).
            spatial (SpatialChannelMixer | None): Optional electrode spatial-encoding mixer, applied to `(..., n_channels, time_steps)` before the
                temporal convolution mixes channels.
        """
        super().__init__()
        self.spatial_mixer = spatial
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
    config: ModelConfig,
    in_dim: int | None,
    raw_shape: tuple[int, int] | None,
    *,
    n_channels: int | None = None,
    bp_features_per_channel: int | None = None,
    montage_csv: str | None = None,
) -> nn.Module:
    """Constructs the frontend selected by `config.frontend`.

    Args:
        config (ModelConfig): Model configuration.
        in_dim (int | None): Flattened band-power size (required for `band_power_mlp`).
        raw_shape (tuple[int, int] | None): `(n_channels, time_steps)` raw window shape
            (required for `raw_conformer`).
        n_channels (int | None): EEG channel count, used to build electrode geometry for `spatial_encoding`. For `raw_conformer` this defaults to
            `raw_shape[0]`; for `band_power_mlp` it must be supplied (with `bp_features_per_channel`) for spatial encoding to activate.
        bp_features_per_channel (int | None): Band-power features per channel (the electrode-token width) for `band_power_mlp` spatial encoding.
        montage_csv (str | None): Optional electrode-coordinate CSV (`channel,x,y,z`) for exact scalp geometry.

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
        mixer = build_spatial_mixer(config, bp_features_per_channel or 0, n_channels, montage_csv)
        return BandPowerMLP(
            in_dim,
            config,
            spatial=mixer,
            n_channels=n_channels,
            bp_features_per_channel=bp_features_per_channel,
        )
    if raw_shape is None:
        raise ValueError('raw_conformer frontend requires raw_shape (n_channels, time_steps).')
    raw_channels, time_steps = raw_shape
    mixer = build_spatial_mixer(config, time_steps, n_channels or raw_channels, montage_csv)
    return RawConformer(raw_channels, time_steps, config, spatial=mixer)
