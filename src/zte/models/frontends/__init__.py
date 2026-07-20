"""Per-token EEG frontends, where a token is one word's neural response.

- `zte.models.frontends.band_power` - flat and band-family-routed MLPs over band-power vectors.
- `zte.models.frontends.raw_conformer` - Conformer encoder over raw EEG windows.

`build_frontend` selects the frontend named by `config.frontend` and wires in the electrode positional-encoding mixer when enabled.
"""

from __future__ import annotations

from torch import nn

from zte.config import ModelConfig
from zte.logging_utils import get_logger
from zte.models.frontends.band_power import BandPowerMLP, BandRoutedMLP
from zte.models.frontends.raw_conformer import RawConformer, _largest_divisor
from zte.models.spatial import SpatialAttention, SpatialChannelMixer, resolve_geometry

_LOG = get_logger('models.frontends')

__all__ = [
    'BandPowerMLP',
    'BandRoutedMLP',
    'RawConformer',
    '_largest_divisor',
    'build_frontend',
    'build_spatial_mixer',
]


def build_spatial_mixer(
    config: ModelConfig,
    feat_dim: int,
    n_channels: int | None,
    montage_csv: str | None,
) -> nn.Module | None:
    """Constructs the electrode spatial-encoding mixer if enabled and applicable.

    `spherical_harmonics` (a `SpatialChannelMixer`) and `spatial_attention` (a `SpatialAttention`) share the
    `(..., n_channels, feat_dim) -> same shape` contract. Both need electrode geometry, so both benefit from a real `montage_csv`;
    without one the approximate coordinate-free fallback is used.

    Args:
        config (ModelConfig): Model configuration (reads the `spatial_*` fields).
        feat_dim (int): Per-electrode feature width the mixer will operate on (raw `time_steps`, or band-power features per channel).
        n_channels (int | None): EEG channel count; `None` disables spatial encoding (geometry cannot be built).
        montage_csv (str | None): Optional electrode-coordinate CSV for exact geometry.

    Returns:
        nn.Module | None: The mixer, or `None` when spatial encoding is off or the channel count is unknown.
    """
    if config.spatial_encoding not in {'spherical_harmonics', 'spatial_attention'}:
        return None
    if not n_channels or n_channels <= 0:
        _LOG.warning(
            'spatial_encoding=%r requested but the channel count is unknown; '
            'disabling electrode positional encoding.',
            config.spatial_encoding,
        )
        return None
    geometry = resolve_geometry(n_channels, montage_csv)
    if config.spatial_encoding == 'spatial_attention':
        return SpatialAttention(
            geometry, feat_dim=feat_dim, n_freqs=config.spatial_attn_freqs, dropout=config.dropout
        )
    return SpatialChannelMixer(
        feat_dim=feat_dim,
        geometry=geometry,
        l_max=config.spatial_harmonic_degree,
        n_heads=config.n_heads,
        dropout=config.dropout,
        learnable=config.spatial_encoding_learnable,
        mix=config.spatial_mix,
    )


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
        raw_shape (tuple[int, int] | None): `(n_channels, time_steps)` raw window shape (required for `raw_conformer`).
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
        # Band-family routing needs the (n_bp, n_channels) layout and a whole number of bands.
        if getattr(config, 'band_routing', False):
            from zte.data.schema import BANDS

            if n_channels and bp_features_per_channel and bp_features_per_channel % len(BANDS) == 0:
                return BandRoutedMLP(in_dim, config, n_channels, bp_features_per_channel)
            _LOG.warning(
                'band_routing requested but layout is unknown (bp_per_channel=%s, n_channels=%s); '
                'using the flat band-power MLP.',
                bp_features_per_channel,
                n_channels,
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
