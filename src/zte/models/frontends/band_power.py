"""Band-power token frontends: a flat MLP and a band-family-routed MLP."""

from __future__ import annotations

import torch
from torch import nn

from zte.config import ModelConfig
from zte.logging_utils import get_logger

_LOG = get_logger('models.frontends')


class BandPowerMLP(nn.Module):
    """MLP token encoder for flattened band-power vectors `(..., in_dim)`.

    When `spatial` is supplied, the leading `n_bp_features * n_channels` columns are reshaped into electrode tokens,
    given their scalp-position encoding, mixed across electrodes and re-flattened, so the MLP sees geometry-aware
    channel features. Any trailing eye-tracking columns pass through untouched.

    Attributes:
        out_dim (int): The hidden dimensionality produced for each token.
    """

    def __init__(
        self,
        in_dim: int,
        config: ModelConfig,
        spatial: nn.Module | None = None,
        n_channels: int | None = None,
        bp_features_per_channel: int | None = None,
    ) -> None:
        """Initialises the band-power MLP.

        Args:
            in_dim (int): Flattened band-power size `n_features` per token.
            config (ModelConfig): Model configuration (uses `hidden_dim`, `n_layers`, `dropout`).
            spatial (SpatialChannelMixer | None): Optional electrode spatial-encoding mixer.
            n_channels (int | None): EEG channel count (required with `spatial`).
            bp_features_per_channel (int | None): Band-power features per channel, i.e. the size of each electrode token
                (required with `spatial`).
        """
        super().__init__()
        self.out_dim = config.hidden_dim
        self.spatial = spatial
        self.n_channels = n_channels
        self.bp_features_per_channel = bp_features_per_channel

        # Leading columns forming the (n_bp_features, n_channels) block; any remainder bypasses the spatial mixer.
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


class BandRoutedMLP(nn.Module):
    """Band-family-routed MLP: theta/gamma (content) and alpha/beta (state) get separate paths.

    Each bp-feature of the `(n_bp_features, n_channels)` block carries a band whose family is known
    (`zte.data.schema.BAND_FAMILY`). Theta and gamma carry lexical-semantic load while alpha and beta carry
    attention/arousal state, so separate sub-encoders let invariance pressure fall asymmetrically. Trailing eye-tracking
    columns join the state path; the two hiddens concatenate to `hidden_dim`.

    Attributes:
        out_dim (int): Hidden dimensionality per token.
    """

    def __init__(self, in_dim: int, config: ModelConfig, n_channels: int, bp_features_per_channel: int) -> None:
        """Builds the two band-family pathways.

        Args:
            in_dim (int): Flattened feature width (channel block + any eye-tracking columns).
            config (ModelConfig): Model configuration.
            n_channels (int): EEG channel count.
            bp_features_per_channel (int): Band-power features per channel (measures x bands).
        """
        super().__init__()
        from zte.data.schema import BAND_FAMILY, BANDS

        self.out_dim = config.hidden_dim

        # Split the flattened columns into a content and a state index set.
        n_bp, n_ch = bp_features_per_channel, n_channels
        n_bands = len(BANDS)
        content_fam = {'theta', 'gamma'}
        content_cols: list[int] = []
        state_cols: list[int] = []
        for i in range(n_bp):
            fam = BAND_FAMILY[BANDS[i % n_bands]]  # measures repeat the band order
            cols = list(range(i * n_ch, (i + 1) * n_ch))
            (content_cols if fam in content_fam else state_cols).extend(cols)
        extra = list(range(n_bp * n_ch, in_dim))  # trailing eye-tracking columns -> state path
        state_cols += extra
        self.register_buffer('_content_cols', torch.tensor(content_cols, dtype=torch.long))
        self.register_buffer('_state_cols', torch.tensor(state_cols, dtype=torch.long))

        # Split the hidden width between the two branches.
        c_out = config.hidden_dim - config.hidden_dim // 2
        s_out = config.hidden_dim // 2
        self.content_path = self._branch(len(content_cols), c_out, config)
        self.state_path = self._branch(len(state_cols), s_out, config)

    @staticmethod
    def _branch(in_dim: int, out_dim: int, config: ModelConfig) -> nn.Module:
        """A small LayerNorm -> (Linear + GELU + Dropout)* branch producing `out_dim`."""
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for _ in range(max(1, config.n_layers)):
            layers += [nn.Linear(prev, out_dim), nn.GELU(), nn.Dropout(config.dropout)]
            prev = out_dim
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Routes band families through their paths and concatenates the hiddens."""
        content = self.content_path(x.index_select(-1, self._content_cols))
        state = self.state_path(x.index_select(-1, self._state_cols))
        return torch.cat([content, state], dim=-1)
