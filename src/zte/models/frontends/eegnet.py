"""The EEGNet and DeepConvNet baselines: the two standard convolutional EEG architectures, as ZTE token frontends."""

from typing import Final

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from zte.config import ModelConfig
from zte.logging_utils import get_logger

_LOG = get_logger('models.frontends')

# Fixed at 16 by the architecture rather than tuned: it is the width EEGNet's depthwise separable filter summarises
# each feature map over, so exposing it as a knob would stop the arm being the published baseline.
_SEPARABLE_KERNEL: Final[int] = 16
"""Length of EEGNet's depthwise separable temporal filter."""

# EEGNet downsamples 4x then 8x, so its flattened head reads `time_steps // 32` points per feature map.
_EEGNET_POOLS: Final[tuple[int, int]] = (4, 8)
"""Average-pool factors of EEGNet's two blocks."""


class _RawConvFrontend(nn.Module):
    """Shared plumbing for the convolutional raw-EEG baselines: leading dims, the spatial mixer and sub-tokens."""

    trunk: nn.Module
    head: nn.Linear

    def __init__(self, config: ModelConfig, spatial: nn.Module | None) -> None:
        """Records the shared state and warns when a second spatial mechanism is stacked in front of the trunk."""
        super().__init__()
        self.spatial_mixer = spatial
        self.grad_checkpoint = bool(config.grad_checkpoint)
        self.out_dim = config.hidden_dim

        if spatial is not None:
            _LOG.warning(
                '%s already filters across electrodes with its own depthwise (n_channels, 1) convolution, and '
                'spatial_encoding=%r puts a second spatial mechanism in front of it. The two are stacked, so read '
                'this run as an explicit double-spatial ablation rather than as the published baseline.',
                type(self).__name__,
                config.spatial_encoding,
            )

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        """Runs the convolution trunk over flattened tokens, giving `(n_tokens, filters, 1, out_steps)`."""
        if self.spatial_mixer is not None:
            x = self.spatial_mixer(x)

        lead = x.shape[:-2]
        c, t = x.shape[-2:]
        flat = x.reshape(-1, 1, c, t)  # Conv2d reads the electrode axis as image height and time as width.

        # A raw batch turns every (sentence, word) pair into its own convolution, so the first block's
        # (filters, n_channels, time_steps) activation is what decides whether the batch fits at all.
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self.trunk, flat, use_reentrant=False), lead

        return self.trunk(flat), lead

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes raw EEG tokens.

        Args:
            x (torch.Tensor): Tensor `(..., n_channels, time_steps)` with arbitrary leading dims.

        Returns:
            torch.Tensor: `(..., hidden_dim)`.
        """
        h, lead = self._features(x)
        out = self.head(h.flatten(1))  # (n_tokens, hidden_dim)

        return out.reshape(*lead, self.out_dim)

    def sub_tokens(self, x: torch.Tensor, n_sub: int) -> torch.Tensor:
        """Encodes each raw EEG window into `n_sub` intra-word sub-tokens instead of one pooled token.

        Note:
            The head is linear over the flattened feature map, so zeroing everything outside a span is exactly that
            head restricted to the span's columns -- every sub-token is read by the same weights as the whole word.
            `n_sub` is a fixed constant, never a per-word count derived from how many word-pieces the reference
            spells that word in, or the EEG encoding would depend on the answer being retrieved.

        Args:
            x (torch.Tensor): Tensor `(..., n_channels, time_steps)` with arbitrary leading dims.
            n_sub (int): Sub-tokens per word; the pooled time axis is split into this many contiguous spans.

        Returns:
            torch.Tensor: `(..., n_sub, hidden_dim)`.

        Raises:
            ValueError: If `n_sub` is not positive.
        """
        if n_sub <= 0:
            raise ValueError(f'n_sub must be positive, got {n_sub}.')

        h, lead = self._features(x)  # (n_tokens, filters, 1, out_steps)
        edges = torch.linspace(0, h.shape[-1], n_sub + 1).round().long().tolist()

        spans: list[torch.Tensor] = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            span = torch.zeros_like(h)
            span[..., lo : max(hi, lo + 1)] = h[..., lo : max(hi, lo + 1)]
            spans.append(self.head(span.flatten(1)))

        out = torch.stack(spans, dim=1)  # (n_tokens, n_sub, hidden_dim)

        return out.reshape(*lead, n_sub, self.out_dim)


class EEGNet(_RawConvFrontend):
    """EEGNet token encoder for raw EEG windows `(..., n_channels, time_steps)`.

    Note:
        The published batch norms are replaced by per-token group norms. This frontend is handed padded word slots as
        all-zero windows and never receives a mask, so batch statistics would be fitted over whatever fraction of the
        batch happens to be padding -- and a word's embedding must not depend on which other words it was batched
        with, or a retrieval gallery would not be reproducible one sentence at a time.

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
        """Initialises the EEGNet frontend.

        Args:
            n_channels (int): EEG channel count.
            time_steps (int): Raw window length (time steps).
            config (ModelConfig): Model configuration (uses `eegnet_f1`, `eegnet_depth`, `eegnet_kernel`,
                `eegnet_dropout`, `hidden_dim`, `grad_checkpoint`).
            spatial (nn.Module | None): Optional electrode spatial-encoding mixer. The depthwise `(n_channels, 1)`
                convolution below is already a spatial filter, so supplying one stacks two spatial mechanisms and is
                logged as a warning.

        Raises:
            ValueError: If `time_steps` is not positive.
        """
        super().__init__(config, spatial)
        if time_steps <= 0:
            raise ValueError(f'eegnet frontend needs a positive raw window, got time_steps={time_steps}.')

        f1 = max(1, config.eegnet_f1)
        f2 = f1 * max(1, config.eegnet_depth)

        # Both pools are clamped to what the window can still divide: at ZuCo's 350-step window they are the published
        # 4 and 8, and a shorter window degrades to a coarser schedule instead of collapsing the time axis to nothing.
        pool_1 = min(_EEGNET_POOLS[0], time_steps)
        after_1 = time_steps // pool_1
        pool_2 = min(_EEGNET_POOLS[1], after_1)
        after_2 = after_1 // pool_2

        self.trunk = nn.Sequential(
            # Temporal conv: F1 trainable band-passes, each reading `eegnet_kernel` samples of every electrode.
            nn.Conv2d(1, f1, (1, max(1, config.eegnet_kernel)), padding='same', bias=False),
            nn.GroupNorm(1, f1),
            # Depthwise spatial conv: `eegnet_depth` scalp projections per band-pass, collapsing the electrode axis.
            nn.Conv2d(f1, f2, (n_channels, 1), groups=f1, bias=False),
            nn.GroupNorm(1, f2),
            nn.ELU(),
            nn.AvgPool2d((1, pool_1)),
            nn.Dropout(config.eegnet_dropout),
            # Separable conv: one depthwise temporal filter per feature map, then a pointwise mix across them.
            nn.Conv2d(f2, f2, (1, _SEPARABLE_KERNEL), padding='same', groups=f2, bias=False),
            nn.Conv2d(f2, f2, (1, 1), bias=False),
            nn.GroupNorm(1, f2),
            nn.ELU(),
            nn.AvgPool2d((1, pool_2)),
            nn.Dropout(config.eegnet_dropout),
        )
        self.head = nn.Linear(f2 * after_2, config.hidden_dim)


class DeepConvNet(_RawConvFrontend):
    """DeepConvNet token encoder for raw EEG windows `(..., n_channels, time_steps)`.

    Note:
        The published batch norms are replaced by per-token group norms, for the same reason as `EEGNet`: the frontend
        sees padded word slots as all-zero windows and no mask, so batch statistics would be fitted over the padding.

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
        """Initialises the DeepConvNet frontend.

        Args:
            n_channels (int): EEG channel count.
            time_steps (int): Raw window length (time steps).
            config (ModelConfig): Model configuration (uses `deepconv_filters`, `deepconv_kernel`, `deepconv_pool`,
                `deepconv_dropout`, `hidden_dim`, `grad_checkpoint`).
            spatial (nn.Module | None): Optional electrode spatial-encoding mixer; stacking one on the first block's
                `(n_channels, 1)` convolution is logged as a warning.

        Raises:
            ValueError: If `time_steps` is not positive, or too short for one block per entry of `deepconv_filters`.
        """
        super().__init__(config, spatial)
        if time_steps <= 0:
            raise ValueError(f'deep_conv_net frontend needs a positive raw window, got time_steps={time_steps}.')

        filters = tuple(config.deepconv_filters) or (25,)
        kernel = max(1, config.deepconv_kernel)
        pools, out_steps = _deepconv_schedule(time_steps, len(filters), kernel, max(1, config.deepconv_pool))

        # Block 1 is the only one that sees the electrode axis: its (n_channels, 1) conv collapses it to height 1.
        layers: list[nn.Module] = [
            nn.Conv2d(1, filters[0], (1, kernel), bias=False),
            nn.Conv2d(filters[0], filters[0], (n_channels, 1), bias=False),
            nn.GroupNorm(1, filters[0]),
            nn.ELU(),
            nn.MaxPool2d((1, pools[0]), stride=(1, pools[0])),
        ]
        prev = filters[0]
        for width, step in zip(filters[1:], pools[1:], strict=True):
            layers += [
                nn.Dropout(config.deepconv_dropout),
                nn.Conv2d(prev, width, (1, kernel), bias=False),
                nn.GroupNorm(1, width),
                nn.ELU(),
                nn.MaxPool2d((1, step), stride=(1, step)),
            ]
            prev = width

        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev * out_steps, config.hidden_dim)


def _deepconv_schedule(time_steps: int, n_blocks: int, kernel: int, pool: int) -> tuple[list[int], int]:
    """Returns the per-block pool factor and the surviving time-axis length, or raises if the window is too short."""
    pools: list[int] = []
    length = time_steps

    for block in range(n_blocks):
        if length < kernel:
            raise ValueError(
                f'deep_conv_net cannot run {n_blocks} blocks over a {time_steps}-step raw window: block '
                f'{block + 1} is left with a length-{length} time axis but its convolution consumes {kernel} '
                f'steps. Raise dataset.raw_window, or shorten model.deepconv_filters to {block} entries.'
            )
        length -= kernel - 1

        # The published schedule pools by 3 every block, which is tuned for 1000+ samples; clamping keeps a 350- or
        # 128-step ZuCo window from being pooled down to a zero-length axis.
        step = min(pool, length)
        pools.append(step)
        length //= step

    return pools, length
