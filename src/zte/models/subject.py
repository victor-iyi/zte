"""Subject adaptation driven by inferred signal statistics rather than an identity lookup.

A per-subject layer indexed by subject id is inert under leave-one-subject-out: an unseen id has no learned entry, so
the held-out person gets the identity map. Keying the adapter on the person's covariance descriptor instead makes a
stranger a point in signature space rather than a missing row. See `docs/SUBJECT_ALIGNMENT.md`.
"""

from __future__ import annotations

import torch
from torch import nn


class SubjectAdapter(nn.Module):
    """Hypernetwork mapping a subject signature to a spatial gain and a FiLM affine.

    Attributes:
        channel_gain (bool): Whether a per-electrode gain is emitted alongside the FiLM parameters.
    """

    def __init__(
        self,
        signature_dim: int,
        hidden_dim: int,
        n_channels: int | None = None,
        width: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channel_gain = n_channels is not None
        self._n_channels = int(n_channels or 0)
        self._hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(signature_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.head = nn.Linear(width, 2 * hidden_dim + self._n_channels)

        # Zero-init so training starts from the unmodified encoder and learns adaptation as a correction.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, signature: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Emits `(channel_gain, film_gamma, film_beta)` for a batch of signatures `(B, signature_dim)`."""
        params = self.head(self.trunk(signature))
        gamma, beta = (
            params[:, : self._hidden_dim],
            params[:, self._hidden_dim : 2 * self._hidden_dim],
        )
        gain = params[:, 2 * self._hidden_dim :] if self.channel_gain else None
        return gain, gamma, beta

    def apply_spatial(self, x: torch.Tensor, gain: torch.Tensor | None) -> torch.Tensor:
        """Scales each electrode of `(B, L, n_channels, T)` raw windows by that subject's emitted gain."""
        if gain is None:
            return x

        # exp() keeps the gain positive and centred on 1 at zero-init, so the adapter starts as a no-op.
        return x * torch.exp(gain).reshape(gain.shape[0], 1, self._n_channels, 1)

    @staticmethod
    def apply_film(hidden: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Applies the emitted feature-wise affine to token hiddens `(B, L, hidden_dim)`."""
        return (1.0 + gamma).unsqueeze(1) * hidden + beta.unsqueeze(1)
