"""EEG-OT-CLIP aligner: project ZTE embeddings into a shared text space.

Composes an EEG projector and a text projector with the composite loss
``λ₁ InfoNCE + λ₂ Sinkhorn-OT``. The ZTE encoder itself is typically frozen;
only the projectors (and optionally a light text-side head) are trained.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from zte.decode.config import AlignConfig
from zte.decode.losses import info_nce_loss, sinkhorn_ot_loss
from zte.logging_utils import get_logger

_LOG = get_logger('decode.alignment')

# Re-export losses so callers can ``from zte.decode.alignment import info_nce_loss``.
__all__ = [
    'EEGProjector',
    'OTCLIPAligner',
    'info_nce_loss',
    'sinkhorn_ot_loss',
]


class EEGProjector(nn.Module):
    """MLP projecting embeddings into the shared EEG–text space.

    Architecture: ``Linear → GELU → Linear``, optional ``LayerNorm``, then
    optional L2 normalisation of the output.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        layer_norm: bool = True,
        normalize: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Builds the projector.

        Args:
            in_dim: Input dimensionality.
            hidden_dim: Hidden layer width.
            out_dim: Shared-space dimensionality.
            layer_norm: Apply ``LayerNorm`` after the MLP.
            normalize: L2-normalise the output.
            dropout: Dropout between layers.
        """
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)
        self.normalize = normalize
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        """Projects ``x`` into the shared space.

        Args:
            x: Tensor with last dimension ``in_dim``.

        Returns:
            Projected tensor with last dimension ``out_dim`` (L2-normalised if configured).
        """
        y = self.net(x)
        if self.normalize:
            y = F.normalize(y, dim=-1)
        return y


class OTCLIPAligner(nn.Module):
    """Composite InfoNCE + Sinkhorn OT aligner over paired EEG / text embeddings.

    Attributes:
        config: Alignment hyper-parameters.
        eeg_proj: Projector for ZTE embeddings.
        text_proj: Projector for text-encoder embeddings.
    """

    def __init__(self, config: AlignConfig | None = None) -> None:
        """Initialises both projectors from ``config``.

        Args:
            config: Alignment configuration (defaults to :class:`AlignConfig`).
        """
        super().__init__()
        self.config = config or AlignConfig()
        self.eeg_proj = EEGProjector(
            self.config.eeg_dim,
            self.config.proj_hidden,
            self.config.proj_dim,
        )
        self.text_proj = EEGProjector(
            self.config.text_dim,
            self.config.proj_hidden,
            self.config.proj_dim,
        )
        _LOG.debug(
            'OTCLIPAligner ready | eeg_dim=%d text_dim=%d proj_dim=%d',
            self.config.eeg_dim,
            self.config.text_dim,
            self.config.proj_dim,
        )

    def project(self, eeg_emb: Tensor, text_emb: Tensor) -> tuple[Tensor, Tensor]:
        """Projects both modalities into the shared space.

        Args:
            eeg_emb: EEG embeddings ``(N, eeg_dim)``.
            text_emb: Text embeddings ``(N, text_dim)``.

        Returns:
            ``(eeg_proj, text_proj)`` each ``(N, proj_dim)`` and L2-normalised.
        """
        return self.eeg_proj(eeg_emb), self.text_proj(text_emb)

    def forward(self, eeg_emb: Tensor, text_emb: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Computes the composite alignment loss.

        Args:
            eeg_emb: EEG embeddings ``(N, eeg_dim)``.
            text_emb: Text embeddings ``(N, text_dim)``, paired row-wise with ``eeg_emb``.

        Returns:
            ``(loss, metrics)`` where ``metrics`` includes ``loss_infonce``, ``loss_ot``,
            ``loss`` and ``alignment_diag_mean`` (mean cosine of matched pairs).
        """
        eeg_z, text_z = self.project(eeg_emb, text_emb)
        loss_nce = info_nce_loss(eeg_z, text_z, temperature=self.config.temperature)
        loss_ot = sinkhorn_ot_loss(
            eeg_z,
            text_z,
            epsilon=self.config.ot_epsilon,
            n_iters=self.config.ot_iters,
        )
        loss = self.config.lambda_infonce * loss_nce + self.config.lambda_ot * loss_ot
        with torch.no_grad():
            diag = (eeg_z * text_z).sum(dim=-1).mean()
        metrics: dict[str, float] = {
            'loss': float(loss.detach()),
            'loss_infonce': float(loss_nce.detach()),
            'loss_ot': float(loss_ot.detach()),
            'alignment_diag_mean': float(diag),
        }
        return loss, metrics

    def encode_eeg(self, eeg_emb: Tensor) -> Tensor:
        """Projects EEG embeddings only (for retrieval / decode).

        Args:
            eeg_emb: EEG embeddings ``(N, eeg_dim)``.

        Returns:
            Shared-space embeddings ``(N, proj_dim)``.
        """
        return self.eeg_proj(eeg_emb)

    def encode_text(self, text_emb: Tensor) -> Tensor:
        """Projects text embeddings only.

        Args:
            text_emb: Text embeddings ``(N, text_dim)``.

        Returns:
            Shared-space embeddings ``(N, proj_dim)``.
        """
        return self.text_proj(text_emb)

    def state_dict_for_checkpoint(self) -> dict[str, Any]:
        """Returns a picklable checkpoint payload for the aligner.

        Returns:
            Dict with ``aligner`` weights and ``config``.
        """
        from dataclasses import asdict

        return {
            'aligner': self.state_dict(),
            'config': asdict(self.config),
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        map_location: str | torch.device = 'cpu',
    ) -> OTCLIPAligner:
        """Restores an aligner from a ``.pt`` checkpoint.

        Args:
            path: Path to a checkpoint saved by the alignment trainer.
            map_location: ``torch.load`` map location.

        Returns:
            A weight-loaded :class:`OTCLIPAligner` in eval mode.
        """
        from dataclasses import fields

        from zte.decode.config import AlignConfig as _AlignConfig

        payload = torch.load(path, map_location=map_location, weights_only=False)
        raw_cfg = payload.get('config', {})
        if isinstance(raw_cfg, _AlignConfig):
            config = raw_cfg
        else:
            allowed = {f.name for f in fields(_AlignConfig)}
            config = _AlignConfig(**{k: v for k, v in dict(raw_cfg).items() if k in allowed})
        aligner = cls(config)
        state = payload.get('aligner', payload.get('model', payload))
        aligner.load_state_dict(state)
        aligner.eval()
        return aligner
