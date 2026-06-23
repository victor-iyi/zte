"""Small reusable network blocks: projection head, predictor and EMA teacher.

These are shared across the self-supervised objectives so the projection geometry and the data2vec teacher/predictor machinery live in one place.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class ProjectionHead(nn.Module):
    """A 2-layer MLP that projects encoder hidden states to the embedding space.

    Attributes:
        net (nn.Sequential): The underlying sequential network.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        """Initialises the projection head.

        Args:
            in_dim (int): Input feature dimensionality.
            hidden_dim (int): Hidden layer width.
            out_dim (int): Output embedding dimensionality.
            dropout (float): Dropout probability between layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects `x` to the embedding space.

        Args:
            x (torch.Tensor): Tensor with last dimension `in_dim`.

        Returns:
            torch.Tensor: Tensor with last dimension `out_dim`.
        """
        return self.net(x)


class Predictor(nn.Module):
    """A bottleneck MLP used by data2vec-style latent prediction.

    Attributes:
        net (nn.Sequential): The underlying sequential network.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        """Initialises the predictor.

        Args:
            dim (int): Input/output dimensionality.
            hidden_dim (int | None): Bottleneck width (defaults to `dim`).
        """
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps student latents toward teacher latents.

        Args:
            x (torch.Tensor): Student latent tensor.

        Returns:
            torch.Tensor: The predicted tensor, same shape as `x`.
        """
        return self.net(x)


class EMATeacher:
    """An exponential-moving-average copy of a module (data2vec target network).

    The teacher is not optimised by gradients; its weights track the student so
    it provides stable latent targets that the student predicts.

    Attributes:
        decay (float): EMA decay applied each update.
        module (nn.Module): The frozen teacher module.
    """

    def __init__(self, student: nn.Module, decay: float = 0.999) -> None:
        """Initialises the teacher as a frozen clone of `student`.

        Args:
            student (nn.Module): The module to mirror.
            decay (float): EMA decay in `[0, 1)`; closer to 1 = slower tracking.
        """
        self.decay = decay
        self.module = copy.deepcopy(student)
        self.module.requires_grad_(False)
        self.module.eval()

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        """Moves teacher weights a step toward the student weights.

        Args:
            student (nn.Module): The current student module.
        """
        for teacher_param, student_param in zip(
            self.module.parameters(), student.parameters(), strict=True
        ):
            teacher_param.mul_(self.decay).add_(student_param.detach(), alpha=1 - self.decay)
        for teacher_buf, student_buf in zip(self.module.buffers(), student.buffers(), strict=True):
            teacher_buf.copy_(student_buf)

    def to(self, device: torch.device) -> EMATeacher:
        """Moves the teacher module to `device`.

        Args:
            device (torch.device): Target device.

        Returns:
            EMATeacher: Self.

        """
        self.module.to(device)
        return self
