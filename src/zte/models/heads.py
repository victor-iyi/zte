"""Small reusable network blocks shared across the objectives: projection head, predictor, EMA teacher and subject adversary."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    """Identity forward; sign-flipped, scaled gradient backward (DANN)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Passes `x` through unchanged, stashing the reversal strength `lambda_`."""
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        """Returns the negated, `lambda_`-scaled gradient for the input."""
        return grad_output.neg() * ctx.lambda_, None


def gradient_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Applies a gradient-reversal layer to `x`.

    Forward is the identity; the backward gradient is negated and scaled by `lambda_`, so an encoder upstream of an adversary head
    trains to fool it -- e.g. to become subject-invariant.

    Args:
        x (torch.Tensor): Any tensor on the encoder's gradient path.
        lambda_ (float): Reversal strength (0 disables the reversal effect).

    Returns:
        torch.Tensor: `x`, unchanged in the forward pass.
    """
    return _GradientReversal.apply(x, lambda_)  # type: ignore[no-any-return]


class SubjectAdversary(nn.Module):
    """A subject classifier trained through a gradient-reversal layer (DANN).

    The head predicts the subject from token hiddens; its input passes through `gradient_reverse`, so the encoder is pushed to strip
    subject identity from its representation.

    Attributes:
        net (nn.Sequential): The subject-classification MLP.
    """

    def __init__(self, in_dim: int, n_subjects: int, hidden_dim: int | None = None) -> None:
        """Initialises the adversary.

        Args:
            in_dim (int): Dimensionality of the hidden representation it reads (the encoder's `hidden_dim`).
            n_subjects (int): Number of subject classes.
            hidden_dim (int | None): Width of the adversary's hidden layer (defaults to `in_dim`).
        """
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_subjects),
        )

    def forward(self, hidden: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        """Predicts subject logits from `hidden` behind a gradient-reversal layer.

        Args:
            hidden (torch.Tensor): Encoder hiddens `(n_tokens, in_dim)`.
            lambda_ (float): Gradient-reversal strength for this step.

        Returns:
            torch.Tensor: Subject logits `(n_tokens, n_subjects)`.
        """
        return self.net(gradient_reverse(hidden, lambda_))


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

    The teacher takes no gradients; its weights track the student, giving stable latent targets for the student to predict.

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
    def update(self, student: nn.Module, decay: float | None = None) -> None:
        """Moves teacher weights a step toward the student weights.

        Args:
            student (nn.Module): The current student module.
            decay (float | None): Override for this step's EMA decay (the data2vec ramp); falls back to `self.decay`.
        """
        d = self.decay if decay is None else decay
        for teacher_param, student_param in zip(
            self.module.parameters(), student.parameters(), strict=True
        ):
            teacher_param.mul_(d).add_(student_param.detach(), alpha=1 - d)
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
