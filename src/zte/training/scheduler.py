"""Linear warmup followed by cosine, linear or constant learning-rate decay."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from zte.config import SchedulerName


def build_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int,
    kind: SchedulerName = 'cosine',
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Builds a warmup + decay learning-rate scheduler.

    Args:
        optimizer (Optimizer): The optimiser whose LR is scheduled.
        total_steps (int): Total number of optimiser steps over training.
        warmup_steps (int): Steps spent linearly warming up from 0 to the peak LR.
        kind (SchedulerName): Decay shape after warmup (`cosine`, `linear` or `constant`).
        min_lr_ratio (float): Floor for the LR multiplier (as a fraction of peak).

    Returns:
        LambdaLR: A `LambdaLR` scheduler.
    """
    total_steps = max(total_steps, 1)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if kind == 'constant':
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        if kind == 'linear':
            return max(min_lr_ratio, 1.0 - progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)
