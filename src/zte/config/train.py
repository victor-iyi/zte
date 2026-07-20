from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zte.config.types import (
    SchedulerName,
    SplitStrategy,
)


@dataclass
class TrainConfig:
    """Optimisation, scheduling, logging and checkpointing."""

    epochs: int = 20
    """Number of passes over the training split."""

    batch_size: int = 64
    """Sentences (or words) per optimisation step."""

    lr: float = 3e-4
    """Peak learning rate."""

    weight_decay: float = 0.01
    """Weight decay for the AdamW optimizer."""

    warmup_ratio: float = 0.1
    """Fraction of total steps spent linearly warming up."""

    scheduler: SchedulerName = 'cosine'
    """Learning-rate schedule after the warmup period."""

    grad_accum_steps: int = 1
    """Number of micro-batches accumulated per optimiser step."""

    grad_clip: float = 1.0
    """Global gradient-norm clip (`0` disables)."""

    device: Literal['auto', 'cpu', 'cuda', 'mps'] = 'auto'
    """Backend preference passed to `resolve_device`."""

    precision: Literal['auto', 'fp32', 'fp16', 'bf16'] = 'auto'
    """Mixed-precision preference."""

    num_workers: int = 0
    """DataLoader worker processes; negative means auto (see `zte.device.auto_num_workers`)."""

    static_shapes: Literal['auto', 'on', 'off'] = 'auto'
    """Pad every batch to one fixed sentence length instead of the per-batch maximum. `auto` enables it only on
    XLA/TPU, where varying shapes force constant recompilation. Accuracy-neutral (padding is masked everywhere),
    at the cost of extra padded compute."""

    split: SplitStrategy = 'by_sentence'
    """What the train/val/test split groups on; `by_subject_loso` holds out whole subjects."""

    val_fraction: float = 0.1
    """Validation fraction for random/by_sentence splits."""

    test_fraction: float = 0.1
    """Held-out test fraction (`0` disables), so evaluation is never in-sample. For `by_subject_loso` the held-out
    subject is the test set regardless of this value."""

    loso_holdout_subject: str | None = None
    """Subject to hold out for `by_subject_loso`; `None` picks the last subject in the dataset."""

    seed: int = 42
    """Global random number generator seed."""

    deterministic: bool = False
    """Request deterministic cuDNN kernels for byte-for-byte reproducible CUDA runs (slower)."""

    log_every: int = 10
    """Log training metrics every N optimiser steps."""

    eval_every: int = 1
    """Run validation every N epochs."""

    ckpt_dir: str = 'res/checkpoints'
    """Directory to save checkpoints to."""

    ckpt_keep_last: int = 3
    """How many recent checkpoints to retain."""

    tensorboard: bool = False
    """Enable TensorBoard logging if installed."""

    drive_backup_dir: str | None = None
    """Optional Google Drive folder id/path to mirror checkpoints to (see `zte.data.io.remote`)."""

    compile_model: bool = False
    """Apply `torch.compile` (skipped on MPS/CPU)."""
