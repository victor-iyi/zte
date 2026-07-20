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
    """DataLoader worker processes. `-1` (or any negative) means **auto** — a few workers on an accelerator
    (CUDA/MPS/TPU), single-process on CPU (see `zte.device.auto_num_workers`)."""

    static_shapes: Literal['auto', 'on', 'off'] = 'auto'
    """Whether to pad every batch to a single fixed sentence length instead of the per-batch maximum.  `auto` enables it
    **only on XLA/TPU**, where varying tensor shapes force constant recompilation; `on`/`off` force it.
    It is accuracy-neutral — the padded positions are masked out of attention, pooling and the loss — at the cost of some extra padded compute,
    which is the right trade only on TPU No effect on CUDA/MPS/CPU under `auto`."""

    split: SplitStrategy = 'by_sentence'
    """The strategy to use for splitting the dataset into training/validation/test sets. `random` splits randomly;
    `by_sentence` splits by sentence; `by_stimulus` splits by stimulus; `by_subject_loso` splits by subject (leaving out the held-out subject);
    `by_task` splits by task. Default is `by_sentence`."""

    val_fraction: float = 0.1
    """Validation fraction for random/by_sentence splits."""

    test_fraction: float = 0.1
    """Held-out test fraction (`0` disables). Defaults to `0.1` so evaluation reports on data the encoder never trained on (without a
    held-out split, runs are scored in-sample, which inflates every metric). For `random`/`by_sentence`/`by_stimulus` a disjoint test set
    is carved out and evaluation runs on it; for `by_subject_loso` the held-out subject is always the test set regardless of this value."""

    loso_holdout_subject: str | None = None
    """The subject to hold out for `by_subject_loso`. If `None`, the held-out subject is the last subject in the dataset."""

    seed: int = 42
    """Global random number generator seed."""

    deterministic: bool = False
    """Request deterministic cuDNN kernels for byte-for-byte reproducible CUDA runs (slower). Always seeds Python/NumPy/Torch.
    This is useful for debugging and reproducibility."""

    log_every: int = 10
    """Log training metrics every N optimiser steps. Default is 10."""

    eval_every: int = 1
    """Run validation every N epochs. Default is 1."""

    ckpt_dir: str = 'res/checkpoints'
    """The directory to save checkpoints to. Default is `res/checkpoints`."""

    ckpt_keep_last: int = 3
    """How many recent checkpoints to retain."""

    tensorboard: bool = False
    """Enable TensorBoard logging if installed. Default is `False`."""

    drive_backup_dir: str | None = None
    """Optional Google Drive folder id/path to mirror checkpoints to (see `zte.data.io.remote`)."""

    compile_model: bool = False
    """Apply `torch.compile` (skipped on MPS/CPU). Default is `False`. This is useful for faster inference."""
