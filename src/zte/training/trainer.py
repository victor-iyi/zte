"""The ZTE Trainer: a device-agnostic, checkpointing self-supervised loop.

The trainer is backend-agnostic (CPU / CUDA / MPS via `zte.device`), uses automatic mixed precision where it is safe, supports gradient accumulation and
clipping, warmup+decay scheduling, rich progress bars, structured + optional TensorBoard logging, EMA-teacher updates for data2vec, and best/last/rotating
checkpoints with optional Google Drive backup.
"""

# pyright: reportFunctionMemberAccess=false, reportPrivateImportUsage=false
# pylint: disable=import-outside-toplevel
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from zte.config import ZTEConfig
from zte.device import DeviceSpec, autocast, configure_backend, resolve_device, seed_everything
from zte.logging_utils import get_logger, progress
from zte.training.checkpoint import CheckpointManager
from zte.training.scheduler import build_scheduler

_LOG = get_logger('training.trainer')


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Moves all tensor values of a batch dict to `device` (non-blocking).

    Args:
        batch: A collated batch dict (some values may be `None`).
        device (torch.device): Target device.

    Returns:
        A new `dict` with tensors relocated and non-tensors passed through.
    """
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


class Trainer:
    """Drives self-supervised pretraining of a ZTE model with a chosen objective.

    Attributes:
        config (ZTEConfig): The full run configuration.
        device (DeviceSpec): The resolved device specification.
        model (nn.Module): The ZTE encoder being trained.
        objective (nn.Module): The self-supervised objective module.
        train_loader (DataLoader[Any]): Training DataLoader (yields collated batch dicts).
        val_loader (DataLoader[Any] | None): Optional validation DataLoader.
        extra_state (dict[str, Any] | None): Picklable extras (normaliser state, subject vocab) to embed in every checkpoint for reproducible inference.
        history (dict[str, list[float]]): Per-epoch metric history (train/val loss, lr).
    """

    def __init__(
        self,
        model: nn.Module,
        objective: nn.Module,
        config: ZTEConfig,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any] | None = None,
        device: DeviceSpec | None = None,
        extra_state: dict[str, Any] | None = None,
        resume: bool = False,
    ) -> None:
        """Wires up the model, optimiser, scheduler, AMP and checkpointing.

        Args:
            model (nn.Module): The ZTE encoder.
            objective (nn.Module): The objective module (may hold its own parameters).
            config (ZTEConfig): Full run configuration.
            train_loader (DataLoader[Any]): Training DataLoader (yields collated batch dicts).
            val_loader (DataLoader[Any] | None): Optional validation DataLoader.
            device (DeviceSpec | None): Pre-resolved device spec; auto-resolved when `None`.
            extra_state (dict[str, Any] | None): Picklable extras (normaliser state, subject vocab) to embed in every checkpoint for reproducible inference.
            resume (bool): If `True` and a `last.pt` exists in the checkpoint dir, restore the model,
                optimiser, scheduler, AMP scaler, objective/EMA-teacher, best metric, history and step,
                and continue from the next epoch. Enables pause/resume of long runs.
        """
        seed_everything(config.train.seed, deterministic=config.train.deterministic)
        self.config = config
        self.device = device or resolve_device(config.train.device, config.train.precision)
        # Backend-global speedups that don't change accuracy (TF32 on Ampere+; no-op elsewhere).
        configure_backend(self.device)
        self.model = model.to(self.device.device)
        self.objective = objective.to(self.device.device)
        self._move_teacher()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.extra_state = extra_state or {}

        params = list(self.model.parameters()) + list(self.objective.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=config.train.lr, weight_decay=config.train.weight_decay
        )
        steps_per_epoch = max(1, len(train_loader) // max(1, config.train.grad_accum_steps))
        self.total_steps = steps_per_epoch * config.train.epochs
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=self.total_steps,
            warmup_steps=int(self.total_steps * config.train.warmup_ratio),
            kind=config.train.scheduler,
        )
        self._use_scaler = self.device.is_cuda and self.device.autocast_dtype is torch.float16
        self.scaler = self._build_scaler()
        self.ckpt = CheckpointManager(
            config.train.ckpt_dir,
            keep_last=config.train.ckpt_keep_last,
            drive_backup_dir=config.train.drive_backup_dir,
            higher_is_better=False,
        )
        self.writer = self._build_tensorboard()
        self.history: dict[str, list[float]] = defaultdict(list)
        self._global_step = 0
        self._last_grad_norm: float | None = None
        self._start_epoch = 1
        if resume:
            self._resume_from_last()
        self._maybe_compile()
        _LOG.info(
            'Trainer ready | device=%s | objective=%s | steps=%d | params=%.2fM',
            self.device.name,
            config.objective.name,
            self.total_steps,
            sum(p.numel() for p in params) / 1e6,
        )

    # -- public API --------------------------------------------------------- #

    def train(self) -> dict[str, list[float]]:
        """Runs the full training loop and returns the metric history.

        Supports pause/resume: a SIGINT (Ctrl-C) or SIGTERM (`kill`) finishes cleanly, leaving a
        `last.pt` from the last completed epoch. Re-running with `resume=True` continues from the next
        epoch. Raises `KeyboardInterrupt` on a pause so the caller can skip downstream stages.

        Returns:
            The `history` dict (`train_loss`, `val_loss`, `lr` lists).

        Raises:
            KeyboardInterrupt: When the run is paused by a signal before all epochs complete.
        """
        total = self.config.train.epochs
        if self._start_epoch > total:
            _LOG.info('Training already complete (%d/%d epochs); nothing to do.', total, total)
            return dict(self.history)
        if self._start_epoch > 1:
            _LOG.info('Resuming training from epoch %d/%d.', self._start_epoch, total)

        previous = self._install_signal_handlers()
        try:
            for epoch in range(self._start_epoch, total + 1):
                train_loss = self._train_one_epoch(epoch)
                self.history['train_loss'].append(train_loss)
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

                val_loss = float('nan')
                if self.val_loader is not None and epoch % self.config.train.eval_every == 0:
                    val_loss = self.evaluate()
                    self.history['val_loss'].append(val_loss)

                monitor = val_loss if not _isnan(val_loss) else train_loss
                self.ckpt.save(self._checkpoint_state(epoch), epoch=epoch, metric=monitor)
                _LOG.info(
                    '[epoch %d/%d] train_loss=%.4f val_loss=%.4f lr=%.2e',
                    epoch,
                    total,
                    train_loss,
                    val_loss,
                    self.optimizer.param_groups[0]['lr'],
                )
                if self.writer is not None:
                    self.writer.add_scalar('loss/train', train_loss, epoch)
                    if not _isnan(val_loss):
                        self.writer.add_scalar('loss/val', val_loss, epoch)
        except KeyboardInterrupt:
            done = len(self.history['train_loss']) + self._start_epoch - 1
            _LOG.warning(
                'Paused at epoch %d/%d. Progress saved to %s. Resume with --resume.',
                done,
                total,
                self.ckpt.ckpt_dir / 'last.pt',
            )
            if self.writer is not None:
                self.writer.close()
            raise
        finally:
            self._restore_signal_handlers(previous)

        self._log_hparams()
        if self.writer is not None:
            self.writer.close()
        return dict(self.history)

    def _log_hparams(self) -> None:
        """Records this run's key knobs joined to its final losses (HParams tab)."""
        if self.writer is None:
            return
        cfg = self.config
        hparams = {
            'objective': cfg.objective.name,
            'frontend': cfg.model.frontend,
            'pos_encoding': cfg.model.pos_encoding,
            'embed_dim': cfg.model.embed_dim,
            'hidden_dim': cfg.model.hidden_dim,
            'lr': cfg.train.lr,
            'include_eye_tracking': cfg.dataset.include_eye_tracking,
        }
        final = {
            'hparam/final_train_loss': self.history['train_loss'][-1]
            if self.history['train_loss']
            else float('nan'),
        }
        if self.history.get('val_loss'):
            final['hparam/final_val_loss'] = self.history['val_loss'][-1]
        clean = {k: v for k, v in final.items() if not _isnan(v)}
        if clean:
            self.writer.add_hparams(hparams, clean)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Computes the mean validation objective loss.

        Returns:
            The average loss over the validation loader (`nan` if no loader).
        """
        if self.val_loader is None:
            return float('nan')
        self.model.eval()
        self.objective.eval()
        total, count = 0.0, 0
        for raw_batch in progress(self.val_loader, description='validating'):
            batch = move_batch(raw_batch, self.device.device)
            with autocast(self.device):
                loss, _ = self.objective.compute(self.model, batch)
            total += float(loss.detach())
            count += 1
        self.model.train()
        self.objective.train()
        return total / max(count, 1)

    # -- internals ---------------------------------------------------------- #

    def _train_one_epoch(self, epoch: int) -> float:
        """Runs one training epoch and returns the mean step loss."""
        self.model.train()
        self.objective.train()
        accum = self.config.train.grad_accum_steps
        running, n_steps = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)

        iterator = progress(
            self.train_loader,
            description=f'epoch {epoch}/{self.config.train.epochs}',
            total=len(self.train_loader),
        )
        for i, raw_batch in enumerate(iterator):
            batch = move_batch(raw_batch, self.device.device)
            # Report B §1.2: hand the objective the current progress so the subject-adversary
            # gradient-reversal strength can ramp 0 -> 1 (a no-op when warmup_ratio is 0).
            if hasattr(self.objective, 'set_progress'):
                self.objective.set_progress(self._global_step, self.total_steps)
            with autocast(self.device):
                loss, metrics = self.objective.compute(self.model, batch)
                loss = loss / accum

            self._backward(loss)
            if (i + 1) % accum == 0:
                self._optimizer_step()
                self._global_step += 1
                if self._global_step % self.config.train.log_every == 0:
                    _LOG.debug(
                        'step %d | loss=%.4f | %s',
                        self._global_step,
                        metrics.get('loss', float('nan')),
                        {k: round(v, 3) for k, v in metrics.items() if k != 'loss'},
                    )
                    self._log_step(metrics)
            running += metrics.get('loss', 0.0)
            n_steps += 1
        return running / max(n_steps, 1)

    def _log_step(self, metrics: dict[str, float]) -> None:
        """Writes per-step training scalars to TensorBoard (loss, lr, grad-norm, ...)."""
        if self.writer is None:
            return
        step = self._global_step
        self.writer.add_scalar('train/loss', metrics.get('loss', float('nan')), step)
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], step)
        if self._last_grad_norm is not None:
            self.writer.add_scalar('train/grad_norm', self._last_grad_norm, step)
        for key, value in metrics.items():
            if key != 'loss' and isinstance(value, (int, float)):
                self.writer.add_scalar(f'train/{key}', value, step)

    def _backward(self, loss: torch.Tensor) -> None:
        """Backpropagates, scaling the loss when an AMP scaler is active."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self) -> None:
        """Applies grad clipping, the optimiser/scheduler step and EMA update."""
        params = list(self.model.parameters()) + list(self.objective.parameters())
        if self.config.train.grad_clip > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            self._last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(params, self.config.train.grad_clip)
            )
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.kind == 'xla':
            # On a Cloud TPU, materialise the accumulated XLA graph for this step.
            try:
                import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

                xm.mark_step()
            except Exception:  # noqa: BLE001 — never let an XLA hiccup abort a step.
                pass
        if getattr(self.objective, 'needs_teacher', False) and hasattr(self.objective, 'post_step'):
            # Pass the global step so the objective can ramp its EMA teacher decay across training.
            self.objective.post_step(
                self.model, step=self._global_step, total_steps=self.total_steps
            )

    def _checkpoint_state(self, epoch: int) -> dict[str, Any]:
        """Builds the checkpoint payload, including objective/teacher/resume state and extras."""
        extra = dict(self.extra_state)
        extra['objective_state'] = self.objective.state_dict()
        # Resume bookkeeping: the EMA teacher is a plain object (not in state_dict), and best/history
        # live on the manager/trainer -- persist them so a resumed run is byte-consistent.
        teacher = getattr(self.objective, 'teacher', None)
        if teacher is not None:
            extra['teacher_state'] = teacher.module.state_dict()
        extra['best_metric'] = self.ckpt.best_metric
        extra['history'] = {k: list(v) for k, v in self.history.items()}
        return CheckpointManager.build_state(
            self.model,
            self.config,
            epoch=epoch,
            step=self._global_step,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            extra=extra,
        )

    def _resume_from_last(self) -> None:
        """Restores model/optimiser/scheduler/scaler/objective/teacher/best/history from `last.pt`."""
        last = self.ckpt.ckpt_dir / 'last.pt'
        if not last.exists():
            _LOG.info('resume requested but no last.pt in %s; starting fresh.', self.ckpt.ckpt_dir)
            return
        ckpt = CheckpointManager.load(last, map_location=self.device.device)
        self.model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt and self.scaler is not None:
            self.scaler.load_state_dict(ckpt['scaler'])
        extra = ckpt.get('extra', {})
        if 'objective_state' in extra:
            self.objective.load_state_dict(extra['objective_state'])
        teacher = getattr(self.objective, 'teacher', None)
        if teacher is not None and 'teacher_state' in extra:
            teacher.module.load_state_dict(extra['teacher_state'])
        self.ckpt.best_metric = extra.get('best_metric', self.ckpt.best_metric)
        for key, values in extra.get('history', {}).items():
            self.history[key] = list(values)
        self._global_step = int(ckpt.get('step', 0))
        self._start_epoch = int(ckpt.get('epoch', 0)) + 1
        # Keep checkpoint rotation aware of the epoch files already on disk.
        self.ckpt._last_paths = sorted(self.ckpt.ckpt_dir.glob('ckpt_epoch*.pt'))  # noqa: SLF001
        _LOG.info(
            'Restored from %s (epoch %d, step %d, best=%.4f).',
            last,
            self._start_epoch - 1,
            self._global_step,
            self.ckpt.best_metric,
        )

    def _install_signal_handlers(self) -> dict[int, Any]:
        """Installs SIGINT/SIGTERM handlers that raise KeyboardInterrupt for a clean pause.

        Returns:
            dict[int, Any]: The previous handlers, keyed by signal number, for later restoration.
        """
        import signal

        def _raise(signum: int, frame: Any) -> None:  # noqa: ARG001
            raise KeyboardInterrupt

        previous: dict[int, Any] = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[sig] = signal.signal(sig, _raise)
            except ValueError, OSError:  # pragma: no cover - not on the main thread
                pass
        return previous

    def _restore_signal_handlers(self, previous: dict[int, Any]) -> None:
        """Restores signal handlers saved by :meth:`_install_signal_handlers`."""
        import signal

        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except ValueError, OSError:  # pragma: no cover
                pass

    def _build_scaler(self) -> Any | None:
        """Creates an AMP grad scaler only when CUDA fp16 is in use."""
        if not self._use_scaler:
            return None
        try:
            from torch.amp import GradScaler

            return GradScaler('cuda')
        except ImportError, TypeError:  # pragma: no cover - older torch
            from torch.cuda.amp import GradScaler as CudaGradScaler

            return CudaGradScaler()

    def _build_tensorboard(self) -> Any | None:
        """Creates a TensorBoard writer when enabled and installed."""
        if not self.config.train.tensorboard:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = Path(self.config.train.ckpt_dir) / 'tb' / self.config.run_name
            return SummaryWriter(log_dir=str(log_dir))
        except ImportError:  # pragma: no cover
            _LOG.warning('TensorBoard requested but unavailable; install tensorboard.')
            return None

    def _move_teacher(self) -> None:
        """Moves a data2vec EMA teacher (if any) onto the compute device."""
        teacher = getattr(self.objective, 'teacher', None)
        if teacher is not None:
            teacher.to(self.device.device)

    def _maybe_compile(self) -> None:
        """Optionally applies `torch.compile` (skipped on MPS/CPU)."""
        if self.config.train.compile_model and self.device.is_cuda:
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
                _LOG.info('Applied torch.compile to the model.')
            except (RuntimeError, AttributeError) as exc:  # pragma: no cover
                _LOG.warning('torch.compile failed: %r', exc)


def _isnan(value: float) -> bool:
    """Returns whether a float is `NaN`."""
    return math.isnan(value)
