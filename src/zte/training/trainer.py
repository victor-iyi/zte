"""The ZTE Trainer: a device-agnostic, pausable, checkpointing self-supervised loop."""

# pyright: reportFunctionMemberAccess=false, reportPrivateImportUsage=false
from __future__ import annotations

import dataclasses
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from zte.config import ZTEConfig
from zte.device import DeviceSpec, autocast, configure_backend, resolve_device, seed_everything
from zte.logging_utils import get_logger, progress
from zte.training import stages
from zte.training.checkpoint import CheckpointManager
from zte.training.scheduler import build_scheduler
from zte.utils.provenance import git_info, package_versions

_LOG = get_logger('training.trainer')

_DECODER_MODULES = ('bridge', 'resampler', 'gap', 'clip_head', 'ladder', 'evidence', 'lexical')
"""Objective submodules a `ZTEDecoder` rebuilds, in the order `decoder_state` records them."""


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Moves all tensor values of a batch dict to `device` (non-blocking).

    Args:
        batch (dict[str, Any]): A collated batch dict (some values may be `None`).
        device (torch.device): Target device.

    Returns:
        dict[str, Any]: A new dict with tensors relocated and non-tensors passed through.
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
        extra_state (dict[str, Any] | None): Picklable extras (normaliser state, subject vocab) to embed in every
            checkpoint for reproducible inference.
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
            extra_state (dict[str, Any] | None): Picklable extras (normaliser state, subject vocab) to embed in every
                checkpoint for reproducible inference.
            resume (bool): Restore model, optimiser, scheduler, scaler, objective/teacher, best metric,
                history and step from `last.pt` and continue from the next epoch.
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

        # Optimiser and schedule, sized from the accumulated (not raw) step count.
        groups = stages.parameter_groups(self.model, self.objective, config)
        # `optimizer.load_state_dict` replaces each group's keys with the saved ones, dropping `name`.
        self._group_names = [str(group.get('name', f'group{i}')) for i, group in enumerate(groups)]
        self.optimizer = torch.optim.AdamW(groups, lr=config.train.lr, weight_decay=config.train.weight_decay)
        self._trainable_params = stages.trainable_parameters(self.model, self.objective)
        steps_per_epoch = max(1, len(train_loader) // max(1, config.train.grad_accum_steps))
        self.total_steps = steps_per_epoch * config.train.epochs
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=self.total_steps,
            warmup_steps=int(self.total_steps * config.train.warmup_ratio),
            kind=config.train.scheduler,
        )

        # Mixed precision, checkpointing and logging side-cars.
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
        self._epochs_since_best = 0
        # Curriculum stage of the most recent epoch (ran or resumed); `None` until there is one to compare against.
        self._stage: str | None = None
        self._wall_start = time.perf_counter()
        self._wall_seconds = 0.0
        if resume:
            self._resume_from_last()
        self._maybe_compile()
        _LOG.info(
            'Trainer ready | device=%s | objective=%s | mode=%s | steps=%d | trainable=%.2fM',
            self.device.name,
            config.objective.name,
            config.train.mode,
            self.total_steps,
            sum(p.numel() for p in self._trainable_params) / 1e6,
        )

    # -- public API --------------------------------------------------------- #

    def train(self) -> dict[str, list[float]]:
        """Runs the full training loop and returns the metric history.

        SIGINT or SIGTERM finishes cleanly, leaving a `last.pt` from the last completed epoch that `resume=True`
        picks up; the re-raised `KeyboardInterrupt` lets the caller skip downstream stages.

        Returns:
            dict[str, list[float]]: The `history` dict (`train_loss`, `val_loss`, `lr` lists).

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
                self._enter_stage(epoch)
                train_loss = self._train_one_epoch(epoch)
                self.history['train_loss'].append(train_loss)
                self._record_learning_rates()

                val_loss = float('nan')
                if self.val_loader is not None and epoch % self.config.train.eval_every == 0:
                    val_loss = self.evaluate()
                    self.history['val_loss'].append(val_loss)

                monitor = val_loss if not _isnan(val_loss) else train_loss
                self._epochs_since_best = 0 if self.ckpt.is_improvement(monitor) else self._epochs_since_best + 1
                self.ckpt.save(self._checkpoint_state(epoch, monitor), epoch=epoch, metric=monitor)
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
                if self._should_early_stop(epoch):
                    break
        except KeyboardInterrupt:
            self._wall_seconds += time.perf_counter() - self._wall_start
            self._wall_start = time.perf_counter()
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

        self._wall_seconds += time.perf_counter() - self._wall_start
        # Closes the interval as well as banking it. `provenance()` adds whatever has elapsed since this mark, so a
        # start left standing would count the whole of training a second time in every manifest it writes.
        self._wall_start = time.perf_counter()
        self._log_hparams()
        if self.writer is not None:
            self.writer.close()
        return dict(self.history)

    def provenance(self) -> dict[str, Any]:
        """Returns what a results table needs to place this run: code, hardware, schedule and library versions.

        Returns:
            dict[str, Any]: JSON-safe provenance, embedded in every checkpoint and consumed by `manifest.json`.
        """
        cfg = self.config
        git = git_info()
        record: dict[str, Any] = {
            'git_commit': git['commit'],
            'git_branch': git['branch'],
            'git_dirty': git['dirty'],
            'device': self.device.name,
            'device_kind': self.device.kind,
            'precision': cfg.train.precision,
            'autocast_dtype': str(self.device.autocast_dtype),
            'batch_size': cfg.train.batch_size,
            'grad_accum_steps': cfg.train.grad_accum_steps,
            'effective_batch_size': cfg.train.batch_size * max(1, cfg.train.grad_accum_steps),
            'seed': cfg.train.seed,
            'mode': cfg.train.mode,
            'wall_seconds': round(self._wall_seconds + (time.perf_counter() - self._wall_start), 3),
            'total_steps': self.total_steps,
            'torch': torch.__version__,
        }
        # `torch.__version__` keeps the build suffix (`+cu121`) that the distribution version drops.
        record.update({k: v for k, v in package_versions().items() if v is not None and k != 'torch'})
        return record

    def _record_learning_rates(self) -> None:
        """Appends this epoch's learning rates; `history['lr']` stays group 0, which is what the run plots read."""
        groups = self.optimizer.param_groups
        self.history['lr'].append(groups[0]['lr'])
        if len(groups) < 2:
            return
        for name, group in zip(self._group_names, groups, strict=False):
            self.history[f'lr_{name}'].append(group['lr'])

    def _enter_stage(self, epoch: int) -> None:
        """Tracks the curriculum stage, resetting the best monitor and patience when a boundary is crossed."""
        stage = stages.stage_for_epoch(epoch, self.config)
        if stage is None:
            return

        # Stage B adds the joint auxiliaries to the val loss too, so a lifetime best would pin `best.pt` to stage A.
        if self._stage is not None and stage != self._stage:
            self.ckpt.reset_best()
            self._epochs_since_best = 0
            _LOG.info(
                'Stage %s -> %s at epoch %d: best-checkpoint monitor and early-stop patience reset.',
                self._stage.upper(),
                stage.upper(),
                epoch,
            )
        self._stage = stage

    def _should_early_stop(self, epoch: int) -> bool:
        """Reports whether patience on the monitored metric has run out (`early_stop_patience=0` disables)."""
        patience = self.config.train.early_stop_patience
        if patience <= 0 or self._epochs_since_best < patience:
            return False
        _LOG.info(
            'Early stop at epoch %d: no improvement in %d epochs (best=%.4f).',
            epoch,
            self._epochs_since_best,
            self.ckpt.best_metric,
        )
        return True

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
            'hparam/final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else float('nan'),
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
            float: The average loss over the validation loader (`nan` if no loader).
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
            # Drain the residual head's stash every step: left to accumulate it would pin one autograd graph per
            # validation batch for the whole loop.
            self._residual_loss({})
        self._set_train_mode()
        return total / max(count, 1)

    # -- internals ---------------------------------------------------------- #

    def _train_one_epoch(self, epoch: int) -> float:
        """Runs one training epoch and returns the mean step loss."""
        if stages.apply_stage(epoch, self.model, self.objective, self.config):
            self._trainable_params = stages.trainable_parameters(self.model, self.objective)
        self._set_train_mode()
        accum = self.config.train.grad_accum_steps
        running, n_steps = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)

        iterator = progress(
            self.train_loader,
            description=f'epoch {epoch}/{self.config.train.epochs}',
            total=len(self.train_loader),
        )
        epoch_metrics: dict[str, list[float]] = {}
        for i, raw_batch in enumerate(iterator):
            batch = move_batch(raw_batch, self.device.device)
            # Progress drives the subject-adversary gradient-reversal ramp.
            if hasattr(self.objective, 'set_progress'):
                self.objective.set_progress(self._global_step, self.total_steps)
            with autocast(self.device):
                loss, metrics = self.objective.compute(self.model, batch)
                loss = loss + self._residual_loss(metrics)
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
            for key, value in metrics.items():
                if key != 'loss' and isinstance(value, (int, float)):
                    epoch_metrics.setdefault(key, []).append(float(value))

        self._record_epoch_metrics(epoch_metrics)
        return running / max(n_steps, 1)

    def _record_epoch_metrics(self, collected: dict[str, list[float]]) -> None:
        """Appends the epoch mean of every objective metric to `history`, padding series that started late.

        These are what `zte-analyze` plots as the mechanism curves -- a consensus term that never engaged or a
        gallery accuracy that never left chance is visible here and nowhere else in the artifacts.
        """
        epochs = len(self.history['train_loss']) + 1
        for key, values in collected.items():
            series = self.history[f'train_{key}']
            series.extend([float('nan')] * (epochs - 1 - len(series)))
            series.append(sum(values) / len(values) if values else float('nan'))

    def _residual_loss(self, metrics: dict[str, float]) -> torch.Tensor:
        """Drains the predictive-residual head's own regression loss and merges its metrics in place.

        The head trains here rather than inside an objective because it belongs to no objective: it de-trends the
        encoder's tokens for whatever loss comes next, and every objective gets the same treatment.
        """
        drain = getattr(self.model, 'take_residual_loss', None)
        if drain is None:
            return torch.zeros((), device=self.device.device)

        loss, extra = drain()
        metrics.update(extra)
        if loss is None:
            return torch.zeros((), device=self.device.device)

        return loss * self.config.model.residual_predict_weight

    def _set_train_mode(self) -> None:
        """Puts the model and objective in train mode, then pins every frozen submodule back to eval.

        A frozen encoder left in train mode keeps its dropout and normalisation statistics live, which makes the
        conditioning vector non-deterministic even though no gradient reaches it.
        """
        self.model.train()
        self.objective.train()
        for module in self._frozen_modules():
            module.eval()

    def _frozen_modules(self) -> list[nn.Module]:
        """Returns the submodules whose parameters all have `requires_grad=False`, so they belong in eval mode."""
        frozen: list[nn.Module] = []
        for module in (self.model, getattr(self.objective, 'lm', None)):
            if module is None:
                continue
            params = list(module.parameters())
            if params and not any(p.requires_grad for p in params):
                frozen.append(module)
        return frozen

    def _log_step(self, metrics: dict[str, float]) -> None:
        """Writes per-step training scalars to TensorBoard (loss, lr, grad-norm, ...)."""
        if self.writer is None:
            return
        step = self._global_step
        self.writer.add_scalar('train/loss', metrics.get('loss', float('nan')), step)
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], step)
        if len(self.optimizer.param_groups) > 1:
            for name, group in zip(self._group_names, self.optimizer.param_groups, strict=False):
                self.writer.add_scalar(f'train/lr/{name}', group['lr'], step)
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
        if self.config.train.grad_clip > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            self._last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self._trainable_params, self.config.train.grad_clip)
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
            except Exception:  # noqa: BLE001 -- never let an XLA hiccup abort a step.
                pass
        if getattr(self.objective, 'needs_teacher', False) and hasattr(self.objective, 'post_step'):
            # Pass the global step so the objective can ramp its EMA teacher decay across training.
            self.objective.post_step(self.model, step=self._global_step, total_steps=self.total_steps)

    def _checkpoint_state(self, epoch: int, monitor: float | None = None) -> dict[str, Any]:
        """Builds the checkpoint payload, including objective/teacher/resume state and extras.

        Args:
            epoch (int): The epoch being written.
            monitor (float | None, optional): This epoch's monitored value. Defaults to `None`; supplying it records
                the best metric this checkpoint will leave behind rather than the one it inherited.
        """
        extra = dict(self.extra_state)
        extra['objective_state'] = self.objective.state_dict()
        # The EMA teacher, best metric and history live outside any state_dict, so persist them here.
        teacher = getattr(self.objective, 'teacher', None)
        if teacher is not None:
            extra['teacher_state'] = teacher.module.state_dict()
        best = self.ckpt.best_metric
        if monitor is not None and self.ckpt.is_improvement(monitor):
            best = monitor
        extra['best_metric'] = best
        extra['epochs_since_best'] = self._epochs_since_best
        if self._stage is not None:
            extra['stage'] = self._stage
        extra['history'] = {k: list(v) for k, v in self.history.items()}
        extra['provenance'] = self.provenance()
        extra.update(self._decoder_extras())
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

    def _decoder_extras(self) -> dict[str, Any]:
        """Collects the decoder-side payload a `ZTEDecoder` needs, or nothing when this run has no bridge.

        `decoder_state` carries the decoder's own surface -- the bridge, the word resampler, the fitted gap correction
        and the text projection -- named as `objective_state` names them. The encoder-side heads stay out: no decoder
        reads them, and `objective_state` in the same checkpoint already holds everything a resume restores.
        """
        if getattr(self.objective, 'bridge', None) is None:
            return {}
        extras: dict[str, Any] = {
            'decoder_state': self._decoder_state(),
            'decoder_config': dataclasses.asdict(self.config.decoder),
        }
        gap = getattr(self.objective, 'gap', None)
        gap_state = getattr(gap, 'state', None)
        if gap_state is not None:
            extras['gap_correction'] = gap_state
        source = self.extra_state.get('encoder_source')
        if source is not None:
            extras['encoder_source'] = source
        extras['lm_provenance'] = self._lm_provenance()
        return extras

    def _decoder_state(self) -> dict[str, torch.Tensor]:
        """Returns the state of every decoder submodule this objective owns, prefixed with the submodule's name."""
        state: dict[str, torch.Tensor] = {}
        for name in _DECODER_MODULES:
            module = getattr(self.objective, name, None)
            if isinstance(module, nn.Module):
                state.update({f'{name}.{key}': v for key, v in module.state_dict().items()})
        return state

    def _lm_provenance(self) -> dict[str, Any]:
        """Returns what pins the frozen LM: its id, revision and tokeniser, so a decode is reproducible."""
        lm = getattr(self.objective, 'lm', None)
        recorded = getattr(lm, 'provenance', None)
        if callable(recorded):
            return dict(recorded())  # type: ignore[return-value]
        decoder = self.config.decoder
        return {
            'lm_source': decoder.lm_source,
            'lm_revision': decoder.lm_revision,
            'tokenizer_source': decoder.tokenizer_source or decoder.lm_source,
            'name_or_path': getattr(lm, 'name_or_path', None),
        }

    def _resume_from_last(self) -> None:
        """Restores model/optimiser/scheduler/scaler/objective/teacher/best/history from the newest checkpoint.

        Reads whichever checkpoint is newest *and* readable, so a write torn apart by a reclaimed VM
        costs one epoch rather than the run.
        """
        # A fresh VM has an empty checkpoint directory even though Drive holds the run. Pull it down before
        # deciding there is nothing to resume: otherwise this restarts at epoch 1, seeds a *new* best from an
        # untrained model, and the next mirror writes that over the good `best.pt` on Drive.
        self.ckpt.stage_from_drive()

        ckpt, last = CheckpointManager.load_latest(self.ckpt.ckpt_dir, map_location=self.device.device)
        if ckpt is None:
            _LOG.info(
                'resume requested but no readable checkpoint in %s; starting fresh.',
                self.ckpt.ckpt_dir,
            )
            return

        # Core torch state.
        self.model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt and self.scaler is not None:
            self.scaler.load_state_dict(ckpt['scaler'])

        # Trainer-side bookkeeping carried in `extra`.
        extra = ckpt.get('extra', {})
        # A decoder objective's frozen LM returns an empty `state_dict`, so only a non-strict load fits it.
        decoder_run = getattr(self.objective, 'bridge', None) is not None
        objective_state = extra.get('objective_state')
        if objective_state is None and decoder_run:
            objective_state = extra.get('decoder_state')
        if objective_state is not None:
            self.objective.load_state_dict(objective_state, strict=not decoder_run)
        teacher = getattr(self.objective, 'teacher', None)
        if teacher is not None and 'teacher_state' in extra:
            teacher.module.load_state_dict(extra['teacher_state'])
        self.ckpt.best_metric = extra.get('best_metric', self.ckpt.best_metric)
        self._epochs_since_best = int(extra.get('epochs_since_best', 0))
        # Older payloads carry no stage; `None` keeps their monitor behaviour unchanged rather than guessing a reset.
        stage = extra.get('stage')
        self._stage = stage if isinstance(stage, str) else None
        self._wall_seconds = float((extra.get('provenance') or {}).get('wall_seconds', 0.0))
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
        """Restores signal handlers saved by `_install_signal_handlers`."""
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
