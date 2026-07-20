"""Checkpoint management: best/last tracking, rotation, resume and Drive backup."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from zte.config import ZTEConfig
from zte.logging_utils import get_logger

_LOG = get_logger('training.checkpoint')


class CheckpointManager:
    """Saves, rotates, restores and (optionally) backs up training checkpoints.

    Attributes:
        ckpt_dir (Path): Local directory holding checkpoint files.
        keep_last (int): How many rolling "last" checkpoints to retain.
        best_metric (float): The best (lowest) monitored value seen so far.
        drive_backup_dir (str | None): Optional remote folder to mirror checkpoints to.
        higher_is_better (bool): Whether a higher monitored metric is better.
    """

    def __init__(
        self,
        ckpt_dir: str | Path,
        keep_last: int = 3,
        drive_backup_dir: str | None = None,
        higher_is_better: bool = False,
    ) -> None:
        """Initialises the manager and ensures the checkpoint directory exists.

        Args:
            ckpt_dir (str | Path): Local checkpoint directory (created if missing).
            keep_last (int): Number of recent "last" checkpoints to keep.
            drive_backup_dir (str | None): Optional Drive folder path/id for remote mirroring.
            higher_is_better (bool): Whether a higher monitored metric is better.
        """
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.drive_backup_dir = drive_backup_dir
        self.higher_is_better = higher_is_better
        self.best_metric = float('-inf') if higher_is_better else float('inf')
        self._last_paths: list[Path] = []

    def is_improvement(self, metric: float) -> bool:
        """Returns whether `metric` improves on the best seen so far.

        Args:
            metric (float): The freshly computed monitored value.

        Returns:
            bool: `True` if it is a new best.
        """
        return metric > self.best_metric if self.higher_is_better else metric < self.best_metric

    def save(
        self,
        state: dict[str, Any],
        epoch: int,
        metric: float | None = None,
        is_best: bool = False,
    ) -> Path:
        """Writes a checkpoint and updates best/rotation bookkeeping.

        Args:
            state (dict[str, Any]): The payload to serialise (model/optimiser/etc.).
            epoch (int): Current epoch (used in the filename).
            metric (float | None): Monitored value to record (and compare for best).
            is_best (bool): Force-mark this as the best checkpoint.

        Returns:
            Path: The path of the written `last` checkpoint.
        """
        path = self.ckpt_dir / f'ckpt_epoch{epoch:04d}.pt'
        torch.save(state, path)
        self._last_paths.append(path)
        self._rotate()
        torch.save(state, self.ckpt_dir / 'last.pt')

        if metric is not None and self.is_improvement(metric):
            self.best_metric = metric
            is_best = True
        if is_best:
            torch.save(state, self.ckpt_dir / 'best.pt')
            _LOG.info(
                'New best checkpoint at epoch %d (metric=%.4f)', epoch, metric or float('nan')
            )
        self._backup_to_drive()
        return path

    def _rotate(self) -> None:
        """Deletes the oldest epoch checkpoints beyond `keep_last`, tolerating mounts that forbid `unlink`."""
        while len(self._last_paths) > self.keep_last:
            old = self._last_paths.pop(0)
            try:
                old.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - filesystem dependent
                _LOG.debug('Could not rotate %s: %r', old, exc)

    def _backup_to_drive(self) -> None:
        """Mirrors the checkpoint directory to Drive when configured."""
        if not self.drive_backup_dir:
            return
        try:
            from zte.data.io.remote import upload_directory

            upload_directory(self.ckpt_dir, self.drive_backup_dir)
        except (RuntimeError, OSError) as exc:  # pragma: no cover - network/cred dependent
            _LOG.warning('Drive backup failed: %r', exc)

    @staticmethod
    def build_state(
        model: torch.nn.Module,
        config: ZTEConfig,
        epoch: int,
        step: int,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assembles a serialisable checkpoint payload.

        Args:
            model (torch.nn.Module): The model to snapshot.
            config (ZTEConfig): The full run config.
            epoch (int): Current epoch.
            step (int): Global optimiser step.
            optimizer (torch.optim.Optimizer | None): Optional optimiser to snapshot.
            scheduler (Any | None): Optional scheduler to snapshot.
            scaler (Any | None): Optional AMP grad scaler to snapshot.
            extra (dict[str, Any] | None): Extra picklable data (normaliser state, subject vocab, ...).

        Returns:
            dict[str, Any]: A dict suitable for `torch.save`.
        """
        state: dict[str, Any] = {
            'model': model.state_dict(),
            'config': asdict(config),
            'epoch': epoch,
            'step': step,
        }
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            state['scheduler'] = scheduler.state_dict()
        if scaler is not None:
            state['scaler'] = scaler.state_dict()
        if extra:
            state['extra'] = extra
        return state

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = 'cpu') -> dict[str, Any]:
        """Loads a checkpoint payload from disk.

        Args:
            path (str | Path): Checkpoint file path.
            map_location (str | torch.device): Device mapping for tensor storage.

        Returns:
            dict[str, Any]: The deserialised checkpoint dict.
        """
        return torch.load(path, map_location=map_location, weights_only=False)
