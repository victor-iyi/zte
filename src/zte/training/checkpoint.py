"""Checkpoint management: best/last tracking, rotation, resume and Drive backup."""

from __future__ import annotations

import os
import pickle
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from zte.config import ZTEConfig
from zte.logging_utils import get_logger

_LOG = get_logger('training.checkpoint')

# Consecutive failed Drive mirrors that raise the log level. The first says "the mount hiccuped"; the tenth says
# "this run has been unrecoverable for ten epochs and nobody noticed".
_MIRROR_ALARMS: frozenset[int] = frozenset({1, 3, 10, 30})

# What a half-written checkpoint raises when torch tries to read it back.
_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    EOFError,
    OSError,
    ValueError,
    zipfile.BadZipFile,
    pickle.UnpicklingError,
)


def _atomic_save(state: dict[str, Any], path: Path) -> None:
    """Serialises `state` to `path` via a temp file, so a killed process cannot truncate it.

    Without this, a VM reclaimed mid-write leaves a corrupt `last.pt` *and* has already overwritten the
    previous good one -- turning a lost epoch into a lost run.
    """
    tmp = path.with_name(f'.{path.name}.tmp')
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    except OSError:
        # A mount that forbids rename-over -- Drive's FUSE layer among them -- still gets a checkpoint, but never
        # by writing over the good one: the previous file is moved aside first, so a kill mid-write costs the epoch
        # rather than the run. Writing straight to `path` here would truncate the only copy that existed.
        previous = path.with_name(f'.{path.name}.prev')
        try:
            if path.is_file():
                previous.unlink(missing_ok=True)
                shutil.move(str(path), str(previous))
            shutil.move(str(tmp), str(path))
            previous.unlink(missing_ok=True)
        except OSError as exc:
            # Losing this epoch's checkpoint costs an epoch; raising would cost the run, and a multi-day sweep is exactly
            # where a transient mount hiccup happens. The previous checkpoint is put back and `--resume` restarts from it.
            _LOG.warning(
                'Could not write %s (%r); the previous checkpoint is kept and this epoch is not saved.',
                path.name,
                exc,
            )
            if previous.is_file() and not path.is_file():
                shutil.move(str(previous), str(path))
        finally:
            tmp.unlink(missing_ok=True)
            if path.is_file():
                previous.unlink(missing_ok=True)


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copies `src` onto `dst` atomically, tolerating filesystems that forbid rename."""
    tmp = dst.with_name(f'.{dst.name}.tmp')
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        tmp.unlink(missing_ok=True)
        _LOG.warning('Could not write %s: %r', dst, exc)


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
        self.reset_best()
        self._last_paths: list[Path] = []
        self.mirror_failures = 0

    def reset_best(self) -> None:
        """Forgets the best metric, so the next monitored save starts a fresh comparison window."""
        self.best_metric = float('-inf') if self.higher_is_better else float('inf')

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
        _atomic_save(state, path)
        self._last_paths.append(path)
        self._rotate()
        # `last.pt` and `best.pt` are copies of the epoch file, not re-serialisations: one
        # `torch.save` per epoch instead of three, and the payload is guaranteed identical.
        _atomic_copy(path, self.ckpt_dir / 'last.pt')

        if metric is not None and self.is_improvement(metric):
            self.best_metric = metric
            is_best = True
        # Evaluation loads `best.pt`, so a run whose monitor is NaN every epoch (a diverged loss) would
        # train for hours and still be unable to finish. Always keep one; the first epoch seeds it.
        best_path = self.ckpt_dir / 'best.pt'
        if not best_path.exists():
            is_best = True
        if is_best:
            _atomic_copy(path, best_path)
            _LOG.info('New best checkpoint at epoch %d (metric=%.4f)', epoch, metric or float('nan'))
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
        """Mirrors the two checkpoints a reclaimed VM actually needs, the moment they are written.

        Note:
            Only `last.pt` and `best.pt` go every epoch -- `last.pt` because `--resume` reads it, `best.pt`
            because it is the result. The rotation files are history: they cost `keep_last` extra copies of a
            multi-hundred-megabyte file per epoch and buy nothing a fresh VM can use, so they ride the run
            directory's stage mirror instead. `mirror_file` skips a file whose bytes already match, so a `best.pt`
            that did not improve is a stat call rather than a copy.
        """
        if not self.drive_backup_dir:
            return
        try:
            from zte.data.io.remote import is_mounted_path, upload_directory

            if not is_mounted_path(self.drive_backup_dir):
                upload_directory(self.ckpt_dir, self.drive_backup_dir)
                return

            from zte.utils.mirror import mirror_file

            destination = Path(self.drive_backup_dir) / self.ckpt_dir.name
            for name in ('best.pt', 'last.pt'):
                mirror_file(self.ckpt_dir / name, destination)
            self._note_mirror(self.ckpt_dir / 'last.pt', destination / 'last.pt')
        except (RuntimeError, OSError) as exc:  # pragma: no cover - network/cred dependent
            self.mirror_failures += 1
            _LOG.warning('Drive backup failed: %r', exc)

    def _note_mirror(self, local: Path, remote: Path) -> None:
        """Tracks consecutive mirror failures and escalates, because a silent one is the dangerous kind.

        Note:
            The check is whether the remote copy still *differs* from the local one, not whether it exists. A
            `last.pt` that landed at epoch 1 and then stopped updating -- a mount gone read-only, a full quota --
            exists at every later epoch, so an existence check would report success forever while the run quietly
            became unrecoverable. `mirror_file` never raises, by design, so this is the only signal there is.
        """
        from zte.utils.mirror import needs_copy

        if not needs_copy(local, remote):
            self.mirror_failures = 0
            return

        self.mirror_failures += 1
        if self.mirror_failures in _MIRROR_ALARMS:
            _LOG.error(
                'Drive backup has not landed for %d consecutive epochs: %s is missing or stale. This run is NOT '
                'recoverable from Drive right now -- check the mount and the quota.',
                self.mirror_failures,
                remote,
            )

    def stage_from_drive(self) -> bool:
        """Pulls `best.pt` and `last.pt` down from Drive when the local directory has nothing to resume from.

        Note:
            Without this the per-epoch mirror is write-only insurance nobody can cash, and worse than useless: a
            fresh VM resuming into an empty directory restores no `best_metric`, so `save()` seeds a *new* best at
            epoch 1 and the next mirror overwrites the good `best.pt` on Drive with it. The staging has to happen
            before the first save, not after.

        Returns:
            bool: Whether anything was staged.
        """
        if not self.drive_backup_dir:
            return False
        if any((self.ckpt_dir / name).is_file() for name in ('last.pt', 'best.pt')):
            return False

        try:
            from zte.data.io.remote import is_mounted_path

            if not is_mounted_path(self.drive_backup_dir):
                return False

            from zte.utils.mirror import mirror_file

            source = Path(self.drive_backup_dir) / self.ckpt_dir.name
            staged = [name for name in ('best.pt', 'last.pt') if mirror_file(source / name, self.ckpt_dir)]
        except (RuntimeError, OSError) as exc:  # pragma: no cover - network/cred dependent
            _LOG.warning('Could not stage checkpoints from Drive: %r', exc)
            return False

        if staged:
            _LOG.info('Staged %s from %s -- resuming a run this machine has never seen.', ', '.join(staged), source)
        return bool(staged)

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
    def load_latest(
        ckpt_dir: str | Path, map_location: str | torch.device = 'cpu'
    ) -> tuple[dict[str, Any] | None, Path | None]:
        """Loads the newest *readable* checkpoint, falling back past corrupt ones.

        Note:
            Tries `last.pt` first, then the epoch checkpoints newest-first, then `best.pt`. A VM killed during a
            write can leave the newest file truncated; the previous epoch is then still a perfectly good resume
            point, so a torn write costs one epoch instead of the whole run. `best.pt` is the last line of that
            defence and matters most on a directory restored from Drive, which carries `last.pt` and `best.pt` but
            no rotation history -- it is a full-state checkpoint like any other, so resuming from it costs the
            epochs since the last improvement rather than the run.

        Args:
            ckpt_dir (str | Path): Directory holding the checkpoints.
            map_location (str | torch.device): Device mapping for tensor storage.

        Returns:
            tuple[dict[str, Any] | None, Path | None]: The payload and the file it came from, or
                `(None, None)` when nothing readable exists.
        """
        directory = Path(ckpt_dir)
        epochs = sorted(directory.glob('ckpt_epoch*.pt'), reverse=True)
        for candidate in [directory / 'last.pt', *epochs, directory / 'best.pt']:
            if not candidate.is_file():
                continue
            try:
                state = CheckpointManager.load(candidate, map_location=map_location)
            except _LOAD_ERRORS as exc:
                _LOG.warning('Checkpoint %s is unreadable (%r); trying the previous one.', candidate, exc)
                continue
            if candidate.name != 'last.pt':
                _LOG.warning('Resuming from %s -- `last.pt` was missing or corrupt.', candidate.name)
            return state, candidate
        return None, None

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
