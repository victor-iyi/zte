"""Tests for surviving a reclaimed VM: atomic checkpoints, corrupt-file fallback and Drive mirroring."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch
import yaml

from zte.config import DatasetConfig, TrainConfig, ZTEConfig
from zte.training.checkpoint import CheckpointManager
from zte.utils.mirror import mirror_file, mirror_tree


def _refuse(*_args: object, **_kwargs: object) -> None:
    """Stands in for a mount that forbids the operation."""
    raise OSError('mount forbids this')


def test_path_fields_are_stored_as_strings(tmp_path: Path) -> None:
    """Assigning a `Path` to a path-like config field stores a `str`.

    Argparse hands `--data-cache` / `--drive-backup` a `Path`, and these are assigned after the config
    is constructed, so `__post_init__` cannot catch them.
    """
    dataset = DatasetConfig(root=tmp_path / 'data', cache_dir=tmp_path / 'cache')
    assert isinstance(dataset.root, str) and isinstance(dataset.cache_dir, str)

    dataset.cache_dir = tmp_path / 'prepared'  # the assignment that broke a real Colab run
    dataset.root = tmp_path / 'zuco'
    dataset.montage_csv = tmp_path / 'montage.csv'
    assert isinstance(dataset.cache_dir, str)
    assert isinstance(dataset.root, str)
    assert isinstance(dataset.montage_csv, str)

    train = TrainConfig()
    train.ckpt_dir = tmp_path / 'checkpoints'
    train.drive_backup_dir = tmp_path / 'drive'
    assert isinstance(train.ckpt_dir, str) and isinstance(train.drive_backup_dir, str)

    # `None` must survive untouched -- it is the "no Drive backup" signal.
    train.drive_backup_dir = None
    assert train.drive_backup_dir is None


def test_config_with_assigned_paths_still_serialises(tmp_path: Path) -> None:
    """A config carrying assigned paths round-trips through YAML, JSON and the checkpoint."""
    config = ZTEConfig()
    config.dataset.cache_dir = tmp_path / 'prepared'
    config.train.ckpt_dir = tmp_path / 'checkpoints'
    config.train.drive_backup_dir = tmp_path / 'drive'

    assert yaml.safe_dump(config.to_dict())  # the run's config.yaml
    assert json.dumps(dataclasses.asdict(config))  # the dataset bundle meta + checkpoint payload

    out = tmp_path / 'config.yaml'
    config.to_yaml(out)
    assert ZTEConfig.from_yaml(out).dataset.cache_dir == str(tmp_path / 'prepared')


def _state(epoch: int) -> dict[str, object]:
    """A small, serialisable checkpoint payload."""
    return {'model': {'w': torch.zeros(4)}, 'epoch': epoch, 'step': epoch * 10}


def test_save_leaves_no_partial_files(tmp_path: Path) -> None:
    """A completed save leaves only real checkpoints, never a stray temp file."""
    manager = CheckpointManager(tmp_path, keep_last=3)
    manager.save(_state(1), epoch=1, metric=1.0)

    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ['best.pt', 'ckpt_epoch0001.pt', 'last.pt']
    assert not list(tmp_path.glob('.*.tmp'))


def test_last_and_best_match_the_epoch_file(tmp_path: Path) -> None:
    """`last.pt` and `best.pt` are byte-identical copies of the epoch checkpoint."""
    manager = CheckpointManager(tmp_path, keep_last=3)
    manager.save(_state(1), epoch=1, metric=0.5)

    epoch_bytes = (tmp_path / 'ckpt_epoch0001.pt').read_bytes()
    assert (tmp_path / 'last.pt').read_bytes() == epoch_bytes
    assert (tmp_path / 'best.pt').read_bytes() == epoch_bytes


def test_load_latest_prefers_last(tmp_path: Path) -> None:
    """A healthy directory resumes from `last.pt`."""
    manager = CheckpointManager(tmp_path, keep_last=3)
    manager.save(_state(1), epoch=1, metric=1.0)
    manager.save(_state(2), epoch=2, metric=0.5)

    state, path = CheckpointManager.load_latest(tmp_path)
    assert path is not None and path.name == 'last.pt'
    assert state is not None and state['epoch'] == 2


def test_load_latest_falls_back_past_a_truncated_last(tmp_path: Path) -> None:
    """A `last.pt` torn apart by a killed VM costs one epoch, not the whole run."""
    manager = CheckpointManager(tmp_path, keep_last=3)
    manager.save(_state(1), epoch=1, metric=1.0)
    manager.save(_state(2), epoch=2, metric=0.5)

    # Simulate a process killed midway through writing `last.pt`.
    intact = (tmp_path / 'last.pt').read_bytes()
    (tmp_path / 'last.pt').write_bytes(intact[: len(intact) // 2])

    state, path = CheckpointManager.load_latest(tmp_path)
    assert path is not None and path.name == 'ckpt_epoch0002.pt'
    assert state is not None and state['epoch'] == 2


def test_load_latest_returns_none_when_nothing_is_readable(tmp_path: Path) -> None:
    """An empty or wholly corrupt directory reports "start fresh" instead of raising."""
    assert CheckpointManager.load_latest(tmp_path) == (None, None)

    (tmp_path / 'last.pt').write_bytes(b'not a checkpoint')
    (tmp_path / 'ckpt_epoch0001.pt').write_bytes(b'also not a checkpoint')
    assert CheckpointManager.load_latest(tmp_path) == (None, None)


def test_mirror_tree_copies_then_skips_unchanged(tmp_path: Path) -> None:
    """Mirroring is incremental: unchanged files are not re-copied on the next pass."""
    src, dst = tmp_path / 'run', tmp_path / 'drive'
    (src / 'evaluation').mkdir(parents=True)
    (src / 'config.yaml').write_text('run_name: x', encoding='utf-8')
    (src / 'evaluation' / 'metrics.json').write_text('{}', encoding='utf-8')

    assert mirror_tree(src, dst) == (2, 0)
    assert (dst / 'evaluation' / 'metrics.json').read_text(encoding='utf-8') == '{}'
    assert mirror_tree(src, dst) == (0, 0)  # nothing changed -> nothing re-copied

    (src / 'config.yaml').write_text('run_name: y', encoding='utf-8')
    assert mirror_tree(src, dst) == (1, 0)
    assert (dst / 'config.yaml').read_text(encoding='utf-8') == 'run_name: y'


def test_mirror_tree_skips_heavy_dirs(tmp_path: Path) -> None:
    """Regenerable heavy directories stay out of the Drive mirror."""
    src, dst = tmp_path / 'run', tmp_path / 'drive'
    (src / 'cache').mkdir(parents=True)
    (src / 'bundle').mkdir(parents=True)
    (src / 'cache' / 'arrays.npz').write_bytes(b'0' * 32)
    (src / 'bundle' / 'meta.json').write_text('{}', encoding='utf-8')
    (src / 'manifest.json').write_text('{}', encoding='utf-8')

    assert mirror_tree(src, dst) == (1, 0)
    assert (dst / 'manifest.json').is_file()
    assert not (dst / 'cache').exists()
    assert not (dst / 'bundle').exists()


def test_mirror_tree_leaves_the_rotation_history_behind_without_losing_last_pt(tmp_path: Path) -> None:
    """`ckpt_epoch*.pt` is history a fresh VM cannot resume from; `best.pt` and `last.pt` are the run itself."""
    src, dst = tmp_path / 'run', tmp_path / 'drive'
    (src / 'checkpoints').mkdir(parents=True)
    for name in ('best.pt', 'last.pt', 'ckpt_epoch03.pt', 'ckpt_epoch07.pt'):
        (src / 'checkpoints' / name).write_bytes(b'0' * 16)

    copied, failed = mirror_tree(src, dst, exclude_files=('ckpt_epoch*.pt',))

    # Two copied and *zero* failed: a file deliberately left behind must never be counted as a mirror failure.
    assert (copied, failed) == (2, 0)
    assert sorted(p.name for p in (dst / 'checkpoints').iterdir()) == ['best.pt', 'last.pt']


def test_mirror_tree_excludes_nothing_by_default(tmp_path: Path) -> None:
    """A caller that names no pattern gets every file, so the exclusion is opt-in rather than a surprise."""
    src, dst = tmp_path / 'run', tmp_path / 'drive'
    (src / 'checkpoints').mkdir(parents=True)
    (src / 'checkpoints' / 'ckpt_epoch03.pt').write_bytes(b'0')
    (src / 'checkpoints' / 'best.pt').write_bytes(b'1')

    assert mirror_tree(src, dst) == (2, 0)
    assert (dst / 'checkpoints' / 'ckpt_epoch03.pt').is_file()


def test_mirror_tree_never_raises_on_a_bad_destination(tmp_path: Path) -> None:
    """An unusable Drive path degrades to a logged failure, never an exception."""
    src = tmp_path / 'run'
    src.mkdir()
    (src / 'a.txt').write_text('a', encoding='utf-8')
    blocker = tmp_path / 'blocker'
    blocker.write_text('I am a file, not a directory', encoding='utf-8')

    assert mirror_tree(src, blocker) == (0, 1)
    assert mirror_tree(tmp_path / 'does_not_exist', tmp_path / 'out') == (0, 0)


def test_mirror_file_is_incremental(tmp_path: Path) -> None:
    """A single-file mirror copies once and then only after a change."""
    index = tmp_path / 'INDEX.md'
    index.write_text('# catalogue', encoding='utf-8')
    dest = tmp_path / 'drive'

    assert mirror_file(index, dest) is True
    assert (dest / 'INDEX.md').read_text(encoding='utf-8') == '# catalogue'
    assert mirror_file(index, dest) is False

    index.write_text('# catalogue\n| run |', encoding='utf-8')
    assert mirror_file(index, dest) is True
    assert mirror_file(tmp_path / 'missing.md', dest) is False


def test_best_checkpoint_exists_even_when_the_metric_is_nan(tmp_path: Path) -> None:
    """A diverged (NaN) metric still leaves a `best.pt` for evaluation to load.

    Evaluation loads `best.pt`; without this, a run whose loss went NaN would train for hours and then
    be unable to finish, identically on every restart.
    """
    manager = CheckpointManager(tmp_path, keep_last=3)
    manager.save(_state(1), epoch=1, metric=float('nan'))
    assert (tmp_path / 'best.pt').is_file()

    manager.save(_state(2), epoch=2, metric=float('nan'))
    state = CheckpointManager.load(tmp_path / 'best.pt')
    assert state['epoch'] == 1  # the NaN epochs never displace the seeded best


def test_best_checkpoint_is_reseeded_if_it_goes_missing(tmp_path: Path) -> None:
    """A deleted `best.pt` is recreated on the next save rather than staying absent."""
    manager = CheckpointManager(tmp_path, keep_last=3)
    manager.save(_state(1), epoch=1, metric=1.0)
    (tmp_path / 'best.pt').unlink()

    manager.save(_state(2), epoch=2, metric=5.0)  # a worse metric, so not an improvement
    assert (tmp_path / 'best.pt').is_file()


# --------------------------------------------------------------------------- #
# The Drive mirror: what a reclaimed VM can actually resume from
# --------------------------------------------------------------------------- #


def test_the_best_checkpoint_reaches_drive_the_moment_it_improves(tmp_path: Path) -> None:
    """The result must never be more than one epoch away from durable storage."""
    drive = tmp_path / 'gdrive'
    local = tmp_path / 'run' / 'checkpoints'
    manager = CheckpointManager(local, keep_last=3, drive_backup_dir=str(drive))
    manager.save(_state(1), epoch=1, metric=1.0)

    mirrored = drive / 'checkpoints' / 'best.pt'
    assert mirrored.is_file()
    assert mirrored.read_bytes() == (local / 'best.pt').read_bytes()

    manager.save(_state(2), epoch=2, metric=0.1)
    assert mirrored.read_bytes() == (local / 'best.pt').read_bytes()


def test_the_resumable_checkpoint_reaches_drive_every_epoch(tmp_path: Path) -> None:
    """`--resume` reads `last.pt`, so a mirror carrying only `best.pt` would restart from an older epoch."""
    drive = tmp_path / 'gdrive'
    manager = CheckpointManager(tmp_path / 'run' / 'checkpoints', keep_last=3, drive_backup_dir=str(drive))
    manager.save(_state(1), epoch=1, metric=1.0)
    manager.save(_state(2), epoch=2, metric=5.0)  # worse, so `best.pt` stays at epoch 1

    state, path = CheckpointManager.load_latest(drive / 'checkpoints')
    assert path is not None and path.name == 'last.pt'
    assert state is not None and state['epoch'] == 2


def test_rotation_history_is_not_mirrored(tmp_path: Path) -> None:
    """Epoch files are `keep_last` extra copies of a large file per epoch and buy a fresh VM nothing."""
    drive = tmp_path / 'gdrive'
    manager = CheckpointManager(tmp_path / 'run' / 'checkpoints', keep_last=3, drive_backup_dir=str(drive))
    for epoch in (1, 2, 3):
        manager.save(_state(epoch), epoch=epoch, metric=float(epoch))

    assert sorted(p.name for p in (drive / 'checkpoints').iterdir()) == ['best.pt', 'last.pt']


def test_a_drive_restored_directory_still_resumes_past_a_torn_last(tmp_path: Path) -> None:
    """It carries no rotation history, so `best.pt` is the whole fallback chain and has to be in it.

    Note:
        This is the case the mirror exists for: a reclaimed VM, a fresh runtime, and a `last.pt` torn by the write
        that was in flight when the machine went away. Without `best.pt` in the chain the run restarts from zero.
    """
    drive = tmp_path / 'gdrive'
    manager = CheckpointManager(tmp_path / 'run' / 'checkpoints', keep_last=3, drive_backup_dir=str(drive))
    manager.save(_state(1), epoch=1, metric=0.5)
    manager.save(_state(2), epoch=2, metric=9.0)

    restored = drive / 'checkpoints'
    intact = (restored / 'last.pt').read_bytes()
    (restored / 'last.pt').write_bytes(intact[: len(intact) // 2])

    state, path = CheckpointManager.load_latest(restored)
    assert path is not None and path.name == 'best.pt'
    assert state is not None and state['epoch'] == 1


def test_an_unwritable_mount_warns_instead_of_killing_the_run(tmp_path: Path) -> None:
    """A full or unmounted Drive must cost a warning, never a multi-hour training run."""
    blocked = tmp_path / 'blocked'
    blocked.write_text('this is a file, not a directory')
    local = tmp_path / 'run' / 'checkpoints'
    manager = CheckpointManager(local, keep_last=2, drive_backup_dir=str(blocked))

    manager.save(_state(1), epoch=1, metric=1.0)

    assert (local / 'best.pt').is_file(), 'training continued regardless'
    assert manager.mirror_failures == 1


def test_repeated_mirror_failures_are_counted_so_they_can_escalate(tmp_path: Path) -> None:
    """`mirror_file` never raises, so a mount that stopped accepting writes looks exactly like a working one."""
    blocked = tmp_path / 'blocked'
    blocked.write_text('not a directory')
    manager = CheckpointManager(tmp_path / 'run' / 'checkpoints', keep_last=2, drive_backup_dir=str(blocked))
    for epoch in (1, 2, 3):
        manager.save(_state(epoch), epoch=epoch, metric=float(epoch))

    assert manager.mirror_failures == 3


def test_a_stale_mirror_is_a_failure_even_though_the_file_exists(tmp_path: Path) -> None:
    """The dangerous case: `last.pt` landed once and then stopped updating.

    Note:
        An existence check reports success at every later epoch while the run quietly becomes unrecoverable, which
        is exactly the "silently failing for forty epochs" case the alarm claims to catch. Liveness has to mean
        "the remote copy matches the local one", not "a file with that name is there".
    """
    drive = tmp_path / 'gdrive'
    local = tmp_path / 'run' / 'checkpoints'
    manager = CheckpointManager(local, keep_last=3, drive_backup_dir=str(drive))
    manager.save(_state(1), epoch=1, metric=1.0)
    assert manager.mirror_failures == 0

    # The mount goes read-only: the epoch-1 copy stays, and every later write silently does nothing.
    (drive / 'checkpoints').chmod(0o500)
    try:
        manager.save(_state(2), epoch=2, metric=0.5)
        assert (drive / 'checkpoints' / 'last.pt').is_file(), 'the stale file is still there'
        assert manager.mirror_failures == 1, 'an existence check would have reported success'
    finally:
        (drive / 'checkpoints').chmod(0o700)


def test_a_fresh_machine_stages_the_run_down_from_drive(tmp_path: Path) -> None:
    """The mirror is write-only insurance unless something cashes it, and nothing else does.

    Note:
        Worse than useless without this: an empty local directory restores no `best_metric`, so `save()` seeds a
        new best from an untrained epoch 1 and the next mirror writes that over the good `best.pt` on Drive.
    """
    drive = tmp_path / 'gdrive'
    trained = CheckpointManager(tmp_path / 'first' / 'checkpoints', keep_last=3, drive_backup_dir=str(drive))
    trained.save(_state(1), epoch=1, metric=1.0)
    trained.save(_state(2), epoch=2, metric=0.2)

    fresh = CheckpointManager(tmp_path / 'second' / 'checkpoints', keep_last=3, drive_backup_dir=str(drive))
    assert fresh.stage_from_drive() is True

    state, path = CheckpointManager.load_latest(fresh.ckpt_dir)
    assert path is not None and path.name == 'last.pt'
    assert state is not None and state['epoch'] == 2


def test_staging_never_overwrites_local_work(tmp_path: Path) -> None:
    """A local checkpoint is always newer than the mirror it produced, so staging must not touch it."""
    drive = tmp_path / 'gdrive'
    manager = CheckpointManager(tmp_path / 'run' / 'checkpoints', keep_last=3, drive_backup_dir=str(drive))
    manager.save(_state(5), epoch=5, metric=0.1)

    assert manager.stage_from_drive() is False


def test_staging_is_a_no_op_without_a_drive(tmp_path: Path) -> None:
    """The local-only path must not pay for machinery it does not use."""
    assert CheckpointManager(tmp_path / 'checkpoints', keep_last=2).stage_from_drive() is False


def test_a_recovered_mount_resets_the_failure_count(tmp_path: Path) -> None:
    """Consecutive, not cumulative -- otherwise one early hiccup shouts for the rest of the run."""
    drive = tmp_path / 'gdrive'
    drive.write_text('blocked')
    manager = CheckpointManager(tmp_path / 'run' / 'checkpoints', keep_last=2, drive_backup_dir=str(drive))
    manager.save(_state(1), epoch=1, metric=1.0)
    assert manager.mirror_failures == 1

    drive.unlink()
    manager.save(_state(2), epoch=2, metric=0.5)

    assert manager.mirror_failures == 0
    assert (drive / 'checkpoints' / 'best.pt').is_file()


def test_a_mount_that_forbids_rename_never_truncates_the_checkpoint_it_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive's FUSE layer can refuse `os.replace`, and the old fallback wrote straight over the only good copy.

    Note:
        `write_mode='drive'` puts every checkpoint of a twelve-fold sweep on that mount, so the fallback is the
        path a reclaimed VM actually takes. A kill mid-write must cost the epoch, never the run.
    """
    from zte.training import checkpoint as ckpt_mod

    path = tmp_path / 'last.pt'
    torch.save({'epoch': 1}, path)
    monkeypatch.setattr(ckpt_mod.os, 'replace', _refuse)

    ckpt_mod._atomic_save({'epoch': 2}, path)

    assert torch.load(path, weights_only=False)['epoch'] == 2, 'the new checkpoint must land'
    assert not list(tmp_path.glob('.*.tmp')), 'no temp file may be left behind'
    assert not list(tmp_path.glob('.*.prev')), 'no shadow copy may be left behind'


def test_a_failed_write_keeps_the_previous_checkpoint_instead_of_killing_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient mount hiccup on day nine of a campaign must cost one epoch, not thirty hours."""
    from zte.training import checkpoint as ckpt_mod

    path = tmp_path / 'last.pt'
    torch.save({'epoch': 7}, path)
    monkeypatch.setattr(ckpt_mod.os, 'replace', _refuse)
    monkeypatch.setattr(ckpt_mod.shutil, 'move', _refuse)

    ckpt_mod._atomic_save({'epoch': 8}, path)  # must not raise

    assert path.is_file(), 'the previous checkpoint must survive a failed write'
    assert torch.load(path, weights_only=False)['epoch'] == 7
