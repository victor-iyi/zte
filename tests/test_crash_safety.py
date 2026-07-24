"""Tests for surviving a reclaimed VM: atomic checkpoints, corrupt-file fallback and Drive mirroring."""

from __future__ import annotations

from pathlib import Path

import torch

from zte.training.checkpoint import CheckpointManager
from zte.utils.mirror import mirror_file, mirror_tree


def _state(epoch: int) -> dict[str, object]:
    """A small, serialisable checkpoint payload."""
    return {'model': {'w': torch.zeros(4)}, 'epoch': epoch, 'step': epoch * 10}


def test_save_leaves_no_partial_files(tmp_path: Path) -> None:
    """Test that a completed save leaves only real checkpoints, never a stray temp file."""
    manager = CheckpointManager(tmp_path, keep_last=3)
    manager.save(_state(1), epoch=1, metric=1.0)

    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ['best.pt', 'ckpt_epoch0001.pt', 'last.pt']
    assert not list(tmp_path.glob('.*.tmp'))


def test_last_and_best_match_the_epoch_file(tmp_path: Path) -> None:
    """Test that `last.pt` and `best.pt` are byte-identical copies of the epoch checkpoint."""
    manager = CheckpointManager(tmp_path, keep_last=3)
    manager.save(_state(1), epoch=1, metric=0.5)

    epoch_bytes = (tmp_path / 'ckpt_epoch0001.pt').read_bytes()
    assert (tmp_path / 'last.pt').read_bytes() == epoch_bytes
    assert (tmp_path / 'best.pt').read_bytes() == epoch_bytes


def test_load_latest_prefers_last(tmp_path: Path) -> None:
    """Test that a healthy directory resumes from `last.pt`."""
    manager = CheckpointManager(tmp_path, keep_last=3)
    manager.save(_state(1), epoch=1, metric=1.0)
    manager.save(_state(2), epoch=2, metric=0.5)

    state, path = CheckpointManager.load_latest(tmp_path)
    assert path is not None and path.name == 'last.pt'
    assert state is not None and state['epoch'] == 2


def test_load_latest_falls_back_past_a_truncated_last(tmp_path: Path) -> None:
    """Test that a `last.pt` torn apart by a killed VM costs one epoch, not the whole run."""
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
    """Test that an empty or wholly corrupt directory reports "start fresh" instead of raising."""
    assert CheckpointManager.load_latest(tmp_path) == (None, None)

    (tmp_path / 'last.pt').write_bytes(b'not a checkpoint')
    (tmp_path / 'ckpt_epoch0001.pt').write_bytes(b'also not a checkpoint')
    assert CheckpointManager.load_latest(tmp_path) == (None, None)


def test_mirror_tree_copies_then_skips_unchanged(tmp_path: Path) -> None:
    """Test that mirroring is incremental: unchanged files are not re-copied on the next pass."""
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
    """Test that regenerable heavy directories stay out of the Drive mirror."""
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


def test_mirror_tree_never_raises_on_a_bad_destination(tmp_path: Path) -> None:
    """Test that an unusable Drive path degrades to a logged failure, never an exception."""
    src = tmp_path / 'run'
    src.mkdir()
    (src / 'a.txt').write_text('a', encoding='utf-8')
    blocker = tmp_path / 'blocker'
    blocker.write_text('I am a file, not a directory', encoding='utf-8')

    assert mirror_tree(src, blocker) == (0, 1)
    assert mirror_tree(tmp_path / 'does_not_exist', tmp_path / 'out') == (0, 0)


def test_mirror_file_is_incremental(tmp_path: Path) -> None:
    """Test that a single-file mirror copies once and then only after a change."""
    index = tmp_path / 'INDEX.md'
    index.write_text('# catalogue', encoding='utf-8')
    dest = tmp_path / 'drive'

    assert mirror_file(index, dest) is True
    assert (dest / 'INDEX.md').read_text(encoding='utf-8') == '# catalogue'
    assert mirror_file(index, dest) is False

    index.write_text('# catalogue\n| run |', encoding='utf-8')
    assert mirror_file(index, dest) is True
    assert mirror_file(tmp_path / 'missing.md', dest) is False
