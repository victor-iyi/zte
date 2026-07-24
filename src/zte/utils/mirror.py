"""Incremental directory mirroring, for keeping a live copy of a run on a mounted Drive.

A Colab VM can be reclaimed at any moment, so everything needed to resume must already be on Drive.
`shutil.copytree` re-copies every byte on every call, which is far too slow over a FUSE mount once
checkpoints are hundreds of megabytes; `mirror_tree` copies only what changed and never raises.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from zte.logging_utils import get_logger

_LOG = get_logger('utils.mirror')

# Reproducible from the shared data cache and large enough to dominate the mirror; excluded by default.
HEAVY_DIRS: frozenset[str] = frozenset({'cache', 'bundle'})

# Below this size a file is compared by content; above it, by size + mtime (checkpoints).
_HASH_MAX_BYTES: int = 8 * 1024 * 1024


def _digest(path: Path) -> str | None:
    """Returns a content hash, or `None` when the file cannot be read."""
    try:
        return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
    except OSError:
        return None


def _needs_copy(src: Path, dst: Path) -> bool:
    """Whether `src` differs from `dst`.

    Small files are compared by content: a metrics.json rewritten in the same second with the same
    length is a real case, and size+mtime alone would silently leave the stale copy on Drive. Large
    files (checkpoints) fall back to size + modification time, since hashing them every epoch would
    cost more than the copy it saves.
    """
    if not dst.exists():
        return True
    try:
        s, d = src.stat(), dst.stat()
    except OSError:
        return True
    if s.st_size != d.st_size:
        return True
    if s.st_size <= _HASH_MAX_BYTES:
        src_digest, dst_digest = _digest(src), _digest(dst)
        return src_digest is None or src_digest != dst_digest
    # FUSE mounts round mtimes, so compare with a tolerance rather than for equality.
    return s.st_mtime > d.st_mtime + 1


def mirror_file(src: str | Path, dst_dir: str | Path) -> bool:
    """Copies a single file into `dst_dir` when it changed, never raising.

    Args:
        src (str | Path): File to copy.
        dst_dir (str | Path): Destination directory (created if missing).

    Returns:
        bool: `True` when the file was copied.
    """
    source, target_dir = Path(src), Path(dst_dir)
    if not source.is_file():
        return False
    target = target_dir / source.name
    if not _needs_copy(source, target):
        return False
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f'.{target.name}.part')
        shutil.copy2(source, tmp)
        os.replace(tmp, target)
    except OSError as exc:
        _LOG.debug('Could not mirror %s: %r', source, exc)
        return False
    return True


def mirror_tree(
    src: str | Path,
    dst: str | Path,
    *,
    exclude_dirs: Iterable[str] = HEAVY_DIRS,
) -> tuple[int, int]:
    """Copies `src` into `dst`, transferring only files that changed since the last call.

    Never raises: a mirror failure must not kill a multi-hour training run, so IO errors are logged
    and reported through the return value instead.

    Args:
        src (str | Path): Directory to mirror from.
        dst (str | Path): Destination directory (created if missing).
        exclude_dirs (Iterable[str]): Directory names to skip anywhere in the tree.

    Returns:
        tuple[int, int]: The number of files copied and the number that failed.
    """
    source, target = Path(src), Path(dst)
    if not source.is_dir():
        return 0, 0
    skip = set(exclude_dirs)
    copied = failed = 0
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _LOG.warning('Mirror to %s skipped: %r', target, exc)
        return 0, 1

    for root, dirs, files in os.walk(source):
        dirs[:] = [d for d in dirs if d not in skip]
        rel = Path(root).relative_to(source)
        out_dir = target / rel
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _LOG.debug('Could not create %s: %r', out_dir, exc)
            failed += len(files)
            continue
        for name in files:
            src_file, dst_file = Path(root) / name, out_dir / name
            if not _needs_copy(src_file, dst_file):
                continue
            try:
                # Copy via a temp name so a killed VM cannot leave a truncated file on Drive.
                tmp = dst_file.with_name(f'.{dst_file.name}.part')
                shutil.copy2(src_file, tmp)
                os.replace(tmp, dst_file)
                copied += 1
            except OSError as exc:
                _LOG.debug('Could not mirror %s: %r', src_file, exc)
                failed += 1

    if copied or failed:
        _LOG.info('Mirrored %s -> %s (%d copied, %d failed).', source, target, copied, failed)
    return copied, failed
