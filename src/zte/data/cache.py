"""A layered, content-addressed store for dataset bundles: fast local cache backed by a persistent remote (docs/DATASET.md)."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from zte.logging_utils import get_logger
from zte.utils.mirror import mirror_tree

_LOG = get_logger('data.cache')

# Extractions live beside the processed bundles but in their own namespace, so tooling that iterates the
# store (or a human browsing Drive) can tell the two apart at a glance.
EXTRACT_SUBDIR: str = '_extracts'

# Set in a Colab session (or a shell profile) to give every ZTE command the same persistent store.
REMOTE_ENV_VAR: str = 'ZTE_CACHE_REMOTE'

# Single-file artifacts (frozen text/meaning matrices, GloVe, the montage) share the persistent store
# under their own namespace, so they survive a reclaimed VM like the bundles do.
ARTIFACT_SUBDIR: str = '_artifacts'


def _artifact_remote(local: str | Path) -> Path | None:
    """Returns the persistent-store path for a local artifact file, or `None` without a remote."""
    remote = remote_from_env()
    return Path(remote) / ARTIFACT_SUBDIR / Path(local).name if remote else None


def fetch_artifact(local: str | Path) -> bool:
    """Stages a single cached artifact down from the persistent store when it is missing locally.

    The frozen encoder passes (contextual BERT, E5 sentence embeddings) cost minutes and their file names
    are already content-addressed, so a name match is a content match.

    Args:
        local (str | Path): Where the artifact is expected on this machine.

    Returns:
        bool: `True` if the file is present locally afterwards.
    """
    local = Path(local)
    if local.is_file():
        return True

    remote = _artifact_remote(local)
    if remote is None or not remote.is_file():
        return False

    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(remote, local)
    except OSError as exc:
        _LOG.warning('Could not stage artifact %s: %r', remote, exc)
        return False
    _LOG.info('Staged %s from the persistent store.', local.name)
    return True


def publish_artifact(local: str | Path) -> None:
    """Copies a freshly built artifact to the persistent store, so the next session reuses it."""
    local = Path(local)
    remote = _artifact_remote(local)
    if remote is None or not local.is_file() or remote.is_file():
        return

    remote.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(local, remote)
    except OSError as exc:
        _LOG.warning('Could not publish artifact %s: %r', local, exc)
        return
    _LOG.info('Published %s to the persistent store.', local.name)


def remote_from_env() -> str | None:
    """Returns the remote cache directory configured via the environment, if any."""
    value = os.environ.get(REMOTE_ENV_VAR, '').strip()
    return value or None


@dataclass(frozen=True)
class BundleStore:
    """A local cache directory optionally backed by a persistent remote one.

    Attributes:
        local (Path): Fast, per-machine cache directory that runs read from.
        remote (Path | None): Persistent directory (e.g. a mounted Drive folder) that survives the machine.
    """

    local: Path
    remote: Path | None = None

    @classmethod
    def create(cls, local: str | Path, remote: str | Path | None = None) -> BundleStore:
        """Builds a store, falling back to the `ZTE_CACHE_REMOTE` environment variable for the remote.

        Args:
            local (str | Path): The local cache directory.
            remote (str | Path | None): The persistent directory, or `None` to read the environment.

        Returns:
            BundleStore: The configured store.
        """
        resolved = remote if remote is not None else remote_from_env()
        return cls(local=Path(local), remote=Path(resolved) if resolved else None)

    def _dirs(self, key: str, kind: str) -> tuple[Path, Path | None]:
        """Returns the (local, remote) directories for one cache entry."""
        suffix = Path(EXTRACT_SUBDIR) / key if kind == 'extract' else Path(key)
        return self.local / suffix, (self.remote / suffix if self.remote else None)

    def has(self, key: str, kind: str = 'bundle') -> str | None:
        """Reports where an entry lives without copying it: `'local'`, `'persistent'` or `None`.

        The cheap counterpart to `find`, for callers that only need to know whether work is required.
        Staging gigabytes down from Drive to answer that question is exactly the waste this avoids.

        Args:
            key (str): The content-addressed entry key.
            kind (str): `'bundle'` for a processed bundle, `'extract'` for a raw extraction.

        Returns:
            str | None: Which layer holds the entry, or `None` if neither does.
        """
        local_dir, remote_dir = self._dirs(key, kind)
        if (local_dir / 'meta.json').is_file():
            return 'local'
        if remote_dir is not None and (remote_dir / 'meta.json').is_file():
            return 'persistent'
        return None

    def find(self, key: str, kind: str = 'bundle') -> Path | None:
        """Locates a cache entry, pulling it down from the remote on first use.

        Args:
            key (str): The content-addressed entry key.
            kind (str): `'bundle'` for a processed bundle, `'extract'` for a raw extraction.

        Returns:
            Path | None: A local directory ready to load, or `None` on a miss.
        """
        local_dir, remote_dir = self._dirs(key, kind)
        if (local_dir / 'meta.json').is_file():
            return local_dir
        if remote_dir is not None and (remote_dir / 'meta.json').is_file():
            _LOG.info(
                'Cache hit on the persistent store; copying %s -> %s ...', remote_dir, local_dir
            )
            copied, failed = mirror_tree(remote_dir, local_dir, exclude_dirs=())
            if failed or not (local_dir / 'meta.json').is_file():
                _LOG.warning('Could not stage %s locally (%d file(s) failed).', remote_dir, failed)
                return None
            _LOG.info('Staged %d file(s) from the persistent store.', copied)
            return local_dir
        return None

    def reserve(self, key: str, kind: str = 'bundle') -> Path:
        """Returns the local directory a new entry should be written to."""
        local_dir, _ = self._dirs(key, kind)
        return local_dir

    def publish(self, key: str, kind: str = 'bundle') -> None:
        """Copies a freshly built entry to the persistent store, so it is never rebuilt.

        Publishing happens as soon as the entry exists rather than at the end of a run: the processing is
        then safe even if training is interrupted seconds later.
        """
        local_dir, remote_dir = self._dirs(key, kind)
        if remote_dir is None or not (local_dir / 'meta.json').is_file():
            return

        # Entries are immutable, so an existing copy is already correct. In particular the memory-mapped
        # raw array is a LOCAL derived artifact: re-deriving it costs minutes, whereas pushing ~24 GB of
        # it up would bloat the persistent store for no gain.
        if (remote_dir / 'meta.json').is_file():
            return
        copied, failed = mirror_tree(local_dir, remote_dir, exclude_dirs=())
        if failed:
            _LOG.warning(
                'Published %s with %d failure(s); it will be retried next time.', key, failed
            )
        else:
            _LOG.info('Published %s to the persistent store (%d file(s)).', key, copied)

    def describe(self) -> str:
        """Returns a one-line human description of where this store reads and writes."""
        return f'{self.local}' + (f' (persistent: {self.remote})' if self.remote else '')
