"""A layered, content-addressed store for dataset bundles: local cache over persistent remote (docs/DATASET.md)."""

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

# A torn copy is the failure this guards: `mirror_tree` walks alphabetically, so `meta.json` can land on
# a store before the pickles it describes -- one file's existence is not an entry's existence.
REQUIRED_ENTRY_FILES: tuple[str, ...] = ('meta.json', 'words.pkl', 'sentences.pkl', 'arrays.npz')
"""Files a cache entry (bundle or extract) must carry before any layer counts it as present."""

# A ZuCo raw bundle is 11 GB for one task and 24 for SR+NR, and the campaign needs four task sets. Nothing
# evicted them, so a long sweep filled the disk and every later run died on a full volume rather than a bad
# number. The headroom is what training itself needs beside the bundle it is reading.
MIN_FREE_GB: float = 12.0
"""Free space, in GB, that staging must leave on the local volume."""

# Set to a number of GB to override `MIN_FREE_GB` on a machine with a different disk.
FREE_SPACE_ENV_VAR: str = 'ZTE_MIN_FREE_GB'


def _free_gb(path: Path) -> float:
    """Free space on the volume holding `path`, in GB, walking up to the nearest directory that exists."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    return shutil.disk_usage(probe).free / 1e9


def _entry_gb(directory: Path) -> float:
    """Size of a cache entry on disk, in GB."""
    return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file()) / 1e9


# Colab's boot volume holds the repo checkout, and it is not always the roomiest disk the VM has: a GPU runtime
# often mounts a larger local scratch elsewhere. An 11-24 GB bundle goes wherever there is room for it.
SCRATCH_CANDIDATES: tuple[str, ...] = ('/content', '/var/scratch', '/scratch', '/mnt/disks/local', '/tmp')
"""Local directories to consider for the bundle cache, most-preferred first, before falling back to the default."""

# Set to a directory to pin the bundle cache and skip the scan entirely.
SCRATCH_ENV_VAR: str = 'ZTE_SCRATCH_DIR'

# Measured from the store: `raw_eeg.npy` is 11.3 GB for NR alone and scales with the word count, so SR+NR is ~24.
_LARGEST_BUNDLE_GB: float = 24.0


def _writable(path: Path) -> bool:
    """Whether an existing directory can be written to, without creating anything to find out."""
    return path.is_dir() and os.access(path, os.W_OK)


def scratch_root(default: str | Path, *, margin_gb: float = 20.0) -> Path:
    """Returns the roomiest writable local directory for the bundle cache, or `default` when none beats it.

    Note:
        A prepared ZuCo bundle is 11 GB for one task and 24 for SR+NR, so where it is staged decides whether a
        twelve-fold sweep finishes. The checkout's own `res/` sits on the boot volume, which on a Colab GPU runtime
        is often the smaller of the disks available.

        A scratch volume is fixed in size and does not survive the machine, and both are fine here: this directory
        only ever holds a staging copy of the persistent store, so losing it costs a re-stage and never a result.
        Nothing durable -- runs, checkpoints, evaluation -- is written here.

    Args:
        default (str | Path): Where the cache lives when nothing roomier is found.
        margin_gb (float, optional): Free space a candidate must beat the default by before it is worth moving to.
            Defaults to 20.0, roughly one SR+NR bundle.

    Returns:
        Path: The chosen directory. It is created if it does not exist.
    """
    if pinned := os.environ.get(SCRATCH_ENV_VAR, '').strip():
        chosen = Path(pinned).expanduser()
        chosen.mkdir(parents=True, exist_ok=True)
        _LOG.info('Bundle cache pinned to %s by %s (%.0f GB free).', chosen, SCRATCH_ENV_VAR, _free_gb(chosen))
        return chosen

    fallback = Path(default)
    best, best_free = fallback, _free_gb(fallback)
    for candidate in SCRATCH_CANDIDATES:
        volume = Path(candidate)
        if not _writable(volume):
            continue

        free = _free_gb(volume)
        if free > best_free + margin_gb:
            best, best_free = volume / 'zte-cache', free

    if best != fallback:
        _LOG.info(
            'Bundle cache -> %s (%.0f of %.0f GB free, against %.0f GB on %s). It is scratch: a staging copy of '
            'the persistent store, so losing it costs a re-stage. Set %s to override.',
            best,
            best_free,
            _total_gb(best),
            _free_gb(fallback),
            fallback,
            SCRATCH_ENV_VAR,
        )
    best.mkdir(parents=True, exist_ok=True)

    # A volume too small for the corpus is worth saying so about now, rather than at 80% of a multi-GB copy.
    if best_free < _LARGEST_BUNDLE_GB:
        _LOG.warning(
            'The bundle cache has %.0f GB free at %s, and an SR+NR raw bundle is about %.0f GB. Staging will evict '
            'aggressively and may still not fit; point %s at a larger volume if one exists.',
            best_free,
            best,
            _LARGEST_BUNDLE_GB,
            SCRATCH_ENV_VAR,
        )

    return best


# A Colab local scratch is a fixed, and sometimes small, volume. A flat reserve sized for a 200 GB disk would
# swallow a 40 GB one whole, so the headroom is capped at a share of the volume it is actually reserving on.
MAX_HEADROOM_SHARE: float = 0.15
"""Largest fraction of a volume that headroom may claim, before `MIN_FREE_GB` applies."""


def min_free_gb(volume: str | Path | None = None) -> float:
    """The local-disk headroom to leave free, from `ZTE_MIN_FREE_GB` when set.

    Note:
        Scaled to the volume when one is named: a Colab scratch disk is fixed and can be far smaller than the boot
        volume, and a reserve that exceeds it would make the cache unusable rather than safe.

    Args:
        volume (str | Path | None, optional): The volume the headroom is for. Defaults to None, which returns the
            unscaled figure.

    Returns:
        float: Headroom in GB.
    """
    raw = os.environ.get(FREE_SPACE_ENV_VAR, '').strip()
    try:
        configured = max(float(raw), 0.0) if raw else MIN_FREE_GB
    except ValueError:
        _LOG.warning('Ignoring unreadable %s=%r; using %.1f GB.', FREE_SPACE_ENV_VAR, raw, MIN_FREE_GB)
        configured = MIN_FREE_GB

    if volume is None:
        return configured

    return min(configured, _total_gb(Path(volume)) * MAX_HEADROOM_SHARE)


def _total_gb(path: Path) -> float:
    """Total size of the volume holding `path`, in GB, walking up to the nearest directory that exists."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    return shutil.disk_usage(probe).total / 1e9


def _is_complete(directory: Path | None) -> bool:
    """Whether a cache directory carries every required file of a finished entry."""
    return directory is not None and all((directory / name).is_file() for name in REQUIRED_ENTRY_FILES)


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
        if _is_complete(local_dir):
            return 'local'
        if _is_complete(remote_dir):
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
        if _is_complete(local_dir):
            return local_dir
        if local_dir.is_dir():
            # A local cache entry is rebuildable by definition, so a torn one is cleared, never trusted.
            _LOG.warning('Local cache entry %s is incomplete (torn copy); discarding it.', local_dir)
            shutil.rmtree(local_dir, ignore_errors=True)

        if remote_dir is not None and remote_dir.is_dir():
            if not _is_complete(remote_dir):
                _LOG.warning(
                    'Persistent cache entry %s is incomplete (torn publish); ignoring it -- the entry will be '
                    'rebuilt and the store repaired on the next publish.',
                    remote_dir,
                )
                return None
            self.make_room(_entry_gb(remote_dir), keep=key)
            _LOG.info('Cache hit on the persistent store; copying %s -> %s ...', remote_dir, local_dir)
            copied, failed = mirror_tree(remote_dir, local_dir, exclude_dirs=())
            if failed or not _is_complete(local_dir):
                _LOG.warning(
                    'Could not stage %s locally (%d file(s) failed); clearing the partial copy.', remote_dir, failed
                )
                shutil.rmtree(local_dir, ignore_errors=True)
                return None
            _LOG.info('Staged %d file(s) from the persistent store.', copied)
            return local_dir

        return None

    def reserve(self, key: str, kind: str = 'bundle') -> Path:
        """Returns the local directory a new entry should be written to."""
        local_dir, _ = self._dirs(key, kind)
        return local_dir

    def staged(self) -> list[Path]:
        """Every complete entry currently on the local disk, least recently used first."""
        if not self.local.is_dir():
            return []

        roots = [self.local, self.local / EXTRACT_SUBDIR]
        entries = [d for root in roots if root.is_dir() for d in root.iterdir() if _is_complete(d)]

        return sorted(entries, key=lambda d: (d / 'meta.json').stat().st_atime)

    def make_room(self, need_gb: float, *, keep: str | None = None) -> list[str]:
        """Evicts least-recently-used local entries until `need_gb` fits with the configured headroom left over.

        Note:
            Only an entry the persistent store holds *complete* is evictable: a local cache entry is rebuildable by
            definition, but rebuilding a ZuCo bundle is a multi-GB extraction, so anything not safely re-stageable
            stays put and the shortfall is reported instead of silently deleting work.

        Args:
            need_gb (float): Space the incoming entry needs.
            keep (str | None, optional): Key that must not be evicted. Defaults to None.

        Returns:
            list[str]: The names of the entries removed, in the order they were removed.
        """
        headroom = min_free_gb(self.local)
        removed: list[str] = []
        for entry in self.staged():
            if _free_gb(self.local) >= need_gb + headroom:
                break
            if entry.name == keep:
                continue

            _, remote_dir = self._dirs(entry.name, 'extract' if entry.parent.name == EXTRACT_SUBDIR else 'bundle')
            if not _is_complete(remote_dir):
                _LOG.info('Keeping %s: the persistent store has no complete copy to re-stage it from.', entry.name)
                continue

            freed = _entry_gb(entry)
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry.name)
            _LOG.info(
                'Evicted %s from the local cache (%.1f GB freed; it is safe in the persistent store).',
                entry.name,
                freed,
            )

        free = _free_gb(self.local)
        if free < need_gb + headroom:
            _LOG.warning(
                'Local cache has %.1f GB free for a %.1f GB entry needing %.1f GB of headroom; staging may fill '
                'the disk. Point --data-cache at a larger volume, or lower %s.',
                free,
                need_gb,
                headroom,
                FREE_SPACE_ENV_VAR,
            )

        return removed

    def publish(self, key: str, kind: str = 'bundle') -> None:
        """Copies a freshly built entry to the persistent store, so it is never rebuilt.

        Publishing happens as soon as the entry exists rather than at the end of a run: the processing is
        then safe even if training is interrupted seconds later.
        """
        local_dir, remote_dir = self._dirs(key, kind)
        if remote_dir is None or not _is_complete(local_dir):
            return

        # A COMPLETE remote entry is immutable and already correct. Completeness, not existence, gates the
        # early return: an interrupted publish can land `meta.json` without the pickles it describes, and an
        # existence check would then freeze that torn entry into the store forever.
        if _is_complete(remote_dir):
            return
        copied, failed = mirror_tree(local_dir, remote_dir, exclude_dirs=())
        if failed or not _is_complete(remote_dir):
            _LOG.warning('Published %s incompletely (%d failure(s)); it will be repaired next time.', key, failed)
        else:
            _LOG.info('Published %s to the persistent store (%d file(s)).', key, copied)

    def describe(self) -> str:
        """Returns a one-line human description of where this store reads and writes."""
        return f'{self.local}' + (f' (persistent: {self.remote})' if self.remote else '')
