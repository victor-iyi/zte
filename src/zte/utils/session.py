"""The dated Colab session layout on Drive, and finding runs and checkpoints across sessions."""

import dataclasses
import datetime
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self

from zte.logging_utils import get_logger
from zte.utils.archive import is_synthetic_run

_LOG = get_logger('utils.session')

# `drive` is slower per epoch but leaves nothing to mirror when the VM is reclaimed mid-run.
type WriteMode = Literal['auto', 'local+mirror', 'drive']
"""Whether long runs write to the VM disk and mirror afterwards, or straight to Drive."""

# One tuple behind every `--write-mode`, so a command cannot refuse a mode another command hands it.
WRITE_MODES: Final[tuple[WriteMode, ...]] = ('auto', 'local+mirror', 'drive')
"""Every accepted write mode, in the order a command lists them."""

# The one shared folder a session mounts: the dated runs, the prepared bundles and the raw archives all hang off it.
DEFAULT_DRIVE_ROOT: Final[str] = '/gdrive/My Drive/Sharables/ZTE'
"""Mounted-Drive root holding every dated session folder."""

# Training writes here rather than to Drive because a checkpoint write per epoch across the mount costs more than
# mirroring the finished stage does.
DEFAULT_LOCAL_RUNS: Final[str] = 'res/experiments'
"""Local directory holding per-run folders on the machine that trains."""

# Bundles are content-addressed and immutable, so this is a staging copy of the Drive store, never a second source.
DEFAULT_PREPARED_LOCAL: Final[str] = 'res/cache/prepared'
"""Local staging directory for prepared feature bundles."""

# A session folder is named by its ISO date, and that name is also what marks a run directory as living on Drive.
_SESSION_DATE: Final[re.Pattern[str]] = re.compile(r'\d{4}-\d{2}-\d{2}')
"""Matches a dated session folder name, and so tells a Drive run from a local one."""


def _session_of(run_dir: Path) -> str | None:
    """The dated session folder a run sits under, or `None` when it is on the local disk."""
    for parent in run_dir.parents:
        if _SESSION_DATE.fullmatch(parent.name):
            return parent.name

    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class DriveSession:
    """One dated Colab session on Drive, and every directory a run in it reads or writes.

    Attributes:
        drive_root (Path): The mounted-Drive root holding every dated session.
        run_date (str): ISO date naming this session's folder.
        resumed (bool): Whether an existing session was reopened rather than today's started.
        write_mode (WriteMode): Whether runs are written locally and mirrored, or written straight to Drive.
    """

    drive_root: Path
    run_date: str
    resumed: bool
    write_mode: WriteMode

    @classmethod
    def create(
        cls,
        drive_root: str | Path = DEFAULT_DRIVE_ROOT,
        *,
        run_date: str | None = None,
        write_mode: WriteMode = 'auto',
        make_dirs: bool = True,
    ) -> Self:
        """Opens today's session, or reopens an earlier one so its runs can be resumed.

        Args:
            drive_root (str | Path, optional): The mounted-Drive root. Defaults to `DEFAULT_DRIVE_ROOT`.
            run_date (str | None, optional): An existing session folder to resume, e.g. `'2026-08-13'`.
                Defaults to None, which starts today's and leaves `resumed` false.
            write_mode (WriteMode, optional): Where long runs write. Defaults to `'auto'`, which writes to Drive
                when it is mounted -- a Colab VM's disk cannot hold a twelve-fold sweep -- and locally otherwise.
            make_dirs (bool, optional): Create the session's Drive directories. Defaults to True.

        Returns:
            Self: The resolved session.
        """
        session = cls(
            drive_root=Path(drive_root).expanduser(),
            run_date=run_date or datetime.date.today().isoformat(),
            resumed=run_date is not None,
            write_mode=write_mode,
        )
        if write_mode == 'auto':
            # Drive mounted means Colab, where the runs of a twelve-fold sweep are ~27 GB against a disk that also
            # has to hold an 11-24 GB dataset bundle. Off Colab the local disk is the fast, sane default.
            resolved: WriteMode = 'drive' if session.drive_mounted else 'local+mirror'
            session = dataclasses.replace(session, write_mode=resolved)
            _LOG.info(
                'Storage: writing runs to %s (write_mode=auto resolved to %r; Drive mounted: %s).',
                session.out_root,
                resolved,
                session.drive_mounted,
            )
        if make_dirs:
            for directory in (session.drive_runs, session.drive_analysis, session.drive_archives):
                # An unmounted Drive is a normal state off Colab, so report it through `created` instead of raising.
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    _LOG.warning('Could not create %s (%r); Drive is probably not mounted.', directory, exc)
                    break

        return session

    @property
    def drive_mounted(self) -> bool:
        """Whether the Drive root is reachable on this machine."""
        return self.drive_root.is_dir()

    @property
    def data_dir(self) -> Path:
        """The raw ZuCo archives, shared by every session."""
        return self.drive_root / 'ZuCo Dataset'

    @property
    def data_dir_present(self) -> bool:
        """Whether the raw ZuCo directory is there to read."""
        return self.data_dir.is_dir()

    @property
    def session_dir(self) -> Path:
        """This session's dated folder."""
        return self.drive_root / self.run_date

    @property
    def drive_runs(self) -> Path:
        """Where this session's runs are catalogued on Drive."""
        return self.session_dir / 'experiments'

    @property
    def drive_analysis(self) -> Path:
        """Where cross-run analysis for this session is written."""
        return self.session_dir / 'analysis'

    @property
    def drive_archives(self) -> Path:
        """Where zipped runs for this session are kept."""
        return self.session_dir / 'archives'

    @property
    def local_runs(self) -> Path:
        """The training machine's own run directory."""
        return Path(DEFAULT_LOCAL_RUNS)

    @property
    def out_root(self) -> Path:
        """Where long runs write, which `write_mode` decides."""
        return self.drive_runs if self.write_mode == 'drive' else self.local_runs

    @property
    def drive_backup(self) -> Path:
        """Where every run is backed up, whichever way it was written."""
        return self.drive_runs

    @property
    def prepared_local(self) -> Path:
        """The local staging copy of the prepared-bundle store, on the roomiest local volume this machine has.

        Note:
            A bundle is 11 GB for one task and 24 for SR+NR, and the checkout's own `res/` sits on the boot volume,
            which on a Colab GPU runtime is often not the largest disk attached. `ZTE_SCRATCH_DIR` pins it.
        """
        from zte.data.cache import scratch_root

        return scratch_root(DEFAULT_PREPARED_LOCAL)

    @property
    def prepared_drive(self) -> Path:
        """The persistent prepared-bundle store, shared by every session because a bundle is not date-stamped."""
        return self.drive_root / 'prepared'

    @property
    def created(self) -> bool:
        """Whether this session's three Drive directories are in place."""
        return self.drive_runs.is_dir() and self.drive_analysis.is_dir() and self.drive_archives.is_dir()

    def as_dict(self) -> dict[str, Any]:
        """Every resolved path and flag of this session, JSON-ready.

        Returns:
            dict[str, Any]: Paths as strings, alongside `drive_mounted`, `data_dir_present` and `created`.
        """
        return {
            'run_date': self.run_date,
            'resumed': self.resumed,
            'write_mode': self.write_mode,
            'drive_root': str(self.drive_root),
            'drive_mounted': self.drive_mounted,
            'data_dir': str(self.data_dir),
            'data_dir_present': self.data_dir_present,
            'session_dir': str(self.session_dir),
            'drive_runs': str(self.drive_runs),
            'drive_analysis': str(self.drive_analysis),
            'drive_archives': str(self.drive_archives),
            'local_runs': str(self.local_runs),
            'out_root': str(self.out_root),
            'drive_backup': str(self.drive_backup),
            'prepared_local': str(self.prepared_local),
            'prepared_drive': str(self.prepared_drive),
            'created': self.created,
        }

    def env(self) -> dict[str, str]:
        """The environment variables every command in this session inherits.

        Returns:
            dict[str, str]: `ZTE_CACHE_REMOTE` is the persistent bundle store `BundleStore.create` falls back to;
            the rest name this session's directories.
        """
        return {
            'ZTE_DRIVE': str(self.drive_root),
            'DATA_DIR': str(self.data_dir),
            'DRIVE_DIR': str(self.session_dir),
            'DRIVE_RUNS': str(self.drive_runs),
            'DRIVE_ANALYSIS': str(self.drive_analysis),
            'DRIVE_BACKUP': str(self.drive_backup),
            'RUN_DATE': self.run_date,
            'OUT_ROOT': str(self.out_root),
            'DATA_CACHE': str(self.prepared_local),
            'ZTE_CACHE_REMOTE': str(self.prepared_drive),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RunRef:
    """One run directory found on disk, and enough about it to decide whether to read it.

    Attributes:
        name (str): The run name, which is its directory name.
        path (Path): The run directory.
        source (Literal['drive', 'local']): Whether it was found under a dated Drive session or on the local disk.
        session (str | None): The dated session it belongs to, or `None` for a local run.
        synthetic (bool): Whether `manifest.json` records `--synthetic`, so it can never be quoted as a result.
            A run that died before the manifest was written reads as false; `evaluated` is what says so.
        evaluated (bool): Whether `evaluation/metrics.json` exists -- false for a run that died before evaluation.
    """

    name: str
    path: Path
    source: Literal['drive', 'local']
    session: str | None
    synthetic: bool
    evaluated: bool


def every_session(drive_root: str | Path) -> list[Path]:
    """Every dated session's run directory on Drive, newest first.

    Args:
        drive_root (str | Path): The mounted-Drive root holding the dated session folders.

    Returns:
        list[Path]: One `<date>/experiments` path per session; a date folder without one is skipped.
    """
    root = Path(drive_root)
    if not root.is_dir():
        return []

    dated = sorted((p for p in root.iterdir() if p.is_dir() and _SESSION_DATE.fullmatch(p.name)), reverse=True)

    return [runs for p in dated if (runs := p / 'experiments').is_dir()]


def discover_runs(roots: Sequence[str | Path]) -> list[RunRef]:
    """Every run reachable across the given roots, in root order, keeping the first of any repeated name.

    A run is anything with a `config.yaml`, which `zte-run` writes as soon as it makes the directory. Keying on
    `manifest.json` or on the evaluation would hide exactly the runs worth asking about, the ones a reclaimed VM
    killed mid-training: both are written only after training finishes.

    Args:
        roots (Sequence[str | Path]): Directories holding per-run folders, most authoritative first. A missing
            root is skipped.

    Returns:
        list[RunRef]: One entry per distinct run directory name. Two runs that genuinely share a name collide and
        only the first is kept, so a `run_name` reused by a second config is invisible here.
    """
    seen: set[str] = set()
    found: list[RunRef] = []

    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue

        # Direct children only: an `rglob` over a mounted Drive costs a network round trip per directory.
        for config in sorted(base.glob('*/config.yaml')):
            run_dir = config.parent
            if run_dir.name in seen:
                continue

            seen.add(run_dir.name)
            session = _session_of(run_dir)
            found.append(
                RunRef(
                    name=run_dir.name,
                    path=run_dir,
                    source='drive' if session else 'local',
                    session=session,
                    synthetic=is_synthetic_run(run_dir),
                    evaluated=(run_dir / 'evaluation' / 'metrics.json').is_file(),
                )
            )

    return found


def find_checkpoint(run_name: str, roots: Sequence[str | Path], which: str = 'best') -> Path | None:
    """Locates one run's checkpoint across the given roots, so a fresh VM can evaluate an earlier session.

    Args:
        run_name (str): The run directory name.
        roots (Sequence[str | Path]): Directories holding per-run folders, searched in order.
        which (str, optional): Checkpoint stem, `best` or `last`. Defaults to `'best'`.

    Returns:
        Path | None: The first matching `<run>/checkpoints/<which>.pt`, or `None`. A missing `best.pt` never falls
        back to `last.pt`: they are different models, and silently swapping them misattributes the number.
    """
    for root in roots:
        candidate = Path(root) / run_name / 'checkpoints' / f'{which}.pt'
        if candidate.is_file():
            return candidate

    return None
