from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

DEFAULT_EXTRACT_DIR: Final[Path] = Path('res/data/zuco_extracted')
DEFAULT_DOWNLOAD_DIR: Final[Path] = Path('res/data/_downloads')

_ROOT_HELP: Final[str] = (
    'Local extracted `.mat` dir, a `.zip` archive, or a folder of task `.zip` archives.'
)
_DRIVE_HELP: Final[str] = (
    'Google Drive folder id or shareable URL (downloads + extracts task archives).'
)
_EXTRACT_HELP: Final[str] = (
    'Where Drive/zips are extracted to (idempotent; default: res/data/zuco_extracted).'
)


def add_data_source_args(
    parser: argparse.ArgumentParser,
    *,
    include_bundle: bool = False,
    include_synthetic: bool = False,
    required: bool = True,
) -> argparse._MutuallyExclusiveGroup:
    """Adds `--root` / `--drive` / optional `--bundle` / `--synthetic` to a parser."""
    group = parser.add_mutually_exclusive_group(required=required)
    if include_bundle:
        group.add_argument('--bundle', type=Path, help='Saved ZuCoDataset bundle directory.')
    group.add_argument('--root', type=Path, help=_ROOT_HELP)
    group.add_argument('--drive', type=str, help=_DRIVE_HELP)
    if include_synthetic:
        group.add_argument(
            '--synthetic',
            action='store_true',
            help='Generate a schema-faithful synthetic ZuCo tree instead.',
        )
    return group


def add_extract_dir(parser: argparse.ArgumentParser) -> None:
    """Adds `--extract-dir` (Drive/zip staging) and `--overwrite` (force re-extraction)."""
    parser.add_argument('--extract-dir', type=Path, default=DEFAULT_EXTRACT_DIR, help=_EXTRACT_HELP)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Re-extract `.mat` files even if already present in the extract dir (else reused).',
    )


def resolve_data_root(
    args: argparse.Namespace,
    *,
    default: str | Path | None = None,
    tasks: object = None,
    subjects: object = None,
) -> Path:
    """Resolves `--root` or `--drive` to a local directory of `.mat` files.

    Only the `.mat` files matching `tasks` / `subjects` are extracted, so selecting `SR,NR` never
    unpacks `TSR`; `--overwrite` (read off `args`) forces re-extraction.

    Args:
        args (argparse.Namespace): The parsed arguments.
        default (str | Path | None): The default root directory.
        tasks (object): The tasks to extract.
        subjects (object): The subjects to extract.

    Returns:
        Path: The resolved root directory.
    """
    from zte.data.io.sources import resolve_source  # pylint: disable=import-outside-toplevel

    spec = getattr(args, 'drive', None) or getattr(args, 'root', None) or default
    if spec is None:
        msg = 'One of --root or --drive is required.'
        raise ValueError(msg)

    # Fall back to the CLI's own --tasks / --subjects so a command extracts only what it will load.
    if tasks is None:
        t = getattr(args, 'tasks', None)
        tasks = t.split(',') if isinstance(t, str) and t else None

    if subjects is None:
        s = getattr(args, 'subjects', None)
        subjects = s.split(',') if isinstance(s, str) and s else None

    extract_dir = getattr(args, 'extract_dir', DEFAULT_EXTRACT_DIR)

    return resolve_source(
        spec,
        extract_dir=extract_dir,
        download_dir=DEFAULT_DOWNLOAD_DIR,
        tasks=tasks,  # type: ignore[arg-type]
        subjects=subjects,  # type: ignore[arg-type]
        overwrite=bool(getattr(args, 'overwrite', False)),
    )
