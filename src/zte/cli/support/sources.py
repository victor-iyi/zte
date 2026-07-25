from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Final

from zte.logging_utils import get_logger

if TYPE_CHECKING:
    from zte.config import DatasetConfig

_LOG = get_logger('cli.sources')

DEFAULT_EXTRACT_DIR: Final[Path] = Path('res/data/zuco_extracted')
DEFAULT_DOWNLOAD_DIR: Final[Path] = Path('res/data/_downloads')

# Stands in for the data root while a cache key is computed. Cache keys exclude the root, so this lets a
# command ask "is the bundle already built?" before paying to resolve the raw source. Must not contain
# 'synthetic' -- that substring is the one part of the root the key does look at.
PENDING_ROOT: Final[str] = '<unresolved>'
SYNTHETIC_ROOT: Final[str] = 'res/data/synthetic_zuco'

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


def bundle_is_cached(dataset: DatasetConfig, synthetic: bool = False) -> str | None:
    """Reports which cache layer already holds this dataset's processed bundle, without copying it.

    Args:
        dataset (DatasetConfig): The dataset config, with `cache_dir`/`cache_remote` already set.
        synthetic (bool): Key against the synthetic tree rather than real ZuCo.

    Returns:
        str | None: `'local'`, `'persistent'`, or `None` when the bundle still has to be built.
    """
    from zte.data.cache import BundleStore
    from zte.data.dataset import ZuCoDataset

    probe = dataclasses.replace(dataset, root=SYNTHETIC_ROOT if synthetic else PENDING_ROOT)
    key = ZuCoDataset(probe)._cache_key()  # noqa: SLF001
    return BundleStore.create(dataset.cache_dir, dataset.cache_remote).has(key)


def resolve_root_if_needed(
    args: argparse.Namespace,
    dataset: DatasetConfig,
    *,
    tasks: object = None,
    subjects: object = None,
) -> str:
    """Resolves the raw data source, but only when the processed bundle is not already cached.

    Resolving means unzipping the ZuCo archives -- tens of gigabytes and several minutes, redone from
    scratch on every fresh Colab runtime. A run whose bundle is cached never reads a single `.mat`, so
    the source is left unresolved and the configured spec is recorded as-is.

    Args:
        args (argparse.Namespace): Parsed CLI arguments (`--root` / `--drive` / `--synthetic`).
        dataset (DatasetConfig): The dataset config, with `cache_dir`/`cache_remote` already set.
        tasks (object): Tasks to extract; falls back to the config's own.
        subjects (object): Subjects to extract; falls back to the config's own.

    Returns:
        str: A local `.mat` directory, or the unresolved spec when the bundle is already cached.
    """
    synthetic = bool(getattr(args, 'synthetic', False))
    synth_out = str(getattr(args, 'synthetic_out', None) or SYNTHETIC_ROOT)

    where = bundle_is_cached(dataset, synthetic=synthetic)
    if where is not None:
        spec = (
            synth_out
            if synthetic
            else str(getattr(args, 'drive', None) or getattr(args, 'root', None) or dataset.root)
        )
        _LOG.info('Processed bundle already %s; skipping raw-data extraction.', where)
        return spec

    if synthetic:
        from zte.data.synthetic import generate_synthetic_zuco

        generate_synthetic_zuco(synth_out, tasks=tuple(dataset.tasks))
        return synth_out

    return str(
        resolve_data_root(
            args,
            default=dataset.root,
            tasks=tasks if tasks is not None else dataset.tasks,
            subjects=subjects if subjects is not None else dataset.subjects,
        )
    )
