"""Resolve a dataset *source* (local dir, zip archives, or Google Drive) to `.mat`.

The one thing every entry point needs is a local directory that contains the ZuCo `results<SUBJECT>_<TASK>.mat` files.
Users have them in different shapes:

- an already-extracted directory (`res/data/zuco_extracted`);
- one or more `.zip` archives (`task1 - SR.zip` ...), or a folder of them;
- a Google Drive folder / shareable link (downloaded via :mod:`zte.data.remote`).

`resolve_source` normalises all of these into a single extracted directory, unzipping as needed and skipping work that is
already done, so the rest of the pipeline only ever deals with a plain local root.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from zte.logging_utils import get_logger, progress

_LOG = get_logger('data.sources')


def _has_mat(directory: Path) -> bool:
    """Returns whether `directory` contains any `.mat` file (recursively)."""
    return any(directory.rglob('*.mat'))


def _unzip_all(archives: list[Path], extract_dir: Path) -> Path:
    """Extracts a list of `.zip` archives into `extract_dir` (idempotent)."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    for archive in progress(archives, description='unzipping'):
        marker = extract_dir / f'.{archive.stem}.done'
        if marker.exists():
            _LOG.info('Skipping already-extracted %s', archive.name)
            continue
        _LOG.info('Extracting %s -> %s', archive.name, extract_dir)
        shutil.unpack_archive(str(archive), str(extract_dir))
        marker.touch()
    return extract_dir


def resolve_source(
    spec: str | Path,
    extract_dir: str | Path = 'res/data/zuco_extracted',
    download_dir: str | Path = 'res/data/_downloads',
) -> Path:
    """Turns any supported source `spec` into a local directory of `.mat` files.

    Resolution order:

    1. **A `.zip` file, or a directory containing `.zip` files** -> extracted into `extract_dir` (idempotently) and returned.
    2. **Directory already holding `.mat` files** -> returned unchanged.
    3. **A Google Drive id / URL** -> downloaded via `download_to_dir`, then resolved recursively.

    Args:
        spec (str | Path): A local directory, a `.zip` path, a directory of zips, or a Drive id/URL.
        extract_dir (str | Path): Where archives are unzipped to (and where extracted data lives).
        download_dir (str | Path): Scratch directory for Drive downloads before extraction.

    Returns:
        A local directory containing the ZuCo `.mat` files.

    Raises:
        FileNotFoundError: If resolution produced no `.mat` files.

    """
    extract_dir = Path(extract_dir)
    spec_str = str(spec)

    # 3) Remote Google Drive id / URL.
    from zte.data.remote import (  # pylint: disable=import-outside-toplevel
        download_to_dir,
        is_drive_spec,
    )

    if is_drive_spec(spec_str):
        _LOG.info('Fetching source from Google Drive: %s', spec_str)
        downloaded = download_to_dir(spec_str, download_dir)
        return resolve_source(downloaded, extract_dir=extract_dir, download_dir=download_dir)

    path = Path(spec_str)
    if not path.exists():
        raise FileNotFoundError(f'Source does not exist: {path}')

    # 1) A zip, or a directory of zips (checked before .mat so staging dirs extract to extract_dir).
    if path.is_file() and path.suffix == '.zip':
        return _finalise(_unzip_all([path], extract_dir))
    if path.is_dir():
        zips = sorted(path.glob('*.zip'))
        if zips:
            return _finalise(_unzip_all(zips, extract_dir))

    # 2) Already-extracted directory.
    if path.is_dir() and _has_mat(path):
        _LOG.info('Using extracted ZuCo data at %s', path)
        return path

    raise FileNotFoundError(
        f'No .mat files (or .zip archives) found under {path}. Point --root at the '
        'extracted ZuCo directory or the folder holding the task .zip archives.'
    )


def _finalise(extract_dir: Path) -> Path:
    """Verifies extraction produced `.mat` files and returns the directory."""
    if not _has_mat(extract_dir):
        raise FileNotFoundError(
            f'Extraction to {extract_dir} produced no .mat files; check the archives.'
        )
    return extract_dir
