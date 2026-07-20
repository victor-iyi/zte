"""Resolve a dataset *source* (local dir, zip archives, or Google Drive) to `.mat`.

The one thing every entry point needs is a local directory that contains the ZuCo `results<SUBJECT>_<TASK>.mat` files.
Users have them in different shapes:

- an already-extracted directory (`res/data/zuco_extracted`);
- one or more `.zip` archives (`task2 - NR.zip` ...), or a folder of them;
- a Google Drive folder / shareable link (downloaded via `zte.data.io.remote`).

`resolve_source` normalises all of these into a single extracted directory, unzipping as needed and skipping work that is
already done, so the rest of the pipeline only ever deals with a plain local root.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Pattern

from zte.logging_utils import get_logger, progress

_LOG = get_logger('data.sources')

# ZuCo per-word files are named `results<SUBJECT>_<TASK>.mat` (e.g. `resultsZAB_SR.mat`), so the
# subject and task are readable straight from the filename — no reliance on how a zip is named.
_MAT_RE: Pattern[str] = re.compile(
    r'results(?P<subj>[A-Za-z0-9]+)_(?P<task>TSR|NR|SR)\.mat$', re.IGNORECASE
)


def _has_mat(directory: Path) -> bool:
    """Returns whether `directory` contains any `.mat` file (recursively)."""
    return any(directory.rglob('*.mat'))


def _parse_mat(member: str) -> tuple[str | None, str | None]:
    """Parses `(subject, task)` from a `.mat` member/file name, or `(None, None)``."""
    m = _MAT_RE.search(member.rsplit('/', 1)[-1])
    return (m.group('subj').upper(), m.group('task').upper()) if m else (None, None)


def _mat_members(archive: Path) -> list[str]:
    """Returns the `.mat` member names inside a zip, reading only its index (no extraction)."""
    try:
        with zipfile.ZipFile(archive) as zf:
            return [n for n in zf.namelist() if n.lower().endswith('.mat')]
    except zipfile.BadZipFile, OSError:
        return []


def _zip_has_mat(archive: Path) -> bool:
    """Returns whether a `.zip` contains any `.mat` member (index-only)."""
    return bool(_mat_members(archive))


def _wanted(member: str, tasks: set[str] | None, subjects: set[str] | None) -> bool:
    """Whether a `.mat` member is needed given the requested tasks/subjects (None = all)."""
    subj, task = _parse_mat(member)
    if tasks is not None and (task is None or task not in tasks):
        return False
    if subjects is not None and (subj is None or subj not in subjects):
        return False
    return True


def _extract_selected(
    archives: list[Path],
    extract_dir: Path,
    *,
    tasks: set[str] | None,
    subjects: set[str] | None,
    overwrite: bool,
) -> Path:
    """Extracts only the needed `.mat` members from each archive (idempotent unless `overwrite`)."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    for archive in progress(archives, description='unzipping'):
        members = _mat_members(archive)
        wanted = [m for m in members if _wanted(m, tasks, subjects)]
        if not wanted:
            continue
        n_new = 0
        try:
            with zipfile.ZipFile(archive) as zf:
                for m in wanted:
                    target = extract_dir / m
                    if target.exists() and not overwrite:
                        continue
                    zf.extract(m, extract_dir)
                    n_new += 1
        except (zipfile.BadZipFile, OSError) as exc:
            _LOG.warning('Skipping unreadable %s (%s)', archive.name, exc)
            continue
        _LOG.info(
            'Extracted %d/%d matching .mat from %s%s',
            n_new,
            len(wanted),
            archive.name,
            '' if n_new == len(wanted) else f' ({len(wanted) - n_new} already present)',
        )
    return extract_dir


def resolve_source(
    spec: str | Path,
    extract_dir: str | Path = 'res/data/zuco_extracted',
    download_dir: str | Path = 'res/data/_downloads',
    *,
    tasks: Iterable[str] | None = None,
    subjects: Iterable[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Turns any supported source `spec` into a local directory of `.mat` files.

    Resolution order:

    1. **A `.zip` file, or a directory containing `.zip` files** -> the needed `.mat` members are extracted into `extract_dir` (idempotently) and returned.
    2. **Directory already holding `.mat` files** -> returned unchanged.
    3. **A Google Drive id / URL** -> downloaded via `download_to_dir`, then resolved recursively.

    Extraction is **selective**: only archives (and, within them, only the `.mat` files) matching the requested `tasks` / `subjects`
    are unpacked, so e.g. selecting `SR,NR` never unzips `task3 - TSR.zip`, and unrelated archives (`scripts.zip`) are skipped.
    Already-extracted files are reused unless `overwrite` is set.

    Args:
        spec (str | Path): A local directory, a `.zip` path, a directory of zips, or a Drive id/URL.
        extract_dir (str | Path): Where archives are unzipped to (and where extracted data lives).
        download_dir (str | Path): Scratch directory for Drive downloads before extraction.
        tasks (Iterable[str] | None): Only extract `.mat` for these tasks (`SR`/`NR`/`TSR`); `None` = all.
        subjects (Iterable[str] | None): Only extract `.mat` for these subject codes; `None` = all.
        overwrite (bool): Re-extract files even when they already exist in `extract_dir`.

    Returns:
        A local directory containing the ZuCo `.mat` files.

    Raises:
        FileNotFoundError: If resolution produced no `.mat` files.

    """
    extract_dir = Path(extract_dir)
    spec_str = str(spec)
    tset = {t.upper() for t in tasks} if tasks else None
    sset = {s.upper() for s in subjects} if subjects else None

    # 3) Remote Google Drive id / URL.
    from zte.data.io.remote import (  # pylint: disable=import-outside-toplevel
        download_to_dir,
        is_drive_spec,
    )

    if is_drive_spec(spec_str):
        _LOG.info('Fetching source from Google Drive: %s', spec_str)
        downloaded = download_to_dir(spec_str, download_dir)
        return resolve_source(
            downloaded,
            extract_dir=extract_dir,
            download_dir=download_dir,
            tasks=tasks,
            subjects=subjects,
            overwrite=overwrite,
        )

    path = Path(spec_str)
    if not path.exists():
        raise FileNotFoundError(f'Source does not exist: {path}')

    def _extract(archives: list[Path]) -> Path:
        return _finalise(
            _extract_selected(archives, extract_dir, tasks=tset, subjects=sset, overwrite=overwrite)
        )

    # 1) A zip, or a directory of zips (checked before .mat so staging dirs extract to extract_dir).
    if path.is_file() and path.suffix == '.zip':
        return _extract([path])
    if path.is_dir():
        # Already-extracted .mat next to some zips? Prefer the data as-is; don't re-unpack.
        if _has_mat(path):
            _LOG.info('Using extracted ZuCo data at %s', path)
            return path
        zips = sorted(path.glob('*.zip'))
        data_zips = [z for z in zips if _zip_has_mat(z)]
        skipped = [z.name for z in zips if z not in data_zips]
        if skipped:
            _LOG.info(
                'Skipping %d non-data zip(s) (no .mat inside): %s',
                len(skipped),
                ', '.join(skipped),
            )
        if data_zips:
            return _extract(data_zips)

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
