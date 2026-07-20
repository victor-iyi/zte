from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from zte.logging_utils import get_logger, progress

_LOG = get_logger('data.drive_download')

MANIFEST_NAME: Final[str] = '.zte_drive_manifest.json'


@dataclass(slots=True)
class DriveFileEntry:
    """One file inside a Drive folder listing."""

    id: str
    """The file's unique Drive ID."""
    name: str
    """The file's name."""
    local_path: Path
    """The file's local path."""


class DriveManifest:
    """Tracks per-file download completion for resume across interruptions."""

    def __init__(self, folder_id: str, files: dict[str, dict[str, Any]] | None = None) -> None:
        self.folder_id = folder_id
        self.files = files if files is not None else {}

    @classmethod
    def load(cls, dest: Path, folder_id: str) -> DriveManifest:
        """Loads a manifest from `dest` or returns an empty one."""
        path = dest / MANIFEST_NAME
        if not path.is_file():
            return cls(folder_id)
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('folder_id') != folder_id:
            return cls(folder_id)
        files = data.get('files', {})
        if not isinstance(files, dict):
            files = {}
        return cls(folder_id, files)

    def save(self, dest: Path) -> None:
        """Persists the manifest after each completed file."""
        dest.mkdir(parents=True, exist_ok=True)
        payload = {
            'folder_id': self.folder_id,
            'updated_at': datetime.now(UTC).isoformat(),
            'files': self.files,
        }
        (dest / MANIFEST_NAME).write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def sync_entries(self, entries: list[DriveFileEntry]) -> None:
        """Registers newly discovered Drive files without clobbering done state."""
        for entry in entries:
            rel = entry.local_path.name
            if entry.id not in self.files:
                self.files[entry.id] = {'name': entry.name, 'local_path': rel, 'status': 'pending'}
            else:
                self.files[entry.id]['name'] = entry.name
                self.files[entry.id]['local_path'] = rel

    def mark_done(self, entry: DriveFileEntry, dest: Path) -> None:
        """Marks a file complete and records its byte size."""
        local = dest / entry.local_path.name
        nbytes = local.stat().st_size if local.is_file() else 0
        self.files[entry.id] = {
            'name': entry.name,
            'local_path': entry.local_path.name,
            'status': 'done',
            'bytes': nbytes,
        }

    def is_done(self, entry: DriveFileEntry, dest: Path) -> bool:
        """Returns whether a file is already fully on disk."""
        meta = self.files.get(entry.id, {})
        if meta.get('status') != 'done':
            return False
        local = dest / entry.local_path.name
        return local.is_file() and local.stat().st_size > 0


def list_drive_folder_files(
    gdown: object, folder_id: str, dest: Path, *, use_cookies: bool
) -> list[DriveFileEntry]:
    """Lists downloadable files in a Drive folder (no bytes transferred)."""
    download_folder = gdown.download_folder  # type: ignore[attr-defined]
    listed = download_folder(
        id=folder_id,
        output=str(dest),
        skip_download=True,
        use_cookies=use_cookies,
        quiet=True,
    )
    entries: list[DriveFileEntry] = []
    for item in listed:
        name = Path(item.path).name  # type: ignore[attr-defined]
        entries.append(
            DriveFileEntry(
                id=item.id,  # type: ignore[attr-defined]
                name=name,
                local_path=Path(name),
            )
        )
    return entries


def download_drive_folder_resumable(gdown: object, folder_id: str, dest: Path) -> None:
    """Downloads every file in a Drive folder, resuming after interruption."""
    download = gdown.download  # type: ignore[attr-defined]
    from gdown.exceptions import DownloadError  # pylint: disable=import-outside-toplevel

    dest.mkdir(parents=True, exist_ok=True)
    manifest = DriveManifest.load(dest, folder_id)

    entries: list[DriveFileEntry] | None = None
    last_err: Exception | None = None
    for use_cookies in (True, False):
        try:
            entries = list_drive_folder_files(gdown, folder_id, dest, use_cookies=use_cookies)
            break
        except DownloadError as exc:
            last_err = exc
            _LOG.warning('Could not list Drive folder (use_cookies=%s): %s', use_cookies, exc)
    if entries is None:
        assert last_err is not None
        raise last_err

    manifest.sync_entries(entries)
    manifest.save(dest)

    pending = [e for e in entries if not manifest.is_done(e, dest)]
    done = len(entries) - len(pending)
    _LOG.info(
        'Drive folder %s: %d file(s) total, %d complete, %d to download (Ctrl+C safe — re-run to resume)',
        folder_id,
        len(entries),
        done,
        len(pending),
    )

    for entry in progress(pending, description='drive files', unit='file'):
        local = dest / entry.local_path.name
        _LOG.info('Downloading %s -> %s', entry.name, local)
        _download_file_resumable(download, DownloadError, entry, local)
        manifest.mark_done(entry, dest)
        manifest.save(dest)
        _LOG.info('Finished %s (%s bytes)', entry.name, local.stat().st_size)

    _LOG.info('Drive download complete: %s', dest.resolve())


def _download_file_resumable(
    download: object,
    download_error: type[Exception],
    entry: DriveFileEntry,
    local: Path,
) -> None:
    """Downloads one Drive file with resume, retrying cookie settings on failure."""
    last_err: Exception | None = None
    for use_cookies in (True, False):
        try:
            download(  # type: ignore[operator]
                id=entry.id,
                output=str(local),
                quiet=False,
                resume=True,
                use_cookies=use_cookies,
            )
            return
        except download_error as exc:
            last_err = exc
            _LOG.warning(
                'Download failed for %s (use_cookies=%s): %s', entry.name, use_cookies, exc
            )
    assert last_err is not None
    raise last_err
