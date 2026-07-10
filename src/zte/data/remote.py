"""Remote (Google Drive) load/save for datasets and checkpoints.

Three transport mechanisms are supported, tried in order of convenience:

1. **Mounted Drive path** -- on Colab (`drive.mount('/content/drive')`) or any machine where Drive is mounted as a folder,
    reads/writes are ordinary file copies. This is the most reliable path.
2. **gdown** -- downloads public/shared files and folders by id or URL (e.g. `uv add 'zte[drive]'` or `pip install 'zte[drive]'`). Download-only.
3. **PyDrive2 / service account** -- authenticated uploads when configured.

Every function degrades gracefully with an actionable error rather than a stack trace when an optional dependency or credential is missing.
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Final, Literal

from zte.logging_utils import get_logger

_LOG = get_logger('data.remote')

type DriveKind = Literal['folder', 'file']
_DRIVE_ID_RE: Final[re.Pattern[str]] = re.compile(r'^[-\w]{10,}$')


def parse_drive_spec(spec: str) -> tuple[DriveKind, str] | None:
    """Parses a Google Drive folder/file id or share URL into `(kind, id)`.

    Strips common shell-escape artifacts (e.g. trailing `\\` before `?usp=`) so pasted terminal URLs work even when over-escaped.

    Args:
        spec (str): A Drive id, share link, or `uc?id=` URL.

    Returns:
        `('folder' | 'file', id)` when `spec` looks like Drive, else `None`.
    """
    cleaned = spec.strip().replace('\\', '')
    if not cleaned:
        return None

    if cleaned.startswith(('http://', 'https://')):
        if 'drive.google.com' not in cleaned and 'docs.google.com' not in cleaned:
            return None
        folder = re.search(r'drive\.google\.com/(?:drive/)?folders/([-\w]+)', cleaned)
        if folder:
            return 'folder', folder.group(1)
        file_match = re.search(r'drive\.google\.com/file/d/([-\w]+)', cleaned)
        if file_match:
            return 'file', file_match.group(1)
        id_match = re.search(r'[?&]id=([-\w]+)', cleaned)
        if id_match:
            if 'folderview' in cleaned or 'folders' in cleaned:
                return 'folder', id_match.group(1)
            return 'file', id_match.group(1)
        return None

    if '/' not in cleaned and '.' not in cleaned and _DRIVE_ID_RE.fullmatch(cleaned):
        return 'folder', cleaned
    return None


def is_drive_spec(spec: str) -> bool:
    """Returns whether `spec` is a Google Drive id or shareable URL."""
    return parse_drive_spec(spec) is not None


def mount_drive(mount_point: str = '/content/drive') -> bool:
    """Mounts Google Drive when running inside Google Colab.

    Args:
        mount_point (str): Where to mount Drive.

    Returns:
        `True` if mounted (or already mounted), `False` outside Colab.
    """
    try:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount(mount_point)
        return True
    except ImportError:
        _LOG.info('Not running in Colab; skipping Drive mount.')
        return False


def is_mounted_path(spec: str) -> bool:
    """Heuristically decides whether `spec` is a local (mounted) filesystem path.

    Args:
        spec (str): A path or remote identifier/URL.

    Returns:
        `True` when `spec` looks like a local path whose parent exists.
    """
    if spec.startswith(('http://', 'https://')):
        return False
    if is_drive_spec(spec):
        return False
    path = Path(spec)
    if path.exists():
        return True
    parent = path.parent
    return parent.exists() and parent != Path('.')


def download_to_dir(remote_spec: str, local_dir: str | Path, *, resume: bool = True) -> Path:
    """Downloads a Drive file/folder (or copies a mounted path) into `local_dir`.

    Folder downloads are **resumable**: each file is fetched individually with `gdown`'s `resume=True`, completed files
    are recorded in `.zte_drive_manifest.json`, and re-running the same command skips finished files and continues
    partial `.part` transfers. Interrupt with Ctrl+C at any time and run again to continue.

    Per-file byte progress is shown via `gdown`/`tqdm`; overall file progress uses the package progress bar.

    Args:
        remote_spec (str): A Drive file/folder id, a shareable URL, or a local path.
        local_dir (str | Path): Local destination directory (created if missing).

    Returns:
        The local directory containing the downloaded bundle.

    Raises:
        RuntimeError: If a network download is required but `gdown` is absent.

    """
    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)

    parsed = parse_drive_spec(remote_spec)
    if parsed is not None:
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                'Remote download needs gdown. Install with: uv add gdown  (or pip install '
                "'zte[drive]'). Alternatively mount Drive and pass a local path."
            ) from exc
        kind, drive_id = parsed
        _LOG.info('Downloading Google Drive %s: %s', kind, drive_id)
        if kind == 'folder':
            from zte.data.drive_download import download_drive_folder_resumable

            download_drive_folder_resumable(gdown, drive_id, dest)
            return dest
        gdown.download(  # type: ignore[reportPrivateImportUsage]
            id=drive_id,
            output=str(dest) + '/',
            quiet=False,
            resume=resume,
        )
        return _maybe_unzip(dest)

    if is_mounted_path(remote_spec):
        src = Path(remote_spec)
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest / src.name)
        return _maybe_unzip(dest)

    msg = f'Not a Google Drive id/URL or mounted local path: {remote_spec!r}'
    raise ValueError(msg)


def upload_directory(local_dir: str | Path, remote_dir: str) -> str:
    """Uploads a directory bundle to Drive (mounted-path copy or PyDrive2).

    Args:
        local_dir (str | Path): The local bundle directory to upload.
        remote_dir (str): A mounted Drive folder path, or a Drive folder id when using PyDrive2.

    Returns:
        A string describing the remote destination.

    Raises:
        RuntimeError: If neither a mounted path nor PyDrive2 credentials are available.

    """
    local = Path(local_dir)
    if is_mounted_path(remote_dir):
        target = Path(remote_dir) / local.name
        shutil.copytree(local, target, dirs_exist_ok=True)
        _LOG.info('Copied bundle to mounted Drive path: %s', target)
        return str(target)

    try:
        from pydrive2.auth import GoogleAuth  # type: ignore[reportMissingImports]
        from pydrive2.drive import GoogleDrive  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            'Drive upload requires either a mounted Drive path or PyDrive2 (uv add pydrive2) with configured credentials.'
        ) from exc

    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    archive = shutil.make_archive(str(local), 'zip', root_dir=local)
    file = drive.CreateFile({'title': Path(archive).name, 'parents': [{'id': remote_dir}]})
    file.SetContentFile(archive)
    file.Upload()
    _LOG.info('Uploaded %s to Drive folder %s', archive, remote_dir)
    return f'drive://{remote_dir}/{Path(archive).name}'


def _maybe_unzip(directory: Path) -> Path:
    """Extracts any single top-level zip in `directory` and returns the dir."""
    zips = list(directory.glob('*.zip'))
    for archive in zips:
        shutil.unpack_archive(str(archive), str(directory))
    return directory
