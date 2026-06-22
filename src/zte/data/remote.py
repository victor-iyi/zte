"""Remote (Google Drive) load/save for datasets and checkpoints.

Three transport mechanisms are supported, tried in order of convenience:

#. **Mounted Drive path** -- on Colab (`drive.mount('/content/drive')`) or any
   machine where Drive is mounted as a folder, reads/writes are ordinary file copies. This is the most reliable path.
#. **gdown** -- downloads public/shared files and folders by id or URL (e.g. `uv add 'zte[drive]` or `pip install 'zte[drive]'). Download-only.
#. **PyDrive2 / service account** -- authenticated uploads when configured.

Every function degrades gracefully with an actionable error rather than a stack
trace when an optional dependency or credential is missing.
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import shutil
from pathlib import Path

from zte.logging_utils import get_logger

_LOG = get_logger('data.remote')


def mount_drive(mount_point: str = '/content/drive') -> bool:
    """Mounts Google Drive when running inside Google Colab.

    Args:
        mount_point: Where to mount Drive.

    Returns:
        ``True`` if mounted (or already mounted), ``False`` outside Colab.
    """
    try:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount(mount_point)
        return True
    except ImportError:
        _LOG.info('Not running in Colab; skipping Drive mount.')
        return False


def is_mounted_path(spec: str) -> bool:
    """Heuristically decides whether ``spec`` is a local (mounted) filesystem path.

    Args:
        spec: A path or remote identifier/URL.

    Returns:
        ``True`` when ``spec`` looks like a local path whose parent exists.
    """
    if spec.startswith(('http://', 'https://')):
        return False
    path = Path(spec)
    return path.exists() or path.parent.exists()


def download_to_dir(remote_spec: str, local_dir: str | Path) -> Path:
    """Downloads a Drive file/folder (or copies a mounted path) into ``local_dir``.

    If the downloaded artifact is a ``.zip`` it is extracted in place. If it is a
    directory bundle (already containing ``meta.json``) it is returned as-is.

    Args:
        remote_spec: A Drive file/folder id, a shareable URL, or a local path.
        local_dir: Local destination directory (created if missing).

    Returns:
        The local directory containing the downloaded bundle.

    Raises:
        RuntimeError: If a network download is required but ``gdown`` is absent.
    """
    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if is_mounted_path(remote_spec):
        src = Path(remote_spec)
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest / src.name)
        return _maybe_unzip(dest)

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            'Remote download needs gdown. Install with: uv add gdown  (or pip install '
            "'zte[drive]'). Alternatively mount Drive and pass a local path."
        ) from exc

    if 'folders' in remote_spec or remote_spec.endswith('/'):
        gdown.download_folder(url=remote_spec, output=str(dest), quiet=False, use_cookies=False)  # type: ignore[reportPrivateImportUsage]
    else:
        gdown.download(url=remote_spec, output=str(dest) + '/', quiet=False)  # type: ignore[reportPrivateImportUsage]
    return _maybe_unzip(dest)


def upload_directory(local_dir: str | Path, remote_dir: str) -> str:
    """Uploads a directory bundle to Drive (mounted-path copy or PyDrive2).

    Args:
        local_dir: The local bundle directory to upload.
        remote_dir: A mounted Drive folder path, or a Drive folder id when using
            PyDrive2.

    Returns:
        A string describing the remote destination.

    Raises:
        RuntimeError: If neither a mounted path nor PyDrive2 credentials are
            available.
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
            'Drive upload requires either a mounted Drive path or PyDrive2 '
            '(uv add pydrive2) with configured credentials.'
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
    """Extracts any single top-level zip in ``directory`` and returns the dir."""
    zips = list(directory.glob('*.zip'))
    for archive in zips:
        shutil.unpack_archive(str(archive), str(directory))
    return directory
