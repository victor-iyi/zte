"""Zip, download-prepare and delete ZTE training runs — the Colab↔local hand-off.

The intended workflow: train on a powerful cloud GPU (Colab Pro), **zip** the finished runs into a small archive
(checkpoints + config + evaluation, without the huge dataset cache / TensorBoard logs by default), **download** them, and **unpack**
on a Mac for inference / offline exploration. A run's checkpoint embeds its input shapes and fitted normaliser, so inference needs only
the checkpoint — which is why the default archive is light.

Everything here is pure-stdlib and platform-agnostic (paths via `pathlib`), so it behaves the same on Linux, macOS and Colab.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from zte.logging_utils import get_logger

_LOG = get_logger('utils.archive')

# Directory names excluded from a run archive by default (heavy and not needed for inference).
_HEAVY_DIRS: set[str] = {'cache', 'tb', 'bundle'}


def human_size(n: int) -> str:
    """Formats a byte count as a short human-readable string (e.g., `'12.3 MB'`)."""
    size = float(n)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size < 1024.0 or unit == 'TB':
            return f'{size:.1f} {unit}' if unit != 'B' else f'{int(size)} B'
        size /= 1024.0
    return f'{size:.1f} TB'


def _dir_size(path: Path) -> int:
    """Total size in bytes of all files under `path`.

    Args:
        path (Path): The path to the directory.

    Returns:
        The total size in bytes of all files under `path`.
    """
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def _run_is_complete(run_dir: Path) -> bool:
    """Heuristic: a run is 'complete' if it has a best/last checkpoint and an evaluation."""
    ckpt = (run_dir / 'checkpoints' / 'best.pt').exists() or (
        run_dir / 'checkpoints' / 'last.pt'
    ).exists()
    return ckpt and (run_dir / 'evaluation' / 'metrics.json').exists()


def list_runs(experiments_root: str | Path = 'res/experiments') -> list[dict]:
    """Lists catalogued runs with sizes and a completeness flag.

    Args:
        experiments_root: (str | Path): The directory holding per-run folders.

    Returns:
        A list of dicts `{name, path, size_bytes, size, has_checkpoint, complete}`, largest first.
    """
    root = Path(experiments_root)
    if not root.is_dir():
        return []
    rows: list[dict] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        size = _dir_size(child)
        rows.append(
            {
                'name': child.name,
                'path': str(child),
                'size_bytes': size,
                'size': human_size(size),
                'has_checkpoint': (child / 'checkpoints' / 'best.pt').exists()
                or (child / 'checkpoints' / 'last.pt').exists(),
                'complete': _run_is_complete(child),
            }
        )
    rows.sort(key=lambda r: r['size_bytes'], reverse=True)
    return rows


def _iter_archive_files(
    run_dir: Path,
    *,
    with_bundle: bool,
    with_cache: bool,
    with_tb: bool,
    best_only: bool = False,
) -> list[tuple[Path, str]]:
    """Yields `(absolute_path, arcname)` pairs for the files to include from one run.

    Args:
        run_dir: (str | Path): The run directory.
        with_bundle: (bool): Include the saved dataset bundle.
        with_cache: (bool): Include the processed dataset cache.
        with_tb: (bool): Include TensorBoard logs.
        best_only: (bool): Keep only the best checkpoint.

    Returns:
        A list of `(absolute_path, arcname)` pairs for the files to include from one run.

    """
    skip = set(_HEAVY_DIRS)
    if with_bundle:
        skip.discard('bundle')
    if with_cache:
        skip.discard('cache')
    if with_tb:
        skip.discard('tb')
    files: list[tuple[Path, str]] = []
    for p in sorted(run_dir.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(run_dir)
        if any(part in skip for part in rel.parts):  # 'tb' also lives under checkpoints/tb
            continue
        # best_only: keep just checkpoints/best.pt, dropping last.pt + every ckpt_epoch*.pt.
        if best_only and rel.parts and rel.parts[0] == 'checkpoints':
            if p.suffix == '.pt' and p.name != 'best.pt':
                continue
        files.append((p, str(Path(run_dir.name) / rel)))
    return files


def zip_run(
    run_dir: str | Path,
    out: str | Path | None = None,
    *,
    with_bundle: bool = False,
    with_cache: bool = False,
    with_tb: bool = False,
    best_only: bool = False,
    move: bool = False,
) -> Path:
    """Zips a single run into an archive suitable for download.

    By default the heavy `cache/`, `tb/` and `bundle/` directories are excluded (a run's checkpoint
    already embeds everything inference needs), keeping the archive small.

    Args:
        run_dir: The run directory (`res/experiments/<name>`).
        out: (str | Path): Output `.zip` path (default `<run>.zip` next to the run). Point this at a mounted Google Drive folder to upload straight to Drive.
        with_bundle: (bool): Include the saved dataset bundle (needed only to _re-evaluate_, not to infer).
        with_cache: (bool): Include the processed dataset cache (large).
        with_tb: (bool): Include TensorBoard event logs.
        best_only: (bool): Keep only `checkpoints/best.pt` (drop `last.pt` + every `ckpt_epoch*.pt`) — smallest archive, enough for inference but not for resuming training.
        move: (bool): Delete the run directory after a successful zip (free local space once it is on Drive).

    Returns:
        The written archive path.

    Raises:
        FileNotFoundError: If `run_dir` does not exist.
    """
    run = Path(run_dir)
    if not run.is_dir():
        raise FileNotFoundError(f'run not found: {run}')
    out_path = Path(out) if out is not None else run.with_suffix('.zip')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = _iter_archive_files(
        run, with_bundle=with_bundle, with_cache=with_cache, with_tb=with_tb, best_only=best_only
    )
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for abs_path, arcname in files:
            zf.write(abs_path, arcname)
    _LOG.info(
        'Zipped %s (%d files) -> %s [%s]',
        run.name,
        len(files),
        out_path,
        human_size(out_path.stat().st_size),
    )
    if move:
        shutil.rmtree(run)
        _LOG.info('Moved: removed local %s after archiving.', run)
    return out_path


def zip_experiments(
    experiments_root: str | Path = 'res/experiments',
    names: list[str] | None = None,
    out: str | Path | None = None,
    *,
    with_bundle: bool = False,
    with_cache: bool = False,
    with_tb: bool = False,
    best_only: bool = False,
    move: bool = False,
) -> Path:
    """Zips several runs (default: all) into one archive, each under its own folder.

    Args:
        experiments_root: The experiments directory.
        names: (list[str]): Run names to include (default: every run directory found).
        out: (str | Path): Output `.zip` path (default `<experiments_root>/zte_experiments.zip`). Point it at a mounted Google Drive folder to upload straight to Drive.
        with_bundle: (bool): Include dataset bundles.
        with_cache: (bool): Include dataset caches.
        with_tb: (bool): Include TensorBoard logs.
        best_only: (bool): Keep only each run's `best.pt` checkpoint (smallest archive; inference-only).
        move: (bool): Delete each run directory after a successful zip (free local space).

    Returns:
        (Path): The written archive path.

    Raises:
        ValueError: If no matching runs are found.
    """
    root = Path(experiments_root)
    selected = (
        [root / n for n in names]
        if names
        else [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith('_')]
    )
    selected = [p for p in selected if p.is_dir()]
    if not selected:
        raise ValueError(f'no runs to zip under {root}')
    out_path = Path(out) if out is not None else root / 'zte_experiments.zip'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for run in selected:
            files = _iter_archive_files(
                run,
                with_bundle=with_bundle,
                with_cache=with_cache,
                with_tb=with_tb,
                best_only=best_only,
            )
            total += len(files)
            for abs_path, arcname in files:
                zf.write(abs_path, arcname)
    _LOG.info(
        'Zipped %d run(s), %d files -> %s [%s]',
        len(selected),
        total,
        out_path,
        human_size(out_path.stat().st_size),
    )
    if move:
        for run in selected:
            shutil.rmtree(run)
        _LOG.info('Moved: removed %d local run dir(s) after archiving.', len(selected))
    return out_path


def unpack(archive: str | Path, dest: str | Path = 'res/experiments') -> list[str]:
    """Extracts a run archive into `dest` (e.g. on your Mac after downloading).

    Args:
        archive: (str | Path): The path to a `.zip` produced by `zip_run` / `zip_experiments`.
        dest: (str | Path): Destination directory (run folders are created underneath).

    Returns:
        The list of top-level run names extracted.

    """
    arc = Path(archive)
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(arc) as zf:
        names = zf.namelist()
        zf.extractall(dest_path)
    tops = sorted({Path(n).parts[0] for n in names if n and not n.startswith('/')})
    _LOG.info('Unpacked %s -> %s (%d run folder(s))', arc, dest_path, len(tops))
    return tops


def delete_run(run_dir: str | Path, *, yes: bool = False) -> bool:
    """Deletes a run directory (guarded).

    Args:
        run_dir: (str | Path): The run directory to remove.
        yes: (bool): Must be `True` to actually delete; otherwise this is a dry run that only logs.

    Returns:
        `True` if the directory was deleted, `False` on a dry run or a missing directory.

    """
    run = Path(run_dir)
    if not run.is_dir():
        _LOG.warning('delete_run: %s does not exist', run)
        return False
    if not yes:
        _LOG.info(
            '[dry-run] would delete %s (%s). Pass yes=True to confirm.',
            run,
            human_size(_dir_size(run)),
        )
        return False
    shutil.rmtree(run)
    _LOG.info('Deleted %s', run)
    return True
