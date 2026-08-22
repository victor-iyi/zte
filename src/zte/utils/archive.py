"""Zip, unpack and delete training runs, for the cloud-to-local hand-off.

A run's checkpoint embeds its input shapes and fitted normaliser, so inference needs only the checkpoint; the default
archive therefore drops the dataset cache, TensorBoard logs and saved bundle.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from zte.logging_utils import get_logger

_LOG = get_logger('utils.archive')

# Directory names excluded from a run archive by default (heavy and not needed for inference).
_HEAVY_DIRS: set[str] = {'cache', 'tb', 'bundle'}


def _write_provenance(zf: zipfile.ZipFile, run_dirs: list[Path], note: str | None) -> None:
    """Writes reproducibility metadata (`PROVENANCE.json` + `PROVENANCE.md`) into the archive root.

    Best-effort: any failure to gather provenance is logged and skipped so it can never break packing.

    Args:
        zf (zipfile.ZipFile): The open archive to write into.
        run_dirs (list[Path]): The run directories included in this archive.
        note (str | None): Optional free-text note stored in the metadata.
    """
    try:
        from zte.utils.provenance import build_provenance, provenance_markdown

        prov = build_provenance(run_dirs, note=note)
        zf.writestr('PROVENANCE.json', json.dumps(prov, indent=2, default=str))
        zf.writestr('PROVENANCE.md', provenance_markdown(prov))
    except Exception as exc:  # noqa: BLE001 -- provenance is a nicety, never a hard requirement.
        _LOG.warning('Skipped provenance metadata: %r', exc)


def human_size(n: int) -> str:
    """Formats a byte count as a short human-readable string (e.g., `'12.3 MB'`)."""
    size = float(n)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size < 1024.0 or unit == 'TB':
            return f'{size:.1f} {unit}' if unit != 'B' else f'{int(size)} B'
        size /= 1024.0
    return f'{size:.1f} TB'


def _dir_size(path: Path) -> int:
    """Total size in bytes of all files under `path`."""
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def is_synthetic_run(run_dir: Path) -> bool:
    """Returns whether a run was produced with `--synthetic`, to keep smoke runs out of Drive backups.

    A run whose manifest is missing or lacks the flag counts as real, so real runs are never dropped by accident.

    Args:
        run_dir (Path): The run directory.

    Returns:
        bool: `True` only when `manifest.json` explicitly records `synthetic: true`.
    """
    manifest = run_dir / 'manifest.json'
    if not manifest.is_file():
        return False
    try:
        return bool(json.loads(manifest.read_text(encoding='utf-8')).get('synthetic', False))
    except OSError, json.JSONDecodeError:
        return False


def _run_is_complete(run_dir: Path) -> bool:
    """Heuristic: a run is 'complete' if it has a best/last checkpoint and an evaluation."""
    ckpt = (run_dir / 'checkpoints' / 'best.pt').exists() or (run_dir / 'checkpoints' / 'last.pt').exists()
    return ckpt and (run_dir / 'evaluation' / 'metrics.json').exists()


def list_runs(experiments_root: str | Path = 'res/experiments') -> list[dict]:
    """Lists catalogued runs with sizes and a completeness flag.

    Args:
        experiments_root (str | Path): The directory holding per-run folders.

    Returns:
        list[dict]: `{name, path, size_bytes, size, has_checkpoint, complete}` per run, largest first.
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
        run_dir (str | Path): The run directory.
        with_bundle (bool): Include the saved dataset bundle.
        with_cache (bool): Include the processed dataset cache.
        with_tb (bool): Include TensorBoard logs.
        best_only (bool): Keep only the best checkpoint.
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
        # best_only keeps checkpoints/best.pt, dropping last.pt and every ckpt_epoch*.pt.
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
    note: str | None = None,
) -> Path:
    """Zips a single run into an archive suitable for download.

    Args:
        run_dir (str | Path): The run directory (`res/experiments/<name>`).
        out (str | Path | None): Output `.zip` path (default `<run>.zip` next to the run); point it at a mounted
            Google Drive folder to upload straight to Drive.
        with_bundle (bool): Include the saved dataset bundle, needed to re-evaluate but not to infer.
        with_cache (bool): Include the processed dataset cache (large).
        with_tb (bool): Include TensorBoard event logs.
        best_only (bool): Keep only `checkpoints/best.pt` -- enough for inference, not for resuming training.
        move (bool): Delete the run directory after a successful zip.
        note (str | None): Free-text note stored in the archive's provenance metadata.

    Returns:
        Path: The written archive path.

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
        _write_provenance(zf, [run], note)
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
    note: str | None = None,
    skip_synthetic: bool = False,
) -> Path:
    """Zips several runs (default: all) into one archive, each under its own folder.

    Args:
        experiments_root (str | Path): The experiments directory.
        names (list[str] | None): Run names to include (default: every run directory found).
        out (str | Path | None): Output `.zip` path (default `<experiments_root>/zte_experiments.zip`).
        with_bundle (bool): Include dataset bundles.
        with_cache (bool): Include dataset caches.
        with_tb (bool): Include TensorBoard logs.
        best_only (bool): Keep only each run's `best.pt` checkpoint (inference-only).
        move (bool): Delete each run directory after a successful zip.
        note (str | None): Free-text note stored in the archive's provenance metadata.
        skip_synthetic (bool): Drop runs produced with `--synthetic`.

    Returns:
        Path: The written archive path.

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
    if skip_synthetic:
        real = [p for p in selected if not is_synthetic_run(p)]
        dropped = len(selected) - len(real)
        if dropped:
            _LOG.info('Skipping %d synthetic run(s) from the archive.', dropped)
        selected = real
    if not selected:
        raise ValueError(f'no runs to zip under {root} (after skip_synthetic filtering)')
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
        _write_provenance(zf, selected, note)
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


#: Default res/ subtrees for a full "continue locally" snapshot.
_SNAPSHOT_TARGETS: tuple[str, ...] = ('experiments', 'cache', 'benchmark', 'explorer')


def zip_res(
    res_root: str | Path = 'res',
    targets: list[str] | tuple[str, ...] | None = None,
    out: str | Path | None = None,
    *,
    note: str | None = None,
    move: bool = False,
    skip_synthetic: bool = False,
) -> Path:
    """Zips whole `res/` subtrees into one archive so a local session can continue without re-training.

    Where `zip_experiments` packs per-run checkpoints for inference, this captures the heavier working state under its
    `res/`-relative paths; `unpack(archive, dest='res')` restores it verbatim.

    Args:
        res_root (str | Path): The `res/` directory root.
        targets (list[str] | tuple[str, ...] | None): Subtree names to include (default:
            `experiments`, `cache`, `benchmark`, `explorer`). Missing subtrees are skipped.
        out (str | Path | None): Output `.zip` path (default `<res_root>/zte_snapshot.zip`). Point it at a
            mounted Drive folder to upload straight to Drive.
        note (str | None): Optional free-text note stored in the archive provenance.
        move (bool): Delete the archived subtrees after a successful zip.
        skip_synthetic (bool): Drop runs produced with `--synthetic`.

    Returns:
        Path: The written archive path.

    Raises:
        ValueError: If none of the requested subtrees exist.
    """
    root = Path(res_root)
    names = tuple(targets) if targets else _SNAPSHOT_TARGETS
    subtrees = [root / n for n in names if (root / n).is_dir()]
    if not subtrees:
        raise ValueError(f'no res/ subtrees to snapshot under {root} (looked for {names}).')
    out_path = Path(out) if out is not None else root / 'zte_snapshot.zip'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Synthetic run dirs to exclude, since smoke runs should not be shipped to Drive.
    synthetic_dirs: set[Path] = set()
    exp_root = root / 'experiments'
    if skip_synthetic and exp_root.is_dir():
        synthetic_dirs = {d for d in exp_root.iterdir() if d.is_dir() and is_synthetic_run(d)}
        if synthetic_dirs:
            _LOG.info('Snapshot: skipping %d synthetic run(s).', len(synthetic_dirs))
    total = 0
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for sub in subtrees:
            for p in sorted(sub.rglob('*')):
                if p.is_file() and not any(sd in p.parents for sd in synthetic_dirs):
                    zf.write(p, str(p.relative_to(root)))
                    total += 1
        run_dirs = (
            [
                d
                for d in sorted(exp_root.iterdir())
                if d.is_dir() and not d.name.startswith('_') and d not in synthetic_dirs
            ]
            if exp_root.is_dir()
            else []
        )
        _write_provenance(zf, run_dirs, note)
    _LOG.info(
        'Snapshot: %d file(s) from %s -> %s [%s]',
        total,
        ', '.join(s.name for s in subtrees),
        out_path,
        human_size(out_path.stat().st_size),
    )
    if move:
        for sub in subtrees:
            shutil.rmtree(sub)
        _LOG.info('Moved: removed %d local subtree(s) after snapshot.', len(subtrees))
    return out_path


def unpack(archive: str | Path, dest: str | Path = 'res/experiments') -> list[str]:
    """Extracts a run archive into `dest` (e.g. on your machine after downloading).

    Args:
        archive (str | Path): Path to a `.zip` produced by `zip_run` / `zip_experiments`.
        dest (str | Path): Destination directory (run folders are created underneath).

    Returns:
        list[str]: The top-level run names extracted.
    """
    arc = Path(archive)
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(arc) as zf:
        names = zf.namelist()
        zf.extractall(dest_path)

    # Root-level files (PROVENANCE.json/md) are metadata, so only nested top-level folders count as runs.
    tops = sorted({Path(n).parts[0] for n in names if n and not n.startswith('/') and len(Path(n).parts) > 1})
    _LOG.info('Unpacked %s -> %s (%d run folder(s))', arc, dest_path, len(tops))
    return tops


def delete_run(run_dir: str | Path, *, yes: bool = False) -> bool:
    """Deletes a run directory (guarded).

    Args:
        run_dir (str | Path): The run directory to remove.
        yes (bool): Must be `True` to actually delete; otherwise this is a dry run that only logs.

    Returns:
        bool: `True` if the directory was deleted, `False` on a dry run or a missing directory.
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
