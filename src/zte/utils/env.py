"""Environment bootstrap: env-var defaults, project-root resolution and `res/` output directories."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from zte.logging_utils import get_logger

_LOG = get_logger('utils.env')

# Named `res/` output subtrees that are safe to wipe to free space / start fresh.
_RES_TARGETS: dict[str, str] = {
    'experiments': 'res/experiments',
    'data': 'res/data',
    'cache': 'res/cache',
    'benchmark': 'res/benchmark',
    'explorer': 'res/explorer',
    'embeddings': 'res/embeddings',
    'all': 'res',
}


def project_root(start: str | Path | None = None) -> Path:
    """Resolves the project root: `ZTE_HOME`, else the nearest ancestor holding a `pyproject.toml`, else the cwd.

    Args:
        start (str | Path | None): Where to begin the upward search (default: current working directory).

    Returns:
        Path: The resolved project-root path.
    """
    env = os.environ.get('ZTE_HOME')
    if env:
        return Path(env).expanduser().resolve()
    here = Path(start).resolve() if start else Path.cwd()
    for candidate in (here, *here.parents):
        if (candidate / 'pyproject.toml').is_file():
            return candidate
    return Path.cwd()


def set_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Sets defaults for headless matplotlib, quiet tokenizers and writable caches, never overwriting existing values.

    Args:
        overrides (dict[str, str] | None): Extra `{name: value}` applied with the same "only if unset" rule.

    Returns:
        dict[str, str]: The env vars this call actually set.
    """
    root = project_root()
    cache = root / 'res' / '.cache'
    defaults = {
        'MPLBACKEND': 'Agg',
        'MPLCONFIGDIR': str(cache / 'matplotlib'),
        'TOKENIZERS_PARALLELISM': 'false',
        'NUMBA_CACHE_DIR': str(cache / 'numba'),
        'XDG_CACHE_HOME': str(cache),
    }
    if overrides:
        defaults.update(overrides)
    applied: dict[str, str] = {}

    # An inherited `module://` MPLBACKEND crashes a headless subprocess, so replace it rather than treat it as set.
    current_mpl = os.environ.get('MPLBACKEND', '')
    if current_mpl.startswith('module://'):
        os.environ['MPLBACKEND'] = 'Agg'
        applied['MPLBACKEND'] = 'Agg'
    for key, value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value
    for sub in ('matplotlib', 'numba'):
        Path(cache / sub).mkdir(parents=True, exist_ok=True)
    return applied


def ensure_dirs(root: str | Path | None = None) -> list[Path]:
    """Creates the standard `res/` output directories so first-run writes never fail.

    Args:
        root (str | Path | None): Project root (default: `project_root`).

    Returns:
        list[Path]: The directories ensured.
    """
    base = Path(root) if root is not None else project_root()
    dirs = [
        base / 'res' / 'experiments',
        base / 'res' / 'data',
        base / 'res' / 'data' / '_downloads',
        base / 'res' / 'cache',
        base / 'res' / 'explorer',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def clean_outputs(targets: list[str] | None = None, root: str | Path | None = None, *, yes: bool = False) -> list[Path]:
    """Deletes selected `res/` output subtrees, to free space or start fresh.

    Args:
        targets (list[str] | None): Names from `_RES_TARGETS`, or `all` for the whole `res/`.
            Defaults to `['experiments']`.
        root (str | Path | None): Project root (default: `project_root`).
        yes (bool): Must be `True` to actually delete; otherwise this is a dry run that only logs.

    Returns:
        list[Path]: The directories removed (empty on a dry run).
    """
    base = Path(root) if root is not None else project_root()
    names = targets or ['experiments']
    paths = (
        [base / _RES_TARGETS['all']] if 'all' in names else [base / _RES_TARGETS[n] for n in names if n in _RES_TARGETS]
    )
    removed: list[Path] = []
    for p in paths:
        if not p.exists():
            _LOG.info('%s does not exist; nothing to remove.', p)
            continue
        if not yes:
            _LOG.info('[dry-run] would remove %s. Pass yes=True to confirm.', p)
            continue
        shutil.rmtree(p)
        _LOG.info('Removed %s', p)
        removed.append(p)
    return removed


def accelerator_info() -> dict[str, Any]:
    """Reports the available accelerator without raising, on any platform.

    Returns:
        dict[str, Any]: `{kind, name, torch_version, cuda, mps, tpu}`, where `kind` is the backend `--device auto`
        picks.
    """
    import torch

    from zte.device import _xla_device

    cuda = bool(torch.cuda.is_available())
    mps = bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())
    tpu = _xla_device() is not None
    if cuda:
        kind, name = 'cuda', torch.cuda.get_device_name(0)
    elif tpu:
        kind, name = 'xla', 'Cloud TPU (torch_xla)'
    elif mps:
        kind, name = 'mps', 'Apple Silicon (MPS)'
    else:
        kind, name = 'cpu', f'CPU ({os.cpu_count() or 1} cores)'
    return {
        'kind': kind,
        'name': name,
        'torch_version': torch.__version__,
        'cuda': cuda,
        'mps': mps,
        'tpu': tpu,
    }


def bootstrap(chdir: bool = False, *, ensure: bool = True, quiet: bool = False) -> dict[str, Any]:
    """Makes any environment ready to run ZTE: env vars, project root, output directories, accelerator report.

    Args:
        chdir (bool): `os.chdir` into the project root, so the CLIs' relative `res/...` paths resolve in a notebook.
        ensure (bool): Create the `res/` directories.
        quiet (bool): Suppress the summary log line.

    Returns:
        dict[str, Any]: `{root, changed_dir, env_set, accelerator}`.
    """
    root = project_root()
    env_set = set_env()
    changed = False
    if chdir and Path.cwd() != root:
        os.chdir(root)
        changed = True
    if ensure:
        ensure_dirs(root)
    accel = accelerator_info()
    if not quiet:
        _LOG.info(
            'ZTE ready | root=%s | device=%s (%s) | torch=%s | env_set=%s',
            root,
            accel['kind'],
            accel['name'],
            accel['torch_version'],
            ','.join(env_set) or 'none',
        )
    return {'root': str(root), 'changed_dir': changed, 'env_set': env_set, 'accelerator': accel}
