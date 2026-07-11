"""Environment bootstrap so ZTE runs cleanly on Colab (and anywhere) without manual fiddling.

Colab does not set the environment variables headless plotting / tokenizer libraries expect, and the CLIs use paths relative to the repo
root — so the two ways a fresh Colab session errors are (a) a missing/writable-config env var and (b) the notebook's working directory not
being the repo root.  `bootstrap` fixes both: it sets sane defaults for the missing env vars (only when unset), resolves the project root,
changes into it if asked, and creates the `res/` output directories. It is idempotent and a no-op-safe on a laptop.
"""

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
    """Resolves the ZTE project root.

    Order: the `ZTE_HOME` env var, else the nearest ancestor of `start`/cwd containing a `pyproject.toml`, else the current working directory.

    Args:
        start: Where to begin the upward search (default: current working directory).

    Returns:
        The resolved project-root path.

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
    """Sets sane defaults for env vars Colab leaves unset (only when they are missing).

    Covers headless matplotlib (`MPLBACKEND`, a writable `MPLCONFIGDIR`), quiet tokenizers, and a
    writable Numba/font cache. Existing values are never overwritten.

    Args:
        overrides: Extra ``{name: value}`` to apply with the same "only if unset" rule.

    Returns:
        The mapping of env vars this call actually set (i.e. that were previously unset).
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
    # An inherited interactive/inline MPLBACKEND (Colab sets `module://…`) crashes a headless
    # subprocess, so replace it rather than treating it as "already set".
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
        root: Project root (default: `project_root`).

    Returns:
        The list of directories ensured.

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


def clean_outputs(
    targets: list[str] | None = None, root: str | Path | None = None, *, yes: bool = False
) -> list[Path]:
    """Deletes selected `res/` output subtrees — to free Colab space or start fresh.

    Args:
        targets: Names to remove, from ``experiments, data, cache, benchmark, explorer, embeddings``
            or ``all`` (the whole ``res/``). Defaults to ``['experiments']``.
        root: Project root (default: :func:`project_root`).
        yes: Must be `True` to actually delete; otherwise this is a dry run that only logs.

    Returns:
        The directories removed (empty on a dry run).
    """
    base = Path(root) if root is not None else project_root()
    names = targets or ['experiments']
    paths = (
        [base / _RES_TARGETS['all']]
        if 'all' in names
        else [base / _RES_TARGETS[n] for n in names if n in _RES_TARGETS]
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
        `{kind, name, torch_version, cuda, mps, tpu}` where `kind` is one of `cuda|mps|xla|cpu`
            (the backend ZTE's `--device auto` would pick).
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
    """One call to make any environment (Colab included) ready to run ZTE.

    Note:
        This function sets the missing env vars, optionally changes into the project root, ensures the output directories exist,
        and reports the accelerator.  It is safe and idempotent everywhere.

    Args:
        chdir (bool): If `True`, `os.chdir` into the resolved project root (handy in a notebook so the CLIs' relative `res/...` paths resolve).
        ensure (bool): Create the `res/` directories.
        quiet (bool): Suppress the summary log line.

    Returns:
        `{root, changed_dir, env_set, accelerator}`.

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
