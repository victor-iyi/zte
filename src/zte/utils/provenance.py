"""Reproducibility metadata embedded in archived runs by `zte-pack zip`.

Everything here is best-effort and never raises: missing git, an unreadable manifest or an absent package degrades to
`null`, so packing a run can never fail because provenance could not be gathered.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from zte.logging_utils import get_logger

_LOG = get_logger('utils.provenance')

#: Packages whose versions pin the numerical result (recorded for reproducibility).
_TRACKED_PACKAGES: tuple[str, ...] = (
    'zte',
    'torch',
    'numpy',
    'scipy',
    'scikit-learn',
    'pandas',
    'mne',
    'transformers',
    'sentence-transformers',
)


def _run_git(args: list[str]) -> str | None:
    """Runs a git command in the repo, returning stripped stdout or `None` on any failure."""
    try:
        out = subprocess.run(
            ['git', *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except OSError, subprocess.SubprocessError:
        return None


def git_info() -> dict[str, Any]:
    """Returns the current git commit/branch/dirty state (best-effort; `null`s off a repo)."""
    commit = _run_git(['rev-parse', 'HEAD'])
    status = _run_git(['status', '--porcelain'])
    return {
        'commit': commit,
        'branch': _run_git(['rev-parse', '--abbrev-ref', 'HEAD']),
        'dirty': bool(status) if status is not None else None,
        'remote': _run_git(['config', '--get', 'remote.origin.url']),
    }


def package_versions() -> dict[str, str | None]:
    """Returns installed versions of the packages that pin the numerical result."""
    out: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _accelerator() -> dict[str, Any] | None:
    """Returns accelerator info if torch is importable, else `None`."""
    try:
        from zte.utils.env import accelerator_info

        return accelerator_info()
    except Exception:  # noqa: BLE001
        return None


def _run_record(run_dir: Path) -> dict[str, Any]:
    """Summarises one run for the provenance manifest: its config + headline metrics/verdict."""
    record: dict[str, Any] = {'name': run_dir.name}
    cfg_path = run_dir / 'config.yaml'
    if cfg_path.is_file():
        try:
            record['config'] = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
        except OSError, yaml.YAMLError:
            record['config'] = None
    manifest_path = run_dir / 'manifest.json'
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except OSError, json.JSONDecodeError:
            manifest = {}

        # Keep the load-bearing summary, not the whole manifest (paths/figures are local-only).
        record['headline'] = {
            key: manifest.get(key)
            for key in ('final_train_loss', 'evaluation', 'verdict', 'dataset')
            if key in manifest
        }
    ckpt = run_dir / 'checkpoints' / 'best.pt'
    record['has_best_checkpoint'] = ckpt.is_file()
    return record


def build_provenance(
    run_dirs: list[Path] | None = None,
    *,
    note: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assembles the reproducibility manifest for a set of runs.

    Args:
        run_dirs (list[Path] | None): Run directories to summarise (config + headline metrics each).
        note (str | None): Optional free-text note to store alongside the metadata.
        now (datetime | None): Timestamp to stamp (defaults to the current UTC time).

    Returns:
        dict[str, Any]: JSON-serialisable, with `created_at`, `git`, `python`, `platform`, `packages`, `accelerator`,
            `command`, `note` and per-run `runs`.
    """
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        'created_at': stamp.isoformat(),
        'git': git_info(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': package_versions(),
        'accelerator': _accelerator(),
        'command': ' '.join(sys.argv),
        'note': note,
        'runs': [_run_record(Path(d)) for d in (run_dirs or [])],
    }


def provenance_markdown(prov: dict[str, Any]) -> str:
    """Renders a short human-readable `PROVENANCE.md` from a provenance dict.

    Args:
        prov (dict[str, Any]): Output of `build_provenance`.

    Returns:
        str: Markdown summarising how the bundle was produced and how to reproduce it.
    """
    git = prov.get('git') or {}
    pkgs = prov.get('packages') or {}
    commit = git.get('commit') or 'unknown'
    dirty = ' (dirty tree)' if git.get('dirty') else ''
    lines = [
        '# ZTE run bundle — provenance',
        '',
        f'- **Created:** {prov.get("created_at")}',
        f'- **Git commit:** `{commit}`{dirty} on `{git.get("branch")}`',
        f'- **Remote:** {git.get("remote") or "—"}',
        f'- **Python:** {prov.get("python")} · **Platform:** {prov.get("platform")}',
        '- **Package versions:** ' + ', '.join(f'{k}={v}' for k, v in pkgs.items() if v),
    ]
    if prov.get('note'):
        lines += ['', f'> {prov["note"]}']
    lines += ['', '## Runs', '']
    for run in prov.get('runs', []):
        head = (run.get('headline') or {}).get('evaluation') or {}
        verdict = (run.get('headline') or {}).get('verdict')
        lines.append(
            f'- **{run["name"]}** — best.pt: {run.get("has_best_checkpoint")}; '
            f'eval: `{json.dumps(head)[:160]}`' + (f'; verdict: `{json.dumps(verdict)[:120]}`' if verdict else '')
        )
    lines += [
        '',
        '## Reproduce',
        '',
        '```sh',
        f'git checkout {commit}',
        'uv sync --group all',
        'uv run zte-run --config <run>/config.yaml --root "<path/to/ZuCo Dataset>" --name <run>',
        '```',
        '',
        'Each run folder carries its exact resolved `config.yaml`; re-running it on the same commit '
        'and data reproduces the result. Headline metrics and the honesty verdict above travel with '
        'the checkpoint so a shared bundle is self-verifying.',
    ]
    return '\n'.join(lines) + '\n'
