"""Skip-if-done guards: an artifact already built from the same inputs is not rebuilt."""

import argparse
import datetime
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from zte.cli.support.io import read_json, write_json
from zte.logging_utils import get_logger

_LOG = get_logger('cli.done')

type Signature = dict[str, Any]
"""Everything an artifact was built from: the options, the data behind it, and the checkpoint it re-scores."""

# Hidden, and named after the artifact rather than the command, so two commands writing different outputs into one
# directory never read each other's record.
STAMP_PREFIX: Final[str] = '.zte-done-'

# Each of these says where an artifact goes, which raw source it is read from, or how the run is logged -- never
# what the artifact contains. The raw source is carried instead by the bundle key, which is identical on every
# machine: keeping `--root` here would rebuild every artifact the first time a Drive path changed.
IGNORED_OPTIONS: Final[frozenset[str]] = frozenset(
    {
        'bundle',
        'command',
        'device',
        'drive',
        'extract_dir',
        'force',
        'log_level',
        'num_workers',
        'out',
        'overwrite',
        'root',
        'synthetic_out',
    }
)
"""Parsed options excluded from every signature."""


def add_force_argument(parser: argparse.ArgumentParser) -> None:
    """Adds `--force`, which rebuilds artifacts that already match their inputs."""
    parser.add_argument(
        '--force',
        action='store_true',
        help='Rebuild even when the artifacts on disk were already built from these exact inputs.',
    )


def stamp_for(artifact: str | Path) -> Path:
    """Where the record for `artifact` lives -- hidden, beside it, named after it."""
    path = Path(artifact)

    return path.with_name(f'{STAMP_PREFIX}{path.stem}.json')


def checkpoint_digest(ckpt: str | Path) -> str:
    """The checkpoint's content hash -- the identity of the weights, not of the path they were copied to."""
    from zte.training.init import file_sha256

    return file_sha256(ckpt)


def signature(
    args: argparse.Namespace,
    *,
    tool: str,
    extra: Mapping[str, Any] | None = None,
    ignore: Iterable[str] = (),
) -> Signature:
    """Records what an artifact is about to be built from.

    Note:
        Every parsed option counts unless it is named in `IGNORED_OPTIONS` or `ignore`, so a knob added to a
        command tightens the guard by default instead of silently serving an artifact that predates it.

    Args:
        args (argparse.Namespace): The parsed arguments.
        tool (str): The command this signature belongs to, e.g. `'decode'`.
        extra (Mapping[str, Any] | None, optional): Inputs that are not options -- the bundle key, the checkpoint
            digest. Defaults to None.
        ignore (Iterable[str], optional): Further options that do not change the artifact. Defaults to ().

    Returns:
        Signature: A JSON-safe dict, ready to compare and to write.
    """
    ignored = IGNORED_OPTIONS | set(ignore)
    options = {name: value for name, value in vars(args).items() if name not in ignored}
    payload: Signature = {'tool': tool, 'options': options, **(extra or {})}

    # Through JSON once, so a Path or a tuple compares equal to the string or list it is read back as.
    return dict(json.loads(json.dumps(payload, sort_keys=True, default=str)))


def is_done(artifacts: Sequence[str | Path], expected: Signature, *, force: bool = False) -> bool:
    """Whether every artifact is on disk and was built from exactly `expected`.

    Note:
        Rebuilding is the answer to every doubt -- a missing file, an unreadable record, an input that moved.
        Minutes spent rebuilding cost less than one stale artifact read as a fresh result.

    Args:
        artifacts (Sequence[str | Path]): The files the command writes. The first names the record.
        expected (Signature): The signature of the invocation about to run.
        force (bool, optional): Rebuild regardless. Defaults to False.

    Returns:
        bool: True when there is nothing left to do.
    """
    if force:
        return False

    paths = [Path(a) for a in artifacts]
    if missing := [p.name for p in paths if not p.exists()]:
        # A first run has nothing to say: it skipped nothing and rejected nothing. A partial one does.
        if len(missing) < len(paths):
            _LOG.info('Rebuilding: %s missing from %s.', ', '.join(missing), paths[0].parent)

        return False

    stamp = stamp_for(paths[0])
    try:
        record = read_json(stamp) if stamp.is_file() else {}
    except (OSError, ValueError) as exc:
        _LOG.warning('Rebuilding %s: its record could not be read (%r).', paths[0].name, exc)

        return False

    recorded = record.get('signature')
    if recorded is None:
        _LOG.info('Rebuilding %s: it carries no record of what it was built from.', paths[0].name)

        return False

    if changed := _changed(recorded, expected):
        _LOG.info('Rebuilding %s: %s changed since it was written.', paths[0].name, ', '.join(changed))

        return False

    # `zte-run`'s own evaluation writes `generation.json` into the directory `zte-decode` defaults to, so a record
    # that matches its inputs is not yet proof it describes the bytes on disk.
    if rewritten := _rewritten(paths, record.get('artifacts')):
        _LOG.info('Rebuilding %s: %s changed on disk since it was recorded.', paths[0].name, ', '.join(rewritten))

        return False

    _LOG.info('Already done: %s was built from these exact inputs. Skipping; pass --force to rebuild.', paths[0])

    return True


def mark_done(artifacts: Sequence[str | Path], sig: Signature) -> Path:
    """Records what `artifacts` were built from, so an identical re-run skips them.

    Args:
        artifacts (Sequence[str | Path]): The files just written. The first names the record.
        sig (Signature): The signature they were built from.

    Returns:
        Path: The written record.
    """
    paths = [Path(a) for a in artifacts]

    return write_json(
        stamp_for(paths[0]),
        {
            'signature': sig,
            'artifacts': {p.name: p.stat().st_size for p in paths},
            'completed': datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds'),
        },
        default=str,
    )


def _rewritten(paths: Sequence[Path], sizes: Any) -> list[str]:
    """The artifacts whose size no longer matches the record -- a size, not an mtime, because mirroring resets one."""
    if not isinstance(sizes, dict):
        return []

    return [p.name for p in paths if sizes.get(p.name) != p.stat().st_size]


def _changed(recorded: Any, expected: Signature) -> list[str]:
    """The input names that differ between a record and this invocation, one level into nested blocks."""
    if not isinstance(recorded, dict):
        return ['the record itself']

    names: list[str] = []
    for key in sorted(set(recorded) | set(expected)):
        old, new = recorded.get(key), expected.get(key)
        if old == new:
            continue

        if isinstance(old, dict) and isinstance(new, dict):
            names += [f'{key}.{k}' for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
        else:
            names.append(key)

    return names
