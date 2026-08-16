"""`zte-colab` -- every Colab capability as JSON on stdout, so the notebook kernel never imports ZTE."""

import argparse
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final

import yaml

from zte.config import DecoderConfig
from zte.data.schema import SUBJECTS_V1
from zte.device import device_plan
from zte.evaluation.analysis import collect_study, panel_builders
from zte.evaluation.analysis.collect import HEADLINES, dig
from zte.evaluation.interactive import generation_payload
from zte.logging_utils import configure_logging, get_logger
from zte.utils.env import accelerator_info, env_defaults, machine_resources, project_root
from zte.utils.mirror import mirror_tree
from zte.utils.session import DEFAULT_DRIVE_ROOT, DriveSession, RunRef, discover_runs, every_session, find_checkpoint

_LOG = get_logger('cli.colab')

# Colab's kernel is an older interpreter than the venv, so it renders payloads rather than computing them; these
# are the tiers it offers as trainable arms. `archive/` is frozen and superseded, so it is never offered.
ARM_TIERS: Final[tuple[str, ...]] = ('flagship', 'decoder', 'ablation', 'benchmark')
"""Experiment tiers `zte-colab arms` reads, in the order the notebook lists them."""

# Rebuildable or unreadable-from-Drive: `cache`/`bundle` are derived, `tb` is large and only ever read locally,
# and the rotation checkpoints are history a fresh VM cannot resume from. `last.pt` and `best.pt` always travel.
MIRROR_EXCLUDE_DIRS: Final[frozenset[str]] = frozenset({'cache', 'bundle', 'tb'})
"""Directories a session mirror never copies."""

MIRROR_EXCLUDE_FILES: Final[tuple[str, ...]] = ('ckpt_epoch*.pt',)
"""Filename patterns a session mirror never copies."""

# Metrics `zte-decode` scores every reading with; the notebook table shows the first two and can ask for more.
DEFAULT_READING_METRICS: Final[tuple[str, ...]] = ('content_f1', 'wer')
"""Per-reading scores `zte-colab readings` keeps unless asked for others."""

# Only reached by an artifact written before the floor travelled in its provenance; a dropped clause would read as
# a passing one, so the registered default stands in and the payload says which floor it used.
DEFAULT_MIN_PREFIX_KL: Final[float] = DecoderConfig().min_prefix_kl
"""Prefix-influence floor in nats used when a decode artifact does not record the run's own."""


def _venv_versions() -> dict[str, str]:
    """The interpreter and package version every `!uv run` command actually executes with."""
    try:
        installed = version('zte')
    except PackageNotFoundError:  # pragma: no cover -- a source checkout that was never installed
        installed = 'unknown'

    return {'python': platform.python_version(), 'zte': installed}


def _evenly_spaced(total: int, count: int) -> list[int]:
    """Indices spread across the whole range, so a sample is never just the first `count` rows."""
    if count >= total:
        return list(range(total))
    if count <= 1:
        return [0]

    step = (total - 1) / (count - 1)

    return sorted({round(i * step) for i in range(count)})


def _run_roots(drive_root: str | Path | None, experiments: list[Path] | None) -> tuple[list[Path], list[Path]]:
    """The dated Drive sessions and the full search order: Drive newest-first, then any explicit local root."""
    sessions = every_session(drive_root) if drive_root else []
    extra = experiments or [Path('res/experiments')]

    return sessions, [*sessions, *(p for p in extra if p not in sessions)]


# Pooled and word-level numbers travel beside the held-out ones because a reader chasing a suspicious result wants
# both; only the `held_out_*` keys are the result, and `pooled_top1` inverted the champion once already.
def _headline_of(run: RunRef) -> dict[str, Any] | None:
    """Every `HEADLINES` metric of a run, or None when it was never evaluated."""
    metrics_path = run.path / 'evaluation' / 'metrics.json'
    if not metrics_path.is_file():
        return None

    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    headline: dict[str, Any] = {name: dig(metrics, *path) for name, path in HEADLINES.items()}
    headline['holdout_subject'] = dig(metrics, 'scoreboard', 'holdout_subject')
    headline['postprocess_fit'] = metrics.get('postprocess_fit')
    headline['length_projection'] = metrics.get('length_projection')
    headline['verdict'] = dig(metrics, 'scoreboard', 'verdict')

    return headline


def _run_entry(run: RunRef, *, headline: bool) -> dict[str, Any]:
    """One run as the notebook sees it: where it is, whether it is usable, and its checkpoints."""
    entry: dict[str, Any] = {
        'name': run.name,
        'path': str(run.path),
        'source': run.source,
        'session': run.session,
        'synthetic': run.synthetic,
        'evaluated': run.evaluated,
        'checkpoints': {
            which: str(path) if (path := find_checkpoint(run.name, [run.path.parent], which)) else None
            for which in ('best', 'last')
        },
    }
    if headline:
        entry['headline'] = _headline_of(run)
        figures = sorted((run.path / 'evaluation' / 'figures').glob('*.png'))
        entry['figures'] = [str(p) for p in figures]

    return entry


def _arm_label(path: Path, config: dict[str, Any]) -> str:
    """A config's own leading `# <stem> -- <what it is>` comment, falling back to its run name."""
    first = path.read_text(encoding='utf-8').lstrip().splitlines()[:1]
    if first and first[0].startswith('#'):
        return first[0].lstrip('# ').strip()

    return str(config.get('run_name') or path.stem)


def _unscoreable_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    """A failing verdict for a block that could not be scored, because a clause that cannot be read must not vanish."""
    return {
        'above_controls': False,
        'clauses': {'applicable': False},
        'reason': payload.get('reason', 'no generation block'),
        'controls_absent': [],
        'controls_missing': [],
    }


def read_decode_artifacts(
    directory: str | Path,
    *,
    rows: int = 12,
    pick: list[int] | None = None,
    metrics: tuple[str, ...] = DEFAULT_READING_METRICS,
) -> dict[str, Any]:
    """Reads a `zte-decode` output directory into the payload the notebook renders, verdict included.

    Note:
        This decodes nothing. `zte-decode` already scored every held-out reading against every pre-registered
        control, so recomputing a subset here would show numbers the verdict does not gate on.

    Args:
        directory (str | Path): A `zte-decode --out` directory holding `generation.json` and `generation.jsonl`.
        rows (int, optional): How many readings to return, spread across the split. Defaults to 12.
        pick (list[int] | None, optional): Exact reading indices, overriding `rows`. Defaults to None.
        metrics (tuple[str, ...], optional): Per-reading scores to keep. Defaults to `DEFAULT_READING_METRICS`.

    Returns:
        dict[str, Any]: `source`, `applicable`, `summary`, `deltas`, `verdict`, `readings` and `provenance`. A block
        that could not be scored comes back with `applicable: False`, a `reason`, and a verdict that fails rather
        than an absent one.

    Raises:
        FileNotFoundError: If either decode artifact is missing.
    """
    out_dir = Path(directory)
    report_path, rows_path = out_dir / 'generation.json', out_dir / 'generation.jsonl'
    for path in (report_path, rows_path):
        if not path.is_file():
            raise FileNotFoundError(f'{path} is missing; run zte-decode --out {out_dir} first.')

    report = json.loads(report_path.read_text(encoding='utf-8'))
    provenance = report.get('provenance') or {}
    block = report.get('generation') or {}

    # The gate the notebook prints has to be the gate the run configured; an artifact written before the floor was
    # recorded falls back to the packaged default rather than dropping the clause.
    floor = provenance.get('min_prefix_kl')
    payload = generation_payload(
        block,
        str(provenance.get('run_name') or out_dir.name),
        float(floor) if floor is not None else DEFAULT_MIN_PREFIX_KL,
    )

    records = [json.loads(line) for line in rows_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    chosen = [i for i in (pick if pick is not None else _evenly_spaced(len(records), rows)) if 0 <= i < len(records)]

    readings: list[dict[str, Any]] = []
    for index in chosen:
        record = records[index]
        conditions = [{'name': 'hypothesis', 'text': record.get('hypothesis'), 'scores': record.get('scores') or {}}]
        conditions.extend(
            {'name': name, 'text': control.get('text'), 'scores': control.get('scores') or {}}
            for name, control in (record.get('controls') or {}).items()
        )
        if record.get('oracle'):
            conditions.append({'name': 'oracle', **record['oracle']})
        readings.append(
            {
                'index': record.get('index', index),
                'subject': record.get('subject'),
                'task': record.get('task'),
                'n_words': record.get('n_words'),
                'target': record.get('reference'),
                'prefix_influence_kl': record.get('prefix_influence_kl'),
                'conditions': [
                    {**c, 'scores': {m: c['scores'][m] for m in metrics if m in c['scores']}} for c in conditions
                ],
            }
        )

    applicable = bool(payload.get('applicable'))

    return {
        'source': {
            'generation_json': str(report_path),
            'generation_jsonl': str(rows_path),
            'run_name': payload.get('run_name'),
            'split': payload.get('split'),
            'split_strategy': payload.get('split_strategy'),
            'n_total': len(records),
            'n_scored': payload.get('n'),
            'n_shown': len(readings),
        },
        'applicable': applicable,
        'reason': payload.get('reason'),
        'primary_metric': payload.get('primary_metric'),
        'conditions': payload.get('condition_order') or [],
        'summary': payload.get('absolute') or {},
        'deltas': payload.get('deltas') or {},
        'verdict': payload.get('verdict') or _unscoreable_verdict(payload),
        'readings': readings,
        'provenance': provenance,
    }


def _env(args: argparse.Namespace) -> dict[str, Any]:
    """Answers what interpreter, accelerator and machine the venv has."""
    root = project_root()

    return {
        'root': str(root),
        'env': env_defaults(root),
        'accelerator': accelerator_info(),
        'plan': device_plan(args.device),
        'resources': machine_resources(root),
        'venv': _venv_versions(),
    }


def _session(args: argparse.Namespace) -> dict[str, Any]:
    """Resolves (and creates) this session's Drive folders."""
    session = DriveSession.create(
        args.drive,
        run_date=args.resume_date,
        write_mode=args.write_mode,
        make_dirs=args.create,
    )

    return {**session.as_dict(), 'env': session.env()}


def _runs(args: argparse.Namespace) -> dict[str, Any]:
    """Lists every run reachable on Drive and locally, newest session first."""
    sessions, roots = _run_roots(args.drive, args.experiments)
    found = [run for run in discover_runs(roots) if args.run is None or run.name == args.run]

    return {
        'drive_root': str(args.drive) if args.drive else None,
        'sessions': [str(p) for p in sessions],
        'roots': [str(p) for p in roots],
        'runs': [_run_entry(run, headline=args.headline) for run in found],
    }


def _arms(args: argparse.Namespace) -> dict[str, Any]:
    """Reads the trainable arms straight off `experiments/`, so a promoted config needs no notebook edit."""
    arms: list[dict[str, Any]] = []
    for tier in args.tiers:
        for path in sorted((args.experiments / tier).glob('*.yaml')):
            config = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
            train, objective = config.get('train') or {}, config.get('objective') or {}
            kind = 'decoder' if train.get('mode') == 'decoder' or objective.get('name') == 'decode' else 'encoder'
            if args.kind not in ('any', kind):
                continue

            arms.append(
                {
                    'path': str(path),
                    'stem': path.stem,
                    'run_name': config.get('run_name'),
                    'tier': tier,
                    'kind': kind,
                    'label': _arm_label(path, config),
                    'objective': objective.get('name'),
                    'mode': train.get('mode'),
                }
            )

    return {'arms': arms, 'holdouts': list(SUBJECTS_V1)}


def _panels(args: argparse.Namespace) -> dict[str, Any]:
    """Draws the study's charts to figure JSON, so a kernel with plotly and no ZTE can render them inline."""
    study = collect_study([str(root) for root in args.experiments])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {name.strip() for name in args.only.split(',') if name.strip()} if args.only else None

    drawn: list[dict[str, Any]] = []
    empty: list[str] = []
    for panel in panel_builders(study, args.montage):
        if wanted is not None and panel.name not in wanted:
            continue

        try:
            figure = panel.build()
        except (KeyError, ValueError, TypeError) as exc:
            _LOG.warning('Panel %r skipped (%r).', panel.name, exc)
            figure = None

        # A panel nothing was collected for is named rather than dropped: a chart that silently vanishes reads as
        # a chart that had nothing to say.
        if figure is None:
            empty.append(panel.name)
            continue

        path = out_dir / f'{panel.name}.json'
        path.write_text(figure.to_json(), encoding='utf-8')
        drawn.append({'name': panel.name, 'section': panel.section, 'caption': panel.caption, 'path': str(path)})

    runs = study.runs
    synthetic = int((~runs['real_data'].astype(bool)).sum()) if 'real_data' in runs else 0
    _LOG.info('Wrote %d panel(s) to %s; %d had no data.', len(drawn), out_dir, len(empty))

    return {
        'out': str(out_dir),
        'roots': [str(root) for root in args.experiments],
        'study': {
            'runs': len(runs),
            'folds': len(study.folds),
            'generations': len(study.generations),
            'synthetic_runs': synthetic,
        },
        'panels': drawn,
        'empty': empty,
    }


def _mirror(args: argparse.Namespace) -> dict[str, Any]:
    """Copies a session between the VM's disk and Drive, skipping what is rebuildable."""
    session = DriveSession.create(args.drive, run_date=args.date, write_mode=args.write_mode, make_dirs=False)
    remote = session.session_dir / args.sub
    local = Path(args.local) if args.local else session.local_runs
    source, destination = (local, remote) if args.direction == 'up' else (remote, local)

    payload: dict[str, Any] = {
        'direction': args.direction,
        'src': str(source),
        'dst': str(destination),
        'exclude_dirs': sorted(MIRROR_EXCLUDE_DIRS),
        'exclude_files': list(MIRROR_EXCLUDE_FILES),
        'copied': 0,
        'failed': 0,
        'skipped_reason': None,
    }

    # Writing straight to Drive makes the mirror a copy onto itself, which would read as a successful backup.
    if source.resolve() == destination.resolve():
        payload['skipped_reason'] = 'source and destination are the same directory'
        return payload

    if not source.is_dir():
        payload['skipped_reason'] = f'{source} does not exist'
        return payload

    copied, failed = mirror_tree(
        source,
        destination,
        exclude_dirs=MIRROR_EXCLUDE_DIRS,
        exclude_files=MIRROR_EXCLUDE_FILES,
    )
    payload['copied'], payload['failed'] = copied, failed
    _LOG.info('Mirrored %s -> %s: %d copied, %d failed.', source, destination, copied, failed)

    return payload


def parse_arguments() -> argparse.Namespace:
    """Parses `zte-colab` arguments."""
    parser = argparse.ArgumentParser(
        description='Every Colab capability as JSON on stdout, so the notebook kernel never imports ZTE.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # A notebook writes the flag after the verb, so `--log-level` is a parent rather than a top-level argument.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--log-level', default='INFO')

    drive = argparse.ArgumentParser(add_help=False)
    drive.add_argument('--drive', default=DEFAULT_DRIVE_ROOT, help='Mounted-Drive root holding the dated sessions.')

    sub = parser.add_subparsers(dest='command', required=True)

    env = sub.add_parser('env', parents=[common], help='Interpreter, accelerator, device plan and machine limits.')
    env.add_argument('--device', default='auto', help="Device to plan for ('auto' picks the best available).")

    session = sub.add_parser('session', parents=[common, drive], help="This session's Drive folders and env vars.")
    session.add_argument('--resume-date', default=None, help="Reopen an earlier session, e.g. '2026-08-13'.")
    session.add_argument('--write-mode', default='local+mirror', choices=('local+mirror', 'drive'))
    session.add_argument('--create', action=argparse.BooleanOptionalAction, default=True)

    runs = sub.add_parser('runs', parents=[common, drive], help='Every run on Drive and locally, with checkpoints.')
    runs.add_argument('--experiments', nargs='*', type=Path, default=None, help='Extra local run roots.')
    runs.add_argument('--run', default=None, help='Only this run name.')
    runs.add_argument('--headline', action='store_true', help="Also read each run's held-out scoreboard block.")

    arms = sub.add_parser('arms', parents=[common], help='Trainable configs, read live from experiments/.')
    arms.add_argument('--kind', default='any', choices=('any', 'encoder', 'decoder'))
    arms.add_argument('--experiments', type=Path, default=Path('experiments'))
    arms.add_argument('--tiers', nargs='*', default=list(ARM_TIERS))

    readings = sub.add_parser('readings', parents=[common], help="A zte-decode run's readings, scored and gated.")
    readings.add_argument('--from', dest='source', required=True, type=Path, help='A zte-decode --out directory.')
    readings.add_argument('--rows', type=int, default=12, help='How many readings to return.')
    readings.add_argument('--pick', default=None, help='Exact reading indices, comma-separated.')
    readings.add_argument('--metrics', default=','.join(DEFAULT_READING_METRICS))

    panels = sub.add_parser('panels', parents=[common], help='Draw the study charts to figure JSON for a renderer.')
    panels.add_argument('--experiments', nargs='+', type=Path, required=True, help='Run roots to collect.')
    panels.add_argument('--out', type=Path, required=True, help='Directory the figure JSON is written to.')
    panels.add_argument('--only', default=None, help='Panel names to draw, comma-separated.')
    panels.add_argument('--montage', default=None, help='Montage CSV for the electrode map.')

    mirror = sub.add_parser('mirror', parents=[common, drive], help='Copy this session between the VM and Drive.')
    mirror.add_argument('--direction', default='up', choices=('up', 'down'))
    mirror.add_argument('--date', default=None, help='Session date; defaults to today.')
    mirror.add_argument('--sub', default='experiments', help="Folder under the session, e.g. 'experiments'.")
    mirror.add_argument('--local', default=None, help='Local side of the mirror.')
    mirror.add_argument('--write-mode', default='local+mirror', choices=('local+mirror', 'drive'))

    return parser.parse_args()


def main() -> None:
    """Entry point for the `zte-colab` console script."""
    args = parse_arguments()

    # The payload owns stdout, so every log line goes to stderr and the kernel can `json.loads` the whole stream.
    configure_logging(args.log_level, stderr=True)

    match args.command:
        case 'env':
            payload = _env(args)
        case 'session':
            payload = _session(args)
        case 'runs':
            payload = _runs(args)
        case 'arms':
            payload = _arms(args)
        case 'readings':
            payload = read_decode_artifacts(
                args.source,
                rows=args.rows,
                pick=[int(i) for i in args.pick.split(',')] if args.pick else None,
                metrics=tuple(m.strip() for m in args.metrics.split(',') if m.strip()),
            )
        case 'panels':
            payload = _panels(args)
        case 'mirror':
            payload = _mirror(args)
        case unknown:  # pragma: no cover -- argparse rejects any verb that has no subparser
            raise SystemExit(f'unhandled zte-colab subcommand {unknown!r}')

    print(json.dumps(payload, indent=2, default=str))


if __name__ == '__main__':
    main()
