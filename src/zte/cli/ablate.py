"""`zte-ablate` -- generate single-variable (or grid) ablation sweeps and diff their scoreboards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zte.cli.support.provision import add_provision_args, provision_from_args
from zte.config import ZTEConfig
from zte.evaluation.ablation import diff_scoreboards, grid_configs, render_diff
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.ablate')


def parse_arguments() -> argparse.Namespace:
    """Parses `zte-ablate` arguments."""
    parser = argparse.ArgumentParser(
        description='Generate single-variable ablation sweeps and diff their scoreboards.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    gen = sub.add_parser(
        'generate',
        help='Write a config sweep to run: one knob (single-variable) or several (a value grid).',
    )
    gen.add_argument('--config', required=True, type=Path, help='Base experiment YAML.')
    gen.add_argument(
        '--knob',
        required=True,
        action='append',
        help="Dotted knob, e.g. 'objective.meaning_distill_weight'. Repeat --knob (paired with a "
        '--values) to sweep the Cartesian product of several knobs.',
    )
    gen.add_argument(
        '--values',
        required=True,
        action='append',
        help='Comma-separated values for the matching --knob (e.g. 0,1). Repeat once per --knob.',
    )
    gen.add_argument(
        '--out-dir', required=True, type=Path, help='Directory to write the config sweep into.'
    )
    add_provision_args(gen)

    dif = sub.add_parser('diff', help='Diff two finished runs on the honest scoreboard.')
    dif.add_argument('--knob', required=True, help='The knob that was varied (for the header).')
    dif.add_argument('--baseline', required=True, type=Path, help="Baseline run's metrics.json.")
    dif.add_argument(
        '--variant', required=True, type=Path, help="One-knob-changed run's metrics.json."
    )
    dif.add_argument('--out', type=Path, default=None, help='Optional Markdown output path.')

    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def _generate(args: argparse.Namespace) -> None:
    """Writes one config per point of the knob grid."""
    if len(args.knob) != len(args.values):
        raise SystemExit(
            f'Got {len(args.knob)} --knob but {len(args.values)} --values; pass one --values per --knob.'
        )
    # Provision the base once so every arm shares it, while a knob can still override per arm.
    base = ZTEConfig.from_yaml(args.config)
    provision_from_args(base, args)

    specs = [
        (knob, [v.strip() for v in values.split(',') if v.strip()])
        for knob, values in zip(args.knob, args.values)
    ]
    pairs = grid_configs(base, specs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, cfg in pairs:
        path = out_dir / f'{cfg.run_name}.yaml'
        cfg.to_yaml(path)
        _LOG.info('Wrote %s (%s)', path, tag)

    _LOG.info('Generated %d config(s) across %d knob(s).', len(pairs), len(specs))
    print('\n'.join(str(out_dir / f'{cfg.run_name}.yaml') for _, cfg in pairs))


def _diff(args: argparse.Namespace) -> None:
    """Renders the scoreboard delta between a baseline run and its one-knob variant."""
    diff = diff_scoreboards(args.baseline, args.variant)
    md = render_diff(
        args.knob,
        f'baseline ({args.baseline.parent.parent.name})',
        f'variant ({args.variant.parent.parent.name})',
        diff,
    )

    if args.out:
        args.out.write_text(md, encoding='utf-8')
        _LOG.info('Ablation diff written to %s', args.out)
    print(md)
    print('\n' + json.dumps(diff, indent=2))


def main() -> None:
    """Entry point for the `zte-ablate` console script."""
    args = parse_arguments()
    configure_logging(args.log_level)
    if args.command == 'generate':
        _generate(args)
    else:
        _diff(args)


if __name__ == '__main__':
    main()
