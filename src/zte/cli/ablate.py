"""`zte-ablate` — generate single-variable ablation sweeps and diff their scoreboards.

Two subcommands:

    # 1) Emit a sweep that changes exactly one knob, ready to run with zte-run:
    zte-ablate generate --config experiments/sota_loso.yaml \
        --knob objective.subject_adversary_weight --values 0,1 --out-dir experiments/ablate_adv

    # 2) After running the pair, isolate the knob's contribution on the LOSO scoreboard:
    zte-ablate diff --knob objective.subject_adversary_weight \
        --baseline res/experiments/<base>/evaluation/metrics.json \
        --variant  res/experiments/<var>/evaluation/metrics.json

The whole point is that every claim is a *single-variable* comparison, so a metric delta is
attributable to one knob — a discipline earlier work could only apply to VICReg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zte.config import ZTEConfig
from zte.evaluation.ablation import diff_scoreboards, render_diff, single_variable_configs
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.ablate')


def parse_arguments() -> argparse.Namespace:
    """Parses `zte-ablate` arguments."""
    parser = argparse.ArgumentParser(
        description='Generate single-variable ablation sweeps and diff their scoreboards.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    gen = sub.add_parser('generate', help='Write a one-knob config sweep to run.')
    gen.add_argument('--config', required=True, help='Base experiment YAML.')
    gen.add_argument(
        '--knob', required=True, help="Dotted knob, e.g. 'objective.meaning_distill_weight'."
    )
    gen.add_argument('--values', required=True, help='Comma-separated values to sweep (e.g. 0,1).')
    gen.add_argument('--out-dir', required=True, help='Directory to write the config sweep into.')

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
    base = ZTEConfig.from_yaml(args.config)
    values = [v.strip() for v in args.values.split(',') if v.strip()]
    pairs = single_variable_configs(base, args.knob, values)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, cfg in pairs:
        path = out_dir / f'{cfg.run_name}.yaml'
        cfg.to_yaml(path)
        _LOG.info('Wrote %s (%s)', path, tag)
    print('\n'.join(str(out_dir / f'{cfg.run_name}.yaml') for _, cfg in pairs))


def _diff(args: argparse.Namespace) -> None:
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
