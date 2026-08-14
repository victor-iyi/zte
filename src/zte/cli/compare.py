"""`zte-compare` -- combine every catalogued run into one offline interactive comparison dashboard."""

from __future__ import annotations

import argparse

from zte.evaluation.interactive import build_comparison
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.compare')


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for `zte-compare`."""
    parser = argparse.ArgumentParser(
        description='Combine all catalogued ZTE runs into one interactive comparison dashboard.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--experiments',
        type=str,
        default='res/experiments',
        help='Directory holding the per-run folders to compare.',
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Output HTML path (default: <experiments>/COMPARE.html).',
    )
    parser.add_argument('--title', type=str, default='ZTE — Experiment Comparison', help='Dashboard title.')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def main() -> None:
    """Entry point for the `zte-compare` console script."""
    args = parse_arguments()
    configure_logging(args.log_level)
    out = build_comparison(args.experiments, args.out, title=args.title)
    _LOG.info('Comparison dashboard written to %s', out)
    print(out)


if __name__ == '__main__':
    main()
