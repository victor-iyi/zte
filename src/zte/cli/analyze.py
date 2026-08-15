"""`zte-analyze` -- read a whole study tree and write the analysis: one HTML page, its tables and its summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from zte.evaluation.analysis import build_dashboard, collect_study, write_summary, write_tables
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.analyze')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-analyze` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Collect every evaluated run under one or more experiment trees and write the study analysis: '
        'a self-contained interactive HTML page, the tidy CSV tables behind it, and a Markdown summary.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        default=['res/experiments'],
        help='One or more directories holding per-run folders. A Drive mirror and a local tree may be given '
        'together; a run present in both is read once.',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=None,
        help='Output directory. Default: <first experiments dir>/analysis.',
    )
    parser.add_argument('--title', type=str, default='ZTE — study analysis', help='Dashboard title.')
    parser.add_argument(
        '--montage',
        type=str,
        default=None,
        help='Montage CSV (channel,x,y,z) for the 3-D electrode map. Without it the map is an approximate spiral '
        'and says so.',
    )
    parser.add_argument(
        '--max-generation-rows',
        type=int,
        default=40000,
        dest='max_generation_rows',
        help='Cap on per-sentence decode rows loaded across all runs.',
    )
    parser.add_argument(
        '--tables',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Also write every tidy frame as CSV, so the analysis can be redone in another tool.',
    )
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    return parser.parse_args()


def main() -> None:
    """Collects the study and writes the dashboard, the tables and the summary."""
    args = parse_arguments()
    configure_logging(args.log_level)

    study = collect_study(args.experiments, max_generation_rows=args.max_generation_rows)
    out_dir = Path(args.out) if args.out else Path(args.experiments[0]) / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    if study.is_empty:
        _LOG.warning(
            'No evaluated run found under %s (each needs evaluation/metrics.json). Writing an empty analysis.',
            ', '.join(args.experiments),
        )

    page = build_dashboard(study, out_dir / 'ANALYSIS.html', title=args.title, montage_csv=args.montage)
    summary = write_summary(study, out_dir / 'ANALYSIS.md')
    if args.tables:
        write_tables(study, out_dir / 'tables')

    # A synthetic run is a wiring check, and the distinction has to survive into the terminal as well as the page.
    if not study.is_empty and 'real_data' in study.runs:
        synthetic = int((~study.runs['real_data'].astype(bool)).sum())
        if synthetic:
            _LOG.warning('%d of %d collected runs are SYNTHETIC and are not results.', synthetic, len(study.runs))

    _LOG.info('Analysis written: %s', page)
    print(page)
    print(summary)


if __name__ == '__main__':
    main()
