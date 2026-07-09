"""`zte-prepare` -- build (or synthesise) a processed ZuCo bundle.

Loads ZuCo `.mat` files (or generates a synthetic tree), processes them into a `ZuCoDataset` with the chosen missing-value strategy
and normalisation, optionally renders the analysis figures, and saves a self-contained bundle that `zte.cli.train` can consume directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zte.cli.sources import add_data_source_args, add_extract_dir, resolve_data_root
from zte.config import DatasetConfig, MissingConfig
from zte.data.dataset import ZuCoDataset
from zte.data.synthetic import generate_synthetic_zuco
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.prepare')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-prepare` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.

    """
    parser = argparse.ArgumentParser(
        description='Prepare a processed ZuCo dataset bundle.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_source_args(parser, include_synthetic=True)
    add_extract_dir(parser)

    parser.add_argument('--synthetic-out', type=str, default='res/data/synthetic_zuco')
    parser.add_argument('--synthetic-subjects', type=str, default='ZAB,ZDM,ZJN')
    parser.add_argument('--synthetic-sentences', type=int, default=12)
    parser.add_argument('--tasks', type=str, default='SR,NR,TSR', help='Comma-separated tasks.')
    parser.add_argument('--subjects', type=str, default=None, help='Comma-separated subjects.')
    parser.add_argument(
        '--representation',
        choices=['band_power', 'raw', 'both'],
        default='band_power',
    )
    parser.add_argument(
        '--missing-method',
        choices=[
            'zero',
            'row_mean',
            'col_mean',
            'global_mean',
            'median',
            'knn',
            'iterative',
            'ffill',
            'interpolate',
            'drop',
            'mask_only',
        ],
        default='mask_only',
        help='Missing-value strategy.',
    )
    parser.add_argument(
        '--normalize',
        choices=['zscore_channel', 'zscore_global', 'minmax', 'none'],
        default='zscore_channel',
    )
    parser.add_argument('--raw-window', type=int, default=128)
    parser.add_argument('--cache-dir', type=str, default='res/cache')
    parser.add_argument('--out', type=str, default='res/bundle')
    parser.add_argument('--figures', type=str, default=None, help='Dir to write overview figures.')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def main() -> None:
    """Runs dataset preparation end-to-end based on parsed arguments."""
    args = parse_arguments()
    configure_logging(args.log_level)

    root = resolve_data_root(args) if not args.synthetic else None
    if args.synthetic:
        subjects = tuple(args.synthetic_subjects.split(','))
        tasks = tuple(args.tasks.split(','))
        generate_synthetic_zuco(
            args.synthetic_out,
            subjects=subjects,
            tasks=tasks,
            n_sentences=args.synthetic_sentences,
        )
        root = args.synthetic_out

    config = DatasetConfig(
        root=root,
        tasks=tuple(args.tasks.split(',')),
        subjects=tuple(args.subjects.split(',')) if args.subjects else None,
        representation=args.representation,
        normalize=args.normalize,
        raw_window=args.raw_window,
        cache_dir=args.cache_dir,
        missing=MissingConfig(method=args.missing_method),
    )
    dataset = ZuCoDataset(config).build()
    _LOG.info('Built dataset: %r', dataset)
    print(json.dumps(dataset.analyze(), indent=2, default=str))

    dataset.save(args.out)
    if args.figures:
        from zte.data.viz import save_overview  # pylint: disable=import-outside-toplevel

        save_overview(dataset, args.figures)
    _LOG.info('Saved bundle to %s', Path(args.out).resolve())


if __name__ == '__main__':
    main()
