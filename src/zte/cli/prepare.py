"""`zte-prepare` -- build (or synthesise) a self-contained, processed ZuCo dataset bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zte.cli.support.sources import add_data_source_args, add_extract_dir, resolve_data_root
from zte.config import DatasetConfig, MissingConfig
from zte.data.cache import REMOTE_ENV_VAR
from zte.data.dataset import ZuCoDataset
from zte.data.synthetic import generate_synthetic_zuco
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.prepare')

# Stands in for the data root while cache keys are computed, so a fully-prepared project never resolves
# (or downloads) the raw ZuCo tree. Must not contain 'synthetic' -- that word is what the key keys on.
_PENDING_ROOT: str = '<unresolved>'


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

    parser.add_argument('--synthetic-out', type=Path, default=Path('res/data/synthetic_zuco'))
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
    parser.add_argument(
        '--configs',
        nargs='*',
        default=None,
        help='Prepare the dataset for these experiment YAMLs instead of the flag-built config: every '
        'distinct dataset is processed ONCE into the cache and never again, so later runs start warm. '
        'Accepts files, directories or globs; pass no value for every config under experiments/.',
    )
    parser.add_argument(
        '--cache-remote',
        type=str,
        default=None,
        dest='cache_remote',
        help='Persistent cache directory (e.g. a mounted Drive folder) layered behind --cache-dir. '
        f'Bundles are published there as soon as they are built. Defaults to ${REMOTE_ENV_VAR}.',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Report what is prepared and exit without building anything. Costs one metadata check per '
        'dataset, so it is safe to gate a session on even when the store lives on Drive.',
    )
    parser.add_argument(
        '--no-extract-cache',
        action='store_true',
        dest='no_extract_cache',
        help='Skip caching the raw .mat extraction (saves disk, but a new dataset config then has to '
        're-parse every .mat file instead of re-deriving in seconds).',
    )
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def _iter_config_paths(patterns: list[str] | None) -> list[Path]:
    """Expands files, directories and globs into a sorted list of experiment YAML paths."""
    patterns = patterns or ['experiments']
    found: set[Path] = set()
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            found.update(path.rglob('*.yaml'))
        elif path.is_file():
            found.add(path)
        else:
            found.update(Path().glob(pattern))
    return sorted(found)


def _prepare_configs(args: argparse.Namespace) -> None:
    """Builds every distinct dataset the given experiment configs need, once."""
    from zte.config import ZTEConfig
    from zte.data.cache import BundleStore

    paths = _iter_config_paths(args.configs)
    if not paths:
        raise SystemExit(f'No experiment configs matched {args.configs!r}.')

    # `root` is excluded from the cache key, so every config can be keyed BEFORE the raw data is
    # resolved. Only the synthetic/real distinction reaches the key, and the placeholder preserves it.
    placeholder = args.synthetic_out if args.synthetic else _PENDING_ROOT

    # One entry per distinct dataset; the configs that share it are recorded for the summary.
    wanted: dict[str, tuple[DatasetConfig, list[str]]] = {}
    for path in paths:
        try:
            cfg = ZTEConfig.from_yaml(path).dataset
        except (OSError, KeyError, TypeError, ValueError) as exc:
            _LOG.warning('Skipping %s: %r', path, exc)
            continue
        cfg.root = placeholder
        cfg.cache_dir = args.cache_dir
        cfg.cache_remote = args.cache_remote
        cfg.cache_extracts = not args.no_extract_cache
        key = ZuCoDataset(cfg)._cache_key()  # noqa: SLF001
        wanted.setdefault(key, (cfg, []))[1].append(path.stem)

    store = BundleStore.create(args.cache_dir, args.cache_remote)
    _LOG.info(
        'Preparing %d distinct dataset(s) for %d config(s) -> %s',
        len(wanted),
        len(paths),
        store.describe(),
    )

    # `has` only stats meta.json on each layer, so this costs nothing even when the store is on Drive.
    located = {key: store.has(key) for key in wanted}
    missing = sorted(key for key, where in located.items() if where is None)

    if missing and not args.check:
        root = _resolve_build_root(args)
        for key in missing:
            cfg, names = wanted[key]
            cfg.root = root
            _LOG.info('Building %s (needed by: %s) ...', key, ', '.join(sorted(names)))
            ZuCoDataset(cfg).build()
            located[key] = 'local'

    status_of = {'local': 'cached', 'persistent': 'on-drive', None: 'MISSING'}
    print(f'\nDataset cache: {store.describe()}\n')
    print(f'{"bundle":52s} {"status":9s} used by')
    print('-' * 110)
    for key in sorted(wanted):
        names = ', '.join(sorted(set(wanted[key][1])))
        print(f'{key:52s} {status_of[located[key]]:9s} {names}')

    still_missing = [key for key, where in located.items() if where is None]
    if still_missing:
        print(f'\n{len(still_missing)} dataset(s) not prepared yet.', end=' ')
        print('Re-run without --check to build them.' if args.check else 'See the errors above.')
    else:
        print('\nEvery dataset these configs need is prepared; no run will re-process anything.')
        print(
            'Entries marked on-drive are pulled down once, on demand, by the first run that needs them.'
        )


def _resolve_build_root(args: argparse.Namespace) -> Path:
    """Resolves (downloading or synthesising if needed) the raw data root -- only when a build is due."""
    if not args.synthetic:
        return resolve_data_root(args)

    generate_synthetic_zuco(
        args.synthetic_out,
        subjects=tuple(args.synthetic_subjects.split(',')),
        tasks=tuple(args.tasks.split(',')),
        n_sentences=args.synthetic_sentences,
    )
    return args.synthetic_out


def main() -> None:
    """Runs dataset preparation end-to-end based on parsed arguments."""
    args = parse_arguments()
    configure_logging(args.log_level)

    # `--configs` is the build-once-for-everything path; the flags below build a single ad-hoc dataset.
    if args.configs is not None:
        _prepare_configs(args)
        return

    # Resolve the data source, fabricating a synthetic tree when asked.
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

    # Build the dataset, then save the bundle and optional overview figures.
    config = DatasetConfig(
        root=root,
        tasks=tuple(args.tasks.split(',')),
        subjects=tuple(args.subjects.split(',')) if args.subjects else None,
        representation=args.representation,
        normalize=args.normalize,
        raw_window=args.raw_window,
        cache_dir=args.cache_dir,
        cache_remote=args.cache_remote,
        cache_extracts=not args.no_extract_cache,
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
