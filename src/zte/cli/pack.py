"""`zte-pack` — list, zip, unpack and delete training runs (the Colab↔local hand-off).

Train on a cloud GPU, `zte-pack zip --all` into a small archive, download it, and `zte-pack unpack` it on your Mac for inference.
Heavy `cache/`, `tb/` and `bundle/` folders are excluded by default (a checkpoint already embeds what inference needs).
"""

from __future__ import annotations

import argparse

from zte.logging_utils import configure_logging, get_logger
from zte.utils.archive import (
    delete_run,
    human_size,
    list_runs,
    unpack,
    zip_experiments,
    zip_res,
    zip_run,
)
from zte.utils.env import clean_outputs

_LOG = get_logger('cli.pack')


def parse_arguments() -> argparse.Namespace:
    """Parses `zte-pack` arguments."""
    parser = argparse.ArgumentParser(
        description='List, zip, unpack and delete ZTE training runs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--experiments', default='res/experiments', help='Experiments directory.')
    parser.add_argument('--log-level', default='INFO')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('list', help='List runs with sizes and completeness.')

    z = sub.add_parser('zip', help='Zip one or more runs (or --all) for download.')
    z.add_argument('names', nargs='*', help='Run names to zip (default with --all: every run).')
    z.add_argument('--all', action='store_true', help='Zip every run into one archive.')
    z.add_argument('--out', default=None, help='Output .zip path.')
    z.add_argument('--with-bundle', action='store_true', help='Include the dataset bundle.')
    z.add_argument('--with-cache', action='store_true', help='Include the dataset cache (large).')
    z.add_argument('--with-tb', action='store_true', help='Include TensorBoard logs.')
    z.add_argument(
        '--best-only',
        action='store_true',
        help="Keep only each run's best.pt (drop last.pt + epoch checkpoints) — smallest, inference-only.",
    )
    z.add_argument(
        '--move',
        action='store_true',
        help='Delete the local run dir(s) after a successful zip (e.g. once the archive is on Drive).',
    )
    z.add_argument(
        '--note',
        default=None,
        help='Free-text note stored in the archive PROVENANCE metadata (e.g. "flagship real-data run").',
    )

    s = sub.add_parser(
        'snapshot',
        help='Zip whole res/ subtrees (experiments+cache+benchmark+explorer) into one archive to continue locally.',
    )
    s.add_argument(
        'targets',
        nargs='*',
        help='res/ subtrees to include (default: experiments cache benchmark explorer). Missing ones are skipped.',
    )
    s.add_argument('--res', default='res', help='The res/ root to snapshot from.')
    s.add_argument(
        '--out', default=None, help='Output .zip path (point at Drive to upload directly).'
    )
    s.add_argument(
        '--note', default=None, help='Free-text note stored in the archive PROVENANCE metadata.'
    )
    s.add_argument(
        '--move', action='store_true', help='Delete the archived subtrees after a successful zip.'
    )

    u = sub.add_parser(
        'unpack',
        help='Extract a run/snapshot archive (e.g. on your Mac). Use --dest res for a snapshot.',
    )
    u.add_argument('archive', help='Path to a .zip produced by zip / snapshot.')
    u.add_argument(
        '--dest', default='res/experiments', help='Where to extract (use res for a snapshot).'
    )

    d = sub.add_parser('delete', help='Delete run directories.')
    d.add_argument('names', nargs='+', help='Run names to delete.')
    d.add_argument('--yes', action='store_true', help='Actually delete (otherwise a dry run).')

    c = sub.add_parser('clean', help='Delete res/ output subfolders (free space / start fresh).')
    c.add_argument(
        'targets',
        nargs='*',
        help='Which res/ subtrees to remove: experiments data cache benchmark explorer embeddings, '
        'or "all" (the whole res/). Default: experiments.',
    )
    c.add_argument('--yes', action='store_true', help='Actually delete (otherwise a dry run).')
    return parser.parse_args()


def main() -> None:
    """Entry point for the `zte-pack` console script."""
    args = parse_arguments()
    configure_logging(args.log_level)
    root = args.experiments

    if args.command == 'list':
        rows = list_runs(root)
        if not rows:
            print(f'No runs under {root}.')
            return
        width = max((len(r['name']) for r in rows), default=4)
        print(f'{"run".ljust(width)}  {"size":>9}  ckpt  complete')
        for r in rows:
            print(
                f'{r["name"].ljust(width)}  {r["size"]:>9}  '
                f'{"yes" if r["has_checkpoint"] else " no"}   {"yes" if r["complete"] else " no"}'
            )
        total = sum(r['size_bytes'] for r in rows)
        print(f'\n{len(rows)} run(s), {human_size(total)} total.')

    elif args.command == 'zip':
        kw = {
            'with_bundle': args.with_bundle,
            'with_cache': args.with_cache,
            'with_tb': args.with_tb,
            'best_only': args.best_only,
            'move': args.move,
            'note': args.note,
        }
        if args.all or len(args.names) != 1:
            out = zip_experiments(root, names=args.names or None, out=args.out, **kw)
        else:
            out = zip_run(f'{root}/{args.names[0]}', out=args.out, **kw)
        print(out)

    elif args.command == 'snapshot':
        out = zip_res(
            args.res,
            targets=args.targets or None,
            out=args.out,
            note=args.note,
            move=args.move,
        )
        print(out)

    elif args.command == 'unpack':
        names = unpack(args.archive, args.dest)
        print('\n'.join(names))

    elif args.command == 'delete':
        for name in args.names:
            delete_run(f'{root}/{name}', yes=args.yes)
        if not args.yes:
            print('Dry run — pass --yes to actually delete.')

    elif args.command == 'clean':
        clean_outputs(args.targets or None, yes=args.yes)
        if not args.yes:
            print('Dry run — pass --yes to actually delete.')


if __name__ == '__main__':
    main()
