"""`zte-download` -- resumable Google Drive folder download for large ZuCo archives."""

from __future__ import annotations

import argparse
from pathlib import Path

from zte.cli.support.sources import add_extract_dir
from zte.data.io.remote import download_to_dir, parse_drive_spec
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.download')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-download` command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Resumable Google Drive download (interrupt with Ctrl+C, re-run to continue).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--drive', type=str, required=True, help='Google Drive folder id or shareable URL.')
    parser.add_argument('--out', type=Path, default=Path('res/data/_downloads'), help='Download directory.')
    add_extract_dir(parser)
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def main() -> None:
    """Downloads a Drive folder with resume support."""
    args = parse_arguments()
    configure_logging(args.log_level)

    parsed = parse_drive_spec(args.drive)
    if parsed is None:
        msg = f'Not a valid Google Drive spec: {args.drive!r}'
        raise SystemExit(msg)

    _LOG.info('Downloading to %s (re-run this command anytime to resume)', args.out)
    download_to_dir(args.drive, args.out)
    _LOG.info(
        'All files downloaded. Extract with: zte-prepare --root %s --extract-dir %s ...',
        args.out,
        args.extract_dir,
    )


if __name__ == '__main__':
    main()
