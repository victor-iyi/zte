"""`zte-schematics` -- render the paper's method schematics and the figures drawn from a session's artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from zte.evaluation.schematics import (
    SCHEMATICS,
    attention_temporal_figure,
    attention_topomap_figure,
    build_all,
    contact_sheet,
    save_figure,
    transfer_heatmap_figure,
)
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.schematics')


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Defines and parses the `zte-schematics` command-line arguments.

    Args:
        argv (list[str] | None, optional): Arguments to parse instead of `sys.argv`. Defaults to None.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Render the ZTE schematics: architecture diagrams, real-montage scalp maps, and the '
        'artifact-driven figures (attention topomap and temporal curve from attention.json, the transfer heatmap '
        'from PARALLAX.json), each as PNG and SVG with a contact sheet to pick from.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--out', type=Path, default=Path('res/figures/schematics'), help='Destination directory.')
    parser.add_argument('--only', default=None, help='Comma-separated schematic names (default: every one).')
    parser.add_argument('--list', action='store_true', help='Print the schematic names and exit.')
    parser.add_argument('--formats', default='png,svg', help='Comma-separated extensions to write (pdf is accepted).')
    parser.add_argument(
        '--attention', type=Path, default=None, help='An attention.json to draw the scalp and temporal figures from.'
    )
    parser.add_argument(
        '--parallax', type=Path, default=None, help='A PARALLAX.json to draw the transfer heatmap from.'
    )
    parser.add_argument('--no-contact-sheet', action='store_true', help='Skip the tiled overview.')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[Path]:
    """Renders what the arguments ask for and returns every file written.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        list[Path]: Every figure file written, plus the contact sheet when one was drawn.
    """
    formats = tuple(f.strip() for f in args.formats.split(',') if f.strip())
    names = [n.strip() for n in args.only.split(',')] if args.only else None
    rendered = build_all(args.out, names, formats)

    if args.attention is not None:
        rendered.append(save_figure(attention_topomap_figure(args.attention), args.out, 'attention_topomap', formats))
        rendered.append(save_figure(attention_temporal_figure(args.attention), args.out, 'attention_temporal', formats))
    if args.parallax is not None:
        rendered.append(save_figure(transfer_heatmap_figure(args.parallax), args.out, 'transfer_heatmap', formats))

    written = [path for item in rendered for path in item.paths]
    if not args.no_contact_sheet:
        sheet = contact_sheet(rendered, args.out)
        written.append(sheet)
        _LOG.info('Contact sheet: %s', sheet)
    _LOG.info('Wrote %d figure(s) to %s.', len(rendered), args.out)

    return written


def main() -> None:
    """Entry point for the `zte-schematics` console script."""
    args = parse_arguments()
    configure_logging(args.log_level)

    if args.list:
        for name in SCHEMATICS:
            print(name)
        return

    run(args)


if __name__ == '__main__':
    main()
