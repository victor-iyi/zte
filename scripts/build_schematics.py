"""Render the ZTE schematics for the paper and dissertation: architecture diagrams, real-montage scalp maps, and the
artifact-driven figures (attention topomap and temporal curve from `attention.json`, the transfer heatmap from
`PARALLAX.json`), each as PNG and SVG with a contact sheet to pick from.
"""

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

_LOG = get_logger('scripts.schematics')


def main() -> None:
    """Parses arguments and writes the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.list:
        for name in SCHEMATICS:
            print(name)
        return

    formats = tuple(f.strip() for f in args.formats.split(',') if f.strip())
    names = [n.strip() for n in args.only.split(',')] if args.only else None
    rendered = build_all(args.out, names, formats)

    if args.attention is not None:
        rendered.append(save_figure(attention_topomap_figure(args.attention), args.out, 'attention_topomap', formats))
        rendered.append(save_figure(attention_temporal_figure(args.attention), args.out, 'attention_temporal', formats))
    if args.parallax is not None:
        rendered.append(save_figure(transfer_heatmap_figure(args.parallax), args.out, 'transfer_heatmap', formats))

    if not args.no_contact_sheet:
        _LOG.info('Contact sheet: %s', contact_sheet(rendered, args.out))
    _LOG.info('Wrote %d figure(s) to %s.', len(rendered), args.out)


if __name__ == '__main__':
    main()
