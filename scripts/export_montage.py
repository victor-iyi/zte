"""Export a `channel,x,y,z,label,region` montage CSV for spatial encoding, from an MNE standard montage.

Pass --zuco105 for the 105 scalp electrodes ZuCo retains, in the order its band-power tensors use.
Requires the optional dependency `mne`; `zte-run --spatial exact` does this per-run instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from zte.data.montage.montage import DEFAULT_MONTAGE, build_montage_csv, zuco105_labels

# Re-exported so this script and `zte-run --spatial exact` share one implementation.
__all__ = ['build_montage_csv', 'zuco105_labels']


def main() -> None:
    """Parses arguments and writes the montage CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='Destination CSV path.')
    parser.add_argument(
        '--montage',
        default=DEFAULT_MONTAGE,
        help='MNE standard montage name (ZuCo uses the 129-net).',
    )
    parser.add_argument(
        '--zuco105',
        action='store_true',
        help="Auto-select ZuCo's 105 scalp electrodes (drop the outer ring) -- no --keep-file needed.",
    )
    parser.add_argument(
        '--keep',
        nargs='*',
        default=None,
        help='Electrode labels to retain, in EEG-channel-axis order.',
    )
    parser.add_argument(
        '--keep-file',
        type=Path,
        default=None,
        help='File of electrode labels (one per line) in channel-axis order.',
    )
    args = parser.parse_args()

    keep = args.keep
    if args.keep_file is not None:
        keep = [ln.strip() for ln in args.keep_file.read_text().splitlines() if ln.strip()]

    out = build_montage_csv(args.out, montage=args.montage, zuco105=args.zuco105, keep=keep, overwrite=True)
    print(f'Wrote montage to {out}')


if __name__ == '__main__':
    main()
