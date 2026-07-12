"""Export a real electrode-coordinate montage CSV for spatial (spherical-harmonic) encoding.

Electrode spatial encoding (`model.spatial_encoding='spherical_harmonics'`) is mathematically exact for whatever coordinates it is given; the only
approximation is in the coordinates themselves. This script writes a `channel,x,y,z,label` CSV -- the format `zte.models.spatial.ScalpGeometry.from_csv`
consumes and that `dataset.montage_csv` points at -- from an MNE standard montage, so the encoding uses the true scalp geometry instead of the
coordinate-free fallback.

ZuCo recorded with a 128-channel EGI Geodesic HydroCel net and retains 105 channels after the 23 outer artefact electrodes are dropped. **The CSV row
order (channel index) must match the channel axis of your EEG tensors.** Pass the retained electrode labels in that exact order via `--keep` (or a file
of labels, one per line); otherwise the full montage is written in its own order, which will not align with the 105-channel subset.

Usage:
    python scripts/export_montage.py --out res/montage_gsn105.csv --keep-file res/zuco_channel_labels.txt
    python scripts/export_montage.py --out res/montage_full.csv            # full 128-channel net, unaligned

Requires the optional dependency `mne` (`pip install mne`).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from zte.models.spatial import ScalpGeometry


def main() -> None:
    """Parses arguments and writes the montage CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='Destination CSV path.')
    parser.add_argument('--montage', default='GSN-HydroCel-128', help='MNE standard montage name.')
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

    geo = ScalpGeometry.from_mne(args.montage, keep=keep)
    labels = geo.labels or tuple(f'ch{c:03d}' for c in range(geo.n_channels))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['channel', 'x', 'y', 'z', 'label'])
        for c, (xyz, label) in enumerate(zip(geo.xyz, labels)):
            writer.writerow([c, f'{xyz[0]:.6f}', f'{xyz[1]:.6f}', f'{xyz[2]:.6f}', label])
    approx = ' (WARNING: unaligned full montage)' if keep is None else ''
    print(f'Wrote {geo.n_channels}-channel montage to {args.out}{approx}')


if __name__ == '__main__':
    main()
