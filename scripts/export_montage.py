"""Export a real electrode-coordinate montage CSV for spatial (spherical-harmonic) encoding.

Electrode spatial encoding (`model.spatial_encoding='spherical_harmonics'`) is mathematically exact for whatever coordinates it is given; the only
approximation is in the coordinates themselves. This script writes a `channel,x,y,z,label` CSV -- the format `zte.models.spatial.ScalpGeometry.from_csv`
consumes and that `dataset.montage_csv` points at -- from an MNE standard montage, so the encoding uses the true scalp geometry instead of the
coordinate-free fallback.

ZuCo recorded with a 128-channel EGI Geodesic HydroCel net and retains 105 channels after the 23 outer artefact electrodes are dropped. The output
also carries a coordinate-derived `region` column, so one CSV serves both the spatial encoding (`x,y,z`) and the eval's region-importance analysis.

Usage:
    python scripts/export_montage.py --out res/montage_gsn105.csv --zuco105                          # recommended: auto 105-channel scalp cap, no manual step
    python scripts/export_montage.py --out res/montage_gsn105.csv --keep-file res/zuco_channel_labels.txt  # exact identity if you have the channel order
    python scripts/export_montage.py --out res/montage_full.csv                                       # full 128-channel net, unaligned

`--zuco105` drops the outer face/neck ring geometrically and keeps 105 electrodes in native E-number order (the EGI export order). The channel
*identity* assumes standard EGI ordering; verify against the recording's channel manifest before trusting per-channel scalp attributions.

Requires the optional dependency `mne` (`pip install mne`).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from zte.models.spatial import ScalpGeometry


def zuco105_labels(montage: str = 'GSN-HydroCel-128') -> list[str]:
    """Derives ZuCo's 105 retained electrodes geometrically -- no hand-made label file.

    ZuCo keeps 105 of the 128-channel EGI net after dropping the outer face/neck ring. That ring
    is exactly the lowest-elevation electrodes, so we keep the 105 highest-z electrodes in native
    E-number order (the order EGI/ZuCo export band power in). This yields a real, standard-layout
    montage with no manual step. The channel *identity* still assumes standard EGI ordering; verify
    against the recording's channel manifest before trusting per-channel scalp attributions.
    """
    import mne  # type: ignore[import-untyped]

    pos = mne.channels.make_standard_montage(montage).get_positions()['ch_pos']
    labels = sorted((l for l in pos if l.startswith('E')), key=lambda s: int(s[1:]))
    z = [pos[l][2] for l in labels]
    drop = set(sorted(range(len(labels)), key=lambda i: z[i])[: len(labels) - 105])
    return [l for i, l in enumerate(labels) if i not in drop]


def _regions_from_geometry(xyz: object) -> list[str]:
    """Assigns each electrode an anterior->posterior scalp region from its front-back (y) position.

    In head coordinates the y-axis runs posterior (-) to anterior (+), so ranking electrodes by y
    and splitting into the 8 `SCALP_REGIONS` bands gives a real, coordinate-derived region map — the
    same anterior->posterior partition the analysis expects, but exact for this montage.
    """
    import numpy as np

    from zte.data.regions import SCALP_REGIONS

    y = np.asarray(xyz, dtype=np.float64)[:, 1]
    # Rank -> quantile bin; most-anterior (largest y) = frontopolar, most-posterior = occipital.
    order = np.argsort(-y)  # anterior first
    rank = np.empty(len(y), dtype=int)
    rank[order] = np.arange(len(y))
    bin_idx = (rank * len(SCALP_REGIONS) // max(len(y), 1)).clip(0, len(SCALP_REGIONS) - 1)
    return [SCALP_REGIONS[b] for b in bin_idx]


def main() -> None:
    """Parses arguments and writes the montage CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='Destination CSV path.')
    parser.add_argument('--montage', default='GSN-HydroCel-128', help='MNE standard montage name.')
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
    elif args.zuco105:
        keep = zuco105_labels(args.montage)

    geo = ScalpGeometry.from_mne(args.montage, keep=keep)
    labels = geo.labels or tuple(f'ch{c:03d}' for c in range(geo.n_channels))
    regions = _regions_from_geometry(geo.xyz)  # anterior->posterior band per electrode (from y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        # Coordinates (x,y,z) drive the spatial encoding; `region` drives the eval's
        # region-importance analysis. Emitting both lets one montage serve the whole pipeline.
        writer.writerow(['channel', 'x', 'y', 'z', 'label', 'region'])
        for c, (xyz, label) in enumerate(zip(geo.xyz, labels)):
            writer.writerow(
                [c, f'{xyz[0]:.6f}', f'{xyz[1]:.6f}', f'{xyz[2]:.6f}', label, regions[c]]
            )
    if keep is None:
        note = ' (WARNING: unaligned full montage)'
    elif args.zuco105:
        note = ' (ZuCo-105 scalp cap, standard EGI order — verify identity before attribution)'
    else:
        note = ''
    print(f'Wrote {geo.n_channels}-channel montage to {args.out}{note}')


if __name__ == '__main__':
    main()
