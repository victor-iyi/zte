"""Export a real electrode-coordinate montage CSV for spatial (spherical-harmonic) encoding.

Electrode spatial encoding (`model.spatial_encoding='spherical_harmonics'`) is mathematically exact for whatever coordinates it is given; the only
approximation is in the coordinates themselves. This script writes a `channel,x,y,z,label` CSV -- the format `zte.models.spatial.ScalpGeometry.from_csv`
consumes and that `dataset.montage_csv` points at -- from an MNE standard montage, so the encoding uses the true scalp geometry instead of the
coordinate-free fallback.

ZuCo v1 and v2 recorded with the 129-channel EGI HydroCel Geodesic Sensor Net (E1..E128 plus the Cz vertex reference, named E129 in ZuCo) and use
**standard EGI channel ordering** — confirmed against the dataset documentation. After preprocessing, 105 scalp channels are retained (the outer
face/neck artefact ring is dropped). Because the ordering is standard, `--zuco105` reproduces that retained set directly. The output also carries a
coordinate-derived `region` column, so one CSV serves both the spatial encoding (`x,y,z`) and the eval's region-importance analysis.

Usage:
    python scripts/export_montage.py --out res/montage_gsn105.csv --zuco105   # recommended: the ZuCo 105-channel scalp cap, no manual step
    python scripts/export_montage.py --out res/montage_full.csv               # full net in its own order (unaligned to the 105-subset)

`--zuco105` starts from the 129-net (the Cz reference is excluded automatically), drops the 23 standard EGI outer-ring electrodes by elevation, and
keeps the 105 scalp electrodes in native E-number order — the order ZuCo's band-power tensors use.

Requires the optional dependency `mne` (`pip install mne`).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from zte.models.spatial import ScalpGeometry


def zuco105_labels(montage: str = 'GSN-HydroCel-129') -> list[str]:
    """Returns ZuCo's 105 retained electrodes -- standard EGI net, standard ordering.

    ZuCo v1/v2 use the EGI HydroCel net with standard channel ordering (confirmed from the dataset
    docs), keeping 105 scalp channels after the outer face/neck artefact ring is dropped. That ring
    is exactly the lowest-elevation electrodes, so — starting from the E1..E128 scalp electrodes (the
    Cz vertex reference is excluded automatically, being the montage's only non-`E` label) — we drop
    the 23 lowest-z and keep the remaining 105 in native E-number order, the order ZuCo's band-power
    tensors use. This reproduces ZuCo's retained set; only per-channel scalp attribution assumes the
    standard reduction, which the standard-ordering guarantee underwrites.
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
    parser.add_argument(
        '--montage',
        default='GSN-HydroCel-129',
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
        note = ' (ZuCo-105 scalp cap, standard EGI order)'
    else:
        note = ''
    print(f'Wrote {geo.n_channels}-channel montage to {args.out}{note}')


if __name__ == '__main__':
    main()
