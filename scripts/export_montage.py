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

    # Turn-key alternative: `zte-run --spatial exact` builds + wires this montage per-run (see zte.cli.provision).

`--zuco105` starts from the 129-net (the Cz reference is excluded automatically), drops the 23 standard EGI outer-ring electrodes by elevation, and
keeps the 105 scalp electrodes in native E-number order — the order ZuCo's band-power tensors use.

Requires the optional dependency `mne` (`pip install mne`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from zte.data.montage import DEFAULT_MONTAGE, build_montage_csv, zuco105_labels

# Re-exported for backward compatibility; the importable core now lives in `zte.data.montage`, shared
# with the `zte-run --spatial exact` flag so the script and the flag stay byte-for-byte identical.
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

    out = build_montage_csv(
        args.out, montage=args.montage, zuco105=args.zuco105, keep=keep, overwrite=True
    )
    print(f'Wrote montage to {out}')


if __name__ == '__main__':
    main()
