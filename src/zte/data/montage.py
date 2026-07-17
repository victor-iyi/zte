"""Build an exact electrode-coordinate montage CSV -- the scalp geometry for spatial encoding + regions.

Electrode spatial encoding (`model.spatial_encoding='spherical_harmonics'`) is mathematically exact for whatever coordinates it is given;
the only approximation is in the coordinates themselves. This module writes a `channel,x,y,z,label,region` CSV -- the format
`zte.models.spatial.ScalpGeometry.from_csv` consumes and that `dataset.montage_csv` points at -- from an MNE standard montage, so the
encoding uses the true scalp geometry instead of the coordinate-free fallback, and the eval's region-importance analysis becomes exact too.

ZuCo v1/v2 recorded with the 129-channel EGI HydroCel Geodesic Sensor Net in **standard EGI channel ordering**; after preprocessing
105 scalp channels are retained (the outer face/neck artefact ring is dropped). `zuco105=True` reproduces that retained set directly,
in native E-number order -- the order ZuCo's band-power tensors use.

This is the importable core shared by `scripts/export_montage.py` and the `zte-run --spatial exact` flag; it requires
the optional dependency `mne` (`pip install mne`) only when it must read coordinates from a standard montage.
"""

from __future__ import annotations

import csv
from pathlib import Path

from zte.logging_utils import get_logger

_LOG = get_logger('data.montage')

DEFAULT_MONTAGE: str = 'GSN-HydroCel-129'
"""The EGI net ZuCo recorded with; `zuco105_labels`/`build_montage_csv` default to it."""


def zuco105_labels(montage: str = DEFAULT_MONTAGE) -> list[str]:
    """Returns ZuCo's 105 retained electrodes -- standard EGI net, standard ordering.

    ZuCo v1/v2 use the EGI HydroCel net with standard channel ordering (confirmed from the dataset docs), keeping 105 scalp channels
    after the outer face/neck artefact ring is dropped. That ring is exactly the lowest-elevation electrodes, so -- starting from the E1..E128
    scalp electrodes (the Cz vertex reference is excluded automatically, being the montage's only non-`E` label) -- we drop the 23 lowest-z
    and keep the remaining 105 in native E-number order, the order ZuCo's band-power tensors use. This reproduces ZuCo's retained set;
    only per-channel scalp attribution assumes the standard reduction, which the standard-ordering guarantee underwrites.
    """
    import mne  # type: ignore[import-untyped]

    pos = mne.channels.make_standard_montage(montage).get_positions()['ch_pos']
    labels = sorted((l for l in pos if l.startswith('E')), key=lambda s: int(s[1:]))
    z = [pos[l][2] for l in labels]
    drop = set(sorted(range(len(labels)), key=lambda i: z[i])[: len(labels) - 105])
    return [l for i, l in enumerate(labels) if i not in drop]


def regions_from_geometry(xyz: object) -> list[str]:
    """Assigns each electrode an anterior->posterior scalp region from its front-back (y) position.

    In head coordinates the y-axis runs posterior (-) to anterior (+), so ranking electrodes by y
    and splitting into the 8 `SCALP_REGIONS` bands gives a real, coordinate-derived region map -- the
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


def build_montage_csv(
    out: str | Path,
    *,
    montage: str = DEFAULT_MONTAGE,
    zuco105: bool = True,
    keep: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Writes a `channel,x,y,z,label,region` montage CSV and returns its path.

    Coordinates (`x,y,z`) drive the spatial encoding; `region` drives the eval's region-importance
    analysis. Emitting both lets one CSV serve the whole pipeline.

    Args:
        out (str | Path): Destination CSV path (parent dirs created).
        montage (str): MNE standard montage name (ZuCo uses the 129-net).
        zuco105 (bool): Auto-select ZuCo's 105 scalp electrodes (drop the outer ring). Ignored when
            an explicit `keep` list is given.
        keep (list[str] | None): Electrode labels to retain, in EEG-channel-axis order. Overrides
            `zuco105`. `None` with `zuco105=False` keeps the full net in its own (unaligned) order.
        overwrite (bool): Rebuild even when `out` already exists. Default `False` reuses the cached CSV
            (the montage is fixed for a given net, so a rerun needs neither `mne` nor a rebuild).

    Returns:
        Path: The written (or reused) CSV path.

    Raises:
        ImportError: If `mne` is not installed and no cached CSV exists (needed to read coordinates).
    """
    out = Path(out)
    if out.is_file() and not overwrite:
        _LOG.info('Reusing cached montage %s.', out)
        return out

    from zte.models.spatial import ScalpGeometry

    if keep is None and zuco105:
        keep = zuco105_labels(montage)

    geo = ScalpGeometry.from_mne(montage, keep=keep)
    labels = geo.labels or tuple(f'ch{c:03d}' for c in range(geo.n_channels))
    regions = regions_from_geometry(geo.xyz)  # anterior->posterior band per electrode (from y)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['channel', 'x', 'y', 'z', 'label', 'region'])
        for c, (xyz, label) in enumerate(zip(geo.xyz, labels)):
            writer.writerow(
                [c, f'{xyz[0]:.6f}', f'{xyz[1]:.6f}', f'{xyz[2]:.6f}', label, regions[c]]
            )
    if keep is None:
        note = ' (WARNING: unaligned full montage)'
    elif zuco105:
        note = ' (ZuCo-105 scalp cap, standard EGI order)'
    else:
        note = ''
    _LOG.info('Wrote %d-channel montage to %s%s', geo.n_channels, out, note)
    return out
