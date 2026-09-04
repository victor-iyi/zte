"""Build the `channel,x,y,z,label,region` montage CSV that `dataset.montage_csv` points at.

Spatial encoding is exact for whatever coordinates it is given, so the only approximation is in the coordinates; reading
them from an MNE standard montage replaces the coordinate-free fallback and makes region importance exact too. The
ZuCo-105 montage ships inside the package, so building it needs `mne` only for a montage other than that one.
"""

from __future__ import annotations

import csv
import shutil
from importlib import resources
from pathlib import Path
from typing import Final

from zte.logging_utils import get_logger

_LOG = get_logger('data.montage')

DEFAULT_MONTAGE: str = 'GSN-HydroCel-129'
"""The EGI net ZuCo recorded with; `zuco105_labels`/`build_montage_csv` default to it."""

# The montage is fixed per net, so a copy built once from `mne` is the montage; shipping it means a fresh checkout
# without `mne` still trains on the real coordinates instead of degrading to the placeholder cap.
PACKAGED_MONTAGE_NAME: Final[str] = 'gsn_hydrocel_105.csv'
"""File name of the ZuCo-105 montage shipped with the package, in the native EGI channel order."""


def packaged_montage_csv() -> Path:
    """Returns the path of the ZuCo-105 montage CSV shipped with the package.

    Returns:
        Path: The `channel,x,y,z,label,region` file for the 105 retained GSN-HydroCel electrodes.
    """
    with resources.as_file(resources.files('zte.data.montage') / PACKAGED_MONTAGE_NAME) as path:
        return Path(path)


def zuco105_labels(montage: str = DEFAULT_MONTAGE) -> list[str]:
    """Returns ZuCo's 105 retained electrodes, in the native E-number order its band-power tensors use.

    The face/neck ring ZuCo drops is exactly the lowest-elevation electrodes, so dropping the 23 lowest-z of E1..E128
    reproduces the retained set. The Cz vertex reference is excluded automatically as the only non-`E` label.
    """
    import mne  # type: ignore[import-untyped]

    pos = mne.channels.make_standard_montage(montage).get_positions()['ch_pos']
    labels = sorted((l for l in pos if l.startswith('E')), key=lambda s: int(s[1:]))
    z = [pos[l][2] for l in labels]
    drop = set(sorted(range(len(labels)), key=lambda i: z[i])[: len(labels) - 105])
    return [l for i, l in enumerate(labels) if i not in drop]


def regions_from_geometry(xyz: object) -> list[str]:
    """Assigns each electrode an anterior->posterior scalp region from its front-back (y) position.

    In head coordinates y runs posterior (-) to anterior (+), so ranking by y and splitting into the 8 `SCALP_REGIONS`
    bands gives the partition the analysis expects, exact for this montage.
    """
    import numpy as np

    from zte.data.montage.regions import SCALP_REGIONS

    y = np.asarray(xyz, dtype=np.float64)[:, 1]

    # Rank into a quantile bin: largest y (most anterior) is frontopolar, smallest is occipital.
    order = np.argsort(-y)
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

    Coordinates drive the spatial encoding and `region` drives region-importance analysis, so one CSV serves both.
    The default request -- ZuCo's 105 electrodes on the GSN-HydroCel net -- is served from the packaged copy when
    `mne` is not installed, so only a different net or channel subset needs the optional dependency.

    Args:
        out (str | Path): Destination CSV path (parent dirs created).
        montage (str): MNE standard montage name (ZuCo uses the 129-net).
        zuco105 (bool): Auto-select ZuCo's 105 scalp electrodes; ignored when `keep` is given.
        keep (list[str] | None): Electrode labels to retain, in EEG-channel-axis order. Overrides `zuco105`;
            `None` with `zuco105=False` keeps the full net in its own unaligned order.
        overwrite (bool): Rebuild even when `out` exists. The montage is fixed per net, so reuse needs no `mne`.

    Returns:
        Path: The written (or reused) CSV path.

    Raises:
        ImportError: If `mne` is not installed and the request is not the packaged ZuCo-105 montage.
    """
    out = Path(out)
    if out.is_file() and not overwrite:
        _LOG.info('Reusing cached montage %s.', out)
        return out

    from zte.models.spatial import ScalpGeometry

    try:
        if keep is None and zuco105:
            keep = zuco105_labels(montage)
        geo = ScalpGeometry.from_mne(montage, keep=keep)
    except ImportError:
        if keep is not None or not zuco105 or montage != DEFAULT_MONTAGE:
            raise

        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(packaged_montage_csv(), out)
        _LOG.info('mne is not installed; wrote the packaged ZuCo-105 montage to %s instead of rebuilding it.', out)
        return out

    labels = geo.labels or tuple(f'ch{c:03d}' for c in range(geo.n_channels))
    regions = regions_from_geometry(geo.xyz)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['channel', 'x', 'y', 'z', 'label', 'region'])
        for c, (xyz, label) in enumerate(zip(geo.xyz, labels)):
            writer.writerow([c, f'{xyz[0]:.6f}', f'{xyz[1]:.6f}', f'{xyz[2]:.6f}', label, regions[c]])
    if keep is None:
        note = ' (WARNING: unaligned full montage)'
    elif zuco105:
        note = ' (ZuCo-105 scalp cap, standard EGI order)'
    else:
        note = ''
    _LOG.info('Wrote %d-channel montage to %s%s', geo.n_channels, out, note)
    return out
