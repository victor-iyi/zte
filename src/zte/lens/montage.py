"""The electrode coordinates a checkpoint was trained under, recovered from disk and proven against its own basis."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from zte.data.cache import fetch_artifact
from zte.data.montage.montage import packaged_montage_csv
from zte.logging_utils import get_logger
from zte.models.spatial import ScalpGeometry, harmonics_match

_LOG = get_logger('lens.montage')

__all__ = [
    'CheckpointMontage',
    'Montage',
    'azimuthal_xy',
    'checkpoint_geometry',
    'load_montage_csv',
    'resolve_checkpoint_montage',
]

type MontageSource = Literal['config', 'store', 'packaged']
"""Where a verified montage was found: the config's own file, the persistent store, or the packaged ZuCo-105 copy."""

# The packaged montage describes exactly ZuCo's retained electrodes, so it is only ever a candidate at that width.
PACKAGED_MONTAGE_CHANNELS: Final[int] = 105
"""Channel count of the packaged ZuCo-105 montage."""

# Where a built encoder keeps its harmonic basis; the same three buffers travel in every raw-frontend checkpoint.
_ENCODING_PREFIX: Final[str] = 'frontend.spatial_mixer.pos.'
"""State-dict prefix of the spherical-harmonic encoding inside a `ZTEModel`."""


@dataclass(slots=True, frozen=True, kw_only=True)
class Montage:
    """Electrode labels, scalp regions and unit-sphere coordinates read from one montage CSV.

    Attributes:
        labels (list[str]): One electrode label per channel, in channel order.
        regions (list[str]): One scalp region per channel, in channel order.
        xyz (np.ndarray): `(n_channels, 3)` coordinates as the CSV stores them.
        path (Path): The file they were read from.
    """

    labels: list[str]
    regions: list[str]
    xyz: np.ndarray
    path: Path


@dataclass(slots=True, frozen=True, kw_only=True)
class CheckpointMontage:
    """A montage whose coordinates rebuild a checkpoint's harmonic basis, and where on this machine it was found.

    Attributes:
        montage (Montage): The verified coordinates, labels and regions.
        source (MontageSource): Which candidate reproduced the basis.
    """

    montage: Montage
    source: MontageSource


def load_montage_csv(path: str | Path, n_channels: int) -> Montage | None:
    """Reads a `channel,x,y,z,label,region` montage CSV, or `None` when it is missing, malformed or misaligned.

    Args:
        path (str | Path): The montage CSV.
        n_channels (int): Channel count the CSV must cover, `0 .. n_channels - 1` exactly.

    Returns:
        Montage | None: The parsed montage, or `None` with the reason logged.
    """
    file = Path(path)
    if not file.is_file():
        _LOG.warning('Montage CSV %s does not exist.', path)
        return None

    rows: dict[int, tuple[float, float, float, str, str]] = {}
    try:
        with file.open(encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                rows[int(row['channel'])] = (
                    float(row['x']),
                    float(row['y']),
                    float(row['z']),
                    str(row.get('label') or f'ch{int(row["channel"]):03d}'),
                    str(row.get('region') or 'unassigned'),
                )
    except (KeyError, ValueError, OSError) as exc:
        _LOG.warning('Could not parse montage CSV %s (%r).', path, exc)
        return None

    if sorted(rows) != list(range(n_channels)):
        _LOG.warning('Montage %s covers %d channels but the data has %d.', path, len(rows), n_channels)
        return None

    return Montage(
        labels=[rows[c][3] for c in range(n_channels)],
        regions=[rows[c][4] for c in range(n_channels)],
        xyz=np.array([rows[c][:3] for c in range(n_channels)], dtype=np.float64),
        path=file,
    )


def azimuthal_xy(xyz: np.ndarray) -> np.ndarray:
    """Top-down azimuthal-equidistant projection of scalp coordinates: vertex at the origin, nose (+y) up.

    Args:
        xyz (np.ndarray): `(n_channels, 3)` coordinates, any radius.

    Returns:
        np.ndarray: `(n_channels, 2)` planar coordinates.
    """
    unit = xyz / np.clip(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-8, None)
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    planar = np.hypot(unit[:, 0], unit[:, 1])
    scale = np.where(planar > 1e-9, theta / np.clip(planar, 1e-9, None), 0.0)

    return np.stack([unit[:, 0] * scale, unit[:, 1] * scale], axis=1)


def resolve_checkpoint_montage(
    harmonics: np.ndarray,
    l_max: int,
    approximate: bool,
    montage_csv: str | None,
    n_channels: int,
) -> tuple[CheckpointMontage | None, str | None]:
    """Finds a montage on this machine whose coordinates rebuild the checkpoint's harmonic basis.

    Three candidates are tried in order -- the CSV the checkpoint's config names, the same file staged from the
    persistent store, and the packaged ZuCo-105 montage -- and the first whose rebuilt basis equals the
    checkpointed one is returned. A file that exists but rebuilds a different basis is refused: a scalp map drawn on
    it would put every electrode somewhere the model never saw.

    Args:
        harmonics (np.ndarray): The checkpoint's `(n_channels, n_harmonics)` basis.
        l_max (int): The degree it was built to.
        approximate (bool): The checkpoint's own placeholder flag.
        montage_csv (str | None): The montage path the checkpoint's config names, if any.
        n_channels (int): The checkpoint's channel count.

    Returns:
        tuple[CheckpointMontage | None, str | None]: The verified montage and `None`, or `None` and the reason no
            candidate could be verified.
    """
    if approximate:
        return (
            None,
            'the checkpoint carries the approximate coordinate-free cap, so no montage describes its channel axis.',
        )

    candidates: list[tuple[MontageSource, Path]] = []
    if montage_csv:
        named = Path(montage_csv)
        if named.is_file():
            candidates.append(('config', named))
        elif fetch_artifact(named):
            candidates.append(('store', named))
    if n_channels == PACKAGED_MONTAGE_CHANNELS:
        candidates.append(('packaged', packaged_montage_csv()))

    for source, path in candidates:
        montage = load_montage_csv(path, n_channels)
        if montage is None:
            continue

        # Rebuilt exactly as training built it, so equality of the bases is equality of the coordinates.
        if harmonics_match(harmonics, ScalpGeometry.from_csv(path, n_channels), l_max):
            _LOG.info(
                'Montage %s (%s) reproduces the checkpoint basis; scalp positions are the trained ones.', path, source
            )
            return CheckpointMontage(montage=montage, source=source), None

        _LOG.warning(
            'Montage %s (%s) does not reproduce the checkpoint basis; refusing to map electrodes onto it.', path, source
        )

    tried = ', '.join(f'{source}: {path}' for source, path in candidates) or 'none available'
    return None, f'no montage on this machine reproduces the checkpoint basis (tried {tried}).'


def checkpoint_geometry(ckpt: str | Path) -> dict[str, Any]:
    """Reads the electrode geometry a checkpoint was trained under, straight from its state dict.

    Nothing is rebuilt: the basis, its degrees and the placeholder flag are persistent buffers, so the answer
    describes the numbers the checkpoint computed rather than whatever montage this machine can find.

    Args:
        ckpt (str | Path): A `best.pt` / `last.pt` checkpoint.

    Returns:
        dict[str, Any]: The run name and holdout, the spatial encoding, `approximate_geometry`, the channel count
            and degree, the montage the config named, and -- when a montage on this machine reproduces the
            basis -- its `montage_source` and `montage_path`; `topomap_readable` is the conjunction a scalp map
            needs, and `reason` says why not.
    """
    from zte.config import ZTEConfig
    from zte.training.checkpoint import CheckpointManager
    from zte.training.init import strip_compile_prefix

    payload = CheckpointManager.load(ckpt, map_location='cpu')
    config = ZTEConfig.from_dict(payload['config'])
    extra = payload.get('extra') or {}
    state = strip_compile_prefix(payload['model'])
    named = extra.get('montage_csv') or config.dataset.montage_csv

    report: dict[str, Any] = {
        'ckpt': str(ckpt),
        'run_name': config.run_name,
        'train_holdout': config.train.loso_holdout_subject,
        'spatial_encoding': config.model.spatial_encoding,
        'has_harmonic_basis': False,
        'approximate_geometry': None,
        'n_channels': None,
        'l_max': None,
        'montage_csv': named,
        'montage_source': None,
        'montage_path': None,
        'montage_verified': False,
        'topomap_readable': False,
        'reason': None,
    }

    harmonics = state.get(_ENCODING_PREFIX + 'harmonics')
    degrees = state.get(_ENCODING_PREFIX + 'degrees')
    approximate = state.get(_ENCODING_PREFIX + 'approximate')
    if harmonics is None or degrees is None or approximate is None:
        report['reason'] = 'the checkpoint carries no spherical-harmonic electrode basis (spatial_encoding is not on).'
        return report

    basis = harmonics.detach().cpu().numpy()
    l_max = int(degrees.max().item())
    approx = bool(approximate.item())
    report.update(has_harmonic_basis=True, approximate_geometry=approx, n_channels=int(basis.shape[0]), l_max=l_max)

    montage, reason = resolve_checkpoint_montage(basis, l_max, approx, named, int(basis.shape[0]))
    if montage is not None:
        report.update(
            montage_source=montage.source,
            montage_path=str(montage.montage.path),
            montage_verified=True,
            topomap_readable=True,
        )
    else:
        report['reason'] = reason

    return report
