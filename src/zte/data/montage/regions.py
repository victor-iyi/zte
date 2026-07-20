"""Scalp-region grouping of EEG channels and region-level importance analysis.

Grouping channels into anterior->posterior regions turns "which channel" into "which part of the cortex", separating the
occipito-parietal systems reading uses from the fronto-central language areas inner speech leans on. The `.mat` files
ship no electrode coordinates, so the default map only approximates the montage; `RegionMap.from_csv` supplies an exact
one and every downstream analysis is exact for whatever mapping it is given.
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from zte.data.schema import N_CHANNELS
from zte.logging_utils import get_logger

_LOG = get_logger('data.regions')

type RegionReduce = Literal['mean', 'max', 'l2']

#: Anterior -> posterior scalp regions used by the default mapping.
SCALP_REGIONS: tuple[str, ...] = (
    'frontopolar',
    'frontal',
    'frontocentral',
    'central',
    'centroparietal',
    'parietal',
    'parieto_occipital',
    'occipital',
)

#: Human-readable descriptions (for plots / reports).
REGION_DESCRIPTIONS: dict[str, str] = {
    'frontopolar': 'Fp / anterior prefrontal',
    'frontal': 'Frontal (F) -- executive / inner speech',
    'frontocentral': 'Fronto-central (FC) -- motor planning',
    'central': 'Central (C) -- sensorimotor',
    'centroparietal': 'Centro-parietal (CP)',
    'parietal': 'Parietal (P) -- attention / integration',
    'parieto_occipital': 'Parieto-occipital (PO)',
    'occipital': 'Occipital (O) -- visual word form',
}

#: Relative anterior->posterior extent of each default region, normalised at build time. Frontal and occipital areas
#: carry more electrodes in the EGI net, so their bands are wider.
_DEFAULT_REGION_WEIGHTS: dict[str, float] = {
    'frontopolar': 0.9,
    'frontal': 1.5,
    'frontocentral': 1.3,
    'central': 1.3,
    'centroparietal': 1.2,
    'parietal': 1.3,
    'parieto_occipital': 1.1,
    'occipital': 1.4,
}


@dataclass(slots=True)
class RegionMap:
    """A channel -> scalp-region assignment plus region-level reductions.

    Attributes:
        names (tuple[str, ...]): Region names in canonical (anterior -> posterior) order.
        channel_region (np.ndarray): `(n_channels,)` int array indexing `names` for each channel.
        approximate (bool): `True` for the coordinate-free default, `False` for a montage-derived mapping.
    """

    names: tuple[str, ...]
    channel_region: np.ndarray
    approximate: bool = True

    @property
    def n_regions(self) -> int:
        """Number of distinct regions."""
        return len(self.names)

    @property
    def n_channels(self) -> int:
        """Number of channels covered by the mapping."""
        return int(self.channel_region.shape[0])

    def channels_in(self, region: str) -> np.ndarray:
        """Returns the channel indices assigned to `region`.

        Args:
            region (str): A region name present in `names`.

        Returns:
            np.ndarray: Integer channel indices belonging to the region.

        Raises:
            KeyError: If `region` is not a known region name.
        """
        if region not in self.names:
            raise KeyError(f'Unknown region {region!r}; known: {self.names}')
        r = self.names.index(region)
        return np.nonzero(self.channel_region == r)[0]

    def region_sizes(self) -> dict[str, int]:
        """Returns the channel count per region."""
        return {name: int((self.channel_region == i).sum()) for i, name in enumerate(self.names)}

    def reduce(self, band_power: np.ndarray, method: RegionReduce = 'mean') -> np.ndarray:
        """Collapses per-channel band power into per-region band power.

        Args:
            band_power (np.ndarray): Array `(n_words, n_bp_features, n_channels)`; `NaN` for omitted words is tolerated.
            method (RegionReduce): `mean` (NaN-aware), `max` or `l2` over the channels of each region.

        Returns:
            np.ndarray: Region-reduced band power `(n_words, n_bp_features, n_regions)`.
        """
        import warnings

        bp = np.asarray(band_power, dtype=np.float32)
        n, f, _ = bp.shape
        out = np.full((n, f, self.n_regions), np.nan, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            for r in range(self.n_regions):
                chans = np.nonzero(self.channel_region == r)[0]
                if chans.size == 0:
                    continue
                block = bp[:, :, chans]
                if method == 'mean':
                    out[:, :, r] = np.nanmean(block, axis=2)
                elif method == 'max':
                    out[:, :, r] = np.nanmax(block, axis=2)
                else:  # l2
                    out[:, :, r] = np.sqrt(np.nanmean(block**2, axis=2))
        return out

    def to_dict(self) -> dict[str, object]:
        """Returns a JSON-serialisable representation for bundling."""
        return {
            'names': list(self.names),
            'channel_region': np.asarray(self.channel_region, dtype=int).tolist(),
            'approximate': bool(self.approximate),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RegionMap:
        """Rebuilds a `RegionMap` from `to_dict` output."""
        return cls(
            names=tuple(data['names']),  # type: ignore[arg-type]
            channel_region=np.asarray(data['channel_region'], dtype=int),
            approximate=bool(data.get('approximate', True)),
        )

    @classmethod
    def from_csv(cls, path: str | Path, n_channels: int = N_CHANNELS) -> RegionMap:
        """Loads an exact channel -> region mapping from a CSV montage file.

        The CSV needs a header plus `channel` (0-based index into the band-power tensor) and `region` columns. Regions
        are ordered by first appearance and any channel absent from the file lands in a trailing `unassigned` region.

        Args:
            path (str | Path): Path to the montage CSV.
            n_channels (int): Expected channel count.

        Returns:
            RegionMap: An exact (`approximate=False`) mapping.
        """
        import csv

        raw: dict[int, str] = {}
        with Path(path).open(encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                raw[int(row['channel'])] = str(row['region']).strip()
        ordered: list[str] = []
        for ch in sorted(raw):
            if raw[ch] not in ordered:
                ordered.append(raw[ch])
        assign = np.full(n_channels, -1, dtype=int)
        for ch, region in raw.items():
            if 0 <= ch < n_channels:
                assign[ch] = ordered.index(region)
        if (assign < 0).any():
            ordered.append('unassigned')
            assign[assign < 0] = len(ordered) - 1
        return cls(names=tuple(ordered), channel_region=assign, approximate=False)


def default_region_map(n_channels: int = N_CHANNELS) -> RegionMap:
    """Partitions channels into contiguous bands whose widths follow `_DEFAULT_REGION_WEIGHTS`.

    Assumes the channel axis is ordered roughly rostro-caudally, the common EGI convention; supply an exact montage via
    `RegionMap.from_csv` when precision matters.

    Args:
        n_channels (int): Number of channels to cover.

    Returns:
        RegionMap: The default, approximate mapping.
    """
    weights = np.array([_DEFAULT_REGION_WEIGHTS[name] for name in SCALP_REGIONS], dtype=np.float64)
    edges = np.concatenate([[0.0], np.cumsum(weights) / weights.sum()])
    bounds = np.round(edges * n_channels).astype(int)
    bounds[-1] = n_channels
    assign = np.empty(n_channels, dtype=int)
    for r in range(len(SCALP_REGIONS)):
        assign[bounds[r] : bounds[r + 1]] = r
    return RegionMap(names=SCALP_REGIONS, channel_region=assign, approximate=True)


def region_feature_names(bp_feature_names: list[str], region_names: tuple[str, ...]) -> list[str]:
    """Names for a flattened `(n_bp_features, n_regions)` region-band matrix (band-major order).

    Args:
        bp_feature_names (list[str]): The `n_bp_features` `(measure, band)` names.
        region_names (tuple[str, ...]): The `n_regions` region names in anterior -> posterior order.

    Returns:
        list[str]: `n_bp_features * n_regions` names like `'TRT_t1::frontal'`.
    """
    return [f'{name}::{region}' for name in bp_feature_names for region in region_names]


def region_importance(
    band_power: np.ndarray,
    targets: dict[str, tuple[np.ndarray, Literal['classification', 'regression']]],
    region_map: RegionMap | None = None,
    presence: np.ndarray | None = None,
    method: Literal['mutual_info', 'f_score'] = 'mutual_info',
) -> list[dict[str, object]]:
    """Scores how informative each scalp region is for each target attribute.

    Per-region band power is scored against each target and normalised to sum to 1 across regions, so a row reads as
    "region `r` carries `x%` of the decodable information about this attribute".

    Args:
        band_power (np.ndarray): Array `(n_words, n_bp_features, n_channels)` with `NaN` for omitted words.
        targets (dict[str, tuple[np.ndarray, Literal['classification', 'regression']]]): Mapping `name -> (values (n_words,), task)`.
        region_map (RegionMap | None): Channel grouping (defaults to `default_region_map`).
        presence (np.ndarray | None): Optional boolean `(n_words,)` restricting scoring to present words.
        method (Literal['mutual_info', 'f_score']): Per-feature scorer -- mutual information or ANOVA F-score.

    Returns:
        list[dict[str, object]]: One row per (target, region) with `target`, `region`, `importance` and raw `score`.
    """
    region_map = region_map or default_region_map(band_power.shape[-1])
    if region_map.approximate:
        _LOG.info(
            'Region importance uses an APPROXIMATE coordinate-free region proxy '
            '(no montage); region labels are indicative. Supply RegionMap.from_csv '
            'for exact per-channel regions.'
        )

    # Reduce to `(n_words, n_bp_features * n_regions)`, band-major, NaN-filled so omitted rows cannot break the scorer.
    region_bp = region_map.reduce(band_power, method='mean')
    n, _, r = region_bp.shape
    flat = region_bp.reshape(n, -1)
    col_mean = np.nan_to_num(np.nanmean(np.where(np.isnan(flat), np.nan, flat), axis=0))
    flat = np.where(np.isnan(flat), col_mean[None, :], flat)
    region_of_col = np.tile(np.arange(r), flat.shape[1] // r)

    # Score each target, then average the per-column scores within each region.
    rows: list[dict[str, object]] = []
    for target_name, (values, task) in targets.items():
        x, y = flat, np.asarray(values)
        if presence is not None:
            x, y = flat[presence], y[presence]
        scores = _feature_scores(x, y, task, method)
        per_region = np.array(
            [
                float(scores[region_of_col == ri].mean()) if (region_of_col == ri).any() else 0.0
                for ri in range(r)
            ]
        )
        total = float(per_region.sum()) or 1.0
        for ri, name in enumerate(region_map.names):
            rows.append(
                {
                    'target': target_name,
                    'region': name,
                    'importance': float(per_region[ri] / total),
                    'score': float(per_region[ri]),
                }
            )
    _LOG.info('Computed region importance for %d targets over %d regions.', len(targets), r)
    return rows


def _feature_scores(
    x: np.ndarray,
    y: np.ndarray,
    task: str,
    method: str,
) -> np.ndarray:
    """Per-column importance scores (mutual information or F-score)."""
    if method == 'f_score':
        from sklearn.feature_selection import f_classif, f_regression

        func = f_classif if task == 'classification' else f_regression
        scores, _ = func(x, y)
        return np.nan_to_num(scores)
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    func = mutual_info_classif if task == 'classification' else mutual_info_regression
    return np.nan_to_num(func(x, y, random_state=0))
