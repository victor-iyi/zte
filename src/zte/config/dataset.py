from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from zte.config.types import Granularity, MissingMethod, Normalization, Representation
from zte.data.schema import BANDS, Band, EyeTrackingMeasure, Task


@dataclass
class MissingConfig:
    """How missing word-level values (omitted words, rejected epochs) are filled."""

    method: MissingMethod = 'mask_only'
    """Imputation strategy. `mask_only` leaves NaNs to the presence mask; `drop` removes incomplete rows; the rest fill."""

    knn_neighbors: int = 5
    """Neighbour count for the `knn` method."""

    iterative_max_iter: int = 10
    """Max rounds for the `iterative` (model-based) method."""

    interpolate_method: Literal['linear', 'nearest', 'spline'] = 'linear'
    """Pandas interpolation kind for `interpolate`."""

    add_missing_indicator: bool = True
    """Emit a boolean presence mask alongside the imputed features so downstream losses can ignore filled entries."""


@dataclass
class DatasetConfig:
    """Everything that controls how raw ZuCo `.mat` files become tensors."""

    root: str = 'res/data/zuco_extracted'
    """Directory holding extracted `.mat` files (searched recursively)."""

    tasks: tuple[Task, ...] = ('SR', 'NR')
    """Which reading tasks to include (`SR`, `NR`, `TSR`)."""

    subjects: tuple[str, ...] | None = None
    """Subject codes to include, or `None` for all discovered."""

    granularity: Granularity = 'word'
    """Token granularity. Only `'word'` is implemented; use `ZTEEmbedder(level='sentence')` for pooled sentences."""

    representation: Representation = 'band_power'
    """Use compact band-power vectors, raw time-series windows, or both."""

    band_power_measures: tuple[EyeTrackingMeasure, ...] = ('TRT',)
    """Eye-tracking measures whose band features are used for the band-power representation."""

    include_eye_tracking: bool = True
    """Append the gaze-behaviour scalars to each word's band-power vector. `False` gives an EEG-only representation,
    suited to imagined thought where no eye-tracking exists; the EEG band-power itself is kept either way."""

    # SFD is excluded: ~60% missing and equal to FFD wherever present.
    eye_tracking_measures: tuple[str, ...] = (
        'FFD',
        'GD',
        'GPT',
        'TRT',
        'n_fixations',
        'mean_pupil',
    )
    """Per-word eye-tracking scalars appended when `include_eye_tracking` is `True`. Names must match the `words`
    table columns written by `mat_loader` -- an unmatched name is silently skipped."""

    bands: tuple[Band, ...] = BANDS
    """Frequency bands used for the band-power representation."""

    raw_field: str = 'rawEEG'
    """Which raw EEG field to read (`rawEEG` per word or `rawData` per sentence)."""

    raw_window: int = 128
    """Fixed time length (samples) raw EEG is padded/truncated to."""

    time_bins: int = 1
    """Time bins each word's raw window is split into for band power. `1` is one whole-fixation vector; `>1` exposes
    the post-word N400 semantic-integration window instead of integrating it away. Requires a raw `representation`."""

    normalize: Normalization = 'zscore_channel'
    """Feature normalisation. `zscore_channel`/`zscore_global` fit one mean/std across the cohort; `zscore_subject` fits
    per subject, removing the constant offset that makes subject identity the cheapest thing to encode."""

    normalizer_fit: Literal['train', 'all'] = 'train'
    """Fit normaliser and imputer statistics on the training split only (`train`) or on everything before splitting
    (`all`). `train` is required for honest held-out and LOSO numbers."""

    montage_csv: str | None = None
    """Electrode-montage CSV (`channel,region` or `channel,x,y,z`) for scalp-region importance. When `None`, an
    approximate rostro-caudal partition is used and every region claim is flagged `approximate=True`."""

    bandpass: tuple[float, float] | None = None
    """Optional `(low, high)` Hz Butterworth band-pass for raw EEG."""

    missing: MissingConfig = field(default_factory=MissingConfig)
    """Missing-value handling configuration."""

    include_omitted: bool = True
    """Keep omitted words as masked tokens, preserving sentence sequence integrity; `False` drops those rows."""

    min_words: int = 1
    """Drop sentences shorter than this many words."""

    max_words: int | None = None
    """Drop sentences longer than this many words (`None` = no cap)."""

    cache_dir: str = 'res/cache'
    """Where processed artifacts are cached."""

    cache_format: Literal['npz', 'parquet', 'hdf5'] = 'npz'
    """On-disk cache format. Only `npz` is implemented; `parquet`/`hdf5` are reserved."""
