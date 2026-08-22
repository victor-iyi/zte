from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from zte.config._paths import PathFields
from zte.config.types import Granularity, MissingMethod, Normalization, RawAlign, Representation
from zte.data.schema import BANDS, Band, EyeTrackingMeasure, Task


@dataclass
class MissingConfig:
    """How missing word-level values (omitted words, rejected epochs) are filled."""

    method: MissingMethod = 'mask_only'
    """Imputation strategy. `mask_only` defers to the presence mask; `drop` removes incomplete rows; the rest fill."""

    knn_neighbors: int = 5
    """Neighbour count for the `knn` method."""

    iterative_max_iter: int = 10
    """Max rounds for the `iterative` (model-based) method."""

    interpolate_method: Literal['linear', 'nearest', 'spline'] = 'linear'
    """Pandas interpolation kind for `interpolate`."""

    add_missing_indicator: bool = True
    """Emit a boolean presence mask alongside the imputed features so downstream losses can ignore filled entries."""


@dataclass
class DatasetConfig(PathFields):
    """Everything that controls how raw ZuCo `.mat` files become tensors."""

    _PATH_FIELDS: ClassVar[tuple[str, ...]] = (
        'root',
        'cache_dir',
        'cache_remote',
        'montage_csv',
    )

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

    raw_align: RawAlign = 'none'
    """Per-subject alignment of raw EEG windows. `euclidean` whitens each subject by their own mean channel
    covariance. Note `normalize` only ever applied to band power, so this is the raw path's only alignment."""

    raw_align_amplitude: bool = False
    """Also divide each subject's windows by their own RMS voltage. The covariance reference is trace-normalised and
    therefore fixes only the shape of a subject's channel geometry; amplitude is the larger identity carrier."""

    raw_align_fit: Literal['train', 'all'] = 'all'
    """Whose windows the alignment maps are fitted on. `all` includes the held-out subject, which is label-free
    calibration rather than leakage; `train` withholds it as the strict ablation."""

    subject_signature: bool = False
    """Export each subject's covariance descriptor for `model.subject_adapter`. Computed before alignment, since
    whitening would flatten it to a constant."""

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

    cache_remote: str | None = None
    """Persistent cache directory (e.g. a mounted Drive folder) layered behind `cache_dir`. A bundle found
    there is copied down once; a freshly built one is published there immediately, so processing survives the
    machine. Defaults to the `ZTE_CACHE_REMOTE` environment variable. Never part of the cache key."""

    cache_extracts: bool = True
    """Also cache the raw `.mat` extraction, so a config that changes only processing (normalisation,
    imputation, eye-tracking, filters) skips the expensive parse. Costs extra disk; never part of the cache key."""

    cache_format: Literal['npz', 'parquet', 'hdf5'] = 'npz'
    """On-disk cache format. Only `npz` is implemented; `parquet`/`hdf5` are reserved."""
