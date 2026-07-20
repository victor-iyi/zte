from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from zte.config.types import Granularity, MissingMethod, Normalization, Representation
from zte.data.schema import BANDS, Band, EyeTrackingMeasure, Task


@dataclass
class MissingConfig:
    """How missing word-level values (omitted words, rejected epochs) are filled."""

    method: MissingMethod = 'mask_only'
    """The imputation strategy. `mask_only` leaves NaNs in place and relies solely on the presence mask;
    `drop` removes incomplete rows; the remainder fill values in different ways."""

    knn_neighbors: int = 5
    """Neighbour count for the `knn` method."""

    iterative_max_iter: int = 10
    """Max rounds for the `iterative` (model-based) method."""

    interpolate_method: Literal['linear', 'nearest', 'spline'] = 'linear'
    """Pandas interpolation kind for `interpolate`."""

    add_missing_indicator: bool = True
    """If `True`, emit a boolean presence mask alongside the imputed features so downstream losses can ignore filled entries."""


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
    """Token granularity. Only `'word'` is implemented; `'sentence'` is reserved (use `ZTEEmbedder(level='sentence')`
    for pooled sentence embeddings at inference time)."""

    representation: Representation = 'band_power'
    """Use compact band-power vectors, raw time-series windows, or both."""

    band_power_measures: tuple[EyeTrackingMeasure, ...] = ('TRT',)
    """Eye-tracking measures whose band features are used for the band-power representation."""

    include_eye_tracking: bool = True
    """Whether eye-tracking *behaviour* (the fixation-duration scalars FFD/SFD/GD/GPT/TRT, `nFixations` and `meanPupilSize`)
    is appended to each word's band-power feature vector. `True` (default) enriches the reading-evoked representation;
    `False` yields an EEG-only representation better suited to imagined thought, where no eye-tracking exists. The EEG band-power
    itself is always kept -- this toggle only governs the extra gaze-behaviour dimensions."""

    eye_tracking_measures: tuple[str, ...] = ('FFD', 'GD', 'GPT', 'TRT', 'nFixations', 'meanPupil')
    """Which per-word eye-tracking scalars are appended when `include_eye_tracking` is `True`."""

    bands: tuple[Band, ...] = BANDS
    """Frequency bands used for the band-power representation."""

    raw_field: str = 'rawEEG'
    """Which raw EEG field to read (`rawEEG` per word or `rawData` per sentence)."""

    raw_window: int = 128
    """Fixed time length (samples) raw EEG is padded/truncated to."""

    time_bins: int = 1
    """Number of time bins to split each word's raw window into when computing band power, to expose the post-word N400
    semantic-integration window.  `1` (default) is the legacy single whole-fixation vector; `>1` yields band power per bin so an N400-window
    feature is exposed to the encoder instead of being integrated away. Requires the raw signal (`representation` includes raw)."""

    normalize: Normalization = 'zscore_channel'
    """Feature normalisation. `zscore_channel`/`zscore_global` fit one mean/std across the whole cohort; `zscore_subject` fits and
    applies a **per-subject** mean/std, which removes the constant per-subject offset that otherwise makes subject identity the cheapest thing
    to encode (a direct attack on the "learns who, not what" failure mode). `minmax`/`none` as named."""

    normalizer_fit: Literal['train', 'all'] = 'train'
    """Whether the normaliser (and imputer) statistics are fit on the training split only (`train`, default -- no leakage into val/test/held-out subject)
    or on the whole dataset before splitting (`all`, the legacy behaviour). `train` is required for honest held-out and LOSO numbers."""

    montage_csv: str | None = None
    """Optional path to an electrode-montage CSV (`channel,region` or `channel,x,y,z`) used for scalp-region importance.
    When `None`, an *approximate* rostro-caudal channel partition is used and every region claim is flagged `approximate=True`."""

    bandpass: tuple[float, float] | None = None
    """Optional `(low, high)` Hz Butterworth band-pass for raw EEG."""

    missing: MissingConfig = field(default_factory=MissingConfig)
    """Missing-value handling configuration."""

    include_omitted: bool = True
    """Whether to keep omitted words as masked tokens (preserves sentence sequence integrity). When `False`, omitted-word rows are dropped
    (as with `missing.method='drop'`)."""

    min_words: int = 1
    """Drop sentences shorter than this many words."""

    max_words: int | None = None
    """Drop sentences longer than this many words (`None` = no cap)."""

    cache_dir: str = 'res/cache'
    """Where processed artifacts are cached."""

    cache_format: Literal['npz', 'parquet', 'hdf5'] = 'npz'
    """On-disk cache format. Only `npz` is implemented; `parquet`/`hdf5` are reserved."""
