"""Canonical ZuCo schema constants, so the rest of the package never hard-codes magic strings."""

from __future__ import annotations

from typing import Final, Literal

# --- Core acquisition constants -------------------------------------------- #

#: EEG sampling rate in Hz (eye-tracker shares the same rate).
SAMPLING_RATE_HZ: Final[float] = 500.0

#: EEG channels retained after the 23 outermost (cheek/neck) electrodes are dropped from the 128-channel cap.
N_CHANNELS: Final[int] = 105

#: Hardware band-pass applied during acquisition (Hz).
ACQUISITION_BANDPASS_HZ: Final[tuple[float, float]] = (0.1, 100.0)

# --- Frequency bands -------------------------------------------------------- #

type Band = Literal['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']

#: The eight ZuCo EEG bands in canonical order.
BANDS: Final[tuple[Band, ...]] = ('t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2')

#: Inclusive frequency ranges (Hz) for each band, per the ZuCo papers.
BAND_RANGES_HZ: Final[dict[Band, tuple[float, float]]] = {
    't1': (4.0, 6.0),
    't2': (6.5, 8.0),
    'a1': (8.5, 10.0),
    'a2': (10.5, 13.0),
    'b1': (13.5, 18.0),
    'b2': (18.5, 30.0),
    'g1': (30.5, 40.0),
    'g2': (40.0, 49.5),
}

#: Human-readable band family for grouping/plots.
BAND_FAMILY: Final[dict[Band, str]] = {
    't1': 'theta',
    't2': 'theta',
    'a1': 'alpha',
    'a2': 'alpha',
    'b1': 'beta',
    'b2': 'beta',
    'g1': 'gamma',
    'g2': 'gamma',
}

# --- Eye-tracking measures -------------------------------------------------- #

type EyeTrackingMeasure = Literal['FFD', 'SFD', 'GD', 'GPT', 'TRT']

#: The five fixation-duration measures (milliseconds), canonical order.
ET_MEASURES: Final[tuple[EyeTrackingMeasure, ...]] = ('FFD', 'SFD', 'GD', 'GPT', 'TRT')

#: Long names for documentation and axis labels.
ET_MEASURE_NAMES: Final[dict[EyeTrackingMeasure, str]] = {
    'FFD': 'First Fixation Duration',
    'SFD': 'Single Fixation Duration',
    'GD': 'Gaze Duration',
    'GPT': 'Go-Past Time',
    'TRT': 'Total Reading Time',
}

#: Additional scalar word fields commonly used as features/targets.
WORD_SCALAR_FIELDS: Final[tuple[str, ...]] = ('nFixations', 'meanPupilSize')

# --- Tasks and subjects ----------------------------------------------------- #

type Task = Literal['SR', 'NR', 'TSR']

#: A subject code such as `'ZAB'`. Left open rather than a `Literal` so ZuCo 1.0, 2.0 and synthetic cohorts share one type.
type Subject = str

#: ZuCo reading tasks. `SR` = sentiment reading (task 1), `NR` = normal reading (task 2), `TSR` = task-specific reading (task 3).
TASKS: Final[tuple[Task, ...]] = ('SR', 'NR', 'TSR')

TASK_NAMES: Final[dict[Task, str]] = {
    'SR': 'Sentiment Reading',
    'NR': 'Normal Reading',
    'TSR': 'Task-Specific Reading',
}

#: The 12 ZuCo 1.0 subject codes (every subject reads the same stimuli).
SUBJECTS_V1: Final[tuple[str, ...]] = (
    'ZAB',
    'ZDM',
    'ZDN',
    'ZGW',
    'ZJM',
    'ZJN',
    'ZJS',
    'ZKB',
    'ZKH',
    'ZKW',
    'ZMG',
    'ZPH',
)

# --- Derived feature naming ------------------------------------------------- #


def band_feature_name(measure: EyeTrackingMeasure, band: Band) -> str:
    """Returns the ZuCo word-level EEG field name for a measure/band pair.

    Args:
        measure (EyeTrackingMeasure): One of the five eye-tracking measures (e.g. `'TRT'`).
        band (Band): One of the eight frequency bands (e.g. `'t1'`).

    Returns:
        str: The `<measure>_<band>` field name, e.g. `'TRT_t1'`.

    Example:
        >>> band_feature_name('TRT', 't1')
        'TRT_t1'
    """
    return f'{measure}_{band}'


def sentence_band_field(band: Band) -> str:
    """Returns the sentence-level mean-EEG field name for a band.

    Args:
        band (Band): One of the eight frequency bands.

    Returns:
        str: The `mean_<band>` field name, e.g. `'mean_t1'`.

    Example:
        >>> sentence_band_field('g2')
        'mean_g2'
    """
    return f'mean_{band}'


def all_word_band_fields(
    measures: tuple[EyeTrackingMeasure, ...] = ET_MEASURES,
    bands: tuple[Band, ...] = BANDS,
) -> list[str]:
    """Enumerates every `<measure>_<band>` word-level EEG field name.

    Args:
        measures (tuple[EyeTrackingMeasure, ...]): Eye-tracking measures to include. Defaults to all five.
        bands (tuple[Band, ...]): Frequency bands to include. Defaults to all eight.

    Returns:
        list[str]: Field names in measure-major, band-minor order.
    """
    return [band_feature_name(m, b) for m in measures for b in bands]
