"""Canonical ZuCo dataset schema constants and helpers.

This module encodes the *verified* structural facts about the Zurich Cognitive Language Processing Corpus (ZuCo 1.0 / 2.0) so the rest of the package never has
to hard-code magic strings. The numbers below are taken from the ZuCo data papers and confirmed against the project's own exploration notebook:

- High-density EEG was recorded with a 128-channel EGI Geodesic Hydrocel system at `500 Hz` (band-pass `0.1-100 Hz`).
  The 23 outermost electrodes (cheeks and neck) are dropped during preprocessing, leaving `105` channels for every word-/sentence-level EEG feature.
- EEG power is split into `8` frequency bands (`t1` ... `g2`).
- Five eye-tracking measures are provided per word.
- Word-level EEG features are named `<measure>_<band>` (e.g. `TRT_t1`) and each is a `105`-dimensional vector (one value per channel).

References:
    Hollenstein et al. (2018), *ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading*, Scientific Data.
    Hollenstein et al. (2020), *ZuCo 2.0*, LREC.
"""

from __future__ import annotations

from typing import Final, Literal

# --- Core acquisition constants -------------------------------------------- #

#: EEG sampling rate in Hz (eye-tracker shares the same rate).
SAMPLING_RATE_HZ: Final[float] = 500.0

#: Number of EEG channels retained after artefact electrodes are removed.
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
        measure: One of the five eye-tracking measures (e.g. `'TRT'`).
        band: One of the eight frequency bands (e.g. `'t1'`).

    Returns:
        The `<measure>_<band>` field name, e.g. `'TRT_t1'`.

    Example:
        >>> band_feature_name('TRT', 't1')
        'TRT_t1'
    """
    return f'{measure}_{band}'


def sentence_band_field(band: Band) -> str:
    """Returns the sentence-level mean-EEG field name for a band.

    Args:
        band: One of the eight frequency bands.

    Returns:
        The `mean_<band>` field name, e.g. `'mean_t1'`.

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
        A `list` of field names in measure-major, band-minor order.

    """
    return [band_feature_name(m, b) for m in measures for b in bands]
