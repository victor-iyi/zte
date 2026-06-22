"""Parsing of ZuCo MATLAB `.mat` files into aligned NumPy/record structures.

ZuCo ships dense, nested MATLAB structs. Pre-v7.3 files are read with :func:`scipy.io.loadmat` (`squeeze_me=True, struct_as_record=False`)
which exposes structs as attribute-style objects; v7.3 (HDF5) files fall back to :mod:`h5py`.
The public entry point is :func:`extract_file`, which flattens one file into sentence rows, word rows,
a band-power tensor and (optionally) a raw EEG tensor -- all index-aligned so the dataset layer can stack them directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from zte.data.schema import (
    ET_MEASURES,
    N_CHANNELS,
    Band,
    EyeTrackingMeasure,
    band_feature_name,
)

_FILENAME_RE = re.compile(r'results([A-Za-z0-9]+)_([A-Za-z]+)')


@dataclass(slots=True)
class FileExtract:
    """The flattened contents of a single ZuCo ``.mat`` file.

    Attributes:
        subject: Subject code parsed from the filename (e.g. ``'ZAB'``).
        task: Task code parsed from the filename (e.g. ``'SR'``).
        sentence_rows: One metadata dict per sentence.
        word_rows: One metadata/scalar dict per word (sentence order preserved).
        band_power: Array ``(n_words, n_bp_features, N_CHANNELS)`` with ``NaN``
            for omitted words / rejected epochs, or ``None`` if not requested.
        bp_feature_names: Names (``'<measure>_<band>'``) for the band-power axis.
        raw_eeg: Array ``(n_words, N_CHANNELS, raw_window)`` or ``None``.
    """

    subject: str
    task: str
    sentence_rows: list[dict[str, Any]]
    word_rows: list[dict[str, Any]]
    band_power: np.ndarray | None
    bp_feature_names: list[str]
    raw_eeg: np.ndarray | None


def parse_subject_task(path: Path) -> tuple[str, str]:
    """Extracts subject and task codes from a ZuCo ``.mat`` filename.

    Args:
        path: Path such as ``resultsZAB_SR.mat``.

    Returns:
        ``(subject, task)``, or ``('?', stem)`` when the name does not match.

    Example:
        >>> parse_subject_task(Path('resultsZAB_SR.mat'))
        ('ZAB', 'SR')
    """
    match = _FILENAME_RE.search(path.stem)
    return (match.group(1), match.group(2)) if match else ('?', path.stem)


def scalar_or_nan(value: object) -> float:
    """Returns a feature as a float, mapping empty arrays to ``NaN``.

    Word-level features are empty for words that received no fixation; those are
    treated as missing.

    Args:
        value: A scipy-loaded scalar or (possibly empty) array.

    Returns:
        The scalar value, or ``NaN`` if the underlying array is empty.
    """
    arr = np.asarray(value).ravel()
    return float(arr[0]) if arr.size else float('nan')


def feature_present(value: object) -> bool:
    """Returns whether a feature array is non-empty (not rejected/omitted).

    Args:
        value: A scipy-loaded feature value.

    Returns:
        ``True`` if the underlying array holds any data.
    """
    return np.asarray(value).size > 0


def _channel_vector(value: object, n_channels: int = N_CHANNELS) -> np.ndarray:
    """Coerces a word band feature into a length-``n_channels`` float vector.

    Args:
        value: A scipy-loaded band feature (ideally a 105-vector).
        n_channels: Expected channel count.

    Returns:
        A ``(n_channels,)`` float32 array; all-``NaN`` when the source is empty,
        truncated/zero-padded when the length is unexpected.
    """
    arr = np.asarray(value, dtype=np.float32).ravel()
    if arr.size == 0:
        return np.full(n_channels, np.nan, dtype=np.float32)
    if arr.size == n_channels:
        return arr
    out = np.full(n_channels, np.nan, dtype=np.float32)
    out[: min(arr.size, n_channels)] = arr[:n_channels]
    return out


def _raw_window(value: object, n_channels: int, window: int) -> np.ndarray:
    """Pads/truncates a raw EEG segment to ``(n_channels, window)``.

    Args:
        value: Raw EEG array, ideally ``(channels, time)``.
        n_channels: Target channel count.
        window: Target time length in samples.

    Returns:
        A ``(n_channels, window)`` float32 array; all-zero when the source is
        empty (an omitted word), so callers must consult the presence mask.
    """
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((n_channels, window), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[0] != n_channels and arr.shape[-1] == n_channels:
        arr = arr.T  # tolerate (time, channels) layout
    chan = min(arr.shape[0], n_channels)
    out = np.zeros((n_channels, window), dtype=np.float32)
    length = min(arr.shape[-1], window)
    out[:chan, :length] = arr[:chan, :length]
    return out


def load_mat(path: str | Path) -> dict[str, Any]:
    """Loads a ZuCo ``.mat`` file, transparently handling v7.3 (HDF5) files.

    Args:
        path: Path to a ``.mat`` file (v5/v6 or v7.3).

    Returns:
        A dict of variables. Pre-v7.3 structs are attribute-style objects.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'No such .mat file: {path}')
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        # v7.3 files are HDF5; expose the raw handle for callers that need it.
        import h5py  # pylint: disable=import-outside-toplevel

        return {'__h5__': h5py.File(path, 'r')}


def extract_file(
    path: str | Path,
    measures: tuple[EyeTrackingMeasure, ...] = ('TRT',),
    bands: tuple[Band, ...] = (),
    load_band_power: bool = True,
    load_raw: bool = False,
    raw_field: str = 'rawEEG',
    raw_window: int = 128,
    eeg_probe: tuple[EyeTrackingMeasure, Band] = ('TRT', 't1'),
) -> FileExtract:
    """Flattens one ZuCo ``.mat`` file into aligned rows and tensors.

    The whole file (raw EEG included) is loaded eagerly by scipy, so callers
    should process one file at a time and let it be garbage-collected.

    Args:
        path: Path to a v5/v6 ZuCo ``.mat`` file.
        measures: Eye-tracking measures whose band features feed the band-power
            tensor.
        bands: Frequency bands for the band-power tensor (empty disables it).
        load_band_power: Whether to assemble the band-power tensor.
        load_raw: Whether to assemble the raw EEG tensor.
        raw_field: Word field holding raw EEG (typically ``'rawEEG'``).
        raw_window: Fixed time length raw EEG is padded/truncated to.
        eeg_probe: ``(measure, band)`` used as the per-word EEG presence probe.

    Returns:
        A populated :class:`FileExtract`.
    """
    subject, task = parse_subject_task(Path(path))
    mat = load_mat(path)
    if '__h5__' in mat:  # pragma: no cover - depends on having a v7.3 file
        mat['__h5__'].close()
        raise NotImplementedError(
            'v7.3 ZuCo files are not yet flattened; re-export as v6 or extend '
            'extract_file() with an h5py reader.'
        )

    sentences = np.atleast_1d(mat['sentenceData'])
    bp_names = [band_feature_name(m, b) for m in measures for b in bands]
    probe_field = band_feature_name(*eeg_probe)

    sent_rows: list[dict[str, Any]] = []
    word_rows: list[dict[str, Any]] = []
    bp_chunks: list[np.ndarray] = []
    raw_chunks: list[np.ndarray] = []

    for s_idx, sent in enumerate(sentences):
        text = str(getattr(sent, 'content', '') or '')
        words = np.atleast_1d(getattr(sent, 'word', []))
        sent_rows.append(
            {
                'subject': subject,
                'task': task,
                'sentence_idx': s_idx,
                'text': text,
                'n_words': int(words.size),
                'n_chars': len(text),
                'omission_rate': scalar_or_nan(getattr(sent, 'omissionRate', np.nan)),
                'has_sentence_eeg': feature_present(getattr(sent, 'mean_t1', [])),
            }
        )
        for w_idx, word in enumerate(words):
            present = feature_present(getattr(word, probe_field, []))
            row: dict[str, Any] = {
                'subject': subject,
                'task': task,
                'sentence_idx': s_idx,
                'word_idx': w_idx,
                'word': str(getattr(word, 'content', '') or ''),
                'n_fixations': scalar_or_nan(getattr(word, 'nFixations', np.nan)),
                'mean_pupil': scalar_or_nan(getattr(word, 'meanPupilSize', np.nan)),
                'has_word_eeg': present,
            }
            for measure in ET_MEASURES:
                row[measure] = scalar_or_nan(getattr(word, measure, np.nan))
            word_rows.append(row)

            if load_band_power and bp_names:
                vecs = [
                    _channel_vector(getattr(word, band_feature_name(m, b), []))
                    for m in measures
                    for b in bands
                ]
                bp_chunks.append(np.stack(vecs, axis=0))  # (n_bp_features, channels)
            if load_raw:
                raw_chunks.append(_raw_window(getattr(word, raw_field, []), N_CHANNELS, raw_window))

    band_power = np.stack(bp_chunks, axis=0) if (load_band_power and bp_chunks) else None
    raw_eeg = np.stack(raw_chunks, axis=0) if (load_raw and raw_chunks) else None
    return FileExtract(
        subject=subject,
        task=task,
        sentence_rows=sent_rows,
        word_rows=word_rows,
        band_power=band_power,
        bp_feature_names=bp_names,
        raw_eeg=raw_eeg,
    )
