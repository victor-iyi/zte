"""A schema-faithful synthetic ZuCo generator for tests, demos and smoke-trains.

The raw ZuCo EEG archives are tens of gigabytes, so this module fabricates small `.mat` files that match ZuCo's *exact*
on-disk structure: a `sentenceData` struct array whose elements expose `content`, `word` (a nested struct array),
`omissionRate` and `mean_<band>` fields, and whose words expose the five eye-tracking scalars, `meanPupilSize`, `rawEEG`
and the `<measure>_<band>` 105-channel band features. Omitted words carry *empty* arrays exactly as the real corpus does.

Crucially the generator injects realistic latent structure -- omission depends on word length/frequency, band power correlates
with word length and frequency, and each subject has its own offset/scale -- so that the dataset's analysis tools and the self-supervised
objectives have genuine signal to find.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from zte.data.schema import (
    BANDS,
    ET_MEASURES,
    N_CHANNELS,
    Band,
    EyeTrackingMeasure,
    band_feature_name,
)
from zte.logging_utils import get_logger, progress

_LOG = get_logger('data.synthetic')

# Small built-in corpus: sentiment-style (SR) and Wikipedia-style (NR) lines.
_CORPUS_SR: tuple[str, ...] = (
    'The film is a tedious and unconvincing melodrama.',
    'A genuinely funny and warm-hearted family comedy.',
    'The performances are flat and the script meanders badly.',
    'A dazzling visual feast with surprising emotional depth.',
    'It tries hard but never quite earns its grand finale.',
    'An exhilarating thriller that grips from the first minute.',
    'The plot collapses under the weight of its own cleverness.',
    'A quiet, moving portrait of grief and reconciliation.',
    'Loud, overlong and almost completely charmless.',
    'A sharp, witty satire of modern corporate life.',
)
_CORPUS_NR: tuple[str, ...] = (
    'She earned a doctorate in physics from the university in 1987.',
    'The river flows north before joining the larger eastern delta.',
    'He served as the regional governor for almost two decades.',
    'The orchestra premiered the symphony to widespread acclaim.',
    'Their research reshaped the field of molecular genetics.',
    'The treaty was signed after months of careful negotiation.',
    'The bridge remains one of the longest of its kind.',
    'The company expanded rapidly across three continents.',
    'A small museum now preserves the inventor early notebooks.',
    'The expedition mapped the coastline over several summers.',
)


def _word_dtype(measures: tuple[EyeTrackingMeasure, ...], bands: tuple[Band, ...]) -> np.dtype:
    """Builds the structured dtype for a single ZuCo word struct.

    Args:
        measures (tuple[EyeTrackingMeasure, ...]): Eye-tracking measures to expose as band-feature fields.
        bands (tuple[Band, ...]): Frequency bands to expose per measure.

    Returns:
        A `np.dtype` with object fields matching ZuCo's layout.
    """
    fields: list[tuple[str, str]] = [
        ('content', 'O'),
        ('nFixations', 'O'),
        ('meanPupilSize', 'O'),
        ('rawEEG', 'O'),
    ]
    fields += [(m, 'O') for m in ET_MEASURES]
    fields += [(band_feature_name(m, b), 'O') for m in measures for b in bands]
    return np.dtype(fields)


def _sentence_dtype() -> np.dtype:
    """Returns the structured dtype for a ZuCo sentence struct."""
    fields: list[tuple[str, str]] = [
        ('content', 'O'),
        ('word', 'O'),
        ('omissionRate', 'O'),
        ('rawData', 'O'),
    ]
    fields += [(f'mean_{b}', 'O') for b in BANDS]
    return np.dtype(fields)


def _word_frequency(word: str) -> float:
    """Cheap pseudo-frequency proxy in `(0, 1]` (short words score higher).

    Args:
        word (str): The surface word form.

    Returns:
        A value where common short tokens approach 1 and long tokens approach 0.
    """
    return float(np.clip(1.0 / (1.0 + 0.35 * len(word.strip('.,;:'))), 0.05, 1.0))


def _band_power_vector(
    band: Band,
    word_len: int,
    freq: float,
    subject_gain: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesises a 105-channel band-power vector with embedded structure.

    Power scales with word length and inverse frequency (longer, rarer words evoke stronger responses), is modulated per band,
    and carries a subject gain plus a spatial gradient across channels and channel-wise noise.

    Args:
        band (Band): The frequency band being generated.
        word_len (int): Word length in characters.
        freq (float): Word-frequency proxy in `(0, 1]`.
        subject_gain (float): Per-subject multiplicative offset.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        A `(N_CHANNELS,)` float32 vector.
    """
    band_scale = {'t': 1.0, 'a': 0.8, 'b': 0.6, 'g': 0.4}[band[0]]
    base = band_scale * subject_gain * (0.5 + 0.08 * word_len + 0.6 * (1.0 - freq))
    gradient = np.linspace(0.85, 1.15, N_CHANNELS, dtype=np.float32)
    noise = rng.normal(0.0, 0.15, size=N_CHANNELS).astype(np.float32)
    return (base * gradient + noise).astype(np.float32)


def _raw_segment(word_len: int, subject_gain: float, rng: np.random.Generator) -> np.ndarray:
    """Synthesises a short `(N_CHANNELS, time_steps)` raw EEG segment for a word.

    Args:
        word_len (int): Word length, which scales the fixation (segment) duration.
        subject_gain (float): Per-subject amplitude offset.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        A `(n_channels, time_steps)` float32 array with band-like oscillations.

    """
    time = int(np.clip(20 + 6 * word_len + rng.integers(-5, 6), 16, 80))
    t = np.arange(time, dtype=np.float32) / 500.0
    freqs = np.array([5.0, 10.0, 20.0, 40.0], dtype=np.float32)
    signal = np.zeros((N_CHANNELS, time), dtype=np.float32)
    for f in freqs:
        phase = rng.uniform(0, 2 * np.pi, size=(N_CHANNELS, 1)).astype(np.float32)
        signal += np.sin(2 * np.pi * f * t[None, :] + phase).astype(np.float32)
    signal *= subject_gain
    signal += rng.normal(0.0, 0.2, size=signal.shape).astype(np.float32)
    return signal


def _build_word(
    word: str,
    omitted: bool,
    subject_gain: float,
    measures: tuple[EyeTrackingMeasure, ...],
    bands: tuple[Band, ...],
    rng: np.random.Generator,
) -> dict[str, object]:
    """Builds one word's field dict; omitted words get empty arrays.

    Args:
        word (str): Surface word form.
        omitted (bool): Whether the reader skipped this word (no fixation).
        subject_gain (float): Per-subject offset.
        measures (tuple[EyeTrackingMeasure, ...]): Band-feature measures to populate.
        bands (tuple[Band, ...]): Band-feature bands to populate.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        dict[str, object]: A mapping from field name to value (scalars wrapped as arrays).

    """
    empty = np.array([], dtype=np.float32)
    fields: dict[str, object] = {'content': word}
    word_len = len(word.strip('.,;:'))
    freq = _word_frequency(word)

    if omitted:
        fields['nFixations'] = np.array(0.0, dtype=np.float32)
        fields['meanPupilSize'] = empty
        fields['rawEEG'] = empty
        for measure in ET_MEASURES:
            fields[measure] = empty
        for measure in measures:
            for band in bands:
                fields[band_feature_name(measure, band)] = empty
        return fields

    # Compute the eye-tracking scalars.
    ffd = float(np.clip(rng.normal(180 + 12 * word_len, 35), 80, 600))
    gd = ffd + float(np.clip(rng.normal(40 + 6 * word_len, 30), 0, 400))
    trt = gd + float(np.clip(rng.normal(50, 40), 0, 500))
    # Compute the number of fixations, mean pupil size and raw EEG segment.
    fields['nFixations'] = np.array(max(1.0, round(trt / 220.0)), dtype=np.float32)
    fields['meanPupilSize'] = np.array(rng.normal(700, 40), dtype=np.float32)
    fields['rawEEG'] = _raw_segment(word_len, subject_gain, rng)
    # Compute the band-power vectors.
    scalars = {'FFD': ffd, 'SFD': ffd, 'GD': gd, 'GPT': gd, 'TRT': trt}
    for measure in ET_MEASURES:
        fields[measure] = np.array(scalars[measure], dtype=np.float32)
    for measure in measures:
        for band in bands:
            fields[band_feature_name(measure, band)] = _band_power_vector(
                band, word_len, freq, subject_gain, rng
            )
    return fields


def generate_subject_file(
    path: str | Path,
    subject: str,  # pylint: disable=unused-argument
    task: str,  # pylint: disable=unused-argument
    sentences: tuple[str, ...],
    measures: tuple[EyeTrackingMeasure, ...] = ET_MEASURES,
    bands: tuple[Band, ...] = BANDS,
    seed: int = 0,
) -> Path:
    """Writes a single `results<SUBJECT>_<TASK>.mat` synthetic file.

    Args:
        path (str | Path): Output `.mat` path.
        subject (str): Subject code embedded in the data and filename logic.
        task (str): Task code.
        sentences (tuple[str, ...]): Stimulus sentences for this subject/task.
        measures (tuple[EyeTrackingMeasure, ...]): Band-feature measures to populate.
        bands (tuple[Band, ...]): Band-feature bands to populate.
        seed (int): Seed for this file's RNG (combine with subject for variety).

    Returns:
        Path: The written path.
    """
    rng = np.random.default_rng(seed)
    subject_gain = float(rng.uniform(0.8, 1.25))
    word_dt = _word_dtype(measures, bands)
    sent_dt = _sentence_dtype()
    sentence_data = np.zeros((len(sentences),), dtype=sent_dt)

    # Build the sentence data.
    for s_idx, text in enumerate(sentences):
        # Tokenize the sentence.
        tokens = text.split()
        word_arr = np.zeros((len(tokens),), dtype=word_dt)
        n_omitted = 0
        # Build the word data.
        for w_idx, token in enumerate(tokens):
            freq = _word_frequency(token)
            p_omit = float(np.clip(0.55 * freq + 0.1, 0.05, 0.85))
            omitted = bool(rng.random() < p_omit)
            n_omitted += int(omitted)
            word_fields = _build_word(token, omitted, subject_gain, measures, bands, rng)
            # Assign the word fields to the word array.
            for key, value in word_fields.items():
                word_arr[w_idx][key] = value

        # Assign the sentence fields to the sentence array.
        sentence_data[s_idx]['content'] = text
        sentence_data[s_idx]['word'] = word_arr
        sentence_data[s_idx]['omissionRate'] = np.array(
            n_omitted / max(len(tokens), 1), dtype=np.float32
        )
        sentence_data[s_idx]['rawData'] = rng.normal(
            0.0, subject_gain, size=(N_CHANNELS, max(8, 4 * len(tokens)))
        ).astype(np.float32)

        # Assign the mean band power to the sentence array.
        for band in bands:
            sentence_data[s_idx][f'mean_{band}'] = _band_power_vector(
                band, len(text), 0.5, subject_gain, rng
            )

    # Save the sentence data to disk.
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(out), {'sentenceData': sentence_data}, do_compression=True)
    return out


def generate_synthetic_zuco(
    out_dir: str | Path,
    subjects: tuple[str, ...] = ('ZAB', 'ZDM', 'ZJN'),
    tasks: tuple[str, ...] = ('SR', 'NR'),
    n_sentences: int = 10,
    measures: tuple[EyeTrackingMeasure, ...] = ET_MEASURES,
    bands: tuple[Band, ...] = BANDS,
    seed: int = 42,
    show_progress: bool = True,
) -> list[Path]:
    """Generates a small synthetic ZuCo tree under `out_dir`.

    Args:
        out_dir (str | Path): Destination directory (created if missing). One `.mat` file is written per subject/task as `results<SUBJECT>_<TASK>.mat`.
        subjects (tuple[str, ...]): Subject codes to fabricate.
        tasks (tuple[str, ...]): Tasks to fabricate (`'SR'` draws sentiment lines, others draw Wikipedia-style lines).
        n_sentences (int): Sentences per subject/task (sampled with replacement from the built-in corpus when larger than it).
        measures (tuple[EyeTrackingMeasure, ...]): Band-feature measures to populate.
        bands (tuple[Band, ...]): Band-feature bands to populate.
        seed (int): Base seed; each file is seeded deterministically from it.
        show_progress (bool): Whether to show a progress bar.

    Returns:
        list[Path]: The list of written `.mat` paths.

    Example:
        >>> paths = generate_synthetic_zuco('/tmp/zuco', n_sentences=4)  # doctest: +SKIP
        >>> len(paths) == 6  # doctest: +SKIP
        True
    """
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    # Sample the sentences for each task.
    task_sentences: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        corpus = _CORPUS_SR if task == 'SR' else _CORPUS_NR
        idx = rng.integers(0, len(corpus), size=n_sentences)
        task_sentences[task] = tuple(corpus[j] for j in idx)

    # Generate the synthetic ZuCo files.
    jobs = [(subj, task) for subj in subjects for task in tasks]
    paths: list[Path] = []

    for i, (subject, task) in enumerate(
        progress(jobs, description='Synthesising ZuCo', disable=not show_progress)
    ):
        sentences = task_sentences[task]
        file_path = out_dir / f'results{subject}_{task}.mat'
        generate_subject_file(
            file_path,
            subject=subject,
            task=task,
            sentences=sentences,
            measures=measures,
            bands=bands,
            seed=seed + i * 101,
        )
        paths.append(file_path)
    _LOG.info('Generated %d synthetic ZuCo files under %s', len(paths), out_dir)
    return paths
