"""A **truly synthetic**, schema-faithful ZuCo generator for tests, demos and smoke-trains.

The raw ZuCo EEG archives are tens of gigabytes and cannot live in this repository, so this module *fabricates* small `.mat`
files that match ZuCo's exact on-disk structure. Crucially, nothing here is copied or excerpted from the real corpus:

- **The language is invented.** Sentences are assembled from a procedurally generated pseudo-vocabulary (pronounceable
  nonsense tokens such as ``tave`` or ``brelon``) drawn from a Zipf--Mandelbrot frequency law, so there is *no* natural-language
  text -- no movie reviews, no encyclopaedia lines, no ZuCo stimuli -- anywhere in the output.
- **The signals are sampled from statistical models**, never lifted from recordings. Eye-tracking durations, omission
  behaviour, pupil size, band-power topographies and raw EEG segments are all draws from parametric distributions.

At the same time the dataset is deliberately built to *mimic* real ZuCo so that the same analysis and evaluation code produces
qualitatively similar charts on both. Every distribution is calibrated against the published ZuCo/reading-literature reference
statistics collected in :data:`ZUCO_REFERENCE`, and the generator injects the same latent structure the real corpus exhibits:

- Omission (word skipping) falls with word length and rises with word frequency, matching the well-known lexical skipping effect.
- Fixation durations (FFD -> SFD -> GD -> GPT -> TRT) form a realistically ordered, strongly-but-imperfectly correlated family
  that lengthens with word length and rarity.
- Band power scales with word length / inverse frequency, carries a per-band scalp topography (theta frontal, alpha posterior,
  ...) and a per-subject gain/offset, so probes, PCA and scalp-region importance recover interpretable structure.
- Every stimulus token carries a **shared content signature** that is identical across subjects (plus a per-subject transform),
  so cross-subject retrieval and subject-vector arithmetic behave as they would on genuine neural data rather than sitting at chance.

The result is 100% synthetic yet good enough that a model trained/encoded on it yields charts that resemble those from the real
dataset.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
from scipy.io import savemat

from zte.data.schema import (
    BAND_FAMILY,
    BANDS,
    ET_MEASURES,
    N_CHANNELS,
    Band,
    EyeTrackingMeasure,
    band_feature_name,
)
from zte.logging_utils import get_logger, progress

_LOG = get_logger('data.synthetic')


class ZuCoReference:
    """Published/reference statistics the synthetic generator is calibrated against.

    These targets come from the ZuCo data papers (Hollenstein et al. 2018, 2020) and the natural-reading eye-tracking
    literature (e.g. Rayner 1998). They are *approximate* by design -- the goal is to reproduce the shape and scale of the real
    corpus (medians, skew, correlations, lexical effects) so downstream analyses and charts line up, not to clone any single
    recording. Durations are in milliseconds.
    """

    #: Median first-fixation duration for a typical (average-length, mid-frequency) word.
    ffd_median_ms: Final[float] = 195.0
    #: Log-normal spread of the first-fixation duration.
    ffd_log_sigma: Final[float] = 0.32
    #: Mean additional single-fixation time over FFD (log space) and its spread.
    sfd_extra_log: Final[tuple[float, float]] = (0.04, 0.09)
    #: Gamma (shape, base-scale, per-extra-char scale) for the gaze-duration refixation surplus over FFD.
    gd_extra_gamma: Final[tuple[float, float, float]] = (2.0, 20.0, 8.0)
    #: Gamma (shape, base-scale, per-extra-char scale) for the go-past regression surplus over GD.
    gpt_extra_gamma: Final[tuple[float, float, float]] = (1.5, 26.0, 6.0)
    #: Gamma (shape, base-scale, per-extra-char scale) for the total-reading re-reading surplus over GD.
    trt_extra_gamma: Final[tuple[float, float, float]] = (1.4, 22.0, 10.0)
    #: Typical single-fixation duration used to convert reading time into a fixation count.
    ms_per_fixation: Final[float] = 210.0
    #: Mean and spread of the (arbitrary-unit) pupil size.
    pupil_mean_std: Final[tuple[float, float]] = (800.0, 45.0)

    #: Word-length distribution (characters): mean and spread used to standardise lexical effects.
    word_len_mean: Final[float] = 4.8
    word_len_spread: Final[float] = 2.5

    #: Overall word-skipping (omission) behaviour, as a clipped linear model on frequency and length.
    omit_base: Final[float] = 0.42
    omit_freq_gain: Final[float] = 0.6
    omit_len_gain: Final[float] = 0.16
    omit_freq_center: Final[float] = 0.45
    omit_clip: Final[tuple[float, float]] = (0.02, 0.9)

    #: Relative band-power weighting by family (theta strongest at fixation onset, gamma weakest).
    band_family_scale: Final[dict[str, float]] = {
        'theta': 1.0,
        'alpha': 0.8,
        'beta': 0.62,
        'gamma': 0.45,
    }
    #: Anterior->posterior focus (0 = frontopolar, 1 = occipital) and spread of each band family's topography.
    band_family_focus: Final[dict[str, tuple[float, float]]] = {
        'theta': (0.18, 0.22),
        'alpha': (0.85, 0.22),
        'beta': (0.5, 0.28),
        'gamma': (0.15, 0.16),
    }

    #: How strongly band-power magnitude grows with (standardised) word length and word rarity.
    bp_len_gain: Final[float] = 0.35
    bp_freq_gain: Final[float] = 0.6
    #: Weight of the shared, content-specific channel pattern relative to the band topography.
    bp_content_weight: Final[float] = 1.5
    #: Per-subject offset / channel-modulation / additive-noise scales for the band power.
    bp_subject_bias: Final[float] = 0.16
    bp_subject_modulation: Final[float] = 0.1
    bp_noise: Final[float] = 0.14


#: The single, shared reference-statistics instance.
ZUCO_REFERENCE: Final[ZuCoReference] = ZuCoReference()

# --- Procedural pseudo-language -------------------------------------------- #

#: Single consonants / vowels and occasional two-letter clusters for pronounceable nonsense tokens.
_CONSONANTS: Final[tuple[str, ...]] = tuple('bcdfghjklmnprstvwz')
_VOWELS: Final[tuple[str, ...]] = tuple('aeiou')
_CLUSTERS: Final[tuple[str, ...]] = (
    'br',
    'cl',
    'dr',
    'fl',
    'gr',
    'pl',
    'pr',
    'sl',
    'sp',
    'st',
    'tr',
    'ch',
    'sh',
    'th',
)

#: Dimensionality of a stimulus token's latent content code.
_LATENT_DIM: Final[int] = 8
#: Fixed seed for the (subject-independent) global structures shared by every file.
_GLOBAL_SEED: Final[int] = 20240607


def _stable_seed(*parts: object) -> int:
    """Returns a process-independent 32-bit seed derived from its arguments.

    Unlike the built-in :func:`hash`, this is stable across interpreter runs (no `PYTHONHASHSEED` salt), so the same word or
    subject always maps to the same latent code / transform -- the property that lets the shared content signature line up
    across separately generated subject files.

    Args:
        *parts (object): Values whose string forms are hashed together.

    Returns:
        int: A seed in `[0, 2**32)`.
    """
    key = '|'.join(str(p) for p in parts).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), 'big')


def _make_pseudo_word(rng: np.random.Generator, target_len: int) -> str:
    """Builds one pronounceable nonsense token of roughly `target_len` characters.

    Args:
        rng (np.random.Generator): Seeded generator.
        target_len (int): Desired character length (the result is clipped to `[1, 15]`).

    Returns:
        str: A lower-case pseudo-word such as `'tave'` or `'brelon'`.
    """
    target_len = int(np.clip(target_len, 1, 15))
    if target_len <= 1:
        return str(rng.choice(_VOWELS))
    # Alternate consonant/vowel to stay pronounceable, occasionally using a two-letter cluster onset.
    chars: list[str] = []
    want_consonant = True
    while len(''.join(chars)) < target_len:
        if want_consonant:
            chars.append(
                str(rng.choice(_CLUSTERS)) if rng.random() < 0.2 else str(rng.choice(_CONSONANTS))
            )
        else:
            chars.append(str(rng.choice(_VOWELS)))
        want_consonant = not want_consonant
    return ''.join(chars)[:target_len]


@lru_cache(maxsize=8)
def _build_vocabulary(size: int) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Generates a Zipf-ranked pseudo-vocabulary (shared across all subjects/tasks).

    Rank-1 tokens are short and very frequent (function-word-like); rare tokens are longer, reproducing the length/frequency
    coupling of natural language. Frequencies follow a Zipf--Mandelbrot law.

    Args:
        size (int): Number of distinct pseudo-words to mint.

    Returns:
        tuple[tuple[str, ...], tuple[float, ...]]: The vocabulary and its (normalised) sampling weights, aligned by index.
    """
    rng = np.random.default_rng(_stable_seed('vocab', size))
    words: list[str] = []
    seen: set[str] = set()
    for rank in range(1, size + 1):
        target_len = int(round(1.6 + 0.72 * np.log2(rank + 1) + rng.normal(0, 0.7)))
        word = _make_pseudo_word(rng, target_len)
        # Keep tokens distinct so a surface form has a single frequency/identity.
        guard = 0
        while word in seen and guard < 8:
            word = _make_pseudo_word(rng, target_len)
            guard += 1
        seen.add(word)
        words.append(word)
    ranks = np.arange(1, size + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks + 2.7, 1.07)  # Zipf-Mandelbrot
    weights /= weights.sum()
    return tuple(words), tuple(float(w) for w in weights)


def _generate_sentences(
    task: str, n_sentences: int, seed: int, vocab_size: int = 400
) -> tuple[str, ...]:
    """Assembles `n_sentences` distinct synthetic sentences for a task.

    Sentence length follows a task-dependent log-normal (sentiment reading is a touch shorter than normal reading), tokens are
    sampled by Zipf weight (so frequent words recur -- giving the corpus a real term-frequency distribution), the first token is
    capitalised and a terminal mark is appended.

    Args:
        task (str): Task code (`'SR'` reads a little shorter and sometimes takes `!`/`?`; anything else is declarative).
        n_sentences (int): Number of distinct sentences to emit.
        seed (int): Seed controlling sentence assembly (the vocabulary itself is globally shared).
        vocab_size (int): Size of the pseudo-vocabulary to draw from.

    Returns:
        tuple[str, ...]: The generated sentences (deterministic for a given `task`/`seed`).
    """
    words, weights = _build_vocabulary(vocab_size)
    word_arr = np.array(words, dtype=object)
    weight_arr = np.array(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)
    target_len = 8.0 if task == 'SR' else 12.0
    endings = ('.', '.', '.', '!', '?') if task == 'SR' else ('.',)

    sentences: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(sentences) < n_sentences and attempts < n_sentences * 20 + 50:
        attempts += 1
        length = int(np.clip(round(rng.lognormal(np.log(target_len), 0.35)), 3, 40))
        idx = rng.choice(len(word_arr), size=length, p=weight_arr)
        tokens = [str(word_arr[j]) for j in idx]
        tokens[0] = tokens[0].capitalize()
        text = ' '.join(tokens) + str(rng.choice(endings))
        if text in seen:
            continue
        seen.add(text)
        sentences.append(text)
    # Guarantee the requested count even for tiny/degenerate vocabularies.
    while len(sentences) < n_sentences:
        sentences.append(sentences[len(sentences) % max(len(sentences), 1)] if sentences else 'Ae.')
    return tuple(sentences)


# --- Fixed, subject-independent neural structure --------------------------- #


@lru_cache(maxsize=1)
def _band_structures() -> tuple[dict[Band, np.ndarray], dict[Band, np.ndarray]]:
    """Builds the (globally shared) per-band scalp topography and content projection.

    The topography is a smooth anterior->posterior profile placed according to each band family's focus (theta/gamma frontal,
    alpha posterior, beta central), so region-importance analyses recover a sensible map. The projection turns a token's latent
    content code into a channel pattern that is *identical across subjects*, which is what makes cross-subject retrieval work.

    Returns:
        tuple[dict[Band, np.ndarray], dict[Band, np.ndarray]]: `(topography, projection)` where each topography is `(105,)` and
            each projection is `(105, _LATENT_DIM)`.
    """
    rng = np.random.default_rng(_GLOBAL_SEED)
    positions = np.linspace(0.0, 1.0, N_CHANNELS, dtype=np.float32)
    topo: dict[Band, np.ndarray] = {}
    proj: dict[Band, np.ndarray] = {}
    for band in BANDS:
        focus, spread = ZUCO_REFERENCE.band_family_focus[BAND_FAMILY[band]]
        profile = np.exp(-0.5 * ((positions - focus) / spread) ** 2).astype(np.float32)
        topo[band] = (0.5 + profile).astype(np.float32)  # keep it strictly positive
        proj[band] = rng.normal(0.0, 1.0, size=(N_CHANNELS, _LATENT_DIM)).astype(np.float32)
    return topo, proj


def _word_latent(word: str) -> np.ndarray:
    """Returns a stimulus token's shared latent content code (identical across subjects).

    Args:
        word (str): Surface word form.

    Returns:
        np.ndarray: A `(_LATENT_DIM,)` float32 vector, deterministic in `word`.
    """
    rng = np.random.default_rng(_stable_seed('latent', word.lower()))
    return rng.standard_normal(_LATENT_DIM).astype(np.float32)


def _subject_transform(subject: str) -> dict[str, np.ndarray | float]:
    """Builds a subject's fixed gain/offset/channel-modulation and reading-speed bias.

    Args:
        subject (str): Subject code.

    Returns:
        dict[str, np.ndarray | float]: `gain` (scalar), `bias`/`modulation` (`(105,)`), and `reading_bias` (scalar effort offset).
    """
    rng = np.random.default_rng(_stable_seed('subject', subject))
    return {
        'gain': float(rng.uniform(0.8, 1.25)),
        'bias': rng.normal(0.0, ZUCO_REFERENCE.bp_subject_bias, size=N_CHANNELS).astype(np.float32),
        'modulation': (
            1.0 + rng.normal(0.0, ZUCO_REFERENCE.bp_subject_modulation, size=N_CHANNELS)
        ).astype(np.float32),
        'reading_bias': float(rng.normal(0.0, 0.25)),
    }


def _word_frequency(word: str) -> float:
    """Length-based pseudo-frequency proxy in `(0, 1]` (short words score higher).

    This mirrors :func:`zte.data.dataset._word_freq_proxy` so the omission/duration effects the generator injects are exactly
    the effects the analysis tooling later measures.

    Args:
        word (str): The surface word form.

    Returns:
        float: A value where common short tokens approach 1 and long tokens approach 0.
    """
    return float(np.clip(1.0 / (1.0 + 0.35 * len(word.strip('.,;:!?'))), 0.05, 1.0))


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


def _band_power_vector(
    band: Band,
    latent: np.ndarray,
    word_len: int,
    freq: float,
    subject: dict[str, np.ndarray | float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesises a 105-channel band-power vector with embedded, partly-shared structure.

    The vector combines: a magnitude that scales with word length and inverse frequency; a band-specific scalp topography; a
    content pattern derived from the token's *shared* latent code (identical across subjects); and a per-subject gain, channel
    modulation, offset and channel noise. The shared content pattern is what lets the same stimulus retrieve across subjects.

    Args:
        band (Band): The frequency band being generated.
        latent (np.ndarray): The token's shared `(_LATENT_DIM,)` content code.
        word_len (int): Word length in characters.
        freq (float): Word-frequency proxy in `(0, 1]`.
        subject (dict[str, np.ndarray | float]): Output of :func:`_subject_transform`.
        rng (np.random.Generator): Seeded generator for per-token noise.

    Returns:
        A `(N_CHANNELS,)` float32 vector.
    """
    topo, proj = _band_structures()
    ref = ZUCO_REFERENCE
    len_norm = (word_len - ref.word_len_mean) / ref.word_len_spread
    magnitude = ref.band_family_scale[BAND_FAMILY[band]] * (
        1.0 + ref.bp_len_gain * len_norm + ref.bp_freq_gain * (1.0 - freq)
    )
    content = proj[band] @ latent  # (105,), shared across subjects
    content = content / (np.linalg.norm(content) + 1e-6) * np.sqrt(N_CHANNELS)
    pattern = topo[band] + ref.bp_content_weight * content
    vec = (
        subject['gain'] * magnitude * (pattern * subject['modulation'])
        + subject['bias']
        + rng.normal(0.0, ref.bp_noise, size=N_CHANNELS)
    )
    return vec.astype(np.float32)


def _raw_segment(
    word_len: int, subject: dict[str, np.ndarray | float], rng: np.random.Generator
) -> np.ndarray:
    """Synthesises a short `(N_CHANNELS, time_steps)` raw EEG segment for a word.

    The segment is a family-weighted mixture of band oscillations (theta strongest, gamma weakest -- as in
    :data:`ZUCO_REFERENCE`) whose duration scales with word length, sampled at 500 Hz.

    Args:
        word_len (int): Word length, which scales the fixation (segment) duration.
        subject (dict[str, np.ndarray | float]): Per-subject transform (its gain sets the amplitude).
        rng (np.random.Generator): Seeded random generator.

    Returns:
        A `(n_channels, time_steps)` float32 array with band-like oscillations.
    """
    time = int(np.clip(20 + 6 * word_len + rng.integers(-5, 6), 16, 80))
    t = np.arange(time, dtype=np.float32) / 500.0
    band_freqs = {'theta': 5.0, 'alpha': 10.0, 'beta': 20.0, 'gamma': 40.0}
    signal = np.zeros((N_CHANNELS, time), dtype=np.float32)
    for family, f in band_freqs.items():
        amp = ZUCO_REFERENCE.band_family_scale[family]
        phase = rng.uniform(0, 2 * np.pi, size=(N_CHANNELS, 1)).astype(np.float32)
        signal += (amp * np.sin(2 * np.pi * f * t[None, :] + phase)).astype(np.float32)
    signal *= float(subject['gain'])
    signal += rng.normal(0.0, 0.2, size=signal.shape).astype(np.float32)
    return signal


def _eye_tracking(
    word_len: int,
    freq: float,
    subject: dict[str, np.ndarray | float],
    rng: np.random.Generator,
) -> dict[EyeTrackingMeasure, float]:
    """Draws the five fixation-duration measures for a fixated word.

    Durations share a latent reading-effort term (so they are strongly but imperfectly correlated) and are built hierarchically
    -- FFD, then a single-fixation surplus, then non-negative refixation/regression/re-reading surpluses -- yielding the
    realistic ordering ``FFD <= SFD``, ``FFD <= GD <= {GPT, TRT}`` with right-skewed, length-sensitive values.

    Args:
        word_len (int): Word length in characters.
        freq (float): Word-frequency proxy in `(0, 1]`.
        subject (dict[str, np.ndarray | float]): Per-subject transform (its reading bias shifts effort).
        rng (np.random.Generator): Seeded generator.

    Returns:
        dict[EyeTrackingMeasure, float]: Millisecond durations for `FFD, SFD, GD, GPT, TRT`.
    """
    ref = ZUCO_REFERENCE
    len_norm = (word_len - ref.word_len_mean) / ref.word_len_spread
    extra_chars = max(word_len - 3, 0)
    # Shared reading-effort in log space: longer / rarer words and slower subjects read for longer.
    effort = (
        0.14 * len_norm
        + 0.3 * (ref.omit_freq_center - freq)
        + float(subject['reading_bias'])
        + rng.normal(0.0, ref.ffd_log_sigma)
    )
    ffd = ref.ffd_median_ms * float(np.exp(effort))
    sfd = ffd * float(np.exp(rng.normal(*ref.sfd_extra_log)))
    gd_shape, gd_base, gd_char = ref.gd_extra_gamma
    gpt_shape, gpt_base, gpt_char = ref.gpt_extra_gamma
    trt_shape, trt_base, trt_char = ref.trt_extra_gamma
    gd = ffd + float(rng.gamma(gd_shape, gd_base + gd_char * extra_chars))
    gpt = gd + float(rng.gamma(gpt_shape, gpt_base + gpt_char * extra_chars))
    trt = gd + float(rng.gamma(trt_shape, trt_base + trt_char * extra_chars))
    return {'FFD': ffd, 'SFD': sfd, 'GD': gd, 'GPT': gpt, 'TRT': trt}


def _build_word(
    word: str,
    omitted: bool,
    subject: dict[str, np.ndarray | float],
    measures: tuple[EyeTrackingMeasure, ...],
    bands: tuple[Band, ...],
    rng: np.random.Generator,
) -> dict[str, object]:
    """Builds one word's field dict; omitted words get empty arrays.

    Args:
        word (str): Surface word form.
        omitted (bool): Whether the reader skipped this word (no fixation).
        subject (dict[str, np.ndarray | float]): Per-subject transform.
        measures (tuple[EyeTrackingMeasure, ...]): Band-feature measures to populate.
        bands (tuple[Band, ...]): Band-feature bands to populate.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        dict[str, object]: A mapping from field name to value (scalars wrapped as arrays).
    """
    empty = np.array([], dtype=np.float32)
    fields: dict[str, object] = {'content': word}
    word_len = len(word.strip('.,;:!?'))
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

    durations = _eye_tracking(word_len, freq, subject, rng)
    fields['nFixations'] = np.array(
        max(1.0, round(durations['TRT'] / ZUCO_REFERENCE.ms_per_fixation)), dtype=np.float32
    )
    fields['meanPupilSize'] = np.array(rng.normal(*ZUCO_REFERENCE.pupil_mean_std), dtype=np.float32)
    fields['rawEEG'] = _raw_segment(word_len, subject, rng)
    for measure in ET_MEASURES:
        fields[measure] = np.array(durations[measure], dtype=np.float32)
    latent = _word_latent(word)
    for measure in measures:
        # A small per-measure gain keeps the five band-feature families distinct but coherent.
        measure_gain = 0.85 + 0.06 * ET_MEASURES.index(measure)
        for band in bands:
            fields[band_feature_name(measure, band)] = measure_gain * _band_power_vector(
                band, latent, word_len, freq, subject, rng
            )
    return fields


def generate_subject_file(
    path: str | Path,
    subject: str,
    task: str,
    sentences: tuple[str, ...],
    measures: tuple[EyeTrackingMeasure, ...] = ET_MEASURES,
    bands: tuple[Band, ...] = BANDS,
    seed: int = 0,
) -> Path:
    """Writes a single `results<SUBJECT>_<TASK>.mat` synthetic file.

    The `sentences` are expected to be *shared* across subjects for a task (the same stimuli everyone reads); the per-subject
    variation comes entirely from the subject transform and RNG, so the same stimulus token stays retrievable across subjects.

    Args:
        path (str | Path): Output `.mat` path.
        subject (str): Subject code; drives the deterministic per-subject transform.
        task (str): Task code.
        sentences (tuple[str, ...]): Stimulus sentences for this subject/task.
        measures (tuple[EyeTrackingMeasure, ...]): Band-feature measures to populate.
        bands (tuple[Band, ...]): Band-feature bands to populate.
        seed (int): Seed for this file's omission/noise RNG (combine with subject for variety).

    Returns:
        Path: The written path.
    """
    rng = np.random.default_rng(seed)
    subject_transform = _subject_transform(subject)
    ref = ZUCO_REFERENCE
    word_dt = _word_dtype(measures, bands)
    sent_dt = _sentence_dtype()
    sentence_data = np.zeros((len(sentences),), dtype=sent_dt)

    for s_idx, text in enumerate(sentences):
        tokens = text.split()
        word_arr = np.zeros((len(tokens),), dtype=word_dt)
        n_omitted = 0
        for w_idx, token in enumerate(tokens):
            freq = _word_frequency(token)
            len_norm = (len(token.strip('.,;:!?')) - ref.word_len_mean) / ref.word_len_spread
            p_omit = float(
                np.clip(
                    ref.omit_base
                    + ref.omit_freq_gain * (freq - ref.omit_freq_center)
                    - ref.omit_len_gain * len_norm,
                    *ref.omit_clip,
                )
            )
            omitted = bool(rng.random() < p_omit)
            n_omitted += int(omitted)
            word_fields = _build_word(token, omitted, subject_transform, measures, bands, rng)
            for key, value in word_fields.items():
                word_arr[w_idx][key] = value

        sentence_data[s_idx]['content'] = text
        sentence_data[s_idx]['word'] = word_arr
        sentence_data[s_idx]['omissionRate'] = np.array(
            n_omitted / max(len(tokens), 1), dtype=np.float32
        )
        sentence_data[s_idx]['rawData'] = rng.normal(
            0.0, subject_transform['gain'], size=(N_CHANNELS, max(8, 4 * len(tokens)))
        ).astype(np.float32)
        sentence_latent = _word_latent(text)
        for band in bands:
            sentence_data[s_idx][f'mean_{band}'] = _band_power_vector(
                band, sentence_latent, len(text) // 5, 0.5, subject_transform, rng
            )

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
    vocab_size: int = 400,
) -> list[Path]:
    """Generates a small, truly synthetic ZuCo tree under `out_dir`.

    The pseudo-language corpus for each task is generated **once and shared across every subject** (everyone reads the same
    stimuli), while each subject's neural/gaze signals are sampled independently from the calibrated models. No natural-language
    text and no real ZuCo data are used anywhere.

    Args:
        out_dir (str | Path): Destination directory (created if missing). One `.mat` file is written per subject/task as `results<SUBJECT>_<TASK>.mat`.
        subjects (tuple[str, ...]): Subject codes to fabricate.
        tasks (tuple[str, ...]): Tasks to fabricate (each gets its own shared pseudo-language corpus).
        n_sentences (int): Distinct sentences per task.
        measures (tuple[EyeTrackingMeasure, ...]): Band-feature measures to populate.
        bands (tuple[Band, ...]): Band-feature bands to populate.
        seed (int): Base seed; each file and each task corpus is seeded deterministically from it.
        show_progress (bool): Whether to show a progress bar.
        vocab_size (int): Size of the shared pseudo-vocabulary to draw sentences from.

    Returns:
        list[Path]: The list of written `.mat` paths.

    Example:
        >>> paths = generate_synthetic_zuco('/tmp/zuco', n_sentences=4)  # doctest: +SKIP
        >>> len(paths) == 6  # doctest: +SKIP
        True
    """
    out_dir = Path(out_dir)
    task_sentences: dict[str, tuple[str, ...]] = {}
    for t_idx, task in enumerate(tasks):
        task_sentences[task] = _generate_sentences(
            task, n_sentences, seed=seed + 7919 * (t_idx + 1), vocab_size=vocab_size
        )

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
    _LOG.info('Generated %d truly-synthetic ZuCo files under %s', len(paths), out_dir)
    return paths
