"""The temporal latency profile: which milliseconds of the word-locked raw window carry the sentence embedding."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from zte.data.dataset import ZuCoDataset
from zte.data.schema import SAMPLING_RATE_HZ
from zte.evaluation.audit.scoreboard import _bootstrap_ci
from zte.inference.embed import ZTEEmbedder
from zte.lens.saliency import DISCLAIMER, Reading, _cosine_drops, _embed_chunks, _reading_batch, _replicate
from zte.logging_utils import get_logger
from zte.utils.provenance import git_info

_LOG = get_logger('lens.temporal')

__all__ = ['CAVEAT', 'DISCLAIMER', 'N400_WINDOW_MS', 'render_markdown', 'temporal_saliency']

# The conventional N400 band, quoted only so a reader can see where the peak fell relative to it -- never as a gate.
N400_WINDOW_MS: Final[tuple[float, float]] = (300.0, 500.0)
"""Millisecond band, relative to word onset, the N400 is conventionally reported in."""

# Every reading costs one clean embed plus one pass per bin and per null draw, so this bounds the whole sweep.
MAX_TEMPORAL_PASSES: Final[int] = 640
"""Ceiling on occlusion forward passes for one temporal profile."""

# Enough draws that the floor's own interval is narrower than the spread between bins, and no more.
DEFAULT_NULL_DRAWS: Final[int] = 8
"""Randomly offset occlusions per reading that make up the null band."""

CAVEAT: Final[str] = (
    'ZuCo word windows are eye-tracking-segmented, so a window starts at a fixation and overlaps the '
    'neighbouring words. A peak in the 300-500 ms band is therefore consistent with an N400 and is not '
    'evidence of one: it is a causal contribution of samples, not a component.'
)
"""The physiological caveat every temporal artifact carries, so no bin can be read as a component."""


def temporal_saliency(
    embedder: ZTEEmbedder,
    dataset: ZuCoDataset,
    readings: Sequence[Reading],
    n_bins: int = 14,
    n_null: int = DEFAULT_NULL_DRAWS,
    max_passes: int = MAX_TEMPORAL_PASSES,
    seed: int = 0,
    ckpt_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Profiles when in the raw window a model's sentence embedding is built, by occluding one time span at a time.

    Each bin is a contiguous span of the word-locked raw window, zeroed in every word of the sentence at once; the
    score is the cosine drop of the sentence embedding against its unoccluded self. That is a causal contribution
    measure, not an attention read-out, so it works on any checkpoint. Bins are aggregated over many readings with a
    percentile bootstrap, and the null band occludes a span of the same width at a random offset *per word*, which
    keeps the removed energy the same while destroying the time-locking -- so a bin only means something above it.

    Note:
        ZuCo word windows come from eye-tracking segmentation and overlap their neighbours, so a peak inside the
        conventional 300-500 ms band is consistent with an N400 and is not proof of one. `peak_in_n400_window`
        records where the peak fell; it makes no physiological claim and gates nothing.

    Args:
        embedder (ZTEEmbedder): The loaded encoder; must be a raw-input model.
        dataset (ZuCoDataset): The built dataset the readings live in, carrying raw EEG windows.
        readings (Sequence[Reading]): Readings to aggregate over -- one reading is a quirk, not a profile.
        n_bins (int, optional): Contiguous time bins the window is split into. Defaults to 14.
        n_null (int, optional): Randomly offset occlusions per reading forming the null band. Defaults to 8.
        max_passes (int, optional): Ceiling on occlusion forward passes. Defaults to `MAX_TEMPORAL_PASSES`.
        seed (int, optional): Seed for the null offsets and the bootstrap. Defaults to 0.
        ckpt_path (str | Path | None, optional): Checkpoint path, hashed into provenance. Defaults to None.

    Returns:
        dict[str, Any] | None: The `temporal.json` payload, or `None` when the model or the dataset cannot answer
            the question (a band-power model, or a dataset built without raw windows).

    Raises:
        ValueError: If no readings are given, or `n_bins` does not fit the raw window.
    """
    if not readings:
        raise ValueError('The temporal profile needs at least one reading; a profile over none is not a profile.')

    if not bool(embedder.model.uses_raw):
        _LOG.warning('Temporal profile skipped: the model reads band-power features, which carry no time axis.')
        return None

    if dataset.raw_eeg is None:
        _LOG.warning('Temporal profile skipped: the dataset holds no raw EEG windows.')
        return None

    window = int(dataset.raw_eeg.shape[-1])
    if not 1 <= n_bins <= window:
        raise ValueError(f'n_bins must be between 1 and the {window}-sample raw window, got {n_bins}.')

    spans = _bin_spans(window, n_bins)
    null_width = max(window // n_bins, 1)

    # One clean embed plus one pass per bin and per null draw, so the ceiling caps readings rather than truncating one.
    per_reading = 1 + n_bins + max(n_null, 0)
    budget = max(max_passes // per_reading, 1)
    used = list(readings[:budget])
    if len(used) < len(readings):
        _LOG.info('Temporal profile: %d of %d readings fit the %d-pass budget.', len(used), len(readings), max_passes)

    drops: list[np.ndarray] = []
    null_means: list[float] = []
    for order, reading in enumerate(used):
        measured = _profile_reading(embedder, dataset, reading, spans, null_width, n_null, seed + order)
        if measured is None:
            return None

        bin_drops, null_drops = measured
        drops.append(bin_drops)
        if null_drops.size:
            null_means.append(float(null_drops.mean()))

    matrix = np.stack(drops)
    null = _null_block(np.asarray(null_means, dtype=np.float64), null_width, seed=seed)

    bins = [_bin_block(index, spans[index], matrix[:, index], seed=seed) for index in range(n_bins)]
    for block in bins:
        block['above_null'] = bool(block['ci_low'] > null['ci_high'])

    peak = int(np.argmax([block['mean_drop'] for block in bins]))
    peak_block = bins[peak]

    from zte.training.init import file_sha256

    return {
        'schema': 'zte.lens.temporal/1',
        'sampling_rate_hz': float(SAMPLING_RATE_HZ),
        'raw_window_samples': window,
        'window_ms': _ms(window),
        'n_readings': len(used),
        'n_words': int(sum(reading.n_words for reading in used)),
        'subjects': sorted({reading.subject for reading in used}),
        'bins': bins,
        'null_band': null,
        'peak': {
            'bin': peak,
            'start_ms': peak_block['start_ms'],
            'end_ms': peak_block['end_ms'],
            'center_ms': peak_block['center_ms'],
            'mean_drop': peak_block['mean_drop'],
            'above_null': peak_block['above_null'],
        },
        'peak_in_n400_window': bool(N400_WINDOW_MS[0] <= peak_block['center_ms'] <= N400_WINDOW_MS[1]),
        'n400_window_ms': list(N400_WINDOW_MS),
        'method': 'temporal_occlusion_cosine_drop',
        'caveat': CAVEAT,
        'disclaimer': DISCLAIMER,
        'provenance': {
            'ckpt': None if ckpt_path is None else str(ckpt_path),
            'ckpt_sha256': None if ckpt_path is None else file_sha256(ckpt_path),
            'run_name': embedder.config.run_name,
            'git_commit': git_info()['commit'],
            'train_holdout': embedder.config.train.loso_holdout_subject,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Renders a temporal profile as the markdown block that ships beside `temporal.json`.

    Args:
        report (dict[str, Any]): A `temporal_saliency` payload.

    Returns:
        str: Markdown: the per-bin table in milliseconds, the null band, the peak, and the caveat.
    """
    null = report.get('null_band') or {}
    peak = report.get('peak') or {}
    lines = [
        '# Temporal latency profile',
        '',
        f'{report.get("n_readings", 0)} readings ({report.get("n_words", 0)} words) over a '
        f'{report.get("raw_window_samples", 0)}-sample window '
        f'({_fmt(report.get("window_ms"))} ms at {_fmt(report.get("sampling_rate_hz"))} Hz), '
        f'occluded one span at a time. Method: `{report.get("method", "unstated")}`.',
        '',
        '| Bin | Onset (ms) | Offset (ms) | Samples | Cosine drop | 95% CI | Above null |',
        '| ---: | ---: | ---: | --- | ---: | --- | --- |',
    ]
    for block in report.get('bins') or []:
        lines.append(
            f'| {block["bin"]} | {_fmt(block.get("start_ms"))} | {_fmt(block.get("end_ms"))} '
            f'| {block.get("start_sample")}-{block.get("end_sample")} '
            f'| {_fmt(block.get("mean_drop"), 4)} '
            f'| [{_fmt(block.get("ci_low"), 4)}, {_fmt(block.get("ci_high"), 4)}] '
            f'| {"yes" if block.get("above_null") else "no"} |'
        )

    lines += [
        '',
        '## The null band -- the floor every bin is read against',
        '',
        f'A span of the same width ({null.get("width_samples", 0)} samples, {_fmt(null.get("width_ms"))} ms) '
        f'occluded at a random offset in each word independently, so the same energy is removed but nothing is '
        f'time-locked: **{_fmt(null.get("mean_drop"), 4)}** '
        f'[{_fmt(null.get("ci_low"), 4)}, {_fmt(null.get("ci_high"), 4)}] '
        f'over {null.get("n_draws", 0)} draws.',
        '',
        '## Peak',
        '',
        f'- Largest drop in bin **{peak.get("bin")}**, {_fmt(peak.get("start_ms"))}-{_fmt(peak.get("end_ms"))} ms '
        f'after word onset ({_fmt(peak.get("mean_drop"), 4)}).',
        f'- Above the null band: **{"yes" if peak.get("above_null") else "no"}**.',
        f'- Falls inside the conventional {_fmt(N400_WINDOW_MS[0])}-{_fmt(N400_WINDOW_MS[1])} ms band: '
        f'**{"yes" if report.get("peak_in_n400_window") else "no"}**.',
        '',
        f'{report.get("caveat", CAVEAT)}',
        '',
        f'_{report.get("disclaimer", DISCLAIMER)}._',
    ]

    return '\n'.join(lines).rstrip() + '\n'


def _bin_spans(window: int, n_bins: int) -> list[tuple[int, int]]:
    """Contiguous `[lo, hi)` sample spans covering the whole window, as evenly as integer rounding allows."""
    cuts = np.round(np.linspace(0.0, float(window), n_bins + 1)).astype(int)

    return [(int(cuts[b]), int(max(cuts[b + 1], cuts[b] + 1))) for b in range(n_bins)]


def _ms(sample: float) -> float:
    """Sample offset from word onset in milliseconds -- the axis a reviewer reads latency on."""
    return round(1000.0 * float(sample) / SAMPLING_RATE_HZ, 3)


def _profile_reading(
    embedder: ZTEEmbedder,
    dataset: ZuCoDataset,
    reading: Reading,
    spans: list[tuple[int, int]],
    null_width: int,
    n_null: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-bin and per-null-draw cosine drops for one reading, or `None` when its batch carries no raw tensor."""
    base = _reading_batch(embedder, dataset, reading)
    if base.get('raw') is None:
        _LOG.warning('Temporal profile skipped: the collated batch carries no raw tensor for this representation.')
        return None

    full = _embed_chunks(embedder, base)[0]

    variants = _replicate(base, len(spans))
    for row, (lo, hi) in enumerate(spans):
        variants['raw'][row][:, :, lo:hi] = 0.0
    bin_drops = _cosine_drops(full, _embed_chunks(embedder, variants))

    if n_null <= 0:
        return bin_drops, np.empty(0, dtype=np.float64)

    window = int(base['raw'].shape[-1])
    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, max(window - null_width, 0) + 1, size=(n_null, reading.n_words))
    nulls = _replicate(base, n_null)
    for row in range(n_null):
        for word in range(reading.n_words):
            lo = int(offsets[row, word])
            nulls['raw'][row][word][:, lo : lo + null_width] = 0.0

    return bin_drops, _cosine_drops(full, _embed_chunks(embedder, nulls))


def _bin_block(index: int, span: tuple[int, int], values: np.ndarray, seed: int) -> dict[str, Any]:
    """One bin's aggregated row: its span in samples and milliseconds, and the bootstrapped drop across readings."""
    lo, hi = span
    mean, ci_low, ci_high = _bootstrap_ci(np.asarray(values, dtype=np.float64), seed=seed)

    return {
        'bin': index,
        'start_sample': lo,
        'end_sample': hi,
        'start_ms': _ms(lo),
        'end_ms': _ms(hi),
        'center_ms': _ms(0.5 * (lo + hi)),
        'mean_drop': float(mean),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'n': int(values.size),
    }


def _null_block(values: np.ndarray, width: int, seed: int) -> dict[str, Any]:
    """The null band: the same occlusion width placed at a random offset per word, bootstrapped across readings."""
    mean, ci_low, ci_high = _bootstrap_ci(values, seed=seed)

    return {
        'width_samples': int(width),
        'width_ms': _ms(width),
        'mean_drop': float(mean),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'n_draws': int(values.size),
        'method': 'random_offset_per_word_occlusion',
    }


def _fmt(value: Any, digits: int = 1) -> str:
    """Formats a table cell, printing a dash rather than `None`."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '--'

    return f'{float(value):.{digits}f}'
