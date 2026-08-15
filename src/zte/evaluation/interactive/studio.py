"""The decode studio: watch a reading go through the encoder, the bridge and the frozen LM, one token at a time."""

from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zte.data.schema import BANDS
from zte.evaluation.generation import content_word_f1, sentence_wer, tokenise
from zte.evaluation.interactive._assets import load_page
from zte.inference.decode import ReadingBatch, ZTEDecoder, paired_shuffle
from zte.logging_utils import get_logger
from zte.models.spatial import ScalpGeometry, resolve_geometry

_LOG = get_logger('evaluation.interactive.studio')

# Band power spans several orders of magnitude across electrodes, so the page receives a per-reading log-scaled
# 0-255 quantisation rather than raw microvolts-squared: the colour map is comparative, and the legend says so.
_QUANT_MAX: int = 255

# Controls the studio decodes for itself. Enough to make one reading readable; the verdict's full pre-registered set
# is `zte-decode`'s job and the page says so rather than implying this is the audit.
_STUDIO_CONTROLS: tuple[str, ...] = ('null_prefix', 'length_only', 'mismatch')


def build_studio(
    decoder: ZTEDecoder,
    dataset: Any,
    readings: ReadingBatch,
    *,
    rows: np.ndarray | None = None,
    controls: tuple[str, ...] = _STUDIO_CONTROLS,
    max_new_tokens: int | None = None,
    run_name: str = 'ZTE run',
    montage_csv: str | None = None,
) -> dict[str, Any]:
    """Decodes a handful of readings and assembles everything the studio page draws.

    Note:
        This is an inspection tool, not an audit. It decodes a few readings so a human can see what the model did;
        the pre-registered controls, the permutation null and the verdict live in `zte-decode` and the evaluation
        report. The page repeats that in its own banner so a screenshot cannot be mistaken for a result.

    Args:
        decoder (ZTEDecoder): A restored decoder.
        dataset (ZuCoDataset): The dataset the readings were embedded from, for the per-word EEG.
        readings (ReadingBatch): The conditioning bundle for a split.
        rows (np.ndarray | None, optional): Which readings to include. Defaults to None, which takes the first few.
        controls (tuple[str, ...], optional): Brain-independent conditions to decode beside the hypothesis.
        max_new_tokens (int | None, optional): Decode cap. Defaults to the configured value.
        run_name (str, optional): Label for the header. Defaults to 'ZTE run'.
        montage_csv (str | None, optional): Electrode-coordinate CSV for the scalp map. Defaults to None.

    Returns:
        dict[str, Any]: The JSON island the page consumes.
    """
    chosen = np.arange(min(8, len(readings))) if rows is None else np.asarray(rows, dtype=int)
    if not len(chosen):
        return _clean({'run_name': run_name, 'applicable': False, 'reason': 'no readings selected'})

    subset = readings.take(chosen)
    subset.meta = readings.meta.iloc[chosen].reset_index(drop=True)
    references = [str(t) for t in subset.meta.get('text', pd.Series([''] * len(chosen)))]

    traces = decoder.decode_trace(subset, max_new_tokens=max_new_tokens)
    hypotheses = [str(record['hypothesis']) for record in traces]
    control_texts = _decode_controls(decoder, subset, controls, max_new_tokens)
    power, geometry = _scalp_power(dataset, subset.meta, montage_csv)

    entries: list[dict[str, Any]] = []
    for i, record in enumerate(traces):
        meta = subset.meta.iloc[i]
        entries.append(
            {
                'index': int(chosen[i]),
                'subject': str(meta.get('subject', '?')),
                'task': str(meta.get('task', '?')),
                'n_words': int(meta.get('n_words', 0) or 0),
                'target': references[i],
                'target_words': tokenise(references[i]) or references[i].split(),
                'hypothesis': hypotheses[i],
                'scores': _scores(hypotheses[i], references[i]),
                'controls': [
                    {'name': name, 'text': texts[i], **_scores(texts[i], references[i])}
                    for name, texts in control_texts.items()
                ],
                'steps': record['steps'],
                # Trimmed to this reading's own words: the pointer arrives padded to the batch's longest reading,
                # and a page that drew those columns would put an always-empty word on the end of every short one.
                'pointer': _trim(record['pointer'], int(meta.get('n_words', 0) or 0)),
                'codes': record['codes'],
                'power': power[i],
            }
        )

    return _clean(
        {
            'run_name': run_name,
            'applicable': True,
            'bands': list(BANDS),
            'montage': _montage_payload(geometry),
            'uses_evidence': bool(decoder.uses_evidence),
            'readings': entries,
        }
    )


def studio_html(payload: dict[str, Any], out_path: str | Path, run_name: str = 'ZTE run') -> Path:
    """Writes the studio as one self-contained offline HTML file.

    Args:
        payload (dict[str, Any]): A dict from `build_studio`.
        out_path (str | Path): Destination `.html` path (parents created; a non-html suffix is rewritten).
        run_name (str, optional): Header label. Defaults to 'ZTE run'.

    Returns:
        Path: The written path.
    """
    out = Path(out_path)
    if out.suffix != '.html':
        out = out.with_suffix('.html')
    out.parent.mkdir(parents=True, exist_ok=True)

    data = json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c')
    out.write_text(_TEMPLATE.replace('__TITLE__', _esc(run_name)).replace('__DATA__', data), encoding='utf-8')
    _LOG.info('Decode studio written to %s (%.1f MB).', out, out.stat().st_size / 1e6)

    return out


def _decode_controls(
    decoder: ZTEDecoder,
    readings: ReadingBatch,
    controls: tuple[str, ...],
    max_new_tokens: int | None,
) -> dict[str, list[str]]:
    """Decodes each requested brain-independent condition through the decoder's own control machinery."""
    out: dict[str, list[str]] = {}
    n = len(readings)
    for name in controls:
        if name == 'null_prefix':
            prefix = decoder.null_prefix(n)
            out[name] = decoder.generate_from_prefix(prefix, max_new_tokens=max_new_tokens)
        elif name == 'length_only':
            out[name] = decoder.generate(readings, max_new_tokens=max_new_tokens, evidence_content=False)
        elif name == 'mismatch':
            partners = paired_shuffle(n, seed=0)
            out[name] = decoder.generate(readings.take(partners), max_new_tokens=max_new_tokens)
        else:
            _LOG.warning('Studio control %r is not one this page can decode; skipping it.', name)
    return out


def _trim(pointer: list[list[float]] | None, n_words: int) -> list[list[float]] | None:
    """Cuts the pointer's padded word axis back to the reading's own length."""
    if pointer is None or n_words <= 0:
        return pointer

    return [row[:n_words] for row in pointer]


def _scores(hypothesis: str, reference: str) -> dict[str, float]:
    """Per-sentence scores for one decode. Absolute values mean nothing alone -- the controls beside them do."""
    return {
        'content_f1': float(content_word_f1([hypothesis], [reference])[0]),
        'wer': float(sentence_wer(hypothesis, reference)),
        'n_tokens': float(len(tokenise(hypothesis))),
    }


def _scalp_power(
    dataset: Any, meta: pd.DataFrame, montage_csv: str | None
) -> tuple[list[list[list[list[int]]]], ScalpGeometry | None]:
    """Returns per-reading `(words, bands, channels)` quantised band power, and the electrode geometry to draw it on.

    Note:
        Quantisation is per reading over the log power, so the colour scale compares electrodes and words *within*
        one reading and never across readings. Two readings whose maps look alike are not therefore alike in
        microvolts, which is why the page labels the scale relative.
    """
    cube = _band_power_cube(dataset)
    if cube is None:
        return [[] for _ in range(len(meta))], None

    words = dataset.words
    geometry = _geometry(int(cube.shape[2]), montage_csv)
    out: list[list[list[list[int]]]] = []
    for _, row in meta.iterrows():
        mask = (
            (words['subject'] == row['subject'])
            & (words['task'] == row['task'])
            & (words['sentence_idx'] == row['sentence_idx'])
        )
        block = cube[np.flatnonzero(mask.to_numpy())]
        out.append(_quantise(block))
    return out, geometry


def _band_power_cube(dataset: Any) -> np.ndarray | None:
    """Returns `(n_words, n_bands, n_channels)` band power, computing it from raw windows when only those exist."""
    if getattr(dataset, 'band_power_raw', None) is not None:
        return np.asarray(dataset.band_power_raw)

    raw = getattr(dataset, 'raw_eeg', None)
    if raw is None:
        _LOG.warning('No band power and no raw windows: the studio will render without the scalp map.')
        return None

    from zte.data.features.transforms import band_power_from_raw

    flat = band_power_from_raw(np.asarray(raw))  # (n, n_channels * n_bands), channel-major
    n_bands = len(BANDS)

    return flat.reshape(flat.shape[0], -1, n_bands).transpose(0, 2, 1)


def _quantise(block: np.ndarray) -> list[list[list[int]]]:
    """Log-scales and 0-255 quantises one reading's `(words, bands, channels)` power for transport."""
    if not block.size:
        return []

    values = np.log1p(np.abs(np.asarray(block, dtype=np.float64)))
    lo, hi = float(values.min()), float(values.max())
    scaled = np.zeros_like(values) if hi <= lo else (values - lo) / (hi - lo)

    return np.rint(scaled * _QUANT_MAX).astype(int).tolist()


def _geometry(n_channels: int, montage_csv: str | None) -> ScalpGeometry | None:
    """Resolves electrode coordinates, degrading to the flagged approximate cap rather than dropping the map."""
    try:
        return resolve_geometry(n_channels, montage_csv)
    except (OSError, ValueError) as exc:
        _LOG.warning('No scalp geometry for %d channels (%r); the studio will render without it.', n_channels, exc)
        return None


def _montage_payload(geometry: ScalpGeometry | None) -> dict[str, Any] | None:
    """Returns the electrode coordinates the page draws, both as a flattened 2-D cap and in 3-D."""
    if geometry is None:
        return None

    xyz = np.asarray(geometry.xyz, dtype=np.float64)
    # Azimuthal-equidistant flattening: the standard topographic projection, where distance from the vertex is
    # preserved along every radius, so a 2-D map is comparable to a published scalp plot rather than merely pretty.
    radius = np.arccos(np.clip(xyz[:, 2], -1.0, 1.0)) / math.pi
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])

    # Rescaled so the outermost electrode sits exactly on the unit circle. A cap that reaches below the equator
    # projects past 0.5, and the page draws a fixed head outline -- unnormalised, the lowest ring falls outside it.
    flat = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1) / max(float(radius.max()), 1e-6)

    return {
        'labels': list(geometry.labels or tuple(f'E{i + 1}' for i in range(len(xyz)))),
        'xy': np.round(flat, 4).tolist(),
        'xyz': np.round(xyz, 4).tolist(),
        'approximate': bool(getattr(geometry, 'approximate', False)),
    }


def _clean(obj: Any) -> Any:
    """Recursively coerces to JSON-safe types, rounding floats and dropping non-finite values."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        value = float(obj)
        return round(value, 5) if math.isfinite(value) else None
    return str(obj)


def _esc(text: str) -> str:
    """Minimal HTML escaping for text substituted into the template."""
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


_TEMPLATE: str = load_page('studio')
