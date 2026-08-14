"""Builds the reference/hypothesis/controls side-by-side page from a `generation_report` dict."""

from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
from typing import Any

from zte.evaluation.interactive._assets import load_page

# Metrics the page tabulates and can sort rows by; the primary one leads.
_METRICS: tuple[str, ...] = (
    'content_f1',
    'rouge1',
    'rouge2',
    'rougeL',
    'bleu1',
    'sentence_bleu4',
    'wer',
)


def generation_html(block: dict[str, Any], out_path: str | Path, run_name: str = 'ZTE run') -> Path:
    """Writes the generation side-by-side as a single offline HTML file.

    Every hypothesis is rendered beside the identical row for each brain-independent control, which is
    the one artifact that makes an absolute BLEU unreadable on its own.

    Args:
        block (dict[str, Any]): The dict from `zte.evaluation.generation.generation_report`; every
            sub-block is optional and read defensively.
        out_path (str | Path): Destination `.html` path (parents are created; a non-html suffix is
            rewritten to `.html`).
        run_name (str, optional): Human label for the run, shown in the header and the page title.
            Defaults to 'ZTE run'.

    Returns:
        Path: The written HTML file path.
    """
    out = Path(out_path)
    if out.suffix != '.html':
        out = out.with_suffix('.html')
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = _build_payload(block or {}, run_name)
    data_json = json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c')
    html = _TEMPLATE.replace('__TITLE__', _esc(run_name)).replace('__DATA__', data_json)
    out.write_text(html, encoding='utf-8')

    return out


def _build_payload(block: dict[str, Any], run_name: str) -> dict[str, Any]:
    """Normalises the generation report into the small JSON island the page consumes."""
    if not block.get('applicable'):
        return _clean(
            {
                'run_name': run_name,
                'applicable': False,
                'reason': block.get('reason', 'no generation block'),
            }
        )

    metric = block.get('primary_metric', 'content_f1')
    absolute = block.get('absolute') or {}
    controls = absolute.get('controls') or {}
    order = ['hypothesis', *controls.keys()]
    scores = {'hypothesis': absolute.get('hypothesis') or {}}
    scores.update({name: value or {} for name, value in controls.items()})
    if absolute.get('oracle'):
        order.append('oracle')
        scores['oracle'] = absolute['oracle']

    from zte.evaluation.report import HONEST_SPLIT

    perm = block.get('permutation') or {}
    worst = block.get('worst_control_ci') or {}
    rows = block.get('rows') or []
    absent = (block.get('controls_unavailable') or {}) | (block.get('controls_skipped') or {})
    payload = {
        'run_name': run_name,
        'applicable': True,
        'n': block.get('n'),
        'split': block.get('split'),
        'split_strategy': block.get('split_strategy'),
        'honest_split': (block.get('split_strategy'), block.get('split')) == HONEST_SPLIT,
        'honest_split_required': f'{HONEST_SPLIT[1]} cell of {HONEST_SPLIT[0]}',
        'free': block.get('n_candidate_sentences') is None,
        'n_candidate_sentences': block.get('n_candidate_sentences'),
        'primary_metric': metric,
        'metrics': [metric, *[m for m in _METRICS if m != metric]],
        'condition_order': order,
        'absolute': scores,
        'deltas': {name: delta.get(metric, {}) for name, delta in (block.get('deltas') or {}).items()},
        'verdict': {
            'beats_all_controls': bool(block.get('beats_all_controls')) and not absent,
            'controls_absent': sorted(absent),
            'worst_control': block.get('worst_control'),
            'worst_ci': worst,
            'permutation_p': perm.get('p_value') if perm.get('applicable') else None,
            'prefix_kl': block.get('prefix_influence_kl'),
        },
        'rows': rows,
        'truncated': bool(block.get('n') and len(rows) < int(block['n'])),
    }
    return _clean(payload)


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
        f = float(obj)
        return round(f, 5) if math.isfinite(f) else None
    return str(obj)


def _esc(text: str) -> str:
    """Minimal HTML escaping for text substituted into the template."""
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


_TEMPLATE: str = load_page('generation')
