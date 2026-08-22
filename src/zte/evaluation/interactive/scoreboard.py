"""Builds the held-out scoreboard dashboard page from a `build_scoreboard` dict."""

from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
from typing import Any

from zte.evaluation.interactive._assets import load_page

_HELD_GEOMETRY_KEYS: tuple[str, ...] = ('held_out_geometry',)
_HELD_RETRIEVAL_KEYS: tuple[str, ...] = ('held_out_retrieval', 'cross_subject_holdout_retrieval')
_INSAMPLE_GEOMETRY_KEYS: tuple[str, ...] = ('in_sample_geometry', 'geometry')
_INSAMPLE_RETRIEVAL_KEYS: tuple[str, ...] = (
    'in_sample_retrieval',
    'in_sample',
    'cross_subject_retrieval',
)


def scoreboard_html(scoreboard: dict, out_path: str | Path, run_name: str = 'ZTE run') -> Path:
    """Writes the interactive held-out scoreboard dashboard as a single offline HTML file.

    Args:
        scoreboard (dict): The dict from `zte.evaluation.audit.scoreboard.build_scoreboard`; every block is optional
            and read defensively.
        out_path (str | Path): Destination `.html` path (parents are created; a non-html suffix is rewritten to
            `.html`).
        run_name (str): Human label for the run, shown in the header and the page title.

    Returns:
        Path: The written HTML file path.
    """
    out = Path(out_path)
    if out.suffix != '.html':
        out = out.with_suffix('.html')
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = _build_payload(scoreboard or {}, run_name)
    data_json = json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c')
    html = _TEMPLATE.replace('__TITLE__', _esc(run_name)).replace('__DATA__', data_json)
    out.write_text(html, encoding='utf-8')

    return out


def _build_payload(board: dict, run_name: str) -> dict[str, Any]:
    """Normalises the raw scoreboard dict into the small JSON island the page consumes."""
    lift_block = board.get('lift_over_raw') or {}
    content_probe = lift_block.get('content_probe') if isinstance(lift_block, dict) else None

    # Flatten the per-target lift table, skipping the `content_probe` sibling block.
    lift_list: list[dict[str, Any]] = []
    if isinstance(lift_block, dict):
        for target, v in lift_block.items():
            if target == 'content_probe' or not isinstance(v, dict):
                continue
            lift_list.append(
                {
                    'target': target,
                    'metric': v.get('metric'),
                    'zte': v.get('zte_linear'),
                    'raw': v.get('raw_linear'),
                    'noise': v.get('noise_linear'),
                    'lift_linear': v.get('lift_linear'),
                    'lift_knn': v.get('lift_knn'),
                    'is_content': bool(v.get('is_content')),
                    'is_identity': bool(v.get('is_identity')),
                }
            )

    # One tab per evaluation regime, omitted when the run carries neither of its blocks.
    views: dict[str, Any] = {}
    held = _view(
        'Held-out (new brain)',
        _first(board, _HELD_GEOMETRY_KEYS),
        _first(board, _HELD_RETRIEVAL_KEYS),
    )
    if held is not None:
        views['held_out'] = held
    insample = _view('In-sample', _first(board, _INSAMPLE_GEOMETRY_KEYS), _first(board, _INSAMPLE_RETRIEVAL_KEYS))
    if insample is not None:
        views['in_sample'] = insample

    payload = {
        'run_name': run_name,
        'is_loso': bool(board.get('is_loso')),
        'holdout_subject': board.get('holdout_subject'),
        'factored': bool(board.get('factored')),
        'view_order': [k for k in ('held_out', 'in_sample') if k in views],
        'views': views,
        'lift': lift_list,
        'content_probe': content_probe,
    }
    return _clean(payload)


def _view(label: str, geometry: Any, retrieval: Any) -> dict[str, Any] | None:
    """Wraps a geometry/retrieval pair into a view, or `None` if both are absent."""
    geometry = geometry if isinstance(geometry, dict) else None
    retrieval = retrieval if isinstance(retrieval, dict) else None
    if geometry is None and retrieval is None:
        return None
    return {'label': label, 'geometry': geometry, 'retrieval': retrieval}


def _first(board: dict, keys: tuple[str, ...]) -> Any:
    """Returns the first present, non-`None` value among `keys`."""
    for k in keys:
        v = board.get(k)
        if v is not None:
            return v
    return None


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


_TEMPLATE: str = load_page('scoreboard')
