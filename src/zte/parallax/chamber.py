"""Builds the Parallax Chamber: one offline page where three task-specific encoders view the same sentences."""

import json
import math
import numbers
from pathlib import Path
from typing import Any, Final

from zte.logging_utils import get_logger
from zte.parallax._page import load_page

_LOG = get_logger('parallax.chamber')

# The page cannot draw a single panel without these; optional sections (capacity, cka, provenance) degrade per panel.
_REQUIRED_KEYS: Final[tuple[str, ...]] = ('tasks', 'points', 'transfer')


def build_chamber(report_dir: Path, out: Path) -> Path:
    """Renders `CHAMBER_DATA.json` from a `zte-parallax report` directory as one self-contained offline HTML page.

    Args:
        report_dir (Path): Directory holding `CHAMBER_DATA.json` (and optionally `PARALLAX.json` for provenance).
        out (Path): Destination `.html` path (parents created; a non-html suffix is rewritten).

    Returns:
        Path: The written path.

    Raises:
        FileNotFoundError: If `report_dir` has no `CHAMBER_DATA.json`.
        ValueError: If `CHAMBER_DATA.json` is not valid JSON or is missing a required top-level key.
        ImportError: If plotly is not installed.
    """
    data = _read_chamber_data(Path(report_dir))
    provenance = _provenance(Path(report_dir))

    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise ImportError('plotly is required to build the Parallax Chamber; `uv sync` installs it.') from exc

    # The payload rides in an application/json island; `<` is escaped so sentence text can never close the tag.
    payload = _clean({'data': data, 'provenance': provenance})
    blob = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')

    out = Path(out)
    if out.suffix != '.html':
        out = out.with_suffix('.html')
    out.parent.mkdir(parents=True, exist_ok=True)

    html = (
        load_page('chamber')
        .replace('/*__CHAMBER_PLOTLY_JS__*/', get_plotlyjs())
        .replace('__CHAMBER_DATA__', blob, 1)
        .replace('__CHAMBER_TITLE__', _esc(str(data.get('holdout', '?'))))
    )
    out.write_text(html, encoding='utf-8')
    _LOG.info('Parallax Chamber written to %s (%.1f MB).', out, out.stat().st_size / 1e6)

    return out


def _read_chamber_data(report_dir: Path) -> dict[str, Any]:
    """Loads and validates `CHAMBER_DATA.json`, failing loudly rather than rendering an empty page."""
    path = report_dir / 'CHAMBER_DATA.json'
    if not path.is_file():
        raise FileNotFoundError(f'No CHAMBER_DATA.json under {report_dir}; run `zte-parallax report` first.')

    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'{path} is not valid JSON: {exc}') from exc

    if not isinstance(data, dict):
        raise ValueError(f'{path} must hold a JSON object, not {type(data).__name__}.')
    if missing := [key for key in _REQUIRED_KEYS if key not in data]:
        raise ValueError(f'{path} is missing required keys: {", ".join(missing)}.')

    return data


def _provenance(report_dir: Path) -> dict[str, Any] | None:
    """Pulls seeds and provenance from `PARALLAX.json` when the report ships one; the page renders without it."""
    path = report_dir / 'PARALLAX.json'
    if not path.is_file():
        return None

    try:
        parallax = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning('PARALLAX.json under %s is unreadable (%r); the page renders without provenance.', report_dir, exc)
        return None

    if not isinstance(parallax, dict):
        return None

    # The footer prints scalars only, so the nested git dict is flattened to the commit it carries.
    prov = parallax.get('provenance')
    if isinstance(prov, dict) and isinstance(prov.get('git'), dict) and 'git_commit' not in prov:
        prov = {**prov, 'git_commit': prov['git'].get('commit')}

    return {'seeds': parallax.get('seeds'), 'provenance': prov}


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
