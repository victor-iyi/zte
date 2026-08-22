"""Renders one `lens.json` capture as a self-contained offline HTML inspection page."""

import json
import math
import numbers
from pathlib import Path
from typing import Any, Final

from zte.lens._page import load_page
from zte.logging_utils import get_logger

_LOG = get_logger('lens.page')

# The page cannot walk one reading without these; optional sections (scalp, decode) degrade per panel.
_REQUIRED_KEYS: Final[tuple[str, ...]] = ('mode', 'reading', 'word_saliency', 'neighbors', 'disclaimer')

# The only two captures `zte-lens` writes; anything else is a corrupted or foreign file, not a render request.
_MODES: Final[tuple[str, ...]] = ('encode', 'decode')


def build_lens_page(json_path: Path, out: Path) -> Path:
    """Renders one `lens.json` from `zte-lens` as one self-contained offline HTML page.

    Args:
        json_path (Path): Path to a `lens.json` capture of a single reading.
        out (Path): Destination `.html` path (parents created; a non-html suffix is rewritten).

    Returns:
        Path: The written path.

    Raises:
        FileNotFoundError: If `json_path` does not exist.
        ValueError: If the file is not valid JSON, is missing a required key, has an unknown mode,
            or carries no disclaimer -- the disclaimer is mandatory, never defaulted in.
        ImportError: If plotly is not installed.
    """
    data = _read_lens_data(Path(json_path))
    reading = data['reading'] if isinstance(data['reading'], dict) else {}

    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise ImportError('plotly is required to build the lens page; `uv sync` installs it.') from exc

    # The payload rides in an application/json island; `<` is escaped so sentence text can never close the tag.
    blob = json.dumps(_clean(data), separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')

    out = Path(out)
    if out.suffix != '.html':
        out = out.with_suffix('.html')
    out.parent.mkdir(parents=True, exist_ok=True)

    # Panel presence is decided server-side so the honest state survives even with scripting disabled.
    has_decode = isinstance(data.get('decode'), dict)
    has_scalp = isinstance(data.get('channel_saliency'), dict)

    subject = str(reading.get('subject', '?'))
    text = str(reading.get('text', ''))
    title = f'{subject} · {text[:57]}...' if len(text) > 60 else f'{subject} · {text}'

    html = (
        load_page('lens')
        .replace('/*__LENS_PLOTLY_JS__*/', get_plotlyjs())
        .replace('__LENS_DATA__', blob, 1)
        .replace('__LENS_TITLE__', _esc(title))
        .replace('__LENS_DECODE_HIDDEN__', '' if has_decode else ' hidden')
        .replace('__LENS_SCALP_PLOT_HIDDEN__', '' if has_scalp else ' hidden')
        .replace('__LENS_SCALP_NOTE_HIDDEN__', ' hidden' if has_scalp else '')
    )
    out.write_text(html, encoding='utf-8')
    _LOG.info('Lens page written to %s (%.1f MB).', out, out.stat().st_size / 1e6)

    return out


def _read_lens_data(json_path: Path) -> dict[str, Any]:
    """Loads and validates one `lens.json`, failing loudly rather than rendering an empty page."""
    if not json_path.is_file():
        raise FileNotFoundError(f'No lens capture at {json_path}; run `zte-lens encode` or `zte-lens decode` first.')

    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'{json_path} is not valid JSON: {exc}') from exc

    if not isinstance(data, dict):
        raise ValueError(f'{json_path} must hold a JSON object, not {type(data).__name__}.')
    if missing := [key for key in _REQUIRED_KEYS if key not in data]:
        raise ValueError(f'{json_path} is missing required keys: {", ".join(missing)}.')
    if data['mode'] not in _MODES:
        raise ValueError(f'{json_path} has unknown mode {data["mode"]!r}; expected one of {", ".join(_MODES)}.')

    # The disclaimer is an honesty guard, not decoration: a capture without one is refused, never patched over.
    if not (isinstance(data['disclaimer'], str) and data['disclaimer'].strip()):
        raise ValueError(f'{json_path} carries no disclaimer; refusing to render an inspection page without one.')

    return data


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
