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

# Every gap note names the notebook cell that produces the missing artifact, so an empty panel is an instruction.
_FIX_POINTS: Final[str] = (
    '&sect;5 of notebooks/zte_parallax.ipynb (the transfer matrix) writes the embeddings this panel reduces.'
)
_FIX_CELLS: Final[str] = '&sect;5 of notebooks/zte_parallax.ipynb (the transfer matrix) produces its transfer cells.'
_FIX_MENU: Final[str] = (
    '&sect;5 of notebooks/zte_parallax.ipynb writes the menu audit into each diagonal transfer cell.'
)
_FIX_TRIAD: Final[str] = '&sect;4 of notebooks/zte_parallax.ipynb trains its arm and &sect;5 measures the pairs.'

# Rendered server-side from the capacity blocks alone, so a healthy page provably carries no trace of it:
# the class name and wording exist nowhere in the page assets.
_GAMED_NOTE: Final[str] = (
    '<div class="missnote srv-gamed-note" style="border-color:#e8a13c;color:#e8a13c;">'
    '<span class="noteicon">&#9888;</span><span>{task}: length-gamed &mdash; disqualified. Word count alone '
    'beats chance in this pool, so no capacity may be read from it.</span></div>'
)


def build_chamber(report_dir: Path, out: Path) -> Path:
    """Renders `CHAMBER_DATA.json` from a `zte-parallax report` directory as one self-contained offline HTML page.

    Args:
        report_dir (Path): Directory holding `CHAMBER_DATA.json` (and optionally `PARALLAX.json`, which
            supplies provenance and the menu decomposition).
        out (Path): Destination `.html` path (parents created; a non-html suffix is rewritten).

    Returns:
        Path: The written path.

    Raises:
        FileNotFoundError: If `report_dir` has no `CHAMBER_DATA.json`.
        ValueError: If `CHAMBER_DATA.json` is not valid JSON or is missing a required top-level key.
        ImportError: If plotly is not installed.
    """
    data = _read_chamber_data(Path(report_dir))
    sidecar = _sidecar(Path(report_dir))

    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise ImportError('plotly is required to build the Parallax Chamber; `uv sync` installs it.') from exc

    # The payload rides in an application/json island; `<` is escaped so sentence text can never close the tag.
    payload = _clean(
        {
            'data': data,
            'provenance': sidecar['provenance'],
            'menu_decomposition': sidecar['menu_decomposition'],
        }
    )
    blob = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')

    out = Path(out)
    if out.suffix != '.html':
        out = out.with_suffix('.html')
    out.parent.mkdir(parents=True, exist_ok=True)

    # Panel gaps are decided server-side so the honest state survives even with scripting disabled.
    decomp = sidecar['menu_decomposition']
    has_decomp = isinstance(decomp, dict) and any(isinstance(row, dict) and row for row in decomp.values())
    notes = _missing_notes(data)

    html = (
        load_page('chamber')
        .replace('/*__CHAMBER_PLOTLY_JS__*/', get_plotlyjs())
        .replace('__CHAMBER_DATA__', blob, 1)
        .replace('__CHAMBER_TITLE__', _esc(str(data.get('holdout', '?'))))
        .replace('__CHAMBER_MISS_PARALLAX__', notes['parallax'])
        .replace('__CHAMBER_MISS_FLOW__', notes['flow'])
        .replace('__CHAMBER_MISS_DIALS__', notes['dials'])
        .replace('__CHAMBER_GAMED_NOTES__', _gamed_notes(data))
        .replace('__CHAMBER_MISS_RAIN__', notes['rain'])
        .replace('__CHAMBER_MISS_TRIAD__', notes['triad'])
        .replace('__CHAMBER_DECOMP_PLOT_HIDDEN__', '' if has_decomp else ' hidden')
        .replace('__CHAMBER_DECOMP_NOTE_HIDDEN__', ' hidden' if has_decomp else '')
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


def _sidecar(report_dir: Path) -> dict[str, Any]:
    """Provenance and the menu decomposition from `PARALLAX.json`; the page renders without either."""
    empty: dict[str, Any] = {'provenance': None, 'menu_decomposition': None}
    path = report_dir / 'PARALLAX.json'
    if not path.is_file():
        return empty

    try:
        parallax = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning('PARALLAX.json under %s is unreadable (%r); the page renders without provenance.', report_dir, exc)
        return empty

    if not isinstance(parallax, dict):
        return empty

    # The footer prints scalars only, so the nested git dict is flattened to the commit it carries.
    prov = parallax.get('provenance')
    if isinstance(prov, dict) and isinstance(prov.get('git'), dict) and 'git_commit' not in prov:
        prov = {**prov, 'git_commit': prov['git'].get('commit')}

    decomp = parallax.get('menu_decomposition')

    return {
        'provenance': {'seeds': parallax.get('seeds'), 'provenance': prov},
        'menu_decomposition': decomp if isinstance(decomp, dict) else None,
    }


def _missing_notes(data: dict[str, Any]) -> dict[str, str]:
    """Static per-panel gap notes naming each declared task a panel cannot draw, and the cell that fills it."""
    tasks = [str(t) for t in data.get('tasks') or []]
    points = _dict_or_empty(data.get('points'))
    transfer = _dict_or_empty(data.get('transfer'))
    capacity = _dict_or_empty(data.get('capacity'))
    cka = _dict_or_empty(data.get('cka'))

    has_points = {t for t in tasks if points.get(t)}
    has_transfer = {
        t for t in tasks if transfer.get(t) or any(t in row for row in transfer.values() if isinstance(row, dict))
    }
    has_capacity = {t for t in tasks if isinstance(capacity.get(t), dict)}
    in_cka = {part for key in cka for part in str(key).split('|')}
    has_triad = {t for t in tasks if t in in_cka or t in has_transfer or t in has_points}

    return {
        'parallax': _notes_for(tasks, has_points, 'has no sentence points in this report', _FIX_POINTS),
        'flow': _notes_for(tasks, has_transfer, 'has no transfer cells in this report', _FIX_CELLS),
        'dials': _notes_for(tasks, has_capacity, 'has no menu-capacity audit in this report', _FIX_MENU),
        'rain': _notes_for(tasks, has_points, 'has no percentile drops in this report', _FIX_POINTS),
        'triad': _notes_for(tasks, has_triad, 'is missing from the CKA triad', _FIX_TRIAD),
    }


def _dict_or_empty(value: Any) -> dict[str, Any]:
    """The value when it is a JSON object, else an empty dict."""
    return value if isinstance(value, dict) else {}


def _gamed_notes(data: dict[str, Any]) -> str:
    """One amber disqualification note per arm whose capacity block (or open diagnostic) is length-gamed."""
    capacity = _dict_or_empty(data.get('capacity'))
    tainted = []
    for task, block in capacity.items():
        if not isinstance(block, dict):
            continue
        open_block = _dict_or_empty(block.get('open'))
        if block.get('gamed') or open_block.get('gamed'):
            tainted.append(str(task))

    return ''.join(_GAMED_NOTE.format(task=_esc(task)) for task in sorted(tainted))


def _notes_for(tasks: list[str], have: set[str], gap: str, fix: str) -> str:
    """One gap-note card per declared-but-absent task; empty when nothing is missing or nothing is drawable."""

    # With nothing drawable at all the page's own full empty-state card owns the panel; a note would double it.
    if not have:
        return ''

    return ''.join(
        f'<div class="missnote"><span class="noteicon">&#9676;</span><span>{_esc(t)} {gap} &mdash; {fix}</span></div>'
        for t in tasks
        if t not in have
    )


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
