"""Builds the Neuron Atlas page: ranked importance and per-neuron selectivity detail."""

from __future__ import annotations

from pathlib import Path

from zte.evaluation.interactive._assets import load_page
from zte.evaluation.interactive._common import _escape, _json_safe
from zte.logging_utils import get_logger

_LOG = get_logger('evaluation.interactive')
_ATLAS_TEMPLATE: str = load_page('atlas')


def neuron_atlas_html(
    neurons: dict,
    out_path: str | Path,
    *,
    title: str = 'ZTE Neuron Atlas',
    max_bars: int | None = None,
) -> Path:
    """Writes a self-contained interactive "Neuron Atlas" from a `neuron_report` dict.

    Args:
        neurons (dict): The report from `zte.evaluation.neurons.neuron_report`; missing keys degrade gracefully.
        out_path (str | Path): Output path (`.html`, or `.png` on the Plotly fallback).
        title (str): Page and figure title.
        max_bars (int | None): Cap on the neurons the ranked chart draws; detail/search still cover all of them.

    Returns:
        Path: The written path (`.html` when Plotly is available, else a static `.png`).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    neurons = neurons or {}

    try:
        from plotly.offline import get_plotlyjs
    except ImportError:
        return _neuron_atlas_fallback(neurons, title, out)

    import json

    payload = {'data': _json_safe(neurons), 'max_bars': max_bars}
    html = (
        _ATLAS_TEMPLATE.replace('/*__ATLAS_PLOTLY_JS__*/', get_plotlyjs())
        .replace('"__ATLAS_PAYLOAD__"', json.dumps(payload, separators=(',', ':')))
        .replace('__ATLAS_TITLE__', _escape(title))
    )
    if out.suffix != '.html':
        out = out.with_suffix('.html')
    out.write_text(html, encoding='utf-8')

    dim = (neurons.get('meta') or {}).get('embed_dim', '?')
    _LOG.info('Wrote Neuron Atlas (%s neurons) to %s', dim, out)
    return out


def _neuron_atlas_fallback(neurons: dict, title: str, out: Path) -> Path:
    """Renders a static ranked-importance PNG when Plotly is unavailable."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Rank by variance share, with the active-neuron threshold on the same scale.
    imp = neurons.get('importance', {}) or {}
    std = imp.get('std', []) or []
    order = imp.get('order') or list(range(len(std)))
    var_share = imp.get('var_share', []) or []
    dominant = (neurons.get('selectivity', {}) or {}).get('dominant', []) or []
    thr = float(imp.get('active_threshold', 0.0) or 0.0)
    tot = float(sum(s * s for s in std)) or 1.0
    thr_vs = (thr * thr) / tot

    # Colour each bar by the attribute that neuron is most selective for.
    palette = {
        'subject': '#eda100',
        'word_len': '#2a78d6',
        'log_freq': '#1baf7a',
        'category': '#4a3aa7',
        'task': '#008300',
        'none': '#b8b6ad',
    }
    ys = [var_share[d] for d in order] if var_share else []
    colors = [palette.get(dominant[d] if d < len(dominant) else 'none', '#8a6cd6') for d in order]

    fig, ax = plt.subplots(figsize=(9, 4))
    if ys:
        ax.bar(range(len(ys)), ys, color=colors, width=1.0)
    ax.axhline(thr_vs, ls='--', color='#e34948', lw=1, label='active threshold')
    ax.set(
        xlabel='importance rank (0 = most important)',
        ylabel='variance share',
        title=f'{title} (static fallback; install plotly for the interactive atlas)',
    )

    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out.with_suffix('.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)

    _LOG.warning('Plotly not installed; wrote static Neuron Atlas PNG to %s', out)
    return out
