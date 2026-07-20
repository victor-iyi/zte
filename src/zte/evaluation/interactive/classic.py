"""Builds the classic word-embedding scatter page: a PCA projection with a colour-by control."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zte.evaluation.interactive._assets import load_page
from zte.evaluation.interactive._common import _pca, _static_fallback
from zte.logging_utils import get_logger

_LOG = get_logger('evaluation.interactive')
_CLASSIC_TEMPLATE: str = load_page('classic')


def _hover_text(meta: pd.DataFrame, columns: list[str]) -> list[str]:
    """Builds per-point hover strings from selected metadata columns."""
    parts: list[str] = []
    for _, row in meta.iterrows():
        parts.append('<br>'.join(f'{c}: {row[c]}' for c in columns if c in meta.columns))
    return parts


def _codes(series: pd.Series) -> list[int]:
    """Integer colour codes for a categorical/continuous column."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).tolist()
    return pd.factorize(series.astype(str))[0].tolist()


def embedding_explorer_html(
    emb: np.ndarray,
    meta: pd.DataFrame,
    out_path: str | Path,
    color_cols: tuple[str, ...] = ('subject', 'task', 'category', 'length_band'),
    hover_cols: tuple[str, ...] = ('word', 'subject', 'task', 'category'),
    title: str = 'ZTE thought-embedding explorer',
    dims: int = 3,
    max_points: int = 8000,
    seed: int = 0,
) -> Path:
    """Writes a self-contained interactive HTML (or a static PNG fallback).

    Args:
        emb (np.ndarray): Embeddings `(n_samples, embed_dim)`.
        meta (pd.DataFrame): Aligned metadata used for colouring and hover.
        out_path (str | Path): Output path (`.html` for interactive, `.png` on fallback).
        color_cols (tuple[str, ...]): Metadata columns offered in the colour-by dropdown.
        hover_cols (tuple[str, ...]): Metadata columns shown on hover.
        title (str): Figure title.
        dims (int): 3 for a 3-D scatter, 2 for a 2-D scatter.
        max_points (int): Subsample cap for responsiveness.
        seed (int): Sampling seed.

    Returns:
        Path: The written path (`.html` when Plotly is available, else `.png`).
    """
    import json

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = meta.reset_index(drop=True)

    # Subsample for responsiveness, then project to the plotted dimensions.
    idx = np.arange(len(emb))
    if len(emb) > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(len(emb), size=max_points, replace=False))
    emb, meta = np.asarray(emb)[idx], meta.iloc[idx].reset_index(drop=True)
    coords = _pca(emb, dims)
    color_cols = tuple(c for c in color_cols if c in meta.columns)

    try:
        import plotly.graph_objects as go
    except ImportError:
        return _static_fallback(coords, meta, color_cols, title, out)

    # Light colorway plus its brightened dark-theme counterpart, swapped in-browser.
    way_light: list[str] = [
        '#5a4bff',
        '#0f9d6b',
        '#e29008',
        '#008300',
        '#ff4d8d',
        '#e5484d',
        '#e87ba4',
        '#eb6834',
    ]

    way_dark: list[str] = [
        '#9a86ff',
        '#3ddc97',
        '#f5b642',
        '#3fc46a',
        '#ff77a9',
        '#ff6b6f',
        '#f2a3c0',
        '#ff9a5c',
    ]

    tokens = {
        'light': {
            'panel': '#ffffff',
            'ink': '#131720',
            'ink2': '#495264',
            'border': '#e3e7ef',
            'grid': '#e3e7ef',
            'plane': '#eef1f6',
        },
        'dark': {
            'panel': '#161b26',
            'ink': '#eef2f9',
            'ink2': '#aab4c6',
            'border': '#232a37',
            'grid': '#232a37',
            'plane': '#0b0e14',
        },
    }

    # Precompute discrete category codes per colour column (JSON-serialisable).
    colorby: dict[str, dict[str, list]] = {}
    for c in color_cols:
        codes, uniques = pd.factorize(meta[c].astype(str))
        colorby[c] = {'codes': [int(k) for k in codes], 'cats': [str(u) for u in uniques]}

    hover = _hover_text(meta, list(hover_cols))
    scatter = go.Scatter3d if dims == 3 else go.Scatter
    axis_kw = dict(x=coords[:, 0], y=coords[:, 1])
    if dims == 3:
        axis_kw['z'] = coords[:, 2]

    first = color_cols[0] if color_cols else None
    marker: dict[str, Any] = dict(size=3 if dims == 3 else 6, opacity=0.82)
    if first is not None:
        cds = colorby[first]['codes']
        marker['color'] = [way_light[k % len(way_light)] for k in cds]
    else:
        marker['color'] = way_light[0]

    # Bake the light theme into the figure so the first paint needs no reflow.
    fig = go.Figure(scatter(**axis_kw, mode='markers', marker=marker, text=hover, hoverinfo='text'))
    lt = tokens['light']
    fig.update_layout(
        paper_bgcolor=lt['panel'],
        plot_bgcolor=lt['panel'],
        font=dict(color=lt['ink'], family='system-ui,-apple-system,"Segoe UI",sans-serif', size=12),
        margin=dict(l=0, r=0, t=8, b=0),
        showlegend=False,
        hoverlabel=dict(bgcolor='#161b26', font=dict(color='#fff', family='system-ui,sans-serif')),
    )
    if dims == 3:
        ax = dict(
            gridcolor=lt['grid'],
            zerolinecolor=lt['border'],
            backgroundcolor=lt['plane'],
            showbackground=True,
            color=lt['ink2'],
            title='',
        )
        fig.update_layout(scene=dict(xaxis=dict(ax), yaxis=dict(ax), zaxis=dict(ax)))
    else:
        ax = dict(
            gridcolor=lt['grid'],
            zerolinecolor=lt['border'],
            linecolor=lt['border'],
            color=lt['ink2'],
            title='',
        )
        fig.update_layout(xaxis=dict(ax), yaxis=dict(ax))

    # Splice the figure and its config island into the page template.
    fig_html = fig.to_html(
        include_plotlyjs=True,
        full_html=False,
        div_id='plot',
        config={'displaylogo': False, 'responsive': True},
    )

    cfg = {
        'dims': dims,
        'palette': {'light': way_light, 'dark': way_dark},
        'colorby': colorby,
        'tokens': tokens,
        'columns': list(color_cols),
    }
    caption = (
        f'PCA projection of <b>{len(emb)}</b> word embeddings, {dims}-D. Rotate, zoom and hover a '
        'point; recolour to test whether the space clusters by <b>content</b> (good) or by '
        'subject / task nuisance (bad).'
    )
    title_html = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    data_json = json.dumps(cfg).replace('<', '\\u003c')
    html = (
        _CLASSIC_TEMPLATE.replace('__TITLE__', title_html)
        .replace('__CAPTION__', caption)
        .replace('__FIGURE__', fig_html)
        .replace('__DATA__', data_json)
    )

    if out.suffix != '.html':
        out = out.with_suffix('.html')
    out.write_text(html, encoding='utf-8')

    _LOG.info('Wrote interactive embedding explorer to %s', out)
    return out
