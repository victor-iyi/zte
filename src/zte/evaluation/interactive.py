"""A self-contained interactive HTML explorer for thought embeddings.

TensorBoard's projector is one interactive view; this is the second, shareable one: a single `.html` file (Plotly embedded)
that opens offline in any browser, shows the embedding cloud in 3-D, lets you rotate/zoom, hover a point to read its word and
metadata, and switch the colouring between subject, task and sentence category from a dropdown -- so you can *see* whether the
space clusters by content (good) or by subject/task nuisance (bad).

Plotly is an optional dependency (the `viz` group). Without it the function falls back to a static PCA scatter PNG so a figure is always produced.
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zte.logging_utils import get_logger

_LOG = get_logger('evaluation.interactive')


def _pca(emb: np.ndarray, dims: int) -> np.ndarray:
    """Projects embeddings to `dims` principal components (centred SVD)."""
    x = np.asarray(emb, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return (x @ vt[:dims].T).astype(np.float32)


def _hover_text(meta: pd.DataFrame, columns: list[str]) -> list[str]:
    """Builds per-point hover strings from selected metadata columns."""
    parts: list[str] = []
    for _, row in meta.iterrows():
        parts.append('<br>'.join(f'{c}: {row[c]}' for c in columns if c in meta.columns))
    return parts


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
        The written path (`.html` when Plotly is available, else `.png`).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = meta.reset_index(drop=True)
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

    hover = _hover_text(meta, list(hover_cols))
    scatter = go.Scatter3d if dims == 3 else go.Scatter
    axis_kw = dict(x=coords[:, 0], y=coords[:, 1])
    if dims == 3:
        axis_kw['z'] = coords[:, 2]

    first = color_cols[0] if color_cols else None
    marker = dict(size=3 if dims == 3 else 6, opacity=0.75)
    if first is not None:
        marker['color'] = _codes(meta[first])
        marker['colorscale'] = 'Turbo'
    fig = go.Figure(scatter(**axis_kw, mode='markers', marker=marker, text=hover, hoverinfo='text'))

    buttons = []
    for col in color_cols:
        buttons.append(
            dict(
                label=f'colour: {col}',
                method='restyle',
                args=[{'marker.color': [_codes(meta[col])]}],
            )
        )
    updatemenus = [dict(buttons=buttons, x=0.0, y=1.12, xanchor='left')] if buttons else None
    fig.update_layout(
        title=f'{title}  (PCA of {len(emb)} embeddings; colour = {first})',
        updatemenus=updatemenus,
        margin=dict(l=0, r=0, t=60, b=0),
        template='plotly_white',
    )
    if out.suffix != '.html':
        out = out.with_suffix('.html')
    fig.write_html(str(out), include_plotlyjs=True, full_html=True)
    _LOG.info('Wrote interactive embedding explorer to %s', out)
    return out


def _codes(series: pd.Series) -> list[int]:
    """Integer colour codes for a categorical/continuous column."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).tolist()
    return pd.factorize(series.astype(str))[0].tolist()


def _static_fallback(
    coords: np.ndarray,
    meta: pd.DataFrame,
    color_cols: tuple[str, ...],
    title: str,
    out: Path,
) -> Path:
    """Renders a static 2-D PCA PNG when Plotly is unavailable."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    color = color_cols[0] if color_cols else None
    fig, ax = plt.subplots(figsize=(7, 6))
    if color is not None:
        for value in pd.unique(meta[color]):
            m = (meta[color] == value).to_numpy()
            ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.6, label=str(value))
        ax.legend(title=color, fontsize=7, markerscale=1.5)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.6)
    ax.set(xlabel='PC1', ylabel='PC2', title=f'{title} (static fallback; install plotly)')
    fig.tight_layout()
    out = out.with_suffix('.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    _LOG.warning('Plotly not installed; wrote static PNG fallback to %s', out)
    return out
