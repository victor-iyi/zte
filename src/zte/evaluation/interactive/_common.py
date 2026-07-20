"""Shared PCA, escaping and fallback-rendering helpers for the interactive page builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def _pca_basis(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the `(mean, components)` PCA basis so new vectors can be projected.

    Args:
        emb (np.ndarray): Embeddings `(n_samples, embed_dim)`.

    Returns:
        tuple[np.ndarray, np.ndarray]: `(mean, vt)` where `mean` is `(embed_dim,)` and
            `vt` is the `(k, embed_dim)` matrix of principal directions (rows).
    """
    x = np.asarray(emb, dtype=np.float64)
    mean = x.mean(axis=0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean.astype(np.float32), vt.astype(np.float32)


def _project(x: np.ndarray, mean: np.ndarray, vt: np.ndarray, dims: int) -> np.ndarray:
    """Projects `x` onto the first `dims` PCA directions of a fitted basis."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return (x - mean) @ vt[:dims].T


def _escape(text: str) -> str:
    """Minimal HTML escaping for the injected title."""
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _json_safe(obj: Any) -> Any:
    """Converts a report dict to strict-JSON / valid-JS-literal types.

    Numpy scalars become builtins and non-finite floats become `None`, so the injected payload parses in the browser.

    Args:
        obj (Any): Any nested combination of dicts, lists, numpy or builtin scalars.

    Returns:
        Any: The same structure using only JSON-safe Python builtins.
    """
    import math

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


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
