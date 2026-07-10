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


# --------------------------------------------------------------------------- #
# Thought-Space Explorer -- the flagship interactive HTML.
# --------------------------------------------------------------------------- #

_REDUCED_DIMS = 64  # PCA width kept for in-browser cosine / analogy arithmetic.


def _round(a: np.ndarray, nd: int = 4) -> list:
    """Rounds an array and returns nested Python lists (JSON-friendly, compact)."""
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def _l2_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalisation."""
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _word_cross_subject_stats(
    emb: np.ndarray, words: np.ndarray, subjects: np.ndarray, min_subjects: int = 2
) -> dict[str, dict[str, float]]:
    """Per-word mean pairwise cosine of its per-subject centroids.

    For each surface word read by at least `min_subjects` subjects, the mean
    embedding per subject is computed and the mean of the upper-triangular
    pairwise cosine similarities across those subjects is returned -- the core
    "does this word mean the same thing in different brains?" statistic.

    Args:
        emb (np.ndarray): Embeddings `(n, d)`.
        words (np.ndarray): Surface word per row `(n,)`.
        subjects (np.ndarray): Subject id per row `(n,)`.
        min_subjects (int): Minimum distinct subjects for a word to qualify.

    Returns:
        dict[str, dict[str, float]]: `word -> {mean_cos, n_subj}`.
    """
    stats: dict[str, dict[str, float]] = {}
    words = np.asarray(words).astype(str)
    subjects = np.asarray(subjects).astype(str)
    for w in np.unique(words):
        wmask = words == w
        subj_here = np.unique(subjects[wmask])
        if subj_here.size < min_subjects:
            continue
        centroids = np.stack([emb[wmask & (subjects == s)].mean(axis=0) for s in subj_here])
        centroids = _l2_rows(centroids)
        sims = centroids @ centroids.T
        iu = np.triu_indices(len(subj_here), k=1)
        stats[str(w)] = {
            'mean_cos': round(float(sims[iu].mean()), 4),
            'n_subj': int(subj_here.size),
        }
    return stats


def _random_baseline_cos(emb: np.ndarray, n_pairs: int = 4000, seed: int = 0) -> float:
    """Mean cosine of random point pairs -- the "unrelated thoughts" baseline."""
    u = _l2_rows(emb)
    if len(u) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(u), size=n_pairs)
    b = rng.integers(0, len(u), size=n_pairs)
    keep = a != b
    if not keep.any():
        return 0.0
    return round(float(np.mean(np.sum(u[a[keep]] * u[b[keep]], axis=1))), 4)


def _analogy_candidates(
    words: np.ndarray, subjects: np.ndarray, *, cap: int = 300, seed: int = 0
) -> list[dict]:
    """Enumerates subject-transfer analogy candidates the browser scores as hit/miss.

    A candidate is a surface word `t` read by at least two subjects, together with
    an ordered source/target pair `A -> B` that both read it. In the browser each
    becomes the analogy `v = emb(t, A) - centroid(A) + centroid(B)`, scored a HIT
    when `v`'s nearest neighbour (cosine, over the reduced vectors) is `t` as read
    by `B`. This removes all manual "which word / which person" guessing: every
    viable analogy is pre-listed and the leaderboard ranks them.

    Args:
        words (np.ndarray): Surface word per row `(n,)` (already subsampled).
        subjects (np.ndarray): Subject id per row `(n,)`.
        cap (int): Maximum number of candidates (random subset if exceeded), to
            keep the in-browser scan fast and the file small.
        seed (int): Sampling seed.

    Returns:
        list[dict]: `{t, A, B, ai, bi}` rows where `ai`/`bi` are the first-occurrence
            row indices of `(t, A)` and `(t, B)` in the subsampled arrays.
    """
    words = np.asarray(words).astype(str)
    subjects = np.asarray(subjects).astype(str)
    first: dict[tuple[str, str], int] = {}
    for i in range(len(words)):
        key = (words[i], subjects[i])
        if key not in first:
            first[key] = i
    by_word: dict[str, list[str]] = {}
    for w, s in first:
        by_word.setdefault(w, []).append(s)
    cands: list[dict] = []
    for w, subs in by_word.items():
        if len(subs) < 2 or not w.strip():
            continue
        ordered = sorted(subs)
        for a in ordered:
            for b in ordered:
                if a != b:
                    cands.append({'t': w, 'A': a, 'B': b, 'ai': first[(w, a)], 'bi': first[(w, b)]})
    if len(cands) > cap:
        sel = np.random.default_rng(seed).choice(len(cands), size=cap, replace=False)
        cands = [cands[i] for i in sorted(sel)]
    return cands


def _coord_block(emb: np.ndarray, mean: np.ndarray, vt: np.ndarray) -> dict[str, list]:
    """Projects embeddings to a 3-D coordinate block for plotting."""
    c = _project(emb, mean, vt, 3)
    block = {'x': _round(c[:, 0], 4), 'y': _round(c[:, 1], 4)}
    block['z'] = _round(c[:, 2], 4) if c.shape[1] > 2 else _round(np.zeros(len(c)), 4)
    return block


def _centroid_blocks(
    emb: np.ndarray,
    subjects: np.ndarray,
    mean: np.ndarray,
    vt: np.ndarray,
    override: dict | None,
) -> tuple[dict[str, list], dict[str, list]]:
    """Per-subject reduced-space and projected-space centroids for arithmetic."""
    reduced: dict[str, list] = {}
    proj: dict[str, list] = {}
    subjects = np.asarray(subjects).astype(str)
    for s in np.unique(subjects):
        vec = None
        if override is not None and s in override:
            vec = np.asarray(override[s], dtype=np.float32)
        if vec is None or vec.shape[0] != emb.shape[1]:
            vec = emb[subjects == s].mean(axis=0)
        reduced[str(s)] = _round(_project(vec, mean, vt, _REDUCED_DIMS)[0], 5)
        proj[str(s)] = _round(_project(vec, mean, vt, 3)[0], 4)
    return reduced, proj


def _build_payload(
    emb: np.ndarray,
    meta: pd.DataFrame,
    *,
    eeg_only_emb: np.ndarray | None,
    centroids: dict | None,
    probe_scores: dict | None,
    emergence: dict | None,
    dims: int,
    title: str,
    seed: int,
) -> dict:
    """Assembles the full JSON payload consumed by the in-browser explorer."""
    n = len(emb)
    reduced_dims = min(_REDUCED_DIMS, emb.shape[1])

    mean, vt = _pca_basis(emb)
    reduced = _project(emb, mean, vt, reduced_dims)
    subjects = meta['subject'].astype(str).to_numpy() if 'subject' in meta else np.array(['S'] * n)
    words = meta['word'].astype(str).to_numpy() if 'word' in meta else np.array([''] * n)

    cats = [c for c in ('subject', 'task', 'category', 'length_band') if c in meta.columns]
    nums = [c for c in ('word_len', 'log_freq') if c in meta.columns]

    meta_block: dict[str, list] = {}
    for c in ('subject', 'task', 'word', 'category', 'length_band'):
        meta_block[c] = meta[c].astype(str).tolist() if c in meta.columns else [''] * n
    for c in ('sentence_idx', 'word_idx'):
        meta_block[c] = meta[c].fillna(-1).astype(int).tolist() if c in meta.columns else [-1] * n
    meta_block['word_len'] = (
        meta['word_len'].fillna(0).astype(int).tolist()
        if 'word_len' in meta.columns
        else [len(w) for w in words]
    )
    meta_block['log_freq'] = (
        _round(np.nan_to_num(meta['log_freq'].to_numpy(dtype=float)), 4)
        if 'log_freq' in meta.columns
        else [0.0] * n
    )
    if 'word_len' not in nums:
        nums.insert(0, 'word_len')

    cent_reduced, cent_proj = _centroid_blocks(emb, subjects, mean, vt, centroids)
    coords = {'et': _coord_block(emb, mean, vt)}
    reduced_block = {'et': _round(reduced, 4)}
    cents = {'et': cent_reduced}
    cents_proj = {'et': cent_proj}
    has_eeg = False
    if eeg_only_emb is not None and len(eeg_only_emb) == n and np.asarray(eeg_only_emb).ndim == 2:
        has_eeg = True
        e_mean, e_vt = _pca_basis(eeg_only_emb)
        coords['eeg'] = _coord_block(eeg_only_emb, e_mean, e_vt)
        reduced_block['eeg'] = _round(
            _project(eeg_only_emb, e_mean, e_vt, min(_REDUCED_DIMS, eeg_only_emb.shape[1])), 4
        )
        er, ep = _centroid_blocks(eeg_only_emb, subjects, e_mean, e_vt, None)
        cents['eeg'] = er
        cents_proj['eeg'] = ep

    return {
        'title': title,
        'n': int(n),
        'dims_default': int(dims),
        'meta': meta_block,
        'coords': coords,
        'reduced': reduced_block,
        'centroids': cents,
        'centroids_proj': cents_proj,
        'fields': cats,
        'numeric_fields': nums,
        'subjects': sorted({str(s) for s in subjects}),
        'words': sorted({str(w) for w in words if str(w).strip()}),
        'word_stats': _word_cross_subject_stats(emb, words, subjects),
        'random_baseline': _random_baseline_cos(emb, seed=seed),
        'analogy_candidates': _analogy_candidates(words, subjects, seed=seed),
        'probe': probe_scores or None,
        'emergence': _json_safe(emergence) if emergence else None,
        'has_eeg_only': has_eeg,
    }


def thought_space_explorer_html(
    emb: np.ndarray,
    meta: pd.DataFrame,
    out_path: str | Path,
    *,
    eeg_only_emb: np.ndarray | None = None,
    centroids: dict | None = None,
    probe_scores: dict | None = None,
    dims: int = 3,
    max_points: int = 6000,
    seed: int = 0,
    title: str = 'ZTE Thought-Space Explorer',
    emergence: dict | None = None,
) -> Path:
    """Writes the flagship self-contained interactive "Thought-Space Explorer".

    One offline `.html` file (Plotly embedded) designed to be interpretable in
    seconds. A plain-language "What am I looking at?" guide and two always-on
    verdict banners -- "Same word across subjects" and "Do meanings cluster across
    people?" -- answer the headline question up front (both computed in-browser
    from the embedded reduced vectors, honestly reporting *clustered / weakly /
    not*). Below sit five wired-up views: (1) one subject / many words, (2) one
    word across many brains (with a cross-subject cosine statistic), (3) thought
    arithmetic `emb(t,A) - centroid(A) + centroid(B)` fronted by an auto-computed,
    sortable analogy *leaderboard* (so the viewer never has to guess which word or
    person), (4) a semantic-neighbourhood view (k nearest neighbours of a chosen
    word + a coherence stat), and (5) an eye-tracking vs EEG-only toggle. Live
    controls (colour-by, visible subjects, word filter, 2-D/3-D, theme) drive
    everything via `Plotly.react`. Falls back to a static PCA PNG (mirroring
    :func:`embedding_explorer_html`) when Plotly is unavailable.

    Args:
        emb (np.ndarray): Word-level embeddings `(n_samples, embed_dim)` (EEG +
            eye-tracking, the primary set).
        meta (pd.DataFrame): Aligned metadata; recognises `subject, task, word,
            sentence_idx, word_idx, category, length_band` and optional `word_len,
            log_freq`. Missing columns degrade gracefully.
        out_path (str | Path): Output path (`.html`, or `.png` on fallback).
        eeg_only_emb (np.ndarray | None): Optional EEG-only embeddings aligned
            row-for-row with `emb`, enabling the view-4 toggle.
        centroids (dict | None): Optional `{subject: vector}` full-dim centroid
            override for the arithmetic offset (else computed from `emb`).
        probe_scores (dict | None): Optional `word_len` linear-probe scores, e.g.
            `{'EEG + eye-tracking': 0.72, 'EEG-only': 0.41}`, rendered as a small bar.
        dims (int): Default projection (3 for 3-D, 2 for 2-D).
        max_points (int): Subsample cap for responsiveness.
        seed (int): Sampling / baseline seed.
        title (str): Page + figure title.
        emergence (dict | None): Optional canonical, full-embedding-space emergence
            report from :func:`zte.evaluation.emergence.emergence_report` (the same
            numbers that land in ``metrics.json``). When supplied, the three verdict
            banners show these authoritative figures and their ``verdict`` strings as
            the headline, and label the in-browser reduced-space number a secondary
            "live estimate (PCA space)". When ``None``, the banners keep today's
            in-browser estimate as the headline (unchanged behaviour).

    Returns:
        Path: The written path (`.html` when Plotly is available, else `.png`).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = meta.reset_index(drop=True)

    idx = np.arange(len(emb))
    if len(emb) > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(len(emb), size=max_points, replace=False))
    emb = np.asarray(emb, dtype=np.float32)[idx]
    meta = meta.iloc[idx].reset_index(drop=True)
    eeg_sub = None
    if eeg_only_emb is not None:
        eeg_arr = np.asarray(eeg_only_emb, dtype=np.float32)
        if eeg_arr.ndim == 2 and len(eeg_arr) == len(emb):
            # Already the subsampled length -- take as-is.
            eeg_sub = eeg_arr
        elif eeg_arr.ndim == 2 and len(idx) and eeg_arr.shape[0] > idx.max():
            # Original length: apply the same subsample so rows stay aligned to `emb`.
            eeg_sub = eeg_arr[idx]
        else:
            _LOG.warning(
                'eeg_only_emb length %d does not align with emb; ignoring for view 4.',
                len(eeg_arr),
            )

    try:
        import plotly.graph_objects as go  # noqa: F401
        from plotly.offline import get_plotlyjs
    except ImportError:
        color_cols = tuple(c for c in ('subject', 'task', 'category') if c in meta.columns)
        return _static_fallback(_pca(emb, 2), meta, color_cols, title, out)

    payload = _build_payload(
        emb,
        meta,
        eeg_only_emb=eeg_sub,
        centroids=centroids,
        probe_scores=probe_scores,
        emergence=emergence,
        dims=dims,
        title=title,
        seed=seed,
    )

    import json

    html = (
        _EXPLORER_TEMPLATE.replace('/*__PLOTLY_JS__*/', get_plotlyjs())
        .replace('"__PAYLOAD__"', json.dumps(payload, separators=(',', ':')))
        .replace('__TITLE__', _escape(title))
    )
    if out.suffix != '.html':
        out = out.with_suffix('.html')
    out.write_text(html, encoding='utf-8')
    _LOG.info('Wrote Thought-Space Explorer (%d points) to %s', payload['n'], out)
    return out


def _escape(text: str) -> str:
    """Minimal HTML escaping for the injected title."""
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


# --------------------------------------------------------------------------- #
# The single-file HTML template. `/*__PLOTLY_JS__*/`, `"__PAYLOAD__"` and
# `__TITLE__` are substituted at write time. Plotly is inlined so the page is
# fully offline with no external hosts.
# --------------------------------------------------------------------------- #

_EXPLORER_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script>/*__PLOTLY_JS__*/</script>
<style>
:root{
  --surface:#fcfcfb; --plane:#f4f4f1; --panel:#ffffff;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --accent:#2a78d6; --good:#0ca30c; --warn:#eda100; --bad:#e34948;
}
:root[data-theme="dark"]{
  --surface:#1a1a19; --plane:#0d0d0d; --panel:#202020;
  --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.12);
  --accent:#3987e5; --good:#0ca30c; --warn:#c98500; --bad:#e66767;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{
  background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; font-size:14px;
  display:grid; grid-template-columns:340px 1fr; grid-template-rows:100vh;
}
a{color:var(--accent)}
.rail{
  background:var(--panel); border-right:1px solid var(--border);
  padding:18px 18px 28px; overflow-y:auto; height:100vh;
}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:4px}
.brand h1{font-size:16px;margin:0;font-weight:650;letter-spacing:.2px}
.brand .dot{width:11px;height:11px;border-radius:50%;
  background:conic-gradient(from 210deg,#2a78d6,#1baf7a,#eda100,#e34948,#4a3aa7,#2a78d6)}
.sub{color:var(--ink2);font-size:12px;line-height:1.5;margin:6px 0 16px}
.tabs{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:16px}
.tab{
  border:1px solid var(--border); background:transparent; color:var(--ink2);
  border-radius:9px; padding:9px 8px; font-size:12px; cursor:pointer; text-align:left;
  line-height:1.25; transition:all .12s;
}
.tab b{display:block;color:var(--ink);font-size:12.5px;font-weight:600}
.tab:hover{border-color:var(--axis)}
.tab.on{border-color:var(--accent); background:color-mix(in srgb,var(--accent) 12%,transparent)}
.tab.on b{color:var(--accent)}
.group{border-top:1px solid var(--border);padding:14px 0}
.group:first-of-type{border-top:none}
.group h3{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);
  margin:0 0 10px;font-weight:650}
.row{margin-bottom:11px}
label.lab{display:block;font-size:12px;color:var(--ink2);margin-bottom:5px}
select,input[type=text],input[type=number]{
  width:100%; padding:7px 9px; border-radius:8px; border:1px solid var(--border);
  background:var(--surface); color:var(--ink); font-size:13px; font-family:inherit;
}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.seg button{border:none;background:transparent;color:var(--ink2);padding:6px 14px;
  font-size:12.5px;cursor:pointer;font-family:inherit}
.seg button.on{background:var(--accent);color:#fff}
.seg button:disabled{opacity:.4;cursor:not-allowed}
.checks{display:flex;flex-wrap:wrap;gap:6px}
.chk{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--border);
  border-radius:999px;padding:4px 10px;font-size:12px;cursor:pointer;user-select:none}
.chk input{accent-color:var(--accent)}
.chk .sw{width:9px;height:9px;border-radius:50%}
.hide{display:none!important}
.main{display:flex;flex-direction:column;height:100vh;min-width:0}
.topbar{display:flex;align-items:center;gap:12px;padding:10px 20px;
  border-bottom:1px solid var(--border);background:var(--panel)}
.topbar .story{font-size:12.5px;color:var(--ink2);line-height:1.45;flex:1;min-width:0}
.topbar .story b{color:var(--ink)}
.stat{display:flex;gap:16px;align-items:center}
.stat .cell{text-align:right}
.stat .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted)}
.stat .v{font-size:18px;font-weight:650;font-variant-numeric:tabular-nums}
.stat .v.good{color:var(--good)} .stat .v.warn{color:var(--warn)} .stat .v.bad{color:var(--bad)}
.chipbtn{border:1px solid var(--border);background:transparent;color:var(--ink2);
  border-radius:8px;padding:6px 11px;font-size:12px;cursor:pointer;font-family:inherit;white-space:nowrap}
.chipbtn:hover{border-color:var(--axis)}
.guide{margin:10px 20px 0;background:var(--panel);border:1px solid var(--border);
  border-radius:12px;padding:13px 16px;font-size:12.5px;color:var(--ink2);line-height:1.55}
.guide b{color:var(--ink)}
.guide .lgd{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:10px;font-size:12px}
.guide .lg{display:inline-flex;align-items:center;gap:6px}
.guide .lg .sw{width:11px;height:11px;border-radius:50%}
.banners{display:flex;gap:12px;padding:12px 20px 4px;flex-wrap:wrap}
.banner{flex:1;min-width:250px;background:var(--panel);border:1px solid var(--border);
  border-radius:12px;padding:12px 14px}
.banner .bt{font-size:12px;color:var(--ink);font-weight:640;margin-bottom:7px}
.banner .brow{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.banner .bnum{font-size:21px;font-weight:660;font-variant-numeric:tabular-nums}
.banner .bsub{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.banner .bcap{font-size:11px;color:var(--muted);line-height:1.5;margin-top:7px}
.bsrc{flex-basis:100%;font-size:10.5px;color:var(--muted);line-height:1.45;padding:0 4px}
.verdict{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11.5px;font-weight:650}
.verdict.good{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.verdict.warn{background:color-mix(in srgb,var(--warn) 20%,transparent);color:var(--warn)}
.verdict.bad{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}
.verdict.na{background:color-mix(in srgb,var(--muted) 18%,transparent);color:var(--muted)}
.plotwrap{position:relative;flex:1;min-height:120px}
#plot{width:100%;height:100%}
.bottom{border-top:1px solid var(--border);background:var(--panel);
  max-height:38vh;overflow:auto;padding:9px 16px 12px}
.lbhead{display:flex;align-items:center;gap:14px;margin:1px 0 8px;flex-wrap:wrap}
.lbhead .hh{font-size:12.5px;font-weight:650;color:var(--ink)}
.lbhead .cap{font-size:11px;color:var(--muted);flex:1;min-width:180px;line-height:1.4}
.lbhead .big{font-size:12.5px;font-weight:650;font-variant-numeric:tabular-nums}
.lbhead .big.good{color:var(--good)} .lbhead .big.warn{color:var(--warn)} .lbhead .big.bad{color:var(--bad)}
table.tbl{width:100%;border-collapse:collapse;font-size:12px}
table.tbl th{text-align:left;color:var(--muted);font-weight:600;font-size:10.5px;
  text-transform:uppercase;letter-spacing:.4px;border-bottom:1px solid var(--border);
  padding:4px 8px;cursor:pointer;user-select:none;white-space:nowrap}
table.tbl th.up::after{content:" \25B2";color:var(--accent)}
table.tbl th.down::after{content:" \25BC";color:var(--accent)}
table.tbl td{padding:4px 8px;border-bottom:1px solid var(--border);color:var(--ink2)}
table.tbl td.num{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}
table.tbl tr.clk{cursor:pointer}
table.tbl tr.clk:hover{background:color-mix(in srgb,var(--accent) 9%,transparent)}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;
  font-weight:650;font-variant-numeric:tabular-nums}
.pill.hit{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.pill.miss{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}
.theme-btn{border:1px solid var(--border);background:transparent;color:var(--ink2);
  border-radius:8px;padding:6px 10px;font-size:12px;cursor:pointer;font-family:inherit}
.note{font-size:11.5px;color:var(--muted);line-height:1.5;margin-top:6px}
#barwrap .cap{font-size:11px;color:var(--muted);padding:2px 4px 4px}
#bar{height:120px}
@media (max-width:860px){
  body{grid-template-columns:1fr;grid-template-rows:auto 1fr}
  .rail{height:auto;max-height:46vh}
}
</style>
</head>
<body>
<aside class="rail">
  <div class="brand"><span class="dot"></span><h1>Thought-Space Explorer</h1></div>
  <div class="sub">Word-level ZTE embeddings, projected with PCA. The question throughout:
    does this neural space organise by <b>what was read</b> (good) or by
    <b>who read it</b> (a nuisance)?</div>

  <div class="tabs" id="tabs"></div>

  <div class="group">
    <h3>Projection &amp; colour</h3>
    <div class="row"><label class="lab">Colour points by</label><select id="colorby"></select></div>
    <div class="row"><label class="lab">Dimensions</label>
      <span class="seg" id="dims">
        <button data-d="2">2-D</button><button data-d="3">3-D</button>
      </span>
    </div>
  </div>

  <div class="group">
    <h3>Visible subjects</h3>
    <div class="checks" id="subjchecks"></div>
  </div>

  <div class="group v-ctrl v1">
    <h3>View 1 &middot; one subject</h3>
    <div class="row"><label class="lab">Highlight subject</label><select id="subj1"></select></div>
    <div class="row"><label class="lab">Shade highlighted by</label><select id="metric1"></select></div>
    <div class="note">Everyone else fades to grey so one reader's cloud stands out.</div>
  </div>

  <div class="group v-ctrl v2 hide">
    <h3>View 2 &middot; one word, many brains</h3>
    <div class="row"><label class="lab">Word</label>
      <input type="text" id="word2" list="wordlist" autocomplete="off"></div>
    <div class="note">Every occurrence of this word lights up, coloured by reader. The stat bar
      compares how alike the word is <b>across brains</b> vs unrelated thoughts.</div>
  </div>

  <div class="group v-ctrl v3 hide">
    <h3>View 3 &middot; thought arithmetic</h3>
    <div class="row"><label class="lab">Word <i>t</i></label>
      <input type="text" id="wordT" list="wordlist" autocomplete="off"></div>
    <div class="row"><label class="lab">Source brain A</label><select id="subjA"></select></div>
    <div class="row"><label class="lab">Target brain B</label><select id="subjB"></select></div>
    <div class="note"><b>v = emb(t,A) &minus; centroid(A) + centroid(B)</b>. The arrow re-aims
      A's thought at B. A hit means v's nearest neighbour is <i>t</i> as read by B. The
      <b>leaderboard below</b> ranks every viable analogy for you &mdash; click a row.</div>
  </div>

  <div class="group v-ctrl v4 hide">
    <h3>View 4 &middot; nearest thoughts</h3>
    <div class="row"><label class="lab">Word</label>
      <input type="text" id="wordN" list="wordlist" autocomplete="off"></div>
    <div class="row"><label class="lab">Read by</label><select id="subjN"></select></div>
    <div class="row"><label class="lab">Neighbours (k)</label>
      <input type="number" id="kN" min="3" max="50" step="1"></div>
    <div class="note">The k closest points in ZTE space (cosine). Coherence = how many are the
      <b>same word or category</b> vs chance &mdash; the concrete "are similar thoughts near
      each other?" test.</div>
  </div>

  <div class="group v-ctrl v5 hide">
    <h3>View 5 &middot; eye-tracking's role</h3>
    <div class="row"><label class="lab">Signal set</label>
      <span class="seg" id="embset">
        <button data-s="et">EEG + eye-tracking</button><button data-s="eeg">EEG-only</button>
      </span>
    </div>
    <div class="note" id="v5note"></div>
  </div>

  <div class="group">
    <button class="theme-btn" id="themebtn">Toggle light / dark</button>
  </div>
</aside>

<main class="main">
  <div class="topbar">
    <button class="chipbtn" id="guidebtn">Hide guide</button>
    <div class="story" id="story"></div>
    <div class="stat" id="statbox"></div>
  </div>

  <div class="guide" id="guide"></div>
  <div class="banners" id="banners"></div>

  <div class="plotwrap"><div id="plot"></div></div>

  <div class="bottom hide" id="bottom">
    <div id="leaderpanel" class="hide">
      <div class="lbhead">
        <span class="hh">Auto-analogy leaderboard</span>
        <span class="big" id="hitrate"></span>
        <button class="chipbtn" id="surprise">Surprise me &rarr;</button>
        <span class="cap">Every word read by &ge;2 people, transferred A&rarr;B via
          <b>v = emb(t,A) &minus; centroid(A) + centroid(B)</b>. <b>Hit</b> = v's nearest neighbour
          is <i>t</i> read by B. <b>rank</b> = position of the true target among B's points.
          Click a row to draw it.</span>
      </div>
      <table class="tbl" id="leaderboard"></table>
    </div>
    <div id="neighpanel" class="hide">
      <div class="lbhead">
        <span class="hh">Nearest thoughts</span>
        <span class="big" id="coherence"></span>
        <span class="cap">k nearest points to the chosen token by cosine over the reduced ZTE
          vectors. <b>same-word / same-category</b> shares are compared to their chance rates
          (how common that word / category is overall).</span>
      </div>
      <table class="tbl" id="neightable"></table>
    </div>
    <div id="barwrap" class="hide">
      <div class="cap">Linear-probe accuracy / R&sup2; on <b>word length</b> &mdash; a decodable
        content attribute, higher = more content carried.</div>
      <div id="bar"></div>
    </div>
  </div>
</main>

<datalist id="wordlist"></datalist>

<script>
const P = "__PAYLOAD__";
const M = P.meta, N = P.n;

const PAL = {
  light:['#2a78d6','#1baf7a','#eda100','#008300','#4a3aa7','#e34948','#e87ba4','#eb6834'],
  dark: ['#3987e5','#199e70','#c98500','#008300','#9085e9','#e66767','#d55181','#d95926'],
};
const SEQ = {
  light:[[0,'#cde2fb'],[0.5,'#3987e5'],[1,'#0d366b']],
  dark: [[0,'#173a63'],[0.5,'#3987e5'],[1,'#cde2fb']],
};
const INK = {
  light:{paper:'#fcfcfb',grid:'#e1e0d9',axis:'#c3c2b7',text:'#0b0b0b',muted:'#898781',faint:'#d7d7d2'},
  dark: {paper:'#1a1a19',grid:'#2c2c2a',axis:'#383835',text:'#ffffff',muted:'#898781',faint:'#3a3a37'},
};

const DOMAINS = {};
P.fields.forEach(f=>{ DOMAINS[f]=[...new Set(M[f])].filter(v=>v!=='').sort(); });
if(!DOMAINS['subject']) DOMAINS['subject']=P.subjects.slice();
const HAS_CAT = P.fields.includes('category') && DOMAINS['category'] && DOMAINS['category'].length>1;

// ---- helpers -------------------------------------------------------------
const pick=(arr,idx)=>idx.map(i=>arr[i]);
const catColor=(field,val)=>{const d=DOMAINS[field]||[];const i=d.indexOf(val);return PAL[state.theme][(i<0?0:i)%8];};
function hover(i){
  return `<b>${M.word[i]||'.'}</b><br>subject ${M.subject[i]} - ${M.task[i]}`
       + `<br>${M.category[i]} - len ${M.word_len[i]}`;
}
function idxVisible(){const o=[];for(let i=0;i<N;i++) if(state.visible.has(M.subject[i])) o.push(i); return o;}
function idxWord(w){const o=[];for(let i=0;i<N;i++) if(M.word[i]===w && state.visible.has(M.subject[i])) o.push(i); return o;}
function findIdx(w,s){for(let i=0;i<N;i++) if(M.word[i]===w && M.subject[i]===s) return i; return -1;}
function findFirst(w){for(let i=0;i<N;i++) if(M.word[i]===w) return i; return -1;}
function norm(v){let s=0;for(const x of v)s+=x*x;s=Math.sqrt(s)||1;return v.map(x=>x/s);}
function dot(a,b){let s=0;for(let k=0;k<a.length;k++)s+=a[k]*b[k];return s;}
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

// reduced (64-d) normalised vectors + per-subject index lists (primary "et" set)
const RN = (P.reduced.et||[]).map(norm);
const SUBJIDX = {};
for(let i=0;i<N;i++){ (SUBJIDX[M.subject[i]]=SUBJIDX[M.subject[i]]||[]).push(i); }

function bestWord(){
  let best=P.words[0]||'', bn=-1;
  for(const w in P.word_stats){ if(P.word_stats[w].n_subj>bn){bn=P.word_stats[w].n_subj;best=w;} }
  return best;
}

const state = {
  view:1,
  colorBy: P.fields.includes('subject')?'subject':(P.fields[0]||P.numeric_fields[0]),
  dims: P.dims_default===2?2:3,
  visible: new Set(P.subjects),
  subj1: P.subjects[0],
  metric1: P.numeric_fields[0]||'word_len',
  word2: bestWord(),
  wordT: bestWord(),
  subjA: P.subjects[0],
  subjB: P.subjects[1]||P.subjects[0],
  wordN: bestWord(),
  subjN: 'any',
  kN: 12,
  embSet:'et',
  guide:true,
  theme: (window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches)?'dark':'light',
  lbSort:'default', lbDir:-1,
};

// ---- in-browser analytics: verdict banners -------------------------------
function verdict(delta){
  if(delta==null) return {label:'n/a', cls:'na'};
  if(delta < 0.03) return {label:'not clustered', cls:'bad'};
  if(delta < 0.10) return {label:'weakly clustered', cls:'warn'};
  return {label:'clustered', cls:'good'};
}
// mean cosine of random point pairs in the reduced space (in-browser baseline)
let RAND_BL = 0;
function randBaseline(){
  let s=0,c=0; if(N<2) return 0;
  for(let t=0;t<4000;t++){ const a=(Math.random()*N)|0, b=(Math.random()*N)|0;
    if(a!==b){ s+=dot(RN[a],RN[b]); c++; } }
  return c? s/c : 0;
}
// same-word across subjects: mean over words of the pairwise cosine of per-subject centroids
function sameWordAcross(){
  let s=0, nw=0, nsub=0;
  const byWord={};
  for(let i=0;i<N;i++){ const w=M.word[i]; if(!w) continue;
    (byWord[w]=byWord[w]||{})[M.subject[i]]=(byWord[w][M.subject[i]]||[]); byWord[w][M.subject[i]].push(i); }
  for(const w in byWord){
    const subs=Object.keys(byWord[w]); if(subs.length<2) continue;
    const cents=subs.map(su=>{ const ids=byWord[w][su]; const c=new Array(RN[0].length).fill(0);
      ids.forEach(i=>{const r=RN[i];for(let k=0;k<c.length;k++)c[k]+=r[k];}); return norm(c); });
    let sm=0,np=0; for(let a=0;a<cents.length;a++)for(let b=a+1;b<cents.length;b++){sm+=dot(cents[a],cents[b]);np++;}
    s+=sm/np; nw++; nsub+=subs.length;
  }
  return nw? {cos:s/nw, words:nw} : null;
}
// same-category across subjects: sampled cosine of cross-subject same-category pairs
function sameCategoryAcross(){
  if(!HAS_CAT) return null;
  const byCat={};
  for(let i=0;i<N;i++){ const c=M.category[i]; if(!c) continue; (byCat[c]=byCat[c]||[]).push(i); }
  const cats=Object.keys(byCat).filter(c=>{
    const subs=new Set(byCat[c].map(i=>M.subject[i])); return subs.size>=2; });
  if(!cats.length) return null;
  let s=0,c=0,tries=0;
  while(c<3000 && tries<160000){ tries++;
    const cat=cats[(Math.random()*cats.length)|0], pool=byCat[cat];
    const i=pool[(Math.random()*pool.length)|0], j=pool[(Math.random()*pool.length)|0];
    if(i!==j && M.subject[i]!==M.subject[j]){ s+=dot(RN[i],RN[j]); c++; }
  }
  return c? {cos:s/c, pairs:c} : null;
}

function vcls(s){ return s==='clustered'?'good':s==='weakly clustered'?'warn':s==='not clustered'?'bad':'na'; }
function fmt(x,d){ return (x==null||!isFinite(x))?'--':(+x).toFixed(d==null?3:d); }
function pctOr(x){ return (x==null||!isFinite(x))?'--':(x*100).toFixed(0)+'%'; }
function transferVerdict(gap){ return gap==null||!isFinite(gap)?{label:'n/a',cls:'na'}
  : gap<0.02?{label:'rarely transfers',cls:'bad'}
  : gap<0.10?{label:'sometimes transfers',cls:'warn'}:{label:'often transfers',cls:'good'}; }
function bannerCard(title,num,sub,vlabel,cls,cap){
  return `<div class="banner">
     <div class="bt">${title}</div>
     <div class="brow"><span class="bnum">${num}</span>
       ${sub?`<span class="bsub">${sub}</span>`:''}
       <span class="verdict ${cls}">${vlabel}</span></div>
     <div class="bcap">${cap}</div>
   </div>`;
}

function renderBanners(){
  RAND_BL = randBaseline();
  const box=document.getElementById('banners'); box.innerHTML='';
  const EM = P.emergence;
  const CS = (EM && EM.cross_subject && EM.cross_subject.applicable) ? EM.cross_subject : null;

  // ---- Banner 1: same word across subjects --------------------------------
  const w=sameWordAcross();
  const liveW = w? w.cos : null, liveWd = w? verdict(w.cos-RAND_BL) : verdict(null);
  const sw = CS ? CS.same_word : null;
  if(sw){
    box.insertAdjacentHTML('beforeend', bannerCard('Same word across subjects',
      fmt(sw.mean_cosine), `vs ${fmt(sw.random_baseline)} random`, sw.verdict, vcls(sw.verdict),
      `Full-embedding-space mean cosine of the same word read by different subjects, minus the random
       cross-subject baseline (&Delta;=${fmt(sw.gap)}).
       <b>Live estimate (PCA space):</b> ${fmt(liveW)} vs ${fmt(RAND_BL)} random &rarr; ${liveWd.label}.`));
  } else {
    box.insertAdjacentHTML('beforeend', bannerCard('Same word across subjects',
      fmt(liveW), `vs ${fmt(RAND_BL)} random`, liveWd.label, liveWd.cls,
      `Mean cosine of the same word's per-subject centroids (${w?w.words:0} words read by &ge;2 people),
       minus the random-pair baseline. &Delta;=${w?fmt(w.cos-RAND_BL):'--'} <i>(in-browser PCA-space estimate)</i>.`));
  }

  // ---- Banner 2: do meanings cluster (same category) ----------------------
  const c=sameCategoryAcross();
  const liveC = c? c.cos : null, liveCd = c? verdict(c.cos-RAND_BL) : null;
  const sm = CS ? CS.same_meaning : null;
  if(sm){
    box.insertAdjacentHTML('beforeend', bannerCard('Do meanings cluster across people?',
      fmt(sm.mean_cosine), `vs ${fmt(sm.random_baseline)} random`, sm.verdict, vcls(sm.verdict),
      `Full-embedding-space mean cosine of <b>same-category, different-subject</b> pairs (meaning proxy =
       <code>category</code>), minus random (&Delta;=${fmt(sm.gap)}).`
       + (c?` <b>Live estimate (PCA space):</b> ${fmt(liveC)} &rarr; ${liveCd.label}.`:'')));
  } else if(c){
    box.insertAdjacentHTML('beforeend', bannerCard('Do meanings cluster across people?',
      fmt(liveC), `vs ${fmt(RAND_BL)} random`, liveCd.label, liveCd.cls,
      `Mean cosine of <b>same-category, different-subject</b> pairs (meaning proxy = <code>category</code>;
       ${c.pairs} sampled), minus random. &Delta;=${fmt(c.cos-RAND_BL)} <i>(in-browser PCA-space estimate)</i>.`));
  } else {
    box.insertAdjacentHTML('beforeend', bannerCard('Do meanings cluster across people?',
      '--', '', 'no category column', 'na',
      `Needs a <code>category</code> label read by &ge;2 subjects to use it as a meaning proxy.
       Falling back to the same-word measure on the left.`));
  }

  // ---- Banner 3: can we translate a thought (analogy) ---------------------
  const hr = LB.length? LB.filter(x=>x.hit).length/LB.length : null;
  const liveHd = transferVerdict(hr);
  const an = (EM && EM.analogy) ? EM.analogy : null;
  const top1 = an ? an.subject_transfer_top1 : null, chance = an ? an.chance_top1 : null;
  if(an && top1!=null && isFinite(top1)){
    const gap = isFinite(chance)? top1-chance : top1;
    const ad = transferVerdict(gap);
    box.insertAdjacentHTML('beforeend', bannerCard('Can we translate a thought between people?',
      pctOr(top1), `vs ${pctOr(chance)} chance`, ad.label, ad.cls,
      `Full-embedding-space subject-transfer Top-1 hit rate: v = emb(t,A) &minus; centroid(A) + centroid(B)
       retrieves <i>t</i> read by B.`
       + (hr==null?'':` <b>Live estimate (PCA space):</b> ${pctOr(hr)} over ${LB.length} analogies &rarr; ${liveHd.label}.`)));
  } else {
    box.insertAdjacentHTML('beforeend', bannerCard('Can we translate a thought between people?',
      pctOr(hr), `${LB.length} analogies`, liveHd.label, liveHd.cls,
      `Share of A&rarr;B transfers whose nearest neighbour is the right word for B
       <i>(in-browser PCA-space estimate)</i>. Open <b>View 3</b> for the ranked leaderboard.`));
  }

  // ---- provenance caption -------------------------------------------------
  box.insertAdjacentHTML('beforeend', EM
    ? `<div class="bsrc">Headline figures are the canonical <b>full-embedding-space</b> values from
        <code>metrics.json</code> (emergence report); "live estimate" numbers are computed in-browser from
        the PCA-reduced vectors and may differ slightly.</div>`
    : `<div class="bsrc">Figures are in-browser estimates over the PCA-reduced vectors. When an emergence
        report is present, these banners instead headline the canonical full-embedding-space values from
        <code>metrics.json</code>.</div>`);
}

function renderGuide(){
  const g=document.getElementById('guide');
  if(!state.guide){ g.classList.add('hide'); return; }
  g.classList.remove('hide');
  const subjLg = P.subjects.slice(0,8).map((s,i)=>
    `<span class="lg"><span class="sw" style="background:${PAL[state.theme][i%8]}"></span>${esc(s)}</span>`).join('');
  g.innerHTML =
    `<b>What am I looking at?</b> Each point is <b>one word read by one person</b>, placed by
     their EEG (brain) response. ZTE's goal is that the <b>same meaning read by different people</b>
     lands in the same place &mdash; the way word embeddings put "cat" and "dog" near each other.
     The banners below measure whether that is happening yet; the five views and colours let you
     test it yourself. <span style="color:var(--muted)">Honest status: ZTE v1 largely encodes
     <i>who</i> is reading, not <i>what</i> &mdash; expect weak cross-subject clustering.</span>
     <div class="lgd"><b style="color:var(--ink2)">subjects:</b>${subjLg}</div>`;
}

// ---- analogy leaderboard (computed once, in-browser, on the primary set) --
let LB = [];
function computeLeaderboard(){
  const R=P.reduced.et, CENT=P.centroids.et, cands=P.analogy_candidates||[];
  LB = cands.map(c=>{
    const A=CENT[c.A], B=CENT[c.B], base=R[c.ai];
    if(!A||!B||!base) return null;
    const v=norm(base.map((x,k)=>x-A[k]+B[k]));
    const bl=SUBJIDX[c.B]||[];
    let best=-1,bestSim=-2,trueSim=-2;
    for(const i of bl){ const s=dot(RN[i],v); if(i===c.bi) trueSim=s; if(s>bestSim){bestSim=s;best=i;} }
    let rank=1; for(const i of bl){ if(i!==c.bi && dot(RN[i],v)>trueSim) rank++; }
    return {t:c.t,A:c.A,B:c.B,ai:c.ai,bi:c.bi,
      hit: best>=0 && M.word[best]===c.t, nn: best>=0?M.word[best]:'-',
      sim:bestSim, rank};
  }).filter(Boolean);
  sortLB();
}
function sortLB(){
  const k=state.lbSort, d=state.lbDir;
  const key={default:x=>x.hit?1e6+x.sim:x.sim, word:x=>x.t, ab:x=>x.A+x.B,
    hit:x=>x.hit?1:0, nn:x=>x.nn, rank:x=>-x.rank, sim:x=>x.sim};
  const f=key[k]||key.default;
  LB.sort((a,b)=>{ const x=f(a),y=f(b);
    if(x<y)return -d; if(x>y)return d; return 0; });
}
function renderLeaderboard(){
  const hr=LB.length? LB.filter(x=>x.hit).length/LB.length : 0;
  const hd=document.getElementById('hitrate');
  const cls=hr<0.15?'bad':hr<0.4?'warn':'good';
  hd.className='big '+cls;
  hd.innerHTML=`hit-rate ${(hr*100).toFixed(0)}% <span style="color:var(--muted);font-weight:500">(${LB.filter(x=>x.hit).length}/${LB.length})</span>`;
  const cols=[['word','word t'],['ab','A → B'],['hit','hit'],['nn','v’s NN'],
    ['rank','true rank'],['sim','cos(v,NN)']];
  let h='<tr>'+cols.map(([k,l])=>{
    const arrow=state.lbSort===k?(state.lbDir<0?'up':'down'):''; // note: default not on a col
    return `<th data-k="${k}" class="${arrow}">${l}</th>`; }).join('')+'</tr>';
  const rows=LB.slice(0,120).map((x,i)=>
    `<tr class="clk" data-i="${LB.indexOf(x)}">
       <td>${esc(x.t)}</td><td>${esc(x.A)} → ${esc(x.B)}</td>
       <td><span class="pill ${x.hit?'hit':'miss'}">${x.hit?'✓':'✗'}</span></td>
       <td>${esc(x.nn)}</td><td class="num">${x.rank}</td><td class="num">${x.sim.toFixed(3)}</td>
     </tr>`).join('');
  const t=document.getElementById('leaderboard'); t.innerHTML=h+rows;
  t.querySelectorAll('th').forEach(th=>th.onclick=()=>{
    const k=th.dataset.k;
    if(state.lbSort===k) state.lbDir=-state.lbDir; else {state.lbSort=k;state.lbDir=-1;}
    sortLB(); renderLeaderboard();
  });
  t.querySelectorAll('tr.clk').forEach(tr=>tr.onclick=()=>openAnalogy(LB[+tr.dataset.i]));
}
function openAnalogy(x){
  if(!x) return;
  state.wordT=x.t; state.subjA=x.A; state.subjB=x.B;
  const wt=document.getElementById('wordT'); if(wt) wt.value=x.t;
  const sa=document.getElementById('subjA'); if(sa) sa.value=x.A;
  const sb=document.getElementById('subjB'); if(sb) sb.value=x.B;
  setView(3);
}

// ---- semantic neighbourhood ----------------------------------------------
function neighbourQuery(){
  return state.subjN==='any'? findFirst(state.wordN) : findIdx(state.wordN, state.subjN);
}
function neighbours(qi, k){
  const q=RN[qi], out=[];
  for(let i=0;i<N;i++){ if(i===qi) continue; out.push([i,dot(RN[i],q)]); }
  out.sort((a,b)=>b[1]-a[1]);
  return out.slice(0,k);
}
function renderNeighTable(qi, nn){
  const t=document.getElementById('neightable');
  if(qi<0){ t.innerHTML='<tr><td class="num" style="color:var(--muted)">token not found</td></tr>';
    document.getElementById('coherence').textContent=''; return; }
  const qw=M.word[qi], qc=M.category[qi];
  let sw=0, sc=0;
  const rows=nn.map(([i,s])=>{
    const isw=M.word[i]===qw, isc=HAS_CAT && M.category[i]===qc; if(isw)sw++; if(isc)sc++;
    const tag = isw?'<span class="pill hit">same word</span>'
      : isc?`<span class="pill" style="background:color-mix(in srgb,var(--accent) 16%,transparent);color:var(--accent)">same cat</span>`:'';
    return `<tr><td>${esc(M.word[i])}</td><td>${esc(M.subject[i])}</td><td>${esc(M.category[i])}</td>
      <td class="num">${s.toFixed(3)}</td><td>${tag}</td></tr>`; }).join('');
  t.innerHTML='<tr><th>neighbour word</th><th>subject</th><th>category</th><th>cosine</th><th></th></tr>'+rows;
  const k=nn.length||1;
  const wc=M.word.filter(w=>w===qw).length, cc=HAS_CAT?M.category.filter(c=>c===qc).length:0;
  const chW=(wc-1)/Math.max(1,N-1), chC=(cc-1)/Math.max(1,N-1);
  const co=document.getElementById('coherence');
  const lift=(sw/k)/Math.max(1e-6,chW);
  const cls=lift>3?'good':lift>1.3?'warn':'bad';
  co.className='big '+cls;
  co.innerHTML=`same-word ${(100*sw/k).toFixed(0)}% <span style="color:var(--muted);font-weight:500">(chance ${(100*chW).toFixed(1)}%)</span>`
    + (HAS_CAT?` &middot; same-cat ${(100*sc/k).toFixed(0)}% <span style="color:var(--muted);font-weight:500">(chance ${(100*chC).toFixed(1)}%)</span>`:'');
}

// ---- plotly trace builders ----------------------------------------------
let CUR=null;
function mk(idx,o){
  o=o||{}; const is3=state.dims===3;
  const t={type:is3?'scatter3d':'scattergl',mode:'markers',
    x:pick(CUR.x,idx), y:pick(CUR.y,idx),
    text:idx.map(hover), hoverinfo:'text',
    name:o.name||'', showlegend:o.showlegend!==false && !!o.name,
    marker:{size:o.size||(is3?3.4:7.5), opacity:o.opacity==null?0.9:o.opacity, line:{width:0}}};
  if(is3) t.z=pick(CUR.z,idx);
  if(o.color) t.marker.color=o.color;
  if(o.cvals){t.marker.color=o.cvals;t.marker.colorscale=o.colorscale;t.marker.showscale=!!o.showscale;
    t.marker.colorbar=o.colorbar;t.marker.cmin=o.cmin;t.marker.cmax=o.cmax;}
  if(o.symbol) t.marker.symbol=o.symbol;
  return t;
}
function markerAt(pt,o){
  const is3=state.dims===3, ink=INK[state.theme];
  const t={type:is3?'scatter3d':'scatter',mode:o.text?'markers+text':'markers',
    x:[pt[0]],y:[pt[1]], text:o.text?[o.text]:undefined, textposition:'top center',
    textfont:{color:ink.text,size:11},
    marker:{size:o.size||(is3?9:15),color:o.color,symbol:o.symbol||'circle',
      line:{width:2,color:ink.paper},opacity:1},
    name:o.name||'', showlegend:o.showlegend!==false, hovertext:[o.hover||o.text||o.name],hoverinfo:'text'};
  if(is3) t.z=[pt[2]];
  return t;
}
function segment(a,b,o){
  const is3=state.dims===3;
  const t={type:is3?'scatter3d':'scatter',mode:'lines',x:[a[0],b[0]],y:[a[1],b[1]],
    line:{width:o.width||5,color:o.color,dash:o.dash||'solid'},hoverinfo:'skip',
    showlegend:!!o.name,name:o.name||''};
  if(is3) t.z=[a[2],b[2]];
  return t;
}
function colouredTraces(idx, field, opt){
  opt=opt||{};
  const traces=[];
  if(P.numeric_fields.includes(field)){
    const vals=idx.map(i=>M[field][i]);
    let mn=Math.min(...vals), mx=Math.max(...vals); if(mn===mx){mx=mn+1;}
    traces.push(mk(idx,{cvals:vals,colorscale:SEQ[state.theme],showscale:true,cmin:mn,cmax:mx,
      size:opt.size,opacity:opt.opacity,
      colorbar:{title:{text:field,side:'right'},thickness:12,len:0.6,x:1.0,
        outlinewidth:0,tickfont:{color:INK[state.theme].muted,size:10}}}));
  } else {
    const dom=DOMAINS[field]||[...new Set(idx.map(i=>M[field][i]))].sort();
    dom.forEach(v=>{
      const sub=idx.filter(i=>M[field][i]===v);
      if(!sub.length) return;
      traces.push(mk(sub,{color:catColor(field,v),name:String(v),showlegend:opt.legend!==false,
        size:opt.size,opacity:opt.opacity}));
    });
  }
  return traces;
}
function baseLayout(){
  const ink=INK[state.theme], is3=state.dims===3;
  const L={paper_bgcolor:ink.paper,plot_bgcolor:ink.paper,
    font:{color:ink.text,family:'system-ui,-apple-system,"Segoe UI",sans-serif',size:12},
    margin:{l:0,r:0,t:6,b:0}, showlegend:true, uirevision:'keep',
    legend:{bgcolor:'rgba(0,0,0,0)',font:{color:ink.text,size:11},itemsizing:'constant',
      orientation:'v',x:0,y:1,xanchor:'left',yanchor:'top'},
    hoverlabel:{bgcolor:ink.paper,bordercolor:ink.axis,font:{color:ink.text,size:12}}};
  if(is3){
    const ax=()=>({showbackground:false,gridcolor:ink.grid,zerolinecolor:ink.axis,
      color:ink.muted,showspikes:false,title:{text:''}});
    L.scene={xaxis:ax(),yaxis:ax(),zaxis:ax(),bgcolor:ink.paper,
      camera:{eye:{x:1.5,y:1.4,z:1.1}}};
  } else {
    const ax=(t)=>({gridcolor:ink.grid,zeroline:false,showline:false,color:ink.muted,
      title:{text:t,font:{size:11,color:ink.muted}}});
    L.xaxis=ax('PC1'); L.yaxis=ax('PC2');
  }
  return L;
}

// ---- the views -----------------------------------------------------------
function render(){
  const set = (state.view===5 && P.has_eeg_only)? state.embSet : 'et';
  CUR = P.coords[set];
  const R = P.reduced[set], CENT=P.centroids[set], CENTP=P.centroids_proj[set];
  const ink=INK[state.theme];
  let traces=[], story='', stat=[];

  if(state.view===1){
    const sel=state.subj1;
    const bg=idxVisible().filter(i=>M.subject[i]!==sel);
    const fg=idxVisible().filter(i=>M.subject[i]===sel);
    if(bg.length) traces.push(mk(bg,{color:ink.faint,opacity:0.45,showlegend:false,size:state.dims===3?2.6:5.5}));
    traces=traces.concat(colouredTraces(fg,state.metric1,{legend:true,size:state.dims===3?4.2:9}));
    story=`Subject <b>${sel}</b> in colour (${fg.length} words), shaded by <b>${state.metric1}</b>; `
        +`everyone else greyed. One reader is a coherent sub-cloud - identity is legible in the space.`;
    stat=[['words',fg.length],['subjects hidden',P.subjects.length-state.visible.size]];
  }

  else if(state.view===2){
    const w=state.word2;
    const base=idxVisible();
    traces.push(mk(base,{color:ink.faint,opacity:0.28,showlegend:false,size:state.dims===3?2.4:5}));
    const hit=idxWord(w);
    traces=traces.concat(colouredTraces(hit,'subject',{legend:true,size:state.dims===3?5.5:12}));
    const st=P.word_stats[w];
    const bl=P.random_baseline;
    story=`The word <b>"${w}"</b> read across brains (${hit.length} occurrences). If a thought code were `
        +`subject-invariant these would coincide - instead they scatter. `
        +`Cross-subject similarity barely beats unrelated thoughts: <b>the core finding.</b>`;
    if(st){
      stat=[['mean cos - across brains',st.mean_cos.toFixed(3)],
            ['random baseline',bl.toFixed(3)],
            ['subjects',st.n_subj]];
      stat._flag = st.mean_cos < (bl+0.15) ? 'warn':'good';
    } else {
      stat=[['note','read by <2 subjects']];
    }
  }

  else if(state.view===3){
    const t=state.wordT, A=state.subjA, B=state.subjB;
    traces.push(mk(idxVisible(),{color:ink.faint,opacity:0.22,showlegend:false,size:state.dims===3?2.2:4.5}));
    const ai=findIdx(t,A), bi=findIdx(t,B);
    if(ai<0 || A===B){
      story=`Pick a word <i>t</i> read by <b>${A}</b>, and a different target brain B, `
          +`or just click a row in the leaderboard below. `
          +(A===B?`Source and target must differ.`:`"${t}" was not found for ${A}.`);
      stat=[['status','-']];
    } else {
      const aP=[CUR.x[ai],CUR.y[ai],CUR.z[ai]];
      const cAp=CENTP[A], cBp=CENTP[B];
      const vP=[aP[0]-cAp[0]+cBp[0], aP[1]-cAp[1]+cBp[1], aP[2]-cAp[2]+cBp[2]];
      const Rv=norm(R[ai].map((x,k)=>x-CENT[A][k]+CENT[B][k]));
      let best=-1,bestSim=-2;
      for(let i=0;i<N;i++){ if(M.subject[i]!==B) continue;
        const s=dot(norm(R[i]),Rv); if(s>bestSim){bestSim=s;best=i;} }
      const hit = best>=0 && M.word[best]===t;
      traces.push(segment(aP,vP,{color:state.theme==='dark'?'#9085e9':'#4a3aa7',width:5}));
      traces.push(markerAt(aP,{color:catColor('subject',A),size:state.dims===3?9:15,
        name:`emb(t, ${A})`,text:'t@'+A,hover:`emb("${t}", ${A})`}));
      traces.push(markerAt(vP,{color:state.theme==='dark'?'#9085e9':'#4a3aa7',symbol:'diamond',
        name:'v = t - A + B',text:'v',hover:'v = emb(t,A) - centroid(A) + centroid(B)'}));
      if(bi>=0){const bP=[CUR.x[bi],CUR.y[bi],CUR.z[bi]];
        traces.push(markerAt(bP,{color:catColor('subject',B),size:state.dims===3?9:15,
          name:`true emb(t, ${B})`,text:'t@'+B,hover:`emb("${t}", ${B})`}));}
      if(best>=0){const nP=[CUR.x[best],CUR.y[best],CUR.z[best]];
        traces.push(markerAt(nP,{color:hit?'#0ca30c':'#e34948',symbol:'x',
          name:'nearest neighbour of v',text:M.word[best],
          hover:`NN of v: "${M.word[best]}" [${M.subject[best]}]`}));}
      story=`<b>"${t}"</b> re-aimed from <b>${A}</b> to <b>${B}</b>. The arrow adds the centroid offset; `
          +`v's nearest neighbour under B is <b>"${best>=0?M.word[best]:'-'}"</b>. `
          +(hit?`It lands on the same word - the offset cancelled identity.`
                :`It misses the target word - identity is not a clean translation.`);
      stat=[['cos(v, NN)',bestSim.toFixed(3)],['analogy',hit?'HIT':'miss']];
      stat._pill = hit?'hit':'miss';
    }
  }

  else if(state.view===4){
    const qi=neighbourQuery(), k=Math.max(3,Math.min(50,state.kN|0||12));
    traces.push(mk(idxVisible(),{color:ink.faint,opacity:0.2,showlegend:false,size:state.dims===3?2.2:4.5}));
    let nn=[];
    if(qi>=0){
      nn=neighbours(qi,k);
      const qw=M.word[qi], qc=M.category[qi];
      const same=[], cat=[], other=[];
      nn.forEach(([i])=>{ if(M.word[i]===qw)same.push(i); else if(HAS_CAT&&M.category[i]===qc)cat.push(i); else other.push(i); });
      if(other.length) traces.push(mk(other,{color:ink.muted,opacity:0.9,
        name:'other',size:state.dims===3?4.5:9}));
      if(cat.length) traces.push(mk(cat,{color:PAL[state.theme][0],name:'same category',
        size:state.dims===3?5.5:11}));
      if(same.length) traces.push(mk(same,{color:PAL[state.theme][1],name:'same word',
        size:state.dims===3?6:12}));
      const qP=[CUR.x[qi],CUR.y[qi],CUR.z[qi]];
      traces.push(markerAt(qP,{color:PAL[state.theme][2],symbol:'star',size:state.dims===3?11:18,
        name:'query',text:qw,hover:`query: "${qw}" [${M.subject[qi]}]`}));
      story=`The <b>${k}</b> nearest thoughts to <b>"${qw}"</b>`
          +(state.subjN==='any'?'':` (read by ${state.subjN})`)
          +`. Green = the same word elsewhere, blue = same category. Coherence far above chance would `
          +`mean similar thoughts really do sit together.`;
      stat=[['neighbours',k]];
    } else {
      story=`Type a word read in the dataset to see its nearest neighbours.`;
      stat=[['status','not found']];
    }
    renderNeighTable(qi,nn);
  }

  else { // view 5
    const field=state.colorBy;
    traces=colouredTraces(idxVisible(),field,{legend:true});
    const label = set==='et'?'EEG + eye-tracking':'EEG-only';
    story=`Whole space recomputed from <b>${label}</b> signals`
        +(P.has_eeg_only?`. Toggle the set - gaze behaviour reshapes the reading-evoked geometry, `
          +`but the imagined-thought (EEG-only) space must stand on neural signal alone.`
          :`. No EEG-only set was supplied, so the toggle is disabled.`);
    stat=[['signal set',label],['coloured by',field]];
  }

  Plotly.react('plot', traces, baseLayout(), {responsive:true,displaylogo:false,
    modeBarButtonsToRemove:['select2d','lasso2d']});
  paintStory(story,stat);
  updateBottom();
}

function updateBottom(){
  const bottom=document.getElementById('bottom');
  const lp=document.getElementById('leaderpanel'), np=document.getElementById('neighpanel'),
    bw=document.getElementById('barwrap');
  lp.classList.add('hide'); np.classList.add('hide'); bw.classList.add('hide');
  let show=false;
  if(state.view===3){ lp.classList.remove('hide'); renderLeaderboard(); show=true; }
  else if(state.view===4){ np.classList.remove('hide'); show=true; }
  else if(state.view===5 && P.probe){ bw.classList.remove('hide'); drawBar(); show=true; }
  bottom.classList.toggle('hide', !show);
}

function paintStory(story,stat){
  document.getElementById('story').innerHTML=story;
  const box=document.getElementById('statbox'); box.innerHTML='';
  (stat||[]).forEach(([k,v])=>{
    const flag = stat._flag && k.indexOf('across')>=0 ? ' '+stat._flag : '';
    let vhtml = String(v);
    if(stat._pill && k==='analogy') vhtml=`<span class="pill ${stat._pill}">${v}</span>`;
    box.insertAdjacentHTML('beforeend',
      `<div class="cell"><div class="k">${k}</div><div class="v${flag}">${vhtml}</div></div>`);
  });
}

let barMade=false;
function drawBar(){
  if(!P.probe) return;
  const reps=Object.keys(P.probe);
  const vals=reps.map(r=>{const v=P.probe[r];return typeof v==='number'?v:(v.word_len!=null?v.word_len:Object.values(v)[0]);});
  const ink=INK[state.theme];
  const colors=reps.map(r=> /eeg-only/i.test(r)?PAL[state.theme][2]:PAL[state.theme][0]);
  Plotly.react('bar',[{type:'bar',x:vals,y:reps,orientation:'h',
    marker:{color:colors,line:{width:0}},
    text:vals.map(v=>Number(v).toFixed(3)),textposition:'auto',
    textfont:{color:'#fff',size:12},hoverinfo:'x+y'}],
    {paper_bgcolor:ink.paper,plot_bgcolor:ink.paper,margin:{l:130,r:16,t:4,b:22},
     font:{color:ink.text,size:11},uirevision:'bar',
     xaxis:{range:[0,Math.max(1,Math.max.apply(null,vals)*1.15)],gridcolor:ink.grid,zeroline:false,
       color:ink.muted},yaxis:{color:ink.text,automargin:true}},
    {displayModeBar:false,responsive:true});
  barMade=true;
}

// ---- controls ------------------------------------------------------------
function buildControls(){
  const tabsMeta=[['1','one subject'],['2','one word, many brains'],
    ['3','thought arithmetic'],['4','nearest thoughts'],['5','eye-tracking']];
  const tabs=document.getElementById('tabs');
  tabsMeta.forEach(([n,d])=>{
    const b=document.createElement('button'); b.className='tab'+(n==='1'?' on':'');
    b.dataset.v=n; b.innerHTML=`<b>${n} - ${d}</b>`;
    b.onclick=()=>setView(+n); tabs.appendChild(b);
  });

  const cb=document.getElementById('colorby');
  P.fields.concat(P.numeric_fields).forEach(f=>{
    const o=document.createElement('option'); o.value=f;
    o.textContent=f+(P.numeric_fields.includes(f)?' (scale)':''); cb.appendChild(o);});
  cb.value=state.colorBy; cb.onchange=e=>{state.colorBy=e.target.value;render();};

  document.querySelectorAll('#dims button').forEach(b=>{
    if(+b.dataset.d===state.dims) b.classList.add('on');
    b.onclick=()=>{state.dims=+b.dataset.d;
      document.querySelectorAll('#dims button').forEach(x=>x.classList.toggle('on',x===b));render();};
  });

  const sc=document.getElementById('subjchecks');
  P.subjects.forEach((s,i)=>{
    const l=document.createElement('label'); l.className='chk';
    l.innerHTML=`<input type="checkbox" checked><span class="sw" style="background:${PAL[state.theme][i%8]}"></span>${s}`;
    l.querySelector('input').onchange=e=>{e.target.checked?state.visible.add(s):state.visible.delete(s);render();};
    sc.appendChild(l);
  });

  fillSel('subj1',P.subjects,state.subj1,v=>{state.subj1=v;});
  fillSel('metric1',P.numeric_fields,state.metric1,v=>{state.metric1=v;});
  fillSel('subjA',P.subjects,state.subjA,v=>{state.subjA=v;});
  fillSel('subjB',P.subjects,state.subjB,v=>{state.subjB=v;});
  fillSel('subjN',['any'].concat(P.subjects),state.subjN,v=>{state.subjN=v;});

  const dl=document.getElementById('wordlist');
  P.words.forEach(w=>{const o=document.createElement('option');o.value=w;dl.appendChild(o);});
  const w2=document.getElementById('word2'); w2.value=state.word2;
  w2.onchange=e=>{if(e.target.value)state.word2=e.target.value;render();};
  const wt=document.getElementById('wordT'); wt.value=state.wordT;
  wt.onchange=e=>{if(e.target.value)state.wordT=e.target.value;render();};
  const wn=document.getElementById('wordN'); wn.value=state.wordN;
  wn.onchange=e=>{if(e.target.value)state.wordN=e.target.value;render();};
  const kn=document.getElementById('kN'); kn.value=state.kN;
  kn.onchange=e=>{state.kN=+e.target.value||12;render();};

  document.querySelectorAll('#embset button').forEach(b=>{
    if(b.dataset.s===state.embSet) b.classList.add('on');
    b.disabled=!P.has_eeg_only;
    b.onclick=()=>{if(!P.has_eeg_only)return;state.embSet=b.dataset.s;
      document.querySelectorAll('#embset button').forEach(x=>x.classList.toggle('on',x===b));render();};
  });
  document.getElementById('v5note').innerHTML = P.has_eeg_only
    ? 'Both spaces are trained independently and PCA-projected. The bar below shows a word-length linear probe for each.'
    : 'Supply <code>eeg_only_emb</code> (and <code>probe_scores</code>) to enable the toggle and the probe bar.';

  document.getElementById('surprise').onclick=()=>{
    if(!LB.length) return;
    const hits=LB.filter(x=>x.hit);
    const pool=hits.length?hits:LB;
    openAnalogy(pool[(Math.random()*pool.length)|0]);
  };
  document.getElementById('guidebtn').onclick=()=>{
    state.guide=!state.guide;
    document.getElementById('guidebtn').textContent=state.guide?'Hide guide':'What am I looking at?';
    renderGuide();
    setTimeout(()=>Plotly.Plots.resize('plot'),0);
  };
  document.getElementById('themebtn').onclick=()=>{
    state.theme=state.theme==='dark'?'light':'dark'; applyTheme(); refreshSwatches();
    renderGuide(); renderBanners(); render();};
}
function fillSel(id,opts,val,cb){
  const s=document.getElementById(id); s.innerHTML='';
  opts.forEach(o=>{const e=document.createElement('option');e.value=o;e.textContent=o;s.appendChild(e);});
  s.value=val; s.onchange=e=>{cb(e.target.value);render();};
}
function refreshSwatches(){
  document.querySelectorAll('#subjchecks .chk').forEach((l,i)=>{
    l.querySelector('.sw').style.background=PAL[state.theme][i%8];});
}
function setView(v){
  state.view=v;
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',+t.dataset.v===v));
  document.querySelectorAll('.v-ctrl').forEach(g=>g.classList.add('hide'));
  document.querySelector('.v-ctrl.v'+v).classList.remove('hide');
  render();
  setTimeout(()=>Plotly.Plots.resize('plot'),0);
}
function applyTheme(){document.documentElement.setAttribute('data-theme',state.theme);}

applyTheme();
computeLeaderboard();
buildControls();
renderGuide();
renderBanners();
render();
window.addEventListener('resize',()=>{Plotly.Plots.resize('plot'); if(barMade)Plotly.Plots.resize('bar');});
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Neuron Atlas -- an interactive tour of which embedding dimensions fire and
# what each one represents. It consumes a precomputed ``neuron_report`` dict and
# visualises it; it never recomputes a metric. One offline HTML, Plotly inlined.
# --------------------------------------------------------------------------- #


def _json_safe(obj: Any) -> Any:
    """Converts a report dict to strict-JSON / valid-JS-literal types.

    numpy scalars and arrays become builtins, and non-finite floats
    (``inf``/``nan`` -- e.g. an infinite ``who_vs_what_ratio``) become ``None``
    so the injected payload parses cleanly in the browser.

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


def neuron_atlas_html(
    neurons: dict,
    out_path: str | Path,
    *,
    title: str = 'ZTE Neuron Atlas',
    max_bars: int | None = None,
) -> Path:
    """Writes a self-contained interactive "Neuron Atlas" from a `neuron_report` dict.

    The atlas visualises -- without recomputing anything -- which embedding
    dimensions fire and what each represents, through four wired views in one
    offline HTML: (1) a summary header with headline tiles and a dominant-attribute
    colour legend, (2) a ranked-importance bar of *every* dimension coloured by its
    dominant attribute with the ``active_threshold`` drawn so the negligible tail is
    visible, (3) a per-neuron detail panel (selectivity bar, activation histogram,
    top/bottom activating words and scalp/band attribution) that updates on click,
    and (4) live controls (sort by importance or by selectivity for a target,
    recolour, hide negligible neurons, neuron search, light/dark toggle).

    Args:
        neurons (dict): The report produced by :func:`zte.evaluation.neurons.neuron_report`
            (its ``meta``/``importance``/``selectivity``/``summary``/``top_neurons`` blocks).
            Robust to missing keys (no ``attribution``, empty ``top_neurons``, ``targets=[]``).
        out_path (str | Path): Output path (``.html``, or ``.png`` on the Plotly fallback).
        title (str): Page and figure title.
        max_bars (int | None): Optional cap on how many neurons the ranked chart draws
            (most-important first); ``None`` draws all ``D``. Detail/search still cover all.

    Returns:
        Path: The written path (``.html`` when Plotly is available, else a static ``.png``).
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

    imp = neurons.get('importance', {}) or {}
    std = imp.get('std', []) or []
    order = imp.get('order') or list(range(len(std)))
    var_share = imp.get('var_share', []) or []
    dominant = (neurons.get('selectivity', {}) or {}).get('dominant', []) or []
    thr = float(imp.get('active_threshold', 0.0) or 0.0)
    tot = float(sum(s * s for s in std)) or 1.0
    thr_vs = (thr * thr) / tot

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


# --------------------------------------------------------------------------- #
# The Neuron Atlas single-file template. `/*__ATLAS_PLOTLY_JS__*/`,
# `"__ATLAS_PAYLOAD__"` and `__ATLAS_TITLE__` are substituted at write time so the
# page is fully offline with no external hosts.
# --------------------------------------------------------------------------- #

_ATLAS_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__ATLAS_TITLE__</title>
<script>/*__ATLAS_PLOTLY_JS__*/</script>
<style>
:root{
  --surface:#fcfcfb; --plane:#f4f4f1; --panel:#ffffff;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --accent:#2a78d6; --who:#eda100; --what:#2a78d6;
}
:root[data-theme="dark"]{
  --surface:#1a1a19; --plane:#0d0d0d; --panel:#202020;
  --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.12);
  --accent:#3987e5; --who:#c98500; --what:#3987e5;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{
  background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; font-size:14px;
  display:grid; grid-template-columns:300px 1fr; grid-template-rows:100vh;
}
.rail{background:var(--panel);border-right:1px solid var(--border);
  padding:18px 18px 28px;overflow-y:auto;height:100vh}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:4px}
.brand h1{font-size:15.5px;margin:0;font-weight:650;letter-spacing:.2px}
.brand .dot{width:11px;height:11px;border-radius:50%;
  background:conic-gradient(from 210deg,#eda100,#2a78d6,#1baf7a,#4a3aa7,#008300,#eda100)}
.sub{color:var(--ink2);font-size:12px;line-height:1.5;margin:6px 0 16px}
.group{border-top:1px solid var(--border);padding:14px 0}
.group:first-of-type{border-top:none}
.group h3{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);
  margin:0 0 10px;font-weight:650}
.row{margin-bottom:11px}
label.lab{display:block;font-size:12px;color:var(--ink2);margin-bottom:5px}
select,input[type=text],input[type=number]{
  width:100%;padding:7px 9px;border-radius:8px;border:1px solid var(--border);
  background:var(--surface);color:var(--ink);font-size:13px;font-family:inherit}
.chk{display:inline-flex;align-items:center;gap:8px;font-size:12.5px;color:var(--ink2);cursor:pointer;user-select:none}
.chk input{accent-color:var(--accent)}
.searchrow{display:flex;gap:6px}
.searchrow input{flex:1}
.btn{border:1px solid var(--border);background:transparent;color:var(--ink2);
  border-radius:8px;padding:7px 12px;font-size:12.5px;cursor:pointer;font-family:inherit}
.btn:hover{border-color:var(--axis)}
.note{font-size:11.5px;color:var(--muted);line-height:1.5;margin-top:6px}
.hide{display:none!important}
.main{height:100vh;overflow-y:auto;padding:16px 20px 40px;min-width:0}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px}
.tile{background:var(--panel);border:1px solid var(--border);border-radius:12px;
  padding:12px 16px;min-width:150px;flex:1}
.tile .v{font-size:23px;font-weight:660;font-variant-numeric:tabular-nums;line-height:1.1}
.tile .v.who{color:var(--who)} .tile .v.what{color:var(--what)}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-top:4px}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin:2px 2px 14px;font-size:12px;color:var(--ink2)}
.lg{display:inline-flex;align-items:center;gap:6px}
.lg .sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;
  padding:12px 14px 8px;margin-bottom:14px}
.card h2{font-size:13px;font-weight:640;margin:0 0 2px}
.card .cap{font-size:11.5px;color:var(--muted);margin:0 0 8px;line-height:1.45}
#ranked{width:100%;height:330px}
.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.detail-grid .panel h3{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);
  margin:0 0 6px;font-weight:650}
#selbar{width:100%;height:150px}
#acthist{width:100%;height:170px}
#attrbar{width:100%;height:180px}
.words{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:6px}
table.wt{width:100%;border-collapse:collapse;font-size:12px}
table.wt th{text-align:left;color:var(--muted);font-weight:600;font-size:10.5px;
  text-transform:uppercase;letter-spacing:.4px;border-bottom:1px solid var(--border);padding:3px 6px}
table.wt td{padding:3px 6px;border-bottom:1px solid var(--border);color:var(--ink2)}
table.wt td.num{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}
.empty{font-size:12px;color:var(--muted);padding:16px 4px}
#detail-title{font-size:13.5px;margin:0 0 8px;line-height:1.4}
#detail-title b{color:var(--ink)}
#detail-note{font-size:11.5px;color:var(--muted);line-height:1.5;margin-top:8px}
@media (max-width:900px){
  body{grid-template-columns:1fr;grid-template-rows:auto 1fr}
  .rail{height:auto;max-height:44vh}
  .detail-grid,.words{grid-template-columns:1fr}
}
</style>
</head>
<body>
<aside class="rail">
  <div class="brand"><span class="dot"></span><h1>Neuron Atlas</h1></div>
  <div class="sub">Every dimension of the ZTE embedding is a neuron with a story: some
    <b>fire</b> (carry variance), most are <b>negligible</b>, and each is more or less
    <b>selective</b> for an attribute. Click a bar to inspect a neuron.</div>

  <div class="group">
    <h3>Ranked chart</h3>
    <div class="row"><label class="lab">Sort neurons by</label><select id="sortby"></select></div>
    <div class="row"><label class="lab">Colour bars by</label><select id="colorby"></select></div>
    <div class="row"><label class="lab">Bar height</label><select id="metric">
      <option value="var_share">variance share</option>
      <option value="std">std (spread)</option>
    </select></div>
    <div class="row"><label class="chk"><input type="checkbox" id="hidedead"> Hide negligible (inactive) neurons</label></div>
  </div>

  <div class="group">
    <h3>Inspect a neuron</h3>
    <div class="row"><label class="lab">Jump to neuron #</label>
      <div class="searchrow"><input type="number" id="search" min="0" placeholder="dim index">
        <button class="btn" id="searchbtn">Go</button></div></div>
    <div class="note">Detailed exemplars, histogram and scalp attribution exist for the
      top neurons; importance &amp; selectivity are shown for every dimension.</div>
  </div>

  <div class="group">
    <button class="btn" id="themebtn">Toggle light / dark</button>
  </div>
</aside>

<main class="main">
  <div class="tiles" id="tiles"></div>
  <div class="legend" id="legend"></div>

  <div class="card">
    <h2>Ranked importance &mdash; every neuron</h2>
    <div class="cap" id="rankedcap">Each bar is one dimension, ordered most-important first and
      coloured by its dominant attribute. Bars below the dashed line are the negligible tail.
      Hover for detail; click to inspect.</div>
    <div id="ranked"></div>
  </div>

  <div class="card">
    <h2 id="detail-title"></h2>
    <div class="detail-grid">
      <div class="panel">
        <h3>Selectivity per target</h3>
        <div id="selbar"></div>
      </div>
      <div class="panel" id="attrwrap">
        <h3>Scalp / band attribution (correlational)</h3>
        <div id="attrbar"></div>
      </div>
    </div>
    <div id="exemplars">
      <div class="panel" style="margin-top:6px">
        <h3>Activation histogram</h3>
        <div id="acthist"></div>
      </div>
      <div class="words">
        <div class="panel"><h3>Top activating words</h3><table class="wt" id="topwords"></table></div>
        <div class="panel"><h3>Bottom activating words</h3><table class="wt" id="botwords"></table></div>
      </div>
    </div>
    <div id="detail-note"></div>
  </div>
</main>

<script>
const RAW = "__ATLAS_PAYLOAD__";
const N = RAW.data || {};
const MAXB = RAW.max_bars || null;
const FONT = 'system-ui,-apple-system,"Segoe UI",sans-serif';

const INK = {
  light:{paper:'#fcfcfb',grid:'#e1e0d9',axis:'#c3c2b7',text:'#0b0b0b',muted:'#898781'},
  dark: {paper:'#1a1a19',grid:'#2c2c2a',axis:'#383835',text:'#ffffff',muted:'#898781'},
};
const SEQ = {
  light:[[0,'#cde2fb'],[0.5,'#3987e5'],[1,'#0d366b']],
  dark: [[0,'#173a63'],[0.5,'#3987e5'],[1,'#cde2fb']],
};
const ATTR_META = {
  subject: {c:{light:'#eda100',dark:'#c98500'}, label:'subject · who'},
  word_len:{c:{light:'#2a78d6',dark:'#3987e5'}, label:'word_len · what'},
  log_freq:{c:{light:'#1baf7a',dark:'#199e70'}, label:'log_freq · what'},
  category:{c:{light:'#4a3aa7',dark:'#9085e9'}, label:'category · what'},
  task:    {c:{light:'#008300',dark:'#008300'}, label:'task'},
  none:    {c:{light:'#b8b6ad',dark:'#5c5b57'}, label:'none · negligible'},
};
const ATTR_ORDER = ['subject','word_len','log_freq','category','task','none'];
const FALLBACK = {light:['#e34948','#e87ba4','#eb6834'],dark:['#e66767','#d55181','#d95926']};
const _fbi = {};

// ---- normalise the report (robust to missing keys) -----------------------
const imp = N.importance || {};
imp.std = imp.std || [];
imp.var_share = imp.var_share || imp.std.map(()=>0);
imp.active = imp.active || imp.std.map(()=>true);
imp.rank = imp.rank || imp.std.map((_,i)=>i);
imp.order = imp.order || imp.std.map((_,i)=>i).sort((a,b)=>(imp.std[b]||0)-(imp.std[a]||0));
if(imp.active_threshold==null) imp.active_threshold = 0;
const sel = N.selectivity || {targets:[],scores:{},dominant:[],dominant_score:[]};
const targets = sel.targets || [];
const dom = sel.dominant || imp.std.map(()=>'none');
const domScore = sel.dominant_score || imp.std.map(()=>0);
const topByDim = {};
(N.top_neurons || []).forEach(t=>{ topByDim[t.dim]=t; });
const D = (N.meta && N.meta.embed_dim) || imp.std.length;

const state = {
  sort:'importance', color:'dominant', metric:'var_share', hideDead:false,
  sel: (imp.order.length? imp.order[0] : 0),
  theme:(window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches)?'dark':'light',
};

// ---- helpers -------------------------------------------------------------
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function pct(x){return x==null?'–':(x*100).toFixed(1)+'%';}
function attrColor(a){
  if(ATTR_META[a]) return ATTR_META[a].c[state.theme];
  if(!(a in _fbi)) _fbi[a]=Object.keys(_fbi).length;
  const arr=FALLBACK[state.theme]; return arr[_fbi[a]%arr.length];
}
function attrLabel(a){return ATTR_META[a]?ATTR_META[a].label:a;}
function value(d){return state.metric==='std'?imp.std[d]:imp.var_share[d];}
function threshold(){
  if(state.metric==='std') return imp.active_threshold;
  const tot = imp.std.reduce((a,s)=>a+s*s,0)||1;
  return imp.active_threshold*imp.active_threshold/tot;
}
function orderedDims(){
  let dims;
  if(state.sort==='importance'){ dims = imp.order.slice(); }
  else { const sc = sel.scores[state.sort]||[];
    dims = Array.from({length:D},(_,i)=>i).sort((a,b)=>(sc[b]||0)-(sc[a]||0)); }
  if(state.hideDead) dims = dims.filter(d=>imp.active[d]);
  if(MAXB && dims.length>MAXB) dims = dims.slice(0,MAXB);
  return dims;
}

// ---- view 1: summary tiles + legend --------------------------------------
function renderTiles(){
  const S = N.summary || {};
  const r = S.who_vs_what_ratio;
  const ratio = (r==null)?'∞':(+r).toFixed(2)+'×';
  const tiles = [
    {k:'neurons active', v:`${S.n_active==null?'–':S.n_active} / ${D}`, cls:''},
    {k:'who variance · subject', v:pct(S.who_variance), cls:'who'},
    {k:'what variance · content', v:pct(S.what_variance), cls:'what'},
    {k:'who / what ratio', v:ratio, cls:(r!=null&&r>1)?'who':'what'},
    {k:'variance in active', v:pct(S.active_variance_share), cls:''},
  ];
  document.getElementById('tiles').innerHTML = tiles.map(o=>
    `<div class="tile"><div class="v ${o.cls}">${o.v}</div><div class="k">${o.k}</div></div>`).join('');
}
function renderLegend(){
  const present = new Set();
  (dom||[]).forEach(x=>present.add(x)); targets.forEach(t=>present.add(t));
  const ordered = ATTR_ORDER.filter(a=>present.has(a))
    .concat([...present].filter(a=>!ATTR_ORDER.includes(a)));
  document.getElementById('legend').innerHTML = ordered.map(a=>
    `<span class="lg"><span class="sw" style="background:${attrColor(a)}"></span>${attrLabel(a)}</span>`
  ).join('') || '<span class="lg">no attributes probed</span>';
}

// ---- view 2: ranked importance of every neuron ---------------------------
function renderRanked(){
  const dims = orderedDims(), ink = INK[state.theme], n = dims.length;
  const cd = d=>[d, imp.std[d], imp.var_share[d], dom[d]||'none', domScore[d]||0];
  const HT = 'neuron #%{customdata[0]}<br>std %{customdata[1]:.3f} · var %{customdata[2]:.3f}'
           + '<br>dominant %{customdata[3]} (%{customdata[4]:.2f})<extra></extra>';
  let traces = [];
  if(state.color==='dominant'){
    const groups = {};
    dims.forEach((d,pos)=>{ const a=dom[d]||'none'; (groups[a]=groups[a]||[]).push([pos,d]); });
    const keys = ATTR_ORDER.filter(a=>groups[a]).concat(Object.keys(groups).filter(a=>!ATTR_ORDER.includes(a)));
    keys.forEach(a=>{ const g=groups[a];
      traces.push({type:'bar', name:attrLabel(a),
        x:g.map(o=>o[0]), y:g.map(o=>value(o[1])), customdata:g.map(o=>cd(o[1])),
        marker:{color:attrColor(a), line:{width:0}}, hovertemplate:HT});
    });
  } else {
    const sc = sel.scores[state.color]||[];
    const HTS = 'neuron #%{customdata[0]}<br>std %{customdata[1]:.3f} · var %{customdata[2]:.3f}'
              + '<br>sel(' + state.color + ') %{customdata[5]:.2f}<extra></extra>';
    traces.push({type:'bar', showlegend:false,
      x:dims.map((d,i)=>i), y:dims.map(value),
      customdata:dims.map(d=>cd(d).concat([sc[d]||0])),
      marker:{color:dims.map(d=>sc[d]||0), colorscale:SEQ[state.theme], cmin:0, cmax:1, showscale:true,
        colorbar:{title:{text:'|sel| '+state.color, side:'right'}, thickness:12, len:0.7,
          outlinewidth:0, tickfont:{color:ink.muted, size:10}}},
      hovertemplate:HTS});
  }
  const thr = threshold();
  Plotly.react('ranked', traces, {
    barmode:'overlay', bargap:0,
    paper_bgcolor:ink.paper, plot_bgcolor:ink.paper,
    font:{color:ink.text, size:12, family:FONT},
    margin:{l:58, r:16, t:8, b:36}, uirevision:'ranked',
    legend:{orientation:'h', y:1.1, x:0, font:{color:ink.text, size:11}, bgcolor:'rgba(0,0,0,0)'},
    hoverlabel:{bgcolor:ink.paper, bordercolor:ink.axis, font:{color:ink.text, size:12}},
    xaxis:{title:{text:(state.sort==='importance'?'importance rank (0 = most important) →'
        :'rank by |sel| '+state.sort+' →'), font:{size:11, color:ink.muted}},
      gridcolor:ink.grid, zeroline:false, color:ink.muted, range:[-0.5, Math.max(0.5,n-0.5)]},
    yaxis:{title:{text:(state.metric==='std'?'std (spread)':'variance share'),
        font:{size:11, color:ink.muted}},
      gridcolor:ink.grid, zeroline:false, color:ink.muted, rangemode:'tozero'},
    shapes:[{type:'line', x0:-0.5, x1:Math.max(0.5,n-0.5), y0:thr, y1:thr,
      line:{color:state.theme==='dark'?'#e66767':'#e34948', width:1.5, dash:'dash'}}],
    annotations:[{x:Math.max(0.5,n-0.5), y:thr, xanchor:'right', yanchor:'bottom',
      text:'active threshold — tail below is negligible', showarrow:false,
      font:{color:state.theme==='dark'?'#e66767':'#e34948', size:10.5}}],
  }, {responsive:true, displaylogo:false, modeBarButtonsToRemove:['select2d','lasso2d']});
}

// ---- view 3: per-neuron detail panel -------------------------------------
function selectNeuron(dim){
  if(dim==null || isNaN(dim) || dim<0 || dim>=D) return;
  state.sel = dim; renderDetail();
}
function fillWords(id, rows){
  const t = document.getElementById(id);
  t.innerHTML = '<tr><th>word</th><th>subject</th><th>act</th></tr>' + (rows||[]).map(r=>
    `<tr><td>${esc(r.word)}</td><td>${esc(r.subject)}</td>`
    + `<td class="num">${(+r.activation).toFixed(2)}</td></tr>`).join('');
}
function renderDetail(){
  const dim = state.sel, ink = INK[state.theme];
  const a = dom[dim]||'none', sc = domScore[dim]||0;
  document.getElementById('detail-title').innerHTML =
    `Neuron <b>#${dim}</b> · rank ${imp.rank[dim]} · dominant: `
    + `<span style="color:${attrColor(a)};font-weight:640">${attrLabel(a)}</span> (${sc.toFixed(2)}) `
    + `· std ${(+imp.std[dim]).toFixed(3)} · var ${((imp.var_share[dim]||0)*100).toFixed(2)}%`;

  if(targets.length){
    const vals = targets.map(t=>(sel.scores[t]||[])[dim]||0);
    Plotly.react('selbar', [{type:'bar', orientation:'h',
      x:vals, y:targets, marker:{color:targets.map(t=>attrColor(t)), line:{width:0}},
      text:vals.map(v=>v.toFixed(2)), textposition:'auto', textfont:{color:'#fff', size:11},
      hovertemplate:'%{y}: %{x:.3f}<extra></extra>'}],
      {paper_bgcolor:ink.paper, plot_bgcolor:ink.paper, font:{color:ink.text, size:11, family:FONT},
       margin:{l:74, r:12, t:6, b:24}, uirevision:'sel',
       xaxis:{range:[0,1], gridcolor:ink.grid, zeroline:false, color:ink.muted},
       yaxis:{color:ink.text, automargin:true}},
      {displayModeBar:false, responsive:true});
  } else {
    Plotly.purge('selbar');
    document.getElementById('selbar').innerHTML='<div class="empty">No probe targets were available.</div>';
  }

  const entry = topByDim[dim];
  const exWrap = document.getElementById('exemplars');
  const attrWrap = document.getElementById('attrwrap');
  if(entry){
    document.getElementById('detail-note').textContent='';
    exWrap.classList.remove('hide');
    const h = entry.activation_hist||{}, edges=h.edges||[], counts=h.counts||[];
    const centers=[], width=[];
    for(let i=0;i<counts.length;i++){ centers.push((edges[i]+edges[i+1])/2); width.push(edges[i+1]-edges[i]); }
    Plotly.react('acthist', [{type:'bar', x:centers, y:counts, width:width,
      marker:{color:attrColor(a), line:{width:0}},
      hovertemplate:'act %{x:.3f}: %{y} words<extra></extra>'}],
      {paper_bgcolor:ink.paper, plot_bgcolor:ink.paper, font:{color:ink.text, size:11, family:FONT},
       margin:{l:46, r:10, t:6, b:30}, uirevision:'hist',
       xaxis:{title:{text:'activation', font:{size:10, color:ink.muted}}, gridcolor:ink.grid, zeroline:false, color:ink.muted},
       yaxis:{title:{text:'words', font:{size:10, color:ink.muted}}, gridcolor:ink.grid, zeroline:false, color:ink.muted}},
      {displayModeBar:false, responsive:true});
    fillWords('topwords', entry.top_words);
    fillWords('botwords', entry.bottom_words);
    const at = entry.attribution;
    if(at && at.length){
      attrWrap.classList.remove('hide');
      const rev = at.slice().reverse();
      Plotly.react('attrbar', [{type:'bar', orientation:'h',
        x:rev.map(o=>o.corr), y:rev.map(o=>o.feature),
        marker:{color:state.theme==='dark'?'#199e70':'#1baf7a', line:{width:0}},
        text:rev.map(o=>o.corr.toFixed(2)), textposition:'auto', textfont:{color:'#fff', size:10},
        hovertemplate:'%{y}: |r|=%{x:.3f}<extra></extra>'}],
        {paper_bgcolor:ink.paper, plot_bgcolor:ink.paper, font:{color:ink.text, size:10, family:FONT},
         margin:{l:132, r:12, t:6, b:22}, uirevision:'attr',
         xaxis:{range:[0,1], gridcolor:ink.grid, zeroline:false, color:ink.muted},
         yaxis:{color:ink.text, automargin:true}},
        {displayModeBar:false, responsive:true});
    } else { attrWrap.classList.add('hide'); }
  } else {
    exWrap.classList.add('hide');
    attrWrap.classList.add('hide');
    document.getElementById('detail-note').textContent =
      'Exemplars, the activation histogram and scalp/band attribution are computed for the '
      + 'top neurons only. This neuron’s importance and selectivity (above) are available for '
      + 'every dimension.';
  }
}

// ---- view 4: controls ----------------------------------------------------
function fill(id, opts, val){
  const s = document.getElementById(id); s.innerHTML='';
  opts.forEach(([v,label])=>{ const o=document.createElement('option');
    o.value=v; o.textContent=label; s.appendChild(o); });
  s.value = val;
}
function buildControls(){
  fill('sortby', [['importance','importance']].concat(targets.map(t=>[t,'selectivity: '+t])), state.sort);
  document.getElementById('sortby').onchange = e=>{ state.sort=e.target.value; renderRanked(); };
  fill('colorby', [['dominant','dominant attribute']].concat(targets.map(t=>[t,'selectivity: '+t])), state.color);
  document.getElementById('colorby').onchange = e=>{ state.color=e.target.value; renderRanked(); };
  document.getElementById('metric').value = state.metric;
  document.getElementById('metric').onchange = e=>{ state.metric=e.target.value; renderRanked(); };
  document.getElementById('hidedead').onchange = e=>{ state.hideDead=e.target.checked; renderRanked(); };
  const go = ()=>{ selectNeuron(parseInt(document.getElementById('search').value,10)); };
  document.getElementById('searchbtn').onclick = go;
  document.getElementById('search').onkeydown = e=>{ if(e.key==='Enter') go(); };
  document.getElementById('themebtn').onclick = ()=>{
    state.theme = state.theme==='dark'?'light':'dark';
    applyTheme(); renderTiles(); renderLegend(); renderRanked(); renderDetail();
  };
}
function applyTheme(){ document.documentElement.setAttribute('data-theme', state.theme); }

// ---- init ----------------------------------------------------------------
applyTheme();
buildControls();
renderTiles();
renderLegend();
if(D>0){
  renderRanked();
  document.getElementById('ranked').on('plotly_click', ev=>{
    const p = ev.points && ev.points[0];
    if(p && p.customdata) selectNeuron(p.customdata[0]);
  });
  selectNeuron(state.sel);
} else {
  document.getElementById('ranked').innerHTML='<div class="empty">Empty report — no neurons to display.</div>';
}
window.addEventListener('resize', ()=>{
  ['ranked','selbar','acthist','attrbar'].forEach(id=>{
    const el=document.getElementById(id); if(el && el.data) Plotly.Plots.resize(id);
  });
});
</script>
</body>
</html>
"""
