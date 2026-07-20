"""Builds the Thought-Space Explorer page, driven in-browser from a reduced-vector payload."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zte.evaluation.interactive._assets import load_page
from zte.evaluation.interactive._common import (
    _escape,
    _json_safe,
    _pca,
    _pca_basis,
    _project,
    _static_fallback,
)
from zte.logging_utils import get_logger

_LOG = get_logger('evaluation.interactive')
_EXPLORER_TEMPLATE: str = load_page('explorer')

_REDUCED_DIMS: int = 64  # PCA width kept for in-browser cosine / analogy arithmetic.


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

    This is the core "does this word mean the same thing in different brains?" statistic.

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

    A candidate is a word `t` read by two subjects `A` and `B`; the browser scores `emb(t, A) - centroid(A) +
    centroid(B)` a hit when its nearest neighbour is `t` as read by `B`. Pre-listing every viable analogy is what
    lets the leaderboard rank them instead of the user guessing which word and person to try.

    Args:
        words (np.ndarray): Surface word per row `(n,)` (already subsampled).
        subjects (np.ndarray): Subject id per row `(n,)`.
        cap (int): Maximum number of candidates, keeping the in-browser scan fast and the file small.
        seed (int): Sampling seed.

    Returns:
        list[dict]: `{t, A, B, ai, bi}` rows, where `ai`/`bi` are the first-occurrence row indices of `(t, A)` and `(t, B)`.
    """
    words = np.asarray(words).astype(str)
    subjects = np.asarray(subjects).astype(str)

    # Index the first row of every (word, subject) reading.
    first: dict[tuple[str, str], int] = {}
    for i in range(len(words)):
        key = (words[i], subjects[i])
        if key not in first:
            first[key] = i
    by_word: dict[str, list[str]] = {}
    for w, s in first:
        by_word.setdefault(w, []).append(s)

    # Every ordered subject pair sharing a word is one candidate.
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


def _sentence_index(meta: pd.DataFrame, max_sentences: int = 160) -> list[dict]:
    """Per-sentence, per-subject ordered token positions for the sentence view.

    Readings are keyed by sentence *text*, not index, so the explorer can draw one path per reader through a single
    shared sentence and show whether different people traverse it the same way.

    Args:
        meta (pd.DataFrame): Aligned token metadata (row order matches the embedding).
        max_sentences (int): Cap on returned sentences (multi-subject, longer ones preferred).

    Returns:
        list[dict]: `[{id, label, n_subj, by_subj: {subject: [row positions in reading order]}}]`.
    """
    needed = {'subject', 'sentence_idx', 'word_idx', 'word'}
    if not needed.issubset(meta.columns):
        return []
    subj = meta['subject'].astype(str).to_numpy()
    sidx = meta['sentence_idx'].fillna(-1).astype(int).to_numpy()
    widx = meta['word_idx'].fillna(-1).astype(int).to_numpy()
    words = meta['word'].astype(str).to_numpy()

    # Group rows into one reading per (subject, sentence).
    groups: dict[tuple[str, int], list[int]] = {}
    for i in range(len(meta)):
        if sidx[i] < 0:
            continue
        groups.setdefault((subj[i], int(sidx[i])), []).append(i)

    # Collapse readings of the same text, in `word_idx` order, across subjects.
    by_text: dict[str, dict] = {}
    for (s, _si), idxs in groups.items():
        idxs.sort(key=lambda i: widx[i])
        text = ' '.join(w for w in words[idxs] if w).strip()
        if len(idxs) < 2 or not text:
            continue
        rec = by_text.setdefault(text.lower(), {'label': text, 'by_subj': {}})
        rec['by_subj'][str(s)] = [int(i) for i in idxs]

    # Prefer the most-read, longest sentences and label them for the picker.
    out = list(by_text.values())
    out.sort(
        key=lambda r: (len(r['by_subj']), sum(len(v) for v in r['by_subj'].values())), reverse=True
    )
    out = out[:max_sentences]
    for j, rec in enumerate(out):
        rec['id'] = j
        rec['n_subj'] = len(rec['by_subj'])
        label = rec['label']
        rec['label'] = (label[:52] + '…') if len(label) > 53 else label
    return out


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

    # One PCA basis serves both the 3-D plot and the wider reduced vectors used for in-browser cosines.
    mean, vt = _pca_basis(emb)
    reduced = _project(emb, mean, vt, reduced_dims)
    subjects = meta['subject'].astype(str).to_numpy() if 'subject' in meta else np.array(['S'] * n)
    words = meta['word'].astype(str).to_numpy() if 'word' in meta else np.array([''] * n)

    cats = [c for c in ('subject', 'task', 'category', 'length_band') if c in meta.columns]
    nums = [c for c in ('word_len', 'log_freq') if c in meta.columns]

    # Every metadata column the page can colour or filter by, defaulted when absent.
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

    # The EEG-only set gets its own basis, so the toggle compares two independently fitted spaces.
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
        'sentences': _sentence_index(meta),
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

    Args:
        emb (np.ndarray): Word-level embeddings `(n_samples, embed_dim)` (EEG + eye-tracking, the primary set).
        meta (pd.DataFrame): Aligned metadata; recognises `subject, task, word, sentence_idx, word_idx, category,
            length_band` and optional `word_len, log_freq`. Missing columns degrade gracefully.
        out_path (str | Path): Output path (`.html`, or `.png` on fallback).
        eeg_only_emb (np.ndarray | None): Optional EEG-only embeddings aligned row-for-row with `emb`, enabling the view-4 toggle.
        centroids (dict | None): Optional `{subject: vector}` full-dim centroid override for the arithmetic offset (else computed from `emb`).
        probe_scores (dict | None): Optional `word_len` linear-probe scores rendered as a small bar.
        dims (int): Default projection (3 for 3-D, 2 for 2-D).
        max_points (int): Subsample cap for responsiveness.
        seed (int): Sampling / baseline seed.
        title (str): Page and figure title.
        emergence (dict | None): Optional full-embedding-space report from `zte.evaluation.emergence.emergence_report`,
            the same numbers as `metrics.json`. When given, the verdict banners headline these figures and demote the
            in-browser reduced-space estimate; when `None` that estimate is the headline.

    Returns:
        Path: The written path (`.html` when Plotly is available, else `.png`).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = meta.reset_index(drop=True)

    # Subsample once; every downstream block must index the same rows.
    idx = np.arange(len(emb))
    if len(emb) > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(len(emb), size=max_points, replace=False))

    emb = np.asarray(emb, dtype=np.float32)[idx]
    meta = meta.iloc[idx].reset_index(drop=True)
    eeg_sub = None

    if eeg_only_emb is not None:
        eeg_arr = np.asarray(eeg_only_emb, dtype=np.float32)
        if eeg_arr.ndim == 2 and len(eeg_arr) == len(emb):
            eeg_sub = eeg_arr  # already the subsampled length
        elif eeg_arr.ndim == 2 and len(idx) and eeg_arr.shape[0] > idx.max():
            eeg_sub = eeg_arr[idx]  # original length; subsample to stay aligned with `emb`
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
