"""Neuron-level interpretability for ZTE embeddings -- *which dimensions fire, and what they encode*.

The 768-d "thought embedding" is opaque as a whole, but each dimension is a neuron with a story:
some carry a lot of variance (they *fire*), most may be near-silent (*negligible*), and each one is,
to some degree, *selective* for an attribute -- the reader's identity, the task, the word's length or
frequency. This module makes that legible without a trained decoder:

- **Importance** -- per-dimension standard deviation and its share of the total embedding variance,
  ranked most-active to negligible, with a dead-neuron count. This is the collapse story at neuron
  resolution.
- **Selectivity** -- for every dimension, the *fraction of its variance explained* by each known
  attribute: r^2 for continuous targets (word length, log-frequency) and eta^2 for categorical ones
  (subject, task, category). Both are on the same 0..1 scale, so each neuron's *dominant* attribute is
  a fair argmax across them.
- **Budget** -- the headline discovery: what share of the space's *variance* is spent encoding
  "who" (subject) versus "what" (lexical content). A subject-dominated space spends most of its
  variance on identity -- exactly the ZTE v1 failure mode, now quantified per neuron.
- **Exemplars** -- the words that most (and least) activate each top neuron, so "neuron 42" gets a
  human-readable meaning.
- **Scalp / band attribution** -- which frequency band x scalp region drives each top neuron, tying
  a dimension back to the brain (and exposing gaze-driven neurons when eye-tracking is included).

Everything here is model-free: it reads the exported embeddings and metadata, so it works anywhere the
evaluation does. Attribution is correlational (not a causal saliency map) and is labelled as such.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from zte.data.regions import RegionMap

# Continuous vs categorical probe targets, if present in the word metadata.
_CONTINUOUS = ('word_len', 'log_freq')
_CATEGORICAL = ('subject', 'task', 'category')
# Which dominant attributes count as "what" (content) vs "who" (identity).
_CONTENT = ('word_len', 'log_freq', 'category')
_IDENTITY = ('subject',)


def _abs_pearson(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Returns `|Pearson r|` between every column of `x` and the vector `y`.

    Args:
        x (np.ndarray): Matrix `(n, d)`.
        y (np.ndarray): Vector `(n,)`.
        eps (float): Numerical floor.

    Returns:
        np.ndarray: `(d,)` absolute correlations in `[0, 1]` (0 where a column or `y` is constant).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(y) < 3:
        return np.zeros(x.shape[1])
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    num = xc.T @ yc
    den = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum()) + eps
    return np.abs(num / den)


def _eta_squared(x: np.ndarray, groups: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Returns eta^2 (between-group variance share) of every column of `x` for a grouping.

    eta^2 in `[0, 1]` is the fraction of a dimension's variance explained by the categorical label --
    a clean "how subject-selective is this neuron?" measure.

    Args:
        x (np.ndarray): Matrix `(n, d)`.
        groups (np.ndarray): Categorical labels `(n,)`.
        eps (float): Numerical floor.

    Returns:
        np.ndarray: `(d,)` eta^2 values.
    """
    x = np.asarray(x, dtype=np.float64)
    codes, _ = _codes(groups)
    k = int(codes.max()) + 1 if len(codes) else 0
    if k < 2:
        return np.zeros(x.shape[1])
    onehot = np.zeros((len(codes), k))
    onehot[np.arange(len(codes)), codes] = 1.0
    counts = onehot.sum(axis=0)  # (k,)
    group_sums = onehot.T @ x  # (k, d)
    group_means = group_sums / counts[:, None].clip(min=1)
    grand = x.mean(axis=0, keepdims=True)  # (1, d)
    ss_between = (counts[:, None] * (group_means - grand) ** 2).sum(axis=0)
    ss_total = ((x - grand) ** 2).sum(axis=0) + eps
    return np.clip(ss_between / ss_total, 0.0, 1.0)


def _codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factorises `values` to contiguous integer codes and their unique labels."""
    import pandas as pd

    codes, uniques = pd.factorize(pd.Series(values))
    return np.asarray(codes), np.asarray(uniques)


def neuron_report(
    emb: np.ndarray,
    meta: pd.DataFrame,
    *,
    band_power: np.ndarray | None = None,
    band_names: tuple[str, ...] | None = None,
    region_map: RegionMap | None = None,
    top_k: int = 24,
    n_exemplars: int = 8,
    active_quantile: float = 0.5,
) -> dict[str, Any]:
    """Computes a full neuron-level interpretability report for an embedding matrix.

    Args:
        emb (np.ndarray): Word embeddings `(n_words, embed_dim)`.
        meta (pd.DataFrame): Aligned word metadata (may hold `word`, `subject`, `task`, `category`,
            `word_len`, `log_freq`). Missing columns are simply skipped.
        band_power (np.ndarray | None): Per-word band power `(n_words, n_bands, n_channels)` for scalp
            attribution (optional).
        band_names (tuple[str, ...] | None): Names for the band axis of `band_power`.
        region_map (RegionMap | None): Channel->region map used to attribute neurons to scalp regions.
        top_k (int): How many of the most-important neurons to describe in detail.
        n_exemplars (int): How many top/bottom activating words to list per detailed neuron.
        active_quantile (float): A neuron counts as "active" if its std exceeds this quantile-scaled
            threshold (see below); also used for the dead-neuron count.

    Returns:
        dict[str, Any]: A JSON-serialisable report with `importance`, `selectivity`, `summary`,
        `top_neurons` and `meta` blocks (see module docstring for the semantics).
    """
    x = np.asarray(emb, dtype=np.float64)
    n, d = x.shape

    # --- Importance ------------------------------------------------------- #
    std = x.std(axis=0)
    var = std**2
    total_var = float(var.sum()) or 1.0
    var_share = var / total_var
    order = np.argsort(-std)  # most active first
    rank = np.empty(d, dtype=int)
    rank[order] = np.arange(d)
    # "Dead" = essentially constant; "active" = carries a non-trivial share of the top neuron's spread.
    dead = std < 1e-4
    active_threshold = max(1e-4, float(np.max(std)) * 0.05)
    active = std >= active_threshold

    # --- Selectivity ------------------------------------------------------ #
    # Every score is a fraction of a neuron's variance explained by the target, so continuous
    # (r^2) and categorical (eta^2) scores are on the same scale and comparable in the argmax.
    scores: dict[str, np.ndarray] = {}
    for col in _CONTINUOUS:
        if col in meta.columns:
            scores[col] = _abs_pearson(x, meta[col].to_numpy()) ** 2  # r^2 == variance explained
    for col in _CATEGORICAL:
        if col in meta.columns and _n_unique(meta[col]) > 1:
            scores[col] = _eta_squared(x, meta[col].to_numpy())
    targets = list(scores.keys())

    if targets:
        score_stack = np.vstack([scores[t] for t in targets])  # (T, d)
        dom_idx = score_stack.argmax(axis=0)
        dom_score = score_stack.max(axis=0)
        dominant = np.array(
            [targets[i] if dom_score[j] > 0.02 else 'none' for j, i in enumerate(dom_idx)]
        )
    else:
        dominant = np.array(['none'] * d)
        dom_score = np.zeros(d)

    # --- Budget: how much of the space's VARIANCE encodes who vs what ----- #
    budget = _budget(dominant, active)
    var_budget = _variance_budget(dominant, var_share, active)
    who = sum(var_budget.get(t, 0.0) for t in _IDENTITY)
    what = sum(var_budget.get(t, 0.0) for t in _CONTENT)
    summary = {
        'embed_dim': int(d),
        'n_active': int(active.sum()),
        'n_dead': int(dead.sum()),
        'active_variance_share': float(var_share[active].sum()),
        'neuron_budget': budget,
        'variance_budget': var_budget,
        'who_variance': float(who),
        'what_variance': float(what),
        'who_vs_what_ratio': float(who / what) if what > 1e-9 else float('inf'),
    }

    # --- Scalp / band attribution (correlational) ------------------------- #
    region_cols, region_labels = _region_features(band_power, band_names, region_map)

    # --- Detailed top neurons -------------------------------------------- #
    words = meta['word'].to_numpy() if 'word' in meta.columns else np.arange(n).astype(str)
    subjects = meta['subject'].to_numpy() if 'subject' in meta.columns else np.array([''] * n)
    top_neurons = []
    for dim in order[:top_k].tolist():
        col = x[:, dim]
        hi = np.argsort(-col)[:n_exemplars]
        lo = np.argsort(col)[:n_exemplars]
        counts, edges = np.histogram(col, bins=30)
        entry: dict[str, Any] = {
            'dim': int(dim),
            'rank': int(rank[dim]),
            'std': float(std[dim]),
            'var_share': float(var_share[dim]),
            'dominant': str(dominant[dim]),
            'dominant_score': float(dom_score[dim]),
            'selectivity': {t: float(scores[t][dim]) for t in targets},
            'activation_hist': {'counts': counts.tolist(), 'edges': edges.tolist()},
            'top_words': [
                {'word': str(words[i]), 'subject': str(subjects[i]), 'activation': float(col[i])}
                for i in hi
            ],
            'bottom_words': [
                {'word': str(words[i]), 'subject': str(subjects[i]), 'activation': float(col[i])}
                for i in lo
            ],
        }
        if region_cols is not None:
            attr = _abs_pearson(region_cols, col)
            top = np.argsort(-attr)[:6]
            entry['attribution'] = [
                {'feature': region_labels[i], 'corr': float(attr[i])} for i in top
            ]
        top_neurons.append(entry)

    # Alternative importance orderings: the default ranks by how much a neuron *fires* (variance);
    # a per-target ranking instead surfaces the neurons most *selective for* that attribute. This is
    # the answer to "importance to what?" -- you choose the axis.
    rankings = {'variance': order.tolist()}
    for t in targets:
        rankings[f'selectivity:{t}'] = np.argsort(-scores[t]).tolist()

    return {
        'meta': {
            'n_words': int(n),
            'embed_dim': int(d),
            'targets': targets,
            'has_attribution': region_cols is not None,
            'attribution_note': 'correlational (not a causal saliency map)',
        },
        'importance': {
            # A neuron "fires" when it VARIES across words; its importance is the share of the total
            # embedding variance it carries: var_share[d] = std[d]^2 / sum_j std[j]^2. `order` ranks
            # dimensions most-active -> least; a "dead" neuron is near-constant (std < 1e-4).
            'definition': 'var_share[d] = std[d]^2 / sum(std^2); rank 0 = highest-variance (fires most)',
            'std': std.tolist(),
            'var_share': var_share.tolist(),
            'rank': rank.tolist(),
            'active': active.tolist(),
            'dead': dead.tolist(),
            'active_threshold': float(active_threshold),
            'order': order.tolist(),
            'rankings': rankings,
        },
        'selectivity': {
            'note': 'variance explained by each attribute: r^2 (continuous) / eta^2 (categorical), 0..1',
            'targets': targets,
            'scores': {t: scores[t].tolist() for t in targets},
            'dominant': dominant.tolist(),
            'dominant_score': dom_score.tolist(),
        },
        'summary': summary,
        'top_neurons': top_neurons,
    }


def _n_unique(series: Any) -> int:
    """Returns the number of distinct non-null values in a pandas Series."""
    return int(series.nunique(dropna=True))


def _budget(dominant: np.ndarray, active: np.ndarray) -> dict[str, int]:
    """Counts active neurons per dominant attribute."""
    out: dict[str, int] = {}
    for label in dominant[active]:
        out[str(label)] = out.get(str(label), 0) + 1
    return out


def _variance_budget(
    dominant: np.ndarray, var_share: np.ndarray, active: np.ndarray
) -> dict[str, float]:
    """Sums the variance share of active neurons per dominant attribute."""
    out: dict[str, float] = {}
    idx = np.where(active)[0]
    for j in idx:
        label = str(dominant[j])
        out[label] = out.get(label, 0.0) + float(var_share[j])
    return out


def _region_features(
    band_power: np.ndarray | None,
    band_names: tuple[str, ...] | None,
    region_map: RegionMap | None,
) -> tuple[np.ndarray | None, list[str]]:
    """Reduces per-word band power to per (band, region) columns for neuron attribution.

    Args:
        band_power (np.ndarray | None): `(n_words, n_bands, n_channels)` band power.
        band_names (tuple[str, ...] | None): Band-axis labels.
        region_map (RegionMap | None): Channel->region map (an approximate default is used if `None`).

    Returns:
        tuple[np.ndarray | None, list[str]]: `(columns (n_words, n_bands*n_regions), labels)` or
        `(None, [])` when band power is unavailable.
    """
    if band_power is None:
        return None, []
    bp = np.asarray(band_power, dtype=np.float32)
    if bp.ndim != 3:
        return None, []
    from zte.data.regions import default_region_map

    rmap = region_map or default_region_map(bp.shape[2])
    reduced = rmap.reduce(bp, method='mean')  # (n_words, n_bands, n_regions)
    n, n_bands, n_regions = reduced.shape
    bands = band_names or tuple(f'band{i}' for i in range(n_bands))
    cols = reduced.reshape(n, n_bands * n_regions)
    # Mean-impute NaNs (omitted words) so correlations are well defined.
    col_mean = np.nanmean(cols, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    cols = np.where(np.isfinite(cols), cols, col_mean)
    labels = [f'{bands[b]} · {rmap.names[r]}' for b in range(n_bands) for r in range(n_regions)]
    return cols, labels
