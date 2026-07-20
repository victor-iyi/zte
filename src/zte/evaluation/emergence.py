"""Emergent-property metrics: do similar thoughts cluster across people, regardless of who produced them?"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Returns row-wise L2-normalised `x` as float32."""
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _verdict(gap: float, weak: float = 0.02, strong: float = 0.1) -> str:
    """Maps a (same - random) cosine gap to a plain-language verdict."""
    if not np.isfinite(gap) or gap < weak:
        return 'not clustered'
    if gap < strong:
        return 'weakly clustered'
    return 'clustered'


def _pair_cosine_mean(
    unit: np.ndarray, left: np.ndarray, right: np.ndarray, rng: np.random.Generator, cap: int
) -> float:
    """Mean cosine over (subsampled) index pairs `(left[i], right[i])` of a unit-normalised matrix."""
    if len(left) == 0:
        return float('nan')
    if len(left) > cap:
        sel = rng.choice(len(left), size=cap, replace=False)
        left, right = left[sel], right[sel]
    return float(np.mean(np.sum(unit[left] * unit[right], axis=1)))


def cross_subject_clustering(
    word_emb: np.ndarray,
    word_meta: pd.DataFrame,
    *,
    max_pairs: int = 200_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Measures whether the same word / same meaning clusters across subjects.

    Args:
        word_emb (np.ndarray): Word embeddings `(n_words, embed_dim)`.
        word_meta (pd.DataFrame): Aligned metadata with `subject`, `word`, and optionally `category`.
        max_pairs (int): Cap on the number of pairs averaged per statistic (for speed).
        seed (int): RNG seed for pair subsampling.

    Returns:
        dict[str, Any]: `same_word` and (if `category` present) `same_meaning` blocks, each with the
        same-pair mean cosine, the random-pair baseline, their gap and a verdict.
    """
    rng = np.random.default_rng(seed)
    unit = _normalize(word_emb)
    n = len(unit)
    out: dict[str, Any] = {}

    subject = word_meta['subject'].to_numpy() if 'subject' in word_meta.columns else None
    if subject is None or len(np.unique(subject)) < 2:
        out['applicable'] = False
        out['reason'] = 'need >=2 subjects'
        return out
    out['applicable'] = True

    # Random cross-subject baseline (pairs guaranteed to be different subjects).
    def _random_xsubj(cap: int) -> float:
        a = rng.integers(0, n, size=cap)
        b = rng.integers(0, n, size=cap)
        keep = subject[a] != subject[b]
        a, b = a[keep], b[keep]
        return _pair_cosine_mean(unit, a, b, rng, cap)

    random_xsubj = _random_xsubj(min(max_pairs, 100_000))

    # Same word, different subject.
    words = word_meta['word'].astype(str).to_numpy()
    left, right = _same_key_cross_subject_pairs(words, subject, rng, max_pairs)
    same_word = _pair_cosine_mean(unit, left, right, rng, max_pairs)
    out['same_word'] = {
        'definition': 'mean cosine of the SAME word read by DIFFERENT subjects, vs random cross-subject pairs',
        'mean_cosine': same_word,
        'random_baseline': random_xsubj,
        'gap': _finite_sub(same_word, random_xsubj),
        'n_word_types_multi_subject': int(_n_multi_subject_keys(words, subject)),
        'verdict': _verdict(_finite_sub(same_word, random_xsubj)),
    }

    # Same category (meaning proxy), different subject.
    if 'category' in word_meta.columns and word_meta['category'].nunique() > 1:
        cats = word_meta['category'].astype(str).to_numpy()
        cl, cr = _same_key_cross_subject_pairs(cats, subject, rng, max_pairs)
        same_cat = _pair_cosine_mean(unit, cl, cr, rng, max_pairs)
        out['same_meaning'] = {
            'definition': 'mean cosine of SAME-category words across DIFFERENT subjects, vs random cross-subject pairs',
            'meaning_proxy': 'sentence category',
            'mean_cosine': same_cat,
            'random_baseline': random_xsubj,
            'gap': _finite_sub(same_cat, random_xsubj),
            'verdict': _verdict(_finite_sub(same_cat, random_xsubj)),
        }
    return out


def semantic_neighbourhood(
    word_emb: np.ndarray,
    word_meta: pd.DataFrame,
    *,
    k: int = 10,
    max_queries: int = 4000,
    seed: int = 0,
) -> dict[str, Any]:
    """Measures how semantically coherent (and cross-subject) each token's neighbourhood is.

    Args:
        word_emb (np.ndarray): Word embeddings `(n_words, embed_dim)`.
        word_meta (pd.DataFrame): Aligned metadata with `subject`, `word`, optionally `category`.
        k (int): Neighbourhood size.
        max_queries (int): Cap on query tokens (for speed).
        seed (int): RNG seed for query subsampling.

    Returns:
        dict[str, Any]: neighbourhood purities (same word / same category vs chance) and the fraction
        of neighbours drawn from a different subject (high = subject-invariant).
    """
    from zte.inference.retrieval import NearestNeighborIndex

    n = len(word_emb)
    if n < k + 2:
        return {'applicable': False, 'reason': 'too few tokens'}
    rng = np.random.default_rng(seed)
    q = np.arange(n) if n <= max_queries else rng.choice(n, size=max_queries, replace=False)

    import pandas as pd

    # Neighbourhoods of the sampled query tokens, self excluded.
    index = NearestNeighborIndex(word_emb, pd.DataFrame({'_i': np.arange(n)}))
    idx, _ = index.query(word_emb[q], k=k, self_indices=q)  # (n_q, k) neighbour row ids

    # Purity: how often a neighbour shares the query's word, and how often it comes from someone else.
    words = word_meta['word'].astype(str).to_numpy()
    subject = word_meta['subject'].to_numpy() if 'subject' in word_meta.columns else np.zeros(n)
    same_word = np.mean(words[idx] == words[q][:, None])
    cross_subj = np.mean(subject[idx] != subject[q][:, None])

    out: dict[str, Any] = {
        'applicable': True,
        'k': int(k),
        'same_word_purity': float(same_word),
        'cross_subject_neighbour_fraction': float(cross_subj),
        'definition': (
            "fraction of each token's k nearest neighbours that share its word / category, and the "
            'fraction that come from a different subject (higher cross-subject fraction = more '
            'subject-invariant neighbourhoods)'
        ),
    }

    # Category purity is only meaningful against the majority-category prior.
    if 'category' in word_meta.columns and word_meta['category'].nunique() > 1:
        cats = word_meta['category'].astype(str).to_numpy()
        same_cat = float(np.mean(cats[idx] == cats[q][:, None]))
        _, counts = np.unique(cats, return_counts=True)
        chance = float((counts / counts.sum()).max())  # majority-category prior
        out['same_category_purity'] = same_cat
        out['category_chance'] = chance
        out['category_coherence'] = same_cat - chance
        out['verdict'] = _verdict(same_cat - chance)
    return out


def emergence_report(
    word_emb: np.ndarray,
    word_meta: pd.DataFrame,
    *,
    analogy: dict[str, Any] | None = None,
    k: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Bundles the emergent-property metrics into one report with a headline verdict.

    Args:
        word_emb (np.ndarray): Word embeddings `(n_words, embed_dim)`.
        word_meta (pd.DataFrame): Aligned word metadata.
        analogy (dict[str, Any] | None): The already-computed analogy report (for the subject-transfer
            hit-rate headline); optional.
        k (int): Neighbourhood size for the coherence metric.
        seed (int): RNG seed.

    Returns:
        dict[str, Any]: `cross_subject`, `neighbourhood`, `analogy`, and a `headline` one-liner.
    """
    cross = cross_subject_clustering(word_emb, word_meta, seed=seed)
    neigh = semantic_neighbourhood(word_emb, word_meta, k=k, seed=seed)
    report: dict[str, Any] = {'cross_subject': cross, 'neighbourhood': neigh}

    if analogy is not None:
        st = analogy.get('subject_transfer', {})
        report['analogy'] = {
            'definition': 'emb(t, A) - centroid(A) + centroid(B) retrieves t read by B (Top-1 hit rate)',
            'subject_transfer_top1': st.get('top1', float('nan')),
            'chance_top1': st.get('chance_top1', float('nan')),
        }

    sw = cross.get('same_word', {}) if cross.get('applicable') else {}
    report['headline'] = (
        'Same word across subjects: cosine '
        f'{sw.get("mean_cosine", float("nan")):.3f} vs random {sw.get("random_baseline", float("nan")):.3f} '
        f'-> {sw.get("verdict", "n/a")}.'
        if sw
        else 'Cross-subject clustering not applicable (need >=2 subjects).'
    )
    return report


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _finite_sub(a: float, b: float) -> float:
    """Returns `a - b`, or nan if either is non-finite."""
    return float(a - b) if np.isfinite(a) and np.isfinite(b) else float('nan')


def _same_key_cross_subject_pairs(
    keys: np.ndarray, subject: np.ndarray, rng: np.random.Generator, cap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Samples index pairs that share a key but differ in subject.

    Args:
        keys (np.ndarray): Grouping key per row (e.g. word or category).
        subject (np.ndarray): Subject label per row.
        rng (np.random.Generator): RNG.
        cap (int): Max pairs to return.

    Returns:
        tuple[np.ndarray, np.ndarray]: `(left, right)` row-index arrays.
    """
    order = np.argsort(keys, kind='stable')
    keys_sorted = keys[order]
    boundaries = np.flatnonzero(keys_sorted[1:] != keys_sorted[:-1]) + 1
    groups = np.split(order, boundaries)
    left: list[int] = []
    right: list[int] = []
    for g in groups:
        if len(g) < 2:
            continue
        subs = subject[g]
        if len(np.unique(subs)) < 2:
            continue
        # Sample a bounded number of cross-subject pairs within this group.
        take = min(len(g), 8)
        a = rng.choice(g, size=take, replace=False)
        b = rng.choice(g, size=take, replace=False)
        for i, j in zip(a, b, strict=True):
            if subject[i] != subject[j]:
                left.append(int(i))
                right.append(int(j))
        if len(left) >= cap:
            break
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def _n_multi_subject_keys(keys: np.ndarray, subject: np.ndarray) -> int:
    """Counts distinct keys that appear for at least two subjects."""
    import pandas as pd

    frame = pd.DataFrame({'k': keys, 's': subject})
    per_key = frame.groupby('k')['s'].nunique()
    return int((per_key >= 2).sum())
