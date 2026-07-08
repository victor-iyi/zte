"""Vector arithmetic on thought embeddings -- the word2vec analogy test for EEG.

Word embeddings famously support `king - man + woman ~ queen`: an attribute is a
*translation* in the space. If ZTE is to be a real thought code, the same structure
should hold for the factors that ought to be *nuisances* -- **who** produced the
signal and **which task** they were doing. Concretely, for the same stimulus token
`t`::

    emb(t, subject A) - centroid(A) + centroid(B)  ~  emb(t, subject B)

i.e. "this thought, as if thought by B". When that arithmetic retrieves the *same
stimulus* read by another subject far above chance, the subject identity is an
additive offset the space can cancel -- the essence of a device-/subject-agnostic
encoding. The same construction with the task centroid tests task-invariance.

Everything here is pure NumPy over embeddings plus a metadata table, so it is
unit-testable in isolation and reused by both the report and TensorBoard layers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from zte.logging_utils import get_logger

_LOG = get_logger('evaluation.analogy')


def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalises rows of `x`."""
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _group_centroids(emb: np.ndarray, groups: np.ndarray) -> dict[Any, np.ndarray]:
    """Returns the mean embedding of each group label."""
    return {g: emb[groups == g].mean(axis=0) for g in np.unique(groups)}


def transfer_analogy(
    emb: np.ndarray,
    groups: np.ndarray,
    contents: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
    max_pairs: int = 40000,
    seed: int = 0,
) -> dict[str, float]:
    """Cross-group translation analogy: `emb(t,A) - c(A) + c(B) -> emb(t,B)`.

    For every ordered pair of groups `(A, B)` and every stimulus present in both,
    the source embedding is translated by the group-centroid offset and matched
    against `B`'s bank; a hit is retrieving the *same stimulus* under `B`.

    Args:
        emb: Embeddings `(n_samples, embed_dim)`.
        groups: Group label per row `(n_samples,)` (e.g. subject or task).
        contents: Stimulus id per row `(n_samples,)` (same content shares an id
            across groups -- e.g. `task|sentence|word_idx`).
        ks: Top-K cut-offs.
        max_pairs: Cap on evaluated source items (sampled) to bound cost.
        seed: Sampling seed.

    Returns:
        `top{k}` for each `k`, `mrr`, `n_queries`, `chance_top1` and `n_groups`.
        All-`nan` when no cross-group stimulus overlap exists.
    """
    emb = np.asarray(emb, dtype=np.float32)
    groups = np.asarray(groups)
    contents = np.asarray(contents)
    centroids = _group_centroids(emb, groups)
    uniq_groups = list(centroids)
    rng = np.random.default_rng(seed)

    hits_at: dict[int, list[bool]] = {k: [] for k in ks}
    recip_ranks: list[float] = []
    chances: list[float] = []
    n_queries = 0
    chunk = 4096  # bound the (chunk x bank) similarity matrix

    for dst in uniq_groups:
        dst_rows = np.nonzero(groups == dst)[0]
        if dst_rows.size == 0:
            continue
        bank = _l2(emb[dst_rows])
        bank_contents = contents[dst_rows]
        # Each stimulus is read once per subject, so one bank row per content.
        bank_index = {c: j for j, c in enumerate(bank_contents.tolist())}
        for src in uniq_groups:
            if src == dst:
                continue
            src_rows = np.nonzero(groups == src)[0]
            overlap = src_rows[np.isin(contents[src_rows], bank_contents)]
            if overlap.size == 0:
                continue
            if overlap.size > max_pairs:
                overlap = rng.choice(overlap, size=max_pairs, replace=False)
            offset = centroids[dst] - centroids[src]
            target_j = np.array([bank_index[c] for c in contents[overlap].tolist()])
            # Rank the *correct* bank item by counting strictly-more-similar items --
            # top-k / MRR without an O(bank log bank) argsort over the whole bank.
            for start in range(0, len(overlap), chunk):
                rows = overlap[start : start + chunk]
                queries = _l2(emb[rows] + offset[None, :])
                sims = queries @ bank.T  # (chunk, bank_size)
                tgt_sim = sims[np.arange(len(rows)), target_j[start : start + chunk]]
                rank = (sims > tgt_sim[:, None]).sum(axis=1) + 1
                for k in ks:
                    hits_at[k].extend((rank <= k).tolist())
                recip_ranks.extend((1.0 / rank).tolist())
            chances.extend([1.0 / len(bank)] * len(overlap))
            n_queries += len(overlap)

    if n_queries == 0:
        return {f'top{k}': float('nan') for k in ks} | {
            'mrr': float('nan'), 'n_queries': 0.0, 'chance_top1': float('nan'),
            'n_groups': float(len(uniq_groups)),
        }
    out = {f'top{k}': float(np.mean(hits_at[k])) for k in ks}
    out['mrr'] = float(np.mean(recip_ranks))
    out['n_queries'] = float(n_queries)
    out['chance_top1'] = float(np.nanmean(chances))
    out['n_groups'] = float(len(uniq_groups))
    return out


def analogy_examples(
    emb: np.ndarray,
    meta: pd.DataFrame,
    group_col: str,
    content_col: str,
    label_col: str,
    n_examples: int = 8,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Concrete `a - A + B` demonstrations for the report (king/queen style).

    Args:
        emb: Embeddings `(n_samples, embed_dim)`.
        meta: Aligned metadata with `group_col`, `content_col` and `label_col`.
        group_col: Column whose centroid offset is added/subtracted (e.g. subject).
        content_col: Stimulus-identity column (same content shares a value).
        label_col: Human-readable label to print (e.g. the word).
        n_examples: How many worked examples to return.
        seed: Sampling seed.

    Returns:
        Rows with `source`/`expression`/`retrieved`/`target`, a `hit` flag and the
        cosine `similarity` -- ready to render as a table.
    """
    emb = np.asarray(emb, dtype=np.float32)
    groups = meta[group_col].to_numpy()
    contents = meta[content_col].to_numpy()
    labels = meta[label_col].astype(str).to_numpy()
    centroids = _group_centroids(emb, groups)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    order = rng.permutation(len(emb))
    for i in order:
        src = groups[i]
        others = [g for g in centroids if g != src]
        if not others:
            continue
        dst = others[int(rng.integers(len(others)))]
        dst_rows = np.nonzero(groups == dst)[0]
        if dst_rows.size == 0:
            continue
        query = _l2((emb[i] + centroids[dst] - centroids[src])[None, :])[0]
        bank = _l2(emb[dst_rows])
        j = int(np.argmax(bank @ query))
        hit = bool(contents[dst_rows[j]] == contents[i])
        rows.append(
            {
                'source': f'{labels[i]!r} [{src}]',
                'expression': f'{labels[i]!r} - {src} + {dst}',
                'retrieved': f'{labels[dst_rows[j]]!r} [{dst}]',
                'target': f'{labels[i]!r} [{dst}]',
                'hit': hit,
                'similarity': round(float(bank[j] @ query), 3),
            }
        )
        if len(rows) >= n_examples:
            break
    return rows


def analogy_report(
    emb: np.ndarray,
    meta: pd.DataFrame,
    raw_feats: np.ndarray | None = None,
) -> dict[str, Any]:
    """Runs subject- and task-transfer analogies (+ a raw-feature control).

    Args:
        emb: Word-level ZTE embeddings `(n_words, embed_dim)`.
        meta: Aligned word metadata with `subject`, `task`, `sentence_idx`, `word_idx` and `word`.
        raw_feats: Optional aligned raw features `(n_words, n_features)` for a control
            showing the arithmetic is a property of the *learned* space, not the inputs.

    Returns:
        A dict with `subject_transfer` / `task_transfer` metric blocks, matched
        raw-feature controls, and human-readable `examples`.
    """
    meta = meta.reset_index(drop=True)
    stimulus = (
        meta['task'].astype(str)
        + '|' + meta['sentence_idx'].astype(str)
        + '|' + meta['word_idx'].astype(str)
    ).to_numpy()

    report: dict[str, Any] = {}
    report['subject_transfer'] = transfer_analogy(emb, meta['subject'].to_numpy(), stimulus)
    if meta['task'].nunique() > 1:
        report['task_transfer'] = transfer_analogy(emb, meta['task'].to_numpy(), stimulus)
    if raw_feats is not None:
        report['subject_transfer_raw'] = transfer_analogy(
            np.asarray(raw_feats, dtype=np.float32), meta['subject'].to_numpy(), stimulus
        )
    report['examples'] = analogy_examples(
        emb, meta.assign(_stim=stimulus), group_col='subject', content_col='_stim',
        label_col='word',
    )
    st = report['subject_transfer']
    _LOG.info(
        'Subject-transfer analogy: Top-1 %.3f vs chance %.3f (%d queries).',
        st.get('top1', float('nan')), st.get('chance_top1', float('nan')), int(st.get('n_queries', 0)),
    )
    return report
