"""Language-decoding metrics for EEG→text evaluation.

Pure-Python / NumPy implementations (no NLTK required). Optional sacrebleu is
used when installed. Cross-modal retrieval wraps
:func:`zte.training.metrics.retrieval_metrics`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from zte.training.metrics import noise_matched, retrieval_metrics


def bleu_score(hyps: list[str], refs: list[str], max_n: int = 4) -> float:
    """Corpus BLEU-ish score (modified n-gram precision + brevity penalty).

    Args:
        hyps: Hypothesis strings.
        refs: Reference strings (same length as ``hyps``).
        max_n: Maximum n-gram order (default 4).

    Returns:
        BLEU in ``[0, 1]``. Returns ``0.0`` on empty input.
    """
    if not hyps or not refs or len(hyps) != len(refs):
        return 0.0
    try:
        import sacrebleu  # type: ignore[import-untyped]

        return float(sacrebleu.corpus_bleu(hyps, [refs]).score) / 100.0
    except ImportError:
        pass

    hyp_len = 0
    ref_len = 0
    for hyp, ref in zip(hyps, refs, strict=True):
        ht = _tokenize(hyp)
        rt = _tokenize(ref)
        hyp_len += len(ht)
        ref_len += len(rt)
    if hyp_len == 0 or ref_len == 0:
        return 0.0

    # Exact normalised match → perfect BLEU (avoids short-sentence n-gram holes).
    if all(_norm(h) == _norm(r) for h, r in zip(hyps, refs, strict=True)):
        return 1.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        overlap = 0
        total = 0
        for hyp, ref in zip(hyps, refs, strict=True):
            ht = _tokenize(hyp)
            rt = _tokenize(ref)
            hyp_ng = _ngrams(ht, n)
            ref_ng = _ngrams(rt, n)
            if not hyp_ng:
                continue
            total += sum(hyp_ng.values())
            for ng, count in hyp_ng.items():
                overlap += min(count, ref_ng.get(ng, 0))
        if total > 0:
            precisions.append(overlap / total)

    if not precisions or any(p <= 0 for p in precisions):
        return 0.0
    log_avg = sum(np.log(p) for p in precisions) / len(precisions)
    bp = 1.0 if hyp_len > ref_len else float(np.exp(1.0 - ref_len / max(hyp_len, 1)))
    return float(bp * np.exp(log_avg))


def exact_match(hyps: list[str], refs: list[str]) -> float:
    """Fraction of hypotheses that exactly match their reference (normalised).

    Args:
        hyps: Hypothesis strings.
        refs: Reference strings.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    if not hyps or len(hyps) != len(refs):
        return 0.0
    return float(np.mean([_norm(h) == _norm(r) for h, r in zip(hyps, refs, strict=True)]))


def token_f1(hyps: list[str], refs: list[str]) -> float:
    """Mean bag-of-words token F1 across pairs.

    Args:
        hyps: Hypothesis strings.
        refs: Reference strings.

    Returns:
        Mean F1 in ``[0, 1]``.
    """
    if not hyps or len(hyps) != len(refs):
        return 0.0
    scores = [_token_f1_one(h, r) for h, r in zip(hyps, refs, strict=True)]
    return float(np.mean(scores))


def wer_score(hyps: list[str], refs: list[str]) -> float:
    """Corpus word error rate (edit distance / reference length).

    Args:
        hyps: Hypothesis strings.
        refs: Reference strings.

    Returns:
        WER (lower is better). ``0.0`` when both sides are empty.
    """
    if not hyps or len(hyps) != len(refs):
        return 1.0
    total_edits = 0
    total_ref = 0
    for hyp, ref in zip(hyps, refs, strict=True):
        ht = _tokenize(hyp)
        rt = _tokenize(ref)
        total_edits += _edit_distance(ht, rt)
        total_ref += max(1, len(rt))
    return float(total_edits / total_ref)


def cross_modal_retrieval(
    eeg: np.ndarray,
    text: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Top-K / MRR for paired EEG↔text embeddings.

    Args:
        eeg: Query embeddings ``(N, D)``.
        text: Key embeddings ``(N, D)`` aligned with ``eeg``.
        ks: Cut-offs for Top-K accuracy.

    Returns:
        Dict with ``top{k}`` and ``mrr``.
    """
    return retrieval_metrics(eeg, text, ks=ks)


def noise_anchored_retrieval(
    eeg: np.ndarray,
    text: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
    seed: int = 0,
) -> dict[str, float]:
    """Compares real retrieval against a noise-matched EEG control.

    Args:
        eeg: Real EEG embeddings ``(N, D)``.
        text: Text embeddings ``(N, D)``.
        ks: Cut-offs for Top-K.
        seed: RNG seed for the noise control.

    Returns:
        Dict with real metrics, ``noise_*`` counterparts, and ``lift_*`` deltas
        (real − noise) plus ``beats_noise_anchor`` (bool as 0/1 on top1).
    """
    real = cross_modal_retrieval(eeg, text, ks=ks)
    noise_eeg = noise_matched(np.asarray(eeg, dtype=np.float32), seed=seed)
    noise = cross_modal_retrieval(noise_eeg, text, ks=ks)
    out: dict[str, float] = {**real}
    for key, value in noise.items():
        out[f'noise_{key}'] = value
        out[f'lift_{key}'] = float(real[key] - value)
    out['beats_noise_anchor'] = float(real.get('top1', 0.0) > noise.get('top1', 0.0))
    return out


def decode_breakdown(
    hyps: list[str],
    refs: list[str],
    meta: pd.DataFrame | dict[str, Any] | None,
) -> dict[str, Any]:
    """Per-subject / per-task exact_match and BLEU.

    Args:
        hyps: Hypothesis strings.
        refs: Reference strings.
        meta: Metadata with optional ``subject`` / ``task`` columns (DataFrame or
            dict of equal-length arrays).

    Returns:
        Nested dict ``{'by_subject': {...}, 'by_task': {...}, 'overall': {...}}``.
    """
    overall = {
        'exact_match': exact_match(hyps, refs),
        'bleu': bleu_score(hyps, refs),
        'token_f1': token_f1(hyps, refs),
        'wer': wer_score(hyps, refs),
        'n': len(hyps),
    }
    if meta is None or len(hyps) == 0:
        return {'overall': overall, 'by_subject': {}, 'by_task': {}}

    frame = meta if isinstance(meta, pd.DataFrame) else pd.DataFrame(meta)
    if len(frame) != len(hyps):
        return {'overall': overall, 'by_subject': {}, 'by_task': {}}

    def _group(col: str) -> dict[str, dict[str, float]]:
        if col not in frame.columns:
            return {}
        out: dict[str, dict[str, float]] = {}
        for key, idx in frame.groupby(col, sort=True).groups.items():
            ii = list(idx)
            h = [hyps[i] for i in ii]
            r = [refs[i] for i in ii]
            out[str(key)] = {
                'exact_match': exact_match(h, r),
                'bleu': bleu_score(h, r),
                'n': float(len(ii)),
            }
        return out

    return {
        'overall': overall,
        'by_subject': _group('subject'),
        'by_task': _group('task'),
    }


def _tokenize(text: str) -> list[str]:
    """Whitespace tokenisation after light normalisation."""
    return _norm(text).split() if text else []


def _norm(text: str) -> str:
    """Lowercase + collapse whitespace."""
    return ' '.join(str(text).strip().lower().split())


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    """Counts n-grams over a token list."""
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _token_f1_one(hyp: str, ref: str) -> float:
    """Bag-of-words F1 for one pair."""
    ht = Counter(_tokenize(hyp))
    rt = Counter(_tokenize(ref))
    if not ht and not rt:
        return 1.0
    if not ht or not rt:
        return 0.0
    overlap = sum((ht & rt).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(ht.values())
    recall = overlap / sum(rt.values())
    return 2 * precision * recall / (precision + recall)


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two token sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]
