"""Free-running text-generation metrics and the paired-control generation report, in pure stdlib and numpy."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from zte.evaluation.metrics import bootstrap_ci

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Function words carry no evidence about which sentence was read, so a fluent LM scores on them for free.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but nor if while of to in on at by for with from into over under about after
    again against before below between down during further off once out through up
    is are was were be been being am do does did done have has had having
    i you he she it we they me him her us them my your his hers its our their mine yours ours theirs
    this that these those there here as so than then too very can could will would shall should may
    might must not no only own same s t don didn doesn isn aren wasn weren won just now what which
    who whom whose when where why how all any both each few more most other some such
    """.split()
)

# Which direction is better, so a paired delta is always signed "hypothesis minus control, higher = better".
_HIGHER_IS_BETTER: dict[str, bool] = {
    'bleu1': True,
    'bleu2': True,
    'bleu3': True,
    'bleu4': True,
    'sentence_bleu4': True,
    'rouge1': True,
    'rouge2': True,
    'rougeL': True,
    'wer': False,
    'content_f1': True,
}

# Per-sentence metrics a paired delta can be taken over.
_SENTENCE_METRICS: tuple[str, ...] = (
    'bleu1',
    'sentence_bleu4',
    'rouge1',
    'rouge2',
    'rougeL',
    'wer',
    'content_f1',
)

# A key matching either suffix is a quarantined diagnostic and must never reach a verdict.
_QUARANTINE_RE = re.compile(r'(_DIAGNOSTIC|_RETRIEVAL)$')


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #


def tokenise(text: str) -> list[str]:
    """Splits `text` into lowercase word tokens, dropping punctuation but keeping intra-word apostrophes.

    Args:
        text (str): Raw hypothesis or reference text.

    Returns:
        list[str]: Word tokens (empty for text with no alphanumeric content).
    """
    return _TOKEN_RE.findall(str(text).lower())


def normalise_text(text: str) -> str:
    """Canonical form used by every metric here: lowercased, punctuation-stripped, single-spaced.

    Args:
        text (str): Raw text.

    Returns:
        str: The whitespace-joined token sequence.
    """
    return ' '.join(tokenise(text))


def content_words(text: str, stopwords: frozenset[str] | None = None) -> set[str]:
    """Content-word types in `text`: the token set minus the stopword list.

    Args:
        text (str): Raw text.
        stopwords (frozenset[str] | None, optional): Words to drop. Defaults to the module list.

    Returns:
        set[str]: Distinct content-word types.
    """
    stop = _STOPWORDS if stopwords is None else stopwords
    return {t for t in tokenise(text) if t not in stop}


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    """Counts the `n`-grams of a token sequence."""
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def bleu(hyps: list[str], refs: list[str], max_n: int = 4, smooth: bool = False) -> dict[str, float]:
    """Corpus BLEU-1..`max_n`: clipped modified n-gram precision with the brevity penalty.

    Corpus BLEU pools the clipped match and total counts across the whole corpus before taking the
    geometric mean, so a single sentence with no 4-gram match does not zero the corpus score. With
    `smooth=False` a precision of exactly zero still zeroes every order at or above it, which is the
    expected outcome at this signal level and must not be hidden.

    Args:
        hyps (list[str]): Hypothesis strings.
        refs (list[str]): Reference strings, aligned with `hyps`.
        max_n (int, optional): Highest n-gram order. Defaults to 4.
        smooth (bool, optional): Add-1 smoothing of the orders above 1. Defaults to False.

    Returns:
        dict[str, float]: `bleu{n}` and `precision{n}` for each order, plus `brevity_penalty`,
            `hyp_len` and `ref_len`.

    Raises:
        ValueError: If `hyps` and `refs` differ in length, which would silently mis-pair the corpus.
    """
    if len(hyps) != len(refs):
        raise ValueError(f'hyps/refs length mismatch: {len(hyps)} vs {len(refs)}')

    matches = [0] * (max_n + 1)
    totals = [0] * (max_n + 1)
    hyp_len = 0
    ref_len = 0
    for hyp, ref in zip(hyps, refs, strict=True):
        h_tok, r_tok = tokenise(hyp), tokenise(ref)
        hyp_len += len(h_tok)
        ref_len += len(r_tok)
        for n in range(1, max_n + 1):
            h_ng, r_ng = _ngrams(h_tok, n), _ngrams(r_tok, n)
            totals[n] += sum(h_ng.values())
            matches[n] += sum(min(c, r_ng[g]) for g, c in h_ng.items())

    out: dict[str, float] = {'hyp_len': float(hyp_len), 'ref_len': float(ref_len)}
    bp = 1.0 if hyp_len > ref_len else (math.exp(1.0 - ref_len / hyp_len) if hyp_len > 0 else 0.0)
    out['brevity_penalty'] = float(bp)

    log_p = 0.0
    for n in range(1, max_n + 1):
        m, t = matches[n], totals[n]
        if smooth and n > 1:
            p = (m + 1.0) / (t + 1.0)
        else:
            p = (m / t) if t > 0 else 0.0
        out[f'precision{n}'] = float(p)
        log_p += math.log(p) if p > 0 else -math.inf
        out[f'bleu{n}'] = float(bp * math.exp(log_p / n)) if math.isfinite(log_p) else 0.0
    return out


def sentence_bleu(hyp: str, ref: str, max_n: int = 4) -> float:
    """Add-1 smoothed sentence BLEU, so a per-sentence paired delta is not identically zero.

    Args:
        hyp (str): Hypothesis string.
        ref (str): Reference string.
        max_n (int, optional): Highest n-gram order. Defaults to 4.

    Returns:
        float: Smoothed BLEU in `[0, 1]`; 0.0 when the hypothesis has no tokens.
    """
    h_tok, r_tok = tokenise(hyp), tokenise(ref)
    if not h_tok or not r_tok:
        return 0.0
    bp = 1.0 if len(h_tok) > len(r_tok) else math.exp(1.0 - len(r_tok) / len(h_tok))
    log_p = 0.0
    for n in range(1, max_n + 1):
        h_ng, r_ng = _ngrams(h_tok, n), _ngrams(r_tok, n)
        m = sum(min(c, r_ng[g]) for g, c in h_ng.items())
        t = sum(h_ng.values())
        p = (m / t) if (n == 1 and t > 0) else ((m + 1.0) / (t + 1.0))
        if p <= 0.0:
            return 0.0
        log_p += math.log(p)
    return float(bp * math.exp(log_p / max_n))


def _rouge_n_f1(h_tok: list[str], r_tok: list[str], n: int) -> float:
    """F1 over clipped `n`-gram overlap between two token sequences."""
    h_ng, r_ng = _ngrams(h_tok, n), _ngrams(r_tok, n)
    h_total, r_total = sum(h_ng.values()), sum(r_ng.values())
    if h_total == 0 or r_total == 0:
        return 1.0 if (h_total == 0 and r_total == 0) else 0.0
    overlap = sum(min(c, r_ng[g]) for g, c in h_ng.items())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / h_total, overlap / r_total
    return float(2.0 * precision * recall / (precision + recall))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length, on a rolling row so memory stays `O(len(b))`."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token in a:
        cur = [0] * (len(b) + 1)
        for j, other in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if token == other else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _rouge_l_f1(h_tok: list[str], r_tok: list[str]) -> float:
    """F1 over the longest common subsequence of two token sequences."""
    if not h_tok or not r_tok:
        return 1.0 if (not h_tok and not r_tok) else 0.0
    lcs = _lcs_length(h_tok, r_tok)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(h_tok), lcs / len(r_tok)
    return float(2.0 * precision * recall / (precision + recall))


def rouge(hyps: list[str], refs: list[str]) -> dict[str, float]:
    """Mean per-sentence ROUGE-1 / ROUGE-2 / ROUGE-L F1.

    Args:
        hyps (list[str]): Hypothesis strings.
        refs (list[str]): Reference strings, aligned with `hyps`.

    Returns:
        dict[str, float]: `rouge1`, `rouge2`, `rougeL` (all `nan` on an empty corpus).

    Raises:
        ValueError: If `hyps` and `refs` differ in length, which would silently mis-pair the corpus.
    """
    if len(hyps) != len(refs):
        raise ValueError(f'hyps/refs length mismatch: {len(hyps)} vs {len(refs)}')
    if not hyps:
        return {'rouge1': float('nan'), 'rouge2': float('nan'), 'rougeL': float('nan')}
    r1, r2, rl = [], [], []
    for hyp, ref in zip(hyps, refs, strict=True):
        h_tok, r_tok = tokenise(hyp), tokenise(ref)
        r1.append(_rouge_n_f1(h_tok, r_tok, 1))
        r2.append(_rouge_n_f1(h_tok, r_tok, 2))
        rl.append(_rouge_l_f1(h_tok, r_tok))
    return {
        'rouge1': float(np.mean(r1)),
        'rouge2': float(np.mean(r2)),
        'rougeL': float(np.mean(rl)),
    }


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two token sequences, on a rolling row."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, token in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, other in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (token != other))
        prev = cur
    return prev[-1]


def wer(hyps: list[str], refs: list[str]) -> float:
    """Corpus word error rate: total edit distance over total reference words.

    Args:
        hyps (list[str]): Hypothesis strings.
        refs (list[str]): Reference strings, aligned with `hyps`.

    Returns:
        float: Edits per reference word; it exceeds 1.0 when the hypotheses are longer than the
            references, and is `nan` when the references hold no words.

    Raises:
        ValueError: If `hyps` and `refs` differ in length, which would silently mis-pair the corpus.
    """
    if len(hyps) != len(refs):
        raise ValueError(f'hyps/refs length mismatch: {len(hyps)} vs {len(refs)}')
    edits = 0
    ref_words = 0
    for hyp, ref in zip(hyps, refs, strict=True):
        h_tok, r_tok = tokenise(hyp), tokenise(ref)
        edits += _edit_distance(h_tok, r_tok)
        ref_words += len(r_tok)
    return float(edits / ref_words) if ref_words else float('nan')


def sentence_wer(hyp: str, ref: str) -> float:
    """Per-sentence word error rate, normalised by the reference length."""
    h_tok, r_tok = tokenise(hyp), tokenise(ref)
    if not r_tok:
        return 0.0 if not h_tok else float('nan')
    return float(_edit_distance(h_tok, r_tok) / len(r_tok))


def _set_f1(a: set[str], b: set[str]) -> float:
    """F1 over two type sets; two empty sets agree exactly and score 1.0."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(b)
    return float(2.0 * precision * recall / (precision + recall))


def content_word_f1(hyps: list[str], refs: list[str], stopwords: frozenset[str] | None = None) -> np.ndarray:
    """Per-sentence F1 over content-word types -- the most sensitive metric at this signal level.

    A frozen LM reproduces function words from its own prior, so BLEU and ROUGE are dominated by
    fluency. Restricting to content-word types measures only what the conditioning vector could have
    supplied. Two empty content sets count as exact agreement.

    Args:
        hyps (list[str]): Hypothesis strings.
        refs (list[str]): Reference strings, aligned with `hyps`.
        stopwords (frozenset[str] | None, optional): Words to drop. Defaults to the module list.

    Returns:
        np.ndarray: Per-sentence F1 `(n,)` in `[0, 1]`.

    Raises:
        ValueError: If `hyps` and `refs` differ in length, which would silently mis-pair the corpus.
    """
    if len(hyps) != len(refs):
        raise ValueError(f'hyps/refs length mismatch: {len(hyps)} vs {len(refs)}')
    return np.asarray(
        [_set_f1(content_words(h, stopwords), content_words(r, stopwords)) for h, r in zip(hyps, refs, strict=True)],
        dtype=np.float64,
    )


def per_sentence_scores(hyps: list[str], refs: list[str]) -> dict[str, np.ndarray]:
    """Every per-sentence metric at once, so a paired bootstrap can run over any of them.

    Args:
        hyps (list[str]): Hypothesis strings.
        refs (list[str]): Reference strings, aligned with `hyps`.

    Returns:
        dict[str, np.ndarray]: `bleu1`, `sentence_bleu4`, `rouge1`, `rouge2`, `rougeL`, `wer`,
            `content_f1`, each `(n,)`.

    Raises:
        ValueError: If `hyps` and `refs` differ in length, which would silently mis-pair the corpus.
    """
    if len(hyps) != len(refs):
        raise ValueError(f'hyps/refs length mismatch: {len(hyps)} vs {len(refs)}')
    n = len(hyps)
    out = {k: np.zeros(n, dtype=np.float64) for k in _SENTENCE_METRICS}
    for i, (hyp, ref) in enumerate(zip(hyps, refs, strict=True)):
        h_tok, r_tok = tokenise(hyp), tokenise(ref)
        out['bleu1'][i] = _unigram_precision(h_tok, r_tok)
        out['sentence_bleu4'][i] = sentence_bleu(hyp, ref)
        out['rouge1'][i] = _rouge_n_f1(h_tok, r_tok, 1)
        out['rouge2'][i] = _rouge_n_f1(h_tok, r_tok, 2)
        out['rougeL'][i] = _rouge_l_f1(h_tok, r_tok)
        out['wer'][i] = sentence_wer(hyp, ref)
    out['content_f1'] = content_word_f1(hyps, refs)
    return out


def _unigram_precision(h_tok: list[str], r_tok: list[str]) -> float:
    """Clipped unigram precision -- sentence BLEU-1 without the brevity penalty."""
    if not h_tok:
        return 0.0
    r_ng = Counter(r_tok)
    h_ng = Counter(h_tok)
    overlap = sum(min(c, r_ng[g]) for g, c in h_ng.items())
    return float(overlap / len(h_tok))


def corpus_scores(hyps: list[str], refs: list[str]) -> dict[str, float]:
    """Corpus-level absolute scores for one decode condition.

    Args:
        hyps (list[str]): Hypothesis strings.
        refs (list[str]): Reference strings, aligned with `hyps`.

    Returns:
        dict[str, float]: Corpus `bleu1`..`bleu4` with `brevity_penalty`/`hyp_len`/`ref_len`, the
            mean ROUGE F1s, corpus `wer`, mean `sentence_bleu4` and mean `content_f1`, plus `n`.

    Raises:
        ValueError: If `hyps` and `refs` differ in length, which would silently mis-pair the corpus.
    """
    if not hyps:
        return {'n': 0.0}
    out = bleu(hyps, refs)
    out.update(rouge(hyps, refs))
    out['wer'] = wer(hyps, refs)
    per = per_sentence_scores(hyps, refs)
    out['sentence_bleu4'] = float(np.mean(per['sentence_bleu4']))
    out['content_f1'] = float(np.nanmean(per['content_f1']))
    out['n'] = float(len(hyps))
    return {k: float(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Paired deltas against the brain-independent controls
# --------------------------------------------------------------------------- #


def paired_delta(
    hyp_scores: np.ndarray,
    control_scores: np.ndarray,
    *,
    metric: str = 'content_f1',
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap CI of the per-sentence `hypothesis - control` delta, signed so higher is better.

    The bootstrap resamples sentences, not conditions, so the pairing is preserved and the interval
    is on the delta itself rather than on the difference of two independent means.

    Args:
        hyp_scores (np.ndarray): Per-sentence scores of the real decode `(n,)`.
        control_scores (np.ndarray): Per-sentence scores of the control `(n,)`.
        metric (str, optional): Metric name, used only for the sign convention. Defaults to 'content_f1'.
        n_boot (int, optional): Bootstrap resamples. Defaults to 2000.
        seed (int, optional): Bootstrap seed. Defaults to 0.

    Returns:
        dict[str, Any]: `{'metric', 'point', 'lo', 'hi', 'n', 'n_boot', 'beats'}` where `beats` is the
            CI lower bound strictly above zero.
    """
    hyp = np.asarray(hyp_scores, dtype=np.float64).ravel()
    ctrl = np.asarray(control_scores, dtype=np.float64).ravel()
    if hyp.size == 0 or hyp.size != ctrl.size:
        return {
            'metric': metric,
            'point': float('nan'),
            'lo': float('nan'),
            'hi': float('nan'),
            'n': int(hyp.size),
            'n_boot': int(n_boot),
            'beats': False,
        }
    delta = hyp - ctrl if _HIGHER_IS_BETTER.get(metric, True) else ctrl - hyp
    keep = np.isfinite(delta)
    delta = delta[keep]
    point, lo, hi = bootstrap_ci(delta, n_boot=n_boot, seed=seed)
    return {
        'metric': metric,
        'point': float(point),
        'lo': float(lo),
        'hi': float(hi),
        'n': int(delta.size),
        'n_boot': int(n_boot),
        'beats': bool(np.isfinite(lo) and lo > 0.0),
    }


def pairwise_metric(hyps: list[str], refs: list[str], metric: str = 'content_f1') -> np.ndarray:
    """Score every hypothesis against every reference, for the permutation null.

    Computing the full matrix once means a permutation costs a fancy-index mean instead of `n` fresh
    string comparisons.

    Args:
        hyps (list[str]): Hypothesis strings `(n_h,)`.
        refs (list[str]): Reference strings `(n_r,)`.
        metric (str, optional): One of the `per_sentence_scores` keys. Defaults to 'content_f1'.

    Returns:
        np.ndarray: Scores `(n_h, n_r)`, float64.

    Raises:
        ValueError: When `metric` is not a known per-sentence metric.
    """
    if metric not in _SENTENCE_METRICS:
        raise ValueError(f'unknown per-sentence metric {metric!r}; expected one of {_SENTENCE_METRICS}')

    # The content-word set F1 factors into a boolean matrix product, which is what makes n = 1260 affordable.
    if metric == 'content_f1':
        h_sets = [content_words(h) for h in hyps]
        r_sets = [content_words(r) for r in refs]
        vocab = sorted({w for s in h_sets for w in s} | {w for s in r_sets for w in s})
        index = {w: i for i, w in enumerate(vocab)}
        h_mat = np.zeros((len(h_sets), len(vocab)), dtype=np.float64)
        r_mat = np.zeros((len(r_sets), len(vocab)), dtype=np.float64)
        for i, s in enumerate(h_sets):
            for w in s:
                h_mat[i, index[w]] = 1.0
        for i, s in enumerate(r_sets):
            for w in s:
                r_mat[i, index[w]] = 1.0
        overlap = h_mat @ r_mat.T
        h_size = h_mat.sum(axis=1)[:, None]
        r_size = r_mat.sum(axis=1)[None, :]
        denom = h_size + r_size
        with np.errstate(invalid='ignore', divide='ignore'):
            out = np.where(denom > 0, 2.0 * overlap / np.where(denom > 0, denom, 1.0), 1.0)
        return np.asarray(out, dtype=np.float64)

    tokens_h = [tokenise(h) for h in hyps]
    tokens_r = [tokenise(r) for r in refs]
    out = np.zeros((len(hyps), len(refs)), dtype=np.float64)
    for i, h_tok in enumerate(tokens_h):
        for j, r_tok in enumerate(tokens_r):
            out[i, j] = _pair_score(metric, hyps[i], refs[j], h_tok, r_tok)
    return out


def _pair_score(metric: str, hyp: str, ref: str, h_tok: list[str], r_tok: list[str]) -> float:
    """One hypothesis-reference score for the metrics that do not factor into a matrix product."""
    if metric == 'bleu1':
        return _unigram_precision(h_tok, r_tok)
    if metric == 'sentence_bleu4':
        return sentence_bleu(hyp, ref)
    if metric == 'rouge1':
        return _rouge_n_f1(h_tok, r_tok, 1)
    if metric == 'rouge2':
        return _rouge_n_f1(h_tok, r_tok, 2)
    if metric == 'rougeL':
        return _rouge_l_f1(h_tok, r_tok)
    return sentence_wer(hyp, ref)


# --------------------------------------------------------------------------- #
# Quarantine
# --------------------------------------------------------------------------- #


def quarantined_keys(block: Any) -> list[str]:
    """Every key in `block` (recursively) whose name ends in `_DIAGNOSTIC` or `_RETRIEVAL`.

    Teacher-forced perplexity and forced-choice accuracy are computed and reported because they are
    informative, but they are not free generation and must never enter a verdict.

    Args:
        block (Any): Any nested dict/list structure.

    Returns:
        list[str]: Sorted distinct quarantined key names.
    """
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and _QUARANTINE_RE.search(k):
                    found.add(k)
                else:
                    walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    walk(block)
    return sorted(found)


def strip_quarantined(block: Any) -> Any:
    """Returns `block` with every `*_DIAGNOSTIC` / `*_RETRIEVAL` key removed at any depth.

    Args:
        block (Any): Any nested dict/list structure.

    Returns:
        Any: A copy safe to hand to a verdict.
    """
    if isinstance(block, dict):
        return {
            k: strip_quarantined(v) for k, v in block.items() if not (isinstance(k, str) and _QUARANTINE_RE.search(k))
        }
    if isinstance(block, list):
        return [strip_quarantined(v) for v in block]
    if isinstance(block, tuple):
        return tuple(strip_quarantined(v) for v in block)
    return block


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def generation_report(
    hypotheses: list[str],
    references: list[str],
    controls: dict[str, list[str]],
    *,
    oracle: list[str] | None = None,
    prefix_kl: float | None = None,
    n_candidate_sentences: int | None = None,
    split: str | None = None,
    primary_metric: str = 'content_f1',
    n_boot: int = 2000,
    n_perm: int = 1000,
    max_rows: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Absolute scores, paired deltas against every control, and the permutation null in one block.

    No absolute score here is a result. The only readable numbers are the paired deltas and the
    permutation p: a decoder that ignores its conditioning vector and recites the corpus scores a high
    absolute BLEU and a delta of exactly zero against `mean_prefix`.

    Args:
        hypotheses (list[str]): Free-running decodes, one per held-out reading.
        references (list[str]): The true sentences, aligned with `hypotheses`.
        controls (dict[str, list[str]]): Brain-independent decodes through the identical path, keyed by
            control name; each list must align with `hypotheses`.
        oracle (list[str] | None, optional): Positive control -- the true text embedding through the same
            bridge and LM. Bounds the achievable score; it is never a delta. Defaults to None.
        prefix_kl (float | None, optional): Mean KL (nats) between a reading's own prefix and another
            reading's, which is 0 when the prompt does not depend on the brain. Below the configured
            floor the decoder is ignoring the brain. Defaults to None.
        n_candidate_sentences (int | None, optional): Size of any candidate set the decode was allowed to
            choose from. `None` means free generation; anything else is retrieval and is labelled so.
            Defaults to None.
        split (str | None, optional): The split these readings came from, carried so the verdict can
            refuse a headline on a split that shares stimuli between train and test. Defaults to None.
        primary_metric (str, optional): Metric the verdict reads. Defaults to 'content_f1'.
        n_boot (int, optional): Paired-bootstrap resamples. Defaults to 2000.
        n_perm (int, optional): Permutations for the null. Defaults to 1000.
        max_rows (int, optional): Per-sentence rows carried for the side-by-side artifact. Defaults to 200.
        seed (int, optional): Seed for the bootstrap and the permutation null. Defaults to 0.

    Returns:
        dict[str, Any]: `{'applicable', 'n', 'split', 'primary_metric', 'absolute', 'deltas',
            'controls_skipped', 'permutation', 'prefix_influence_kl', 'n_candidate_sentences',
            'controls_beaten', 'beats_all_controls', 'worst_control', 'worst_control_ci', 'rows'}`,
            or `{'applicable': False, 'reason'}`.
    """
    if len(hypotheses) != len(references):
        return {
            'applicable': False,
            'reason': f'hyps/refs length mismatch: {len(hypotheses)} vs {len(references)}',
        }
    n = len(hypotheses)
    if n < 4:
        return {'applicable': False, 'reason': 'need >= 4 held-out sentences'}
    if primary_metric not in _SENTENCE_METRICS:
        return {'applicable': False, 'reason': f'unknown primary metric {primary_metric!r}'}

    hyp_per = per_sentence_scores(hypotheses, references)
    absolute: dict[str, Any] = {'hypothesis': corpus_scores(hypotheses, references)}
    control_absolute: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    control_per: dict[str, dict[str, np.ndarray]] = {}

    for name, texts in controls.items():
        if len(texts) != n:
            skipped[name] = f'length {len(texts)} != {n}'
            continue
        control_absolute[name] = corpus_scores(texts, references)
        control_per[name] = per_sentence_scores(texts, references)
        deltas[name] = {
            m: paired_delta(hyp_per[m], control_per[name][m], metric=m, n_boot=n_boot, seed=seed)
            for m in _SENTENCE_METRICS
        }
    absolute['controls'] = control_absolute
    if oracle is not None and len(oracle) == n:
        absolute['oracle'] = corpus_scores(oracle, references)

    from zte.evaluation.audit.honesty import generation_permutation_test

    permutation = generation_permutation_test(hypotheses, references, metric=primary_metric, n_perm=n_perm, seed=seed)

    beaten = [name for name, d in deltas.items() if d[primary_metric]['beats']]
    worst = min(deltas, key=lambda k: deltas[k][primary_metric]['lo']) if deltas else None
    return {
        'applicable': True,
        'n': int(n),
        'split': split,
        'primary_metric': primary_metric,
        'absolute': absolute,
        'deltas': deltas,
        'controls_skipped': skipped,
        'permutation': permutation,
        'prefix_influence_kl': None if prefix_kl is None else float(prefix_kl),
        'n_candidate_sentences': n_candidate_sentences,
        'controls_beaten': beaten,
        'beats_all_controls': bool(deltas) and len(beaten) == len(deltas),
        'worst_control': worst,
        'worst_control_ci': deltas[worst][primary_metric] if worst is not None else None,
        'rows': _rows(hypotheses, references, controls, oracle, hyp_per, control_per, max_rows),
    }


def _rows(
    hypotheses: list[str],
    references: list[str],
    controls: dict[str, list[str]],
    oracle: list[str] | None,
    hyp_per: dict[str, np.ndarray],
    control_per: dict[str, dict[str, np.ndarray]],
    max_rows: int,
) -> list[dict[str, Any]]:
    """Builds the per-sentence side-by-side rows the interactive page renders."""
    limit = min(len(hypotheses), max(0, max_rows))
    rows: list[dict[str, Any]] = []
    for i in range(limit):
        row: dict[str, Any] = {
            'index': i,
            'reference': references[i],
            'hypothesis': hypotheses[i],
            'scores': {m: float(hyp_per[m][i]) for m in _SENTENCE_METRICS},
            'controls': {},
        }
        for name in control_per:
            row['controls'][name] = {
                'text': controls[name][i],
                'scores': {m: float(control_per[name][m][i]) for m in _SENTENCE_METRICS},
            }
        if oracle is not None and i < len(oracle):
            row['oracle'] = oracle[i]
        rows.append(row)
    return rows
