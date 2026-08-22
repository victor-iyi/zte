"""The length-confound audit: what held-out sentence retrieval is worth once word count is accounted for."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

type PieceSignature = Literal['profile', 'total', 'multiset', 'words']
"""Which sub-word statistic a piece oracle is allowed to resolve a sentence by."""

from zte.evaluation.audit.scoreboard import _binom_tail_p, _bootstrap_ci

# The 3 x 2 grid the audit reports; `transductive` is the published path and is included to be contrasted.
POSTPROCESS_CONDITIONS: tuple[str, ...] = ('none', 'train_fitted', 'transductive')
GALLERY_CONDITIONS: tuple[str, ...] = ('full', 'length_stratified')


# --------------------------------------------------------------------------- #
# Post-processing, fitted on the train split only
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class PostProcess:
    """A whitening + all-but-the-top transform with its parameters frozen at fit time.

    Holding the parameters rather than refitting per call is the whole point: the transform applied to
    a held-out row is a function of the training rows alone, so a decoder that sees one sentence at a
    time could reproduce it.
    """

    mean: np.ndarray
    inv_sqrt: np.ndarray | None
    directions: np.ndarray | None
    n_train: int

    def __call__(self, emb: np.ndarray) -> np.ndarray:
        """Applies the fitted transform to `(n, d)` rows, returning float32."""
        x = np.asarray(emb, dtype=np.float64) - self.mean
        if self.inv_sqrt is not None:
            x = x @ self.inv_sqrt
        if self.directions is not None and len(self.directions):
            x = x - (x @ self.directions.T) @ self.directions
        return x.astype(np.float32)


def fit_postprocess(train_rows: np.ndarray, *, whiten: bool = True, n_top: int = 1, eps: float = 1e-3) -> PostProcess:
    """Fits ZCA whitening then all-but-the-top on the TRAIN rows only.

    Handed no training rows, `report._postprocess` fits `whiten_features` and `all_but_the_top` on the
    union of every subject's embeddings -- the held-out subject included -- and that rebound array is
    what reaches `build_scoreboard`. Those transforms are label-free, so it is a soft leak rather than
    label leakage, but a decoder scoring one sentence cannot reproduce them. This is the non-transductive
    counterpart `report._postprocess` calls instead when it has training rows: fit here, apply anywhere,
    parameters never move.

    Args:
        train_rows (np.ndarray): Training-split embeddings `(n_train, d)`.
        whiten (bool, optional): Apply ZCA whitening. Defaults to True.
        n_top (int, optional): Leading principal directions removed after whitening. Defaults to 1.
        eps (float, optional): Eigenvalue floor for the whitening inverse square root. Defaults to 1e-3.

    Returns:
        PostProcess: The frozen transform, callable on any `(n, d)` array.
    """
    x = np.asarray(train_rows, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2:
        dim = x.shape[1] if x.ndim == 2 else 1
        return PostProcess(np.zeros(dim), None, None, int(len(x)))

    mean = x.mean(axis=0)
    centred = x - mean
    inv_sqrt: np.ndarray | None = None
    if whiten:
        cov = (centred.T @ centred) / (len(centred) - 1)
        vals, vecs = np.linalg.eigh(cov)
        inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.clip(vals, eps, None))) @ vecs.T
        centred = centred @ inv_sqrt

    directions: np.ndarray | None = None
    if n_top > 0:
        # ABTT runs in the whitened frame, matching the whiten-then-ABTT order `report._postprocess` applies.
        _, _, vt = np.linalg.svd(centred - centred.mean(axis=0, keepdims=True), full_matrices=False)
        directions = vt[: min(n_top, vt.shape[0])]
    return PostProcess(mean, inv_sqrt, directions, int(len(x)))


# --------------------------------------------------------------------------- #
# The floor: what sentence length alone already buys
# --------------------------------------------------------------------------- #


def length_oracle(lengths: np.ndarray, *, tol: int = 0, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    """Retrieval scores achievable from word count alone -- the floor every encoder number sits on.

    The oracle knows each query's word count to within `tol` and nothing else, so it ranks the
    matching-length stratum in a uniformly random order and everything else behind it. The scores are
    exact expectations over that ordering rather than a simulation, so there is no seed and no Monte
    Carlo error. Word count is not a modelling choice on ZuCo: the eye-tracking word segmentation hands
    it to the model for free through the pad mask width.

    Args:
        lengths (np.ndarray): Word count per gallery item `(n,)`, one row per distinct sentence.
        tol (int, optional): Word-count tolerance the oracle resolves. Defaults to 0 (exact length).
        ks (tuple[int, ...], optional): Top-K cut-offs. Defaults to (1, 5, 10).

    Returns:
        dict[str, float]: `top{k}`, `mrr`, `rank_percentile`, `chance_top1`, `mean_stratum`,
            `median_stratum`, `min_stratum`, `max_stratum`, `tol` and `n`.
    """
    arr = np.asarray(lengths, dtype=np.float64).ravel()
    n = arr.size
    out: dict[str, float] = {'tol': float(tol), 'n': float(n)}
    if n == 0:
        return out | {f'top{k}': float('nan') for k in ks}

    # Stratum size per query: how many gallery items share its length to within the tolerance.
    strata = np.array([int(np.sum(np.abs(arr - value) <= tol)) for value in arr], dtype=np.float64)
    strata = np.clip(strata, 1.0, None)

    for k in ks:
        out[f'top{k}'] = float(np.mean(np.minimum(k, strata) / strata))

    # Expected reciprocal rank of a uniform position inside the stratum is the harmonic number over its size.
    max_m = int(strata.max())
    harmonic = np.concatenate(([0.0], np.cumsum(1.0 / np.arange(1, max_m + 1))))
    out['mrr'] = float(np.mean(harmonic[strata.astype(int)] / strata))

    expected_rank = (strata + 1.0) / 2.0
    out['rank_percentile'] = float(np.mean(1.0 - (expected_rank - 1.0) / max(n - 1, 1)))
    out['mean_rank'] = float(np.mean(expected_rank))
    out['chance_top1'] = float(1.0 / n)
    out['mean_stratum'] = float(strata.mean())
    out['median_stratum'] = float(np.median(strata))
    out['min_stratum'] = float(strata.min())
    out['max_stratum'] = float(strata.max())
    return out


def signature_oracle(
    signatures: Sequence[Hashable], *, name: str = 'signature', ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """Retrieval scores achievable from a discrete per-sentence signature alone, and the bits it gives away.

    The oracle resolves each query to the set of gallery items carrying an identical signature, ranks that stratum
    uniformly at random and everything else behind it. Scores are exact expectations over that ordering, so there is
    no seed and no Monte Carlo error -- the same construction `length_oracle` uses for word count.

    Note:
        This is the generalisation `length_oracle` could not express. Word count is one integer per sentence; a
        sub-word piece profile is a vector of them, and on a 700-sentence gallery it resolves nearly every sentence
        uniquely. Any representation that is handed such a signature is not being measured on the brain, so a
        headline computed beside a signature it had access to has to be read against this floor first.

    Args:
        signatures (Sequence[Hashable]): One hashable signature per gallery sentence.
        name (str, optional): Label carried into the returned block. Defaults to 'signature'.
        ks (tuple[int, ...], optional): Top-K cut-offs. Defaults to (1, 5, 10).

    Returns:
        dict[str, float]: `top{k}`, `mrr`, `rank_percentile`, `chance_top1`, `mean_stratum`, `median_stratum`,
            `unique_fraction`, `entropy_bits`, `conditional_entropy_bits`, `information_bits`, `n` and `signature`.
    """
    counts = Counter(signatures)
    n = len(list(signatures))
    out: dict[str, Any] = {'signature': name, 'n': float(n)}
    if n == 0:
        return out | {f'top{k}': float('nan') for k in ks}

    strata = np.array([counts[s] for s in signatures], dtype=np.float64)
    for k in ks:
        out[f'top{k}'] = float(np.mean(np.minimum(k, strata) / strata))

    max_m = int(strata.max())
    harmonic = np.concatenate(([0.0], np.cumsum(1.0 / np.arange(1, max_m + 1))))
    out['mrr'] = float(np.mean(harmonic[strata.astype(int)] / strata))

    expected_rank = (strata + 1.0) / 2.0
    out['rank_percentile'] = float(np.mean(1.0 - (expected_rank - 1.0) / max(n - 1, 1)))
    out['mean_rank'] = float(np.mean(expected_rank))
    out['chance_top1'] = float(1.0 / n)
    out['mean_stratum'] = float(strata.mean())
    out['median_stratum'] = float(np.median(strata))
    out['unique_fraction'] = float(np.mean(strata == 1.0))

    # H(identity) - H(identity | signature): what the signature alone tells you about which sentence this is.
    identity_bits = float(np.log2(n))
    conditional = float(np.mean(np.log2(strata)))
    out['entropy_bits'] = identity_bits
    out['conditional_entropy_bits'] = conditional
    out['information_bits'] = identity_bits - conditional
    return out


def piece_signatures(word_pieces: np.ndarray, kind: PieceSignature = 'profile') -> list[Hashable]:
    """Turns a `(n_text, max_words)` sub-word-count table into one hashable signature per sentence.

    Args:
        word_pieces (np.ndarray): `TokenAlignment.word_pieces`; `0` past the end of a sentence.
        kind (PieceSignature, optional): `'profile'` is the ordered per-word vector, `'total'` the summed piece
            count, `'multiset'` the counts with their order destroyed, `'words'` the word count alone.
            Defaults to 'profile'.

    Returns:
        list[Hashable]: One signature per row.

    Raises:
        ValueError: If `kind` is not one of the four supported signatures.
    """
    table = np.asarray(word_pieces, dtype=np.int64)
    rows = [row[row > 0] for row in table]
    match kind:
        case 'profile':
            return [tuple(int(v) for v in row) for row in rows]
        case 'total':
            return [int(row.sum()) for row in rows]
        case 'multiset':
            return [tuple(sorted(int(v) for v in row)) for row in rows]
        case 'words':
            return [int(row.size) for row in rows]
        case unknown:
            raise ValueError(f'Unknown piece signature {unknown!r}; expected profile, total, multiset or words.')


def piece_profile_report(
    word_pieces: np.ndarray,
    *,
    observed_top1: float | None = None,
    gate_signature: PieceSignature = 'total',
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any]:
    """Scores every sub-word piece signature as a retrieval oracle, and says whether a number clears them.

    Note:
        A token-level objective that gives a word as many EEG sub-tokens as the reference spells it word-pieces
        hands the model this signature. On a 700-sentence gallery the ordered profile resolves almost every
        sentence uniquely, so it is a larger channel than the sentence's own identity entropy leaves room for --
        which makes it the floor a token-level headline is measured against, exactly as word count is for the
        sentence level.

    Note:
        The **ordered profile is the ceiling, not the gate**. It is what a design would give away that sized a
        word's EEG by how many pieces its reference spells it in, and on a real gallery it resolves 99.6% of
        sentences -- so gating on it would print `below the floor` whatever the encoder did, which is a column
        carrying no information rather than a check. This encoder is fixed-K, so what it can actually reach is
        the *total* piece count, through the length channel; that is the default gate, and every signature's
        verdict is reported beside it so the choice is visible rather than assumed.

    Args:
        word_pieces (np.ndarray): `TokenAlignment.word_pieces`, `(n_text, max_words)`.
        observed_top1 (float | None, optional): A run's held-out Top-1, compared against every oracle.
            Defaults to None.
        gate_signature (PieceSignature, optional): Which oracle decides `beats_oracles`. Defaults to 'total'.
        ks (tuple[int, ...], optional): Top-K cut-offs. Defaults to (1, 5, 10).

    Returns:
        dict[str, Any]: `oracles` and `clears` per signature, the `gate_*` and `ceiling_*` blocks,
            `observed_top1`, `beats_oracles` and `verdict`.
    """
    kinds: tuple[PieceSignature, ...] = ('words', 'total', 'multiset', 'profile')
    oracles = {kind: signature_oracle(piece_signatures(word_pieces, kind), name=kind, ks=ks) for kind in kinds}
    observed = None if observed_top1 is None else float(observed_top1)
    clears = {
        kind: None if observed is None else bool(observed > float(block['top1'])) for kind, block in oracles.items()
    }

    gate = oracles[gate_signature]
    beats = clears[gate_signature]
    verdict = (
        'not measured'
        if beats is None
        else (
            f'clears the {gate_signature} piece oracle'
            if beats
            else f'BELOW the {gate_signature} piece oracle -- not evidence of decoding'
        )
    )
    return {
        'oracles': oracles,
        'clears': clears,
        'gate_signature': gate_signature,
        'gate_top1': float(gate['top1']),
        'gate_bits': float(gate['information_bits']),
        'ceiling_signature': 'profile',
        'ceiling_top1': float(oracles['profile']['top1']),
        'worst_case_top1': max(float(block['top1']) for block in oracles.values()),
        'worst_case_signature': max(oracles, key=lambda k: float(oracles[k]['top1'])),
        'observed_top1': observed,
        'beats_oracles': beats,
        'verdict': verdict,
    }


def bit_budget(
    n_words: np.ndarray, *, mean_rank: float | None = None, n_gallery: int | None = None
) -> dict[str, float]:
    """How many bits of sentence identity are needed, how many length gives away, how many the encoder adds.

    Args:
        n_words (np.ndarray): Word count per distinct gallery sentence `(n,)`.
        mean_rank (float | None, optional): Mean 1-based rank of the correct sentence under the encoder,
            from which `bits_from_eeg = log2(n / mean_rank)`. Defaults to None.
        n_gallery (int | None, optional): Gallery size. Defaults to `len(n_words)`.

    Returns:
        dict[str, float]: `bits_needed`, `bits_from_length`, `bits_from_eeg`, `ratio`,
            `entropy_identity`, `entropy_identity_given_length`, `n_gallery`.
    """
    arr = np.asarray(n_words, dtype=np.float64).ravel()
    n = int(n_gallery if n_gallery is not None else arr.size)
    if n <= 1:
        return {'bits_needed': float('nan'), 'n_gallery': float(n)}

    bits_needed = float(np.log2(n))
    _, counts = np.unique(arr, return_counts=True)
    conditional = float(np.sum((counts / arr.size) * np.log2(counts))) if arr.size else float('nan')
    bits_from_eeg = (
        float(np.log2(n / mean_rank))
        if (mean_rank is not None and np.isfinite(mean_rank) and mean_rank > 0)
        else float('nan')
    )
    return {
        'bits_needed': bits_needed,
        'entropy_identity': bits_needed,
        'entropy_identity_given_length': conditional,
        'bits_from_length': bits_needed - conditional,
        'bits_from_eeg': bits_from_eeg,
        'ratio': bits_from_eeg / bits_needed if np.isfinite(bits_from_eeg) else float('nan'),
        'n_gallery': float(n),
    }


# --------------------------------------------------------------------------- #
# Retrieval inside a matched-length gallery
# --------------------------------------------------------------------------- #


def stratified_retrieval(
    sent_emb: np.ndarray,
    content_ids: np.ndarray,
    subjects: np.ndarray,
    holdout: str,
    n_words: np.ndarray | None = None,
    *,
    length_tol: int = 1,
    ks: tuple[int, ...] = (1, 5, 10),
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Held-out cross-subject retrieval with the gallery restricted to a matched word count.

    Same contract as `scoreboard.cross_subject_holdout_retrieval` -- rank percentile, exact binomial
    tail and a bootstrap CI -- but every distractor has the query's length, so a hit cannot be a
    sentence-length shortcut. Passing `n_words=None` removes the restriction and reproduces the full
    700-item gallery through the identical code path, which is what makes the two grid columns
    comparable.

    Args:
        sent_emb (np.ndarray): Sentence embeddings `(n, d)`.
        content_ids (np.ndarray): Stimulus id per reading `(n,)`.
        subjects (np.ndarray): Subject code per reading `(n,)`.
        holdout (str): The held-out subject code, whose readings are the queries.
        n_words (np.ndarray | None, optional): Word count per reading `(n,)`. `None` disables the
            length restriction. Defaults to None.
        length_tol (int, optional): Word-count tolerance defining the gallery. Defaults to 1.
        ks (tuple[int, ...], optional): Top-K cut-offs. Defaults to (1, 5, 10).
        n_boot (int, optional): Bootstrap resamples behind `rank_percentile_ci`. Defaults to 2000.
        seed (int, optional): Bootstrap seed. Defaults to 0.

    Returns:
        dict | None: `top{k}`, `top{k}_p`, `mrr`, `rank_percentile`, `rank_percentile_ci`,
            `mean_rank`, `median_rank`, `chance_top1`, `mean_gallery`, `n_queries`,
            `excluded_no_positive`, `headline_metric`; `None` when there are too few subjects or queries.
    """
    subjects = np.asarray(subjects)
    content_ids = np.asarray(content_ids)
    if len(np.unique(subjects)) < 2:
        return None
    q_mask = subjects == holdout
    if int(q_mask.sum()) < 2:
        return None

    emb = np.asarray(sent_emb, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    lengths = None if n_words is None else np.asarray(n_words, dtype=np.float64).ravel()

    q_idx = np.where(q_mask)[0]
    sims = emb[q_idx] @ emb.T  # (n_queries, n) -- never the full n x n matrix

    hits = {k: 0.0 for k in ks}
    reciprocal = 0.0
    chances: list[float] = []
    percentiles: list[float] = []
    ranks: list[float] = []
    galleries: list[float] = []
    excluded = 0

    for row, i in enumerate(q_idx):
        cross = subjects != subjects[i]
        if lengths is not None:
            cross = cross & (np.abs(lengths - lengths[i]) <= length_tol)
        # A query whose truth cannot appear in its own gallery is unanswerable, not wrong: excluded and counted,
        # never zero-scored -- zero-scoring once manufactured a below-chance rank percentile out of stratum misses.
        if not cross.any():
            excluded += 1
            continue

        cand = np.where(cross)[0]
        same_mask = content_ids[cand] == content_ids[i]
        if not same_mask.any():
            excluded += 1
            continue

        galleries.append(float(cand.size))
        chances.append(float(same_mask.mean()))
        order = cand[np.argsort(-sims[row, cand])]
        same = content_ids[order] == content_ids[i]

        for k in ks:
            hits[k] += float(same[:k].any())
        rank = int(np.argmax(same)) + 1
        reciprocal += 1.0 / rank
        ranks.append(float(rank))
        percentiles.append(1.0 - (rank - 1) / max(cand.size - 1, 1))

    n_scored = len(percentiles)
    if n_scored == 0:
        return None

    out: dict[str, Any] = {f'top{k}': hits[k] / n_scored for k in ks}
    out['mrr'] = reciprocal / n_scored
    out['chance_top1'] = float(np.mean(chances)) if chances else float('nan')
    out['rank_percentile'] = float(np.mean(percentiles))
    out['mean_rank'] = float(np.mean(ranks))
    out['median_rank'] = float(np.median(ranks))
    out['mean_gallery'] = float(np.mean(galleries)) if galleries else float('nan')
    out['median_gallery'] = float(np.median(galleries)) if galleries else float('nan')
    out['n_queries'] = int(n_scored)
    out['excluded_no_positive'] = int(excluded)
    out['length_tol'] = None if lengths is None else int(length_tol)
    for k in ks:
        out[f'top{k}_p'] = _binom_tail_p(round(out[f'top{k}'] * n_scored), n_scored, out['chance_top1'] * k)
    out['rank_percentile_ci'] = _bootstrap_ci(np.asarray(percentiles, dtype=np.float64), n_boot=n_boot, seed=seed)
    out['headline_metric'] = 'rank_percentile'
    return out


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def rebaseline_report(
    sent_emb: np.ndarray,
    content_ids: np.ndarray,
    subjects: np.ndarray,
    holdout: str,
    n_words: np.ndarray,
    *,
    tasks: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    length_tol: int = 1,
    oracle_tols: tuple[int, ...] = (0, 1, 2, 4),
    whiten: bool = True,
    n_top: int = 1,
    ks: tuple[int, ...] = (1, 5, 10),
    menu_ks: tuple[int, ...] | None = None,
    menu_target: float = 0.8,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """The full audit: 3 post-processing conditions x 2 galleries, the length-oracle floor, the bit budget.

    This is a diagnostic. It never raises and never refuses: a cell that cannot be computed is reported
    as `None` with its reason, and the floor comparison is recorded as a number, not enforced.

    Args:
        sent_emb (np.ndarray): Sentence embeddings `(n, d)`, before any post-processing.
        content_ids (np.ndarray): Stimulus id per reading `(n,)`.
        subjects (np.ndarray): Subject code per reading `(n,)`.
        holdout (str): The held-out subject code.
        n_words (np.ndarray): Word count per reading `(n,)`.
        tasks (np.ndarray | None, optional): Task label per reading `(n,)`, enabling the menu audit's
            task-matched headline flavor. Defaults to None.
        train_mask (np.ndarray | None, optional): Boolean `(n,)` marking the rows the post-processing may
            be fitted on. Defaults to every non-holdout row.
        length_tol (int, optional): Word-count tolerance for the stratified gallery. Defaults to 1.
        oracle_tols (tuple[int, ...], optional): Tolerances for the length-only floor. Defaults to (0, 1, 2, 4).
        whiten (bool, optional): Whether the fitted conditions whiten. Defaults to True.
        n_top (int, optional): Leading directions removed by all-but-the-top. Defaults to 1.
        ks (tuple[int, ...], optional): Top-K cut-offs. Defaults to (1, 5, 10).
        menu_ks (tuple[int, ...] | None, optional): Menu sizes for the closed-set capacity audit.
            Defaults to None, which uses `menu.DEFAULT_MENU_KS`.
        menu_target (float, optional): Accuracy a menu size must clear to be certified. Defaults to 0.8.
        n_boot (int, optional): Bootstrap resamples behind every cell's interval. Defaults to 2000.
        seed (int, optional): Bootstrap seed. Defaults to 0.

    Returns:
        dict[str, Any]: `{'holdout', 'n_readings', 'n_stimuli', 'length_tol', 'grid', 'length_oracle',
            'bit_budget', 'floor_comparison', 'menu', 'errors'}`.
    """
    emb = np.asarray(sent_emb, dtype=np.float32)
    content_ids = np.asarray(content_ids)
    subjects = np.asarray(subjects)
    lengths = np.asarray(n_words, dtype=np.float64).ravel()
    mask = np.asarray(subjects != holdout) if train_mask is None else np.asarray(train_mask, dtype=bool)
    errors: dict[str, str] = {}

    # One length per distinct stimulus, so the oracle and the bit budget describe the gallery, not the readings.
    order = np.argsort(content_ids, kind='stable')
    uniq_ids, first = np.unique(content_ids[order], return_index=True)
    stimulus_lengths = lengths[order][first]

    variants: dict[str, np.ndarray] = {'none': emb}
    for name, fit_rows in (('train_fitted', emb[mask]), ('transductive', emb)):
        try:
            variants[name] = fit_postprocess(fit_rows, whiten=whiten, n_top=n_top)(emb)
        except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover - defensive
            errors[name] = f'{type(exc).__name__}: {exc}'

    grid: dict[str, Any] = {}
    for cond in POSTPROCESS_CONDITIONS:
        grid[cond] = {gallery: None for gallery in GALLERY_CONDITIONS}
        if cond not in variants:
            continue
        for gallery, words in (('full', None), ('length_stratified', lengths)):
            try:
                grid[cond][gallery] = stratified_retrieval(
                    variants[cond],
                    content_ids,
                    subjects,
                    holdout,
                    words,
                    length_tol=length_tol,
                    ks=ks,
                    n_boot=n_boot,
                    seed=seed,
                )
            except (ValueError, IndexError, MemoryError) as exc:  # pragma: no cover - defensive
                errors[f'{cond}/{gallery}'] = f'{type(exc).__name__}: {exc}'

    oracle = {str(tol): length_oracle(stimulus_lengths, tol=tol, ks=ks) for tol in oracle_tols}
    honest = (grid.get('train_fitted') or {}).get('full') or {}
    budget = bit_budget(stimulus_lengths, mean_rank=honest.get('mean_rank'), n_gallery=int(uniq_ids.size))

    # Deferred import: `menu` reuses this module's fitted post-processing, so a top-level import would cycle.
    from zte.evaluation.audit.menu import DEFAULT_MENU_KS, menu_report

    menu: dict[str, Any] | None = None
    try:
        menu = menu_report(
            emb,
            content_ids,
            subjects,
            holdout,
            lengths,
            tasks=tasks,
            train_mask=mask,
            ks=menu_ks if menu_ks is not None else DEFAULT_MENU_KS,
            target=menu_target,
            whiten=whiten,
            n_top=n_top,
            n_boot=n_boot,
            seed=seed,
        )
    except (ValueError, np.linalg.LinAlgError, MemoryError) as exc:  # pragma: no cover - defensive
        errors['menu'] = f'{type(exc).__name__}: {exc}'

    return {
        'holdout': str(holdout),
        'n_readings': int(len(emb)),
        'n_stimuli': int(uniq_ids.size),
        'n_train_rows': int(mask.sum()),
        'length_tol': int(length_tol),
        'grid': grid,
        'length_oracle': oracle,
        'bit_budget': budget,
        'floor_comparison': _floor_comparison(grid, oracle, length_tol),
        'menu': menu,
        'errors': errors,
    }


def _floor_comparison(grid: dict[str, Any], oracle: dict[str, Any], length_tol: int) -> dict[str, Any]:
    """Puts the honest cell next to the matched length-only oracle; reported, never enforced."""
    cell = (grid.get('train_fitted') or {}).get('length_stratified') or {}
    floor = oracle.get(str(length_tol)) or {}
    ci = cell.get('rank_percentile_ci') or (float('nan'),) * 3
    encoder_lo = float(ci[1])
    floor_value = float(floor.get('rank_percentile', float('nan')))
    return {
        'condition': 'train_fitted',
        'gallery': 'length_stratified',
        'metric': 'rank_percentile',
        'encoder': cell.get('rank_percentile'),
        'encoder_ci_low': encoder_lo,
        'oracle_tol': int(length_tol),
        'oracle': floor_value if np.isfinite(floor_value) else None,
        'clears_floor': bool(np.isfinite(encoder_lo) and np.isfinite(floor_value) and encoder_lo > floor_value),
        'note': 'Diagnostic only: this comparison is reported and gates nothing.',
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Renders the audit as the Markdown block that ships beside `rebaseline.json`.

    Args:
        report (dict[str, Any]): The dict from `rebaseline_report`.

    Returns:
        str: A Markdown document with the 3 x 2 grid, the oracle floor and the bit budget.
    """
    budget = report.get('bit_budget') or {}
    lines = [
        f'# Length-confound audit -- held-out subject `{report.get("holdout")}`',
        '',
        f'{report.get("n_readings", 0)} readings over {report.get("n_stimuli", 0)} distinct '
        f'sentences; post-processing fitted on {report.get("n_train_rows", 0)} training rows.',
        '',
        '## Held-out retrieval: post-processing x gallery',
        '',
        '`transductive` fits whitening and all-but-the-top on every subject including the held-out '
        'one, which is the published path; `train_fitted` fits on the training rows alone, which is '
        'the only version a decoder could reproduce.',
        '',
        '| post-processing | gallery | Top-1 | Top-5 | rank percentile (95% CI) | mean gallery | n |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for cond in POSTPROCESS_CONDITIONS:
        for gallery in GALLERY_CONDITIONS:
            cell = (report.get('grid') or {}).get(cond, {}).get(gallery)
            if not cell:
                lines.append(f'| {cond} | {gallery} | — | — | — | — | — |')
                continue
            ci = cell.get('rank_percentile_ci') or (float('nan'),) * 3
            lines.append(
                f'| {cond} | {gallery} | {cell.get("top1", float("nan")):.4f} '
                f'| {cell.get("top5", float("nan")):.4f} '
                f'| {ci[0]:.4f} ({ci[1]:.4f}–{ci[2]:.4f}) '
                f'| {cell.get("mean_gallery", float("nan")):.1f} | {cell.get("n_queries", 0)} |'
            )

    lines += [
        '',
        '## Length-only oracle -- the floor',
        '',
        'An oracle that knows the word count to within the tolerance and nothing else, ranking the '
        'matching-length stratum in random order. Word count is free on ZuCo: the eye-tracking word '
        'segmentation sets it.',
        '',
        '| tolerance | Top-1 | Top-5 | Top-10 | MRR | rank percentile | mean stratum |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for tol, row in (report.get('length_oracle') or {}).items():
        lines.append(
            f'| ±{tol} | {row.get("top1", float("nan")):.4f} | {row.get("top5", float("nan")):.4f} '
            f'| {row.get("top10", float("nan")):.4f} | {row.get("mrr", float("nan")):.4f} '
            f'| {row.get("rank_percentile", float("nan")):.4f} '
            f'| {row.get("mean_stratum", float("nan")):.1f} |'
        )

    floor = report.get('floor_comparison') or {}
    lines += [
        '',
        '## Bit budget',
        '',
        f'- Sentence identity needs **{budget.get("bits_needed", float("nan")):.4f}** bits '
        f'over {int(budget.get("n_gallery", 0))} sentences.',
        f'- Word count alone carries **{budget.get("bits_from_length", float("nan")):.4f}** bits '
        f'(H(identity | n_words) = {budget.get("entropy_identity_given_length", float("nan")):.4f}).',
        f'- The encoder carries **{budget.get("bits_from_eeg", float("nan")):.4f}** bits '
        f'(train-fitted, full gallery), a ratio of {budget.get("ratio", float("nan")):.4f}.',
        '',
        f'Honest cell rank percentile {_fmt(floor.get("encoder"))} '
        f'(CI low {_fmt(floor.get("encoder_ci_low"))}) against the ±{floor.get("oracle_tol")} '
        f'length oracle at {_fmt(floor.get("oracle"))} -- clears floor: '
        f'**{floor.get("clears_floor")}**. {floor.get("note", "")}',
        '',
    ]
    menu = report.get('menu')
    if menu:
        from zte.evaluation.audit.menu import menu_markdown_lines

        lines += menu_markdown_lines(menu)
    errors = report.get('errors') or {}
    if errors:
        lines += ['## Cells that could not be computed', '']
        lines += [f'- `{k}`: {v}' for k, v in errors.items()]
        lines.append('')
    return '\n'.join(lines)


def _fmt(value: Any) -> str:
    """Formats a score, or a dash when missing."""
    if value is None or not np.isfinite(float(value)):
        return '—'
    return f'{float(value):.4f}'
