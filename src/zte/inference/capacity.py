"""One full-gallery LM pass, sliced into menus, and the conditioning arms the capacity audit compares."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
import pandas as pd

from zte.inference.decode import ReadingBatch, ZTEDecoder, paired_shuffle
from zte.logging_utils import get_logger

_LOG = get_logger('inference.capacity')

SCORE_FAMILIES: Final[tuple[str, ...]] = ('raw', 'pmi')
"""Score families the arms can be reported in -- `pmi` is the headline."""

# Exact match, not the tol 1 the generation controls pair at: the certified menus are built at exact
# stimulus-level word count, so a partner one word away would carry a length the menu never contained.
PARTNER_TOL: Final[int] = 0
"""Word-count tolerance of the `length_only` prefix and of the `mismatch` partner."""


@dataclass(slots=True, frozen=True, kw_only=True)
class GalleryScores:
    """Every gallery sentence scored under every query, plus the query-independent null-prefix score.

    Note:
        Scoring is per-(query, candidate) independent -- nothing in the rescoring path normalises across
        candidates or across queries -- so a K-way menu is a column slice of this matrix, not a rescoring
        run of its own. That is what makes a full capacity sweep cost one gallery pass instead of one per menu.

    Attributes:
        texts (list[str]): The gallery, in the order the columns follow.
        raw (np.ndarray): `(n_query, n_gallery)` length-normalised sequence log-probabilities.
        null (np.ndarray): `(n_gallery,)` scores under the learned null prefix; no query enters them.
        gallery_n_words (np.ndarray): Word count per gallery sentence `(n_gallery,)`.
        gallery_tasks (np.ndarray | None): Task label per gallery sentence, when the split carries one.
        evidence_content (bool): Whether the word-level EEG evidence path was left on for the model arm.
    """

    texts: list[str]
    raw: np.ndarray
    null: np.ndarray
    gallery_n_words: np.ndarray
    gallery_tasks: np.ndarray | None = None
    evidence_content: bool = True

    @property
    def pmi(self) -> np.ndarray:
        """Raw score minus each candidate's null-prefix score, cancelling candidate-side familiarity bias."""
        return self.raw - self.null[None, :]


def gallery_scores(
    decoder: ZTEDecoder,
    readings: ReadingBatch,
    texts: Sequence[str],
    *,
    gallery_n_words: np.ndarray,
    gallery_tasks: np.ndarray | None = None,
    batch_size: int = 8,
    evidence_content: bool = True,
) -> GalleryScores:
    """Scores the whole gallery under every reading once, so every menu is a slice rather than a pass.

    Note:
        `evidence_content` must be `not decoder.uses_evidence`. `ZTEDecoder.rescore` always runs the
        word-synchronous evidence path when the checkpoint has one, and a control built from a bare
        conditioning vector has no words to run it on -- so leaving it on would let the model arm beat
        every control through per-word EEG the controls could never receive. `False` strips the word
        tensors for *all* arms, the model's included, and the choice travels in the returned bundle.

    Args:
        decoder (ZTEDecoder): The decoder whose bridge, LM and scaffold every arm goes through.
        readings (ReadingBatch): The query conditioning bundle.
        texts (Sequence[str]): The gallery, in the order the returned columns follow.
        gallery_n_words (np.ndarray): Word count per gallery sentence `(n_gallery,)`.
        gallery_tasks (np.ndarray | None, optional): Task label per gallery sentence. Defaults to None.
        batch_size (int, optional): Queries per LM pass. Defaults to 8.
        evidence_content (bool, optional): Keep the word-level evidence path. Defaults to True.

    Returns:
        GalleryScores: The raw score matrix, the null column and the provenance of the evidence choice.

    Raises:
        ValueError: If `gallery_n_words` or `gallery_tasks` does not have one entry per gallery sentence.
    """
    gallery = [str(t) for t in texts]
    n_words = np.asarray(gallery_n_words).ravel()
    if n_words.size != len(gallery):
        raise ValueError(f'gallery_n_words/texts length mismatch: {n_words.size} vs {len(gallery)}.')

    tasks = None if gallery_tasks is None else np.asarray(gallery_tasks).ravel()
    if tasks is not None and tasks.size != len(gallery):
        raise ValueError(f'gallery_tasks/texts length mismatch: {tasks.size} vs {len(gallery)}.')

    scored = readings if evidence_content else _without_evidence(readings)
    raw = decoder.rescore(scored, gallery, batch_size=batch_size, pmi=False)

    return GalleryScores(
        texts=gallery,
        raw=raw,
        null=decoder.null_rescore(gallery),
        gallery_n_words=n_words,
        gallery_tasks=tasks,
        evidence_content=evidence_content,
    )


def capacity_arms(
    bundle: GalleryScores,
    decoder: ZTEDecoder,
    readings: ReadingBatch,
    *,
    query_n_words: np.ndarray,
    query_content_ids: np.ndarray,
    score: str,
    train: ReadingBatch | None = None,
    seed: int = 0,
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """Builds the conditioning arms the capacity audit certifies against, all on the scored gallery.

    Every arm reaches the frozen LM through the identical bridge, scaffold and length normalisation; only the
    conditioning changes. `shuffled_eeg` and `mismatch` are row permutations of the model's own matrix, so they
    cost nothing, and `null_prefix` is the bundle's null column broadcast over the queries -- which makes it
    identically zero under PMI, so it is reported under `raw` alone. `length_only` is the one arm needing its
    own pass: a training prefix matched on word count and nothing else, deduplicated over distinct counts
    because the matched vector is a function of the count alone. An arm whose ingredients are missing is
    omitted rather than approximated, and its certification clause then fails.

    Args:
        bundle (GalleryScores): The scored gallery from `gallery_scores`.
        decoder (ZTEDecoder): The decoder that produced the bundle.
        readings (ReadingBatch): The query conditioning bundle.
        query_n_words (np.ndarray): Word count per query `(n_query,)`.
        query_content_ids (np.ndarray): Stimulus id per query `(n_query,)`.
        score (str): `'raw'` or `'pmi'`.
        train (ReadingBatch | None, optional): Training-split conditioning, whose `meta` carries `n_words`;
            it is what the `length_only` prefix is averaged from. Defaults to None, which omits that arm.
        seed (int, optional): Seed for the derangement and the mismatch pairing. Defaults to 0.
        batch_size (int, optional): Queries per LM pass for the `length_only` arm. Defaults to 8.

    Returns:
        dict[str, np.ndarray]: Arm name to `(n_query, n_gallery)` scores in the requested family.

    Raises:
        ValueError: If `score` names no family, if the query arrays disagree with the score matrix, or if the
            model arm kept an evidence path no control can have.
    """
    if score not in SCORE_FAMILIES:
        raise ValueError(f'score={score!r} is not one of {SCORE_FAMILIES}.')
    if decoder.uses_evidence and bundle.evidence_content:
        raise ValueError(
            'This checkpoint decodes with the word-synchronous evidence path, so the model arm carries per-word EEG '
            'that a vector-built control cannot receive, and every control would lose for that reason alone. Score '
            'the gallery with evidence_content=not decoder.uses_evidence.'
        )

    base = bundle.pmi if score == 'pmi' else bundle.raw
    n = int(base.shape[0])
    lengths = np.asarray(query_n_words).ravel()
    ids = np.asarray(query_content_ids).ravel()
    if lengths.size != n or ids.size != n:
        raise ValueError(f'query arrays disagree with the score matrix: {lengths.size}/{ids.size} vs {n} queries.')

    arms: dict[str, np.ndarray] = {'model': base}

    length_only = _length_only_scores(bundle, decoder, lengths, train=train, batch_size=batch_size)
    if length_only is not None:
        arms['length_only'] = length_only - bundle.null[None, :] if score == 'pmi' else length_only

    # A single query has no other brain to be handed, so both partner arms are undefined rather than degenerate.
    if n > 1:
        arms['shuffled_eeg'] = base[paired_shuffle(n, seed)]
    else:
        _LOG.warning('Arm %r needs at least two queries; it is omitted and fails its clause.', 'shuffled_eeg')

    partners = _mismatch_rows(lengths, ids, seed)
    if partners is not None:
        arms['mismatch'] = base[partners]
    else:
        _LOG.warning('Arm %r found no differing stimulus to pair against; it fails its clause.', 'mismatch')

    # The null prefix is the same score for every query, so PMI subtracts it from itself.
    if score == 'raw':
        arms['null_prefix'] = np.repeat(bundle.null[None, :], n, axis=0)

    return arms


def _without_evidence(readings: ReadingBatch) -> ReadingBatch:
    """Returns the batch with every per-word tensor dropped, which switches the evidence path off."""
    return replace(readings, words=None, valid=None, durations=None)


def _length_only_scores(
    bundle: GalleryScores,
    decoder: ZTEDecoder,
    query_n_words: np.ndarray,
    *,
    train: ReadingBatch | None,
    batch_size: int,
) -> np.ndarray | None:
    """Scores the gallery under a training prefix matched on word count alone, one pass per distinct count."""
    if train is None or len(train) == 0:
        _LOG.warning('Arm %r has no training split to average; it is omitted and fails its clause.', 'length_only')
        return None

    train_words = _train_n_words(train)
    counts, inverse = np.unique(np.asarray(query_n_words).ravel(), return_inverse=True)
    matched = decoder.length_matched_z(train.z, train_words, counts, tol=PARTNER_TOL)
    per_count = decoder.rescore(ReadingBatch.from_vectors(matched), bundle.texts, batch_size=batch_size, pmi=False)

    return per_count[inverse.ravel()]


def _train_n_words(train: ReadingBatch) -> np.ndarray:
    """Returns the training split's word counts, which the length-matched prefix is averaged within.

    Raises:
        ValueError: If the batch carries no `n_words` column, without which the control would be a plain mean
            prefix -- a strictly weaker control that leaves the 5.14 bits of length untested.
    """
    meta: pd.DataFrame = train.meta
    if 'n_words' not in meta:
        raise ValueError("The training batch carries no 'n_words' column, so no length-matched prefix can be built.")

    return meta['n_words'].to_numpy()


def _mismatch_rows(n_words: np.ndarray, content_ids: np.ndarray, seed: int) -> np.ndarray | None:
    """Returns each query's partner row -- a different stimulus at the same word count -- or `None` if none exists."""
    lengths = np.asarray(n_words, dtype=np.float64).ravel()
    ids = np.asarray(content_ids).ravel()
    rng = np.random.default_rng(seed)
    rows = np.empty(lengths.size, dtype=np.int64)

    for i in range(lengths.size):
        other = np.flatnonzero(ids != ids[i])
        if other.size == 0:
            return None

        gap = np.abs(lengths[other] - lengths[i])
        # A word count holding a single stimulus has no partner inside its own stratum, so that query widens to
        # the nearest length rather than being handed its own sentence back.
        pool = other[gap == 0.0] if np.any(gap == 0.0) else other[gap == gap.min()]
        rows[i] = int(rng.choice(pool))

    return rows
