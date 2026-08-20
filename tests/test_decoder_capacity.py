"""The gallery pass the capacity audit slices, and the conditioning arms it certifies against."""

import os
from dataclasses import replace
from typing import Final

# The whole decoder test path is offline: `lm_source='tiny'` builds its LM and tokeniser locally.
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import pandas as pd
import pytest
import torch

from zte.config import DecoderConfig, ModelConfig, ZTEConfig
from zte.device import resolve_device
from zte.inference.capacity import GalleryScores, capacity_arms, gallery_scores
from zte.inference.decode import ReadingBatch, ZTEDecoder
from zte.models.decoder import GapCorrector, build_bridge, build_lm
from zte.models.decoder.evidence import WordEvidence
from zte.models.embedding import build_model
from zte.models.objectives.lexical import LexicalAligner

_Z_DIM: Final[int] = 16
"""Width of the conditioning vectors the tiny bridge reads."""

_GALLERY: Final[tuple[str, ...]] = (
    'the cat sat',
    'a dog ran far',
    'birds fly south',
    'rivers run deep',
    'the moon is bright',
    'she read a long book today',
)
"""Six candidate sentences; the menus in these tests are column slices of this gallery."""

_GALLERY_WORDS: Final[tuple[int, ...]] = (3, 4, 3, 3, 4, 6)
"""Word count per gallery sentence, which the menus stratify on."""


def _tiny_decoder(*, evidence: bool = False) -> ZTEDecoder:
    """An untrained decoder over the offline tiny LM: scoring needs shapes and wiring, not learned weights."""
    torch.manual_seed(0)
    decoder_config = DecoderConfig(
        lm_source='tiny',
        tokenizer_source='tiny',
        max_target_tokens=24,
        prefix_slots=2,
        bottleneck=8,
        rescore_chunk=3,
        rescore_pmi=False,
    )
    model_config = ModelConfig(embed_dim=16, hidden_dim=16, n_layers=1, n_heads=2, projection_hidden=16)
    model = build_model(model_config, in_dim=40)
    lm = build_lm(decoder_config, encoder=model)
    bridge, _ = build_bridge(decoder_config, _Z_DIM, 16, lm.hidden_dim)

    # The gate starts at zero, so a trained-from-scratch evidence path would nudge nothing and the
    # evidence-on and evidence-off matrices would be trivially equal.
    word_path = None if not evidence else WordEvidence(_Z_DIM, lm.hidden_dim, rank=4, gate_init=1.0)

    return ZTEDecoder(
        model=model,
        config=ZTEConfig(model=model_config, decoder=decoder_config),
        decoder_config=decoder_config,
        bridge=bridge,
        lm=lm,
        gap=GapCorrector(_Z_DIM, mode='none'),
        evidence=word_path,
        lexical=None if word_path is None else LexicalAligner(16, _Z_DIM),
        device=resolve_device('cpu'),
    )


def _readings(n: int, n_words: tuple[int, ...], seed: int = 0) -> ReadingBatch:
    """`n` queries with distinct conditioning vectors and the given word counts."""
    z = np.random.default_rng(seed).standard_normal((n, _Z_DIM)).astype(np.float32)

    return ReadingBatch(z=z, meta=pd.DataFrame({'n_words': list(n_words)}))


def _with_words(readings: ReadingBatch, lm_dim: int, max_words: int = 5, seed: int = 1) -> ReadingBatch:
    """Attaches per-word LM-space vectors, which is what switches the evidence path on."""
    rng = np.random.default_rng(seed)
    n = len(readings)

    return replace(
        readings,
        words=rng.standard_normal((n, max_words, lm_dim)).astype(np.float32),
        valid=np.ones((n, max_words), dtype=bool),
    )


def _bundle(decoder: ZTEDecoder, readings: ReadingBatch, *, evidence_content: bool = True) -> GalleryScores:
    """Scores the whole gallery once."""
    return gallery_scores(
        decoder,
        readings,
        list(_GALLERY),
        gallery_n_words=np.asarray(_GALLERY_WORDS),
        evidence_content=evidence_content,
    )


def _source_rows(matrix: np.ndarray, base: np.ndarray) -> list[int]:
    """Maps each row of `matrix` to the row of `base` it was copied from."""
    rows: list[int] = []
    for row in matrix:
        hits = np.flatnonzero((base == row[None, :]).all(axis=1))
        assert hits.size == 1, 'the model rows must be distinguishable for the permutation to be readable'
        rows.append(int(hits[0]))

    return rows


def test_a_menu_slice_equals_rescoring_that_menu_alone() -> None:
    """Scoring is per-(query, candidate), so one gallery pass sliced in numpy IS per-menu rescoring."""
    decoder = _tiny_decoder()
    readings = _readings(4, (3, 4, 3, 6))
    cols = [1, 3, 4]

    bundle = _bundle(decoder, readings)
    menu = decoder.rescore(readings, [_GALLERY[c] for c in cols], pmi=False)

    np.testing.assert_allclose(bundle.raw[:, cols], menu, rtol=0.0, atol=1e-6)


def test_evidence_is_off_for_every_arm_when_the_run_has_an_evidence_path() -> None:
    """`evidence_content=False` strips the word tensors the controls cannot have, and it changes the scores."""
    decoder = _tiny_decoder(evidence=True)
    assert decoder.uses_evidence
    readings = _with_words(_readings(4, (3, 4, 3, 6)), decoder.lm.hidden_dim)

    off = _bundle(decoder, readings, evidence_content=False)
    on = _bundle(decoder, readings, evidence_content=True)
    stripped = decoder.rescore(replace(readings, words=None, valid=None, durations=None), list(_GALLERY), pmi=False)

    np.testing.assert_allclose(off.raw, stripped, rtol=0.0, atol=0.0)
    assert off.evidence_content is False
    assert not np.allclose(off.raw, on.raw)


def test_arms_refuse_a_model_that_kept_an_evidence_path_the_controls_cannot_have() -> None:
    """Comparing an evidence-carrying model arm with vector-built controls is refused, not silently reported."""
    decoder = _tiny_decoder(evidence=True)
    readings = _with_words(_readings(3, (3, 4, 3)), decoder.lm.hidden_dim)
    bundle = _bundle(decoder, readings, evidence_content=True)

    with pytest.raises(ValueError, match='evidence'):
        capacity_arms(
            bundle,
            decoder,
            readings,
            query_n_words=np.asarray([3, 4, 3]),
            query_content_ids=np.asarray([0, 1, 2]),
            score='raw',
        )


def test_length_only_arm_is_constant_within_a_word_count() -> None:
    """The length control carries word count and nothing else, and its dedup matches the per-query path."""
    decoder = _tiny_decoder()
    words = (3, 4, 3, 6, 4)
    readings = _readings(5, words)
    train = ReadingBatch(
        z=np.random.default_rng(7).standard_normal((9, _Z_DIM)).astype(np.float32),
        meta=pd.DataFrame({'n_words': [3, 3, 3, 4, 4, 6, 6, 4, 3]}),
    )
    bundle = _bundle(decoder, readings)

    arms = capacity_arms(
        bundle,
        decoder,
        readings,
        query_n_words=np.asarray(words),
        query_content_ids=np.asarray([0, 1, 2, 3, 4]),
        score='raw',
        train=train,
    )
    length_only = arms['length_only']

    np.testing.assert_allclose(length_only[0], length_only[2], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(length_only[1], length_only[4], rtol=0.0, atol=0.0)
    assert not np.allclose(length_only[0], length_only[1])
    assert not np.allclose(length_only[0], length_only[3])

    naive = decoder.rescore(
        ReadingBatch.from_vectors(
            decoder.length_matched_z(train.z, train.meta['n_words'].to_numpy(), np.asarray(words), tol=0)
        ),
        list(_GALLERY),
        pmi=False,
    )
    np.testing.assert_allclose(length_only, naive, rtol=0.0, atol=1e-6)


def test_length_only_arm_is_omitted_without_a_training_split() -> None:
    """No training split means no length control, so the arm is absent rather than approximated."""
    decoder = _tiny_decoder()
    readings = _readings(3, (3, 4, 3))
    bundle = _bundle(decoder, readings)

    arms = capacity_arms(
        bundle,
        decoder,
        readings,
        query_n_words=np.asarray([3, 4, 3]),
        query_content_ids=np.asarray([0, 1, 2]),
        score='raw',
    )

    assert 'length_only' not in arms


def test_null_prefix_arm_is_identically_zero_under_pmi() -> None:
    """The null prefix is the same score for every query, so PMI subtracts it from itself and it is dropped."""
    decoder = _tiny_decoder()
    readings = _readings(4, (3, 4, 3, 6))
    bundle = _bundle(decoder, readings)
    words, ids = np.asarray([3, 4, 3, 6]), np.asarray([0, 1, 2, 3])

    raw = capacity_arms(bundle, decoder, readings, query_n_words=words, query_content_ids=ids, score='raw')
    pmi = capacity_arms(bundle, decoder, readings, query_n_words=words, query_content_ids=ids, score='pmi')

    np.testing.assert_allclose(raw['null_prefix'], np.repeat(bundle.null[None, :], 4, axis=0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(raw['null_prefix'] - bundle.null[None, :], 0.0, rtol=0.0, atol=0.0)
    assert 'null_prefix' not in pmi


def test_pmi_is_raw_minus_the_null_column() -> None:
    """The bundle's PMI is exactly the raw score minus the query-independent null, matching the decoder's own path."""
    decoder = _tiny_decoder()
    readings = _readings(3, (3, 4, 3))
    bundle = _bundle(decoder, readings)

    np.testing.assert_allclose(bundle.pmi, bundle.raw - bundle.null[None, :], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(bundle.pmi, decoder.rescore(readings, list(_GALLERY), pmi=True), rtol=0.0, atol=1e-6)


def test_shuffled_arm_is_a_derangement() -> None:
    """No query keeps its own EEG, so the arm cannot inherit the model's own score row."""
    decoder = _tiny_decoder()
    readings = _readings(6, (3, 4, 3, 6, 4, 3))
    bundle = _bundle(decoder, readings)

    arms = capacity_arms(
        bundle,
        decoder,
        readings,
        query_n_words=np.asarray([3, 4, 3, 6, 4, 3]),
        query_content_ids=np.asarray([0, 1, 2, 3, 4, 5]),
        score='pmi',
    )

    sources = _source_rows(arms['shuffled_eeg'], arms['model'])
    assert sorted(sources) == list(range(6))
    assert all(source != query for query, source in enumerate(sources))


def test_mismatch_partner_never_shares_the_query_stimulus() -> None:
    """The mismatch arm answers 'which brain', so its partner is another sentence read at the same length."""
    decoder = _tiny_decoder()
    words = np.asarray([3, 4, 3, 4, 3, 6])
    ids = np.asarray([0, 1, 2, 3, 4, 5])
    readings = _readings(6, tuple(int(w) for w in words))
    bundle = _bundle(decoder, readings)

    arms = capacity_arms(
        bundle,
        decoder,
        readings,
        query_n_words=words,
        query_content_ids=ids,
        score='pmi',
    )

    for query, source in enumerate(_source_rows(arms['mismatch'], arms['model'])):
        assert ids[source] != ids[query]
        if np.any((words == words[query]) & (ids != ids[query])):
            assert words[source] == words[query]
