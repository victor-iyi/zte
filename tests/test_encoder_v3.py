"""The exp16 encoder mechanisms: cross-reader consensus, predictive residual, gallery negatives, length projection."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zte.config import ModelConfig, ObjectiveConfig
from zte.models.encoder.consensus import ConsensusBank, ConsensusDistiller, build_consensus
from zte.models.encoder.gallery import GalleryContrast, build_gallery_contrast, text_word_counts
from zte.models.encoder.nuisance import LengthProjector, length_leakage
from zte.models.encoder.residual import PredictiveResidual, build_predictive_residual

# -- Cross-reader consensus ------------------------------------------------- #


def test_a_cold_prototype_is_the_batch_mean_rather_than_a_decayed_zero() -> None:
    """The first write must land whole, or every prototype spends its early life shrunk toward the origin."""
    bank = ConsensusBank(n_keys=4, dim=3, n_subjects=2, decay=0.9, min_readers=1)
    vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    bank.update(torch.tensor([2, 2]), vectors, torch.tensor([0, 1]))

    expected = torch.nn.functional.normalize(vectors, dim=-1).mean(0)
    assert torch.allclose(bank.prototypes[2], expected, atol=1e-6)


def test_a_prototype_is_served_only_once_distinct_people_have_read_it() -> None:
    """Two passes by one reader is one reading twice, not a consensus."""
    bank = ConsensusBank(n_keys=2, dim=3, n_subjects=4, min_readers=2)
    same_reader = torch.tensor([[1.0, 0.0, 0.0]])
    bank.update(torch.tensor([0]), same_reader, torch.tensor([1]))
    bank.update(torch.tensor([0]), same_reader, torch.tensor([1]))

    _, ready = bank.lookup(torch.tensor([0]))
    assert not bool(ready[0])

    bank.update(torch.tensor([0]), same_reader, torch.tensor([3]))
    _, ready = bank.lookup(torch.tensor([0]))
    assert bool(ready[0])


def test_a_negative_or_out_of_range_key_never_reaches_the_bank() -> None:
    """Padding positions carry `-1`, and they must not write over key 0 or read back as ready."""
    bank = ConsensusBank(n_keys=2, dim=2, n_subjects=2, min_readers=1)
    bank.update(torch.tensor([-1, 7]), torch.ones(2, 2), torch.tensor([0, 1]))

    assert float(bank.prototypes.abs().sum()) == 0.0
    _, ready = bank.lookup(torch.tensor([-1]))
    assert not bool(ready[0])


def test_the_teacher_is_built_from_earlier_steps_only() -> None:
    """Read then write: a prototype written by this very batch would make the pull term partly self-referential."""
    distiller = ConsensusDistiller(n_keys=3, dim=4, n_subjects=2, min_readers=1)
    distiller.train()
    keys, subject = torch.tensor([0, 1]), torch.tensor([0, 1])

    loss, metrics = distiller.compute(torch.randn(2, 4), keys, subject, pull_weight=1.0, gallery_weight=0.0, prefix='c')
    assert 'c_pull' not in metrics  # nothing was in the bank when the loss read it
    assert float(loss) == 0.0
    assert int(distiller.bank.readers.sum()) == 2  # ...but the write happened

    _, metrics = distiller.compute(torch.randn(2, 4), keys, subject, pull_weight=1.0, gallery_weight=0.0, prefix='c')
    assert 'c_pull' in metrics


def test_the_bank_is_not_written_outside_training() -> None:
    """A validation pass must leave the teacher exactly as training left it."""
    distiller = ConsensusDistiller(n_keys=3, dim=4, n_subjects=2, min_readers=1)
    distiller.eval()
    distiller.compute(
        torch.randn(2, 4), torch.tensor([0, 1]), torch.tensor([0, 1]), pull_weight=1.0, gallery_weight=0.0, prefix='c'
    )

    assert int(distiller.bank.readers.sum()) == 0


def test_the_consensus_gallery_scores_each_reading_against_its_own_prototype() -> None:
    """The label is the anchor's own key's position in the served gallery, not its position in the batch."""
    distiller = ConsensusDistiller(n_keys=4, dim=3, n_subjects=2, min_readers=1, temperature=1.0)
    distiller.train()
    basis = torch.eye(3)

    # Seed keys 1 and 3 with orthogonal prototypes, then hand back exactly those vectors.
    distiller.bank.update(torch.tensor([1, 3]), basis[:2], torch.tensor([0, 1]))
    _, metrics = distiller.compute(
        basis[:2], torch.tensor([1, 3]), torch.tensor([0, 1]), pull_weight=0.0, gallery_weight=1.0, prefix='c'
    )
    assert metrics['c_gallery_top1'] == pytest.approx(1.0)
    assert metrics['c_gallery_size'] == pytest.approx(2.0)


def test_consensus_is_built_only_for_the_weights_that_are_set() -> None:
    """Each level is its own lever, so an ablation can turn off the word bank without touching the sentence bank."""
    config = ObjectiveConfig(consensus_weight=1.0, consensus_word_weight=0.0)
    sentence, word = build_consensus(config, n_sentences=10, n_content=50, dim=8)
    assert sentence is not None
    assert word is None

    off = ObjectiveConfig()
    assert build_consensus(off, n_sentences=10, n_content=50, dim=8) == (None, None)


# -- Predictive residual ---------------------------------------------------- #


def test_a_zero_gate_leaves_the_token_untouched() -> None:
    """The knob has to default to a no-op, or every existing run silently changes."""
    coder = PredictiveResidual(dim=8, gate=0.0)
    hidden = torch.randn(2, 5, 8)
    residual, _, metrics = coder(hidden, torch.ones(2, 5, dtype=torch.bool))

    assert torch.allclose(residual, hidden)
    assert metrics['residual_gate'] == pytest.approx(0.0)


def test_the_expectation_head_cannot_push_the_encoder_to_become_predictable() -> None:
    """Its regression loss reaches the head and stops: otherwise collapse would be the cheapest way to cut it."""
    coder = PredictiveResidual(dim=8, gate=1.0)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    _, predict_loss, _ = coder(hidden, torch.ones(2, 5, dtype=torch.bool))
    predict_loss.backward()

    assert hidden.grad is None or float(hidden.grad.abs().sum()) == 0.0
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in coder.parameters())


def test_the_residual_still_carries_the_objective_gradient() -> None:
    """De-trending shifts the token by a constant; it must not cut the encoder off from its own loss."""
    coder = PredictiveResidual(dim=8, gate=1.0)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    residual, _, _ = coder(hidden, torch.ones(2, 5, dtype=torch.bool))
    residual.sum().backward()

    assert hidden.grad is not None
    assert float(hidden.grad.abs().sum()) > 0.0


def test_the_expectation_of_the_first_word_uses_no_word() -> None:
    """Position 0 sees only the learned BOS, so changing later tokens cannot change what it predicted."""
    coder = PredictiveResidual(dim=8, gate=1.0).eval()
    valid = torch.ones(1, 4, dtype=torch.bool)
    hidden = torch.randn(1, 4, 8)
    other = hidden.clone()
    other[0, 1:] = torch.randn(3, 8)

    with torch.no_grad():
        assert torch.allclose(coder.expectation(hidden, valid)[0, 0], coder.expectation(other, valid)[0, 0], atol=1e-6)


def test_residual_coding_is_off_by_default() -> None:
    """A default `ModelConfig` must build no residual coder at all."""
    assert build_predictive_residual(ModelConfig(), dim=16) is None
    assert build_predictive_residual(ModelConfig(residual_coding=True), dim=16) is not None


# -- Gallery negatives ------------------------------------------------------ #


def test_the_length_band_keeps_only_same_length_distractors() -> None:
    """A denominator of same-length texts is what makes counting words worth nothing."""
    contrast = GalleryContrast(band=1, min_candidates=1)
    contrast.attach_lengths(torch.tensor([5, 6, 10, 20]))
    mask = contrast.candidate_mask(torch.tensor([0]), n_texts=4)

    assert mask.tolist() == [[True, True, False, False]]


def test_the_answer_is_in_its_own_denominator_under_every_band() -> None:
    """A cross-entropy whose target column is masked saturates and stops reading the model, so this must always hold."""
    contrast = GalleryContrast(band=1, min_candidates=1)
    contrast.attach_lengths(torch.tensor([5, 40, 41, 42]))
    for band in (0, 1, 3, 100):
        contrast.band = band
        mask = contrast.candidate_mask(torch.arange(4), n_texts=4)
        assert bool(mask.diagonal().all()), f'band {band} masked an anchor out of its own softmax'


def test_a_band_that_strands_an_anchor_widens_to_the_whole_gallery() -> None:
    """Three distractors would make that anchor's loss small rather than hard, which is the opposite of the point."""
    contrast = GalleryContrast(band=1, min_candidates=3)
    contrast.attach_lengths(torch.tensor([5, 40, 41, 42]))
    mask = contrast.candidate_mask(torch.tensor([0]), n_texts=4)

    assert bool(mask.all())


def test_the_gallery_loss_falls_when_the_answer_is_the_nearest_text() -> None:
    """The term has to actually score retrieval, not merely run."""
    contrast = GalleryContrast(band=0)
    gallery = torch.eye(4)
    right = contrast.compute(torch.eye(4)[:2], gallery, torch.tensor([0, 1]), torch.tensor(10.0))[0]
    wrong = contrast.compute(torch.eye(4)[:2], gallery, torch.tensor([2, 3]), torch.tensor(10.0))[0]

    assert float(right) < float(wrong)


def test_the_gallery_reports_the_chance_its_own_denominator_implies() -> None:
    """A length-matched denominator has a different chance level, and quoting the 1/700 one would flatter it."""
    contrast = GalleryContrast(band=1, min_candidates=1)
    contrast.attach_lengths(torch.tensor([5, 5, 30, 30]))
    _, metrics = contrast.compute(torch.eye(4)[:2], torch.eye(4), torch.tensor([0, 1]), torch.tensor(1.0))

    assert metrics['gallery_candidates'] == pytest.approx(2.0)
    assert metrics['gallery_chance'] == pytest.approx(0.5)


def test_word_counts_come_from_the_gallery_texts_in_row_order() -> None:
    """Row `i` of the count vector must be the length of the text at row `i` of the frozen matrix."""
    assert text_word_counts(['a b c', '', 'one two']).tolist() == [3, 0, 2]


def test_the_denominator_holds_only_the_texts_the_split_reads() -> None:
    """The frozen matrix is indexed by a whole-dataset id, so a stimulus-holding-out split must mask its own test
    sentences out of the negatives -- training against a held-out text still teaches the encoder where not to map."""
    contrast = GalleryContrast(band=0)
    contrast.restrict_to([0, 1], n_texts=4)
    mask = contrast.candidate_mask(torch.tensor([0, 1]), n_texts=4)

    assert mask.tolist() == [[True, True, False, False], [True, True, False, False]]


def test_the_split_restriction_survives_the_length_band_and_the_widening() -> None:
    """Both the band and the min-candidates rescue must stay inside the admissible set, or the mask leaks anyway."""
    contrast = GalleryContrast(band=1, min_candidates=3)
    contrast.attach_lengths(torch.tensor([5, 5, 5, 40]))
    contrast.restrict_to([0, 1], n_texts=4)

    banded = contrast.candidate_mask(torch.tensor([0]), n_texts=4)
    assert banded.tolist() == [[True, True, False, False]], 'the band reached a held-out text'

    contrast.attach_lengths(torch.tensor([5, 40, 41, 42]))
    widened = contrast.candidate_mask(torch.tensor([0]), n_texts=4)
    assert widened.tolist() == [[True, True, False, False]], 'the widening reached past the admissible set'


def test_an_unrestricted_gallery_scores_every_text() -> None:
    """Under a subject-only split every stimulus is a training stimulus, so the restriction is a no-op there."""
    contrast = GalleryContrast(band=0)
    assert bool(contrast.candidate_mask(torch.tensor([0]), n_texts=4).all())


def test_a_cold_prototype_is_never_served_as_a_distractor() -> None:
    """The consensus bank is sized whole-dataset but written from training rows, so held-out keys stay unserved.

    Note:
        This is why the consensus term needs no explicit split restriction: a stimulus nobody in the training split
        read has zero readers, sits below `min_readers`, and never enters `ready_keys`.
    """
    bank = ConsensusBank(n_keys=6, dim=3, n_subjects=4, min_readers=2)
    for subject in (0, 1):
        bank.update(torch.tensor([2]), torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([subject]))

    assert bank.ready_keys().tolist() == [2]
    assert bank.coverage()['consensus_keys_ready'] == 1.0


def test_the_gallery_term_is_off_by_default() -> None:
    """A default objective must build no gallery scorer."""
    assert build_gallery_contrast(ObjectiveConfig(), n_texts=100) is None
    assert build_gallery_contrast(ObjectiveConfig(gallery_weight=1.0), n_texts=100) is not None


# -- Length projection ------------------------------------------------------ #


def _length_confounded_cloud(n: int = 400, dim: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns embeddings whose first three dimensions are a pure function of word count, plus noise."""
    rng = np.random.default_rng(seed)
    n_words = rng.integers(5, 40, size=n).astype(np.float64)
    z = rng.normal(scale=0.05, size=(n, dim))
    z[:, 0] += n_words
    z[:, 1] += np.log(n_words) * 3.0
    z[:, 2] += n_words**2 / 50.0
    return z.astype(np.float32), n_words


def test_the_projection_removes_the_length_it_was_fitted_on() -> None:
    """The measurement the mechanism exists for: word count explains far less variance afterwards."""
    z, n_words = _length_confounded_cloud()
    projector = LengthProjector(dim=z.shape[1])
    projector.fit(z, n_words)

    before = length_leakage(z, n_words)
    after = length_leakage(projector.transform(z, n_words), n_words)
    assert before > 0.9
    assert after < 0.01


def test_the_projection_generalises_to_rows_it_never_saw() -> None:
    """Fitted on train, applied to test -- the only version of this that is not transductive."""
    train_z, train_n = _length_confounded_cloud(seed=1)
    test_z, test_n = _length_confounded_cloud(seed=2)
    projector = LengthProjector(dim=train_z.shape[1])
    projector.fit(train_z, train_n)

    assert length_leakage(projector.transform(test_z, test_n), test_n) < 0.05


def test_an_unfitted_projector_passes_embeddings_through() -> None:
    """It must be the identity until someone fits it, so a missing train split degrades rather than corrupts."""
    z, n_words = _length_confounded_cloud(n=20, dim=4)
    assert np.allclose(LengthProjector(dim=4).transform(z, n_words), z)


def test_fitting_refuses_shapes_it_cannot_honour() -> None:
    """A silent shape coercion here would fit the projection against the wrong sentences."""
    z, n_words = _length_confounded_cloud(n=40, dim=8)
    with pytest.raises(ValueError, match='must be'):
        LengthProjector(dim=4).fit(z, n_words)
    with pytest.raises(ValueError, match='word counts'):
        LengthProjector(dim=8).fit(z, n_words[:10])
    with pytest.raises(ValueError, match='more than'):
        LengthProjector(dim=8).fit(z[:3], n_words[:3])


def test_leakage_is_zero_when_every_sentence_is_the_same_length() -> None:
    """With no variance in the confound there is nothing to explain, and a spurious number would read as leakage."""
    z = np.random.default_rng(0).normal(size=(50, 8)).astype(np.float32)
    assert length_leakage(z, np.full(50, 12.0)) == 0.0


def test_the_projection_is_off_by_default() -> None:
    """Everything new here defaults to a no-op, so an existing run stays behaviourally identical."""
    config = ObjectiveConfig()
    assert config.length_projection is False
    assert config.consensus_weight == 0.0
    assert config.consensus_gallery_weight == 0.0
    assert config.consensus_word_weight == 0.0
    assert config.gallery_weight == 0.0
    assert ModelConfig().residual_coding is False
