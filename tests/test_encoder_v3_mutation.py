"""Break the exp16 encoder's guarantees on purpose and watch the paired checks go red.

A passing suite is not evidence that a guarantee holds. Every test here reproduces the exact mutation the guarantee
exists to prevent and asserts the property is lost, so if a refactor drops the guarantee silently the paired test in
`test_encoder_v3.py` starts lying and this one starts passing for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zte.models.encoder.consensus import ConsensusBank, ConsensusDistiller
from zte.models.encoder.gallery import GalleryContrast
from zte.models.encoder.nuisance import LengthProjector, length_leakage
from zte.models.encoder.residual import PredictiveResidual


def test_writing_before_reading_makes_a_reading_its_own_teacher() -> None:
    """Write-then-read is the mutation the read-then-write ordering exists to prevent.

    Note:
        With one reader in the bank, a prototype written by this very batch *is* the anchor, so the pull term reads
        a cosine of 1 against a target the encoder just produced. The loss goes to zero while learning nothing --
        the failure looks like fast convergence, which is why only the ordering catches it.
    """
    honest = ConsensusDistiller(n_keys=2, dim=4, n_subjects=2, min_readers=1)
    honest.train()
    z = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    keys, subject = torch.tensor([0, 1]), torch.tensor([0, 1])

    _, metrics = honest.compute(z, keys, subject, pull_weight=1.0, gallery_weight=0.0, prefix='c')
    assert 'c_pull' not in metrics, 'the real distiller reads a bank this batch has not touched'

    # The mutation: update first, then read. Now the teacher is the anchor itself.
    leaky = ConsensusDistiller(n_keys=2, dim=4, n_subjects=2, min_readers=1)
    leaky.train()
    leaky.bank.update(keys, z, subject)
    _, leaked = leaky.compute(z, keys, subject, pull_weight=1.0, gallery_weight=0.0, prefix='c')

    assert leaked['c_pull'] == pytest.approx(0.0, abs=1e-5), 'a self-teacher is satisfied for free'


def test_counting_writes_instead_of_readers_serves_a_consensus_of_one() -> None:
    """One person reading twice is not two readers, and a write counter cannot tell the difference."""
    bank = ConsensusBank(n_keys=1, dim=3, n_subjects=4, min_readers=2)
    vector = torch.tensor([[1.0, 0.0, 0.0]])
    for _ in range(5):
        bank.update(torch.tensor([0]), vector, torch.tensor([2]))

    _, ready = bank.lookup(torch.tensor([0]))
    assert not bool(ready[0]), 'five passes by one reader must still be one reader'
    assert int(bank.writes[0]) == 5, 'the write counter would have said otherwise'


def test_an_expectation_head_left_attached_lets_collapse_pay() -> None:
    """Not detaching the expectation is the mutation that turns a de-trender into a collapse incentive.

    Note:
        With the target attached, driving every token to the same constant sends the regression loss to zero. The
        detached version gives the encoder no gradient at all from this loss, which is the whole guarantee.
    """
    coder = PredictiveResidual(dim=8, gate=1.0)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    _, honest_loss, _ = coder(hidden, torch.ones(2, 5, dtype=torch.bool))
    honest_loss.backward()
    assert hidden.grad is None or float(hidden.grad.abs().sum()) == 0.0

    # The mutation: regress against the live tensor, so the encoder can cut the loss by becoming constant.
    leaky_hidden = torch.randn(2, 5, 8, requires_grad=True)
    predicted = coder.expectation(leaky_hidden, torch.ones(2, 5, dtype=torch.bool))
    ((predicted - leaky_hidden) ** 2).mean().backward()

    assert leaky_hidden.grad is not None
    assert float(leaky_hidden.grad.abs().sum()) > 0.0, 'the encoder now has a reason to make itself predictable'


def test_a_band_left_unwidened_hands_a_lone_anchor_a_free_pass() -> None:
    """Dropping the `min_candidates` rescue is the mutation that turns a hard denominator into an empty one.

    Note:
        The length band is meant to make the loss harder, not smaller. An outlier-length sentence whose band holds
        only itself scores a cross-entropy of exactly zero -- it wins a race it ran alone -- and the mean loss falls
        for a reason that has nothing to do with the encoder.
    """
    contrast = GalleryContrast(band=1, min_candidates=3)
    contrast.attach_lengths(torch.tensor([5, 40, 41, 42]))
    gallery, z = torch.eye(4), torch.eye(4)[:1]
    honest_loss, honest_metrics = contrast.compute(z, gallery, torch.tensor([0]), torch.tensor(1.0))
    assert honest_metrics['gallery_candidates'] == pytest.approx(4.0)
    assert float(honest_loss) > 0.0

    # The mutation: the raw band, with no widening. The anchor's denominator is now itself alone.
    contrast.min_candidates = 0
    lone_loss, lone_metrics = contrast.compute(z, gallery, torch.tensor([0]), torch.tensor(1.0))

    assert lone_metrics['gallery_candidates'] == pytest.approx(1.0)
    assert float(lone_loss) == pytest.approx(0.0, abs=1e-6), 'a one-candidate softmax is always right'


def test_fitting_the_length_projection_on_the_scored_rows_erases_the_confound_it_should_measure() -> None:
    """Transductive fitting is the mutation the train-split-only rule exists to prevent.

    Note:
        Fitted on the rows it is about to transform, the projection removes length essentially perfectly and the
        report would show `length_leakage_after` near zero -- a number that says nothing about the encoder and
        cannot be reproduced by a decoder scoring one sentence at a time.
    """
    rng = np.random.default_rng(3)
    n_words = rng.integers(5, 40, size=200).astype(np.float64)
    z = rng.normal(scale=0.05, size=(200, 12))
    z[:, 0] += n_words
    z = z.astype(np.float32)

    honest = LengthProjector(dim=12)
    honest.fit(z[:100], n_words[:100])
    honest_after = length_leakage(honest.transform(z[100:], n_words[100:]), n_words[100:])

    transductive = LengthProjector(dim=12)
    transductive.fit(z[100:], n_words[100:])
    cheating_after = length_leakage(transductive.transform(z[100:], n_words[100:]), n_words[100:])

    assert cheating_after < honest_after / 10.0, 'fitting on the scored rows flatters the number by an order of'
    assert cheating_after < 1e-3, 'magnitude, down to essentially zero -- which is why the fit provenance travels'
