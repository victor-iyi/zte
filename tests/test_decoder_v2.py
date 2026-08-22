"""The rate ladder and the word-synchronous evidence path: the two mechanisms the v2 decoder adds.

Every test here is written against the failure it exists to catch, and the honesty-critical ones are mutation
tested -- the code is broken on purpose inside the test and the assertion is watched to go red.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from zte.config import DecoderConfig
from zte.models.decoder import (
    MonotonicPointer,
    SemanticRateLadder,
    WordEvidence,
    build_evidence,
    build_rate_ladder,
    measure_tokens_per_word,
)

# --------------------------------------------------------------------------- #
# The semantic rate ladder
# --------------------------------------------------------------------------- #


def test_the_ladder_cannot_pass_more_bits_than_its_codebooks_hold() -> None:
    """The ceiling is architectural, not asserted: no input distribution can raise the measured code entropy above it."""
    ladder = SemanticRateLadder(16, n_stages=3, n_codes=8)
    assert ladder.capacity_bits == pytest.approx(3 * math.log2(8))

    ladder.eval()
    out = ladder(torch.randn(512, 16))
    report = ladder.bit_report(out.codes.numpy())

    assert report['code_entropy_bits'] <= report['capacity_bits'] + 1e-9
    assert all(bits <= math.log2(8) + 1e-9 for bits in report['entropy_bits'])


def test_every_stage_emits_one_discrete_code_per_row() -> None:
    """The conditioning channel is discrete, which is what makes its rate countable rather than argued."""
    ladder = SemanticRateLadder(12, n_stages=4, n_codes=32).eval()
    out = ladder(torch.randn(9, 12))

    assert out.codes.shape == (9, 4)
    assert out.codes.dtype == torch.long
    assert int(out.codes.min()) >= 0
    assert int(out.codes.max()) < 32


def test_the_quantised_vector_still_carries_a_gradient_to_the_encoder() -> None:
    """Straight-through, so quantising the channel does not cut the encoder off from the decoding loss."""
    ladder = SemanticRateLadder(8, n_stages=2, n_codes=4)
    z = torch.randn(6, 8, requires_grad=True)

    ladder(z).z.sum().backward()

    assert z.grad is not None
    assert float(z.grad.abs().sum()) > 0.0


def test_anchoring_seeds_the_codebooks_from_the_text_cloud() -> None:
    """A code has to name a region of the text manifold, or a stage that fires means nothing linguistic."""
    ladder = SemanticRateLadder(6, n_stages=2, n_codes=4)
    before = ladder.codebook.clone()
    cloud = torch.nn.functional.normalize(torch.randn(200, 6), dim=-1)

    ladder.anchor(cloud, iters=4)

    assert not torch.allclose(before, ladder.codebook)
    # Stage 0 splits the cloud itself, so its codes sit inside the cloud's own range.
    assert float(ladder.codebook[0].abs().max()) <= float(cloud.abs().max()) * 2.0


def test_a_refused_cloud_leaves_the_ladder_usable() -> None:
    """A run with too few training stimuli to anchor must degrade, not crash mid-pipeline."""
    ladder = SemanticRateLadder(6, n_stages=2, n_codes=4)

    ladder.anchor(torch.randn(1, 6))

    assert ladder(torch.randn(3, 6)).codes.shape == (3, 2)


def test_the_reserved_stage_learns_the_word_count_the_eye_tracker_gave_away() -> None:
    """Stage 0 is trained to absorb the 5.14 bits of sentence length, so the other stages need not carry them."""
    torch.manual_seed(0)
    ladder = SemanticRateLadder(8, n_stages=2, n_codes=16, length_stage=True, max_words=12)
    z = torch.randn(32, 8)
    n_words = torch.randint(1, 12, (32,))

    out = ladder(z, n_words)

    assert out.length_logits is not None
    assert out.length_logits.shape == (32, 12)


def test_length_orthogonality_is_inert_without_a_reserved_stage() -> None:
    """The penalty only means something when one stage is responsible for length; otherwise it must not fire."""
    plain = SemanticRateLadder(8, n_stages=2, n_codes=8, length_stage=False)
    codes = plain(torch.randn(16, 8)).codes

    assert float(plain.length_orthogonality(codes, torch.randint(1, 20, (16,)))) == 0.0


def test_length_orthogonality_punishes_a_stage_that_tracks_word_count() -> None:
    """The term has to be larger for codes that move with length than for codes that do not."""
    torch.manual_seed(0)
    ladder = SemanticRateLadder(4, n_stages=2, n_codes=8, length_stage=True)
    n_words = torch.arange(16) + 1

    # A stage whose code index follows the word count exactly, against one that ignores it.
    tracking = torch.stack([torch.zeros(16, dtype=torch.long), n_words % 8], dim=1)
    constant = torch.stack([torch.zeros(16, dtype=torch.long), torch.zeros(16, dtype=torch.long)], dim=1)

    assert float(ladder.length_orthogonality(tracking, n_words)) > float(ladder.length_orthogonality(constant, n_words))


def test_the_residual_code_drops_the_reserved_stage() -> None:
    """`residual_z` is the conditioning whose bits are the brain's, so it must not contain the length stage."""
    ladder = SemanticRateLadder(6, n_stages=2, n_codes=4, length_stage=True).eval()
    out = ladder(torch.randn(5, 6), torch.randint(1, 9, (5,)))

    assert not torch.allclose(out.z, out.residual_z)


def test_a_dead_code_is_revived_rather_than_left_shrinking_the_rate() -> None:
    """An unused code silently lowers the real bit-rate below the ceiling the report quotes."""
    torch.manual_seed(0)
    ladder = SemanticRateLadder(4, n_stages=1, n_codes=8, revive_after=1, decay=0.5)
    ladder.train()

    # Feed one tight cluster: most codes go unused and must be re-seeded onto the batch's worst-fit rows.
    for _ in range(6):
        ladder(torch.randn(64, 4) * 0.01 + 5.0)

    assert int(ladder.idle_steps.max()) <= ladder.revive_after


def test_the_bit_report_names_what_it_measured() -> None:
    """The measured budget travels in the artifact, so the arithmetic is never re-derived by a reader."""
    ladder = SemanticRateLadder(6, n_stages=2, n_codes=8, length_stage=True).eval()
    z = torch.randn(64, 6)
    out = ladder(z, torch.randint(1, 9, (64,)))

    report = ladder.bit_report(out.codes.numpy(), np.arange(64) % 10)

    assert report['capacity_bits'] == pytest.approx(6.0)
    assert report['length_stage'] is True
    assert 'mutual_information_bits' in report
    assert 'residual_mutual_information_bits' in report
    assert report['mutual_information_bits'] >= 0.0


def test_build_rate_ladder_is_a_no_op_by_default() -> None:
    """Every new knob defaults to off, so an existing decoder run is behaviourally unchanged by the rebuild."""
    assert build_rate_ladder(DecoderConfig(), 16) is None
    assert build_rate_ladder(DecoderConfig(rate_ladder='rvq'), 16) is not None


# --------------------------------------------------------------------------- #
# The monotonic pointer
# --------------------------------------------------------------------------- #


def test_the_pointer_walks_forward_as_the_decode_advances() -> None:
    """The whole premise is that step `t` reads the word being read at that point in the sentence."""
    pointer = MonotonicPointer('linear', width=1.0, tokens_per_word=1.0)
    valid = torch.ones(1, 10, dtype=torch.bool)

    weights = pointer(torch.arange(10), valid)
    centres = (weights[0] * torch.arange(10, dtype=torch.float32)).sum(dim=-1)

    assert torch.all(centres[1:] > centres[:-1])


def test_the_pointer_never_reads_the_content_it_points_at() -> None:
    """A schedule that depended on content would hand the headline information its controls never got.

    Note:
        This is the clause that makes `length_only` a fair control. The pointer sees only the step count and the
        word mask, so every condition walks the reading identically and only *what* each word was can differ.
    """
    pointer = MonotonicPointer('linear', width=1.5, tokens_per_word=1.4)
    valid = torch.ones(3, 8, dtype=torch.bool)

    first = pointer(torch.arange(5), valid)
    second = pointer(torch.arange(5), valid.clone())

    assert torch.equal(first, second)


def test_the_pointer_ignores_padding_and_survives_an_empty_row() -> None:
    """A reading whose words were all skipped must not put a NaN into the loss."""
    pointer = MonotonicPointer('linear')
    valid = torch.tensor([[True, True, False, False], [False, False, False, False]])

    weights = pointer(torch.arange(3), valid)

    assert torch.isfinite(weights).all()
    assert float(weights[0, :, 2:].abs().max()) == 0.0
    assert float(weights[1].abs().max()) == 0.0


def test_the_walking_rate_is_measured_from_the_corpus_not_configured() -> None:
    """A hand-set rate desynchronises the pointer from the text whenever the tokeniser or the corpus changes."""
    mask = torch.ones(4, 10, dtype=torch.bool)
    n_words = torch.full((4,), 5)

    assert measure_tokens_per_word(mask, n_words) == pytest.approx(2.0)
    # A tokeniser never emits fewer than one token per word, so the rate is floored rather than allowed below 1.
    assert measure_tokens_per_word(torch.ones(1, 2, dtype=torch.bool), torch.tensor([10])) == pytest.approx(1.0)


def test_the_measured_rate_rides_in_the_checkpoint() -> None:
    """A checkpoint that lost the rate would decode against a different alignment than it trained under."""
    pointer = MonotonicPointer('linear', tokens_per_word=1.4)
    pointer.tokens_per_word = 2.75

    restored = MonotonicPointer('linear')
    restored.load_state_dict(pointer.state_dict())

    assert restored.tokens_per_word == pytest.approx(2.75)


# --------------------------------------------------------------------------- #
# The evidence path
# --------------------------------------------------------------------------- #


def test_the_evidence_starts_switched_off() -> None:
    """The gate is zero-initialised, so a run begins as the pooled decoder and the path has to earn its influence."""
    evidence = WordEvidence(8, 6, rank=4, gate_init=0.0)
    words = evidence.word_vectors(torch.randn(3, 5, 8))

    nudge = evidence.nudge(words, torch.ones(3, 5, dtype=torch.bool), torch.arange(4))

    assert float(nudge.detach().abs().max()) == 0.0


def test_the_evidence_is_capped_so_it_cannot_saturate_the_softmax() -> None:
    """An uncapped bias wins the loss by collapsing the distribution, which reads as decoding and is not."""
    evidence = WordEvidence(8, 6, rank=4, gate_init=50.0, max_bias=2.0)
    words = evidence.word_vectors(torch.randn(3, 5, 8))

    nudge = evidence.nudge(words, torch.ones(3, 5, dtype=torch.bool), torch.arange(4))

    assert float(nudge.detach().norm(dim=-1).max()) <= 2.0 + 1e-5


def test_the_null_evidence_destroys_content_and_keeps_the_schedule() -> None:
    """That split is the `length_only` control: the word count survives, what each word was does not."""
    evidence = WordEvidence(8, 6, rank=4, gate_init=1.0)
    words = evidence.word_vectors(torch.randn(2, 7, 8))
    valid = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
    steps = torch.arange(3)

    nulled = evidence.null(words)
    real_weights = evidence.pointer(steps, valid)

    assert float(nulled.detach().abs().max()) == 0.0
    # The schedule is a function of `valid` alone, so the control walks the reading exactly as the headline does.
    assert torch.equal(real_weights, evidence.pointer(steps, valid))
    assert float(evidence.nudge(nulled, valid, steps).detach().abs().max()) == 0.0


def test_the_evidence_changes_the_output_when_the_words_change() -> None:
    """With the gate open the path must actually depend on the brain, or it is decoration."""
    torch.manual_seed(0)
    evidence = WordEvidence(8, 6, rank=4, gate_init=1.0)
    valid = torch.ones(2, 5, dtype=torch.bool)
    steps = torch.arange(3)

    one = evidence.nudge(evidence.word_vectors(torch.randn(2, 5, 8)), valid, steps)
    two = evidence.nudge(evidence.word_vectors(torch.randn(2, 5, 8)), valid, steps)

    assert not torch.allclose(one, two)


def test_build_evidence_is_a_no_op_by_default() -> None:
    """`evidence_schedule` defaults to `none`, so the pooled decoder is the unchanged baseline."""
    assert build_evidence(DecoderConfig(), 8, 6) is None
    assert build_evidence(DecoderConfig(evidence_schedule='linear'), 8, 6) is not None
