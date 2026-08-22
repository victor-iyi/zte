"""Break the decoder's honesty machinery on purpose and watch the checks go red.

A passing suite is not evidence that a gate works. Every test here mutates the exact line the gate depends on and
asserts that the guarantee is lost -- so if a future refactor removes the guarantee silently, the paired test in
`test_decoder_v2.py` starts lying and this one starts passing for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zte.cli.decode import mismatch_partners
from zte.inference.decode import paired_shuffle
from zte.models.decoder import MonotonicPointer, SemanticRateLadder, WordEvidence


def test_a_ladder_with_one_live_code_delivers_no_bits() -> None:
    """The mutation the revival logic exists to prevent: a codebook collapsed onto a single entry.

    Note:
        This is what a silently dead codebook looks like from the report's side -- the ceiling still says 32 bits
        and the channel delivers zero. The measured entropy, not the configured stage count, is the number to read.
    """
    ladder = SemanticRateLadder(6, n_stages=2, n_codes=16, revive_after=0).eval()
    with torch.no_grad():
        ladder.codebook.zero_()  # every code identical -> argmin always picks index 0

    out = ladder(torch.randn(64, 6))
    report = ladder.bit_report(out.codes.numpy(), np.arange(64) % 8)

    assert report['code_entropy_bits'] == pytest.approx(0.0, abs=1e-9)
    assert report['mutual_information_bits'] == pytest.approx(0.0, abs=1e-9)
    assert report['capacity_bits'] == pytest.approx(8.0), 'the ceiling is unchanged, which is why it is not the claim'


def test_a_pointer_that_read_content_would_break_the_length_only_control() -> None:
    """Mutating the schedule to depend on content is exactly the cheat `length_only` cannot detect.

    Note:
        The real `MonotonicPointer` takes no content argument at all, which is the structural guarantee. This test
        stands the mutation up beside it so the difference is visible: once the walk depends on the words, the
        control and the headline no longer share a schedule and their paired delta stops meaning anything.
    """
    pointer = MonotonicPointer('linear', width=1.0, tokens_per_word=1.0)
    valid = torch.ones(2, 6, dtype=torch.bool)
    steps = torch.arange(4)

    honest = pointer(steps, valid)
    assert torch.equal(honest, pointer(steps, valid)), 'the real pointer is a function of the mask and step alone'

    # The mutation: bias the window by a per-word content score. Two different readings now walk differently.
    def leaky(content: torch.Tensor) -> torch.Tensor:
        centre = torch.arange(6, dtype=torch.float32).expand(2, 6) + content
        distance = centre[:, None, :] - (steps.float() / 1.0)[None, :, None]
        return torch.softmax(-0.5 * distance**2, dim=-1)

    assert not torch.allclose(leaky(torch.randn(2, 6)), leaky(torch.randn(2, 6)))


def test_an_uncapped_evidence_bias_can_swamp_the_language_model() -> None:
    """Why `evidence_max_bias` exists: without it the path wins the loss by saturating the output distribution."""
    evidence = WordEvidence(8, 6, rank=4, gate_init=100.0, max_bias=1.0)
    words = evidence.word_vectors(torch.randn(2, 5, 8))
    valid = torch.ones(2, 5, dtype=torch.bool)

    capped = evidence.nudge(words, valid, torch.arange(3)).detach()
    assert float(capped.norm(dim=-1).max()) <= 1.0 + 1e-5

    # The mutation: drop the cap. The same gate now produces a nudge two orders of magnitude larger.
    evidence.max_bias = 1e9
    assert float(evidence.nudge(words, valid, torch.arange(3)).detach().norm(dim=-1).max()) > 10.0


def test_the_mismatch_partner_is_never_the_row_itself() -> None:
    """A control paired with its own reading is not a control; it is the headline under another name."""
    lengths = np.array([5, 5, 6, 6, 7, 7, 20, 21])
    ids = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    partner = mismatch_partners(lengths, ids, length_tol=1, seed=0)

    assert not np.any(partner == np.arange(len(ids))), 'no fixed point'
    assert not np.any(ids[partner] == ids), 'and never the same stimulus'


def test_the_mismatch_partner_keeps_the_word_count_matched() -> None:
    """Word count carries 5.14 bits here, so an unstratified mismatch control is easier than the real decode."""
    lengths = np.concatenate([np.full(20, 5.0), np.full(20, 40.0)])
    ids = np.arange(40)

    partner = mismatch_partners(lengths, ids, length_tol=1, seed=0)

    assert np.all(np.abs(lengths[partner] - lengths) <= 2.0)


def test_the_shuffled_control_is_a_derangement() -> None:
    """`shuffled_z` answers "does it matter which brain"; a fixed point silently answers a different question."""
    for n in (2, 5, 64, 105):
        perm = paired_shuffle(n, seed=3)
        assert sorted(perm.tolist()) == list(range(n)), 'still a permutation'
        assert not np.any(perm == np.arange(n)), f'no fixed point at n={n}'


def test_beam_search_is_refused_rather_than_becoming_a_second_decode_path() -> None:
    """Every control and the headline must run byte-identical code, which a second decode strategy would break."""
    from zte.models.decoder.lm import FrozenLM

    lm = FrozenLM('tiny')
    prefix = torch.randn(2, 3, lm.hidden_dim)

    with pytest.raises(ValueError, match='beams'):
        lm.generate_from_prefix(prefix, beams=4)

    # Greedy is accepted and deterministic, so the same prefix always decodes to the same string.
    assert lm.generate_from_prefix(prefix, max_new_tokens=4) == lm.generate_from_prefix(prefix, max_new_tokens=4)


def test_free_running_decode_is_a_pure_function_of_the_prefix() -> None:
    """The teacher-forcing trap, stated as an invariant: no reference token can reach the decode loop.

    Note:
        `generate_from_prefix` takes no reference and no length. This asserts the consequence rather than the
        signature -- decoding the same prefix twice with entirely different references in scope returns the same
        string -- because a future change could smuggle a reference in through module state rather than an argument.
    """
    from zte.models.decoder.lm import FrozenLM

    lm = FrozenLM('tiny')
    torch.manual_seed(0)
    prefix = torch.randn(3, 4, lm.hidden_dim)

    first = lm.generate_from_prefix(prefix, max_new_tokens=8)
    _ = lm.target_token_logprobs(prefix, torch.randint(3, 60, (3, 6)), torch.ones(3, 6, dtype=torch.bool))
    second = lm.generate_from_prefix(prefix, max_new_tokens=8)

    assert first == second


def test_a_decode_that_stops_early_does_not_leak_padding_into_its_text() -> None:
    """A finished row keeps emitting pad ids to keep the batch rectangular; those must never reach the metrics."""
    from zte.models.decoder.lm import FrozenLM

    lm = FrozenLM('tiny')
    row = [5, 6, lm.eos_id, 7, 8, lm.pad_id]

    decoded = lm.decode(torch.tensor(row))

    assert decoded == lm.decode(torch.tensor([5, 6]))
