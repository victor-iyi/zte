"""Tests for the trainable decoder surface: the prefix bridge, the word resampler, the gap fit and the frozen LM."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

# The whole decoder test path is offline: `lm_source='tiny'` builds its LM and tokeniser locally.
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from zte.config import DecoderConfig, ModelConfig
from zte.models.decoder import (
    FrozenLM,
    GapCorrector,
    PrefixBridge,
    WordResampler,
    build_bridge,
    build_lm,
)
from zte.models.embedding import build_model

_Z_DIM, _LM_DIM, _SLOTS, _BOTTLENECK = 768, 896, 8, 128


@pytest.fixture(scope='module')
def tiny_lm() -> FrozenLM:
    """The offline 22,688-parameter LM every decoder test drives, built once for the module."""
    return build_lm(DecoderConfig(lm_source='tiny', tokenizer_source='tiny')).eval()


def _batch(n: int = 4, length: int = 6, in_dim: int = 40, seed: int = 0) -> dict[str, torch.Tensor]:
    """A collated band-power batch with every word present."""
    generator = torch.Generator().manual_seed(seed)
    return {
        'features': torch.randn(n, length, in_dim, generator=generator),
        'pad_mask': torch.ones(n, length, dtype=torch.bool),
        'presence': torch.ones(n, length, dtype=torch.bool),
        'subject': torch.zeros(n, dtype=torch.long),
    }


# --------------------------------------------------------------------------- #
# the prefix bridge
# --------------------------------------------------------------------------- #
def test_the_bridge_is_exactly_the_budgeted_size() -> None:
    """226,560 parameters against roughly 120k supervised target tokens; anything larger memorises the corpus."""
    bridge = PrefixBridge(_Z_DIM, _LM_DIM, slots=_SLOTS, bottleneck=_BOTTLENECK)
    assert bridge(torch.randn(4, _Z_DIM)).shape == (4, _SLOTS, _LM_DIM)
    assert sum(p.numel() for p in bridge.parameters() if p.requires_grad) == 226_560


def test_every_slot_starts_out_different() -> None:
    """Zero-initialised FiLM would make all eight slots one vector, i.e. a prompt one position wide."""
    bridge = PrefixBridge(32, 16, slots=4, bottleneck=8)
    prefix = bridge(torch.randn(2, 32))
    assert not torch.allclose(prefix[:, 0], prefix[:, 1], atol=1e-4)


def test_null_prefix_dropout_is_all_or_nothing() -> None:
    """At `p=0` the conditional prefix passes through untouched; at `p=1` it is exactly the learned null prefix."""
    bridge = PrefixBridge(32, 16, slots=4, bottleneck=8)
    prefix = bridge(torch.randn(5, 32))

    kept, replaced = bridge.dropout_null(prefix, 0.0)
    assert torch.equal(kept, prefix) and not bool(replaced.any())

    dropped, replaced = bridge.dropout_null(prefix, 1.0)
    assert torch.equal(dropped, bridge.null(5)) and bool(replaced.all())
    assert bridge.null(5).shape == (5, 4, 16)


def test_build_bridge_adds_a_resampler_only_for_word_conditioning() -> None:
    """The word slots are a registered ablation arm, so the default configuration must not build them."""
    pooled, none = build_bridge(DecoderConfig(prefix_slots=4, bottleneck=8), 32, 24, 16)
    assert none is None and pooled.slots == 4

    config = DecoderConfig(
        conditioning='pooled_plus_words', prefix_slots=4, word_slots=3, bottleneck=8
    )
    _, resampler = build_bridge(config, 32, 24, 16)
    assert resampler is not None and resampler.slots == 3


# --------------------------------------------------------------------------- #
# the word resampler
# --------------------------------------------------------------------------- #
def test_the_resampler_cannot_see_a_padded_position() -> None:
    """Padding carries the sentence length, so a resampler that read it would read the length confound."""
    torch.manual_seed(0)
    resampler = WordResampler(24, 16, slots=3, n_blocks=2, n_heads=4).eval()
    hidden = torch.randn(2, 7, 24)
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[:, 4:] = False

    before = resampler(hidden, valid)
    hidden[:, 4:] = torch.randn(2, 3, 24) * 100.0
    assert torch.equal(before, resampler(hidden, valid))


def test_the_resampler_survives_a_fully_masked_row() -> None:
    """A reading whose every word was dropped must fall back to attending everything, not return NaN."""
    resampler = WordResampler(24, 16, slots=3).eval()
    valid = torch.ones(2, 5, dtype=torch.bool)
    valid[1] = False
    assert torch.isfinite(resampler(torch.randn(2, 5, 24), valid)).all()


# --------------------------------------------------------------------------- #
# the modality-gap correction
# --------------------------------------------------------------------------- #
def _clouds(dim: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """An EEG cloud and a text cloud with deliberately different means and scales."""
    rng = np.random.default_rng(seed)
    eeg = torch.from_numpy((rng.standard_normal((256, dim)) * 0.2 - 3.0).astype(np.float32))
    txt = torch.from_numpy((rng.standard_normal((256, dim)) * 2.0 + 1.0).astype(np.float32))
    return eeg, txt


def test_the_gap_correction_moves_the_eeg_cloud_onto_the_text_cloud() -> None:
    """The bridge reads a text space, so an EEG vector on its own shell is out of distribution for the frozen LM."""
    eeg, txt = _clouds()
    gap = GapCorrector(8, mode='mean_scale')
    gap.fit(eeg, txt)
    moved = gap(eeg)
    assert torch.allclose(moved.mean(0), txt.mean(0), atol=1e-4)
    assert torch.allclose(moved.std(0), txt.std(0), atol=1e-3)
    assert gap.n_fit == 256


def test_the_none_mode_is_the_identity_and_an_unfitted_corrector_passes_through() -> None:
    """A configuration that asks for no correction must change no number, fitted or not."""
    eeg, txt = _clouds()
    off = GapCorrector(8, mode='none')
    assert torch.equal(off(eeg), eeg)
    off.fit(eeg, txt)
    assert torch.equal(off(eeg), eeg)
    assert torch.equal(GapCorrector(8, mode='mean_scale')(eeg), eeg)


def test_whitening_is_not_a_synonym_for_mean_scaling() -> None:
    """`whiten` registers the two covariance maps that make it a full ZCA, or the config option would be a lie."""
    eeg, txt = _clouds()
    whiten, scale = GapCorrector(8, mode='whiten'), GapCorrector(8, mode='mean_scale')
    whiten.fit(eeg, txt)
    scale.fit(eeg, txt)
    assert whiten.whiten_eeg.shape == (8, 8) and whiten.colour_txt.shape == (8, 8)
    assert not torch.allclose(whiten(eeg), scale(eeg), atol=1e-3)
    assert torch.allclose(whiten(eeg).mean(0), txt.mean(0), atol=1e-3)


def test_the_gap_state_round_trips_exactly() -> None:
    """The correction travels in the checkpoint, so inference must reproduce training to the bit."""
    eeg, txt = _clouds()
    for mode in ('none', 'mean_scale', 'whiten'):
        gap = GapCorrector(8, mode=mode)
        gap.fit(eeg, txt)
        restored = GapCorrector.from_state(gap.state)
        assert restored.mode == mode and restored.n_fit == gap.n_fit
        assert torch.equal(restored(eeg), gap(eeg))


def test_the_gap_refuses_a_cloud_it_cannot_describe() -> None:
    """One row has no scale and the wrong width is a wiring mistake; both fail loudly rather than fitting noise."""
    eeg, txt = _clouds()
    gap = GapCorrector(8, mode='mean_scale')
    with pytest.raises(ValueError, match='at least 2 rows'):
        gap.fit(eeg[:1], txt)
    with pytest.raises(ValueError, match=r'must be \(n, 8\)'):
        gap.fit(eeg[:, :4], txt)


# --------------------------------------------------------------------------- #
# the frozen LM
# --------------------------------------------------------------------------- #
def test_the_language_model_is_frozen_and_uncheckpointed(tiny_lm: FrozenLM) -> None:
    """An empty state dict is mandatory: the trainer writes `objective.state_dict()` into every epoch checkpoint."""
    assert tiny_lm.state_dict() == {}
    assert not any(p.requires_grad for p in tiny_lm.parameters())
    assert tiny_lm.train(True).training is False
    incompatible = tiny_lm.load_state_dict({'nonsense': torch.zeros(1)})
    assert not incompatible.missing_keys and not incompatible.unexpected_keys


def test_a_checkpoint_carrying_the_language_model_stays_small(
    tiny_lm: FrozenLM, tmp_path: Path
) -> None:
    """The LM sits inside the objective, so without the empty state dict every epoch would add its weights."""
    holder = torch.nn.Module()
    holder.lm = tiny_lm
    holder.bridge = PrefixBridge(64, tiny_lm.hidden_dim, slots=4, bottleneck=16)
    state = holder.state_dict()
    assert not any(k.startswith('lm.') for k in state)

    path = tmp_path / 'ckpt.pt'
    torch.save({'extra': {'objective_state': state}}, path)
    assert path.stat().st_size < 1_000_000


def test_the_prompt_is_assembled_as_bos_prefix_scaffold_target(tiny_lm: FrozenLM) -> None:
    """The target span starts after the prefix and the scaffold, and padded target tokens are masked out."""
    prefix = torch.randn(3, 5, tiny_lm.hidden_dim)
    ids = torch.randint(4, 60, (3, 7))
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[:, 5:] = False

    embeds, attention, start = tiny_lm.assemble(prefix, ids, mask)
    assert start == 1 + 5 + int(tiny_lm.scaffold.shape[0])
    assert embeds.shape == (3, start + 7, tiny_lm.hidden_dim)
    assert bool(attention[:, :start].all())
    assert torch.equal(attention[:, start:].bool(), mask)


def test_target_scoring_is_differentiable_and_ignores_padding(tiny_lm: FrozenLM) -> None:
    """Teacher-forced cross-entropy is the training signal, so gradient must reach the prefix and skip the padding."""
    prefix = torch.randn(2, 4, tiny_lm.hidden_dim, requires_grad=True)
    ids = torch.randint(4, 60, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[:, 4:] = False

    logprobs = tiny_lm.target_token_logprobs(prefix, ids, mask)
    assert logprobs.shape == (2, 6)
    assert torch.equal(logprobs[:, 4:], torch.zeros(2, 2))

    loss = tiny_lm.forward_with_prefix(prefix, ids, mask)
    loss.backward()
    assert prefix.grad is not None and float(prefix.grad.abs().sum()) > 0.0


def test_gallery_rescoring_is_the_no_grad_twin_of_the_grounding_score(tiny_lm: FrozenLM) -> None:
    """The grounding loss and the retrieval readout must be the same function, one of them merely wrapped."""
    prefix = torch.randn(2, 4, tiny_lm.hidden_dim, requires_grad=True)
    ids = torch.randint(4, 60, (5, 6))
    mask = torch.ones(5, 6, dtype=torch.bool)

    differentiable = tiny_lm.candidate_logprobs(prefix, ids, mask, chunk=3)
    with torch.no_grad():
        scored = tiny_lm.sequence_logprob(prefix, ids, mask, chunk=3)
    assert differentiable.shape == (2, 5)
    assert torch.allclose(differentiable.detach(), scored)
    assert differentiable.requires_grad and not scored.requires_grad


def test_the_prefix_influence_detector_is_zero_against_itself(tiny_lm: FrozenLM) -> None:
    """A bridge collapsed to a constant prompt scores ~0 here whatever its loss curve looked like."""
    torch.manual_seed(0)
    prefix = torch.randn(3, 4, tiny_lm.hidden_dim)
    assert torch.allclose(tiny_lm.next_token_kl(prefix, prefix), torch.zeros(3), atol=1e-6)
    assert float(tiny_lm.next_token_kl(prefix, torch.randn_like(prefix)).mean()) > 0.0


def test_free_running_decode_returns_only_what_it_wrote(tiny_lm: FrozenLM) -> None:
    """Generating from `inputs_embeds` yields the new ids alone, and greedy decoding repeats exactly."""
    torch.manual_seed(0)
    prefix = torch.randn(3, 4, tiny_lm.hidden_dim)
    texts = tiny_lm.generate_from_prefix(prefix, max_new_tokens=8)
    assert len(texts) == 3
    assert all(isinstance(t, str) and len(t) <= 8 for t in texts)
    assert tiny_lm.generate_from_prefix(prefix, max_new_tokens=8) == texts


def test_the_language_model_records_what_pins_it(tiny_lm: FrozenLM) -> None:
    """A decode is only reproducible if the weights and the tokeniser are both named in the manifest."""
    record = tiny_lm.provenance()
    assert record['source'] == 'tiny'
    assert record['n_parameters'] == 22_688
    assert record['tokenizer'] == 'tiny' and record['tokenizer_fingerprint']
    assert record['prompt_template'] == DecoderConfig().prompt_template


# --------------------------------------------------------------------------- #
# the conditioning vector
# --------------------------------------------------------------------------- #
def test_sentence_hidden_is_the_differentiable_half_of_embed_sentence() -> None:
    """The decoder loss conditions on the same pooled vector retrieval exports, and needs its gradient."""
    model = build_model(
        ModelConfig(embed_dim=16, hidden_dim=16, n_layers=1, n_heads=2), in_dim=40
    ).eval()
    batch = _batch()

    hidden = model.sentence_hidden(batch, contextual=True)
    assert hidden.shape == (4, model.hidden_dim)
    assert hidden.requires_grad
    assert torch.allclose(model.project(hidden), model.embed_sentence(batch), atol=1e-6)
    assert not model.embed_sentence(batch).requires_grad


def test_embed_sentence_still_routes_by_objective() -> None:
    """Skip-gram never trains its contextual path, so its exported vector must stay the per-token one."""
    model = build_model(
        ModelConfig(embed_dim=16, hidden_dim=16, n_layers=1, n_heads=2), in_dim=40
    ).eval()
    batch = _batch()
    with torch.no_grad():
        flat = model.project(model.sentence_hidden(batch, contextual=False))
    assert torch.allclose(model.embed_sentence(batch, 'skipgram'), flat, atol=1e-6)
    assert not torch.allclose(model.embed_sentence(batch, 'masked'), flat, atol=1e-4)


def test_a_reading_with_no_present_word_pools_without_nan() -> None:
    """Attention pooling returns NaN for a fully masked row, so the mask falls back to the padding mask."""
    model = build_model(
        ModelConfig(embed_dim=16, hidden_dim=16, n_layers=1, n_heads=2, pool='attention'), in_dim=40
    ).eval()
    batch = _batch()
    batch['presence'][1] = False

    valid = model.pooling_mask(batch)
    assert bool(valid[1].all())
    assert torch.isfinite(model.sentence_hidden(batch)).all()
