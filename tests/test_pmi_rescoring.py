"""PMI gallery rescoring: the null-prefix subtraction that cancels candidate-side familiarity bias."""

import os
from typing import Final

# The whole decoder test path is offline: `lm_source='tiny'` builds its LM and tokeniser locally.
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import pytest
import torch
from torch import nn

from zte.cli.decode import _pmi_vs_raw, _rank_percentiles
from zte.config import DecoderConfig, ModelConfig, ZTEConfig
from zte.device import resolve_device
from zte.inference.decode import ReadingBatch, ZTEDecoder
from zte.models.decoder import EvidenceFn, GapCorrector, build_bridge, build_lm
from zte.models.embedding import build_model

_Z_DIM: Final[int] = 16
"""Width of the conditioning vectors the tiny bridge reads."""

_GALLERY: Final[tuple[str, ...]] = ('the cat sat', 'a dog ran far', 'birds fly south', 'rivers run deep')
"""Four candidate sentences; each planted-bias query's truth is the candidate sharing its index."""


def _tiny_decoder(rescore_pmi: bool = False) -> ZTEDecoder:
    """An untrained decoder over the offline tiny LM: rescoring needs shapes and wiring, not learned weights."""
    torch.manual_seed(0)
    decoder_config = DecoderConfig(
        lm_source='tiny',
        tokenizer_source='tiny',
        max_target_tokens=24,
        prefix_slots=2,
        bottleneck=8,
        rescore_chunk=3,
        rescore_pmi=rescore_pmi,
    )
    model_config = ModelConfig(embed_dim=16, hidden_dim=16, n_layers=1, n_heads=2, projection_hidden=16)
    model = build_model(model_config, in_dim=40)
    lm = build_lm(decoder_config, encoder=model)
    bridge, _ = build_bridge(decoder_config, _Z_DIM, 16, lm.hidden_dim)

    return ZTEDecoder(
        model=model,
        config=ZTEConfig(model=model_config, decoder=decoder_config),
        decoder_config=decoder_config,
        bridge=bridge,
        lm=lm,
        gap=GapCorrector(_Z_DIM, mode='none'),
        device=resolve_device('cpu'),
    )


class _MarkerBridge(nn.Module):
    """Writes each query's true candidate index into slot (0, 0); the null prefix is all -1, so the LM stub can tell."""

    def __init__(self, slots: int = 2, lm_dim: int = 4) -> None:
        super().__init__()
        self.null_prefix = nn.Parameter(torch.full((slots, lm_dim), -1.0), requires_grad=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns a zero prefix carrying `z[:, 0]` -- the query's true candidate index -- as its marker."""
        prefix = torch.zeros(z.shape[0], *self.null_prefix.shape)
        prefix[:, 0, 0] = z[:, 0]

        return prefix

    def null(self, batch_size: int) -> torch.Tensor:
        """Returns the all-negative unconditional prefix broadcast over a batch."""
        return self.null_prefix.unsqueeze(0).expand(batch_size, -1, -1)


def _plant_bias(decoder: ZTEDecoder, bias: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Rigs scoring to logp(c | prefix) = bias[c] + 1[c is the query's truth]: a pure per-candidate familiarity bonus."""
    calls = {'null': 0, 'conditional': 0}
    offsets = torch.from_numpy(np.asarray(bias, dtype=np.float32))

    def scored(
        prefix: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_mask: torch.Tensor,
        length_normalise: bool = True,
        chunk: int = 64,
        evidence: EvidenceFn | None = None,
    ) -> torch.Tensor:
        scores = offsets.unsqueeze(0).repeat(int(prefix.shape[0]), 1)
        marker = prefix[:, 0, 0]
        if bool((marker < 0).all()):
            calls['null'] += 1
            return scores

        calls['conditional'] += 1
        for row in range(int(scores.shape[0])):
            scores[row, int(marker[row].item())] += 1.0
        return scores

    monkeypatch.setattr(decoder, 'bridge', _MarkerBridge().eval())
    monkeypatch.setattr(decoder.lm, 'sequence_logprob', scored)
    return calls


def _readings(truths: list[int]) -> ReadingBatch:
    """One query per entry of `truths`, whose conditioning vector names its true candidate."""
    z = np.zeros((len(truths), _Z_DIM), dtype=np.float32)
    z[:, 0] = truths

    return ReadingBatch.from_vectors(z)


def test_knob_off_leaves_the_raw_scores_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default `rescore_pmi=False` scores exactly as the raw likelihood path and never reaches the null branch."""
    assert DecoderConfig().rescore_pmi is False

    decoder = _tiny_decoder()
    readings = ReadingBatch.from_vectors(np.random.default_rng(0).standard_normal((3, _Z_DIM)).astype(np.float32))
    gallery = list(_GALLERY)

    def boom(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError('the null branch must not run when rescore_pmi is off')

    monkeypatch.setattr(decoder, '_null_candidate_scores', boom)
    scores = decoder.rescore(readings, gallery)

    ids, mask = decoder._tokenise(gallery)  # noqa: SLF001 -- pinning the raw path against the LM directly
    prefix = decoder.bridge(torch.from_numpy(readings.z))
    expected = decoder.lm.sequence_logprob(prefix, ids, mask, True, decoder.decoder_config.rescore_chunk, None)
    assert scores.shape == (3, len(gallery))
    assert np.allclose(scores, expected.float().cpu().numpy(), atol=1e-6)
    assert np.array_equal(scores, decoder.rescore(readings, gallery, pmi=False))


def test_pmi_recovers_the_ranking_a_planted_familiarity_bias_distorts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A constant per-candidate likelihood bonus flips the raw ranking; the null-prefix subtraction restores it."""
    decoder = _tiny_decoder()
    bias = np.array([0.0, 2.0, 4.0, 6.0])
    _plant_bias(decoder, bias, monkeypatch)
    readings = _readings([0, 1, 2, 3])

    raw = decoder.rescore(readings, list(_GALLERY), pmi=False)
    pmi = decoder.rescore(readings, list(_GALLERY), pmi=True)

    # The brain's +1 cannot outweigh the planted familiarity, so the raw ranking is the bias ranking.
    assert np.argmax(raw, axis=1).tolist() == [3, 3, 3, 3]
    # The subtraction cancels the bias exactly, leaving only what the query's prefix added.
    assert np.argmax(pmi, axis=1).tolist() == [0, 1, 2, 3]
    assert np.allclose(pmi, raw - bias[None, :], atol=1e-6)


def test_null_scores_are_computed_once_per_gallery(monkeypatch: pytest.MonkeyPatch) -> None:
    """The unconditional gallery pass is query-independent and runs exactly once, however many query batches run."""
    decoder = _tiny_decoder()
    calls = _plant_bias(decoder, np.zeros(len(_GALLERY)), monkeypatch)
    readings = _readings([0, 1, 2, 3, 0, 1])

    scores = decoder.rescore(readings, list(_GALLERY), batch_size=2, pmi=True)

    assert scores.shape == (6, len(_GALLERY))
    assert calls == {'null': 1, 'conditional': 3}


def test_config_knob_turns_rescore_into_the_pmi_score() -> None:
    """With `rescore_pmi` on, `rescore` hands downstream exactly the raw matrix minus the broadcast null scores."""
    decoder = _tiny_decoder(rescore_pmi=True)
    readings = ReadingBatch.from_vectors(np.random.default_rng(1).standard_normal((3, _Z_DIM)).astype(np.float32))
    gallery = list(_GALLERY)

    raw = decoder.rescore(readings, gallery, pmi=False)
    null = decoder.null_rescore(gallery)
    pmi = decoder.rescore(readings, gallery)

    assert null.shape == (len(gallery),)
    assert np.allclose(pmi, raw - null[None, :], atol=1e-6)
    # The tiny LM's log-likelihoods are far from zero, so the identity above had something to fail on.
    assert not np.allclose(pmi, raw)


def test_pmi_vs_raw_reports_a_paired_per_query_delta_with_a_ci() -> None:
    """The metrics entry pairs each query's rank percentile under both scores and bounds the delta."""
    gallery_ids = np.arange(4)
    query_ids = np.arange(4)
    bias = np.array([0.0, 2.0, 4.0, 6.0])
    truth = np.eye(4)

    per_raw = _rank_percentiles(truth + bias[None, :], query_ids, gallery_ids)
    assert per_raw.tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])

    block = _pmi_vs_raw(truth, truth + bias[None, :], query_ids, gallery_ids, n_boot=200, seed=0)
    assert block['metric'] == 'rank_percentile'
    assert block['n_queries'] == 4
    assert block['pmi_rank_percentile'] == pytest.approx(1.0)
    assert block['raw_rank_percentile'] == pytest.approx(0.5)
    assert block['per_query_rank_percentile_raw'] == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert block['per_query_rank_percentile_pmi'] == pytest.approx([1.0, 1.0, 1.0, 1.0])

    delta = block['rank_percentile_delta']
    assert delta['point'] == pytest.approx(0.5)
    assert delta['lo'] <= delta['point'] <= delta['hi']
