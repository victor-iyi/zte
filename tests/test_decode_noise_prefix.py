"""The Gaussian noise-prefix control: the arm that measures how much of a decode is the frozen LM's own prior.

The published claim this answers is blunt -- a standard decoder handed pure noise scores as well on content metrics as
one handed EEG. `noise` cannot answer it, because it destroys the encoder's *input* and still runs the result through
the encoder, so the bridge is handed a vector the encoder produced. `noise_prefix` hands the bridge a moment-matched
vector no encoder ever produced, decodes it through the identical `generate_from_prefix` call the headline uses, and
fails its verdict clause loudly rather than quietly returning the real decode.
"""

import os
from typing import Any, Final

# The whole decoder test path is offline: `lm_source='tiny'` builds its LM and tokeniser locally.
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import pandas as pd
import pytest
import torch

from zte.cli.decode import CONTROLS, DecodeOptions, _controls, noise_prefix_z
from zte.config import DecoderConfig, ModelConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.device import resolve_device
from zte.evaluation.generation import generation_report
from zte.evaluation.report import generation_verdict
from zte.inference.decode import ReadingBatch, ZTEDecoder
from zte.models.decoder import GapCorrector, build_bridge, build_lm
from zte.models.embedding import build_model

_Z_DIM: Final[int] = 16
"""Width of the conditioning vectors the tiny bridge reads."""

_REFERENCES: Final[tuple[str, ...]] = (
    'the surgeon rebuilt the shattered wrist',
    'a diplomat resigned over the treaty',
    'volcanic ash grounded every flight',
    'the orchestra rehearsed until midnight',
    'researchers sequenced the coral genome',
    'floodwater closed the coastal highway',
)
"""Six references with disjoint content words, so a paired content-F1 delta has something to move on."""

# Far above the conditioning cloud's own scale, so a surrogate drawn from N(0, I) instead of the matched moments
# would sit an order of magnitude away and the moment assertions would catch the substitution.
_OFFSET: Final[float] = 40.0
"""Per-feature mean the synthetic conditioning cloud is shifted by."""


def _tiny_decoder() -> ZTEDecoder:
    """An untrained decoder over the offline tiny LM: the control's wiring needs shapes, not learned weights."""
    torch.manual_seed(0)
    decoder_config = DecoderConfig(
        lm_source='tiny',
        tokenizer_source='tiny',
        max_target_tokens=24,
        max_new_tokens=8,
        prefix_slots=2,
        bottleneck=8,
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


def _cloud(n: int, seed: int = 0) -> np.ndarray:
    """A conditioning cloud that is offset and anisotropic, so a matched draw is distinguishable from N(0, I)."""
    rng = np.random.default_rng(seed)
    scale = np.linspace(0.5, 4.0, _Z_DIM, dtype=np.float32)

    return (rng.standard_normal((n, _Z_DIM)).astype(np.float32) * scale + _OFFSET).astype(np.float32)


def _readings(seed: int = 0) -> ReadingBatch:
    """One conditioning batch per reference, with the metadata the scored blocks read back."""
    n = len(_REFERENCES)

    return ReadingBatch(
        z=_cloud(n, seed),
        meta=pd.DataFrame(
            {
                'text': list(_REFERENCES),
                'text_id': np.arange(n, dtype=np.int64),
                'n_words': [len(t.split()) for t in _REFERENCES],
            }
        ),
    )


@pytest.fixture()
def decoder() -> ZTEDecoder:
    """The untrained tiny decoder every arm below decodes through."""
    return _tiny_decoder()


@pytest.fixture()
def readings() -> ReadingBatch:
    """The conditioning batch the headline and the control share."""
    return _readings()


def _noise_prefix_arm(
    decoder: ZTEDecoder, dataset: ZuCoDataset, readings: ReadingBatch
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Runs the control ladder restricted to `noise_prefix`, which touches no split and no dataset row."""
    options = DecodeOptions(controls=('noise_prefix',), batch_size=4, oracle=False, rescore=False)

    return _controls(decoder, dataset, None, readings, options, ZTEConfig())


# --------------------------------------------------------------------------- #
# The surrogate itself
# --------------------------------------------------------------------------- #
def test_the_noise_prefix_is_moment_matched_and_not_a_standard_normal() -> None:
    """An off-manifold N(0, I) prefix is a trivially weak control; the matched draw is the empirical floor.

    Note:
        The assertion is on both sides on purpose. Tracking the real per-feature mean and standard deviation is what
        makes the arm a floor rather than a straw man, and being far from N(0, I) is what proves the matching is the
        code path taken -- a standard normal would pass the first clause vacuously on a centred cloud.
    """
    z = _cloud(8192)
    surrogate = noise_prefix_z(z, seed=0)
    spread = surrogate.std(axis=0)

    assert surrogate.shape == z.shape
    assert surrogate.dtype == np.float32
    assert np.allclose(surrogate.mean(axis=0), z.mean(axis=0), atol=0.25)
    assert np.allclose(spread, z.std(axis=0), rtol=0.06)

    # What a standard normal could not reproduce: the cloud's offset and its per-feature spread ratio.
    assert float(np.abs(surrogate.mean(axis=0)).min()) > 0.5 * _OFFSET
    assert float(spread.max() / spread.min()) > 4.0


def test_the_noise_prefix_is_reproducible_from_its_seed() -> None:
    """A reported control that cannot be redrawn is not a pre-registered control."""
    z = _readings().z

    assert np.array_equal(noise_prefix_z(z, seed=3), noise_prefix_z(z, seed=3))
    assert not np.array_equal(noise_prefix_z(z, seed=3), noise_prefix_z(z, seed=4))


@pytest.mark.parametrize(
    ('z', 'message'),
    [
        (np.zeros((0, _Z_DIM), dtype=np.float32), 'non-empty'),
        (np.zeros((4, 3, 2), dtype=np.float32), 'non-empty'),
        (np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32), 'non-finite'),
        (np.full((6, _Z_DIM), 2.5, dtype=np.float32), 'no per-feature variance'),
    ],
)
def test_the_noise_prefix_refuses_a_cloud_it_cannot_perturb(z: np.ndarray, message: str) -> None:
    """Refusing loudly is the whole point: a silent fallback to `z` decodes the headline under a control's name."""
    with pytest.raises(ValueError, match=message):
        noise_prefix_z(z)


# --------------------------------------------------------------------------- #
# The arm inside the control ladder
# --------------------------------------------------------------------------- #
def test_the_noise_prefix_arm_decodes_a_prefix_no_encoder_produced(
    decoder: ZTEDecoder,
    readings: ReadingBatch,
    small_dataset: ZuCoDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The arm reaches the identical decode call the headline uses, carrying a prefix built from noise.

    Note:
        Both halves matter. A control decoded through a different call has a meaningless paired delta, and a control
        carrying the real prefix is the headline wearing a control's label.
    """
    prefixes: list[torch.Tensor] = []
    real = decoder.lm.generate_from_prefix

    def spy(prefix: torch.Tensor, *args: Any, **kwargs: Any) -> list[str]:
        prefixes.append(prefix.detach().clone())
        return real(prefix, *args, **kwargs)

    monkeypatch.setattr(decoder.lm, 'generate_from_prefix', spy)
    headline = decoder.generate(readings, batch_size=4)
    seen_by_headline = len(prefixes)
    controls, unavailable = _noise_prefix_arm(decoder, small_dataset, readings)

    assert unavailable == {}
    assert set(controls) == {'noise_prefix'}
    assert len(controls['noise_prefix']) == len(headline)

    # One decode call per span for the headline and the same number for the control: one shared code path.
    assert len(prefixes) == 2 * seen_by_headline
    for own, surrogate in zip(prefixes[:seen_by_headline], prefixes[seen_by_headline:], strict=True):
        assert own.shape == surrogate.shape
        assert not torch.allclose(own, surrogate), 'the control must not carry the real prefix'

    assert controls['noise_prefix'] != headline, 'a noise prefix that decodes the headline text is not a control'


def test_the_noise_prefix_arm_fails_its_clause_instead_of_returning_the_real_decode(
    decoder: ZTEDecoder, small_dataset: ZuCoDataset
) -> None:
    """A collapsed conditioning cloud leaves nothing to perturb, and the ledger says so rather than passing."""
    collapsed = ReadingBatch(z=np.full((len(_REFERENCES), _Z_DIM), 0.25, dtype=np.float32), meta=_readings().meta)

    controls, unavailable = _noise_prefix_arm(decoder, small_dataset, collapsed)

    assert controls == {}
    assert 'no per-feature variance' in unavailable['noise_prefix']


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #
def test_the_noise_prefix_arm_is_scored_and_named_in_the_verdicts_control_set(
    decoder: ZTEDecoder,
    readings: ReadingBatch,
    small_dataset: ZuCoDataset,
) -> None:
    """The arm produces a paired delta on the primary metric and is a clause the gate ANDs over.

    Note:
        The untrained tiny bridge decodes nothing real, so the delta is expected to be non-positive and the verdict
        is expected to refuse. What is asserted is that the arm is *read*: an unbeaten `noise_prefix` is named in
        `generation_controls_missing`, so it can never be dropped from the AND without leaving a trace.
    """
    hypotheses = decoder.generate(readings, batch_size=4)
    controls, unavailable = _noise_prefix_arm(decoder, small_dataset, readings)
    assert unavailable == {}

    block = generation_report(hypotheses, list(_REFERENCES), controls, split='test', n_boot=64, n_perm=32, seed=0)
    block['split_strategy'] = 'by_subject_and_stimulus'
    block['controls_requested'] = ['noise_prefix']
    block['controls_unavailable'] = unavailable

    assert 'noise_prefix' in block['deltas']
    assert np.isfinite(block['deltas']['noise_prefix']['content_f1']['point'])
    assert block['worst_control'] == 'noise_prefix'

    verdict = generation_verdict(block, min_prefix_kl=0.05)
    assert verdict['generation_above_controls'] is False
    assert 'noise_prefix' in verdict['generation_controls_missing']


def test_a_pre_registered_noise_prefix_tightens_the_generation_gate() -> None:
    """Adding the arm can only ever cost a verdict, which is what makes it worth pre-registering.

    The same decode is scored twice against a fabricated block: once with `noise_prefix` beaten, and once with it
    pre-registered but unavailable. An unavailable control is not a control that was beaten, so the gate must fall.
    """
    beaten = {
        'metric': 'content_f1',
        'point': 0.4,
        'lo': 0.2,
        'hi': 0.6,
        'n': len(_REFERENCES),
        'n_boot': 64,
        'beats': True,
    }
    block: dict[str, Any] = {
        'applicable': True,
        'primary_metric': 'content_f1',
        'split': 'test',
        'split_strategy': 'by_subject_and_stimulus',
        'deltas': {name: {'content_f1': dict(beaten)} for name in ('mean_prefix', 'noise_prefix')},
        'controls_requested': ['mean_prefix', 'noise_prefix'],
        'controls_unavailable': {},
        'permutation': {'applicable': True, 'p_value': 0.001},
        'prefix_influence_kl': 0.5,
        'n_candidate_sentences': None,
        'worst_control': 'noise_prefix',
        'worst_control_ci': dict(beaten),
    }
    assert generation_verdict(block, min_prefix_kl=0.05)['generation_above_controls'] is True

    # The one change: the arm was pre-registered and could not run. Nothing else about the decode moves.
    block['deltas'] = {'mean_prefix': {'content_f1': dict(beaten)}}
    block['controls_unavailable'] = {'noise_prefix': 'the cloud has no per-feature variance left to match'}
    demoted = generation_verdict(block, min_prefix_kl=0.05)

    assert demoted['generation_above_controls'] is False
    assert demoted['generation_clauses']['beats_every_control'] is False
    assert demoted['generation_controls_missing'] == ['noise_prefix']


def test_the_noise_prefix_arm_is_pre_registered_everywhere_a_control_is_named() -> None:
    """A control the CLI knows but the shipped configuration does not is a control no run would ever decode."""
    assert 'noise_prefix' in CONTROLS
    assert 'noise_prefix' in DecoderConfig().generation_controls
    assert 'noise_prefix' in DecodeOptions().controls


# --------------------------------------------------------------------------- #
# Mutation: break the arm and watch the checks go red
# --------------------------------------------------------------------------- #
def test_a_noise_prefix_arm_returning_the_real_z_is_caught(
    decoder: ZTEDecoder,
    readings: ReadingBatch,
    small_dataset: ZuCoDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mutation the arm exists to prevent: the surrogate quietly falling back to the real conditioning vectors.

    Note:
        A passing suite is not evidence the control works. Under this mutation the arm decodes the headline text, its
        paired delta is exactly zero, and `beats` is False -- so the assertions above go red rather than the gate
        passing on a control that measured nothing.
    """
    monkeypatch.setattr('zte.cli.decode.noise_prefix_z', lambda z, seed=0: np.asarray(z, dtype=np.float32))

    hypotheses = decoder.generate(readings, batch_size=4)
    controls, unavailable = _noise_prefix_arm(decoder, small_dataset, readings)

    assert unavailable == {}
    assert controls['noise_prefix'] == hypotheses, 'the mutation is exactly a headline decode under a control name'

    block = generation_report(hypotheses, list(_REFERENCES), controls, split='test', n_boot=64, n_perm=32, seed=0)
    delta = block['deltas']['noise_prefix']['content_f1']

    assert delta['point'] == pytest.approx(0.0, abs=1e-12)
    assert delta['beats'] is False
