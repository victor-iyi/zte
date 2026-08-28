"""Tests for the EEGNet and DeepConvNet baseline frontends and the frontend registry that builds them."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

import pytest
import torch

from zte.config import ModelConfig
from zte.data.dataset import ZuCoDataset
from zte.data.torch_dataset import collate_sentences
from zte.models.embedding import build_model
from zte.models.frontends import DeepConvNet, EEGNet, build_frontend

# ZuCo's live raw window is 350 samples (700 ms at 500 Hz); the archived configs use 128. Both must be exercised.
LIVE_WINDOW = 350
ARCHIVED_WINDOW = 128


def _config(frontend: str, **overrides: object) -> ModelConfig:
    """Builds a small model config for the named baseline frontend."""
    fields: dict[str, object] = {'frontend': frontend, 'embed_dim': 32, 'hidden_dim': 24} | overrides

    return ModelConfig(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize('frontend', ['eegnet', 'deep_conv_net'])
@pytest.mark.parametrize('lead', [(7,), (3, 5), (2, 3, 4)])
def test_baseline_frontend_maps_arbitrary_leading_dims(frontend: str, lead: tuple[int, ...]) -> None:
    """Both baselines map `(..., n_channels, time_steps)` to `(..., hidden_dim)` for any leading shape."""
    config = _config(frontend, deepconv_filters=(8, 16))
    module = build_frontend(config, None, (16, ARCHIVED_WINDOW))

    out = module(torch.randn(*lead, 16, ARCHIVED_WINDOW))

    assert out.shape == (*lead, config.hidden_dim)


@pytest.mark.parametrize('frontend', ['eegnet', 'deep_conv_net'])
def test_baseline_out_dim_equals_hidden_dim(frontend: str) -> None:
    """`out_dim` matches `hidden_dim`, which is what the subject FiLM table and the transformer are sized from."""
    config = _config(frontend, hidden_dim=48, deepconv_filters=(8, 16))
    module = build_frontend(config, None, (16, ARCHIVED_WINDOW))

    assert module.out_dim == 48
    assert module(torch.randn(2, 3, 16, ARCHIVED_WINDOW)).shape[-1] == 48


@pytest.mark.parametrize('frontend', ['eegnet', 'deep_conv_net'])
def test_baseline_sub_tokens_shape_and_span_sensitivity(frontend: str) -> None:
    """`sub_tokens` returns `(..., n_sub, hidden_dim)` and each sub-token reads a different span of the window."""
    config = _config(frontend, deepconv_filters=(8, 16))
    module = build_frontend(config, None, (16, LIVE_WINDOW)).eval()
    x = torch.randn(2, 3, 16, LIVE_WINDOW)

    sub_tokens = cast(Callable[[torch.Tensor, int], torch.Tensor], module.sub_tokens)
    with torch.no_grad():
        sub = sub_tokens(x, 4)

    assert sub.shape == (2, 3, 4, config.hidden_dim)
    # Distinct spans of one word's time course must not collapse to the same vector, or the token alignment level
    # would be aligning four copies of the same thing against four different word-pieces.
    assert not torch.allclose(sub[..., 0, :], sub[..., 3, :])


@pytest.mark.parametrize('frontend', ['eegnet', 'deep_conv_net'])
def test_baseline_sub_tokens_rejects_non_positive_count(frontend: str) -> None:
    """`sub_tokens` refuses a non-positive `n_sub` rather than returning an empty axis."""
    module = build_frontend(_config(frontend, deepconv_filters=(8, 16)), None, (16, ARCHIVED_WINDOW))

    sub_tokens = cast(Callable[[torch.Tensor, int], torch.Tensor], module.sub_tokens)
    with pytest.raises(ValueError, match='n_sub must be positive'):
        sub_tokens(torch.randn(2, 16, ARCHIVED_WINDOW), 0)


def test_eegnet_handles_both_windows_and_shrinks_its_pool_schedule() -> None:
    """EEGNet runs at 350 and at 128 time steps, and at a window shorter than its own pool factors."""
    for time_steps in (LIVE_WINDOW, ARCHIVED_WINDOW, 3):
        module = EEGNet(16, time_steps, _config('eegnet'))
        assert module(torch.randn(2, 16, time_steps)).shape == (2, 24)


def test_deep_conv_net_runs_four_blocks_at_the_live_window() -> None:
    """The published four-block DeepConvNet survives ZuCo's 350-step window."""
    module = DeepConvNet(16, LIVE_WINDOW, _config('deep_conv_net'))

    assert module(torch.randn(2, 16, LIVE_WINDOW)).shape == (2, 24)


def test_deep_conv_net_raises_on_a_window_too_short_for_its_depth() -> None:
    """Four blocks over the archived 128-step window raise, naming the block and the knob that fixes it."""
    with pytest.raises(ValueError, match=r'block 4 is left with a length-2 time axis'):
        DeepConvNet(16, ARCHIVED_WINDOW, _config('deep_conv_net'))


def test_deep_conv_net_fits_the_archived_window_at_a_shallower_depth() -> None:
    """Dropping to three blocks is what makes the 128-step window run, and the error message says so."""
    module = DeepConvNet(16, ARCHIVED_WINDOW, _config('deep_conv_net', deepconv_filters=(25, 50, 100)))

    assert module(torch.randn(2, 16, ARCHIVED_WINDOW)).shape == (2, 24)


def test_deep_conv_net_never_produces_a_zero_length_time_axis() -> None:
    """The pool schedule is clamped per block, so a window that survives the convolutions keeps a non-empty axis."""
    module = DeepConvNet(16, 20, _config('deep_conv_net', deepconv_filters=(8, 16), deepconv_kernel=5))

    assert module(torch.randn(2, 16, 20)).shape == (2, 24)


def test_build_frontend_raises_on_an_unknown_name() -> None:
    """An unrecognised `frontend` is an error, not a silent fallthrough to the raw conformer."""
    config = _config('band_power_mlp')
    config.frontend = 'raw_confromer'  # type: ignore[assignment]

    with pytest.raises(ValueError, match='Unknown frontend'):
        build_frontend(config, 64, (16, ARCHIVED_WINDOW))


def test_build_frontend_still_dispatches_every_known_name(small_dataset: ZuCoDataset) -> None:
    """Each name in the frontend literal builds its own class, so the match arms stay in step with the type."""
    features = small_dataset.features
    assert features is not None
    in_dim = features.shape[1]
    raw = small_dataset.raw_eeg
    assert raw is not None
    channels, time_steps = raw.shape[1], raw.shape[2]
    expected = {'band_power_mlp': 'BandPowerMLP', 'raw_conformer': 'RawConformer', 'eegnet': 'EEGNet'}

    for name, class_name in expected.items():
        built = build_frontend(_config(name, conformer_filters=8), in_dim, (channels, time_steps))
        assert type(built).__name__ == class_name

    deep = build_frontend(_config('deep_conv_net', deepconv_filters=(8,)), in_dim, (channels, time_steps))
    assert type(deep).__name__ == 'DeepConvNet'


def test_stacked_spatial_encoding_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Putting a spatial mixer in front of a baseline's own electrode filter warns instead of passing silently."""
    config = _config('eegnet', spatial_encoding='spherical_harmonics', spatial_harmonic_degree=2)

    with caplog.at_level(logging.WARNING, logger='zte.models.frontends'):
        module = build_frontend(config, None, (16, ARCHIVED_WINDOW), n_channels=16)

    # The mixer is applied, not dropped -- the warning exists so the run's log records the double count.
    assert module.spatial_mixer is not None
    assert any('double-spatial ablation' in record.message for record in caplog.records)


def test_no_warning_without_a_spatial_mixer(caplog: pytest.LogCaptureFixture) -> None:
    """The default `spatial_encoding='none'` builds no mixer and emits no double-spatial warning."""
    with caplog.at_level(logging.WARNING, logger='zte.models.frontends'):
        module = build_frontend(_config('eegnet'), None, (16, ARCHIVED_WINDOW), n_channels=16)

    assert module.spatial_mixer is None
    assert not any('double-spatial' in record.message for record in caplog.records)


def test_baseline_frontends_are_batch_independent() -> None:
    """A word's embedding does not depend on which other words it was batched with.

    The frontend is handed padded word slots as all-zero windows and never a mask, so any cross-token normalisation
    would let the padding move a real word's vector -- and a retrieval gallery must be reproducible one word at a time.
    """
    for frontend in ('eegnet', 'deep_conv_net'):
        # Training mode with dropout off: the only thing left that could couple the tokens is a normaliser fitting
        # statistics over the batch, which is exactly what this asserts is absent.
        config = _config(frontend, deepconv_filters=(8, 16), eegnet_dropout=0.0, deepconv_dropout=0.0)
        module = build_frontend(config, None, (16, ARCHIVED_WINDOW)).train()
        word = torch.randn(1, 16, ARCHIVED_WINDOW)
        padded = torch.cat([word, torch.zeros(5, 16, ARCHIVED_WINDOW)], dim=0)

        with torch.no_grad():
            alone = module(word)
            batched = module(padded)[:1]

        assert torch.allclose(alone, batched, atol=1e-5), f'{frontend} embedding moved when padding shared the batch'


@pytest.mark.parametrize('frontend', ['eegnet', 'deep_conv_net'])
def test_baseline_model_trains_end_to_end(small_dataset: ZuCoDataset, frontend: str) -> None:
    """A full `ZTEModel` on either baseline embeds a collated batch and backpropagates into the frontend."""
    torch.manual_seed(0)
    torch_ds = small_dataset.to_torch(representation='both')
    batch = collate_sentences([torch_ds[i] for i in range(4)])
    raw = small_dataset.raw_eeg
    assert raw is not None
    channels, time_steps = raw.shape[1], raw.shape[2]

    model = build_model(_config(frontend, deepconv_filters=(8, 16)), raw_shape=(channels, time_steps))
    assert model.uses_raw is True

    out = model(batch, contextual=True)
    assert out.shape == (*batch['pad_mask'].shape, 32)

    # `embed_sentence` is `@torch.no_grad()`; `sentence_hidden` is its differentiable half.
    model.project(model.sentence_hidden(batch)).sum().backward()
    grad = sum(float(p.grad.abs().sum()) for p in model.frontend.parameters() if p.grad is not None)
    assert grad > 0, f'{frontend} received no gradient'
