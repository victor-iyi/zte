"""Tests for device handling, frontends, the ZTE model and every objective."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from zte.config import ModelConfig, ObjectiveConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.torch_dataset import collate_sentences
from zte.device import autocast, resolve_device
from zte.models.embedding import build_model
from zte.models.objectives import build_objective


def _band_batch(dataset: ZuCoDataset, n: int = 4) -> dict:
    """Collates the first `n` sentences of a dataset into a batch dict."""
    torch_ds = dataset.to_torch(representation='both')
    samples = [torch_ds[i] for i in range(min(n, len(torch_ds)))]
    return collate_sentences(samples)


def test_device_cpu_and_autocast() -> None:
    """CPU device resolves with AMP disabled and a no-op autocast context."""
    spec = resolve_device('cpu')
    assert spec.kind == 'cpu'
    assert spec.use_amp is False
    with autocast(spec):
        y = torch.ones(2, 2) @ torch.ones(2, 2)
    assert y.shape == (2, 2)


def test_config_yaml_roundtrip(tmp_path: Path) -> None:
    """A config survives a YAML save/load with tuple fields preserved."""
    cfg = ZTEConfig()
    cfg.objective.name = 'masked'
    cfg.dataset.tasks = ('SR', 'NR', 'TSR')
    path = cfg.to_yaml(tmp_path / 'cfg.yaml')
    back = ZTEConfig.from_yaml(path)
    assert back.objective.name == 'masked'
    assert back.dataset.tasks == ('SR', 'NR', 'TSR')
    assert isinstance(back.dataset.tasks, tuple)


def test_band_power_model_forward(small_dataset: ZuCoDataset) -> None:
    """Band-power model returns (B, L, embed_dim) for both context modes."""
    batch = _band_batch(small_dataset)
    in_dim = small_dataset.features.shape[1]
    model = build_model(
        ModelConfig(frontend='band_power_mlp', embed_dim=64, hidden_dim=48), in_dim=in_dim
    )
    tok = model(batch, contextual=False)
    ctx = model(batch, contextual=True)
    b, length = batch['pad_mask'].shape
    assert tok.shape == (b, length, 64)
    assert ctx.shape == (b, length, 64)
    assert model.embed_sentence(batch).shape == (b, 64)


def test_raw_conformer_forward(small_dataset: ZuCoDataset) -> None:
    """Raw Conformer frontend consumes (B, L, C, T) windows."""
    batch = _band_batch(small_dataset)
    c, t = small_dataset.raw_eeg.shape[1], small_dataset.raw_eeg.shape[2]
    model = build_model(
        ModelConfig(frontend='raw_conformer', embed_dim=32, hidden_dim=32, conformer_filters=16),
        raw_shape=(c, t),
    )
    out = model(batch, contextual=True)
    assert out.shape[-1] == 32


@pytest.mark.parametrize('name', ['skipgram', 'cbow', 'masked', 'cpc'])
def test_objective_forward_and_backward(small_dataset: ZuCoDataset, name: str) -> None:
    """Each objective yields a finite scalar loss that backpropagates."""
    torch.manual_seed(0)
    batch = _band_batch(small_dataset, n=6)
    in_dim = small_dataset.features.shape[1]
    model = build_model(
        ModelConfig(frontend='band_power_mlp', embed_dim=48, hidden_dim=40), in_dim=in_dim
    )
    objective = build_objective(
        ObjectiveConfig(name=name, mask_ratio=0.5, cpc_steps=2), model, feature_dim=in_dim
    )

    loss, metrics = objective.compute(model, batch)
    assert torch.isfinite(loss), f'{name} produced non-finite loss'
    assert 'loss' in metrics
    loss.backward()
    grad_norm = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, f'{name} produced no gradients on the encoder'


def test_masked_latent_teacher_updates(small_dataset: ZuCoDataset) -> None:
    """The data2vec EMA teacher tracks the student after a post_step."""
    in_dim = small_dataset.features.shape[1]
    model = build_model(
        ModelConfig(frontend='band_power_mlp', embed_dim=32, hidden_dim=32), in_dim=in_dim
    )
    objective = build_objective(
        ObjectiveConfig(name='masked', masked_target='latent'), model, feature_dim=in_dim
    )
    before = next(objective.teacher.module.parameters()).clone()
    # Perturb the student then update the teacher.
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    objective.post_step(model)
    after = next(objective.teacher.module.parameters())
    assert not torch.equal(before, after)


@pytest.mark.parametrize('frontend', ['band_power_mlp', 'raw_conformer'])
def test_masked_reconstruct_both_frontends(small_dataset: ZuCoDataset, frontend: str) -> None:
    """Masked 'reconstruct' target works for band-power AND raw frontends.

    Regression test: the raw frontend must reconstruct the flattened raw window
    (C*T), not the band-power `features` (which are absent for a raw model).
    """
    torch.manual_seed(0)
    batch = _band_batch(small_dataset, n=6)
    in_dim = small_dataset.features.shape[1]
    channels, time = small_dataset.raw_eeg.shape[1], small_dataset.raw_eeg.shape[2]
    if frontend == 'band_power_mlp':
        model = build_model(
            ModelConfig(frontend=frontend, embed_dim=32, hidden_dim=32, n_layers=2), in_dim=in_dim
        )
        recon_dim = in_dim
    else:
        model = build_model(
            ModelConfig(
                frontend=frontend, embed_dim=32, hidden_dim=32, n_layers=2, conformer_filters=16
            ),
            raw_shape=(channels, time),
        )
        recon_dim = channels * time
    objective = build_objective(
        ObjectiveConfig(name='masked', masked_target='reconstruct', mask_ratio=0.5),
        model,
        feature_dim=recon_dim,
    )
    loss, metrics = objective.compute(model, batch)
    assert torch.isfinite(loss)
    assert 'loss' in metrics
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
