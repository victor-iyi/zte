"""Tests for Unit A (meaning distillation + hard negatives) and Unit B (behaviour)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from zte.config import DatasetConfig, ModelConfig, ObjectiveConfig
from zte.data.dataset import ZuCoDataset
from zte.data.synthetic import generate_synthetic_zuco
from zte.data.targets.behaviour import build_behaviour_matrix
from zte.data.targets.meaning import build_meaning_matrix
from zte.data.torch_dataset import make_dataloader
from zte.models.embedding import build_model
from zte.models.frontends import BandRoutedMLP
from zte.models.objectives import build_objective


@pytest.fixture(scope='module')
def three_subject_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a synthetic ZuCo dataset with three subjects and two tasks."""
    out = tmp_path_factory.mktemp('zuco_ab')
    generate_synthetic_zuco(
        out, subjects=('ZAB', 'ZDM', 'ZPH'), tasks=('SR', 'NR'), n_sentences=10, show_progress=False
    )
    return out


def _dataset(root: Path, tmp: Path) -> ZuCoDataset:
    return ZuCoDataset(DatasetConfig(root=str(root), cache_dir=str(tmp / 'cache'))).build(
        show_progress=False
    )


# --- meaning teacher ------------------------------------------------------- #


def test_meaning_matrix_hash_is_deterministic_and_normalised() -> None:
    """Test that the meaning matrix is deterministic and normalised."""
    vocab = {'the': 0, 'brain': 1, 'reads': 2}
    a = build_meaning_matrix(vocab, source='hash', dim=32)
    b = build_meaning_matrix(vocab, source=None, dim=32)
    assert a.shape == (3, 32)
    assert np.allclose(a, b)  # deterministic
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)  # L2-normalised


def test_meaning_matrix_from_glove_file(tmp_path: Path) -> None:
    """Test that the meaning matrix is built from a GloVe file."""
    f = tmp_path / 'vecs.txt'
    f.write_text('brain 1 0 0\nreads 0 1 0\n', encoding='utf-8')
    mat = build_meaning_matrix({'brain': 0, 'reads': 1, 'oov': 2}, source=str(f), dim=3)
    assert mat.shape == (3, 3)
    assert np.allclose(mat[0], [1, 0, 0])  # aligned to vocab id
    assert np.isfinite(mat[2]).all()  # OOV -> mean vector, still finite


# --- behaviour targets ----------------------------------------------------- #


def test_behaviour_matrix_regression_time_and_binary(
    three_subject_dir: Path, tmp_path: Path
) -> None:
    """Test that the behaviour matrix is built correctly."""
    ds = _dataset(three_subject_dir, tmp_path)
    mat, names, binary = build_behaviour_matrix(ds.words, ('TRT', 'regression_time', 'is_omitted'))
    assert mat.shape == (len(ds.words), len(names))  # type: ignore[index]
    assert 'is_omitted' in names and binary[names.index('is_omitted')]
    # is_omitted column is exactly 0/1 (no NaN); TRT has NaN on skipped words.
    om = mat[:, names.index('is_omitted')]
    assert set(np.unique(om[np.isfinite(om)]).tolist()) <= {0.0, 1.0}


# --- Unit A/B wired through the objective ---------------------------------- #


def test_meaning_distillation_trains(three_subject_dir: Path, tmp_path: Path) -> None:
    """Test that meaning distillation trains."""
    ds = _dataset(three_subject_dir, tmp_path)
    model = build_model(
        ModelConfig(embed_dim=48, hidden_dim=32, n_subjects=3), in_dim=ds.features.shape[1]
    )
    obj = build_objective(
        ObjectiveConfig(name='skipgram', meaning_distill_weight=1.0, meaning_dim=16), model
    )
    td = ds.to_torch()
    mat = build_meaning_matrix(td.word_vocab, source='hash', dim=16)
    obj.attach_auxiliary(meaning_matrix=torch.from_numpy(mat))
    loss, metrics = obj.compute(model, next(iter(make_dataloader(td, batch_size=8))))
    assert torch.isfinite(loss)
    assert 'meaning_loss' in metrics and metrics['meaning_loss'] >= 0.0
    loss.backward()
    grad = sum(
        float(p.grad.abs().sum()) for p in obj.meaning_head.parameters() if p.grad is not None
    )
    assert grad > 0.0  # the meaning head actually trains


def test_behaviour_supervision_trains(three_subject_dir: Path, tmp_path: Path) -> None:
    """Test that behaviour supervision trains."""
    ds = _dataset(three_subject_dir, tmp_path)
    model = build_model(
        ModelConfig(embed_dim=48, hidden_dim=32, n_subjects=3), in_dim=ds.features.shape[1]
    )
    targets = ('TRT', 'regression_time', 'is_omitted')
    obj = build_objective(
        ObjectiveConfig(name='skipgram', behaviour_weight=1.0, behaviour_targets=targets), model
    )
    td = ds.to_torch(behaviour_targets=targets)
    obj.attach_auxiliary(behaviour_binary=torch.from_numpy(td.behaviour_binary))
    batch = next(iter(make_dataloader(td, batch_size=8)))
    assert batch['behaviour_target'] is not None
    loss, metrics = obj.compute(model, batch)
    assert torch.isfinite(loss)
    assert 'behaviour_loss' in metrics
    loss.backward()
    grad = sum(
        float(p.grad.abs().sum()) for p in obj.behaviour_head.parameters() if p.grad is not None
    )
    assert grad > 0.0


def test_hard_negatives_run(three_subject_dir: Path, tmp_path: Path) -> None:
    """Test that hard negatives run."""
    ds = _dataset(three_subject_dir, tmp_path)
    model = build_model(
        ModelConfig(embed_dim=48, hidden_dim=32, n_subjects=3), in_dim=ds.features.shape[1]
    )
    obj = build_objective(ObjectiveConfig(name='skipgram', hard_negatives=True), model)
    loss, _ = obj.compute(model, next(iter(make_dataloader(ds.to_torch(), batch_size=8))))
    assert torch.isfinite(loss)


def test_aux_heads_absent_when_disabled(three_subject_dir: Path, tmp_path: Path) -> None:
    """Test that auxiliary heads are absent when disabled."""
    ds = _dataset(three_subject_dir, tmp_path)
    model = build_model(
        ModelConfig(embed_dim=48, hidden_dim=32, n_subjects=3), in_dim=ds.features.shape[1]
    )
    obj = build_objective(ObjectiveConfig(name='skipgram'), model)
    assert obj.meaning_head is None and obj.behaviour_head is None


def test_band_routing_frontend_selected_and_trains(three_subject_dir: Path, tmp_path: Path) -> None:
    """Test that band routing selects the split-pathway frontend and trains end-to-end."""
    ds = _dataset(three_subject_dir, tmp_path)
    in_dim = ds.features.shape[1]
    bp = (in_dim - 6) // 105  # 6 trailing eye-tracking columns, 105 channels
    model = build_model(
        ModelConfig(embed_dim=48, hidden_dim=32, n_subjects=3, band_routing=True),
        in_dim=in_dim,
        n_channels=105,
        bp_features_per_channel=bp,
    )
    assert isinstance(model.frontend, BandRoutedMLP)
    obj = build_objective(ObjectiveConfig(name='skipgram'), model)
    loss, _ = obj.compute(model, next(iter(make_dataloader(ds.to_torch(), batch_size=8))))
    assert torch.isfinite(loss)
    loss.backward()


def test_factored_routes_heads_to_content_slice(three_subject_dir: Path, tmp_path: Path) -> None:
    """Test that when factored, meaning + subject-adversary heads act on the content subspace."""
    ds = _dataset(three_subject_dir, tmp_path)
    model = build_model(
        ModelConfig(embed_dim=48, hidden_dim=32, n_subjects=3, factored=True, content_dim=24),
        in_dim=ds.features.shape[1],
    )
    obj = build_objective(
        ObjectiveConfig(
            name='skipgram',
            meaning_distill_weight=1.0,
            meaning_dim=16,
            subject_adversary_weight=0.5,
        ),
        model,
    )
    assert obj._content_dim == 24 and obj._factored
    mat = build_meaning_matrix(ds.to_torch().word_vocab, 'hash', 16)
    obj.attach_auxiliary(meaning_matrix=torch.from_numpy(mat))
    assert (
        obj.meaning_head.in_features == 24 and obj.meaning_head.out_features == 16
    )  # content slice; matrix width
    loss, metrics = obj.compute(model, next(iter(make_dataloader(ds.to_torch(), batch_size=8))))
    assert torch.isfinite(loss) and 'meaning_loss' in metrics and 'adv_loss' in metrics
    loss.backward()  # content-slice adversary + meaning head both differentiate cleanly
