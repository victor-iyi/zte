"""Pause / resume of interrupted training runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from zte.config import DatasetConfig, ModelConfig, ObjectiveConfig, TrainConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.synthetic import generate_synthetic_zuco
from zte.training import trainer as trainer_mod
from zte.training.checkpoint import CheckpointManager
from zte.training.pipeline import run_training


@pytest.fixture(scope='module')
def dataset(tmp_path_factory: pytest.TempPathFactory) -> ZuCoDataset:
    """A tiny built dataset for resume tests."""
    root = tmp_path_factory.mktemp('zuco_resume')
    generate_synthetic_zuco(root, subjects=('ZAB', 'ZDM'), tasks=('SR', 'NR'), n_sentences=6, show_progress=False)
    cfg = DatasetConfig(root=str(root), cache_dir=str(root / 'cache'))
    return ZuCoDataset(cfg).build(show_progress=False)


def _config(ckpt_dir: Path, epochs: int) -> ZTEConfig:
    return ZTEConfig(
        model=ModelConfig(embed_dim=32, hidden_dim=24, n_layers=1, projection_hidden=24, n_subjects=2),
        objective=ObjectiveConfig(name='skipgram'),
        train=TrainConfig(
            epochs=epochs,
            batch_size=8,
            device='cpu',
            precision='fp32',
            tensorboard=False,
            ckpt_dir=str(ckpt_dir),
            test_fraction=0.0,
        ),
    )


def test_resume_continues_not_restarts(dataset: ZuCoDataset, tmp_path: Path) -> None:
    """Resuming picks up at the next epoch and preserves earlier history."""
    ckpt = tmp_path / 'ckpts'
    run_training(_config(ckpt, epochs=2), dataset)
    first = CheckpointManager.load(ckpt / 'last.pt')
    assert first['epoch'] == 2

    # Resume with more epochs -> should train only epochs 3 and 4, keeping the first two in history.
    art = run_training(_config(ckpt, epochs=4), dataset, resume=True)
    assert len(art.history['train_loss']) == 4
    assert CheckpointManager.load(ckpt / 'last.pt')['epoch'] == 4


def test_resume_already_complete_is_noop(dataset: ZuCoDataset, tmp_path: Path) -> None:
    """Resuming a finished run does no further training."""
    ckpt = tmp_path / 'ckpts'
    run_training(_config(ckpt, epochs=2), dataset)
    art = run_training(_config(ckpt, epochs=2), dataset, resume=True)
    assert len(art.history['train_loss']) == 2
    assert CheckpointManager.load(ckpt / 'last.pt')['epoch'] == 2


def test_interrupt_then_resume(dataset: ZuCoDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A KeyboardInterrupt mid-run pauses cleanly; resume completes the remaining epochs."""
    ckpt = tmp_path / 'ckpts'
    real_epoch = trainer_mod.Trainer._train_one_epoch

    def interrupt_on_epoch_2(self: trainer_mod.Trainer, epoch: int) -> float:
        if epoch == 2:
            raise KeyboardInterrupt
        return real_epoch(self, epoch)

    monkeypatch.setattr(trainer_mod.Trainer, '_train_one_epoch', interrupt_on_epoch_2)
    with pytest.raises(KeyboardInterrupt):
        run_training(_config(ckpt, epochs=4), dataset)

    # Only epoch 1 completed and was checkpointed.
    assert CheckpointManager.load(ckpt / 'last.pt')['epoch'] == 1

    # Resume without the interrupt -> finishes epochs 2, 3, 4.
    monkeypatch.setattr(trainer_mod.Trainer, '_train_one_epoch', real_epoch)
    art = run_training(_config(ckpt, epochs=4), dataset, resume=True)
    assert CheckpointManager.load(ckpt / 'last.pt')['epoch'] == 4
    assert len(art.history['train_loss']) == 4
