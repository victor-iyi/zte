"""Stage-aware best-checkpoint monitoring across the joint run's A->B boundary."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from zte.config import TrainConfig, ZTEConfig
from zte.training import stages
from zte.training.checkpoint import CheckpointManager
from zte.training.trainer import Trainer

# The defect this file guards: stage B adds the joint auxiliaries to the validation loss, so the monitored scalar
# jumps by several units at the boundary. A lifetime `<` comparison then pins `best.pt` to a stage-A epoch whose
# encoder is bit-for-bit the loaded checkpoint, and patience ends the run -- the joint arm re-measures the frozen
# encoder. The trajectory below reproduces that shape: stage A improves, the boundary jumps, stage B improves but
# never dips under the stage-A best.
_STAGE_A_LOSSES = (3.0, 2.5)
_STAGE_B_LOSSES = (8.0, 7.0, 6.5, 6.0)


def _config(ckpt_dir: Path, *, epochs: int, patience: int = 0) -> ZTEConfig:
    """A joint-mode config whose encoder unfreezes after two stage-A epochs."""
    return ZTEConfig(
        train=TrainConfig(
            mode='joint',
            freeze_encoder=False,
            stage_a_epochs=2,
            early_stop_patience=patience,
            epochs=epochs,
            batch_size=4,
            device='cpu',
            precision='fp32',
            ckpt_dir=str(ckpt_dir),
            test_fraction=0.0,
        ),
    )


def _scripted_trainer(
    config: ZTEConfig,
    monkeypatch: pytest.MonkeyPatch,
    val_losses: Sequence[float],
    *,
    resume: bool = False,
) -> Trainer:
    """A CPU trainer whose epochs are stubs: no batch runs, and `evaluate` replays `val_losses` in order."""
    losses = iter(list(val_losses))

    def fake_epoch(self: Trainer, epoch: int) -> float:  # noqa: ARG001
        return 1.0

    def fake_eval(self: Trainer) -> float:  # noqa: ARG001
        return next(losses)

    monkeypatch.setattr(Trainer, '_train_one_epoch', fake_epoch)
    monkeypatch.setattr(Trainer, 'evaluate', fake_eval)
    loader = DataLoader(TensorDataset(torch.zeros(8, 4)), batch_size=4)
    return Trainer(nn.Linear(4, 4), nn.Linear(4, 4), config, loader, val_loader=loader, resume=resume)


# --------------------------------------------------------------------------- #
# The boundary reset: best.pt can leave stage A, and patience survives the jump
# --------------------------------------------------------------------------- #


def test_best_checkpoint_leaves_stage_a_despite_the_boundary_jump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The monitor restarts at the A->B boundary, so the best stage-B epoch owns `best.pt`."""
    trainer = _scripted_trainer(_config(tmp_path, epochs=6), monkeypatch, _STAGE_A_LOSSES + _STAGE_B_LOSSES)
    trainer.train()

    best = CheckpointManager.load(tmp_path / 'best.pt')
    assert best['epoch'] == 6, 'the best stage-B epoch, not the last stage-A one'
    assert best['extra']['stage'] == stages.STAGE_B
    assert best['extra']['best_metric'] == pytest.approx(6.0)


def test_without_the_boundary_reset_best_pt_locks_to_stage_a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation the fix exists for: with the reset disabled, `best.pt` never leaves the frozen-encoder stage.

    Note:
        This is what shipped the wrong measurement -- evaluation read a `best.pt` whose encoder was byte-identical
        to the loaded checkpoint, and patience ended the run a few epochs into stage B. If a refactor drops the
        reset, the two tests above start failing exactly like this one passes.
    """
    trainer = _scripted_trainer(_config(tmp_path, epochs=6, patience=2), monkeypatch, _STAGE_A_LOSSES + _STAGE_B_LOSSES)

    def no_reset(self: CheckpointManager) -> None:
        return None

    monkeypatch.setattr(CheckpointManager, 'reset_best', no_reset)
    trainer.train()

    best = CheckpointManager.load(tmp_path / 'best.pt')
    assert best['epoch'] == 2 and best['extra']['stage'] == stages.STAGE_A
    assert len(trainer.history['train_loss']) == 4, 'patience killed the run two epochs into stage B'


def test_patience_survives_the_boundary_jump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A patience shorter than the boundary jump no longer ends the run before stage B is measured."""
    trainer = _scripted_trainer(_config(tmp_path, epochs=6, patience=2), monkeypatch, _STAGE_A_LOSSES + _STAGE_B_LOSSES)
    history = trainer.train()

    assert len(history['train_loss']) == 6, 'every stage-B epoch ran'
    assert CheckpointManager.load(tmp_path / 'best.pt')['epoch'] == 6


# --------------------------------------------------------------------------- #
# Resume: the stage travels in the payload and the monitor state comes back
# --------------------------------------------------------------------------- #


def test_resume_restores_the_stage_and_resets_across_the_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run paused at the end of stage A still resets when the resumed run crosses into stage B."""
    _scripted_trainer(_config(tmp_path, epochs=2), monkeypatch, _STAGE_A_LOSSES).train()
    saved = CheckpointManager.load(tmp_path / 'last.pt')
    assert saved['extra']['stage'] == stages.STAGE_A
    assert saved['extra']['best_metric'] == pytest.approx(2.5)

    resumed = _scripted_trainer(_config(tmp_path, epochs=6), monkeypatch, _STAGE_B_LOSSES, resume=True)
    assert resumed._stage == stages.STAGE_A  # noqa: SLF001 -- the restored monitor state is the behaviour under test
    assert resumed.ckpt.best_metric == pytest.approx(2.5)
    resumed.train()

    best = CheckpointManager.load(tmp_path / 'best.pt')
    assert best['epoch'] == 6 and best['extra']['stage'] == stages.STAGE_B
    assert CheckpointManager.load(tmp_path / 'last.pt')['extra']['best_metric'] == pytest.approx(6.0)


def test_an_old_payload_without_a_stage_resumes_without_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint written before stage tracking loads cleanly and keeps its lifetime-best behaviour."""
    _scripted_trainer(_config(tmp_path, epochs=2), monkeypatch, _STAGE_A_LOSSES).train()
    state = CheckpointManager.load(tmp_path / 'last.pt')
    del state['extra']['stage']
    torch.save(state, tmp_path / 'last.pt')

    resumed = _scripted_trainer(_config(tmp_path, epochs=4), monkeypatch, _STAGE_B_LOSSES[:2], resume=True)
    assert resumed._stage is None  # noqa: SLF001 -- no stage in the payload means no reset gets guessed
    history = resumed.train()

    assert len(history['train_loss']) == 4
    assert CheckpointManager.load(tmp_path / 'best.pt')['epoch'] == 2, 'the pre-fix lifetime comparison, unchanged'


# --------------------------------------------------------------------------- #
# The curriculum contract stages.stage_for_epoch enforces
# --------------------------------------------------------------------------- #


def test_stage_for_epoch_maps_the_curriculum() -> None:
    """Joint runs flip to stage B after `stage_a_epochs`; decoder runs stay in stage A; encoder runs have none."""
    joint = ZTEConfig(train=TrainConfig(mode='joint', freeze_encoder=False, stage_a_epochs=2))
    assert [stages.stage_for_epoch(e, joint) for e in (1, 2, 3)] == [stages.STAGE_A, stages.STAGE_A, stages.STAGE_B]

    eager = ZTEConfig(train=TrainConfig(mode='joint', freeze_encoder=False, stage_a_epochs=0))
    assert stages.stage_for_epoch(1, eager) == stages.STAGE_B

    assert stages.stage_for_epoch(99, ZTEConfig(train=TrainConfig(mode='decoder'))) == stages.STAGE_A
    assert stages.stage_for_epoch(1, ZTEConfig()) is None


def test_a_decoder_run_that_unfreezes_its_encoder_is_refused() -> None:
    """`mode='decoder'` keeps the encoder frozen throughout, so `freeze_encoder=false` is a contradiction."""
    config = ZTEConfig(train=TrainConfig(mode='decoder', freeze_encoder=False))
    with pytest.raises(ValueError, match='freeze_encoder'):
        stages.stage_for_epoch(1, config)
    with pytest.raises(ValueError, match="mode='joint'"):
        stages.apply_stage(1, nn.Linear(2, 2), nn.Linear(2, 2), config)
