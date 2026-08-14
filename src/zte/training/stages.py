"""Parameter groups and the staged unfreezing curriculum for encoder / decoder / joint runs."""

from __future__ import annotations

from typing import Any

from torch import nn

from zte.config import ZTEConfig
from zte.logging_utils import get_logger

_LOG = get_logger('training.stages')

STAGE_A = 'a'
"""Bridge-only stage: the encoder is frozen and only the decoder head receives gradients."""

STAGE_B = 'b'
"""Joint stage: the encoder trains alongside the bridge at `bridge_lr * encoder_lr_scale`."""


def parameter_groups(model: nn.Module, objective: nn.Module, config: ZTEConfig) -> list[dict[str, Any]]:
    """Builds the optimiser's named parameter groups for this run's training mode.

    Args:
        model (nn.Module): The encoder.
        objective (nn.Module): The objective, which owns the bridge in decoder/joint mode.
        config (ZTEConfig): The full run configuration.

    Returns:
        list[dict[str, Any]]: Groups of `{'params', 'lr', 'weight_decay', 'name'}`, in a stable order.
    """
    train = config.train
    bridge = getattr(objective, 'bridge', None)
    if bridge is None:
        params = list(model.parameters()) + list(objective.parameters())
        return [
            {
                'params': params,
                'lr': train.lr,
                'weight_decay': train.weight_decay,
                'name': 'encoder',
            }
        ]

    resampler = getattr(objective, 'resampler', None)
    # The frozen LM is never handed to the optimiser: it has no gradients and would bloat its state.
    claimed = _param_ids(getattr(objective, 'lm', None))
    groups: list[dict[str, Any]] = []

    for name, module, lr in (
        ('bridge', bridge, train.bridge_lr),
        ('resampler', resampler, train.bridge_lr),
    ):
        members = _unclaimed(module, claimed)
        if members:
            groups.append(
                {
                    'params': members,
                    'lr': lr,
                    'weight_decay': train.weight_decay,
                    'name': name,
                }
            )

    encoder = _unclaimed(model, claimed) + _unclaimed(objective, claimed)
    if encoder:
        groups.append(
            {
                'params': encoder,
                'lr': train.bridge_lr * train.encoder_lr_scale,
                'weight_decay': train.weight_decay,
                'name': 'encoder',
            }
        )
    summary = [f'{g["name"]}={sum(p.numel() for p in g["params"]) / 1e6:.2f}M@{g["lr"]:.2e}' for g in groups]
    _LOG.info('Optimiser groups: %s', ', '.join(summary))
    return groups


def apply_stage(epoch: int, model: nn.Module, objective: nn.Module, config: ZTEConfig) -> bool:
    """Applies the curriculum for `epoch` and reports whether the trainable parameter set changed.

    Args:
        epoch (int): The 1-based epoch about to run.
        model (nn.Module): The encoder.
        objective (nn.Module): The objective, notified through `set_stage` when it implements it.
        config (ZTEConfig): The full run configuration.

    Returns:
        bool: `True` when the encoder's `requires_grad` flipped, so the caller must rebuild any cached parameter list.
    """
    train = config.train
    if train.mode == 'encoder':
        return False
    stage = STAGE_B if _encoder_trains(epoch, config) else STAGE_A
    set_stage = getattr(objective, 'set_stage', None)
    if callable(set_stage):
        set_stage(stage)
    changed = _set_trainable(model, stage == STAGE_B)
    if changed:
        _LOG.info(
            'Epoch %d enters stage %s: encoder %s.',
            epoch,
            stage.upper(),
            'unfrozen' if stage == STAGE_B else 'frozen',
        )
    return changed


def trainable_parameters(model: nn.Module, objective: nn.Module) -> list[nn.Parameter]:
    """Returns every parameter that can receive a gradient, encoder first, without duplicates.

    Args:
        model (nn.Module): The encoder.
        objective (nn.Module): The objective.

    Returns:
        list[nn.Parameter]: The parameters to clip, ordered encoder first and then objective.
    """
    seen: set[int] = set()
    out: list[nn.Parameter] = []
    for param in list(model.parameters()) + list(objective.parameters()):
        if param.requires_grad and id(param) not in seen:
            seen.add(id(param))
            out.append(param)
    return out


def _encoder_trains(epoch: int, config: ZTEConfig) -> bool:
    """Returns whether the encoder should receive gradients during `epoch`."""
    train = config.train
    if train.freeze_encoder:
        return False
    return train.mode != 'joint' or epoch > train.stage_a_epochs


def _set_trainable(module: nn.Module, trainable: bool) -> bool:
    """Sets `requires_grad` across `module` and returns whether that changed anything."""
    if any(p.requires_grad for p in module.parameters()) == trainable:
        return False
    module.requires_grad_(trainable)
    return True


def _param_ids(module: nn.Module | None) -> set[int]:
    """Returns the identity set of a submodule's parameters, or an empty set when it is absent."""
    return set() if module is None else {id(p) for p in module.parameters()}


def _unclaimed(module: nn.Module | None, claimed: set[int]) -> list[nn.Parameter]:
    """Returns `module`'s parameters that no earlier group took, marking them claimed."""
    if module is None:
        return []
    out: list[nn.Parameter] = []
    for param in module.parameters():
        if id(param) not in claimed:
            claimed.add(id(param))
            out.append(param)
    return out
