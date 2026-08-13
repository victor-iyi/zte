"""Cross-run weight initialisation: rebuild a frozen encoder from another run's checkpoint."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from zte.config import ZTEConfig
from zte.device import DeviceSpec, resolve_device
from zte.logging_utils import get_logger
from zte.models.embedding import ZTEModel, build_model
from zte.training.checkpoint import CheckpointManager

_LOG = get_logger('training.init')

# Prefix `torch.compile` leaves on every key of a wrapped module's state dict.
_COMPILE_PREFIX = '_orig_mod.'

# Keys of `extra` that describe the shapes the source encoder was built with.
_SHAPE_KEYS = (
    'in_dim',
    'raw_shape',
    'n_channels',
    'bp_features_per_channel',
    'montage_csv',
    'signature_dim',
)


@dataclass(slots=True)
class EncoderSource:
    """A rebuilt encoder plus everything the run that produced it fitted alongside it.

    Attributes:
        model (ZTEModel): The rebuilt encoder with the source weights strict-loaded, on the target device.
        config (ZTEConfig): The source run's full configuration, as stored in the checkpoint.
        shapes (dict[str, Any]): The frontend shapes the source encoder was built with.
        normalizer_state (dict[str, Any] | None): The source run's fitted feature normaliser state, if any.
        aligner_state (dict[str, Any] | None): The source run's fitted raw subject aligner state, if any.
        objective_state (dict[str, Any]): The source objective's `state_dict`, which is where a CLIP head lives.
        subject_vocab (dict[str, int] | None): The source run's subject vocabulary.
        epoch (int): Epoch the checkpoint was written at.
        step (int): Global optimiser step the checkpoint was written at.
        sha256 (str): Digest of the checkpoint file, so a decoder run can prove which encoder produced it.
        path (str): Path the checkpoint was read from.
        frozen (bool): Whether the model was returned with gradients disabled and in eval mode.
    """

    model: ZTEModel
    config: ZTEConfig
    shapes: dict[str, Any] = field(default_factory=dict)
    normalizer_state: dict[str, Any] | None = None
    aligner_state: dict[str, Any] | None = None
    objective_state: dict[str, Any] = field(default_factory=dict)
    subject_vocab: dict[str, int] | None = None
    epoch: int = 0
    step: int = 0
    sha256: str = ''
    path: str = ''
    frozen: bool = True

    def provenance(self) -> dict[str, Any]:
        """Returns the JSON-safe record of this source, for `extra['encoder_source']` and the run manifest."""
        return {
            'path': self.path,
            'sha256': self.sha256,
            'epoch': self.epoch,
            'step': self.step,
            'run_name': self.config.run_name,
            'objective': self.config.objective.name,
            'frontend': self.config.model.frontend,
            'frozen': self.frozen,
        }


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Returns the hex sha256 of a file, read in chunks so a multi-GB checkpoint never lands in memory.

    Args:
        path (str | Path): File to digest.
        chunk_size (int, optional): Bytes per read. Defaults to 1 MiB.

    Returns:
        str: The hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def strip_compile_prefix(state: dict[str, Any]) -> dict[str, Any]:
    """Removes the `_orig_mod.` key prefix `torch.compile` adds, so a compiled run's weights load into a plain model.

    Args:
        state (dict[str, Any]): A model state dict.

    Returns:
        dict[str, Any]: The same mapping with the prefix stripped from every key that carries it.
    """
    if not any(key.startswith(_COMPILE_PREFIX) for key in state):
        return state
    return {
        (key[len(_COMPILE_PREFIX) :] if key.startswith(_COMPILE_PREFIX) else key): value
        for key, value in state.items()
    }


def load_encoder(
    path: str | Path,
    config: ZTEConfig,
    device: DeviceSpec | None = None,
    freeze: bool | None = None,
) -> EncoderSource:
    """Rebuilds the encoder held in `path` at the shapes it was trained with, and strict-loads its weights.

    The shapes come from the checkpoint, never from the current run's dataset: a decoder stage that re-derived them
    would silently build a differently sized frontend and either fail to load or load a reinterpreted tensor.

    Args:
        path (str | Path): A `best.pt`/`last.pt` written by a previous run.
        config (ZTEConfig): The current run's configuration; supplies the device preference and the freeze policy, and
            is compared against the source for a divergence warning.
        device (DeviceSpec | None, optional): Pre-resolved device spec. Defaults to `None`, which resolves from
            `config.train`.
        freeze (bool | None, optional): Override the freeze policy. Defaults to `None`, which uses
            `config.train.freeze_encoder`.

    Returns:
        EncoderSource: The rebuilt model plus the source run's fitted states and provenance.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the checkpoint carries no input shapes, or a stored tensor's shape conflicts with the
            rebuilt model's (the message names the offending key).
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'Encoder checkpoint not found: {ckpt_path}')
    device = device or resolve_device(config.train.device, config.train.precision)
    payload = CheckpointManager.load(ckpt_path, map_location=str(device.device))

    source_config = ZTEConfig.from_dict(payload.get('config') or {})
    extra: dict[str, Any] = payload.get('extra') or {}
    shapes = {key: extra.get(key) for key in _SHAPE_KEYS}
    raw_shape = shapes['raw_shape']
    shapes['raw_shape'] = tuple(raw_shape) if raw_shape is not None else None
    if shapes['in_dim'] is None and shapes['raw_shape'] is None:
        raise ValueError(
            f'{ckpt_path} carries no input shapes; it predates shape-bearing checkpoints and cannot '
            'seed a decoder run.'
        )
    if source_config.model != config.model:
        _LOG.warning(
            'Encoder checkpoint model config differs from this run; the source config wins for the encoder.'
        )

    model = build_model(
        source_config.model,
        in_dim=shapes['in_dim'],
        raw_shape=shapes['raw_shape'],
        n_channels=shapes['n_channels'],
        bp_features_per_channel=shapes['bp_features_per_channel'],
        montage_csv=shapes['montage_csv'],
        signature_dim=int(shapes['signature_dim'] or 0),
    )
    state = strip_compile_prefix(payload['model'])
    _assert_loadable(model, state, ckpt_path)
    model.load_state_dict(state, strict=True)
    model = model.to(device.device)

    frozen = config.train.freeze_encoder if freeze is None else freeze
    if frozen:
        model.requires_grad_(False)
        model.eval()

    source = EncoderSource(
        model=model,
        config=source_config,
        shapes=shapes,
        normalizer_state=extra.get('normalizer'),
        aligner_state=extra.get('aligner'),
        objective_state=extra.get('objective_state') or {},
        subject_vocab=extra.get('subject_vocab'),
        epoch=int(payload.get('epoch', 0)),
        step=int(payload.get('step', 0)),
        sha256=file_sha256(ckpt_path),
        path=str(ckpt_path),
        frozen=frozen,
    )
    _LOG.info(
        'Loaded encoder %s (epoch %d, step %d, %s, frozen=%s, sha256=%s).',
        ckpt_path,
        source.epoch,
        source.step,
        source_config.model.frontend,
        frozen,
        source.sha256[:12],
    )
    return source


def _assert_loadable(model: ZTEModel, state: dict[str, Any], ckpt_path: Path) -> None:
    """Raises before `load_state_dict` with the offending key named, which torch's own message buries.

    Raises:
        ValueError: On the first missing, unexpected or shape-conflicting key.
    """
    target = model.state_dict()
    for key, value in state.items():
        if key not in target:
            raise ValueError(f'{ckpt_path}: key {key!r} is not present in the rebuilt encoder.')
        if torch.is_tensor(value) and tuple(value.shape) != tuple(target[key].shape):
            raise ValueError(
                f'{ckpt_path}: shape conflict at {key!r} -- checkpoint has {tuple(value.shape)}, '
                f'the rebuilt encoder expects {tuple(target[key].shape)}.'
            )
    missing = [key for key in target if key not in state]
    if missing:
        raise ValueError(f'{ckpt_path}: missing {len(missing)} key(s), first is {missing[0]!r}.')
