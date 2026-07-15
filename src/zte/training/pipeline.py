"""High-level training orchestration tying a dataset to a trained ZTE model.

`run_training` is the one call the CLI (and the smoke-test) make: it builds leakage-aware splits, sizes the encoder to the data,
constructs the chosen objective, wires DataLoaders with a shared subject vocabulary, and runs the `Trainer`. The fitted normaliser
and subject vocabulary are embedded in every checkpoint for reproducible inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zte.config import ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.torch_dataset import ZuCoTorchDataset, build_subject_vocab, make_dataloader
from zte.device import DeviceSpec, auto_num_workers, resolve_device
from zte.logging_utils import get_logger
from zte.models.embedding import build_model
from zte.models.objectives import build_objective
from zte.training.trainer import Trainer

_LOG = get_logger('training.pipeline')


@dataclass(slots=True)
class TrainingArtifacts:
    """Everything `run_training` produces.

    Attributes:
        trainer (Trainer): The (already-run) trainer, exposing the model and checkpoints.
        history (dict[str, list[float]]): The per-epoch metric history.
        device (DeviceSpec): The resolved device spec used for training.
        test_indices (np.ndarray | None): Held-out test row indices (from `train.test_fraction`), or
            `None` when no test split was carved. Never seen during training.
    """

    trainer: Trainer
    history: dict[str, list[float]]
    device: DeviceSpec
    test_indices: np.ndarray | None = None


def run_training(
    config: ZTEConfig,
    dataset: ZuCoDataset,
    device: DeviceSpec | None = None,
    resume: bool = False,
) -> TrainingArtifacts:
    """Builds and runs a full ZTE pretraining job over `dataset`.

    Args:
        config (ZTEConfig): The complete run configuration.
        dataset (ZuCoDataset): A built `ZuCoDataset`.
        device (DeviceSpec | None): Optional pre-resolved device spec.
        resume (bool): Continue an interrupted run from its `last.pt` checkpoint (see `Trainer`).

    Returns:
        TrainingArtifacts: A `TrainingArtifacts` with the trainer, history and device.

    Raises:
        ValueError: If the configured representation has no matching tensors in the dataset (e.g. raw frontend but band-power-only dataset).
    """
    device = device or resolve_device(config.train.device, config.train.precision)
    splits = dataset.split(
        config.train.split,
        val_fraction=config.train.val_fraction,
        test_fraction=config.train.test_fraction,
        holdout_subject=config.train.loso_holdout_subject,
        seed=config.train.seed,
    )

    # Fit the normaliser (and imputer) on the TRAIN split only, so val/test/held-out-subject
    # statistics never leak in. A no-op when `dataset.normalizer_fit == 'all'` (legacy behaviour).
    dataset.refit_normalizer(splits['train'])

    in_dim, raw_shape, feature_dim = _shapes(dataset, config)
    n_channels, bp_features_per_channel = _channel_shape(dataset, config, raw_shape)
    model = build_model(
        config.model,
        in_dim=in_dim,
        raw_shape=raw_shape,
        n_channels=n_channels,
        bp_features_per_channel=bp_features_per_channel,
        montage_csv=config.dataset.montage_csv,
    )
    objective = build_objective(config.objective, model, feature_dim=feature_dim)

    vocab = build_subject_vocab(dataset)
    # Auto-pick DataLoader workers per backend when config.train.num_workers < 0 (else honour it).
    workers = auto_num_workers(device, config.train.num_workers)
    # Unit B: only emit per-word behaviour targets when the behaviour head is active.
    beh_targets = (
        config.objective.behaviour_targets if config.objective.behaviour_weight > 0.0 else ()
    )
    # Build the torch datasets up front so static-shape padding can be sized from actual lengths.
    train_td = dataset.to_torch(
        split=splits['train'], subject_vocab=vocab, behaviour_targets=beh_targets
    )
    val_td = (
        dataset.to_torch(split=splits['val'], subject_vocab=vocab, behaviour_targets=beh_targets)
        if len(splits['val']) > 0
        else None
    )

    # Units A & B: attach dataset-derived auxiliary targets to the objective now that the word
    # vocabulary and behaviour spec exist. The meaning matrix is frozen (a distillation teacher).
    if config.objective.meaning_distill_weight > 0.0 or config.objective.behaviour_weight > 0.0:
        import torch as _torch

        from zte.data.meaning import build_meaning_matrix

        meaning_mat = None
        if config.objective.meaning_distill_weight > 0.0:
            mat = build_meaning_matrix(
                train_td.word_vocab, config.objective.meaning_source, config.objective.meaning_dim
            )
            meaning_mat = _torch.from_numpy(mat)
        beh_binary = _torch.from_numpy(train_td.behaviour_binary) if beh_targets else None
        objective.attach_auxiliary(meaning_matrix=meaning_mat, behaviour_binary=beh_binary)
    # Static shapes (XLA/TPU only under 'auto'): pad every batch to the dataset-wide max sentence
    # length so XLA compiles a single graph instead of recompiling per shape. Accuracy-neutral
    # (padded positions are masked out); `None` keeps the smallest per-batch padding on GPU/CPU/MPS.
    pad_to = _static_pad_length(config.train.static_shapes, device, train_td, val_td)
    # Cross-subject positives need the same stimulus (read by different subjects) to co-occur in a
    # batch, so route the train loader through the stimulus-grouped sampler when enabled.
    group_by_stimulus = bool(config.objective.cross_subject_positives)
    train_loader = make_dataloader(
        train_td,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.supports_pin_memory,
        drop_last=config.objective.name in {'skipgram', 'cbow', 'cpc'},
        group_by_stimulus=group_by_stimulus,
        seed=config.train.seed,
        pad_to=pad_to,
    )
    val_loader = None
    if val_td is not None:
        val_loader = make_dataloader(
            val_td,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.supports_pin_memory,
            pad_to=pad_to,
        )

    extra = {
        'subject_vocab': vocab,
        'normalizer': None if dataset.normalizer is None else dataset.normalizer.state,
        'in_dim': in_dim,
        'raw_shape': raw_shape,
        'n_channels': n_channels,
        'bp_features_per_channel': bp_features_per_channel,
        'montage_csv': config.dataset.montage_csv,
        'feature_names': dataset.feature_names,
    }
    trainer = Trainer(
        model=model,
        objective=objective,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        extra_state=extra,
        resume=resume,
    )
    history = trainer.train()
    _LOG.info(
        'Training complete: %d epochs, final train_loss=%.4f',
        config.train.epochs,
        history['train_loss'][-1] if history['train_loss'] else float('nan'),
    )
    return TrainingArtifacts(
        trainer=trainer, history=history, device=device, test_indices=splits.get('test')
    )


def _shapes(
    dataset: ZuCoDataset, config: ZTEConfig
) -> tuple[int | None, tuple[int, int] | None, int | None]:
    """Resolves frontend input shapes from the dataset and config.

    Args:
        dataset (ZuCoDataset): A built dataset.
        config (ZTEConfig): The run configuration.

    Returns:
        `(in_dim, raw_shape, feature_dim)` for model/objective construction.

    Raises:
        ValueError: If required tensors are missing for the chosen frontend.
    """
    in_dim = None if dataset.features is None else int(dataset.features.shape[1])
    raw_shape = (
        None
        if dataset.raw_eeg is None
        else (int(dataset.raw_eeg.shape[1]), int(dataset.raw_eeg.shape[2]))
    )
    if config.model.frontend == 'band_power_mlp' and in_dim is None:
        raise ValueError('band_power_mlp frontend needs band-power features in the dataset.')
    if config.model.frontend == 'raw_conformer' and raw_shape is None:
        raise ValueError('raw_conformer frontend needs raw EEG in the dataset.')
    # The masked-reconstruct head predicts the per-token input, so its target
    # dimension follows the frontend: n_features for band power, or
    # n_channels * time_steps for raw windows.
    recon_dim = in_dim
    if config.model.frontend == 'raw_conformer' and raw_shape is not None:
        recon_dim = raw_shape[0] * raw_shape[1]
    return in_dim, raw_shape, recon_dim


def _static_pad_length(
    setting: str,
    device: DeviceSpec,
    train_td: ZuCoTorchDataset,
    val_td: ZuCoTorchDataset | None,
) -> int | None:
    """Resolves the fixed padding length for static shapes, or `None` for per-batch padding.

    `auto` enables static shapes only on XLA/TPU (where dynamic shapes force recompilation); `on`/`off`
    force it. When active, the length is the dataset-wide maximum sentence length (over train + val),
    so no sentence is ever truncated and XLA compiles a single graph.

    Args:
        setting (str): `'auto'`, `'on'` or `'off'`.
        device (DeviceSpec): The resolved device (its `kind` gates the `auto` decision).
        train_td (Any): The training torch dataset (exposes `.sequences`).
        val_td (Any): The validation torch dataset or `None`.

    Returns:
        int | None: The fixed pad length, or `None` to pad each batch to its own maximum.
    """
    active = setting == 'on' or (setting == 'auto' and device.kind == 'xla')
    if not active:
        return None
    lengths = [len(s) for s in train_td.sequences]
    if val_td is not None:
        lengths += [len(s) for s in val_td.sequences]
    return max(lengths) if lengths else None


def _channel_shape(
    dataset: ZuCoDataset, config: ZTEConfig, raw_shape: tuple[int, int] | None
) -> tuple[int | None, int | None]:
    """Resolves the EEG channel geometry needed for electrode spatial encoding.

    Args:
        dataset (ZuCoDataset): A built dataset.
        config (ZTEConfig): The run configuration.
        raw_shape (tuple[int, int] | None): `(n_channels, time_steps)` for the raw frontend.

    Returns:
        `(n_channels, bp_features_per_channel)`. `bp_features_per_channel` is `None` for the raw frontend (each electrode token is a time window, not a
        band-power vector); either value is `None` when it cannot be determined, which disables spatial encoding.
    """
    if config.model.frontend == 'raw_conformer':
        return (raw_shape[0] if raw_shape is not None else None), None
    bp = dataset.band_power_raw
    if bp is None:
        return None, None
    return int(bp.shape[2]), int(bp.shape[1])
