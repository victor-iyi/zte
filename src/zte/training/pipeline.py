"""High-level training orchestration tying a dataset to a trained ZTE model.

:func:`run_training` is the one call the CLI (and the smoke-test) make: it builds
leakage-aware splits, sizes the encoder to the data, constructs the chosen
objective, wires DataLoaders with a shared subject vocabulary, and runs the
:class:`~zte.training.trainer.Trainer`. The fitted normaliser and subject
vocabulary are embedded in every checkpoint for reproducible inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zte.config import ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.torch_dataset import build_subject_vocab, make_dataloader
from zte.device import DeviceSpec, resolve_device
from zte.logging_utils import get_logger
from zte.models.embedding import build_model
from zte.models.objectives import build_objective
from zte.training.trainer import Trainer

_LOG = get_logger('training.pipeline')


@dataclass(slots=True)
class TrainingArtifacts:
    """Everything :func:`run_training` produces.

    Attributes:
        trainer: The (already-run) trainer, exposing the model and checkpoints.
        history: The per-epoch metric history.
        device: The resolved device spec used for training.
        test_indices: Held-out test row indices (from `train.test_fraction`), or
            `None` when no test split was carved. Never seen during training.
    """

    trainer: Trainer
    history: dict[str, list[float]]
    device: DeviceSpec
    test_indices: np.ndarray | None = None


def run_training(
    config: ZTEConfig, dataset: ZuCoDataset, device: DeviceSpec | None = None
) -> TrainingArtifacts:
    """Builds and runs a full ZTE pretraining job over `dataset`.

    Args:
        config: The complete run configuration.
        dataset: A built :class:`ZuCoDataset`.
        device: Optional pre-resolved device spec.

    Returns:
        A :class:`TrainingArtifacts` with the trainer, history and device.

    Raises:
        ValueError: If the configured representation has no matching tensors in
            the dataset (e.g. raw frontend but band-power-only dataset).
    """
    device = device or resolve_device(config.train.device, config.train.precision)
    splits = dataset.split(
        config.train.split,
        val_fraction=config.train.val_fraction,
        test_fraction=config.train.test_fraction,
        holdout_subject=config.train.loso_holdout_subject,
        seed=config.train.seed,
    )

    in_dim, raw_shape, feature_dim = _shapes(dataset, config)
    model = build_model(config.model, in_dim=in_dim, raw_shape=raw_shape)
    objective = build_objective(config.objective, model, feature_dim=feature_dim)

    vocab = build_subject_vocab(dataset)
    train_loader = make_dataloader(
        dataset.to_torch(split=splits['train'], subject_vocab=vocab),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=device.supports_pin_memory,
        drop_last=config.objective.name in {'skipgram', 'cbow', 'cpc'},
    )
    val_loader = None
    if len(splits['val']) > 0:
        val_loader = make_dataloader(
            dataset.to_torch(split=splits['val'], subject_vocab=vocab),
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
            pin_memory=device.supports_pin_memory,
        )

    extra = {
        'subject_vocab': vocab,
        'normalizer': None if dataset.normalizer is None else dataset.normalizer.state,
        'in_dim': in_dim,
        'raw_shape': raw_shape,
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
        dataset: A built dataset.
        config: The run configuration.

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
