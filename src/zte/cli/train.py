"""`zte-train` -- pretrain a ZTE model with a chosen self-supervised objective."""

from __future__ import annotations

import argparse
from pathlib import Path

from zte.cli.support.sources import (
    PENDING_ROOT,
    add_data_source_args,
    add_extract_dir,
    resolve_root_if_needed,
)
from zte.config import DatasetConfig, MissingConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.logging_utils import configure_logging, get_logger
from zte.training.pipeline import run_training

_LOG = get_logger('cli.train')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-train` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Pretrain a ZuCo Thought Embedding model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_source_args(parser, include_bundle=True, include_synthetic=True)
    add_extract_dir(parser)

    parser.add_argument('--config', type=str, default=None, help='Optional base YAML config.')
    parser.add_argument('--objective', choices=['skipgram', 'cbow', 'masked', 'cpc'], default=None)
    parser.add_argument('--frontend', choices=['band_power_mlp', 'raw_conformer'], default=None)
    parser.add_argument('--representation', choices=['band_power', 'raw', 'both'], default=None)
    parser.add_argument('--embed-dim', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument(
        '--split', choices=['random', 'by_sentence', 'by_subject_loso', 'by_task'], default=None
    )
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default=None)
    parser.add_argument('--precision', choices=['auto', 'fp32', 'fp16', 'bf16'], default=None)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--drive-backup-dir', type=str, default=None)
    parser.add_argument('--ckpt-dir', type=str, default=None)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--synthetic-out', type=str, default='res/data/synthetic_zuco')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ZTEConfig:
    """Builds the run config from an optional YAML base plus CLI overrides.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        ZTEConfig: The merged `ZTEConfig`.
    """
    config = ZTEConfig.from_yaml(args.config) if args.config else ZTEConfig()
    overrides = {
        ('objective', 'name'): args.objective,
        ('model', 'frontend'): args.frontend,
        ('model', 'embed_dim'): args.embed_dim,
        ('dataset', 'representation'): args.representation,
        ('train', 'epochs'): args.epochs,
        ('train', 'batch_size'): args.batch_size,
        ('train', 'lr'): args.lr,
        ('train', 'split'): args.split,
        ('train', 'device'): args.device,
        ('train', 'precision'): args.precision,
        ('train', 'ckpt_dir'): args.ckpt_dir,
        ('train', 'drive_backup_dir'): args.drive_backup_dir,
    }
    for (section, field), value in overrides.items():
        if value is not None:
            setattr(getattr(config, section), field, value)
    if args.tensorboard:
        config.train.tensorboard = True
    if args.run_name:
        config.run_name = args.run_name
    return config


def load_dataset(args: argparse.Namespace, config: ZTEConfig) -> ZuCoDataset:
    """Loads or builds the dataset implied by the CLI source flags.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        config (ZTEConfig): The run config (its dataset section is used when building).

    Returns:
        ZuCoDataset: A built `ZuCoDataset`.
    """
    if args.bundle:
        if args.representation is not None:
            _LOG.warning(
                'Ignoring --representation=%s: bundle %r fixed its representation at '
                'prepare time. Re-run zte-prepare to change it.',
                args.representation,
                args.bundle,
            )
        return ZuCoDataset.load(args.bundle)
    ds_config = DatasetConfig(
        root=PENDING_ROOT,
        representation=config.dataset.representation,
        missing=MissingConfig(method=config.dataset.missing.method),
        normalize=config.dataset.normalize,
        raw_window=config.dataset.raw_window,
    )

    # Keyed first, resolved second: a cached bundle skips unzipping the archives entirely.
    ds_config.root = resolve_root_if_needed(args, ds_config)
    return ZuCoDataset(ds_config).build()


def main() -> None:
    """Runs a full training job from the command line."""
    args = parse_arguments()
    configure_logging(args.log_level)
    config = build_config(args)
    dataset = load_dataset(args, config)

    artifacts = run_training(config, dataset)
    out_dir = Path(config.train.ckpt_dir)
    config.to_yaml(out_dir / 'config.yaml')

    # Training curves are a convenience; a missing viz backend must not fail the run.
    try:
        from zte.data.viz import plot_training_curves  # pylint: disable=import-outside-toplevel

        fig = plot_training_curves(artifacts.history)
        fig.savefig(out_dir / 'training_curves.png', dpi=120, bbox_inches='tight')
    except (ImportError, ValueError) as exc:  # pragma: no cover
        _LOG.warning('Could not render training curves: %r', exc)
    _LOG.info('Done. Checkpoints + config in %s', out_dir.resolve())


if __name__ == '__main__':
    main()
