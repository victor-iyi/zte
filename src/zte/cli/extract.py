"""`zte-extract` -- export thought embeddings + aligned metadata from a trained checkpoint as `.npz`."""

from __future__ import annotations

import argparse
import json

from zte.cli.support.sources import add_data_source_args, add_extract_dir, resolve_data_root
from zte.config import DatasetConfig, MissingConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.inference.embed import ZTEEmbedder
from zte.logging_utils import configure_logging, get_logger
from zte.training.checkpoint import CheckpointManager
from zte.training.metrics import linear_probe

_LOG = get_logger('cli.extract')


def load_dataset(args: argparse.Namespace) -> ZuCoDataset:
    """Loads the bundle, or builds a dataset from `.mat` files matching the checkpoint.

    When building from `--root`, the representation and band/measure/raw settings come from the
    checkpoint's embedded config, since otherwise a raw or `both` encoder would find no tensors.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        ZuCoDataset: A built dataset ready for embedding.
    """
    if args.bundle:
        return ZuCoDataset.load(args.bundle)
    payload = CheckpointManager.load(args.ckpt)
    cfg = ZTEConfig.from_dict(payload['config'])
    ds_config = DatasetConfig(
        root=resolve_data_root(args),
        representation=cfg.dataset.representation,
        band_power_measures=cfg.dataset.band_power_measures,
        bands=cfg.dataset.bands,
        raw_field=cfg.dataset.raw_field,
        raw_window=cfg.dataset.raw_window,
        normalize=cfg.dataset.normalize,
        missing=MissingConfig(method=cfg.dataset.missing.method),
    )
    return ZuCoDataset(ds_config).build()


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-extract` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Extract ZTE thought embeddings from a checkpoint.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint (best.pt/last.pt).')
    add_data_source_args(parser, include_bundle=True)
    add_extract_dir(parser)

    parser.add_argument('--level', choices=['word', 'sentence'], default='word')
    parser.add_argument('--out', type=str, default='res/embeddings/embeddings.npz')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument(
        '--probe-target',
        type=str,
        default='log_freq',
        help='Word column to linear-probe (word level only, e.g. `log_freq`).',
    )
    parser.add_argument('--drive-dir', type=str, default=None, help='Optional Drive upload folder.')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def main() -> None:
    """Runs embedding extraction end-to-end from the command line."""
    args = parse_arguments()
    configure_logging(args.log_level)

    dataset = load_dataset(args)
    from zte.device import resolve_device

    embedder = ZTEEmbedder.from_checkpoint(args.ckpt, dataset, device=resolve_device(args.device))
    embeddings, meta = embedder.embed(dataset, level=args.level)
    out = embedder.export(embeddings, meta, args.out)

    if args.level == 'word' and args.probe_target in meta and len(embeddings) > 10:
        score = linear_probe(embeddings, meta[args.probe_target].to_numpy())
        _LOG.info('Linear probe (%s): %s', args.probe_target, json.dumps(score))

    if args.drive_dir:
        from zte.data.io.remote import upload_directory

        upload_directory(out.parent, args.drive_dir)
    _LOG.info('Wrote %d embeddings to %s', len(embeddings), out.resolve())


if __name__ == '__main__':
    main()
