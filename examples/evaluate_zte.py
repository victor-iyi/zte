"""End-to-end ZTE evaluation on synthetic data: train -> embed -> evidence.

Produces figures, tables and a Markdown report showing the encoder turns EEG into a structured, re-purposable space -- transfer
probes (vs raw features and a noise control), geometry/health (no collapse), and cross-subject content retrieval.

If `--ckpt` is omitted a small model is trained on synthetic ZuCo first, so this runs end-to-end with no downloads.
Point `--ckpt`/`--root` at real artifacts to evaluate a real run.

Example:
    >>> # uv run python examples/evaluate_zte.py --out res/evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zte.cli.evaluate import collect_embeddings
from zte.config import DatasetConfig, MissingConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.synthetic import generate_synthetic_zuco
from zte.device import resolve_device
from zte.evaluation.report import evaluate_representation
from zte.inference.embed import ZTEEmbedder
from zte.logging_utils import configure_logging, get_logger
from zte.training.pipeline import run_training

_LOG = get_logger('examples.evaluate_zte')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the example's command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate a ZTE model end-to-end on synthetic data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint; trained if omitted.')
    parser.add_argument('--workdir', type=str, default='res/eval_demo')
    parser.add_argument('--out', type=str, default='res/eval_demo/evaluation')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--epochs', type=int, default=15)
    return parser.parse_args()


def build_dataset(workdir: Path) -> ZuCoDataset:
    """Synthesises a multi-subject ZuCo tree and builds a band-power dataset.

    Args:
        workdir (Path): Working directory for synthetic data and cache.

    Returns:
        ZuCoDataset: The built dataset.
    """
    root = workdir / 'zuco'
    # Several subjects + both tasks so probes and cross-subject retrieval are meaningful.
    generate_synthetic_zuco(
        root, subjects=('ZAB', 'ZDM', 'ZJN', 'ZKH'), tasks=('SR', 'NR'), n_sentences=14
    )
    return ZuCoDataset(
        DatasetConfig(
            root=str(root),
            representation='band_power',
            missing=MissingConfig(method='knn'),
            cache_dir=str(workdir / 'cache'),
        )
    ).build()


def train_quick(dataset: ZuCoDataset, workdir: Path, device: str, epochs: int) -> Path:
    """Trains a small skip-gram ZTE model and returns its best checkpoint.

    Args:
        dataset (ZuCoDataset): The training dataset.
        workdir (Path): Where checkpoints are written.
        device (str): Device preference.
        epochs (int): Training epochs.

    Returns:
        Path: Path to `best.pt`.
    """
    config = ZTEConfig(run_name='eval-demo')
    config.objective.name = 'skipgram'
    config.model.embed_dim = 96
    config.model.hidden_dim = 96
    config.model.n_layers = 3
    config.train.epochs = epochs
    config.train.batch_size = 32
    config.train.device = device
    config.train.precision = 'fp32'
    config.train.ckpt_dir = str(workdir / 'checkpoints')
    run_training(config, dataset)
    return workdir / 'checkpoints' / 'best.pt'


def main() -> None:
    """Runs training (optional) and the full evaluation, printing a verdict."""
    args = parse_arguments()
    configure_logging('INFO')
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(work)
    ckpt = Path(args.ckpt) if args.ckpt else train_quick(dataset, work, args.device, args.epochs)

    embedder = ZTEEmbedder.from_checkpoint(ckpt, dataset, device=resolve_device(args.device))
    word_emb, word_meta, raw_feats, sent_emb, sent_ids, sent_meta, word_bp = collect_embeddings(
        embedder, dataset
    )
    metrics = evaluate_representation(
        word_emb,
        word_meta,
        raw_feats,
        sent_emb,
        sent_ids,
        out_dir=args.out,
        run_name='eval-demo',
        sent_meta=sent_meta,
        word_band_power=word_bp,
        config=embedder.config,
        tensorboard=True,
        interactive=True,
    )

    print(json.dumps(metrics['verdict'], indent=2))
    print(f'\nReport:      {Path(args.out) / "report.md"}')
    print(f'Figures:     {Path(args.out) / "figures"}')
    print(f'Interactive: {Path(args.out) / "interactive" / "word_explorer.html"}')
    print(f'TensorBoard: tensorboard --logdir {Path(args.out) / "tb"}')


if __name__ == '__main__':
    main()
