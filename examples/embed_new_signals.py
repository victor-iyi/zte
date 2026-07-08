"""Embed NEW EEG signals with a trained ZTE checkpoint.

Two complementary flows are shown:

1. **New recordings as ZuCo `.mat` files** -> build a `ZuCoDataset` and call `ZTEEmbedder.embed` (word- and sentence-level, with aligned metadata).
2. **New signals already in memory** (arbitrary arrays from a custom/streaming pipeline) -> call `ZTEEmbedder.embed_signals`,
   which applies the checkpoint's fitted normaliser so the inputs are scaled exactly as in training.

If `--ckpt` is omitted a tiny model is trained on synthetic data first, so the example runs end-to-end with no downloads.

Example:
    >>> # uv run python examples/embed_new_signals.py
    >>> # uv run python examples/embed_new_signals.py --ckpt res/checkpoints/best.pt
"""

# pyright: reportOptionalSubscript=false
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zte.config import DatasetConfig, MissingConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.features import flatten_band_power
from zte.data.synthetic import generate_synthetic_zuco
from zte.device import resolve_device
from zte.inference.embed import ZTEEmbedder
from zte.logging_utils import configure_logging, get_logger
from zte.training.pipeline import run_training

_LOG = get_logger('examples.embed_new_signals')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the example's command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Embed new EEG signals with a trained ZTE model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default=None,
        help='Trained checkpoint (best.pt). If omitted, a tiny model is trained first.',
    )
    parser.add_argument('--workdir', type=str, default='res/embed_demo')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument(
        '--epochs', type=int, default=5, help='Epochs for the fallback synthetic training.'
    )
    return parser.parse_args()


def train_quick_checkpoint(workdir: Path, device: str, epochs: int) -> Path:
    """Trains a tiny band-power skip-gram model on synthetic data.

    Args:
        workdir (Path): Working directory for synthetic data, cache and checkpoints.
        device (str): Device preference passed to the trainer.
        epochs (int): Number of training epochs.

    Returns:
        Path: Path to the written `best.pt` checkpoint.
    """
    root = workdir / 'train_zuco'
    generate_synthetic_zuco(root, subjects=('ZAB', 'ZDM'), tasks=('SR', 'NR'), n_sentences=10)
    dataset = ZuCoDataset(
        DatasetConfig(
            root=str(root),
            representation='band_power',
            missing=MissingConfig(method='knn'),
            cache_dir=str(workdir / 'train_cache'),
        )
    ).build()

    config = ZTEConfig(run_name='embed-demo')
    config.objective.name = 'skipgram'
    config.model.embed_dim = 64
    config.model.hidden_dim = 64
    config.model.n_layers = 2
    config.train.epochs = epochs
    config.train.batch_size = 16
    config.train.device = device
    config.train.precision = 'fp32'
    config.train.ckpt_dir = str(workdir / 'checkpoints')
    run_training(config, dataset)
    return workdir / 'checkpoints' / 'best.pt'


def build_new_recordings(embedder: ZTEEmbedder, workdir: Path) -> ZuCoDataset:
    """Synthesises and builds a fresh ZuCo dataset matching the checkpoint's config.

    Args:
        embedder (ZTEEmbedder): The restored embedder (source of representation/bands).
        workdir (Path): Working directory for the synthetic "new" recordings.

    Returns:
        ZuCoDataset: A built dataset of brand-new recordings.
    """
    new_root = workdir / 'new_zuco'
    generate_synthetic_zuco(new_root, subjects=('ZNEW',), tasks=('NR',), n_sentences=6, seed=123)
    return ZuCoDataset(
        DatasetConfig(
            root=str(new_root),
            representation=embedder.config.dataset.representation,
            bands=embedder.config.dataset.bands,
            band_power_measures=embedder.config.dataset.band_power_measures,
            raw_window=embedder.config.dataset.raw_window,
            missing=MissingConfig(method='mask_only'),
            cache_dir=str(workdir / 'new_cache'),
        )
    ).build()


def embed_from_mat_files(
    embedder: ZTEEmbedder, new_ds: ZuCoDataset, workdir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Flow 1: embed new recordings supplied as ZuCo `.mat` files.

    Args:
        embedder (ZTEEmbedder): The restored embedder.
        new_ds (ZuCoDataset): The new recordings to embed.
        workdir (Path): Where the `.npz` export is written.

    Returns:
        tuple[np.ndarray, np.ndarray]: `(word_embeddings, sentence_embeddings)`.
    """
    word_emb, word_meta = embedder.embed(new_ds, level='word')
    sent_emb, _ = embedder.embed(new_ds, level='sentence')
    out = embedder.export(word_emb, word_meta, workdir / 'new_word_embeddings.npz')
    _LOG.info(
        'Flow 1: %d word + %d sentence embeddings from new .mat files; exported -> %s',
        len(word_emb),
        len(sent_emb),
        out,
    )
    return word_emb, sent_emb


def embed_from_memory(embedder: ZTEEmbedder, new_ds: ZuCoDataset) -> np.ndarray:
    """Flow 2: embed new signals already held in memory as arrays.

    Any `(N, F*C)` band-power array (or `(N, C, T)` raw array for a raw model) works;
    here we use the present word tokens from the new recordings as a stand-in for a
    custom feature pipeline.

    Args:
        embedder (ZTEEmbedder): The restored embedder.
        new_ds (ZuCoDataset): The new recordings (source of example token arrays).

    Returns:
        np.ndarray: `(N, embed_dim)` embeddings for the in-memory signals.
    """
    if embedder.model.uses_raw:
        signals = new_ds.raw_eeg[new_ds.presence]  # (N, C, T)
        mem_emb = embedder.embed_signals(raw=signals)
    else:
        feats = flatten_band_power(new_ds.band_power_raw)  # (N, F*C), NaN at omissions
        signals = feats[new_ds.presence]  # keep present (finite) tokens
        mem_emb = embedder.embed_signals(band_power=signals)  # normaliser applied internally
    _LOG.info(
        'Flow 2: embedded %d in-memory signals -> shape %s', len(mem_emb), tuple(mem_emb.shape)
    )
    return mem_emb


def main() -> None:
    """Runs both embedding flows end-to-end."""
    args = parse_arguments()
    configure_logging('INFO')
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.ckpt) if args.ckpt else train_quick_checkpoint(work, args.device, args.epochs)

    # Restore the trained encoder AND its normaliser -- no dataset required.
    embedder = ZTEEmbedder.from_checkpoint(ckpt, device=resolve_device(args.device))

    new_ds = build_new_recordings(embedder, work)
    word_emb, sent_emb = embed_from_mat_files(embedder, new_ds, work)
    mem_emb = embed_from_memory(embedder, new_ds)

    if len(mem_emb) > 4:
        neighbours = embedder.nearest_neighbors(mem_emb, query_idx=0, k=3)
        _LOG.info('Nearest neighbours of in-memory signal 0: %s', neighbours)

    print(f'Flow 1 (new .mat) word embeddings:     {word_emb.shape}')
    print(f'Flow 1 (new .mat) sentence embeddings: {sent_emb.shape}')
    print(f'Flow 2 (in-memory) embeddings:         {mem_emb.shape}')


if __name__ == '__main__':
    main()
