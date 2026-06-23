"""End-to-end ZTE demo: synthesise ZuCo, pretrain a thought embedding, evaluate.

Runs the entire pipeline on synthetic (but schema-faithful) data so it works
with no downloads: generate -> build dataset -> pretrain (chosen objective) ->
extract word embeddings -> linear-probe the embedding -> render figures.

Example:
    >>> # uv run python examples/run_demo.py --objective skipgram --epochs 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zte.config import DatasetConfig, MissingConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.synthetic import generate_synthetic_zuco
from zte.data.viz import plot_training_curves, save_overview
from zte.inference.embed import ZTEEmbedder
from zte.logging_utils import configure_logging, get_logger
from zte.training.metrics import linear_probe, noise_matched
from zte.training.pipeline import run_training

_LOG = get_logger('demo')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the demo's command-line arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description='Run the end-to-end ZTE demo on synthetic data.')
    parser.add_argument(
        '--objective', choices=['skipgram', 'cbow', 'masked', 'cpc'], default='skipgram'
    )
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--subjects', type=str, default='ZAB,ZDM,ZJN')
    parser.add_argument('--sentences', type=int, default=10)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--workdir', type=str, default='res/demo')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='cpu')
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ZTEConfig:
    """Builds a small, fast :class:`ZTEConfig` for the demo.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The configured run.
    """
    cfg = ZTEConfig(run_name=f'demo-{args.objective}')
    cfg.objective.name = args.objective
    cfg.model.embed_dim = args.embed_dim
    cfg.model.hidden_dim = 96
    cfg.model.n_layers = 3
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = 16
    cfg.train.device = args.device
    cfg.train.precision = 'fp32'
    cfg.train.split = 'by_sentence'
    cfg.train.ckpt_dir = str(Path(args.workdir) / 'checkpoints')
    cfg.train.log_every = 5
    return cfg


def main() -> None:
    """Runs the full demo and writes figures + a JSON summary."""
    args = parse_arguments()
    configure_logging('INFO')
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    # 1) Synthesise + build the dataset.
    data_root = work / 'synthetic_zuco'
    generate_synthetic_zuco(
        data_root,
        subjects=tuple(args.subjects.split(',')),
        tasks=('SR', 'NR'),
        n_sentences=args.sentences,
        show_progress=True,
    )
    ds_cfg = DatasetConfig(
        root=str(data_root),
        representation='band_power',
        missing=MissingConfig(method='knn'),
        cache_dir=str(work / 'cache'),
    )
    dataset = ZuCoDataset(ds_cfg).build()
    _LOG.info('Dataset: %r', dataset)

    # 2) Pretrain the thought embedding.
    cfg = build_config(args)
    artifacts = run_training(cfg, dataset)

    # 3) Extract word-level embeddings from the best checkpoint.
    embedder = ZTEEmbedder.from_checkpoint(work / 'checkpoints' / 'best.pt', dataset)
    emb, meta = embedder.embed(dataset, level='word')
    embedder.export(emb, meta, work / 'word_embeddings.npz')

    # 4) Evaluate the embedding with linear probes (vs a noise-matched floor).
    # Probe only non-degenerate targets (extracted words are all present, so
    # 'is_omitted' would be single-class and uninformative here).
    probes = {
        target: linear_probe(emb, meta[target].to_numpy())
        for target in ('log_freq', 'word_len')
        if target in meta and meta[target].nunique() > 1
    }
    noise = noise_matched(dataset.features)  # type: ignore[arg-type]
    _LOG.info('Linear probes: %s', json.dumps(probes, indent=2))

    # 5) Figures.
    fig_dir = work / 'figures'
    save_overview(dataset, fig_dir)
    fig = plot_training_curves(artifacts.history)
    fig.savefig(fig_dir / 'training_curves.png', dpi=120, bbox_inches='tight')

    summary = {
        'objective': args.objective,
        'dataset': dataset.analyze(),
        'final_train_loss': artifacts.history['train_loss'][-1],
        'n_word_embeddings': int(emb.shape[0]),
        'embed_dim': int(emb.shape[1]),
        'probes': probes,
        'noise_floor_shape': list(noise.shape),
        'device': artifacts.device.name,
    }
    (work / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    _LOG.info('Demo complete. Artifacts in %s', work.resolve())
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
