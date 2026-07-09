"""`zte-benchmark` -- a reproducible sweep over the knobs that matter.

To claim ZTE is a *standard* way to encode EEG, its choices must be benchmarked, not asserted.
This command trains and evaluates a small model across a grid of **objective x positional-encoding x eye-tracking x seed**,
all from fixed seeds, and aggregates the headline metrics into a sortable table (CSV + Markdown).
Every cell writes its own resolved `config.yaml` so any row can be reproduced exactly.

The defaults are intentionally light (a quick, CPU-friendly sweep); widen the grid via the flags for a full run on a GPU.
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zte.cli.sources import add_data_source_args, add_extract_dir, resolve_data_root
from zte.config import DatasetConfig, MissingConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.evaluation import metrics as M
from zte.evaluation.analogy import analogy_report
from zte.inference.embed import ZTEEmbedder
from zte.logging_utils import configure_logging, get_logger
from zte.training.metrics import noise_matched
from zte.training.pipeline import run_training

_LOG = get_logger('cli.benchmark')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-benchmark` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Reproducible ZTE benchmark sweep (objective x pos-encoding x eye-tracking).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_source_args(parser, include_synthetic=True)
    add_extract_dir(parser)

    parser.add_argument('--tasks', type=str, default='SR,NR')
    parser.add_argument('--subjects', type=str, default=None)
    parser.add_argument('--objectives', type=str, default='skipgram')
    parser.add_argument('--pos-encodings', type=str, default='rope,learned')
    parser.add_argument('--eye-tracking', choices=['both', 'on', 'off'], default='both')
    parser.add_argument('--seeds', type=str, default='42')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--out', type=str, default='res/benchmark')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def _dataset_for(args: argparse.Namespace, include_et: bool) -> ZuCoDataset:
    """Builds (and caches) the band-power dataset for an eye-tracking setting."""
    if args.synthetic:
        from zte.data.synthetic import generate_synthetic_zuco

        root = 'res/data/synthetic_zuco'
        generate_synthetic_zuco(root, tasks=tuple(args.tasks.split(',')))
    else:
        root = resolve_data_root(args)
    cfg = DatasetConfig(
        root=root,
        tasks=tuple(args.tasks.split(',')),
        subjects=tuple(args.subjects.split(',')) if args.subjects else None,
        representation='band_power',
        include_eye_tracking=include_et,
        missing=MissingConfig(method='mask_only'),
    )
    return ZuCoDataset(cfg).build()


def headline_metrics(embedder: ZTEEmbedder, dataset: ZuCoDataset) -> dict[str, float]:
    """Computes a compact, comparable metric block for one trained model.

    Args:
        embedder (ZTEEmbedder): A restored embedder.
        dataset (ZuCoDataset): The dataset it was trained on.

    Returns:
        dict[str, float]: Headline metrics: sentence retrieval Top-1 (+ lift over chance), embedding effective-rank ratio and
            anisotropy, the number of probe targets that beat the noise control, and subject-transfer analogy Top-1 (+ lift).

    """
    from zte.cli.evaluate import collect_embeddings

    word_emb, word_meta, raw_feats, sent_emb, sent_ids, _sent_meta, _bp = collect_embeddings(
        embedder, dataset
    )
    sent_ret = M.content_retrieval(sent_emb, np.asarray(sent_ids))
    health = M.embedding_health(word_emb)
    analogy = analogy_report(word_emb, word_meta, raw_feats)['subject_transfer']

    # How many attributes beat a noise-matched control (linear probe).
    reps = {'ZTE': word_emb, 'noise': noise_matched(np.asarray(raw_feats, dtype=np.float32))}
    tgts: dict[str, tuple[np.ndarray, str]] = {}
    if 'word_len' in word_meta:
        tgts['word_len'] = (word_meta['word_len'].to_numpy(), 'regression')
    if word_meta['subject'].nunique() > 1:
        tgts['subject'] = (pd.factorize(word_meta['subject'])[0], 'classification')
    comparison = M.representation_comparison(reps, tgts)
    zte = {r['target']: r['linear_score'] for r in comparison if r['representation'] == 'ZTE'}
    noise = {r['target']: r['linear_score'] for r in comparison if r['representation'] == 'noise'}
    beats = sum(1 for t in zte if zte[t] > noise.get(t, -1) + 1e-3)

    return {
        'sent_retrieval_top1': round(float(sent_ret.get('top1', float('nan'))), 4),
        'sent_retrieval_lift': round(
            float(sent_ret.get('top1', 0) - sent_ret.get('chance_top1', 0)), 4
        ),
        'eff_rank_ratio': round(float(health['effective_rank_ratio']), 4),
        'anisotropy': round(float(health['anisotropy']), 4),
        'beats_noise': int(beats),
        'subject_transfer_top1': round(float(analogy.get('top1', float('nan'))), 4),
        'subject_transfer_lift': round(
            float(analogy.get('top1', 0) - analogy.get('chance_top1', 0)), 4
        ),
    }


def main() -> None:
    """Runs the benchmark sweep and writes the aggregated tables."""
    args = parse_arguments()
    configure_logging(args.log_level)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    objectives = args.objectives.split(',')
    pos_encodings = args.pos_encodings.split(',')
    et_settings = {'both': [True, False], 'on': [True], 'off': [False]}[args.eye_tracking]
    seeds = [int(s) for s in args.seeds.split(',')]
    datasets = {et: _dataset_for(args, et) for et in et_settings}

    rows: list[dict[str, Any]] = []
    grid = list(itertools.product(objectives, pos_encodings, et_settings, seeds))
    _LOG.info('Benchmarking %d configurations.', len(grid))
    for objective, pos, include_et, seed in grid:
        tag = f'{objective}_{pos}_et{int(include_et)}_s{seed}'
        cfg = ZTEConfig()
        cfg.objective.name = objective
        cfg.model.pos_encoding = pos
        cfg.dataset.include_eye_tracking = include_et
        cfg.train.epochs = args.epochs
        cfg.train.batch_size = args.batch_size
        cfg.train.device = args.device
        cfg.train.seed = seed
        cfg.train.deterministic = True
        cfg.train.ckpt_dir = str(out / 'runs' / tag)
        cfg.run_name = tag

        run_training(cfg, datasets[include_et])
        cfg.to_yaml(out / 'runs' / tag / 'config.yaml')
        embedder = ZTEEmbedder.from_checkpoint(out / 'runs' / tag / 'best.pt', datasets[include_et])
        metrics = headline_metrics(embedder, datasets[include_et])
        rows.append(
            {
                'objective': objective,
                'pos_encoding': pos,
                'eye_tracking': include_et,
                'seed': seed,
                **metrics,
            }
        )
        _LOG.info('[%s] %s', tag, json.dumps(metrics))

    frame = pd.DataFrame(rows).sort_values('subject_transfer_lift', ascending=False)
    frame.to_csv(out / 'benchmark.csv', index=False)
    (out / 'benchmark.md').write_text(_render_markdown(frame), encoding='utf-8')
    _LOG.info('Benchmark written to %s (%d runs)', out, len(rows))
    print(frame.to_string(index=False))


def _render_markdown(frame: pd.DataFrame) -> str:
    """Renders the benchmark table as a Markdown document (no extra deps)."""
    cols = list(frame.columns)
    head = '| ' + ' | '.join(cols) + ' |\n| ' + ' | '.join(['---'] * len(cols)) + ' |\n'
    body = ''.join('| ' + ' | '.join(str(v) for v in row) + ' |\n' for row in frame.to_numpy())
    header = (
        '# ZTE benchmark\n\nEach row is a fixed-seed, reproducible run; sorted by '
        'subject-transfer lift (higher = more subject-agnostic).\n\n'
    )
    return header + head + body


if __name__ == '__main__':
    main()
