"""`zte-benchmark` -- fixed-seed sweep over objective x positional-encoding x eye-tracking x seed."""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import argparse
import copy
import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zte.cli.support.render import dataframe_to_markdown
from zte.cli.support.sources import (
    PENDING_ROOT,
    add_data_source_args,
    add_extract_dir,
    resolve_root_if_needed,
)
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

    parser.add_argument(
        '--base-config',
        type=Path,
        default=None,
        dest='base_config',
        help='Experiment YAML to sweep ON TOP OF, so every cell inherits the full recipe (encoder, '
        'spatial encoding, invariance stack) and only the swept axes differ. Without it each cell is '
        'a bare default config, which benchmarks the objective in isolation rather than the flagship.',
    )
    parser.add_argument(
        '--loso-holdout',
        type=str,
        default=None,
        dest='loso_holdout',
        help='Held-out subject code; forces split=by_subject_loso so rows are held-out comparable.',
    )
    parser.add_argument('--tasks', type=str, default='SR,NR')
    parser.add_argument('--subjects', type=str, default=None)
    parser.add_argument('--objectives', type=str, default='clip,skipgram')
    parser.add_argument('--pos-encodings', type=str, default='rope')
    parser.add_argument('--eye-tracking', choices=['both', 'on', 'off'], default='off')
    parser.add_argument('--seeds', type=str, default='42')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip grid cells that already have metrics on disk and continue interrupted ones from '
        'their last checkpoint. Safe to pass always; makes the sweep restartable after a lost VM.',
    )
    parser.add_argument(
        '--drive-backup',
        type=str,
        default=None,
        dest='drive_backup',
        help="Mounted Drive folder to mirror each cell's checkpoints to after every epoch.",
    )
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument(
        '--precision',
        choices=['auto', 'fp32', 'fp16', 'bf16'],
        default='auto',
        help='Mixed-precision override (auto = bf16 on Ampere+/TPU, fp16 on older CUDA, fp32 on MPS/CPU).',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=-1,
        dest='num_workers',
        help='DataLoader workers; -1 = auto per backend (default).',
    )
    parser.add_argument(
        '--compile',
        choices=['on', 'off'],
        default='off',
        help='torch.compile the model (CUDA only).',
    )
    parser.add_argument('--out', type=str, default='res/benchmark')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def _dataset_for(
    args: argparse.Namespace, include_et: bool, base: ZTEConfig | None = None
) -> ZuCoDataset:
    """Builds (and caches) the dataset for one eye-tracking setting.

    With `--base-config` the base recipe's dataset settings are kept (representation, normalisation,
    window, montage), so a raw-conformer flagship is benchmarked on raw EEG rather than band power.
    """
    if base is not None:
        cfg = replace(
            base.dataset,
            root=PENDING_ROOT,
            tasks=tuple(args.tasks.split(',')),
            subjects=tuple(args.subjects.split(',')) if args.subjects else base.dataset.subjects,
            include_eye_tracking=include_et,
        )
    else:
        cfg = DatasetConfig(
            root=PENDING_ROOT,
            tasks=tuple(args.tasks.split(',')),
            subjects=tuple(args.subjects.split(',')) if args.subjects else None,
            representation='band_power',
            include_eye_tracking=include_et,
            missing=MissingConfig(method='mask_only'),
        )

    # Keyed first, resolved second: a cached bundle skips unzipping the archives entirely.
    cfg.root = resolve_root_if_needed(args, cfg)
    return ZuCoDataset(cfg).build()


def headline_metrics(embedder: ZTEEmbedder, dataset: ZuCoDataset) -> dict[str, float]:
    """Computes a compact, comparable metric block for one trained model.

    Args:
        embedder (ZTEEmbedder): A restored embedder.
        dataset (ZuCoDataset): The dataset it was trained on.

    Returns:
        dict[str, float]: Sentence retrieval Top-1 (+ lift over chance), embedding effective-rank ratio
            and anisotropy, the number of probe targets beating the noise control, and subject-transfer
            analogy Top-1 (+ lift).
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

    # A target beats noise only when the paired per-fold gap's bootstrap CI clears an effect-size floor.
    zte = {r['target']: r for r in comparison if r['representation'] == 'ZTE'}
    noise = {r['target']: r for r in comparison if r['representation'] == 'noise'}
    effect_floor = 0.01
    beats = 0
    for t, z_row in zte.items():
        z_scores = np.asarray(z_row.get('linear_scores', []), dtype=np.float64)
        n_scores = np.asarray(noise.get(t, {}).get('linear_scores', []), dtype=np.float64)
        if z_scores.size and z_scores.size == n_scores.size:
            _, lo, _ = M.bootstrap_ci(z_scores - n_scores)
        else:  # No per-fold scores (tiny target) -> fall back to the point difference.
            lo = float(z_row['linear_score'] - noise.get(t, {}).get('linear_score', 0.0))
        if lo > effect_floor:
            beats += 1

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

    base = ZTEConfig.from_yaml(args.base_config) if args.base_config else None
    if base is not None:
        _LOG.info('Sweeping on top of %s (run_name %r).', args.base_config, base.run_name)

    objectives = args.objectives.split(',')
    pos_encodings = args.pos_encodings.split(',')
    et_settings = {'both': [True, False], 'on': [True], 'off': [False]}[args.eye_tracking]
    seeds = [int(s) for s in args.seeds.split(',')]
    datasets = {et: _dataset_for(args, et, base) for et in et_settings}

    # One training run + metric block per grid cell, each writing its own resolved config.
    rows: list[dict[str, Any]] = []
    grid = list(itertools.product(objectives, pos_encodings, et_settings, seeds))
    _LOG.info('Benchmarking %d configurations.', len(grid))
    for objective, pos, include_et, seed in grid:
        tag = f'{objective}_{pos}_et{int(include_et)}_s{seed}'
        cell = out / 'runs' / tag
        cached = cell / 'metrics.json'

        # A finished cell is reused verbatim, so a lost VM costs only the cell that was in flight.
        if args.resume and cached.is_file():
            metrics = json.loads(cached.read_text(encoding='utf-8'))
            _LOG.info('[%s] resume: reusing metrics on disk.', tag)
        else:
            # Deep, not shallow: the sub-configs are mutated below and must not leak across cells.
            cfg = copy.deepcopy(base) if base is not None else ZTEConfig()
            cfg.objective.name = objective
            cfg.model.pos_encoding = pos
            cfg.dataset = copy.deepcopy(datasets[include_et].config)
            cfg.train.epochs = args.epochs
            cfg.train.batch_size = args.batch_size
            cfg.train.device = args.device
            cfg.train.precision = args.precision
            cfg.train.num_workers = args.num_workers
            cfg.train.compile_model = args.compile == 'on'
            cfg.train.seed = seed
            cfg.train.deterministic = True
            cfg.train.ckpt_dir = str(cell)
            cfg.run_name = tag
            if args.loso_holdout:
                cfg.train.split = 'by_subject_loso'
                cfg.train.loso_holdout_subject = args.loso_holdout
            if args.drive_backup:
                cfg.train.drive_backup_dir = str(Path(args.drive_backup) / tag)

            cell.mkdir(parents=True, exist_ok=True)
            cfg.to_yaml(
                cell / 'config.yaml'
            )  # written first, so an interrupted cell is reproducible
            run_training(cfg, datasets[include_et], resume=args.resume)
            embedder = ZTEEmbedder.from_checkpoint(cell / 'best.pt', datasets[include_et])
            metrics = headline_metrics(embedder, datasets[include_et])
            (cell / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

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

    # Aggregate into the sortable CSV + Markdown tables.
    frame = pd.DataFrame(rows).sort_values('subject_transfer_lift', ascending=False)
    frame.to_csv(out / 'benchmark.csv', index=False)
    (out / 'benchmark.md').write_text(_render_markdown(frame), encoding='utf-8')
    _LOG.info('Benchmark written to %s (%d runs)', out, len(rows))
    print(frame.to_string(index=False))


def _render_markdown(frame: pd.DataFrame) -> str:
    """Renders the benchmark table as a Markdown document (no extra deps)."""
    header = (
        '# ZTE benchmark\n\nEach row is a fixed-seed, reproducible run; sorted by '
        'subject-transfer lift (higher = more subject-agnostic).\n\n'
    )
    return header + dataframe_to_markdown(frame)


if __name__ == '__main__':
    main()
