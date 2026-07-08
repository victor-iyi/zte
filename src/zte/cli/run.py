"""`zte-run` -- one command from data source to catalogued experiment.

This is the easy, reproducible front door. Given an experiment YAML (a `ZTEConfig`) and a data source,
it runs the whole pipeline and writes every artifact under `res/experiments/<run_name>/`::

    experiments/<run_name>/
      config.yaml         # the exact, resolved config (re-run with this)
      bundle/             # processed, cached ZuCoDataset (arrays + tables + normaliser)
      checkpoints/        # best.pt / last.pt / rotating + training curves
      figures/            # dataset overview figures
      evaluation/         # metrics.json, report.md, figures, interactive HTML
      exploration/        # brain-region + eye-tracking analysis
      tb/                 # TensorBoard (training + evaluation, incl. projector)
      manifest.json       # data source, headline metrics, verdict, paths
      README.md           # human summary of this run

The data source may be a local extracted directory, one or more `.zip` archives, a Google Drive id/URL, or `--synthetic`
for a no-data smoke run -- all normalised by `resolve_source`.

Examples::

    zte-run --config experiments/exp1_skipgram_rope_et.yaml --root res/data/zuco_extracted
    zte-run --config experiments/exp2_masked_eegonly.yaml --drive <folder-id-or-url>
    zte-run --config experiments/exp1_skipgram_rope_et.yaml --synthetic --epochs 5
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from zte.config import ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.run')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-run` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.

    """
    parser = argparse.ArgumentParser(
        description='Run a full, catalogued ZTE experiment from a config + data source.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, required=True, help='Experiment YAML (a ZTEConfig).')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--root', type=str, help='Local extracted dir, a .zip, or a folder of zips.')
    source.add_argument('--drive', type=str, help='Google Drive folder id / shareable URL.')
    source.add_argument('--synthetic', action='store_true', help='Generate a synthetic tree.')

    parser.add_argument('--name', type=str, default=None, help='Run name (default: config run_name).')
    parser.add_argument('--out-root', type=str, default='res/experiments')
    parser.add_argument('--extract-dir', type=str, default='res/data/zuco_extracted')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default=None)
    parser.add_argument('--epochs', type=int, default=None, help='Override config epochs.')
    parser.add_argument('--subjects', type=str, default=None,
                        help='Comma-separated subject subset (overrides config).')
    parser.add_argument('--tasks', type=str, default=None,
                        help='Comma-separated task subset (overrides config).')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--skip-explore', action='store_true')
    parser.add_argument('--no-tensorboard', action='store_true')
    parser.add_argument('--no-interactive', action='store_true')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def _resolve_root(args: argparse.Namespace, config: ZTEConfig) -> str:
    """Resolves the data source to a local directory of `.mat` files."""
    from zte.data.sources import resolve_source

    if args.synthetic:
        from zte.data.synthetic import generate_synthetic_zuco

        root = 'res/data/synthetic_zuco'
        generate_synthetic_zuco(root, tasks=config.dataset.tasks)
        return root
    spec = args.root or args.drive or config.dataset.root
    return str(resolve_source(spec, extract_dir=args.extract_dir))


def main() -> None:
    """Runs the whole pipeline and catalogues it under the run directory."""
    args = parse_arguments()
    configure_logging(args.log_level)

    config = ZTEConfig.from_yaml(args.config)
    if args.name:
        config.run_name = args.name
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.device is not None:
        config.train.device = args.device
    if args.subjects is not None:
        config.dataset.subjects = tuple(args.subjects.split(','))
    if args.tasks is not None:
        config.dataset.tasks = tuple(args.tasks.split(','))

    run_dir = Path(args.out_root) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info('=== Experiment %r -> %s ===', config.run_name, run_dir)

    # Point every output at the run directory so the experiment is self-contained.
    config.dataset.root = _resolve_root(args, config)
    config.dataset.cache_dir = str(run_dir / 'cache')
    config.train.ckpt_dir = str(run_dir / 'checkpoints')
    config.train.tensorboard = not args.no_tensorboard

    manifest: dict[str, Any] = {'run_name': config.run_name, 'data_root': config.dataset.root}

    # 1) Prepare (build + cache + save bundle + overview figures).
    _LOG.info('[1/4] Preparing dataset ...')
    dataset = ZuCoDataset(config.dataset).build()
    dataset.save(run_dir / 'bundle')
    manifest['dataset'] = dataset.analyze()
    _save_overview(dataset, run_dir / 'figures')

    # 2) Train.
    _LOG.info('[2/4] Training (%s / %s / pos=%s) ...',
              config.objective.name, config.model.frontend, config.model.pos_encoding)
    from zte.training.pipeline import run_training

    artifacts = run_training(config, dataset)
    config.to_yaml(run_dir / 'config.yaml')
    manifest['final_train_loss'] = (
        artifacts.history['train_loss'][-1] if artifacts.history['train_loss'] else None
    )
    _save_curves(artifacts.history, run_dir / 'checkpoints' / 'training_curves.png')

    # 3) Evaluate.
    if not args.skip_eval:
        _LOG.info('[3/4] Evaluating ...')
        manifest['evaluation'] = _evaluate(config, dataset, run_dir, args)

    # 4) Explore (brain regions + eye-tracking) when band power is available.
    if not args.skip_explore and dataset.band_power_raw is not None:
        _LOG.info('[4/4] Exploring brain regions + eye-tracking ...')
        from zte.cli.explore import run_exploration

        summary = run_exploration(dataset, run_dir / 'exploration')
        manifest['region_map_approximate'] = summary['region_map_approximate']

    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    (run_dir / 'README.md').write_text(_render_run_readme(config, manifest, run_dir), encoding='utf-8')
    _catalogue(Path(args.out_root), config.run_name, manifest)
    _LOG.info('Done. Everything catalogued under %s', run_dir.resolve())


def _evaluate(
    config: ZTEConfig, dataset: ZuCoDataset, run_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Embeds the best checkpoint and runs the full evaluation suite."""
    from zte.cli.evaluate import collect_embeddings
    from zte.evaluation.report import evaluate_representation
    from zte.inference.embed import ZTEEmbedder

    embedder = ZTEEmbedder.from_checkpoint(run_dir / 'checkpoints' / 'best.pt', dataset)
    word_emb, word_meta, raw_feats, sent_emb, sent_ids, sent_meta, word_bp = collect_embeddings(
        embedder, dataset
    )
    metrics = evaluate_representation(
        word_emb, word_meta, raw_feats, sent_emb, sent_ids,
        out_dir=run_dir / 'evaluation', run_name=config.run_name,
        sent_meta=sent_meta, word_band_power=word_bp, config=config,
        tensorboard=str(run_dir / 'tb' / 'eval') if not args.no_tensorboard else False,
        interactive=not args.no_interactive,
    )
    return {
        'verdict': metrics['verdict'],
        'sentence_retrieval_top1': metrics['sentence_retrieval'].get('top1'),
        'subject_transfer_top1': metrics['analogy'].get('subject_transfer', {}).get('top1'),
        'effective_rank_ratio': metrics['embedding_health'].get('effective_rank_ratio'),
    }


def _save_overview(dataset: ZuCoDataset, out: Path) -> None:
    """Renders dataset overview figures, tolerating a missing viz backend."""
    try:
        from zte.data.viz import save_overview

        save_overview(dataset, out)
    except (ImportError, ValueError) as exc:  # pragma: no cover
        _LOG.warning('Overview figures skipped: %r', exc)


def _save_curves(history: dict[str, list[float]], path: Path) -> None:
    """Saves the training-curve figure, tolerating a missing viz backend."""
    try:
        from zte.data.viz import plot_training_curves

        fig = plot_training_curves(history)
        fig.savefig(path, dpi=120, bbox_inches='tight')
    except (ImportError, ValueError) as exc:  # pragma: no cover
        _LOG.warning('Training-curve figure skipped: %r', exc)


def _render_run_readme(config: ZTEConfig, manifest: dict[str, Any], run_dir: Path) -> str:
    """Builds the per-run README summarising configuration and headline results."""
    ev = manifest.get('evaluation', {})
    lines = [
        f'# Experiment: {config.run_name}', '',
        '## Configuration', '',
        f'- Objective: **{config.objective.name}** | frontend: **{config.model.frontend}** | '
        f'positional encoding: **{config.model.pos_encoding}**',
        f'- Eye-tracking: **{"included" if config.dataset.include_eye_tracking else "excluded"}** '
        f'| representation: **{config.dataset.representation}** | tasks: {list(config.dataset.tasks)}',
        f'- embed_dim {config.model.embed_dim} | hidden {config.model.hidden_dim} | '
        f'layers {config.model.n_layers} | epochs {config.train.epochs} | split {config.train.split}',
        '', '## Data', '',
        f'- Source: `{manifest.get("data_root")}`',
        f'- Words: {manifest.get("dataset", {}).get("n_words")} | '
        f'sentences: {manifest.get("dataset", {}).get("n_sentences")} | '
        f'subjects: {manifest.get("dataset", {}).get("n_subjects")}',
        '', '## Headline results', '',
        f'- Final train loss: {manifest.get("final_train_loss")}',
        f'- Cross-subject sentence retrieval Top-1: {ev.get("sentence_retrieval_top1")}',
        f'- Subject-transfer analogy Top-1: {ev.get("subject_transfer_top1")}',
        f'- Embedding effective-rank ratio: {ev.get("effective_rank_ratio")}',
        f'- Verdict: `{json.dumps(ev.get("verdict", {}))}`',
        '', '## Reproduce', '',
        '```sh',
        f'zte-run --config {run_dir.name}/config.yaml --root <data> --name {config.run_name}',
        '```', '',
        'Artifacts: `bundle/`, `checkpoints/`, `evaluation/report.md`, '
        '`exploration/report.md`, `tb/` (TensorBoard incl. embedding projector).', '',
    ]
    return '\n'.join(lines)


def _catalogue(out_root: Path, run_name: str, manifest: dict[str, Any]) -> None:
    """Appends/updates a one-line entry for this run in the experiments index."""
    index = out_root / 'INDEX.md'
    ev = manifest.get('evaluation', {})
    row = (
        f'| {run_name} | {manifest.get("dataset", {}).get("n_words", "-")} | '
        f'{ev.get("sentence_retrieval_top1", "-")} | {ev.get("subject_transfer_top1", "-")} | '
        f'{ev.get("effective_rank_ratio", "-")} |'
    )
    header = (
        '# ZTE experiment catalogue\n\n'
        '| run | words | sent. retrieval Top-1 | subject-transfer Top-1 | eff-rank ratio |\n'
        '| --- | --- | --- | --- | --- |\n'
    )
    existing = index.read_text(encoding='utf-8') if index.exists() else header
    lines = [ln for ln in existing.splitlines() if not ln.startswith(f'| {run_name} |')]
    if not lines or not lines[0].startswith('# ZTE'):
        lines = header.rstrip('\n').splitlines()
    index.write_text('\n'.join([*lines, row]) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
