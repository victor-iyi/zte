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

from zte.cli.sources import add_data_source_args, add_extract_dir, resolve_data_root
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
    add_data_source_args(parser, include_synthetic=True)
    add_extract_dir(parser)

    parser.add_argument(
        '--name', type=str, default=None, help='Run name (default: config run_name).'
    )
    parser.add_argument('--out-root', type=str, default='res/experiments')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default=None)
    parser.add_argument('--epochs', type=int, default=None, help='Override config epochs.')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override config seed (for multi-seed sweeps; also appended to the run name if --name is unset).',
    )
    parser.add_argument(
        '--subjects',
        type=str,
        default=None,
        help='Comma-separated subject subset (overrides config).',
    )
    parser.add_argument(
        '--tasks', type=str, default=None, help='Comma-separated task subset (overrides config).'
    )
    parser.add_argument(
        '--loso-holdout',
        type=str,
        default=None,
        help='Leave-one-subject-out held-out subject code (overrides config.train.loso_holdout_subject; '
        'forces split=by_subject_loso). When --name is unset the run name is suffixed with _lo<SUBJ> so '
        'each held-out subject gets its own resumable run directory.',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume an interrupted run: reuse the cached bundle, continue training from the last '
        'checkpoint, and skip stages (prepare / eval / explore) whose outputs already exist. Safe to '
        'pass on a first run (nothing to resume) and to re-run repeatedly.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='With --resume, redo already-completed stages instead of skipping them.',
    )
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--skip-explore', action='store_true')
    parser.add_argument('--no-tensorboard', action='store_true')
    parser.add_argument('--no-interactive', action='store_true')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def _resolve_root(args: argparse.Namespace, config: ZTEConfig) -> str:
    """Resolves the data source to a local directory of `.mat` files."""
    if args.synthetic:
        from zte.data.synthetic import generate_synthetic_zuco

        root = 'res/data/synthetic_zuco'
        generate_synthetic_zuco(root, tasks=config.dataset.tasks)
        return root
    return resolve_data_root(
        args,
        default=config.dataset.root,
        tasks=config.dataset.tasks,
        subjects=config.dataset.subjects,
    )


def main() -> None:
    """Runs the whole pipeline, catalogues it, and supports pause/resume via `--resume`."""
    args = parse_arguments()
    configure_logging(args.log_level)
    try:
        _run(args)
    except KeyboardInterrupt:
        _LOG.warning(
            '\n⏸  Paused. Re-run the exact same command with --resume to continue where you left off.'
        )
        raise SystemExit(130) from None


def _run(args: argparse.Namespace) -> None:
    """The pipeline body (separated so `main` can wrap it for clean pause/resume)."""

    config = ZTEConfig.from_yaml(args.config)
    if args.seed is not None:
        config.train.seed = args.seed
    if args.loso_holdout is not None:
        config.train.loso_holdout_subject = args.loso_holdout
        config.train.split = 'by_subject_loso'
    if args.name:
        config.run_name = args.name
    else:
        if args.loso_holdout is not None:
            config.run_name = f'{config.run_name}_lo{args.loso_holdout}'
        if args.seed is not None:
            config.run_name = f'{config.run_name}_s{args.seed}'
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

    # Fast resume: a fully-complete run is skipped instantly, without reloading the (large) bundle.
    if args.resume and not args.force and _run_is_complete(run_dir, config, args):
        _LOG.info('Resume: %r already complete (all stages done); skipping.', config.run_name)
        return

    _LOG.info('=== Experiment %r -> %s ===', config.run_name, run_dir)

    # Point every output at the run directory so the experiment is self-contained.
    config.dataset.root = _resolve_root(args, config)
    config.dataset.cache_dir = str(run_dir / 'cache')
    config.train.ckpt_dir = str(run_dir / 'checkpoints')
    config.train.tensorboard = not args.no_tensorboard

    manifest: dict[str, Any] = {'run_name': config.run_name, 'data_root': config.dataset.root}

    bundle_dir = run_dir / 'bundle'

    # 1) Prepare (build + cache + save bundle + overview figures). On --resume, reuse the bundle.
    if args.resume and (bundle_dir / 'meta.json').exists() and not args.force:
        _LOG.info('[1/4] Resume: loading cached dataset bundle (skipping prepare) ...')
        dataset = ZuCoDataset.load(bundle_dir)
    else:
        _LOG.info('[1/4] Preparing dataset ...')
        dataset = ZuCoDataset(config.dataset).build()
        dataset.save(bundle_dir)
    manifest['dataset'] = dataset.analyze()
    _save_overview(dataset, run_dir / 'figures')

    # 2) Train (resumes from the last checkpoint when --resume).
    _LOG.info(
        '[2/4] Training (%s / %s / pos=%s)%s ...',
        config.objective.name,
        config.model.frontend,
        config.model.pos_encoding,
        ' [resume]' if args.resume else '',
    )
    from zte.training.pipeline import run_training

    artifacts = run_training(config, dataset, resume=args.resume)
    config.to_yaml(run_dir / 'config.yaml')
    manifest['final_train_loss'] = (
        artifacts.history['train_loss'][-1] if artifacts.history['train_loss'] else None
    )
    _save_curves(artifacts.history, run_dir / 'checkpoints' / 'training_curves.png')

    # 3) Evaluate (skipped on --resume only if metrics are at least as new as the checkpoint, so a
    #    run whose training advanced this invocation is always re-evaluated on the fresh model).
    metrics_path = run_dir / 'evaluation' / 'metrics.json'
    best_ckpt = run_dir / 'checkpoints' / 'best.pt'
    eval_fresh = metrics_path.exists() and (
        not best_ckpt.exists() or metrics_path.stat().st_mtime >= best_ckpt.stat().st_mtime
    )
    if not args.skip_eval:
        if args.resume and eval_fresh and not args.force:
            _LOG.info('[3/4] Resume: evaluation already up to date, skipping.')
            manifest['evaluation'] = _eval_summary_from_disk(metrics_path)
        else:
            _LOG.info('[3/4] Evaluating ...')
            manifest['evaluation'] = _evaluate(config, dataset, run_dir, args)

    # 4) Explore (brain regions + eye-tracking) when band power is available.
    explore_done = (run_dir / 'exploration' / 'report.md').exists()
    if not args.skip_explore and dataset.band_power_raw is not None:
        if args.resume and explore_done and not args.force:
            _LOG.info('[4/4] Resume: exploration already done, skipping.')
        else:
            _LOG.info('[4/4] Exploring brain regions + eye-tracking ...')
            from zte.cli.explore import run_exploration

            summary = run_exploration(dataset, run_dir / 'exploration')
            manifest['region_map_approximate'] = summary['region_map_approximate']

    (run_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, default=str), encoding='utf-8'
    )
    (run_dir / 'README.md').write_text(
        _render_run_readme(config, manifest, run_dir), encoding='utf-8'
    )
    _catalogue(Path(args.out_root), config.run_name, manifest)
    _LOG.info('Done. Everything catalogued under %s', run_dir.resolve())


def _last_completed_epoch(ckpt_dir: Path) -> int:
    """Returns the highest completed epoch from the `ckpt_epoch*.pt` filenames (no torch.load)."""
    epochs = [int(p.stem.removeprefix('ckpt_epoch')) for p in ckpt_dir.glob('ckpt_epoch*.pt')]
    return max(epochs) if epochs else 0


def _run_is_complete(run_dir: Path, config: ZTEConfig, args: argparse.Namespace) -> bool:
    """Whether a run has finished every stage, so `--resume` can skip it without loading anything.

    Checks training (last checkpoint at the final epoch) and, unless skipped, evaluation
    (`metrics.json`) and exploration (`exploration/report.md`) outputs.

    Args:
        run_dir (Path): The run directory.
        config (ZTEConfig): The (resolved) run config, for the target epoch count.
        args (argparse.Namespace): Parsed CLI args (for `--skip-eval` / `--skip-explore`).

    Returns:
        bool: `True` when nothing remains to do.
    """
    if not (run_dir / 'checkpoints' / 'last.pt').exists():
        return False
    if _last_completed_epoch(run_dir / 'checkpoints') < config.train.epochs:
        return False
    if not args.skip_eval and not (run_dir / 'evaluation' / 'metrics.json').exists():
        return False
    if not args.skip_explore and not (run_dir / 'exploration' / 'report.md').exists():
        # Exploration only runs when band power is available; absence of the marker is only
        # decisive when the run could have produced it. A prior manifest confirms it did/should.
        if (run_dir / 'manifest.json').exists():
            manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
            if 'region_map_approximate' not in manifest:
                return True  # this run legitimately has no exploration stage
        return False
    return True


def _eval_summary_from_disk(metrics_path: Path) -> dict[str, Any]:
    """Rebuilds the manifest evaluation summary from an existing `metrics.json` (resume path)."""
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    return {
        'verdict': metrics.get('verdict'),
        'sentence_retrieval_top1': metrics.get('sentence_retrieval', {}).get('top1'),
        'subject_transfer_top1': metrics.get('analogy', {}).get('subject_transfer', {}).get('top1'),
        'effective_rank_ratio': metrics.get('embedding_health', {}).get('effective_rank_ratio'),
    }


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
        word_emb,
        word_meta,
        raw_feats,
        sent_emb,
        sent_ids,
        out_dir=run_dir / 'evaluation',
        run_name=config.run_name,
        sent_meta=sent_meta,
        word_band_power=word_bp,
        config=config,
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
        f'# Experiment: {config.run_name}',
        '',
        '## Configuration',
        '',
        f'- Objective: **{config.objective.name}** | frontend: **{config.model.frontend}** | '
        f'positional encoding: **{config.model.pos_encoding}**',
        f'- Eye-tracking: **{"included" if config.dataset.include_eye_tracking else "excluded"}** '
        f'| representation: **{config.dataset.representation}** | tasks: {list(config.dataset.tasks)}',
        f'- embed_dim {config.model.embed_dim} | hidden {config.model.hidden_dim} | '
        f'layers {config.model.n_layers} | epochs {config.train.epochs} | split {config.train.split}',
        '',
        '## Data',
        '',
        f'- Source: `{manifest.get("data_root")}`',
        f'- Words: {manifest.get("dataset", {}).get("n_words")} | '
        f'sentences: {manifest.get("dataset", {}).get("n_sentences")} | '
        f'subjects: {manifest.get("dataset", {}).get("n_subjects")}',
        '',
        '## Headline results',
        '',
        f'- Final train loss: {manifest.get("final_train_loss")}',
        f'- Cross-subject sentence retrieval Top-1: {ev.get("sentence_retrieval_top1")}',
        f'- Subject-transfer analogy Top-1: {ev.get("subject_transfer_top1")}',
        f'- Embedding effective-rank ratio: {ev.get("effective_rank_ratio")}',
        f'- Verdict: `{json.dumps(ev.get("verdict", {}))}`',
        '',
        '## Reproduce',
        '',
        '```sh',
        f'zte-run --config {run_dir.name}/config.yaml --root <data> --name {config.run_name}',
        '```',
        '',
        'Artifacts: `bundle/`, `checkpoints/`, `evaluation/report.md`, '
        '`exploration/report.md`, `tb/` (TensorBoard incl. embedding projector).',
        '',
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
