"""`zte-run` -- prepare, train, evaluate and catalogue one experiment under `res/experiments/<run_name>/`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

from zte.cli.support.io import read_json, write_json
from zte.cli.support.provision import add_provision_args, provision_from_args
from zte.cli.support.sources import add_data_source_args, add_extract_dir, resolve_root_if_needed
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
    add_provision_args(parser)

    parser.add_argument('--name', type=str, default=None, help='Run name (default: config run_name).')
    parser.add_argument('--out-root', type=Path, default=Path('res/experiments'))
    parser.add_argument(
        '--data-cache',
        type=Path,
        default=None,
        dest='data_cache',
        help='Shared directory for the PROCESSED dataset bundle, content-addressed by the dataset '
        'config. Point it at a persistent/Drive path to build the bundle ONCE and reuse it across '
        'every run and session: on a cache hit the expensive `.mat` load + processing is skipped '
        'entirely. Default: a per-run cache under the run directory (no cross-run reuse).',
    )
    parser.add_argument(
        '--data-cache-remote',
        type=str,
        default=None,
        dest='data_cache_remote',
        help='Persistent cache directory (e.g. a mounted Drive folder) layered behind --data-cache. A '
        'bundle found there is copied down once; a freshly built one is published there immediately, so '
        'the processing survives a reclaimed VM. Defaults to $ZTE_CACHE_REMOTE.',
    )
    parser.add_argument(
        '--drive-backup',
        type=str,
        default=None,
        dest='drive_backup',
        help="Mounted Drive folder to mirror each run's checkpoints to after every epoch (train "
        'locally, keep a live Drive copy). Each run backs up to <root>/<run_name>/checkpoints.',
    )
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default=None)
    parser.add_argument(
        '--precision',
        choices=['auto', 'fp32', 'fp16', 'bf16'],
        default=None,
        help='Mixed-precision override. auto = bf16 on Ampere+/TPU, fp16 on older CUDA, fp32 on MPS/CPU.',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        dest='num_workers',
        help='DataLoader workers; -1 = auto per backend (a few on an accelerator, 0 on CPU).',
    )
    parser.add_argument(
        '--compile',
        choices=['on', 'off'],
        default=None,
        help='torch.compile the model (CUDA only; can speed up training, first step is slower).',
    )
    parser.add_argument(
        '--static-shapes',
        choices=['auto', 'on', 'off'],
        default=None,
        dest='static_shapes',
        help='Fixed-length padding for XLA/TPU (auto = on only on TPU). Accuracy-neutral.',
    )
    parser.add_argument('--epochs', type=int, default=None, help='Override config epochs.')
    parser.add_argument(
        '--mode',
        choices=['encoder', 'decoder', 'joint'],
        default=None,
        help='Training stage: the encoder alone, a decoder over a frozen encoder, or both jointly '
        '(overrides config.train.mode).',
    )
    parser.add_argument(
        '--encoder-ckpt',
        type=str,
        default=None,
        dest='encoder_ckpt',
        help='Frozen encoder checkpoint a decoder/joint run starts from; its stored shapes, normaliser '
        'and aligner are reused verbatim rather than refitted.',
    )
    parser.add_argument(
        '--lm',
        type=str,
        default=None,
        help="Override config.decoder.lm_source ('tiny' builds a 2-layer model locally, with no network).",
    )
    parser.add_argument(
        '--conditioning',
        choices=['pooled', 'pooled_plus_words'],
        default=None,
        help='What the prefix bridge reads (overrides config.decoder.conditioning).',
    )
    parser.add_argument(
        '--stage0-epochs',
        type=int,
        default=None,
        dest='stage0_epochs',
        help='Text-only bridge pretraining epochs on train-split stimuli (0 disables Stage 0).',
    )
    parser.add_argument(
        '--decode-eval',
        action='store_true',
        dest='decode_eval',
        help='Force config.objective.eval_generation: decode the held-out cell with its controls after training.',
    )
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
    parser.add_argument('--tasks', type=str, default=None, help='Comma-separated task subset (overrides config).')
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
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    return parser.parse_args()


def _resolve_root(args: argparse.Namespace, config: ZTEConfig) -> str:
    """Resolves the data source, skipping extraction entirely when the bundle is already cached."""
    return resolve_root_if_needed(args, config.dataset)


def warn_on_split_override(config: ZTEConfig, requested_split: str) -> bool:
    """Warns when `--loso-holdout` traded a decoder run's honest split for `by_subject_loso`.

    Args:
        config (ZTEConfig): The configuration after every CLI override has been applied.
        requested_split (str): The split the YAML named, before `--loso-holdout` touched it.

    Returns:
        bool: Whether the warning fired.

    Note:
        `by_subject_loso` shares all 700 stimuli between train and val, which is the one configuration in which
        a decoder recites the corpus rather than reading one -- and the generation verdict refuses a headline on
        it. An encoder run loses nothing by the swap, so this reports rather than refuses.
    """
    if (
        config.train.mode == 'encoder'
        or config.train.split != 'by_subject_loso'
        or requested_split == 'by_subject_loso'
    ):
        return False

    _LOG.warning(
        '--loso-holdout replaced split %r with %r for a %r run: every held-out sentence is now also a '
        'training sentence, so the generation verdict will fail its honest_split clause. Name '
        'train.loso_holdout_subject in the config instead and drop the flag.',
        requested_split,
        config.train.split,
        config.train.mode,
    )
    return True


def main() -> None:
    """Runs the whole pipeline, catalogues it, and supports pause/resume via `--resume`."""
    args = parse_arguments()
    configure_logging(args.log_level)
    try:
        _run(args)
    except KeyboardInterrupt:
        _LOG.warning('\n⏸  Paused. Re-run the exact same command with --resume to continue where you left off.')
        raise SystemExit(130) from None


def _run(args: argparse.Namespace) -> None:
    """The pipeline body (separated so `main` can wrap it for clean pause/resume)."""
    # CLI overrides, applied before anything derives from the config.
    config = ZTEConfig.from_yaml(args.config)
    requested_split = config.train.split
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
    if args.precision is not None:
        config.train.precision = args.precision
    if args.num_workers is not None:
        config.train.num_workers = args.num_workers
    if args.compile is not None:
        config.train.compile_model = args.compile == 'on'
    if args.static_shapes is not None:
        config.train.static_shapes = args.static_shapes
    if args.subjects is not None:
        config.dataset.subjects = tuple(args.subjects.split(','))
    if args.tasks is not None:
        config.dataset.tasks = tuple(args.tasks.split(','))
    if args.mode is not None:
        config.train.mode = args.mode
    if args.encoder_ckpt is not None:
        config.train.encoder_ckpt = args.encoder_ckpt
    if args.lm is not None:
        config.decoder.lm_source = args.lm
    if args.conditioning is not None:
        config.decoder.conditioning = args.conditioning
    if args.stage0_epochs is not None:
        config.decoder.stage0_epochs = args.stage0_epochs
    if args.decode_eval:
        config.objective.eval_generation = True

    warn_on_split_override(config, requested_split)

    run_dir = Path(args.out_root) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Fast resume: a fully-complete run is skipped instantly, without reloading the (large) bundle.
    if args.resume and not args.force and _run_is_complete(run_dir, config, args):
        _LOG.info('Resume: %r already complete (all stages done); skipping.', config.run_name)
        return

    _LOG.info('=== Experiment %r -> %s ===', config.run_name, run_dir)

    # Settle the cache location BEFORE resolving the root: `_resolve_root` asks the store whether the
    # bundle already exists, and skips the multi-GB, multi-minute extraction if it does.
    config.dataset.cache_dir = args.data_cache or str(run_dir / 'cache')
    if args.data_cache_remote:
        config.dataset.cache_remote = args.data_cache_remote

    # `str`, not `Path`: `DatasetConfig.root` is typed `str` and a Path is not YAML-serialisable.
    config.dataset.root = _resolve_root(args, config)
    config.train.ckpt_dir = str(run_dir / 'checkpoints')
    config.train.tensorboard = not args.no_tensorboard
    if args.drive_backup:
        # The checkpoint manager copies its `checkpoints/` dir into this path, so it must be per-run.
        config.train.drive_backup_dir = str(Path(args.drive_backup) / config.run_name)

    # Written before anything expensive starts: a run killed mid-training is then still reproducible
    # (and resumable) from its own directory, without reconstructing the CLI overrides by hand.
    config.to_yaml(run_dir / 'config.yaml')

    # Recorded because "it OOMed / it was slow" is unanswerable without knowing which accelerator ran it,
    # and a run's artifacts otherwise carry no trace of the hardware.
    from zte.device import resolve_device

    _spec = resolve_device(config.train.device, config.train.precision)

    manifest: dict[str, Any] = {
        'run_name': config.run_name,
        'data_root': config.dataset.root,
        # Lets tooling exclude smoke runs from backups (zte-pack --skip-synthetic).
        'synthetic': bool(args.synthetic),
        'device': _spec.name,
        'device_kind': _spec.kind,
        'batch_size': config.train.batch_size,
        'grad_checkpoint': bool(config.model.grad_checkpoint),
    }

    # A shared cache already holds the processed bundle, making the per-run copy redundant.
    bundle_dir = run_dir / 'bundle'
    shared_cache = args.data_cache is not None

    # 1) Prepare (build + cache + save bundle + overview figures). On --resume, reuse the bundle.
    if args.resume and not shared_cache and (bundle_dir / 'meta.json').exists() and not args.force:
        _LOG.info('[1/4] Resume: loading cached dataset bundle (skipping prepare) ...')
        dataset = ZuCoDataset.load(bundle_dir)
    else:
        _LOG.info(
            '[1/4] Preparing dataset%s ...',
            f' (shared cache: {args.data_cache})' if shared_cache else '',
        )
        dataset = ZuCoDataset(config.dataset).build(force=args.force)
        if not shared_cache:
            dataset.save(bundle_dir)
    manifest['dataset'] = dataset.analyze()
    _save_overview(dataset, run_dir / 'figures')

    # 1b) Provision --spatial / --meaning after the bundle, so the GloVe file can be vocab-restricted.
    _apply_provisioning(config, args, dataset)
    config.to_yaml(run_dir / 'config.yaml')  # now includes the provisioned montage / meaning target
    _mirror_to_drive(run_dir, args, 'prepare')

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
    manifest['final_train_loss'] = artifacts.history['train_loss'][-1] if artifacts.history['train_loss'] else None
    # Code state, hardware, schedule and library versions: without them a number cannot be placed.
    manifest['provenance'] = artifacts.trainer.provenance()
    _save_curves(artifacts.history, run_dir / 'checkpoints' / 'training_curves.png')
    # The per-epoch curves in machine-readable form: `zte-analyze` plots the mechanism metrics from this file, and a
    # PNG cannot be re-read.
    (run_dir / 'history.json').write_text(json.dumps(artifacts.history, indent=2), encoding='utf-8')
    _mirror_to_drive(run_dir, args, 'train')

    # 3) Evaluate, unless the metrics on disk are at least as new as the checkpoint.
    metrics_path = run_dir / 'evaluation' / 'metrics.json'
    best_ckpt = run_dir / 'checkpoints' / 'best.pt'
    needs_generation = config.train.mode != 'encoder' and getattr(config.objective, 'eval_generation', False)
    eval_fresh = (
        metrics_path.exists()
        and (not best_ckpt.exists() or metrics_path.stat().st_mtime >= best_ckpt.stat().st_mtime)
        and (not needs_generation or (run_dir / 'evaluation' / 'generation.json').exists())
    )
    if not args.skip_eval:
        if args.resume and eval_fresh and not args.force:
            _LOG.info('[3/4] Resume: evaluation already up to date, skipping.')
            manifest['evaluation'] = _eval_summary_from_disk(metrics_path)
        else:
            _LOG.info('[3/4] Evaluating ...')
            manifest['evaluation'] = _evaluate(config, dataset, run_dir, args)
            _mirror_to_drive(run_dir, args, 'evaluate')

    # 4) Explore (brain regions + eye-tracking) when band power is available.
    explore_done = (run_dir / 'exploration' / 'report.md').exists()
    if not args.skip_explore and dataset.band_power_raw is not None:
        if args.resume and explore_done and not args.force:
            _LOG.info('[4/4] Resume: exploration already done, skipping.')
        else:
            _LOG.info('[4/4] Exploring brain regions + eye-tracking ...')
            from zte.cli.explore import run_exploration

            summary = run_exploration(dataset, run_dir / 'exploration', montage_csv=config.dataset.montage_csv)
            manifest['region_map_approximate'] = summary['region_map_approximate']
    elif not args.skip_explore:
        # Exploration needs band power, which `representation: raw` never loads.
        _LOG.warning(
            '[4/4] Skipped: exploration needs band power, but representation=%r loads none. '
            "Use representation: 'both' to explore regions for a raw frontend.",
            config.dataset.representation,
        )
        manifest['exploration_skipped'] = f'no band power (representation={config.dataset.representation})'

    # Catalogue: manifest, per-run README and the shared index row.
    write_json(run_dir / 'manifest.json', manifest, default=str)
    (run_dir / 'README.md').write_text(_render_run_readme(config, manifest, run_dir), encoding='utf-8')
    _catalogue(Path(args.out_root), config.run_name, manifest, remote_index=_remote_index_path(args))
    _mirror_to_drive(run_dir, args, 'catalogue', index=Path(args.out_root) / 'INDEX.md')
    _LOG.info('Done. Everything catalogued under %s', run_dir.resolve())


def _mirror_to_drive(run_dir: Path, args: argparse.Namespace, stage: str, index: Path | None = None) -> None:
    """Mirrors the run directory to the mounted Drive backup folder after a completed stage.

    The checkpoint manager already mirrors `checkpoints/` every epoch; this adds everything else the
    run produces (config, figures, evaluation, TensorBoard) so a reclaimed Colab VM never costs more
    than the epoch in flight. Heavy, regenerable directories (`cache/`, `bundle/`) are skipped -- point
    `--data-cache` at a Drive path to persist those once instead of per run.

    Args:
        run_dir (Path): The run directory to mirror.
        args (argparse.Namespace): Parsed CLI args (for `--drive-backup`).
        stage (str): Stage name, for the log line.
        index (Path | None): Optional catalogue file to copy alongside the run.
    """
    if not args.drive_backup:
        return
    from zte.data.io.remote import is_mounted_path

    if not is_mounted_path(args.drive_backup):
        # A Drive id/URL cannot be mirrored file-by-file; the checkpoint zip-upload path covers it.
        return
    from zte.utils.mirror import mirror_file, mirror_tree

    dest = Path(args.drive_backup) / run_dir.name
    copied, failed = mirror_tree(run_dir, dest)
    if index is not None:
        mirror_file(index, Path(args.drive_backup))
    _LOG.info(
        '[drive] %s: %d file(s) mirrored to %s%s',
        stage,
        copied,
        dest,
        f' ({failed} failed)' if failed else '',
    )


def _apply_provisioning(config: ZTEConfig, args: argparse.Namespace, dataset: ZuCoDataset) -> None:
    """Builds + wires the `--spatial` / `--meaning` ingredients into `config`.

    The training word set is pulled from the built dataset so `--meaning static` writes only the GloVe
    rows this dataset needs; a no-op when both flags are left at `keep`.
    """
    vocab: set[str] | None = None
    words = getattr(dataset, 'words', None)
    if words is not None and 'word' in getattr(words, 'columns', []):
        vocab = {w.lower() for w in words['word'].dropna().astype(str) if w}
    provision_from_args(config, args, vocab=vocab)


def _last_completed_epoch(ckpt_dir: Path) -> int:
    """Returns the highest completed epoch from the `ckpt_epoch*.pt` filenames (no torch.load)."""
    epochs = [int(p.stem.removeprefix('ckpt_epoch')) for p in ckpt_dir.glob('ckpt_epoch*.pt')]
    return max(epochs) if epochs else 0


def _run_is_complete(run_dir: Path, config: ZTEConfig, args: argparse.Namespace) -> bool:
    """Whether a run has finished every stage, so `--resume` can skip it without loading anything.

    Returns:
        bool: `True` when nothing remains to do.
    """
    if not (run_dir / 'checkpoints' / 'last.pt').exists():
        return False
    if _last_completed_epoch(run_dir / 'checkpoints') < config.train.epochs:
        return False
    if not args.skip_eval and not (run_dir / 'evaluation' / 'metrics.json').exists():
        return False
    # A decoder run's deliverable is the generation block, so it is not complete without one.
    if (
        not args.skip_eval
        and config.train.mode != 'encoder'
        and getattr(config.objective, 'eval_generation', False)
        and not (run_dir / 'evaluation' / 'generation.json').exists()
    ):
        return False
    if not args.skip_explore and not (run_dir / 'exploration' / 'report.md').exists():
        # Exploration only runs when band power is available, so a prior manifest is the arbiter.
        if (run_dir / 'manifest.json').exists():
            manifest = read_json(run_dir / 'manifest.json')
            if 'region_map_approximate' not in manifest:
                return True  # this run legitimately has no exploration stage
        return False
    return True


def _eval_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Extracts the manifest evaluation summary from a full metrics dict.

    For a LOSO run the honest headline is `held_out_retrieval` (the never-seen subject alone), not the
    pooled `sentence_retrieval` -- which is dominated by the training subjects and reads far higher. Both
    are carried so the catalogue can show the honest one and flag the pooled one as inflated.
    """
    board = metrics.get('scoreboard') or {}
    held = board.get('held_out_retrieval') or {}
    rescoring = board.get('decoder_rescoring_retrieval') or {}
    generation = metrics.get('generation') or {}
    worst = generation.get('worst_control_ci') or {}
    return {
        'verdict': metrics.get('verdict'),
        'sentence_retrieval_top1': metrics.get('sentence_retrieval', {}).get('top1'),
        'held_out_retrieval_top1': held.get('top1'),
        'held_out_retrieval_lift': held.get('lift_top1'),
        'subject_transfer_top1': metrics.get('analogy', {}).get('subject_transfer', {}).get('top1'),
        'effective_rank_ratio': metrics.get('embedding_health', {}).get('effective_rank_ratio'),
        # Generation is reported as a delta against its worst control, never as an absolute score.
        'generation_worst_control': generation.get('worst_control'),
        'generation_delta': worst.get('point'),
        'generation_delta_ci': [worst.get('lo'), worst.get('hi')] if worst else None,
        'decoder_rescoring_rank_percentile': rescoring.get('rank_percentile'),
    }


def _eval_summary_from_disk(metrics_path: Path) -> dict[str, Any]:
    """Rebuilds the manifest evaluation summary from an existing `metrics.json` (resume path)."""
    return _eval_summary(read_json(metrics_path))


def _evaluate(config: ZTEConfig, dataset: ZuCoDataset, run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Embeds the best checkpoint and runs the full evaluation suite."""
    from zte.cli.decode import decoder_blocks
    from zte.cli.evaluate import (
        collect_embeddings,
        phase_shuffled_sent_emb,
        phase_shuffled_word_emb,
        train_split_sent_emb,
        training_vocab,
    )
    from zte.evaluation.report import evaluate_representation
    from zte.inference.embed import ZTEEmbedder

    # `last.pt` is the fallback so a run whose best checkpoint never materialised (a diverged metric)
    # can still be evaluated instead of stranding hours of finished training behind a FileNotFoundError.
    ckpt = run_dir / 'checkpoints' / 'best.pt'
    if not ckpt.is_file():
        ckpt = run_dir / 'checkpoints' / 'last.pt'
        _LOG.warning('No best.pt; evaluating %s instead.', ckpt.name)
    embedder = ZTEEmbedder.from_checkpoint(ckpt, dataset)
    word_emb, word_meta, raw_feats, sent_emb, sent_ids, sent_meta, word_bp = collect_embeddings(embedder, dataset)

    # Opt-in evaluation-hardening inputs (config-gated so default runs stay fast).
    phase_shuffle = bool(getattr(config.objective, 'eval_phase_shuffle', False))
    phase_emb = phase_shuffled_word_emb(embedder, dataset) if phase_shuffle else None
    phase_sent = phase_shuffled_sent_emb(embedder, dataset) if phase_shuffle else None

    # Post-processing fitted on these rows is reproducible one sentence at a time; the scored rows are not.
    train_sent, train_sent_n_words = train_split_sent_emb(embedder, dataset, config)

    train_vocab = training_vocab(dataset, config) if getattr(config.objective, 'eval_seen_novel', False) else None

    generation, rescoring = decoder_blocks(
        ckpt,
        dataset,
        config,
        out_dir=run_dir / 'evaluation',
        device=config.train.device,
        run_name=config.run_name,
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
        phase_word_emb=phase_emb,
        train_vocab=train_vocab,
        phase_sent_emb=phase_sent,
        train_sent_emb=train_sent,
        train_sent_n_words=train_sent_n_words,
        sent_n_words=_sent_n_words(config, sent_meta),
        generation=generation,
        rescoring=rescoring,
        min_prefix_kl=config.decoder.min_prefix_kl,
    )

    return _eval_summary(metrics)


def _sent_n_words(config: ZTEConfig, sent_meta: Any) -> Any:
    """Word count per reading, which turns on the length-stratified gallery beside the full one."""
    if not getattr(config.objective, 'eval_length_stratified', False):
        return None
    return None if 'n_words' not in sent_meta else sent_meta['n_words'].to_numpy()


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
        *(
            [
                f'- **Held-out retrieval Top-1 (the honest LOSO headline): '
                f'{ev.get("held_out_retrieval_top1")}** (lift over chance {ev.get("held_out_retrieval_lift")})',
                f'- Pooled sentence retrieval Top-1 (inflated by training subjects): '
                f'{ev.get("sentence_retrieval_top1")}',
            ]
            if ev.get('held_out_retrieval_top1') is not None
            else [f'- Cross-subject sentence retrieval Top-1: {ev.get("sentence_retrieval_top1")}']
        ),
        *(
            [
                f'- Free-running generation delta vs its worst control '
                f'(`{ev.get("generation_worst_control")}`): {ev.get("generation_delta")} '
                f'CI {ev.get("generation_delta_ci")} -- an absolute score here would not be a result',
                f'- Decoder-rescoring **retrieval** rank percentile: {ev.get("decoder_rescoring_rank_percentile")}',
            ]
            if ev.get('generation_delta') is not None or ev.get('decoder_rescoring_rank_percentile') is not None
            else []
        ),
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


# Normalised on every write: rows from any older column layout cannot be reconciled with this table.
_INDEX_HEADER: Final[str] = (
    '# ZTE experiment catalogue\n\n'
    '| run | words | held-out retrieval Top-1 (honest) | pooled retrieval Top-1 (inflated) | '
    'subject-transfer Top-1 | eff-rank ratio |\n'
    '| --- | --- | --- | --- | --- | --- |\n'
)
"""Header every catalogue write emits before its rows."""


def _remote_index_path(args: argparse.Namespace) -> Path | None:
    """Returns the Drive-side `INDEX.md` the catalogue mirror will overwrite, or `None` without a mounted target."""
    if not args.drive_backup:
        return None
    from zte.data.io.remote import is_mounted_path

    if not is_mounted_path(args.drive_backup):
        return None

    return Path(args.drive_backup) / 'INDEX.md'


def _index_rows(index: Path | None) -> dict[str, str]:
    """Returns a catalogue's data rows keyed by run name in file order; empty when absent, unreadable or pre-held-out."""
    if index is None:
        return {}
    try:
        text = index.read_text(encoding='utf-8')
    except OSError:
        return {}
    # A pre-held-out INDEX has a different column layout, so its rows cannot be reconciled with the header.
    if 'held-out retrieval' not in text:
        return {}

    rows: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith('|'):
            continue
        name = line.split('|')[1].strip()
        # The header row and the `| --- |` separator are re-emitted from `_INDEX_HEADER`, never carried as data.
        if name and name != 'run' and set(name) != {'-'}:
            rows[name] = line

    return rows


# The catalogue mirror pushes the local file whole over the Drive copy, and a fresh VM starts with no local
# rows at all -- so the write must be the union of both indexes keyed by run name, local rows winning on
# conflict. A session can then add or update its own runs but never erase another session's; an unreachable
# remote degrades to local-only.
def _catalogue(out_root: Path, run_name: str, manifest: dict[str, Any], remote_index: Path | None = None) -> None:
    """Appends/updates this run's row in the experiments index, merged with the mirrored Drive copy."""
    index = out_root / 'INDEX.md'
    ev = manifest.get('evaluation', {})
    # Held-out retrieval is the honest LOSO headline; pooled sentence retrieval is shown but flagged as
    # inflated (it counts the training subjects). A non-LOSO run leaves the held-out column blank.
    held = ev.get('held_out_retrieval_top1')
    row = (
        f'| {run_name} | {manifest.get("dataset", {}).get("n_words", "-")} | '
        f'{held if held is not None else "-"} | '
        f'{ev.get("sentence_retrieval_top1", "-")} | {ev.get("subject_transfer_top1", "-")} | '
        f'{ev.get("effective_rank_ratio", "-")} |'
    )

    # Remote rows first, then local rows over them, then this run's row: last write wins per run name.
    rows = _index_rows(remote_index)
    rows |= _index_rows(index)
    rows[run_name] = row
    index.write_text(_INDEX_HEADER + '\n'.join(rows.values()) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
