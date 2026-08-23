"""`zte-parallax` -- per-task encoders scored across tasks: transfer cells, the matrix report, the chamber."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from zte.cli.support.done import add_force_argument, checkpoint_digest, is_done, mark_done, signature
from zte.cli.support.io import read_json
from zte.cli.support.sources import add_data_source_args, add_extract_dir, dataset_for_config, dataset_key
from zte.config import ZTEConfig
from zte.logging_utils import configure_logging, get_logger
from zte.parallax.study import PARALLAX_TASKS, cell_name, derive_eval_config, resolve_transfer_holdout
from zte.parallax.transfer import transfer_report, write_cell

_LOG = get_logger('cli.parallax')


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Defines and parses the `zte-parallax` command-line arguments.

    Args:
        argv (list[str] | None, optional): Arguments to parse instead of `sys.argv`. Defaults to None.

    Returns:
        argparse.Namespace: The parsed argument namespace, with the subcommand under `command`.
    """
    parser = argparse.ArgumentParser(
        description='The parallax study: three per-task encoders scored across tasks on a never-seen subject. '
        '`transfer` scores one matrix cell, `report` aggregates a directory of cells, `chamber` renders the report.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    sub = parser.add_subparsers(dest='command', required=True)

    transfer = sub.add_parser(
        'transfer',
        help='Score one checkpoint on one eval task -- one cell of the 3x3 matrix.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    transfer.add_argument('--ckpt', type=str, required=True, help='Checkpoint (best.pt/last.pt) of a parallax arm.')
    transfer.add_argument(
        '--eval-task',
        choices=list(PARALLAX_TASKS),
        required=True,
        dest='eval_task',
        help='Task whose readings the model is scored on.',
    )
    transfer.add_argument('--out', type=Path, required=True, help='Directory transfer cells are written under.')
    add_data_source_args(transfer, include_bundle=True, include_synthetic=True)
    add_extract_dir(transfer)
    transfer.add_argument(
        '--holdout',
        type=str,
        default=None,
        help="Held-out subject whose readings are the queries. Defaults to the checkpoint's training holdout, "
        'and must match it -- scoring a subject the model trained on would be labelled held-out falsely.',
    )
    transfer.add_argument('--n-boot', type=int, default=2000, dest='n_boot', help='Bootstrap resamples behind each CI.')
    transfer.add_argument('--seed', type=int, default=0)
    transfer.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    add_force_argument(transfer)

    report = sub.add_parser(
        'report',
        help='Aggregate transfer cells into PARALLAX.json, PARALLAX.md and CHAMBER_DATA.json.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    report.add_argument('--transfers', type=Path, required=True, help='Directory of transfer cell directories.')
    report.add_argument('--out', type=Path, required=True, help='Directory the three report files are written to.')

    chamber = sub.add_parser(
        'chamber',
        help='Render the interactive chamber page from a report directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    chamber.add_argument(
        '--report-dir', type=Path, required=True, dest='report_dir', help='Directory holding CHAMBER_DATA.json.'
    )
    chamber.add_argument('--out', type=Path, required=True, help='Output HTML path.')

    return parser.parse_args(argv)


def run_transfer(args: argparse.Namespace) -> dict[str, Any]:
    """Scores one transfer cell from a checkpoint and writes it under `--out`.

    Args:
        args (argparse.Namespace): The parsed `transfer` arguments.

    Returns:
        dict[str, Any]: The cell's `transfer.json` dict.

    Raises:
        SystemExit: If a cross-task cell is requested from a `--bundle` (disjointness unverifiable), the
            checkpoint names no LOSO holdout, or `--holdout` differs from the training holdout.
        ValueError: If the sentence metadata lacks the columns the audit stratifies on.
    """
    from zte.cli.evaluate import collect_embeddings
    from zte.device import resolve_device
    from zte.inference.embed import ZTEEmbedder
    from zte.training.checkpoint import CheckpointManager
    from zte.training.init import file_sha256
    from zte.utils.provenance import git_info

    payload = CheckpointManager.load(args.ckpt, map_location='cpu')
    config = ZTEConfig.from_dict(payload['config'])
    train_tasks = tuple(str(t) for t in config.dataset.tasks)
    train_task = train_tasks[0] if len(train_tasks) == 1 else '+'.join(train_tasks)

    # The held-out claim rests on this guard: the queries must be the subject the checkpoint actually held out.
    try:
        holdout = resolve_transfer_holdout(config, args.holdout)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # A bundle holds exactly one processed dataset; a cross-task cell needs the training task built
    # alongside the eval task to measure stimulus overlap, so the raw source is mandatory there.
    cross_task = set(train_tasks) != {args.eval_task}
    if cross_task and getattr(args, 'bundle', None):
        raise SystemExit(
            'A cross-task cell cannot be scored from --bundle: verifying stimulus disjointness needs the '
            'training task too. Pass --root, --drive or --synthetic instead.'
        )

    eval_config = derive_eval_config(config, args.eval_task)

    # A cell is one checkpoint scored on one task: same weights, same eval task, same bootstrap, same cell. The
    # guard sits before the build so a re-run costs a hash rather than a multi-GB stage and an embedding pass.
    cell_dir = Path(args.out) / cell_name(train_task, args.eval_task, args.seed)
    artifacts = (cell_dir / 'transfer.json', cell_dir / 'embeddings.npz')
    sig = signature(
        args,
        tool='parallax-transfer',
        extra={
            'ckpt_sha256': checkpoint_digest(args.ckpt),
            'dataset': dataset_key(eval_config.dataset),
            'holdout': holdout,
        },
        ignore=('ckpt', 'holdout'),
    )
    if is_done(artifacts, sig, force=args.force):
        existing = dict(read_json(artifacts[0]))
        _log_cell(cell_dir, existing)

        return existing

    eval_dataset = dataset_for_config(args, eval_config.dataset)

    embedder = ZTEEmbedder.from_checkpoint(args.ckpt, eval_dataset, device=resolve_device(args.device))
    _, _, _, sent_emb, sent_ids, sent_meta, _ = collect_embeddings(embedder, eval_dataset)
    for column in ('subject', 'n_words', 'text'):
        if column not in sent_meta.columns:
            raise ValueError(f'Sentence metadata carries no {column!r} column; cannot score the cell.')

    subjects = sent_meta['subject'].astype(str).to_numpy()
    n_words = sent_meta['n_words'].to_numpy()
    texts = sent_meta['text'].astype(str).to_numpy()

    if cross_task:
        train_dataset = dataset_for_config(args, config.dataset)
        train_stimuli = set(train_dataset.sentences['text'].astype(str)) if len(train_dataset.sentences) else set()
    else:
        train_stimuli = {str(t) for t in texts}

    git = git_info()
    provenance = {
        'ckpt': str(args.ckpt),
        'ckpt_sha256': file_sha256(args.ckpt),
        'git_commit': git['commit'],
        'run_name': config.run_name,
        'train_tasks': list(train_tasks),
        'train_holdout': holdout,
        # best.pt is picked on val loss, and by_subject_loso validates on the holdout: parameters never saw the
        # holdout, model *selection* did. The claim wording in PARALLAX.md carries this qualifier.
        'checkpoint_selection': 'val loss (val split = holdout under by_subject_loso)',
    }
    report = transfer_report(
        sent_emb,
        np.asarray(sent_ids),
        subjects,
        n_words,
        texts,
        train_task=train_task,
        eval_task=args.eval_task,
        holdout=holdout,
        train_stimulus_texts=train_stimuli,
        seed=args.seed,
        n_boot=args.n_boot,
        provenance=provenance,
    )

    write_cell(
        cell_dir,
        report,
        sent_emb=sent_emb,
        content_ids=np.asarray(sent_ids),
        subjects=subjects,
        n_words=n_words,
        texts=texts,
    )
    mark_done(artifacts, sig)
    _log_cell(cell_dir, report)

    return report


def _log_cell(cell_dir: Path, report: dict[str, Any]) -> None:
    """Logs one transfer cell's headline, whether it was just scored or read back from disk."""
    held = report.get('held_out') or {}
    _LOG.info(
        'Cell %s: rank percentile %s (CI %s), novel stimuli %s, %s queries.',
        cell_dir.name,
        held.get('rank_percentile'),
        held.get('rank_percentile_ci'),
        report.get('novel_stimuli'),
        report.get('n_queries'),
    )


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    """Aggregates a directory of transfer cells into the three report files.

    Args:
        args (argparse.Namespace): The parsed `report` arguments.

    Returns:
        dict[str, Any]: The dict written to `PARALLAX.json`.
    """
    from zte.parallax.report import build_report

    return build_report(args.transfers, args.out)


def run_chamber(args: argparse.Namespace) -> Path:
    """Renders the chamber page from a report directory, through the deferred chamber module.

    Args:
        args (argparse.Namespace): The parsed `chamber` arguments.

    Returns:
        Path: The written HTML path.

    Raises:
        SystemExit: If the chamber renderer is not available in this build.
    """
    try:
        from zte.parallax.chamber import build_chamber
    except ImportError as exc:
        raise SystemExit(f'The chamber renderer is not available in this build: {exc}') from exc

    out = build_chamber(Path(args.report_dir), Path(args.out))
    _LOG.info('Chamber written to %s.', out)
    return out


def main() -> None:
    """Dispatches the `zte-parallax` subcommands."""
    args = parse_arguments()
    configure_logging(args.log_level)
    match args.command:
        case 'transfer':
            run_transfer(args)
        case 'report':
            run_report(args)
        case 'chamber':
            run_chamber(args)


if __name__ == '__main__':
    main()
