"""`zte-rebaseline` -- re-score an existing checkpoint's held-out retrieval against the sentence-length floor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import numpy as np

from zte.cli.support.io import write_json
from zte.cli.support.sources import add_data_source_args, add_extract_dir, dataset_for_config
from zte.config import ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.device import DeviceKind, resolve_device
from zte.evaluation.audit.rebaseline import rebaseline_report, render_markdown
from zte.inference.embed import ZTEEmbedder
from zte.logging_utils import configure_logging, get_logger
from zte.training.checkpoint import CheckpointManager

_LOG = get_logger('cli.rebaseline')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-rebaseline` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Audit held-out sentence retrieval against the word-count floor: three post-processing '
        'conditions x two galleries, the length-only oracle and the bit budget. Reports; gates nothing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint (best.pt/last.pt).')
    add_data_source_args(parser, include_bundle=True, include_synthetic=True)
    add_extract_dir(parser)

    parser.add_argument(
        '--out',
        type=Path,
        default=None,
        help="Output directory. Default: the run's own `rebaseline/` beside `checkpoints/`.",
    )
    parser.add_argument(
        '--holdout',
        type=str,
        default=None,
        help='Held-out subject whose readings are the queries. Default: the run config'
        "'s loso_holdout_subject.",
    )
    parser.add_argument(
        '--length-tol',
        type=int,
        default=1,
        dest='length_tol',
        help='Word-count tolerance defining the length-stratified gallery.',
    )
    parser.add_argument(
        '--oracle-tol',
        type=str,
        default='0,1,2,4',
        dest='oracle_tol',
        help='Comma-separated tolerances for the length-only oracle floor.',
    )
    parser.add_argument(
        '--n-boot',
        type=int,
        default=2000,
        dest='n_boot',
        help='Bootstrap resamples behind the rank-percentile confidence interval.',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument(
        '--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    )
    return parser.parse_args()


def default_out_dir(ckpt: str | Path) -> Path:
    """Returns `<run>/rebaseline` for a checkpoint at `<run>/checkpoints/<name>.pt`."""
    return Path(ckpt).resolve().parent.parent / 'rebaseline'


def resolve_holdout(config: ZTEConfig, requested: str | None, subjects: np.ndarray) -> str | None:
    """Picks the query subject: the CLI's, else the run's LOSO holdout, else nothing to audit.

    Args:
        config (ZTEConfig): The checkpoint's configuration.
        requested (str | None): The `--holdout` value, or None.
        subjects (np.ndarray): Subject code per reading.

    Returns:
        str | None: The held-out subject code, or `None` when the run held nobody out and none was named.
    """
    if requested:
        return requested
    holdout = config.train.loso_holdout_subject
    if holdout:
        return str(holdout)
    unique = np.unique(subjects)
    if unique.size < 2:
        return None
    _LOG.warning(
        'Run config names no held-out subject; auditing %r. Pass --holdout to choose another.',
        str(unique[-1]),
    )
    return str(unique[-1])


def run_rebaseline(
    ckpt: str | Path,
    dataset: ZuCoDataset,
    *,
    out_dir: Path,
    holdout: str | None = None,
    length_tol: int = 1,
    oracle_tols: tuple[int, ...] = (0, 1, 2, 4),
    n_boot: int = 2000,
    seed: int = 0,
    device: DeviceKind | Literal['auto'] = 'auto',
) -> dict[str, Any]:
    """Embeds a checkpoint's sentences and writes the length-confound audit beside them.

    Args:
        ckpt (str | Path): The checkpoint to re-score. No training happens.
        dataset (ZuCoDataset): The built dataset the checkpoint was trained on.
        out_dir (Path): Where `rebaseline.json` and `rebaseline.md` are written.
        holdout (str | None, optional): Query subject. Defaults to None, which reads the run config.
        length_tol (int, optional): Word-count tolerance for the stratified gallery. Defaults to 1.
        oracle_tols (tuple[int, ...], optional): Tolerances for the floor. Defaults to (0, 1, 2, 4).
        n_boot (int, optional): Bootstrap resamples. Defaults to 2000.
        seed (int, optional): Bootstrap seed. Defaults to 0.
        device (DeviceKind | Literal['auto'], optional): Device selector. Defaults to 'auto'.

    Returns:
        dict[str, Any]: The audit report, also written to `rebaseline.json`.

    Raises:
        ValueError: If the embedded readings carry no subject or word-count column to stratify on.
    """
    from zte.cli.evaluate import collect_embeddings

    embedder = ZTEEmbedder.from_checkpoint(ckpt, dataset, device=resolve_device(device))
    _, _, _, sent_emb, sent_ids, sent_meta, _ = collect_embeddings(embedder, dataset)
    for column in ('subject', 'n_words'):
        if column not in sent_meta.columns:
            raise ValueError(
                f'Sentence metadata carries no {column!r} column; cannot audit length.'
            )

    subjects = sent_meta['subject'].astype(str).to_numpy()
    n_words = sent_meta['n_words'].to_numpy()
    subject = resolve_holdout(embedder.config, holdout, subjects)
    if subject is None:
        raise ValueError(
            'Nothing to hold out: the dataset has a single subject and none was named.'
        )

    report = rebaseline_report(
        sent_emb,
        np.asarray(sent_ids),
        subjects,
        subject,
        n_words,
        length_tol=length_tol,
        oracle_tols=oracle_tols,
        ks=(1, 5, 10),
        n_boot=n_boot,
        seed=seed,
    )
    report['provenance'] = _provenance(ckpt, embedder.config, n_boot, seed, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / 'rebaseline.json', report, default=str)
    (out_dir / 'rebaseline.md').write_text(render_markdown(report), encoding='utf-8')
    return report


def _provenance(
    ckpt: str | Path, config: ZTEConfig, n_boot: int, seed: int, device: str
) -> dict[str, Any]:
    """Records what a results table needs to place this audit next to the run it re-scores."""
    from zte.training.init import file_sha256
    from zte.utils.provenance import git_info

    git = git_info()
    payload = CheckpointManager.load(ckpt, map_location='cpu')
    return {
        'ckpt': str(ckpt),
        'ckpt_sha256': file_sha256(ckpt),
        'ckpt_epoch': payload.get('epoch'),
        'run_name': config.run_name,
        'objective': config.objective.name,
        'frontend': config.model.frontend,
        'split': config.train.split,
        'whiten': bool(config.objective.whiten),
        'all_but_top': int(config.objective.all_but_top or 0),
        'n_boot': int(n_boot),
        'seed': int(seed),
        'device': device,
        'git_commit': git['commit'],
        'git_dirty': git['dirty'],
    }


def main() -> None:
    """Runs the length-confound audit against an existing checkpoint, with no retraining."""
    args = parse_arguments()
    configure_logging(args.log_level)

    payload = CheckpointManager.load(args.ckpt, map_location='cpu')
    config = ZTEConfig.from_dict(payload['config'])
    dataset = dataset_for_config(args, config.dataset)

    out_dir = Path(args.out) if args.out else default_out_dir(args.ckpt)
    report = run_rebaseline(
        args.ckpt,
        dataset,
        out_dir=out_dir,
        holdout=args.holdout,
        length_tol=args.length_tol,
        oracle_tols=tuple(int(t) for t in str(args.oracle_tol).split(',') if t.strip()),
        n_boot=args.n_boot,
        seed=args.seed,
        device=args.device,
    )

    floor = report.get('floor_comparison') or {}
    budget = report.get('bit_budget') or {}
    _LOG.info(
        'Honest cell rank percentile %s (CI low %s) vs the ±%s length oracle at %s -- clears floor: %s.',
        floor.get('encoder'),
        floor.get('encoder_ci_low'),
        floor.get('oracle_tol'),
        floor.get('oracle'),
        floor.get('clears_floor'),
    )
    _LOG.info(
        'Bit budget: %.4f needed, %.4f from word count, %.4f from EEG.',
        budget.get('bits_needed', float('nan')),
        budget.get('bits_from_length', float('nan')),
        budget.get('bits_from_eeg', float('nan')),
    )
    _LOG.info('Audit written to %s (rebaseline.json + rebaseline.md).', out_dir)


if __name__ == '__main__':
    main()
