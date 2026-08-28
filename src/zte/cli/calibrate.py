"""`zte-calibrate` -- how much a new reader's held-out retrieval improves per labelled sentence they read."""

import argparse
from pathlib import Path
from typing import Any, Literal

import numpy as np

from zte.cli.support.done import add_force_argument, checkpoint_digest, is_done, mark_done, signature
from zte.cli.support.io import read_json, write_json
from zte.cli.support.sources import add_data_source_args, add_extract_dir, dataset_for_config, dataset_key
from zte.config import ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.device import DeviceKind, resolve_device
from zte.evaluation.audit.calibration import DEFAULT_ANCHOR_COUNTS, MapFamily, calibration_curve, render_markdown
from zte.inference.embed import ZTEEmbedder
from zte.logging_utils import configure_logging, get_logger
from zte.training.checkpoint import CheckpointManager

_LOG = get_logger('cli.calibrate')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-calibrate` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Sweep held-out sentence retrieval against the number of labelled readings a brand-new reader '
        'supplies. No retraining: a map is fitted from the anchors alone, and every anchor stimulus is dropped from '
        'the queries and from the gallery. Reports; gates nothing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint (best.pt/last.pt).')
    add_data_source_args(parser, include_bundle=True, include_synthetic=True)
    add_extract_dir(parser)

    parser.add_argument(
        '--out',
        type=Path,
        default=None,
        help="Output directory. Default: the run's own `calibration/` beside `checkpoints/`.",
    )
    parser.add_argument(
        '--holdout',
        type=str,
        default=None,
        help="The new reader, whose readings are the queries. Default: the run config's loso_holdout_subject.",
    )
    parser.add_argument(
        '--anchor-counts',
        type=str,
        default=','.join(str(c) for c in DEFAULT_ANCHOR_COUNTS),
        dest='anchor_counts',
        help='Comma-separated anchor counts to sweep. 0 is the uncalibrated control.',
    )
    parser.add_argument(
        '--draws',
        type=int,
        default=5,
        help='Seeded anchor draws per count -- which sentences the reader got is a real source of variance.',
    )
    parser.add_argument(
        '--family',
        choices=['procrustes', 'ridge', 'both'],
        default='both',
        help='Map family: a rotation that cannot rescale a score, a regularised affine map that can, or both.',
    )
    parser.add_argument(
        '--ridge-alpha',
        type=float,
        default=1.0,
        dest='ridge_alpha',
        help='Ridge penalty for the affine family.',
    )
    parser.add_argument(
        '--postprocess',
        action='store_true',
        help='Fit whitening and all-but-the-top on the cohort rows before calibrating, making these numbers '
        'comparable to the train-fitted cell of `zte-rebaseline`.',
    )
    parser.add_argument(
        '--length-tol',
        type=int,
        default=1,
        dest='length_tol',
        help='Word-count tolerance defining the length-stratified gallery.',
    )
    parser.add_argument(
        '--n-boot',
        type=int,
        default=2000,
        dest='n_boot',
        help='Bootstrap resamples behind every interval.',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    add_force_argument(parser)
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    return parser.parse_args()


def default_out_dir(ckpt: str | Path) -> Path:
    """Returns `<run>/calibration` for a checkpoint at `<run>/checkpoints/<name>.pt`."""
    return Path(ckpt).resolve().parent.parent / 'calibration'


def resolve_families(choice: str) -> tuple[MapFamily, ...]:
    """Turns the `--family` choice into the map families to fit.

    Args:
        choice (str): One of `'procrustes'`, `'ridge'` or `'both'`.

    Returns:
        tuple[MapFamily, ...]: The families to sweep.
    """
    match choice:
        case 'procrustes':
            return ('procrustes',)
        case 'ridge':
            return ('ridge',)
        case _:
            return ('procrustes', 'ridge')


def resolve_holdout(config: ZTEConfig, requested: str | None, subjects: np.ndarray) -> str | None:
    """Picks the new reader: the CLI's, else the run's LOSO holdout, else nothing to calibrate.

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
        'Run config names no held-out subject; calibrating %r. Pass --holdout to choose another.',
        str(unique[-1]),
    )
    return str(unique[-1])


def run_calibration(
    ckpt: str | Path,
    dataset: ZuCoDataset,
    *,
    out_dir: Path,
    holdout: str | None = None,
    anchor_counts: tuple[int, ...] = DEFAULT_ANCHOR_COUNTS,
    families: tuple[MapFamily, ...] = ('procrustes', 'ridge'),
    draws: int = 5,
    ridge_alpha: float = 1.0,
    postprocess: bool = False,
    length_tol: int = 1,
    n_boot: int = 2000,
    seed: int = 0,
    device: DeviceKind | Literal['auto'] = 'auto',
) -> dict[str, Any]:
    """Embeds a checkpoint's sentences and writes the anchor-calibration curve beside them.

    Args:
        ckpt (str | Path): The checkpoint to calibrate against. No training happens.
        dataset (ZuCoDataset): The built dataset the checkpoint was trained on.
        out_dir (Path): Where `calibration.json` and `calibration.md` are written.
        holdout (str | None, optional): The new reader. Defaults to None, which reads the run config.
        anchor_counts (tuple[int, ...], optional): The sweep. Defaults to `DEFAULT_ANCHOR_COUNTS`.
        families (tuple[MapFamily, ...], optional): Map families to fit. Defaults to both.
        draws (int, optional): Seeded anchor draws per count. Defaults to 5.
        ridge_alpha (float, optional): Ridge penalty. Defaults to 1.0.
        postprocess (bool, optional): Fit cohort-only whitening and all-but-the-top first. Defaults to False.
        length_tol (int, optional): Word-count tolerance for the stratified gallery. Defaults to 1.
        n_boot (int, optional): Bootstrap resamples. Defaults to 2000.
        seed (int, optional): Anchor-draw and bootstrap seed. Defaults to 0.
        device (DeviceKind | Literal['auto'], optional): Device selector. Defaults to 'auto'.

    Returns:
        dict[str, Any]: The curve report, also written to `calibration.json`.

    Raises:
        ValueError: If the embedded readings carry no subject column, or the dataset holds a single subject.
    """
    from zte.cli.evaluate import collect_embeddings

    embedder = ZTEEmbedder.from_checkpoint(ckpt, dataset, device=resolve_device(device))
    _, _, _, sent_emb, sent_ids, sent_meta, _ = collect_embeddings(embedder, dataset)
    if 'subject' not in sent_meta.columns:
        raise ValueError("Sentence metadata carries no 'subject' column; there is no new reader to calibrate.")

    subjects = sent_meta['subject'].astype(str).to_numpy()
    subject = resolve_holdout(embedder.config, holdout, subjects)
    if subject is None:
        raise ValueError('Nothing to hold out: the dataset has a single subject and none was named.')

    n_words = sent_meta['n_words'].to_numpy() if 'n_words' in sent_meta.columns else None
    if n_words is None:
        _LOG.warning('Sentence metadata carries no word count; the length-stratified gallery cannot be reported.')

    report = calibration_curve(
        sent_emb,
        np.asarray(sent_ids),
        subjects,
        subject,
        n_words,
        anchor_counts=anchor_counts,
        families=families,
        draws=draws,
        ridge_alpha=ridge_alpha,
        postprocess=postprocess,
        length_tol=length_tol,
        n_boot=n_boot,
        seed=seed,
    )
    report['provenance'] = _provenance(ckpt, embedder.config, n_boot, seed, str(device))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / 'calibration.json', report, default=str)
    (out_dir / 'calibration.md').write_text(render_markdown(report), encoding='utf-8')
    return report


def _provenance(ckpt: str | Path, config: ZTEConfig, n_boot: int, seed: int, device: str) -> dict[str, Any]:
    """Records what a results table needs to place this curve next to the run it calibrates."""
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
        'n_boot': int(n_boot),
        'seed': int(seed),
        'device': device,
        'git_commit': git['commit'],
        'git_dirty': git['dirty'],
    }


def main() -> None:
    """Sweeps the anchor-calibration curve against an existing checkpoint, with no retraining."""
    args = parse_arguments()
    configure_logging(args.log_level)

    payload = CheckpointManager.load(args.ckpt, map_location='cpu')
    config = ZTEConfig.from_dict(payload['config'])

    out_dir = Path(args.out) if args.out else default_out_dir(args.ckpt)
    artifacts = (out_dir / 'calibration.json', out_dir / 'calibration.md')
    sig = signature(
        args,
        tool='calibrate',
        extra={'ckpt_sha256': checkpoint_digest(args.ckpt), 'dataset': dataset_key(config.dataset)},
        ignore=('ckpt',),
    )

    # Decided before the dataset is built, which is the multi-GB half of this command: nothing here retrains, so
    # the same weights swept the same way give the curve already on disk.
    if is_done(artifacts, sig, force=args.force):
        report = read_json(artifacts[0])
    else:
        dataset = dataset_for_config(args, config.dataset)
        report = run_calibration(
            args.ckpt,
            dataset,
            out_dir=out_dir,
            holdout=args.holdout,
            anchor_counts=tuple(int(c) for c in str(args.anchor_counts).split(',') if c.strip()),
            families=resolve_families(args.family),
            draws=args.draws,
            ridge_alpha=args.ridge_alpha,
            postprocess=args.postprocess,
            length_tol=args.length_tol,
            n_boot=args.n_boot,
            seed=args.seed,
            device=args.device,
        )
        mark_done(artifacts, sig)

    for family, verdict in (report.get('verdict') or {}).items():
        if not verdict.get('measured'):
            _LOG.info('%s: no anchor count could be scored.', family)
            continue

        _LOG.info(
            '%s at %d anchors: rank-percentile lift %.4f (CI %.4f-%.4f) against a shuffled-anchor lift of %.4f -- %s.',
            family,
            verdict['best_n_anchors'],
            verdict['lift'],
            verdict['lift_ci'][1],
            verdict['lift_ci'][2],
            verdict['shuffled_lift'],
            verdict['verdict'],
        )

    _LOG.info('Wrote %s and %s.', artifacts[0], artifacts[1])


if __name__ == '__main__':
    main()
