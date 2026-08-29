"""`zte-lens` -- one reading through a trained checkpoint: saliency, neighbours, and the decode trace."""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zte.cli.support.done import add_force_argument, checkpoint_digest, is_done, mark_done, signature
from zte.cli.support.sources import add_data_source_args, add_extract_dir, dataset_for_config, dataset_key
from zte.config import ZTEConfig
from zte.logging_utils import configure_logging, get_logger

if TYPE_CHECKING:
    from zte.data.dataset import ZuCoDataset
    from zte.inference.embed import ZTEEmbedder
    from zte.lens.saliency import Reading

_LOG = get_logger('cli.lens')


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Defines and parses the `zte-lens` command-line arguments.

    Args:
        argv (list[str] | None, optional): Arguments to parse instead of `sys.argv`. Defaults to None.

    Returns:
        argparse.Namespace: The parsed argument namespace, with the subcommand under `command`.
    """
    parser = argparse.ArgumentParser(
        description='The lens: pick one reading (one subject reading one sentence), run it through a trained '
        'checkpoint, and see what the model did. `encode` inspects the thought embedding; `decode` also traces '
        'the generated text. Inspection only -- no lens output is a metric.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    sub = parser.add_subparsers(dest='command', required=True)

    encode = sub.add_parser(
        'encode',
        help='Embed one reading: word and channel saliency, and its neighbourhood in the gallery.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_reading_args(encode)
    encode.add_argument('--top-k', type=int, default=10, dest='top_k', help='Gallery neighbours to report.')
    _add_output_args(encode)

    decode = sub.add_parser(
        'decode',
        help='Trace one reading through a decoder checkpoint: generation, slot influence, the null-prefix twin.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_reading_args(decode)
    decode.add_argument(
        '--max-new-tokens',
        type=int,
        default=48,
        dest='max_new_tokens',
        help='Free-running decode cap for the trace.',
    )
    _add_output_args(decode)

    return parser.parse_args(argv)


def _add_reading_args(parser: argparse.ArgumentParser) -> None:
    """Adds the checkpoint, data-source and reading-selection flags shared by both subcommands."""
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint (best.pt/last.pt) to inspect through.')
    parser.add_argument('--out', type=Path, required=True, help='Directory the lens directory is written under.')
    add_data_source_args(parser, include_bundle=True, include_synthetic=True)
    add_extract_dir(parser)
    parser.add_argument(
        '--subject',
        type=str,
        default=None,
        help="Subject whose reading is inspected. Defaults to the checkpoint's LOSO holdout.",
    )
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help="Which of the subject's readings to inspect, in the built dataset's deterministic order.",
    )
    parser.add_argument(
        '--contains',
        type=str,
        default=None,
        help='Count only readings whose sentence text contains this substring (case-insensitive).',
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Adds the device, rendering and temporal-profile flags shared by both subcommands."""
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--html', action='store_true', help='Also render LENS.html beside lens.json.')
    parser.add_argument(
        '--temporal',
        action='store_true',
        help='Also write temporal.json/temporal.md: the occlusion latency profile over the raw window. '
        'Raw-input checkpoints only.',
    )
    parser.add_argument(
        '--temporal-bins',
        type=int,
        default=14,
        dest='temporal_bins',
        help='Contiguous time bins the raw window is split into for the temporal profile.',
    )
    parser.add_argument(
        '--temporal-sentences',
        type=int,
        default=12,
        dest='temporal_sentences',
        help="Readings the temporal profile aggregates over, from --index onwards through the subject's readings.",
    )
    add_force_argument(parser)


def resolve_subject(config: ZTEConfig, requested: str | None) -> str:
    """Returns the subject to inspect: `--subject` when given, else the checkpoint's LOSO holdout.

    Args:
        config (ZTEConfig): The checkpoint's run config.
        requested (str | None): The `--subject` flag, or None to fall back to the training holdout.

    Returns:
        str: The subject code.

    Raises:
        SystemExit: If no subject was requested and the checkpoint names no LOSO holdout.
    """
    if requested:
        return requested

    holdout = config.train.loso_holdout_subject
    if holdout:
        return str(holdout)

    raise SystemExit(
        'The checkpoint names no LOSO holdout (train.loso_holdout_subject is unset), so there is no default '
        'reading to inspect -- pass --subject explicitly.'
    )


def write_lens_json(report: dict[str, Any], out: Path, run_name: str, subject: str, index: int) -> Path:
    """Writes one lens report as `<out>/<run_name>_<subject>_<index>/lens.json`.

    Args:
        report (dict[str, Any]): The lens report dict, carrying the mandatory disclaimer field.
        out (Path): Directory the lens directory is created under.
        run_name (str): The checkpoint's `run_name`.
        subject (str): The inspected subject.
        index (int): The requested `--index` (not the positional row), so the directory names the request.

    Returns:
        Path: The written `lens.json` path.

    Raises:
        SystemExit: If the report does not carry the disclaimer -- an artifact without it could be quoted as a result.
    """
    from zte.lens.saliency import DISCLAIMER

    if report.get('disclaimer') != DISCLAIMER:
        raise SystemExit(
            'Refusing to write a lens report without its disclaimer -- the lens is an inspection surface, and '
            'every artifact must say so.'
        )

    target = Path(out) / f'{run_name}_{subject}_{index}'
    target.mkdir(parents=True, exist_ok=True)
    path = target / 'lens.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    return path


def temporal_readings(
    dataset: ZuCoDataset, subject: str, index: int, contains: str | None, limit: int
) -> list[Reading]:
    """Collects up to `limit` of the subject's readings from `--index` onwards, so the profile is not one sentence.

    Args:
        dataset (ZuCoDataset): The built dataset.
        subject (str): Subject whose readings are profiled.
        index (int): First reading to take, in the dataset's deterministic order.
        contains (str | None): The `--contains` filter, applied exactly as the single-reading selection applies it.
        limit (int): Most readings to take; the subject may have fewer.

    Returns:
        list[Reading]: The selected readings, in dataset order.
    """
    from zte.lens.saliency import select_reading

    readings: list[Reading] = []
    for offset in range(max(int(limit), 1)):
        try:
            readings.append(select_reading(dataset, subject, index + offset, contains))
        except ValueError:
            break

    return readings


def write_temporal(
    embedder: ZTEEmbedder,
    dataset: ZuCoDataset,
    subject: str,
    args: argparse.Namespace,
    target: Path,
) -> Path | None:
    """Profiles when in the raw window the embedding is built, writing `temporal.json` and `temporal.md`.

    Args:
        embedder (ZTEEmbedder): The loaded encoder.
        dataset (ZuCoDataset): The built dataset the readings live in.
        subject (str): The subject whose readings are profiled.
        args (argparse.Namespace): The parsed arguments, for the reading selection and the profile's knobs.
        target (Path): The lens directory the artifacts are written into.

    Returns:
        Path | None: The written `temporal.json`, or `None` when this checkpoint has no time axis to profile.

    Raises:
        SystemExit: If the subject has no readings, or the profile comes back without its disclaimer.
    """
    from zte.lens.saliency import DISCLAIMER
    from zte.lens.temporal import render_markdown, temporal_saliency

    readings = temporal_readings(dataset, subject, int(args.index), args.contains, int(args.temporal_sentences))
    if not readings:
        raise SystemExit(f'Subject {subject!r} has no reading to profile, so there is no temporal profile to write.')

    profile = temporal_saliency(
        embedder,
        dataset,
        readings,
        n_bins=int(args.temporal_bins),
        ckpt_path=Path(args.ckpt),
    )
    if profile is None:
        _LOG.warning('Temporal profile not written: this checkpoint reads no raw time axis.')
        return None

    if profile.get('disclaimer') != DISCLAIMER:
        raise SystemExit(
            'Refusing to write a temporal profile without its disclaimer -- the lens is an inspection surface, '
            'and every artifact must say so.'
        )

    path = target / 'temporal.json'
    path.write_text(json.dumps(profile, indent=2), encoding='utf-8')
    (target / 'temporal.md').write_text(render_markdown(profile), encoding='utf-8')

    return path


def render_page(json_path: Path) -> Path:
    """Renders `LENS.html` beside a written `lens.json`, through the deferred page module.

    Args:
        json_path (Path): The written `lens.json`.

    Returns:
        Path: The written HTML path.

    Raises:
        SystemExit: If the lens page renderer is not available in this build.
    """
    try:
        from zte.lens.page import build_lens_page
    except ImportError as exc:
        raise SystemExit(f'The lens page renderer is not available in this build: {exc}') from exc

    return build_lens_page(json_path, json_path.parent / 'LENS.html')


def run_lens(args: argparse.Namespace, mode: str) -> Path:
    """Builds one lens report for the selected reading and writes its artifacts.

    Args:
        args (argparse.Namespace): The parsed `encode` or `decode` arguments.
        mode (str): `'encode'` or `'decode'`.

    Returns:
        Path: The written `lens.json` path.

    Raises:
        SystemExit: If a decode is requested from a checkpoint with no trained bridge, the subject cannot be
            resolved or selected, or the lens report builder is not available in this build.
    """
    from zte.training.checkpoint import CheckpointManager

    payload = CheckpointManager.load(args.ckpt, map_location='cpu')
    config = ZTEConfig.from_dict(payload['config'])

    # Refused before any data is built: an encoder-only checkpoint has no bridge, so there is nothing to trace.
    if mode == 'decode':
        extra: dict[str, Any] = payload.get('extra') or {}
        if not extra.get('decoder_state'):
            raise SystemExit(
                f'{args.ckpt} carries no decoder_state -- it is an encoder-only checkpoint with no trained '
                'bridge, so there is no generation to trace. Use `zte-lens encode` on it, or train with '
                'train.mode=decoder first.'
            )

    subject = resolve_subject(config, args.subject)
    holdout = config.train.loso_holdout_subject
    if holdout and subject != str(holdout):
        _LOG.warning('Subject %s is a TRAINING brain for this checkpoint (holdout: %s).', subject, holdout)
    elif not holdout:
        _LOG.warning('The checkpoint names no LOSO holdout, so every subject here is a training brain.')

    # Decided before the dataset is built and before a single occlusion pass runs, which together are the whole
    # cost of this command. The lens recomputes nothing about the model, so the same weights inspected the same
    # way give the artifacts already on disk.
    target = Path(args.out) / f'{config.run_name}_{subject}_{int(args.index)}'
    artifacts = [target / 'lens.json']
    if args.temporal:
        artifacts += [target / 'temporal.json', target / 'temporal.md']
    if args.html:
        artifacts.append(target / 'LENS.html')

    sig = signature(
        args,
        tool=f'lens-{mode}',
        extra={'ckpt_sha256': checkpoint_digest(args.ckpt), 'dataset': dataset_key(config.dataset)},
        ignore=('ckpt',),
    )
    if is_done(artifacts, sig, force=args.force):
        _LOG.info('Lens already built at %s from these weights and flags.', artifacts[0])

        return artifacts[0]

    dataset = dataset_for_config(args, config.dataset)

    try:
        from zte.lens.saliency import lens_report, select_reading
    except ImportError as exc:
        raise SystemExit(f'The lens report builder is not available in this build: {exc}') from exc

    try:
        reading = select_reading(dataset, subject, args.index, args.contains)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    from zte.device import resolve_device
    from zte.inference.embed import ZTEEmbedder

    device = resolve_device(args.device)
    decoder = None
    if mode == 'decode':
        from zte.inference.decode import ZTEDecoder

        decoder = ZTEDecoder.from_checkpoint(args.ckpt, dataset, device=device)
        embedder = ZTEEmbedder(decoder.model, decoder.config, decoder.device)
    else:
        embedder = ZTEEmbedder.from_checkpoint(args.ckpt, dataset, device=device)

    report = lens_report(
        embedder,
        dataset,
        reading,
        decoder=decoder,
        ckpt_path=Path(args.ckpt),
        top_k=int(getattr(args, 'top_k', 10)),
        max_new_tokens=int(args.max_new_tokens) if mode == 'decode' else None,
    )

    json_path = write_lens_json(report, Path(args.out), config.run_name, subject, int(args.index))
    _LOG.info('Lens written to %s.', json_path)

    if args.temporal:
        temporal_path = write_temporal(embedder, dataset, subject, args, json_path.parent)
        if temporal_path is not None:
            _LOG.info('Temporal profile written to %s.', temporal_path)

    if args.html:
        html_path = render_page(json_path)
        _LOG.info('Page written to %s.', html_path)

    mark_done(artifacts, sig)

    return json_path


def main() -> None:
    """Dispatches the `zte-lens` subcommands."""
    args = parse_arguments()
    configure_logging(args.log_level)
    run_lens(args, args.command)


if __name__ == '__main__':
    main()
