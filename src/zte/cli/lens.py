"""`zte-lens` -- one reading through a trained checkpoint: saliency, neighbours, and the decode trace."""

import argparse
import json
from pathlib import Path
from typing import Any

from zte.cli.support.sources import add_data_source_args, add_extract_dir, dataset_for_config
from zte.config import ZTEConfig
from zte.logging_utils import configure_logging, get_logger

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
    """Adds the device and rendering flags shared by both subcommands."""
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--html', action='store_true', help='Also render LENS.html beside lens.json.')


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

    if args.html:
        html_path = render_page(json_path)
        _LOG.info('Page written to %s.', html_path)

    return json_path


def main() -> None:
    """Dispatches the `zte-lens` subcommands."""
    args = parse_arguments()
    configure_logging(args.log_level)
    run_lens(args, args.command)


if __name__ == '__main__':
    main()
