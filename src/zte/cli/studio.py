"""`zte-studio` -- decode a handful of readings and write the interactive page that shows what the decoder did."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zte.cli.decode import split_indices
from zte.cli.support.sources import dataset_for_config
from zte.config import ZTEConfig
from zte.device import resolve_device
from zte.evaluation.interactive.studio import build_studio, studio_html
from zte.inference.decode import ZTEDecoder
from zte.logging_utils import configure_logging, get_logger
from zte.training.checkpoint import CheckpointManager

_LOG = get_logger('cli.studio')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-studio` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Decode a few held-out readings and write the interactive studio: the scalp field, the target '
        "sentence with the pointer walking it, the text coming back token by token, and the decoder's own "
        'per-step behaviour. An inspection tool -- the verdict lives in `zte-decode`.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ckpt', type=str, required=True, help='Decoder checkpoint (best.pt/last.pt).')
    parser.add_argument('--root', type=str, default=None, help='ZuCo data root; omit to reuse the run config.')
    parser.add_argument('--extract-dir', type=str, default=None, dest='extract_dir')
    parser.add_argument('--bundle', type=str, default=None, help='Prepared bundle directory.')
    parser.add_argument('--synthetic', action='store_true', help='Build a synthetic dataset instead of reading ZuCo.')
    parser.add_argument('--drive', action='store_true', help='Resolve the data root on a mounted Drive.')
    parser.add_argument(
        '--split',
        default='test',
        choices=['train', 'val', 'test'],
        help='Which split to draw readings from. The honest one is `test`.',
    )
    parser.add_argument(
        '--rows',
        type=int,
        default=8,
        help='How many readings to decode. Each carries its own trace and scalp cube, so the page grows with it.',
    )
    parser.add_argument(
        '--pick',
        type=str,
        default=None,
        help='Comma-separated reading indices to include instead of the first --rows.',
    )
    parser.add_argument(
        '--controls',
        type=str,
        default='null_prefix,length_only,mismatch',
        help='Brain-independent conditions decoded beside the hypothesis, comma-separated.',
    )
    parser.add_argument('--max-new-tokens', type=int, default=None, dest='max_new_tokens')
    parser.add_argument('--montage', type=str, default=None, help='Montage CSV (channel,x,y,z) for the scalp map.')
    parser.add_argument('--out', type=Path, default=None, help='Output .html path.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--run-name', type=str, default=None, dest='run_name')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    return parser.parse_args()


def main() -> None:
    """Decodes the chosen readings and writes the studio page."""
    args = parse_arguments()
    configure_logging(args.log_level)

    payload = CheckpointManager.load(args.ckpt, map_location='cpu')
    config = ZTEConfig.from_dict(payload['config'])
    dataset = dataset_for_config(args, config.dataset)
    decoder = ZTEDecoder.from_checkpoint(args.ckpt, dataset, device=resolve_device(args.device))

    readings = decoder.conditioning(dataset, split_indices(dataset, config, args.split))
    if not len(readings):
        _LOG.error('The %r split holds no readings; nothing to decode.', args.split)
        return

    rows = _rows(args, len(readings))
    _LOG.info('Decoding %d of %d %r readings with a full per-step trace ...', len(rows), len(readings), args.split)
    studio = build_studio(
        decoder,
        dataset,
        readings,
        rows=rows,
        controls=tuple(c.strip() for c in args.controls.split(',') if c.strip()),
        max_new_tokens=args.max_new_tokens,
        run_name=args.run_name or config.run_name,
        montage_csv=args.montage or config.dataset.montage_csv,
    )

    out = Path(args.out) if args.out else Path(args.ckpt).resolve().parent.parent / 'evaluation' / 'STUDIO.html'
    studio_html(studio, out, run_name=args.run_name or config.run_name)
    _LOG.info('Open it in any browser -- everything is inlined, no server and no network: %s', out)


def _rows(args: argparse.Namespace, available: int) -> np.ndarray:
    """Resolves which readings to decode: an explicit pick, or an even spread across the split.

    An even spread rather than the first `n`: readings arrive grouped by subject, so the head of the split is one
    person reading consecutive sentences, which is the least informative page this tool could open with.
    """
    if args.pick:
        picked = [int(p) for p in args.pick.split(',') if p.strip()]
        return np.asarray([p for p in picked if 0 <= p < available], dtype=int)

    count = max(min(int(args.rows), available), 1)
    return np.unique(np.linspace(0, available - 1, count).astype(int))


if __name__ == '__main__':  # pragma: no cover - console-script entry point
    main()
