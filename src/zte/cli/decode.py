"""`zte-decode` -- retrieval or prefix-LM decoding from aligned EEG.

Examples::

    zte-decode --align-ckpt res/decode/alignment/best.pt --ckpt res/checkpoints/best.pt \\
        --bundle res/bundle --mode retrieval --out res/decode/predictions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zte.cli.sources import add_data_source_args, add_extract_dir, resolve_data_root
from zte.config import DatasetConfig, MissingConfig
from zte.data.dataset import ZuCoDataset
from zte.decode.alignment import OTCLIPAligner
from zte.decode.config import DecoderConfig
from zte.decode.pairing import build_sentence_pairs, build_word_pairs
from zte.decode.text_encoder import build_text_encoder
from zte.decode.train_decode import run_decode_training
from zte.inference.embed import ZTEEmbedder
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.decode')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-decode` command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Decode language from aligned EEG (retrieval or prefix-LM).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--align-ckpt', type=str, required=True, help='OT-CLIP aligner checkpoint.')
    parser.add_argument('--ckpt', type=str, required=True, help='ZTE checkpoint.')
    add_data_source_args(parser, include_bundle=True, include_synthetic=True, required=False)
    add_extract_dir(parser)

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--mode', choices=['retrieval', 'prefix_lm', 'both'], default=None)
    parser.add_argument('--backend', choices=['auto', 'transformers', 'hash', 'toy'], default=None)
    parser.add_argument('--epochs', type=int, default=None, help='Prefix-LM epochs.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default=None)
    parser.add_argument('--out', type=str, default='res/decode/predictions')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--synthetic-out', type=str, default='res/data/synthetic_zuco')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def main() -> None:
    """Fits a decoder and writes predictions."""
    args = parse_arguments()
    configure_logging(args.log_level)

    cfg = DecoderConfig.from_yaml(args.config) if args.config else DecoderConfig()
    if args.mode is not None:
        cfg.mode = args.mode
    if args.epochs is not None:
        cfg.generative.epochs = args.epochs
    if args.device is not None:
        cfg.align.device = args.device
        cfg.generative.device = args.device
        cfg.text.device = args.device
    if args.backend == 'toy':
        cfg.generative.backend = 'toy'
    elif args.backend in {'auto', 'transformers', 'hash'}:
        cfg.text.backend = args.backend  # type: ignore[assignment]
        if args.backend == 'hash':
            cfg.text.model_name = 'hash'
    if args.run_name:
        cfg.run_name = args.run_name
    cfg.out_dir = args.out

    dataset = _load_dataset(args)
    embedder = ZTEEmbedder.from_checkpoint(args.ckpt, dataset)
    aligner = OTCLIPAligner.from_checkpoint(args.align_ckpt)
    text_encoder = build_text_encoder(cfg.text)

    level = aligner.config.level
    pair_fn = build_word_pairs if level == 'word' else build_sentence_pairs
    eeg, text_emb, texts, meta = pair_fn(dataset, embedder, text_encoder)

    arts = run_decode_training(
        eeg_emb=eeg,
        text_emb=text_emb,
        texts=texts,
        config=cfg,
        aligner=aligner,
        mode=cfg.mode,
        out_dir=args.out,
    )

    # Write predictions for the fitted decoder.
    hyps: list[str]
    if arts.retrieval is not None:
        with torch.no_grad():
            eeg_z = aligner.encode_eeg(torch.from_numpy(eeg)).cpu().numpy()
        hyps = arts.retrieval.decode(eeg_z, k=1)
    elif arts.generative is not None:
        with torch.no_grad():
            eeg_z = aligner.encode_eeg(torch.from_numpy(eeg)).cpu().numpy()
        hyps = arts.generative.generate(eeg_z)
    else:
        hyps = [''] * len(texts)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / 'predictions.jsonl'
    with pred_path.open('w', encoding='utf-8') as fh:
        for hyp, ref, (_, row) in zip(hyps, texts, meta.iterrows(), strict=False):
            fh.write(
                json.dumps(
                    {
                        'hyp': hyp,
                        'ref': ref,
                        'subject': row.get('subject'),
                        'task': row.get('task'),
                    },
                    default=str,
                )
                + '\n'
            )
    summary = {
        'mode': arts.mode,
        'decoder_path': str(arts.decoder_path) if arts.decoder_path else None,
        'metrics': arts.metrics,
        'n_predictions': len(hyps),
        'predictions': str(pred_path),
    }
    (out / 'decode_summary.json').write_text(
        json.dumps(summary, indent=2, default=str), encoding='utf-8'
    )
    cfg.to_yaml(out / 'config.yaml')
    _LOG.info('Decode done -> %s', out)
    print(json.dumps(summary, indent=2, default=str))


def _load_dataset(args: argparse.Namespace) -> ZuCoDataset:
    if args.bundle:
        return ZuCoDataset.load(args.bundle)
    if args.synthetic:
        from zte.data.synthetic import generate_synthetic_zuco

        generate_synthetic_zuco(args.synthetic_out, show_progress=False)
        root = args.synthetic_out
    elif args.root or args.drive:
        root = resolve_data_root(args)
    else:
        raise SystemExit('Provide --bundle, --root/--drive, or --synthetic.')
    return ZuCoDataset(
        DatasetConfig(root=root, representation='band_power', missing=MissingConfig(method='mask_only'))
    ).build()


if __name__ == '__main__':
    main()
