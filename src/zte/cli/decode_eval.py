"""`zte-decode-eval` -- full EEG→language evaluation report.

Examples::

    zte-decode-eval --align-ckpt res/decode/alignment/best.pt \\
        --ckpt res/checkpoints/best.pt --bundle res/bundle \\
        --out res/decode/eval
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
from zte.decode.decoders import PrefixLanguageDecoder, RetrievalDecoder
from zte.decode.evaluate import evaluate_decoding
from zte.decode.pairing import build_sentence_pairs, build_word_pairs
from zte.decode.text_encoder import build_text_encoder
from zte.inference.embed import ZTEEmbedder
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.decode_eval')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-decode-eval` command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate EEG→language decoding (retrieval + generative metrics).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--align-ckpt', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True, help='ZTE checkpoint.')
    parser.add_argument(
        '--decoder-ckpt',
        type=str,
        default=None,
        help='Optional retrieval bank / prefix-LM checkpoint.',
    )
    add_data_source_args(parser, include_bundle=True, include_synthetic=True, required=False)
    add_extract_dir(parser)

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--backend', choices=['auto', 'transformers', 'hash'], default='hash')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--out', type=str, default='res/decode/eval')
    parser.add_argument('--run-name', type=str, default='decode-eval')
    parser.add_argument('--synthetic-out', type=str, default='res/data/synthetic_zuco')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def main() -> None:
    """Runs the full decode evaluation suite."""
    args = parse_arguments()
    configure_logging(args.log_level)

    cfg = DecoderConfig.from_yaml(args.config) if args.config else DecoderConfig()
    cfg.text.backend = args.backend  # type: ignore[assignment]
    if args.backend == 'hash':
        cfg.text.model_name = 'hash'
    cfg.text.device = args.device  # type: ignore[assignment]

    dataset = _load_dataset(args)
    embedder = ZTEEmbedder.from_checkpoint(args.ckpt, dataset)
    aligner = OTCLIPAligner.from_checkpoint(args.align_ckpt)
    text_encoder = build_text_encoder(cfg.text)

    level = aligner.config.level
    pair_fn = build_word_pairs if level == 'word' else build_sentence_pairs
    eeg, text_emb, texts, meta = pair_fn(dataset, embedder, text_encoder)

    decoder = _maybe_load_decoder(args.decoder_ckpt, texts, text_emb, aligner)

    metrics = evaluate_decoding(
        eeg_emb=eeg,
        text_emb=text_emb,
        texts=texts,
        meta=meta,
        aligner=aligner,
        decoder=decoder,
        out_dir=args.out,
        run_name=args.run_name,
    )
    _LOG.info('Eval complete -> %s', Path(args.out).resolve())
    print(json.dumps({k: v for k, v in metrics.items() if k != 'breakdown'}, indent=2, default=str))


def _maybe_load_decoder(
    path: str | None,
    texts: list[str],
    text_emb,
    aligner: OTCLIPAligner,
) -> RetrievalDecoder | PrefixLanguageDecoder | None:
    if path is None:
        return None
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if 'text_bank_emb' in payload or 'texts' in payload:
        try:
            return RetrievalDecoder.from_checkpoint(path, texts=texts, text_bank_emb=text_emb)
        except ValueError:
            pass
    if 'decoder' in payload or 'generative' in payload:
        from dataclasses import fields

        from zte.decode.config import GenerativeConfig

        raw = payload.get('generative_config') or payload.get('config') or {}
        allowed = {f.name for f in fields(GenerativeConfig)}
        gen_cfg = GenerativeConfig(**{k: v for k, v in dict(raw).items() if k in allowed})
        gen_cfg.backend = 'toy'
        decoder = PrefixLanguageDecoder(gen_cfg, texts=texts)
        state = payload.get('decoder') or payload.get('generative')
        if state is not None:
            decoder.load_state_dict(state, strict=False)
        decoder.eval()
        return decoder
    return RetrievalDecoder(
        text_emb, texts, aligner=aligner, bank_already_aligned=False
    )


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
        DatasetConfig(
            root=root, representation='band_power', missing=MissingConfig(method='mask_only')
        )
    ).build()


if __name__ == '__main__':
    main()
