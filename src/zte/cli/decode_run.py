"""`zte-decode-run` -- one-command ZTE → align → decode → evaluate.

Synthetic smoke::

    zte-decode-run --synthetic --epochs 3 --align-epochs 5 --backend hash \\
        --out res/decode/demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zte.cli.sources import add_data_source_args, add_extract_dir, resolve_data_root
from zte.config import DatasetConfig, MissingConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.decode.config import DecoderConfig
from zte.decode.evaluate import evaluate_decoding
from zte.decode.pairing import build_sentence_pairs
from zte.decode.text_encoder import build_text_encoder
from zte.decode.train_align import run_alignment
from zte.decode.train_decode import run_decode_training
from zte.inference.embed import ZTEEmbedder
from zte.logging_utils import configure_logging, get_logger
from zte.training.pipeline import run_training

_LOG = get_logger('cli.decode_run')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-decode-run` command-line arguments."""
    parser = argparse.ArgumentParser(
        description='End-to-end EEG→language decode: prepare → ZTE → align → decode → eval.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ckpt', type=str, default=None, help='Existing ZTE checkpoint (skip train).')
    parser.add_argument('--align-ckpt', type=str, default=None, help='Existing aligner (skip align).')
    add_data_source_args(parser, include_bundle=True, include_synthetic=True, required=False)
    add_extract_dir(parser)

    parser.add_argument('--config', type=str, default=None, help='Optional DecoderConfig YAML.')
    parser.add_argument('--backend', choices=['auto', 'transformers', 'hash'], default='hash')
    parser.add_argument('--mode', choices=['retrieval', 'prefix_lm', 'both'], default='retrieval')
    parser.add_argument('--epochs', type=int, default=3, help='ZTE pretrain epochs (if no --ckpt).')
    parser.add_argument('--align-epochs', type=int, default=5)
    parser.add_argument('--decode-epochs', type=int, default=3, help='Prefix-LM epochs when used.')
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--out', type=str, default='res/decode/demo')
    parser.add_argument('--run-name', type=str, default='decode-run')
    parser.add_argument('--synthetic-out', type=str, default=None)
    parser.add_argument('--skip-zte-train', action='store_true')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def main() -> None:
    """Runs the full decode stack and writes a manifest."""
    args = parse_arguments()
    configure_logging(args.log_level)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = DecoderConfig.from_yaml(args.config) if args.config else DecoderConfig()
    cfg.mode = args.mode
    cfg.text.backend = args.backend  # type: ignore[assignment]
    if args.backend == 'hash':
        cfg.text.model_name = 'hash'
        cfg.text.embed_dim = args.embed_dim
    cfg.align.epochs = args.align_epochs
    cfg.align.eeg_dim = args.embed_dim
    cfg.align.text_dim = args.embed_dim
    cfg.align.proj_dim = args.embed_dim
    cfg.align.proj_hidden = max(32, args.embed_dim)
    cfg.align.device = args.device  # type: ignore[assignment]
    cfg.align.split = 'by_sentence'
    cfg.align.ckpt_dir = str(out / 'alignment')
    cfg.align.run_name = args.run_name
    cfg.generative.epochs = args.decode_epochs
    cfg.generative.backend = 'toy'
    cfg.generative.prefix_dim = args.embed_dim
    cfg.generative.device = args.device  # type: ignore[assignment]
    cfg.generative.ckpt_dir = str(out / 'generative')
    cfg.out_dir = str(out)
    cfg.run_name = args.run_name

    dataset = _prepare_dataset(args, out)
    dataset.save(out / 'bundle')

    zte_ckpt = Path(args.ckpt) if args.ckpt else out / 'checkpoints' / 'best.pt'
    if not zte_ckpt.is_file():
        if args.skip_zte_train:
            raise SystemExit(f'No ZTE checkpoint at {zte_ckpt} and --skip-zte-train set.')
        _LOG.info('[1/4] Training tiny ZTE (%d epochs) ...', args.epochs)
        zte_cfg = _zte_config(args, out)
        run_training(zte_cfg, dataset)
        zte_ckpt = Path(zte_cfg.train.ckpt_dir) / 'best.pt'
    else:
        _LOG.info('[1/4] Using existing ZTE checkpoint %s', zte_ckpt)

    align_ckpt = Path(args.align_ckpt) if args.align_ckpt else None
    if align_ckpt is None or not align_ckpt.is_file():
        _LOG.info('[2/4] Aligning EEG↔text (%d epochs) ...', cfg.align.epochs)
        arts = run_alignment(
            dataset=dataset,
            zte_ckpt=zte_ckpt,
            config=cfg,
            text_config=cfg.text,
            out_dir=cfg.align.ckpt_dir,
        )
        align_ckpt = arts.aligner_path
        align_metrics = arts.metrics
    else:
        _LOG.info('[2/4] Using existing aligner %s', align_ckpt)
        align_metrics = {}

    from zte.decode.alignment import OTCLIPAligner

    aligner = OTCLIPAligner.from_checkpoint(str(align_ckpt))
    embedder = ZTEEmbedder.from_checkpoint(zte_ckpt, dataset)
    text_encoder = build_text_encoder(cfg.text)
    eeg, text_emb, texts, meta = build_sentence_pairs(dataset, embedder, text_encoder)

    _LOG.info('[3/4] Fitting decoder (mode=%s) ...', cfg.mode)
    decode_arts = run_decode_training(
        eeg_emb=eeg,
        text_emb=text_emb,
        texts=texts,
        config=cfg,
        aligner=aligner,
        mode=cfg.mode,
        out_dir=out / 'decoder',
    )

    _LOG.info('[4/4] Evaluating ...')
    metrics = evaluate_decoding(
        eeg_emb=eeg,
        text_emb=text_emb,
        texts=texts,
        meta=meta,
        aligner=aligner,
        decoder=decode_arts.retrieval or decode_arts.generative,
        out_dir=out / 'evaluation',
        run_name=args.run_name,
    )

    cfg.to_yaml(out / 'config.yaml')
    manifest = {
        'run_name': args.run_name,
        'zte_ckpt': str(zte_ckpt),
        'align_ckpt': str(align_ckpt),
        'decoder_path': str(decode_arts.decoder_path) if decode_arts.decoder_path else None,
        'align_metrics': align_metrics,
        'decode_metrics': decode_arts.metrics,
        'evaluation': {k: v for k, v in metrics.items() if k != 'breakdown'},
        'verdict': metrics.get('verdict'),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    _LOG.info('Decode-run complete -> %s', out.resolve())
    print(json.dumps(manifest, indent=2, default=str))


def _prepare_dataset(args: argparse.Namespace, out: Path) -> ZuCoDataset:
    if args.bundle:
        return ZuCoDataset.load(args.bundle)
    if args.synthetic or (not args.root and not args.drive and args.ckpt is None):
        from zte.data.synthetic import generate_synthetic_zuco

        root = args.synthetic_out or str(out / 'synthetic_zuco')
        generate_synthetic_zuco(
            root,
            subjects=('ZAB', 'ZDM', 'ZJN'),
            tasks=('SR', 'NR'),
            n_sentences=8,
            show_progress=True,
        )
    elif args.root or args.drive:
        root = resolve_data_root(args)
    else:
        raise SystemExit('Provide --synthetic, --bundle, or --root/--drive.')
    return ZuCoDataset(
        DatasetConfig(
            root=root,
            representation='band_power',
            missing=MissingConfig(method='mask_only'),
            cache_dir=str(out / 'cache'),
        )
    ).build()


def _zte_config(args: argparse.Namespace, out: Path) -> ZTEConfig:
    cfg = ZTEConfig(run_name=f'{args.run_name}-zte')
    cfg.objective.name = 'skipgram'
    cfg.model.embed_dim = args.embed_dim
    cfg.model.hidden_dim = 96
    cfg.model.n_layers = 2
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = 16
    cfg.train.device = args.device
    cfg.train.precision = 'fp32'
    cfg.train.split = 'by_sentence'
    cfg.train.ckpt_dir = str(out / 'checkpoints')
    cfg.train.log_every = 5
    cfg.train.tensorboard = False
    return cfg


if __name__ == '__main__':
    main()
