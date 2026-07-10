"""`zte-align` -- train an EEG-OT-CLIP aligner on ZTE embeddings + text.

Examples::

    zte-align --ckpt res/checkpoints/best.pt --bundle res/bundle \\
        --config experiments/decode/align_otclip_loso.yaml --out res/decode/alignment

    zte-align --synthetic --epochs 5 --backend hash --out res/decode/alignment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zte.cli.sources import add_data_source_args, add_extract_dir, resolve_data_root
from zte.config import DatasetConfig, MissingConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.decode.config import AlignConfig, DecoderConfig, TextEncoderConfig
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.align')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-align` command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train EEG-OT-CLIP alignment (InfoNCE + Sinkhorn OT).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default=None,
        help='ZTE checkpoint (best.pt). Required unless --synthetic trains one.',
    )
    add_data_source_args(parser, include_bundle=True, include_synthetic=True, required=False)
    add_extract_dir(parser)

    parser.add_argument('--config', type=str, default=None, help='DecoderConfig / Align YAML.')
    parser.add_argument('--backend', choices=['auto', 'transformers', 'hash'], default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--level', choices=['word', 'sentence'], default=None)
    parser.add_argument(
        '--split', choices=['random', 'by_sentence', 'by_subject_loso', 'by_task'], default=None
    )
    parser.add_argument('--loso-holdout', type=str, default=None)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default=None)
    parser.add_argument('--out', type=str, default='res/decode/alignment')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--synthetic-out', type=str, default='res/data/synthetic_zuco')
    parser.add_argument(
        '--zte-epochs',
        type=int,
        default=3,
        help='When --synthetic and no --ckpt, quick-train ZTE for this many epochs.',
    )
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def main() -> None:
    """Runs EEG-OT-CLIP alignment from the command line."""
    args = parse_arguments()
    configure_logging(args.log_level)

    decoder_cfg = DecoderConfig.from_yaml(args.config) if args.config else DecoderConfig()
    align_cfg = decoder_cfg.align
    text_cfg = decoder_cfg.text

    _apply_overrides(args, align_cfg, text_cfg, decoder_cfg)
    align_cfg.ckpt_dir = args.out
    if args.run_name:
        align_cfg.run_name = args.run_name
        decoder_cfg.run_name = args.run_name

    dataset, zte_ckpt = _load_dataset_and_ckpt(args, decoder_cfg)
    from zte.decode.train_align import run_alignment

    arts = run_alignment(
        dataset=dataset,
        zte_ckpt=zte_ckpt,
        config=decoder_cfg,
        text_config=text_cfg,
        out_dir=args.out,
    )
    summary = {
        'aligner_path': str(arts.aligner_path),
        'metrics': arts.metrics,
        'history_len': len(arts.history),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / 'align_summary.json').write_text(
        json.dumps(summary, indent=2, default=str), encoding='utf-8'
    )
    decoder_cfg.to_yaml(Path(args.out) / 'config.yaml')
    _LOG.info('Alignment done -> %s', arts.aligner_path)
    print(json.dumps(summary, indent=2, default=str))


def _apply_overrides(
    args: argparse.Namespace,
    align_cfg: AlignConfig,
    text_cfg: TextEncoderConfig,
    decoder_cfg: DecoderConfig,
) -> None:
    if args.backend is not None:
        text_cfg.backend = args.backend
        if args.backend == 'hash':
            text_cfg.model_name = 'hash'
    if args.epochs is not None:
        align_cfg.epochs = args.epochs
    if args.batch_size is not None:
        align_cfg.batch_size = args.batch_size
    if args.lr is not None:
        align_cfg.lr = args.lr
    if args.level is not None:
        align_cfg.level = args.level
    if args.split is not None:
        align_cfg.split = args.split
    if args.loso_holdout is not None:
        align_cfg.loso_holdout_subject = args.loso_holdout
    if args.device is not None:
        align_cfg.device = args.device
        text_cfg.device = args.device
    decoder_cfg.align = align_cfg
    decoder_cfg.text = text_cfg
    decoder_cfg.out_dir = args.out


def _load_dataset_and_ckpt(
    args: argparse.Namespace, decoder_cfg: DecoderConfig
) -> tuple[ZuCoDataset, Path]:
    """Resolves dataset + ZTE checkpoint, optionally quick-training on synthetic."""
    if args.bundle:
        dataset = ZuCoDataset.load(args.bundle)
    elif args.synthetic:
        from zte.data.synthetic import generate_synthetic_zuco

        generate_synthetic_zuco(args.synthetic_out, show_progress=True)
        ds_cfg = DatasetConfig(
            root=args.synthetic_out,
            representation='band_power',
            missing=MissingConfig(method='mask_only'),
            cache_dir=str(Path(args.out) / 'cache'),
        )
        dataset = ZuCoDataset(ds_cfg).build()
    elif args.root or args.drive:
        root = resolve_data_root(args)
        ds_cfg = DatasetConfig(
            root=root,
            representation='band_power',
            missing=MissingConfig(method='mask_only'),
        )
        dataset = ZuCoDataset(ds_cfg).build()
    else:
        raise SystemExit('Provide --bundle, --root/--drive, or --synthetic.')

    if args.ckpt:
        return dataset, Path(args.ckpt)

    if not args.synthetic:
        raise SystemExit('--ckpt is required unless --synthetic (quick ZTE train).')

    _LOG.info('No --ckpt; quick-training ZTE for %d epochs on synthetic data.', args.zte_epochs)
    from zte.training.pipeline import run_training

    zte_cfg = ZTEConfig(run_name='align-zte-pretrain')
    zte_cfg.model.embed_dim = decoder_cfg.align.eeg_dim
    zte_cfg.model.hidden_dim = 96
    zte_cfg.model.n_layers = 2
    zte_cfg.train.epochs = args.zte_epochs
    zte_cfg.train.batch_size = 16
    zte_cfg.train.device = args.device or 'cpu'
    zte_cfg.train.precision = 'fp32'
    zte_cfg.train.split = 'by_sentence'
    zte_cfg.train.ckpt_dir = str(Path(args.out) / 'zte_checkpoints')
    run_training(zte_cfg, dataset)
    return dataset, Path(zte_cfg.train.ckpt_dir) / 'best.pt'


if __name__ == '__main__':
    main()
