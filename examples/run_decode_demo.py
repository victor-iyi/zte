"""End-to-end EEG→language decode demo on synthetic ZuCo.

1. Generate synthetic ZuCo
2. Quick-train a tiny ZTE encoder
3. Align with a hash text encoder (EEG-OT-CLIP)
4. Fit a retrieval decoder and evaluate
5. Print metrics; write artifacts under ``res/decode_demo/``

Example::

    $ uv run python examples/run_decode_demo.py --epochs 3 --align-epochs 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zte.config import DatasetConfig, MissingConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.synthetic import generate_synthetic_zuco
from zte.decode.alignment import OTCLIPAligner
from zte.decode.config import AlignConfig, DecoderConfig, TextEncoderConfig
from zte.decode.evaluate import evaluate_decoding
from zte.decode.pairing import build_sentence_pairs
from zte.decode.text_encoder import build_text_encoder
from zte.decode.train_align import run_alignment
from zte.decode.train_decode import run_decode_training
from zte.inference.embed import ZTEEmbedder
from zte.logging_utils import configure_logging, get_logger
from zte.training.pipeline import run_training

_LOG = get_logger('demo.decode')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the decode demo CLI arguments."""
    parser = argparse.ArgumentParser(description='Synthetic EEG→language decode demo.')
    parser.add_argument('--epochs', type=int, default=3, help='ZTE pretrain epochs.')
    parser.add_argument('--align-epochs', type=int, default=5)
    parser.add_argument('--subjects', type=str, default='ZAB,ZDM,ZJN')
    parser.add_argument('--sentences', type=int, default=8)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--workdir', type=str, default='res/decode_demo')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='cpu')
    return parser.parse_args()


def main() -> None:
    """Runs the full synthetic decode demo."""
    args = parse_arguments()
    configure_logging('INFO')
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    dim = args.embed_dim

    # 1) Synthesise + build.
    data_root = work / 'synthetic_zuco'
    generate_synthetic_zuco(
        data_root,
        subjects=tuple(args.subjects.split(',')),
        tasks=('SR', 'NR'),
        n_sentences=args.sentences,
        show_progress=True,
    )
    dataset = ZuCoDataset(
        DatasetConfig(
            root=str(data_root),
            representation='band_power',
            missing=MissingConfig(method='mask_only'),
            cache_dir=str(work / 'cache'),
        )
    ).build()
    _LOG.info('Dataset: %r', dataset)

    # 2) Quick ZTE pretrain.
    zte_cfg = ZTEConfig(run_name='decode-demo-zte')
    zte_cfg.objective.name = 'skipgram'
    zte_cfg.model.embed_dim = dim
    zte_cfg.model.hidden_dim = 96
    zte_cfg.model.n_layers = 2
    zte_cfg.train.epochs = args.epochs
    zte_cfg.train.batch_size = 16
    zte_cfg.train.device = args.device
    zte_cfg.train.precision = 'fp32'
    zte_cfg.train.split = 'by_sentence'
    zte_cfg.train.ckpt_dir = str(work / 'checkpoints')
    zte_cfg.train.tensorboard = False
    run_training(zte_cfg, dataset)
    zte_ckpt = Path(zte_cfg.train.ckpt_dir) / 'best.pt'

    # 3) Align with hash text encoder.
    text_cfg = TextEncoderConfig(
        model_name='hash',
        embed_dim=dim,
        backend='hash',
        normalize=True,
        device=args.device,  # type: ignore[arg-type]
        cache_dir=str(work / 'text_cache'),
    )
    align_cfg = AlignConfig(
        eeg_dim=dim,
        text_dim=dim,
        proj_dim=dim,
        proj_hidden=max(32, dim),
        epochs=args.align_epochs,
        batch_size=16,
        lr=1e-3,
        split='by_sentence',
        val_fraction=0.2,
        device=args.device,  # type: ignore[arg-type]
        ckpt_dir=str(work / 'alignment'),
        run_name='demo-align',
        level='sentence',
    )
    decoder_cfg = DecoderConfig(
        text=text_cfg,
        align=align_cfg,
        mode='retrieval',
        out_dir=str(work),
        run_name='decode-demo',
    )
    align_arts = run_alignment(
        dataset=dataset,
        zte_ckpt=zte_ckpt,
        config=decoder_cfg,
        text_config=text_cfg,
        out_dir=align_cfg.ckpt_dir,
    )
    aligner = OTCLIPAligner.from_checkpoint(str(align_arts.aligner_path))

    # 4) Retrieval decode + evaluate.
    embedder = ZTEEmbedder.from_checkpoint(zte_ckpt, dataset)
    text_encoder = build_text_encoder(text_cfg)
    eeg, text_emb, texts, meta = build_sentence_pairs(dataset, embedder, text_encoder)
    decode_arts = run_decode_training(
        eeg_emb=eeg,
        text_emb=text_emb,
        texts=texts,
        config=decoder_cfg,
        aligner=aligner,
        mode='retrieval',
        out_dir=work / 'decoder',
    )
    metrics = evaluate_decoding(
        eeg_emb=eeg,
        text_emb=text_emb,
        texts=texts,
        meta=meta,
        aligner=aligner,
        decoder=decode_arts.retrieval,
        out_dir=work / 'evaluation',
        run_name='decode-demo',
    )

    summary = {
        'zte_ckpt': str(zte_ckpt),
        'aligner_path': str(align_arts.aligner_path),
        'align_metrics': align_arts.metrics,
        'decode_metrics': decode_arts.metrics,
        'evaluation': {k: v for k, v in metrics.items() if k != 'breakdown'},
        'verdict': metrics.get('verdict'),
        'n_pairs': len(texts),
        'embed_dim': dim,
    }
    (work / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    _LOG.info('Decode demo complete. Artifacts in %s', work.resolve())
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
