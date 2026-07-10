"""Dataset-aware EEG-OT-CLIP alignment training.

Wraps the precomputed-embedding trainer in :mod:`zte.decode.train` with a
ZuCo + ZTE checkpoint path that builds sentence/word pairs, splits, evaluates
retrieval, and saves a best aligner checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from zte.data.dataset import ZuCoDataset
from zte.decode.alignment import OTCLIPAligner
from zte.decode.config import AlignConfig, DecoderConfig, TextEncoderConfig
from zte.decode.pairing import build_sentence_pairs, build_word_pairs
from zte.decode.text_encoder import TextEmbeddingCache, build_text_encoder
from zte.decode.train import run_alignment as _run_alignment_embeddings
from zte.logging_utils import get_logger
from zte.training.metrics import retrieval_metrics

_LOG = get_logger('decode.train_align')


@dataclass
class AlignmentArtifacts:
    """Outputs of dataset- or embedding-based alignment training.

    Attributes:
        aligner_path: Path to the best aligner checkpoint.
        config: Alignment (or full decoder) config used for the run.
        metrics: Final validation / retrieval metrics.
        history: Per-epoch metric dicts (list form for JSON friendliness).
    """

    aligner_path: Path
    config: DecoderConfig | AlignConfig
    metrics: dict[str, Any]
    history: list[dict[str, Any]]


def run_alignment_from_embeddings(
    eeg_train: np.ndarray,
    text_train: np.ndarray,
    eeg_val: np.ndarray,
    text_val: np.ndarray,
    config: AlignConfig,
    out_dir: str | Path | None = None,
    texts_train: list[str] | None = None,
    texts_val: list[str] | None = None,
) -> AlignmentArtifacts:
    """Trains an OT-CLIP aligner from precomputed train/val embedding pairs.

    Useful for unit tests that skip a real ZTE checkpoint.

    Args:
        eeg_train: Train EEG embeddings ``(N, D)``.
        text_train: Train text embeddings ``(N, D)``.
        eeg_val: Val EEG embeddings ``(M, D)``.
        text_val: Val text embeddings ``(M, D)``.
        config: Alignment hyper-parameters.
        out_dir: Checkpoint directory (overrides ``config.ckpt_dir``).
        texts_train: Optional train surface strings.
        texts_val: Optional val surface strings.

    Returns:
        :class:`AlignmentArtifacts` with path, metrics and history.
    """
    align_cfg = _copy_align(config)
    if out_dir is not None:
        align_cfg.ckpt_dir = str(out_dir)
    texts_train = texts_train if texts_train is not None else [''] * len(eeg_train)
    texts_val = texts_val if texts_val is not None else [''] * len(eeg_val)

    # Persist the full text bank (train+val) for retrieval decode at inference.
    text_all = (
        np.concatenate([text_train, text_val], axis=0) if len(eeg_val) else text_train
    )
    texts_all = list(texts_train) + list(texts_val)

    # Train with an internal random split disabled: pass only train, val_fraction=0.
    train_cfg = _copy_align(align_cfg)
    train_cfg.val_fraction = 0.0
    arts = _run_alignment_embeddings(
        eeg_train,
        text_train,
        texts_train,
        config=train_cfg,
        val_fraction=0.0,
    )
    aligner = arts.aligner
    metrics = _retrieval_on_split(aligner, eeg_val, text_val) if len(eeg_val) else {}
    if not metrics and len(eeg_train):
        metrics = _retrieval_on_split(aligner, eeg_train, text_train)

    # Re-save with full bank (train+val texts) under out_dir for retrieval decode.
    out = Path(align_cfg.ckpt_dir) / align_cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    aligner_path = out / 'best.pt'
    torch.save(
        {
            'aligner': aligner.state_dict(),
            'config': asdict(align_cfg),
            'texts': texts_all,
            'text_bank_emb': np.asarray(text_all, dtype=np.float32),
            'metrics': metrics,
            'zte_ckpt': None,
            'text_config': None,
        },
        aligner_path,
    )
    history = _history_as_list(arts.history)
    return AlignmentArtifacts(
        aligner_path=aligner_path,
        config=align_cfg,
        metrics=metrics,
        history=history,
    )


def run_alignment(
    *,
    dataset: ZuCoDataset,
    zte_ckpt: str | Path,
    config: AlignConfig | DecoderConfig,
    text_config: TextEncoderConfig | None = None,
    out_dir: str | Path | None = None,
) -> AlignmentArtifacts:
    """Full alignment pipeline: ZTE embed → text encode → train OT-CLIP.

    Args:
        dataset: Built :class:`~zte.data.dataset.ZuCoDataset`.
        zte_ckpt: Path to a trained ZTE ``best.pt`` / ``last.pt``.
        config: :class:`AlignConfig` or :class:`DecoderConfig`.
        text_config: Optional text-encoder config (else from ``DecoderConfig.text``
            or a hash-friendly default).
        out_dir: Output directory for checkpoints (overrides config).

    Returns:
        :class:`AlignmentArtifacts`.
    """
    from zte.inference.embed import ZTEEmbedder

    align_cfg, text_cfg, decoder_cfg = _resolve_configs(config, text_config)
    if out_dir is not None:
        align_cfg.ckpt_dir = str(out_dir)

    embedder = ZTEEmbedder.from_checkpoint(zte_ckpt, dataset)
    text_encoder = build_text_encoder(text_cfg)
    cache = TextEmbeddingCache(text_cfg.cache_dir, text_cfg.model_name)

    splits = dataset.split(
        align_cfg.split,
        val_fraction=align_cfg.val_fraction,
        holdout_subject=align_cfg.loso_holdout_subject,
        seed=align_cfg.seed,
    )
    train_idx = splits['train']
    val_idx = splits['val']

    pair_fn = build_word_pairs if align_cfg.level == 'word' else build_sentence_pairs
    eeg_tr, text_tr, texts_tr, _meta_tr = pair_fn(
        dataset, embedder, text_encoder, indices=train_idx, cache=cache
    )
    eeg_va, text_va, texts_va, _meta_va = pair_fn(
        dataset, embedder, text_encoder, indices=val_idx, cache=cache
    )
    _LOG.info(
        'Alignment pairs | train=%d val=%d level=%s split=%s',
        len(texts_tr),
        len(texts_va),
        align_cfg.level,
        align_cfg.split,
    )

    # Sync dims from actual embeddings.
    align_cfg = AlignConfig(
        **{
            **asdict(align_cfg),
            'eeg_dim': int(eeg_tr.shape[1]) if len(eeg_tr) else align_cfg.eeg_dim,
            'text_dim': int(text_tr.shape[1]) if len(text_tr) else align_cfg.text_dim,
        }
    )

    arts = run_alignment_from_embeddings(
        eeg_tr,
        text_tr,
        eeg_va,
        text_va,
        config=align_cfg,
        out_dir=align_cfg.ckpt_dir,
        texts_train=texts_tr,
        texts_val=texts_va,
    )

    # Enrich checkpoint with ZTE / text-config references.
    payload = torch.load(arts.aligner_path, map_location='cpu', weights_only=False)
    payload['zte_ckpt'] = str(zte_ckpt)
    payload['text_config'] = asdict(text_cfg)
    payload['metrics'] = arts.metrics
    torch.save(payload, arts.aligner_path)

    return AlignmentArtifacts(
        aligner_path=arts.aligner_path,
        config=decoder_cfg if decoder_cfg is not None else align_cfg,
        metrics=arts.metrics,
        history=arts.history,
    )


def _resolve_configs(
    config: AlignConfig | DecoderConfig,
    text_config: TextEncoderConfig | None,
) -> tuple[AlignConfig, TextEncoderConfig, DecoderConfig | None]:
    """Unpacks align / text configs from a DecoderConfig or AlignConfig."""
    if isinstance(config, DecoderConfig):
        align_cfg = _copy_align(config.align)
        text_cfg = text_config or config.text
        return align_cfg, text_cfg, config
    return _copy_align(config), text_config or TextEncoderConfig(backend='hash', model_name='hash'), None


def _copy_align(config: AlignConfig) -> AlignConfig:
    """Shallow-copies an AlignConfig via dataclass fields."""
    return AlignConfig(**{f.name: getattr(config, f.name) for f in fields(AlignConfig)})


def _retrieval_on_split(
    aligner: OTCLIPAligner,
    eeg: np.ndarray,
    text: np.ndarray,
) -> dict[str, float]:
    """Projects a split and computes retrieval metrics."""
    if len(eeg) == 0:
        return {}
    aligner.eval()
    with torch.no_grad():
        eeg_z = aligner.encode_eeg(torch.from_numpy(np.asarray(eeg, dtype=np.float32)))
        text_z = aligner.encode_text(torch.from_numpy(np.asarray(text, dtype=np.float32)))
    return retrieval_metrics(
        eeg_z.cpu().numpy(),
        text_z.cpu().numpy(),
        ks=(1, 5, 10),
    )


def _history_as_list(history: dict[str, list[float]]) -> list[dict[str, Any]]:
    """Converts column-wise history into a list of per-epoch dicts."""
    if not history:
        return []
    n = max((len(v) for v in history.values()), default=0)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {'epoch': i + 1}
        for key, values in history.items():
            if i < len(values):
                row[key] = values[i]
        rows.append(row)
    return rows


__all__ = [
    'AlignmentArtifacts',
    'run_alignment',
    'run_alignment_from_embeddings',
]
