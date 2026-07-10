"""Train retrieval or prefix-LM decoders on aligned EEG embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from zte.decode.alignment import OTCLIPAligner
from zte.decode.config import DecoderConfig, GenerativeConfig
from zte.decode.decoders import LanguageDecoder, PrefixLanguageDecoder, RetrievalDecoder
from zte.decode.train import GenerativeArtifacts
from zte.decode.train import run_decode_training as _run_prefix_training
from zte.logging_utils import get_logger

_LOG = get_logger('decode.train_decode')

type DecodeMode = Literal['retrieval', 'prefix_lm', 'both']


@dataclass
class DecodeArtifacts:
    """Outputs of :func:`run_decode_training`.

    Attributes:
        mode: Decode path used.
        decoder_path: Checkpoint / bank path written (if any).
        retrieval: Fitted :class:`RetrievalDecoder` when applicable.
        generative: Trained :class:`PrefixLanguageDecoder` when applicable.
        metrics: Lightweight fit-time metrics.
        history: Training history (empty for retrieval).
    """

    mode: DecodeMode
    decoder_path: Path | None
    retrieval: RetrievalDecoder | None
    generative: PrefixLanguageDecoder | None
    metrics: dict[str, Any]
    history: list[dict[str, Any]]


def run_decode_training(
    *,
    eeg_emb: np.ndarray,
    text_emb: np.ndarray,
    texts: list[str],
    config: DecoderConfig | GenerativeConfig | None = None,
    aligner: OTCLIPAligner | None = None,
    mode: DecodeMode | None = None,
    out_dir: str | Path | None = None,
) -> DecodeArtifacts:
    """Trains a retrieval bank and/or a prefix-LM decoder.

    For ``mode='retrieval'`` there is no gradient training — the text bank is
    projected (optionally via ``aligner``) and indexed. For ``prefix_lm`` /
    ``both``, a :class:`PrefixLanguageDecoder` is trained with teacher forcing
    (toy or transformers backend).

    Args:
        eeg_emb: EEG (or already-aligned) embeddings ``(N, D)``.
        text_emb: Text embeddings ``(N, D_text)`` for the retrieval bank.
        texts: Target / bank strings of length ``N``.
        config: :class:`DecoderConfig` or :class:`GenerativeConfig`.
        aligner: Optional OT-CLIP aligner for projecting into the shared space.
        mode: Override for ``DecoderConfig.mode``.
        out_dir: Where to write decoder checkpoints / bank.

    Returns:
        :class:`DecodeArtifacts`.
    """
    decoder_cfg, gen_cfg, resolved_mode = _resolve(config, mode)
    out = Path(out_dir or (decoder_cfg.out_dir if decoder_cfg else gen_cfg.ckpt_dir))
    out.mkdir(parents=True, exist_ok=True)

    eeg = np.asarray(eeg_emb, dtype=np.float32)
    text = np.asarray(text_emb, dtype=np.float32)
    texts = list(texts)

    # Project into shared space when an aligner is provided.
    if aligner is not None:
        aligner.eval()
        with torch.no_grad():
            eeg_z = aligner.encode_eeg(torch.from_numpy(eeg)).cpu().numpy().astype(np.float32)
            text_z = aligner.encode_text(torch.from_numpy(text)).cpu().numpy().astype(np.float32)
    else:
        eeg_z, text_z = eeg, text

    retrieval: RetrievalDecoder | None = None
    generative: PrefixLanguageDecoder | None = None
    history: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {'n': len(texts), 'mode': resolved_mode}
    decoder_path: Path | None = None

    if resolved_mode in ('retrieval', 'both'):
        retrieval = RetrievalDecoder(
            text_z, texts, aligner=None, bank_already_aligned=True
        )
        decoder_path = out / 'retrieval_bank.pt'
        torch.save(
            {
                'text_bank_emb': text_z,
                'texts': texts,
                'mode': 'retrieval',
                'aligner': aligner.state_dict() if aligner is not None else None,
                'config': asdict(aligner.config) if aligner is not None else None,
            },
            decoder_path,
        )
        # Sanity: identical bank should retrieve itself at top-1.
        decoded = retrieval.decode(eeg_z if aligner is None else eeg_z, k=1)
        metrics['retrieval_train_string_acc'] = float(
            np.mean([a == b for a, b in zip(decoded, texts, strict=True)])
            if texts
            else 0.0
        )
        _LOG.info(
            'Retrieval bank fitted | n=%d train_acc=%.3f -> %s',
            len(texts),
            metrics['retrieval_train_string_acc'],
            decoder_path,
        )

    if resolved_mode in ('prefix_lm', 'both'):
        if decoder_cfg is not None:
            gen_cfg = decoder_cfg.generative
        gen_cfg = GenerativeConfig(
            **{**asdict(gen_cfg), 'prefix_dim': int(eeg_z.shape[1]), 'ckpt_dir': str(out)}
        )
        arts: GenerativeArtifacts = _run_prefix_training(eeg_z, texts, config=gen_cfg)
        generative = arts.decoder
        decoder_path = arts.ckpt_path or (out / 'best.pt')
        history = _history_as_list(arts.history)
        metrics['final_train_loss'] = (
            arts.history['train_loss'][-1] if arts.history.get('train_loss') else None
        )
        _LOG.info('Prefix-LM training done | path=%s', decoder_path)

    if resolved_mode == 'both' and retrieval is not None and generative is not None:
        # Persist a combined LanguageDecoder-friendly payload.
        combo = out / 'language_decoder.pt'
        torch.save(
            {
                'mode': 'both',
                'text_bank_emb': text_z,
                'texts': texts,
                'generative': generative.state_dict(),
                'generative_config': asdict(gen_cfg),
                'backend': generative.backend,
                'char_to_id': getattr(generative, 'char_to_id', {}),
            },
            combo,
        )
        decoder_path = combo

    return DecodeArtifacts(
        mode=resolved_mode,
        decoder_path=decoder_path,
        retrieval=retrieval,
        generative=generative,
        metrics=metrics,
        history=history,
    )


def build_language_decoder(
    config: DecoderConfig,
    *,
    text_bank_emb: np.ndarray | None = None,
    texts: list[str] | None = None,
    aligner: OTCLIPAligner | None = None,
) -> LanguageDecoder:
    """Convenience constructor for :class:`LanguageDecoder`."""
    return LanguageDecoder.from_config(
        config,
        text_bank_emb=text_bank_emb,
        texts=texts,
        aligner=aligner,
    )


def _resolve(
    config: DecoderConfig | GenerativeConfig | None,
    mode: DecodeMode | None,
) -> tuple[DecoderConfig | None, GenerativeConfig, DecodeMode]:
    if config is None:
        cfg = DecoderConfig()
        return cfg, cfg.generative, mode or cfg.mode
    if isinstance(config, DecoderConfig):
        return config, config.generative, mode or config.mode
    return None, config, mode or 'prefix_lm'


def _history_as_list(history: dict[str, list[float]]) -> list[dict[str, Any]]:
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
