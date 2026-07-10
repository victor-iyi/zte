"""EEG→language decode package: text encoders, EEG-OT-CLIP alignment, decoders.

Typical flow::

    from zte.decode import DecoderConfig, build_text_encoder, run_alignment
    from zte.decode.evaluate import evaluate_decoding

    cfg = DecoderConfig()
    arts = run_alignment(eeg, text_emb, texts, cfg)  # precomputed embeddings
    # Dataset path: zte.decode.train_align.run_alignment(dataset=..., zte_ckpt=..., ...)
    metrics = evaluate_decoding(
        eeg_emb=eeg, text_emb=text_emb, texts=texts, out_dir='res/decode/eval',
    )
"""

# pylint: disable=undefined-all-variable,import-outside-toplevel
from __future__ import annotations

from zte.decode.config import (
    AlignConfig,
    DecoderConfig,
    GenerativeConfig,
    TextEncoderConfig,
)

__version__ = '0.1.0'

__all__ = [
    'DecoderConfig',
    'AlignConfig',
    'TextEncoderConfig',
    'GenerativeConfig',
    'TextEncoder',
    'HashTextEncoder',
    'TextEmbeddingCache',
    'build_text_encoder',
    'EEGProjector',
    'OTCLIPAligner',
    'info_nce_loss',
    'sinkhorn_ot_loss',
    'RetrievalDecoder',
    'PrefixLanguageDecoder',
    'LanguageDecoder',
    'build_sentence_pairs',
    'build_word_pairs',
    'PairedEmbeddingDataset',
    'make_paired_loader',
    'AlignmentArtifacts',
    'run_alignment',
    'run_alignment_from_embeddings',
    'run_decode_training',
    'evaluate_decoding',
    'bleu_score',
    'exact_match',
    'token_f1',
    'wer_score',
    'cross_modal_retrieval',
    'noise_anchored_retrieval',
    '__version__',
]


def __getattr__(name: str) -> object:
    """Lazily exposes heavier decode entry points.

    Args:
        name: Attribute being accessed.

    Returns:
        The requested object.

    Raises:
        AttributeError: If ``name`` is not a known lazy export.
    """
    if name in {'TextEncoder', 'HashTextEncoder', 'TextEmbeddingCache', 'build_text_encoder'}:
        from zte.decode import text_encoder as _te

        return getattr(_te, name)
    if name in {'EEGProjector', 'OTCLIPAligner', 'info_nce_loss', 'sinkhorn_ot_loss'}:
        from zte.decode import alignment as _alignment

        return getattr(_alignment, name)
    if name in {'RetrievalDecoder', 'PrefixLanguageDecoder', 'LanguageDecoder'}:
        from zte.decode import decoders as _decoders

        return getattr(_decoders, name)
    if name in {
        'build_sentence_pairs',
        'build_word_pairs',
        'PairedEmbeddingDataset',
        'make_paired_loader',
    }:
        from zte.decode import pairing as _pairing

        return getattr(_pairing, name)
    if name == 'AlignmentArtifacts':
        from zte.decode.train_align import AlignmentArtifacts

        return AlignmentArtifacts
    if name == 'run_alignment_from_embeddings':
        from zte.decode.train_align import run_alignment_from_embeddings

        return run_alignment_from_embeddings
    if name == 'run_alignment':
        from zte.decode.train import run_alignment

        return run_alignment
    if name == 'run_decode_training':
        from zte.decode.train_decode import run_decode_training

        return run_decode_training
    if name == 'evaluate_decoding':
        from zte.decode.evaluate import evaluate_decoding

        return evaluate_decoding
    if name in {
        'bleu_score',
        'exact_match',
        'token_f1',
        'wer_score',
        'cross_modal_retrieval',
        'noise_anchored_retrieval',
    }:
        from zte.decode import metrics as _metrics

        return getattr(_metrics, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
