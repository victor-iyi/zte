"""EEG→language decode package: text encoders, EEG-OT-CLIP alignment, decoders.

This is the planned downstream stage after ZTE pretraining. Typical flow::

    from zte.decode import (
        DecoderConfig,
        build_text_encoder,
        OTCLIPAligner,
        run_alignment,
        RetrievalDecoder,
        evaluate_decoding,
    )

    cfg = DecoderConfig()
    text_enc = build_text_encoder(cfg.text)          # hash backend in CI
    # ... build EEG/text pairs via zte.decode.pairing ...
    arts = run_alignment(eeg, text_emb, texts, cfg)
    metrics = evaluate_decoding(eeg, text_emb, texts, aligner=arts.aligner)
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
    'build_text_encoder',
    'EEGProjector',
    'OTCLIPAligner',
    'info_nce_loss',
    'sinkhorn_ot_loss',
    'RetrievalDecoder',
    'PrefixLanguageDecoder',
    'LanguageDecoder',
    'run_alignment',
    'run_decode_training',
    'evaluate_decoding',
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
    if name in {'TextEncoder', 'build_text_encoder'}:
        from zte.decode.text_encoder import TextEncoder, build_text_encoder

        return {'TextEncoder': TextEncoder, 'build_text_encoder': build_text_encoder}[name]
    if name in {'EEGProjector', 'OTCLIPAligner', 'info_nce_loss', 'sinkhorn_ot_loss'}:
        from zte.decode import alignment as _alignment

        return getattr(_alignment, name)
    if name in {'RetrievalDecoder', 'PrefixLanguageDecoder', 'LanguageDecoder'}:
        from zte.decode import decoders as _decoders

        return getattr(_decoders, name)
    if name == 'run_alignment':
        # Embedding-level trainer (positional arrays). Dataset-aware path:
        # ``zte.decode.train_align.run_alignment``.
        from zte.decode.train import run_alignment

        return run_alignment
    if name == 'run_decode_training':
        # High-level retrieval / prefix-LM trainer (keyword API).
        from zte.decode.train_decode import run_decode_training

        return run_decode_training
    if name == 'evaluate_decoding':
        from zte.decode.evaluate import evaluate_decoding

        return evaluate_decoding
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
