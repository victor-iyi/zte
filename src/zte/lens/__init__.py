"""The lens: run one reading through a trained checkpoint and see what the model did -- inspection, never evaluation.

- `zte.lens.saliency` -- reading selection, occlusion saliency, the neighbour gallery, and `lens_report`.
- `zte.lens.trace` -- the greedy decode trace for decoder checkpoints.
- `zte.lens.temporal` -- the temporal occlusion latency profile over the word-locked raw window.
- `zte.lens.attention` -- the encoder's own attention, read through forward hooks: when in the word, where on the scalp.
- `zte.lens.page` -- the self-contained HTML page rendered from one `lens.json`.
"""

__all__ = [
    'DISCLAIMER',
    'Reading',
    'attention_profile',
    'build_lens_page',
    'channel_saliency',
    'decode_trace',
    'lens_report',
    'neighbors',
    'select_reading',
    'temporal_saliency',
    'word_saliency',
]

_SALIENCY_EXPORTS = {
    'DISCLAIMER',
    'Reading',
    'channel_saliency',
    'lens_report',
    'neighbors',
    'select_reading',
    'word_saliency',
}


def __getattr__(name: str) -> object:
    """Lazily resolves the lens API, keeping torch out of the package import."""
    if name in _SALIENCY_EXPORTS:
        from zte.lens import saliency

        return getattr(saliency, name)
    if name == 'decode_trace':
        from zte.lens.trace import decode_trace

        return decode_trace
    if name == 'temporal_saliency':
        from zte.lens.temporal import temporal_saliency

        return temporal_saliency
    if name == 'attention_profile':
        from zte.lens.attention import attention_profile

        return attention_profile
    if name == 'build_lens_page':
        # Resolved by name so importing the analysis core never requires the page renderer to be importable.
        from importlib import import_module

        return import_module('zte.lens.page').build_lens_page
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
