"""Frozen per-word-type text embeddings: the target for token-level lexical alignment."""

from __future__ import annotations

import re

import numpy as np

from zte.data.targets.text import build_sentence_text_matrix
from zte.logging_utils import get_logger

_LOG = get_logger('data.lexical')

# ZuCo's word column keeps the punctuation the reader saw ("colonel." / "Bread,"), which is part of the stimulus but
# not part of the word, and would send two spellings of one word to two different embeddings.
_EDGE_PUNCT = re.compile(r'^[^\w]+|[^\w]+$')


def normalise_word(word: str) -> str:
    """Returns the lexical form of a ZuCo word token: edge punctuation stripped, case preserved.

    Note:
        Case is kept because the frozen encoders that supply the target are case-sensitive and a sentence-initial
        capital is information the reader saw. Only the punctuation the tokeniser would split off is removed.

    Args:
        word (str): The word as it appears in the stimulus.

    Returns:
        str: The stripped form, or the original when stripping would empty it.
    """
    stripped = _EDGE_PUNCT.sub('', str(word))
    return stripped or str(word)


def build_lexical_matrix(
    vocab: dict[str, int],
    source: str | None,
    *,
    backend: str = 'auto',
    prefix: str = '',
    device: str = 'cpu',
    cache_dir: str = 'res/cache/text',
) -> tuple[np.ndarray | None, int]:
    """Builds the L2-normalised `(n_word_types, dim)` frozen word-embedding matrix indexed by `word_id`.

    Args:
        vocab (dict[str, int]): The `{word: word_id}` map from `ZuCoTorchDataset.word_vocab`.
        source (str | None): Frozen text-encoder model id, or `None` / `'hash'` for no target.
        backend (str, optional): `'sentence-transformers'`, `'hf'` or `'auto'`. Defaults to 'auto'.
        prefix (str, optional): Instruction prefix the encoder expects. Defaults to ''.
        device (str, optional): Torch device for the encoder pass. Defaults to 'cpu'.
        cache_dir (str, optional): On-disk embedding cache. Defaults to 'res/cache/text'.

    Returns:
        tuple[np.ndarray | None, int]: `((n_word_types, dim) float32, dim)`, or `(None, 0)` when unavailable.

    Note:
        The same encoder that supplies the sentence-level CLIP target supplies this one, so a word and the sentence
        containing it land in one space. That is what lets the decoder's evidence path read per-word vectors through
        the very map it already learned for the pooled vector.
    """
    if not vocab:
        return None, 0

    words = [''] * len(vocab)
    for word, index in vocab.items():
        if 0 <= index < len(words):
            words[index] = normalise_word(word)

    matrix, dim = build_sentence_text_matrix(
        words,
        source,
        backend=backend,
        prefix=prefix,
        device=device,
        cache_dir=cache_dir,
    )
    if matrix is None:
        _LOG.warning(
            'Lexical target unavailable for source %r: token-level alignment has nothing to align against and is '
            'switched off for this run. Install the `meaning` group or pre-download the encoder.',
            source,
        )
        return None, 0

    _LOG.info('Built lexical target: %d word types x %d dims (%s).', len(words), dim, source)
    return matrix, dim
