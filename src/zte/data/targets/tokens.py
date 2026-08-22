"""Tokenised reference sentences: the supervision target for the frozen-LM prefix decoder."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from zte.data.cache import fetch_artifact, publish_artifact
from zte.logging_utils import get_logger

_LOG = get_logger('data.tokens')

# The offline stand-in for a pretrained tokeniser, sized to the tiny test LM's 64-token vocabulary.
TINY_SOURCE: str = 'tiny'
_TINY_ALPHABET: str = ' abcdefghijklmnopqrstuvwxyz0123456789.,\'"!?;:-()'

# A fixed probe whose ids go into the fingerprint, so a silent tokeniser upgrade cannot pass unnoticed.
_PROBE: str = 'The quick brown fox jumps over the lazy dog.'

# Bump whenever `_encode` changes what it writes. The cache is keyed by corpus and tokeniser, neither of which moves
# when the encoding rule does, so without this a bundle already on Drive would be reused under the new semantics.
_SCHEMA: Final[int] = 2
"""Encoding-rule version folded into the token cache key."""


class TinyByteTokenizer:
    """A deterministic 64-token character tokeniser for the offline `'tiny'` LM.

    It exists so the decoder path -- targets, prompt assembly, generation -- can run in tests and on a laptop with no
    network and no pretrained weights. Characters outside `_TINY_ALPHABET` map to `unk_id`, so `decode(encode(t))` is
    lossy for them by construction. One token is one character, so a `max_target_tokens` tuned for word-pieces
    truncates hard here; that only ever costs a smoke run its tail, never a reported number.

    Attributes:
        vocab_size (int): Token count, matching the tiny LM's embedding table.
        pad_id (int): Padding id.
        bos_id (int): Beginning-of-sequence id.
        eos_id (int): End-of-sequence id.
        unk_id (int): Id for characters outside the alphabet.
    """

    vocab_size: int = 64
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3

    def __init__(self) -> None:
        """Builds the character-to-id table."""
        self._to_id = {c: i + 4 for i, c in enumerate(_TINY_ALPHABET)}
        self._to_char = {i: c for c, i in self._to_id.items()}

    @property
    def name_or_path(self) -> str:
        """The identifier recorded in provenance for this tokeniser."""
        return TINY_SOURCE

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        """Encodes `text` to ids, lowercasing and mapping unknown characters to `unk_id`.

        Args:
            text (str): The sentence to encode.
            add_eos (bool, optional): Append `eos_id`. Defaults to True.

        Returns:
            list[int]: The token ids.
        """
        ids = [self._to_id.get(c, self.unk_id) for c in text.lower()]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decodes ids back to text, dropping the pad/bos/eos specials.

        Args:
            ids (Sequence[int]): Token ids.

        Returns:
            str: The decoded string.
        """
        return ''.join(self._to_char.get(int(i), '') for i in ids)


@dataclass(slots=True)
class TextTargets:
    """Padded target token ids and mask for one reference sentence per row.

    Attributes:
        keys (tuple[str, ...]): Row names, one per reference (the stimulus keys for a ZuCo gallery).
        texts (tuple[str, ...]): The reference strings the rows were built from.
        ids (np.ndarray): `(n_text, max_length)` int64 token ids, right-padded with `pad_id`.
        mask (np.ndarray): `(n_text, max_length)` bool, `True` at real target tokens.
        pad_id (int): The padding id used, so a loss can mask it without re-deriving it.
        fingerprint (str): Tokeniser identity hash, recorded in run provenance.
        truncation_rate (float): Fraction of references that hit `max_length`, which is a silently-clipped target.
    """

    keys: tuple[str, ...]
    texts: tuple[str, ...]
    ids: np.ndarray
    mask: np.ndarray
    pad_id: int
    fingerprint: str
    truncation_rate: float
    _rows: dict[str, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Indexes the rows by key."""
        self._rows = {k: i for i, k in enumerate(self.keys)}

    def __len__(self) -> int:
        """Returns the number of reference sentences."""
        return len(self.keys)

    @property
    def max_length(self) -> int:
        """The padded target width."""
        return int(self.ids.shape[1])

    def index(self, key: str) -> int:
        """Returns the row of `key`, or `-1` when it is not in the gallery."""
        return self._rows.get(key, -1)

    def lookup(self, keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Gathers the rows for `keys`.

        Args:
            keys (Sequence[str]): Stimulus keys to look up.

        Returns:
            tuple[np.ndarray, np.ndarray]: `(ids, mask)`, each `(len(keys), max_length)`. Unknown keys give an
                all-padding row, which the mask marks entirely invalid rather than failing the batch.
        """
        rows = [self.index(k) for k in keys]
        ids = np.full((len(rows), self.max_length), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(rows), self.max_length), dtype=bool)
        for i, r in enumerate(rows):
            if r >= 0:
                ids[i] = self.ids[r]
                mask[i] = self.mask[r]
        return ids, mask


def _cache_path(texts: Sequence[str], source: str, revision: str | None, max_length: int, cache_dir: str) -> Path:
    """Deterministic cache file for a (corpus, tokeniser, revision, width) target-token matrix."""
    h = hashlib.sha1()
    h.update(f'{source}|{revision or "main"}|{max_length}|{len(texts)}|v{_SCHEMA}'.encode())
    for t in texts:
        h.update(t.encode('utf-8', 'ignore'))
        h.update(b'\x00')
    return Path(cache_dir) / f'tokens_{h.hexdigest()[:16]}.npz'


def load_tokenizer(source: str, revision: str | None, cache_dir: str | None) -> Any:
    """Loads the HuggingFace tokeniser for `source`, or the offline stub for `'tiny'`.

    Args:
        source (str): Tokeniser id, or `'tiny'`.
        revision (str | None): Pinned commit SHA.
        cache_dir (str | None): Local snapshot directory.

    Returns:
        Any: A `TinyByteTokenizer` or a HuggingFace fast tokeniser.

    Raises:
        RuntimeError: If the tokeniser cannot be loaded. Decoder targets have no meaningful fallback -- a hash or a
            character stub would silently change what the model is trained to say -- so this fails loudly.
    """
    if source == TINY_SOURCE:
        return TinyByteTokenizer()
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(source, revision=revision, cache_dir=cache_dir)
    except (ImportError, OSError, ValueError) as exc:
        raise RuntimeError(
            f'Could not load the decoder target tokeniser {source!r} (revision={revision!r}): {exc!r}. '
            "Install `transformers` and pre-download the model, or use lm_source='tiny' for offline runs."
        ) from exc


def fingerprint_tokenizer(tokenizer: Any, source: str, revision: str | None) -> str:
    """Hashes the tokeniser identity plus its ids for a fixed probe, so version drift is detectable.

    Args:
        tokenizer (Any): A `TinyByteTokenizer` or a HuggingFace tokeniser.
        source (str): The tokeniser id it was loaded from.
        revision (str | None): Pinned commit SHA, or `None` for `main`.

    Returns:
        str: A 16-character hash of the identity and the probe's ids.
    """
    if isinstance(tokenizer, TinyByteTokenizer):
        probe = tokenizer.encode(_PROBE, add_eos=False)
        vocab_size = tokenizer.vocab_size
    else:
        probe = list(tokenizer(_PROBE, add_special_tokens=False)['input_ids'])
        vocab_size = int(len(tokenizer))
    h = hashlib.sha1()
    h.update(f'{source}|{revision or "main"}|{vocab_size}|{probe}'.encode())
    return h.hexdigest()[:16]


def _rows(tokenizer: Any, texts: Sequence[str]) -> tuple[list[list[int]], int, int | None]:
    """Encodes `texts` to unpadded id rows, returning them with the pad id and the end-of-sequence id."""
    if isinstance(tokenizer, TinyByteTokenizer):
        return [tokenizer.encode(t, add_eos=False) for t in texts], tokenizer.pad_id, tokenizer.eos_id

    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    eos_id = None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
    encoded = tokenizer(list(texts))['input_ids']

    return [[int(t) for t in row] for row in encoded], pad_id, eos_id


def _encode(tokenizer: Any, texts: Sequence[str], max_length: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Encodes `texts` into padded `(ids, mask)` plus the pad id and the count of truncated rows.

    Note:
        Every row that fits ends in the end-of-sequence id, and that id is inside the mask. Without it the bridge is
        never supervised to stop, and free-running decode then runs to `max_new_tokens` on every row -- which is not
        a scoring detail but an unbounded hypothesis measured against a 19.6-word reference. Several tokenisers,
        Qwen2's among them, add no special tokens of their own, so it is appended here rather than left to the
        library. A row long enough to be truncated loses it, because at that point the reference does not fit either.
    """
    rows, pad_id, eos_id = _rows(tokenizer, texts)
    ids = np.full((len(rows), max_length), pad_id, dtype=np.int64)
    mask = np.zeros((len(rows), max_length), dtype=bool)

    n_trunc = 0
    for i, encoded in enumerate(rows):
        terminated = encoded if eos_id is None or (encoded and encoded[-1] == eos_id) else [*encoded, eos_id]
        n_trunc += int(len(terminated) > max_length)
        kept = terminated[:max_length]
        ids[i, : len(kept)] = kept
        mask[i, : len(kept)] = True

    return ids, mask, pad_id, n_trunc


def build_target_tokens(
    texts: Sequence[str],
    tokenizer_source: str,
    *,
    keys: Sequence[str] | None = None,
    revision: str | None = None,
    max_length: int = 96,
    cache_dir: str = 'res/cache/tokens',
    model_cache_dir: str | None = None,
) -> TextTargets:
    """Tokenises the reference sentences a decoder is trained to produce, caching the matrix on disk.

    Row `i` is `texts[i]`, so the caller keeps whatever ordering it indexes by -- for ZuCo that is the `text_vocab`
    id carried in every batch as `sentence_text_id`. Targets must be the readable punctuated sentence, never the
    lowercased `stimulus_key` used for grouping.

    Args:
        texts (Sequence[str]): Reference sentences, one per gallery row.
        tokenizer_source (str): HuggingFace tokeniser id, or `'tiny'` for the offline 64-token stub.
        keys (Sequence[str] | None, optional): Row names for `TextTargets.lookup` (stimulus keys). Defaults to None,
            which names each row by its own text.
        revision (str | None, optional): Pinned commit SHA for the tokeniser. Defaults to None.
        max_length (int, optional): Padded target width; longer references are truncated. Defaults to 96.
        cache_dir (str, optional): Directory for the on-disk token cache. Defaults to 'res/cache/tokens'.
        model_cache_dir (str | None, optional): Local snapshot directory for the tokeniser weights. Defaults to None.

    Returns:
        TextTargets: Padded ids/mask, the tokeniser fingerprint and the truncation rate.

    Raises:
        ValueError: If `keys` is given with a different length than `texts`.
        RuntimeError: If the tokeniser cannot be loaded (see `load_tokenizer`).
    """
    if keys is not None and len(keys) != len(texts):
        raise ValueError(f'keys/texts length mismatch: {len(keys)} vs {len(texts)}.')
    row_keys = tuple(str(k) for k in (keys if keys is not None else texts))

    cache = _cache_path(texts, tokenizer_source, revision, max_length, cache_dir)
    fetch_artifact(cache)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as blob:
            meta = json.loads(str(blob['meta']))
            ids, mask = blob['ids'].astype(np.int64), blob['mask'].astype(bool)
        if len(ids) == len(texts):
            _LOG.info('Loaded cached decoder targets %s (%d sentences).', cache.name, len(ids))
            return TextTargets(
                keys=row_keys,
                texts=tuple(texts),
                ids=ids,
                mask=mask,
                pad_id=int(meta['pad_id']),
                fingerprint=str(meta['fingerprint']),
                truncation_rate=float(meta['truncation_rate']),
            )

    tokenizer = load_tokenizer(tokenizer_source, revision, model_cache_dir)
    ids, mask, pad_id, n_trunc = _encode(tokenizer, texts, max_length)
    rate = float(n_trunc) / max(len(texts), 1)
    fingerprint = fingerprint_tokenizer(tokenizer, tokenizer_source, revision)

    cache.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(
        {
            'pad_id': pad_id,
            'fingerprint': fingerprint,
            'truncation_rate': rate,
            'source': tokenizer_source,
        }
    )
    np.savez(cache, ids=ids, mask=mask, meta=np.array(meta_json))
    publish_artifact(cache)
    if n_trunc:
        _LOG.warning(
            'Truncated %d/%d decoder targets at %d tokens (%.1f%%); those references cannot be produced in full.',
            n_trunc,
            len(texts),
            max_length,
            100.0 * rate,
        )
    _LOG.info(
        'Tokenised %d decoder targets with %s (fingerprint %s) -> cached %s.',
        len(texts),
        tokenizer_source,
        fingerprint,
        cache.name,
    )
    return TextTargets(
        keys=row_keys,
        texts=tuple(texts),
        ids=ids,
        mask=mask,
        pad_id=pad_id,
        fingerprint=fingerprint,
        truncation_rate=rate,
    )


# ---- Word-to-sub-word alignment ---- #

# Bumped whenever `_align_rows` changes what it writes. The id cache above is keyed separately and is deliberately
# left alone, so adding alignment never invalidates a token bundle already sitting on Drive.
_ALIGN_SCHEMA: Final[int] = 1
"""Alignment-rule version folded into the token-alignment cache key."""


@dataclass(slots=True)
class TokenAlignment:
    """Which sub-word slots of each reference sentence belong to which of its words.

    Note:
        The word axis is ZuCo's within-sentence `word_idx`, which is exactly what `content_id` is built from, so a
        batch joins to this table through an id it already carries -- no collate change, and no assumption that a
        word's position in the batch equals its position in the sentence.
    """

    token_word: np.ndarray
    """`(n_text, max_length)` int32 word index per sub-word slot; `-1` at padding, specials and unmatched slots."""

    piece_index: np.ndarray
    """`(n_text, max_length)` int16 rank of a slot inside its own word; `-1` wherever `token_word` is `-1`."""

    word_pieces: np.ndarray
    """`(n_text, max_words)` int16 sub-word count per word; `0` past the end of the sentence."""

    fingerprint: str
    """Tokeniser identity hash, so a silent tokeniser upgrade cannot pass unnoticed."""

    coverage: float
    """Fraction of words that received at least one sub-word slot; below 1 the tail was truncated or unmatched."""

    @property
    def max_words(self) -> int:
        """The padded word-axis width."""
        return int(self.word_pieces.shape[1])

    @property
    def max_length(self) -> int:
        """The padded sub-word-axis width, which matches the `TextTargets` it was built beside."""
        return int(self.token_word.shape[1])

    def pieces_per_word(self, text_ids: np.ndarray, word_idx: np.ndarray) -> np.ndarray:
        """Gathers the sub-word count of each `(text_id, word_idx)` pair, `0` where either index is out of range.

        Args:
            text_ids (np.ndarray): `(n,)` sentence-text ids.
            word_idx (np.ndarray): `(n,)` within-sentence word indices.

        Returns:
            np.ndarray: `(n,)` int16 sub-word counts.
        """
        t = np.asarray(text_ids, dtype=np.int64)
        w = np.asarray(word_idx, dtype=np.int64)
        ok = (t >= 0) & (t < self.word_pieces.shape[0]) & (w >= 0) & (w < self.max_words)
        out = np.zeros(t.shape, dtype=np.int16)
        out[ok] = self.word_pieces[t[ok], w[ok]]
        return out


def _word_start_offsets(text: str, words: Sequence[str]) -> list[int]:
    """Character offset at which each word begins, scanned forward so a repeated word never collapses onto one span."""
    lowered = text.lower()
    starts: list[int] = []
    cursor = 0
    for word in words:
        token = word.strip().lower()
        found = lowered.find(token, cursor) if token else -1
        if found < 0:
            # ZuCo's word list sometimes normalises a character the reference spells differently. An unmatched word
            # keeps the cursor where it was, so every later word still aligns; it is counted against `coverage`.
            starts.append(-1)
            continue

        starts.append(found)
        cursor = found + len(token)
    return starts


def _slot_words(offsets: Sequence[tuple[int, int]], starts: Sequence[int], n_words: int) -> np.ndarray:
    """Maps each sub-word slot's character span to the word it falls in.

    Note:
        A word owns the text from where it starts up to where the next word starts, so the punctuation and spacing
        between two words belong to the earlier one. Without that rule every trailing comma and full stop would be
        an unsupervised slot, which on ZuCo is a sixth of the sub-word sequence.

        The slot is placed by its **last** character, never its first. A byte-level BPE -- Qwen's and GPT-2's among
        them -- folds the preceding space into a word-initial token, so that token's span begins one character
        inside the previous word and probing its start would attribute every word to the one before it.
    """
    anchors = np.array([s for s in starts if s >= 0], dtype=np.int64)
    owners = np.array([i for i, s in enumerate(starts) if s >= 0], dtype=np.int64)
    out = np.full(len(offsets), -1, dtype=np.int32)
    if anchors.size == 0:
        return out

    for slot, (begin, end) in enumerate(offsets):
        if end <= begin:  # a special token carries a zero-width span
            continue

        position = int(np.searchsorted(anchors, end - 1, side='right')) - 1
        if 0 <= position < owners.size:
            out[slot] = int(owners[position])
    return np.where(out < n_words, out, -1)


def _align_rows(
    tokenizer: Any, texts: Sequence[str], words: Sequence[Sequence[str]], max_length: int, max_words: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Builds the `(token_word, piece_index, word_pieces, coverage)` tables for `texts`."""
    n_text = len(texts)
    token_word = np.full((n_text, max_length), -1, dtype=np.int32)
    piece_index = np.full((n_text, max_length), -1, dtype=np.int16)
    word_pieces = np.zeros((n_text, max_words), dtype=np.int16)

    offsets_per_text = _offsets(tokenizer, texts, max_length)
    covered = total = 0
    for row, (text, offsets) in enumerate(zip(texts, offsets_per_text, strict=True)):
        row_words = list(words[row])[:max_words]
        total += len(row_words)
        owners = _slot_words(offsets, _word_start_offsets(text, row_words), len(row_words))
        token_word[row, : len(owners)] = owners

        seen: dict[int, int] = {}
        for slot, owner in enumerate(owners):
            if owner < 0:
                continue

            piece_index[row, slot] = seen.get(int(owner), 0)
            seen[int(owner)] = seen.get(int(owner), 0) + 1
        for owner, count in seen.items():
            word_pieces[row, owner] = min(count, np.iinfo(np.int16).max)
        covered += len(seen)

    return token_word, piece_index, word_pieces, float(covered) / max(total, 1)


def _offsets(tokenizer: Any, texts: Sequence[str], max_length: int) -> list[list[tuple[int, int]]]:
    """Character spans of every sub-word slot, truncated exactly as `_encode` truncates the ids beside them."""
    if isinstance(tokenizer, TinyByteTokenizer):
        # One token is one character, so the span of slot i is character i -- and the appended EOS has none.
        return [[(i, i + 1) for i in range(min(len(t), max_length))] for t in texts]

    if not bool(getattr(tokenizer, 'is_fast', False)):
        raise RuntimeError(
            f'{tokenizer.name_or_path!r} is a slow tokeniser and reports no character offsets, so a sub-word slot '
            'cannot be attributed to a word. Install `tokenizers` so the fast implementation is available.'
        )

    # `_rows` above tokenises with the library's default `add_special_tokens=True`, so these offsets must too or the
    # two sequences differ by however many specials the tokeniser prepends and every slot is attributed to the word
    # one place to its left. A special carries a zero-width span, which `_slot_words` drops to -1.
    encoded = tokenizer(list(texts), return_offsets_mapping=True)['offset_mapping']
    return [[(int(a), int(b)) for a, b in row[:max_length]] for row in encoded]


def _align_cache_path(texts: Sequence[str], source: str, revision: str | None, max_length: int, cache_dir: str) -> Path:
    """Deterministic cache file for a (corpus, tokeniser, revision, width) alignment table."""
    h = hashlib.sha1()
    h.update(f'{source}|{revision or "main"}|{max_length}|{len(texts)}|align{_ALIGN_SCHEMA}'.encode())
    for t in texts:
        h.update(t.encode('utf-8', 'ignore'))
        h.update(b'\x00')
    return Path(cache_dir) / f'align_{h.hexdigest()[:16]}.npz'


def build_token_alignment(
    texts: Sequence[str],
    words: Sequence[Sequence[str]],
    tokenizer_source: str,
    *,
    revision: str | None = None,
    max_length: int = 96,
    max_words: int = 0,
    cache_dir: str = 'res/cache/tokens',
    model_cache_dir: str | None = None,
) -> TokenAlignment:
    """Builds the word-to-sub-word map the token-level alignment is supervised against.

    Args:
        texts (Sequence[str]): Reference sentences, row `i` being the text of `sentence_text_id == i`.
        words (Sequence[Sequence[str]]): The ZuCo word list of each sentence, in `word_idx` order.
        tokenizer_source (str): HuggingFace tokeniser id, or `'tiny'` for the offline stub.
        revision (str | None, optional): Pinned commit SHA for the tokeniser. Defaults to None.
        max_length (int, optional): Sub-word width, matching the `TextTargets` beside it. Defaults to 96.
        max_words (int, optional): Word-axis width. Defaults to 0, which sizes it from the longest sentence.
        cache_dir (str, optional): Directory for the on-disk cache. Defaults to 'res/cache/tokens'.
        model_cache_dir (str | None, optional): Local snapshot directory for the tokeniser. Defaults to None.

    Returns:
        TokenAlignment: The alignment tables, their tokeniser fingerprint and the word coverage achieved.

    Raises:
        ValueError: If `texts` and `words` disagree on length.
        RuntimeError: If the tokeniser cannot be loaded, or reports no character offsets.
    """
    if len(words) != len(texts):
        raise ValueError(f'texts/words length mismatch: {len(texts)} vs {len(words)}.')

    width = max_words or max((len(w) for w in words), default=1)

    cache = _align_cache_path(texts, tokenizer_source, revision, max_length, cache_dir)
    fetch_artifact(cache)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as blob:
            meta = json.loads(str(blob['meta']))
            tables = (blob['token_word'], blob['piece_index'], blob['word_pieces'])
        if tables[0].shape[0] == len(texts) and tables[2].shape[1] >= width:
            _LOG.info('Loaded cached token alignment %s (%d sentences).', cache.name, len(texts))
            return TokenAlignment(
                token_word=tables[0].astype(np.int32),
                piece_index=tables[1].astype(np.int16),
                word_pieces=tables[2].astype(np.int16),
                fingerprint=str(meta['fingerprint']),
                coverage=float(meta['coverage']),
            )

    tokenizer = load_tokenizer(tokenizer_source, revision, model_cache_dir)
    token_word, piece_index, word_pieces, coverage = _align_rows(tokenizer, texts, words, max_length, width)
    fingerprint = fingerprint_tokenizer(tokenizer, tokenizer_source, revision)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        token_word=token_word,
        piece_index=piece_index,
        word_pieces=word_pieces,
        meta=np.array(json.dumps({'fingerprint': fingerprint, 'coverage': coverage, 'source': tokenizer_source})),
    )
    publish_artifact(cache)

    # Coverage below one means words whose sub-words the loss can never reach, so the token level is training on
    # less than it appears to; loud enough to notice, not fatal, because a few unmatched words is normal.
    if coverage < 0.99:
        _LOG.warning(
            'Token alignment covers %.1f%% of words; the remainder is truncated or unmatched and carries no '
            'token-level target.',
            100.0 * coverage,
        )
    _LOG.info(
        'Aligned %d sentences to sub-word slots with %s (fingerprint %s, coverage %.3f) -> cached %s.',
        len(texts),
        tokenizer_source,
        fingerprint,
        coverage,
        cache.name,
    )
    return TokenAlignment(
        token_word=token_word,
        piece_index=piece_index,
        word_pieces=word_pieces,
        fingerprint=fingerprint,
        coverage=coverage,
    )


# ---- Frozen sub-word embedding target ---- #


@dataclass(slots=True)
class SubwordTargets:
    """The frozen embedding of every sub-word type the corpus actually uses.

    Note:
        Restricted to the types present, not the whole tokeniser: Qwen2.5's table is 151,936 x 896, which is 544 MB
        of frozen buffer to carry a corpus that spells 700 sentences with a few thousand distinct pieces.
    """

    rows: dict[int, int]
    """`{lm_token_id: compact row}`, so a target id in the batch indexes `matrix` without a 151k-row gather."""

    matrix: np.ndarray
    """`(n_types, dim)` float32, L2-normalised, row `rows[token_id]` being that token's frozen embedding."""

    source: str
    """The model the embeddings were read from, recorded in provenance."""

    @property
    def dim(self) -> int:
        """Width of the frozen sub-word space."""
        return int(self.matrix.shape[1])

    def compact(self, token_ids: np.ndarray) -> np.ndarray:
        """Maps LM token ids onto compact rows, `-1` for a type this target does not carry.

        Args:
            token_ids (np.ndarray): Any-shaped array of LM token ids.

        Returns:
            np.ndarray: The same shape, int64, holding compact rows.
        """
        flat = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        out = np.array([self.rows.get(int(t), -1) for t in flat], dtype=np.int64)
        return out.reshape(np.asarray(token_ids).shape)


def _hashed_subword_matrix(types: Sequence[int], dim: int) -> np.ndarray:
    """A deterministic hash embedding, so the token level stays runnable offline and in tests."""
    out = np.zeros((len(types), dim), dtype=np.float32)
    for row, token in enumerate(types):
        # `hash()` is salted per process, so a checkpoint written on one run would meet a different target on the
        # next; the digest is stable across processes and across machines, which is what "deterministic" has to mean.
        seed = int.from_bytes(hashlib.sha256(f'zte-subword|{int(token)}'.encode()).digest()[:8], 'big')
        rng = np.random.default_rng(seed)
        out[row] = rng.standard_normal(dim, dtype=np.float32)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _lm_input_embeddings(source: str, revision: str | None, model_cache_dir: str | None) -> np.ndarray | None:
    """Reads a causal LM's input embedding table to CPU float32, or returns `None` when it cannot be loaded."""
    try:
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            source, revision=revision, cache_dir=model_cache_dir, dtype=torch.float32
        )
        embeddings: Any = model.get_input_embeddings()
        with torch.no_grad():
            table = embeddings.weight.detach().to('cpu', torch.float32).numpy().copy()
        del model
        return table
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        _LOG.warning(
            'Could not read the sub-word embedding table from %r (%r); the token level falls back to a hash target, '
            'which carries no semantics -- nothing lexical from this run is meaningful.',
            source,
            exc,
        )
        return None


def build_subword_matrix(
    token_ids: np.ndarray,
    source: str,
    *,
    revision: str | None = None,
    dim: int = 256,
    model_cache_dir: str | None = None,
) -> SubwordTargets:
    """Builds the frozen sub-word embedding target for every token type present in `token_ids`.

    Args:
        token_ids (np.ndarray): The corpus's target token ids (a `TextTargets.ids` matrix is the usual argument).
        source (str): The LM whose input embedding table supplies the vectors, or `'tiny'` for the offline hash.
        revision (str | None, optional): Pinned commit SHA. Defaults to None.
        dim (int, optional): Width used only by the offline hash fallback. Defaults to 256.
        model_cache_dir (str | None, optional): Local snapshot directory for the weights. Defaults to None.

    Returns:
        SubwordTargets: The compact vocabulary and its L2-normalised embedding matrix.
    """
    types = sorted({int(t) for t in np.asarray(token_ids).reshape(-1) if int(t) >= 0})
    rows = {token: i for i, token in enumerate(types)}

    table = None if source == TINY_SOURCE else _lm_input_embeddings(source, revision, model_cache_dir)
    if table is None:
        matrix = _hashed_subword_matrix(types, dim)
        _LOG.info('Built a hashed sub-word target: %d types x %d dims (offline).', len(types), dim)
        return SubwordTargets(rows=rows, matrix=matrix, source=f'{source}#hash')

    keep = [t for t in types if 0 <= t < table.shape[0]]
    if len(keep) != len(types):
        _LOG.warning(
            '%d of %d sub-word types fall outside the embedding table and are dropped.',
            len(types) - len(keep),
            len(types),
        )
        rows = {token: i for i, token in enumerate(keep)}
    matrix = table[np.asarray(keep, dtype=np.int64)] if keep else np.zeros((0, table.shape[1]), dtype=np.float32)
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    _LOG.info('Built the sub-word target: %d types x %d dims from %s.', matrix.shape[0], matrix.shape[1], source)

    return SubwordTargets(rows=rows, matrix=matrix.astype(np.float32), source=source)
