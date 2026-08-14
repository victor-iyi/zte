"""Tokenised reference sentences: the supervision target for the frozen-LM prefix decoder."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from zte.data.cache import fetch_artifact, publish_artifact
from zte.logging_utils import get_logger

_LOG = get_logger('data.tokens')

# The offline stand-in for a pretrained tokeniser, sized to the tiny test LM's 64-token vocabulary.
TINY_SOURCE: str = 'tiny'
_TINY_ALPHABET: str = ' abcdefghijklmnopqrstuvwxyz0123456789.,\'"!?;:-()'

# A fixed probe whose ids go into the fingerprint, so a silent tokeniser upgrade cannot pass unnoticed.
_PROBE: str = 'The quick brown fox jumps over the lazy dog.'


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
    h.update(f'{source}|{revision or "main"}|{max_length}|{len(texts)}'.encode())
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


def _encode(tokenizer: Any, texts: Sequence[str], max_length: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Encodes `texts` into padded `(ids, mask)` plus the pad id and the count of truncated rows."""
    n = len(texts)
    if isinstance(tokenizer, TinyByteTokenizer):
        pad_id = tokenizer.pad_id
        ids = np.full((n, max_length), pad_id, dtype=np.int64)
        mask = np.zeros((n, max_length), dtype=bool)
        n_trunc = 0
        for i, text in enumerate(texts):
            row = tokenizer.encode(text)
            n_trunc += int(len(row) > max_length)
            row = row[:max_length]
            ids[i, : len(row)] = row
            mask[i, : len(row)] = True
        return ids, mask, pad_id, n_trunc

    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    enc = tokenizer(
        list(texts),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='np',
    )
    ids = np.asarray(enc['input_ids'], dtype=np.int64)
    mask = np.asarray(enc['attention_mask'], dtype=bool)

    # A row that fills the width was clipped, since every reference otherwise ends before it.
    n_trunc = int((mask.sum(axis=1) >= max_length).sum())
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
