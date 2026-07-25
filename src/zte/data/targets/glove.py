"""Provision frozen static word vectors for `objective.meaning_source`, in the GloVe text format.

The meaning-distillation target must carry real semantics; the built-in hash fallback only verifies the mechanism.
Available models (dimension is the trailing number) are `glove-wiki-gigaword-50/-100/-200/-300` and
`glove-twitter-25/-50/-100/-200`.
"""

from __future__ import annotations

import gzip
import tempfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path

from zte.data.cache import fetch_artifact, publish_artifact
from zte.logging_utils import get_logger

_LOG = get_logger('data.glove')

# Public gensim-data release files: plain gzipped word2vec text, fetched without gensim.
_RELEASE_BASE: str = 'https://github.com/RaRe-Technologies/gensim-data/releases/download'

DEFAULT_MODEL: str = 'glove-wiki-gigaword-300'


def download_glove(model: str = DEFAULT_MODEL) -> Path:
    """Downloads `<model>.gz` from the gensim-data releases to a cached temp file (stdlib only)."""
    url = f'{_RELEASE_BASE}/{model}/{model}.gz'
    tmp = Path(tempfile.gettempdir()) / f'{model}.gz'
    if tmp.exists() and tmp.stat().st_size > 0:
        _LOG.info('Using cached download %s', tmp)
        return tmp
    _LOG.info('Downloading %s …', url)
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed, trusted https host
    except Exception as exc:
        raise SystemExit(
            f'Could not download {url} ({exc}). Check the model name (see the module docstring) and your network.'
        ) from exc
    _LOG.info('Downloaded %.0f MB to %s', tmp.stat().st_size / 1e6, tmp)
    return tmp


def _glove_dim(path: Path) -> int:
    """Reads the vector dimensionality from the first data row of a GloVe text file."""
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            parts = line.rstrip().split(' ')
            if len(parts) >= 3:  # skip a possible "n dim" header row
                return len(parts) - 1
    return 0


def provision_glove(
    out: str | Path,
    *,
    vocab: Iterable[str] | None = None,
    model: str = DEFAULT_MODEL,
    top: int = 50000,
    overwrite: bool = False,
) -> tuple[Path, int]:
    """Downloads a GloVe embedding and writes the (optionally vocab-restricted) subset in GloVe text format.

    Args:
        out (str | Path): Destination `.txt` (GloVe format); parent dirs are created.
        vocab (Iterable[str] | None): Keep only these words, matched case-insensitively; a dataset's own vocabulary
            yields a file of a few thousand rows. `None` keeps the `top` most frequent words instead.
        model (str): GloVe model name from gensim-data (dimension is the trailing number).
        top (int): Keep the N most frequent words when `vocab` is `None` (0 = all).
        overwrite (bool): Rebuild even when `out` exists. The vectors are corpus-independent, so reuse is safe.

    Returns:
        tuple[Path, int]: The written (or reused) path and the vector dimensionality.
    """
    out = Path(out)

    # Layered onto the persistent store: without it a fresh Colab runtime re-downloads GloVe every session.
    if not overwrite:
        fetch_artifact(out)
    if out.is_file() and not overwrite:
        dim = _glove_dim(out)
        _LOG.info('Reusing cached GloVe vectors %s (dim %d).', out, dim)
        return out, dim
    keep = {w.lower() for w in vocab if w} if vocab is not None else None
    gz = download_glove(model)

    # Stream the gzip straight to the output, filtering by vocabulary or frequency rank.
    out.parent.mkdir(parents=True, exist_ok=True)
    written, dim = 0, 0
    with gzip.open(gz, 'rt', encoding='utf-8') as fh, out.open('w', encoding='utf-8') as fout:
        for i, line in enumerate(fh):
            parts = line.rstrip().split(' ')
            if len(parts) < 3:  # skip the "n_words dim" header row
                continue
            word = parts[0]
            dim = dim or len(parts) - 1
            if keep is not None:
                if word.lower() not in keep:
                    continue
            elif top and i > top:
                break
            fout.write(line if line.endswith('\n') else line + '\n')
            written += 1

    _LOG.info('Wrote %d vectors (dim %d) to %s', written, dim, out)
    publish_artifact(out)
    return out, dim
