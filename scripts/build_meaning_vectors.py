"""Provision frozen word-meaning vectors for Unit A (`objective.meaning_source`).

The meaning-distillation target must carry real semantics to attack `content = 0%`; the
built-in hash fallback only verifies the mechanism. This script downloads a standard static
GloVe embedding and writes it in the GloVe text format (`word v1 v2 …` per line) that
`zte.data.meaning.build_meaning_matrix` consumes.

It fetches directly from the public `gensim-data` release files (plain gzipped word2vec text)
with the Python standard library only — **no `gensim` dependency**, which does not build on
Python 3.14 (its bundled Cython C code still references the removed `PyDictObject.ma_version_tag`).

Restricting with `--vocab-from` to the words that actually occur in a dataset (the ZuCo
vocabulary is tiny) yields a file of a few thousand rows; otherwise `--top` keeps the N most
frequent words.

Available `--model` names (dimension is the trailing number):
    glove-wiki-gigaword-50 / -100 / -200 / -300   ·   glove-twitter-25 / -50 / -100 / -200

Examples::

    # ZuCo-only GloVe-300 (smallest, exact) — the path sota_loso.yaml expects:
    python scripts/build_meaning_vectors.py --out res/vectors/glove.300d.txt \
        --vocab-from experiments/sota_loso.yaml --root res/data/zuco_extracted

    # 50k most frequent GloVe-300 vectors (no dataset needed):
    python scripts/build_meaning_vectors.py --out res/vectors/glove.300d.txt --top 50000
"""

from __future__ import annotations

import argparse
import gzip
import tempfile
import urllib.request
from pathlib import Path

from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('scripts.meaning_vectors')

# Public gensim-data release files: plain gzipped word2vec text, fetched without gensim.
_RELEASE_BASE = 'https://github.com/RaRe-Technologies/gensim-data/releases/download'


def _vocab_from_config(config_path: str, root: str | None) -> set[str] | None:
    """Returns the lowercased word set of the dataset a config defines (or None on failure)."""
    try:
        from zte.config import ZTEConfig
        from zte.data.dataset import ZuCoDataset

        cfg = ZTEConfig.from_yaml(config_path).dataset
        if root:
            cfg.root = root
        ds = ZuCoDataset(cfg).build(show_progress=False)
        words = ds.words['word'].dropna().astype(str)
        return {w.lower() for w in words if w}
    except Exception as exc:  # pragma: no cover - provisioning convenience only
        _LOG.warning(
            'Could not read vocab from %s (%s); writing the full top-N instead.', config_path, exc
        )
        return None


def _download(model: str) -> Path:
    """Downloads `<model>.gz` from the gensim-data releases to a temp file (stdlib only)."""
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
            f'Could not download {url} ({exc}). Check the --model name (see --help) and your network.'
        ) from exc
    _LOG.info('Downloaded %.0f MB to %s', tmp.stat().st_size / 1e6, tmp)
    return tmp


def main() -> None:
    """Downloads a GloVe embedding and writes it in GloVe text format (no gensim)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--out', required=True, type=Path, help='Destination .txt (GloVe format).')
    parser.add_argument(
        '--model', default='glove-wiki-gigaword-300', help='gensim-data GloVe model name.'
    )
    parser.add_argument(
        '--top', type=int, default=50000, help='Keep the N most frequent words (0 = all).'
    )
    parser.add_argument(
        '--vocab-from', default=None, help='Experiment YAML to restrict the vocabulary to.'
    )
    parser.add_argument('--root', default=None, help='Override dataset root for --vocab-from.')
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args()
    configure_logging(args.log_level)

    keep = _vocab_from_config(args.vocab_from, args.root) if args.vocab_from else None
    gz = _download(args.model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written, dim = 0, 0
    with gzip.open(gz, 'rt', encoding='utf-8') as fh, args.out.open('w', encoding='utf-8') as out:
        for i, line in enumerate(fh):
            parts = line.rstrip().split(' ')
            if len(parts) < 3:  # skip the "n_words dim" header row
                continue
            word = parts[0]
            dim = dim or len(parts) - 1
            if keep is not None:
                if word.lower() not in keep:
                    continue
            elif args.top and i > args.top:
                break
            out.write(line if line.endswith('\n') else line + '\n')
            written += 1

    _LOG.info('Wrote %d vectors (dim %d) to %s', written, dim, args.out)
    print(f'{args.out}  (dim={dim}, rows={written})')
    print(f'→ set objective.meaning_source: {args.out}  and  objective.meaning_dim: {dim}')


if __name__ == '__main__':
    main()
