"""Provision frozen word-meaning vectors for Unit A (`objective.meaning_source`).

The meaning-distillation target must carry real semantics to attack `content = 0%`; the built-in hash fallback only verifies the mechanism.
This script downloads a standard static GloVe embedding and writes it in the GloVe text format (`word v1 v2 …` per line) that `zte.data.meaning.build_meaning_matrix` consumes.

It fetches directly from the public `gensim-data` release files (plain gzipped word2vec text) with the Python standard library only — **no `gensim` dependency**,
which does not build on Python 3.14 (its bundled Cython C code still references the removed `PyDictObject.ma_version_tag`).

Restricting with `--vocab-from` to the words that actually occur in a dataset (the ZuCo vocabulary is tiny) yields a file of a few thousand rows;
otherwise `--top` keeps the N most frequent words.

Available `--model` names (dimension is the trailing number):
    glove-wiki-gigaword-50 / -100 / -200 / -300   ·   glove-twitter-25 / -50 / -100 / -200

Examples::

    # ZuCo-only GloVe-300 (smallest, exact) — the path sota_loso.yaml expects:
    python scripts/build_meaning_vectors.py --out res/vectors/glove.300d.txt \
        --vocab-from experiments/sota_loso.yaml --root res/data/zuco_extracted

    # (Turn-key alternative: `zte-run --meaning static` builds + wires this per-run; see zte.cli.provision.)

    # 50k most frequent GloVe-300 vectors (no dataset needed):
    python scripts/build_meaning_vectors.py --out res/vectors/glove.300d.txt --top 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from zte.data.glove import DEFAULT_MODEL, provision_glove
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('scripts.meaning_vectors')


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


def main() -> None:
    """Downloads a GloVe embedding and writes it in GloVe text format (no gensim)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--out', required=True, type=Path, help='Destination .txt (GloVe format).')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='gensim-data GloVe model name.')
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
    out, dim = provision_glove(args.out, vocab=keep, model=args.model, top=args.top, overwrite=True)

    print(f'{out}  (dim={dim})')
    print(f'→ set objective.meaning_source: {out}  and  objective.meaning_dim: {dim}')


if __name__ == '__main__':
    main()
