"""Provision frozen word-meaning vectors for Unit A (`objective.meaning_source`).

The meaning-distillation target must carry real semantics to attack `content = 0%`; the built-in hash fallback only verifies the mechanism.
This script downloads a standard static embedding via `gensim`'s downloader and writes it in the GloVe text format (`word v1 v2 …` per line)
that `zte.data.meaning.build_meaning_matrix` consumes.

Restricting to the most frequent `--top` words keeps the file small; a `--vocab-from` config restricts further to the words that actually
occur in that dataset (the ZuCo vocabulary is tiny, so this yields a file of a few thousand rows).

Examples::

    # ~50k most frequent GloVe-300 vectors (a few tens of MB):
    python scripts/build_meaning_vectors.py --out res/vectors/glove.300d.txt

    # Only the words in the ZuCo corpus the SOTA config uses (smallest, exact):
    python scripts/build_meaning_vectors.py --out res/vectors/zuco.300d.txt \
        --vocab-from experiments/sota_loso.yaml --root res/data/zuco_extracted

Requires the optional dependency `gensim` (`pip install gensim`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    """Downloads an embedding and writes it in GloVe text format."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--out', required=True, type=Path, help='Destination .txt (GloVe format).')
    parser.add_argument(
        '--model', default='glove-wiki-gigaword-300', help='gensim-downloader model name.'
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

    try:
        import gensim.downloader as api  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise SystemExit('gensim is required: pip install gensim') from exc

    _LOG.info('Downloading %s via gensim (first run caches it) …', args.model)
    kv = api.load(args.model)
    keep = _vocab_from_config(args.vocab_from, args.root) if args.vocab_from else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open('w', encoding='utf-8') as fh:
        for i, word in enumerate(kv.index_to_key):
            if args.top and keep is None and i >= args.top:
                break
            if keep is not None and word.lower() not in keep:
                continue
            vec = ' '.join(f'{v:.5f}' for v in kv[word])
            fh.write(f'{word} {vec}\n')
            written += 1
    _LOG.info('Wrote %d vectors (dim %d) to %s', written, kv.vector_size, args.out)
    print(f'{args.out}  (dim={kv.vector_size}, rows={written})')
    print(
        f'→ set objective.meaning_source: {args.out}  and  objective.meaning_dim: {kv.vector_size}'
    )


if __name__ == '__main__':
    main()
