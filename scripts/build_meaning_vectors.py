"""Provision frozen GloVe word-meaning vectors for `objective.meaning_source` (`zte-run --meaning static` does this
per-run).

Available --model names (the trailing number is the dimension):
    glove-wiki-gigaword-50 / -100 / -200 / -300, glove-twitter-25 / -50 / -100 / -200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from zte.data.targets.glove import DEFAULT_MODEL, provision_glove
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('scripts.meaning_vectors')


def _vocab_from_config(config_path: str, root: str | None) -> set[str] | None:
    """Returns the lowercased word set of the dataset a config defines, or `None` on failure."""
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
        _LOG.warning('Could not read vocab from %s (%s); writing the full top-N instead.', config_path, exc)
        return None


def main() -> None:
    """Downloads a GloVe embedding and writes it in GloVe text format (no gensim)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, type=Path, help='Destination .txt (GloVe format).')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='gensim-data GloVe model name.')
    parser.add_argument('--top', type=int, default=50000, help='Keep the N most frequent words (0 = all).')
    parser.add_argument('--vocab-from', default=None, help='Experiment YAML to restrict the vocabulary to.')
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
