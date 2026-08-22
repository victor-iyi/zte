"""`zte-audit` -- quantify how entangled ZTE's factors are, before designing invariance against them."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from zte.cli.support.datasets import synthetic_root
from zte.cli.support.io import write_json
from zte.config import DatasetConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.evaluation.audit.confound import confound_report, render_markdown
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.audit')


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for `zte-audit`."""
    parser = argparse.ArgumentParser(
        description='Audit factor confounds in the ZuCo word table before designing invariance.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument('--config', type=str, help='Experiment YAML; builds the dataset it defines.')
    src.add_argument('--root', type=str, help='Directory of ZuCo .mat files to audit directly.')

    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Fabricate a small schema-faithful tree and audit it.',
    )
    parser.add_argument(
        '--piece-oracle',
        action='store_true',
        dest='piece_oracle',
        help='Also score the sub-word piece signatures the gallery gives away. Model-free like the rest of this '
        'audit -- it reads the corpus, not a checkpoint -- so it answers what a token-level arm would have to '
        'clear before one is trained.',
    )
    parser.add_argument(
        '--tokenizer',
        default=None,
        help="Tokeniser the piece oracle spells the gallery with. Defaults to the decoder's own LM.",
    )
    parser.add_argument('--out', type=Path, default='docs/confound_audit.md', help='Output Markdown path.')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    return parser.parse_args()


def _dataset_config(args: argparse.Namespace) -> DatasetConfig:
    """Resolves a DatasetConfig from --config, --root, or --synthetic."""
    if args.config:
        cfg = ZTEConfig.from_yaml(args.config).dataset
    else:
        cfg = DatasetConfig()
    if args.synthetic:
        cfg.root = str(synthetic_root(cfg.tasks, show_progress=False))
    elif args.root:
        cfg.root = str(args.root)
    return cfg


def _piece_oracle(dataset: ZuCoDataset, cfg: DatasetConfig, tokenizer: str | None) -> dict[str, Any]:
    """Scores what the gallery's sub-word piece signatures give away, with no model involved.

    Note:
        The oracle is a property of the corpus, not of any encoder, so it needs no checkpoint and answers the
        question a token-level arm has to answer before it is worth training: how much of sentence identity does
        spelling hand over for free.
    """
    from zte.config import DecoderConfig
    from zte.data.targets.tokens import build_token_alignment
    from zte.data.torch_dataset import ZuCoTorchDataset
    from zte.evaluation.audit.rebaseline import piece_profile_report

    torch_ds = ZuCoTorchDataset(dataset)
    source = tokenizer or DecoderConfig().lm_source
    alignment = build_token_alignment(
        torch_ds.ordered_texts(),
        torch_ds.ordered_words(),
        source,
        cache_dir=str(Path(cfg.cache_dir) / 'tokens'),
    )
    block = piece_profile_report(alignment.word_pieces)
    block['tokenizer'] = source
    block['alignment_coverage'] = alignment.coverage
    block['n_gallery'] = int(alignment.word_pieces.shape[0])
    _LOG.info(
        'Piece oracle over %d gallery sentences (%s, coverage %.3f): gate %s at Top-1 %.4f (%.2f bits), '
        'ceiling %s at Top-1 %.4f.',
        block['n_gallery'],
        source,
        alignment.coverage,
        block['gate_signature'],
        block['gate_top1'],
        block['gate_bits'],
        block['ceiling_signature'],
        block['ceiling_top1'],
    )

    return block


def main() -> None:
    """Entry point for the `zte-audit` console script."""
    args = parse_arguments()
    configure_logging(args.log_level)

    # Build the word table exactly as a training run would.
    cfg = _dataset_config(args)
    _LOG.info('Building dataset from %s ...', cfg.root)
    dataset = ZuCoDataset(cfg).build(show_progress=False)

    report = confound_report(dataset.words)
    if args.piece_oracle:
        report['piece_oracle'] = _piece_oracle(dataset, cfg, args.tokenizer)
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(report), encoding='utf-8')
    write_json(out.with_suffix('.json'), report)

    # Headline the decisive task<->stimulus overlap.
    ts = report['task_stimulus']
    if ts.get('available'):
        _LOG.info(
            'task↔stimulus: %d/%d shared, V=%.3f — %s',
            ts['n_shared_across_tasks'],
            ts['n_stimuli'],
            ts['cramers_v_task_stimulus'],
            'FULLY CONFOUNDED' if ts['fully_confounded'] else 'partial',
        )
    _LOG.info('Confound audit written to %s (+ .json)', out)
    print(out)


if __name__ == '__main__':
    main()
