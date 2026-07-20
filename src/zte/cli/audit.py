"""`zte-audit` — quantify how entangled ZTE's factors are, *before* fighting them.

Builds (or loads) the word-level metadata table exactly as a training run would, then runs the confound audit from
`zte.evaluation.audit.confound`: the decisive task<->stimulus overlap query, the nuisance→content bleed table,
the behaviour<->lexical table, and the full association matrix. Writes a Markdown report and a JSON sidecar.

Examples::

    zte-audit --synthetic                         # smoke-audit on a fake tree
    zte-audit --config experiments/exp8_*.yaml    # audit the real dataset a run uses
    zte-audit --root /path/to/zuco --out docs/confound_audit.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
        '--out', type=Path, default='docs/confound_audit.md', help='Output Markdown path.'
    )
    parser.add_argument(
        '--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    )

    return parser.parse_args()


def _dataset_config(args: argparse.Namespace) -> DatasetConfig:
    """Resolves a DatasetConfig from --config, --root, or --synthetic."""
    if args.config:
        cfg = ZTEConfig.from_yaml(args.config).dataset
    else:
        cfg = DatasetConfig()
    if args.synthetic:
        cfg.root = synthetic_root(cfg.tasks, show_progress=False)
    elif args.root:
        cfg.root = args.root
    return cfg


def main() -> None:
    """Entry point for the `zte-audit` console script."""
    args = parse_arguments()
    configure_logging(args.log_level)

    cfg = _dataset_config(args)
    _LOG.info('Building dataset from %s ...', cfg.root)
    dataset = ZuCoDataset(cfg).build(show_progress=False)

    report = confound_report(dataset.words)
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(report), encoding='utf-8')
    write_json(out.with_suffix('.json'), report)

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
