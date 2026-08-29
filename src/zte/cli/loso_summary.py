"""`zte-loso-summary` -- aggregate a LOSO sweep into its honest held-out trend (see docs/EVALUATION.md)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from zte.logging_utils import configure_logging, get_logger

# A run records its scoreboard here, so this is both what a sweep root is globbed for and what marks a
# path as already being a run directory rather than a root holding several.
_METRICS: Final[str] = 'evaluation/metrics.json'
"""Where a run records the metrics a LOSO fold row is read from, relative to the run directory."""

_LOG = get_logger('cli.loso_summary')

# Folds with POOLED retrieval below this failed to learn a subject-invariant sentence code; folds tend to either converge (>=0.10) or collapse (<0.01).
_CONVERGED_FLOOR: float = 0.10
_COLLAPSED_CEIL: float = 0.01


def _get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Reads a dotted path out of a nested dict, returning `default` on any missing link."""
    cur: Any = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _fold_row(metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Extracts the honest per-fold row from one run's `metrics.json`, or `None` if not a LOSO run."""
    holdout = _get(metrics, 'honesty.loso_holdout')
    if not holdout:
        return None
    ho = _get(metrics, 'scoreboard.held_out_retrieval', {}) or {}
    cat = _get(metrics, 'honesty.cross_subject_decode.targets.category', {}) or {}
    control = _get(metrics, 'scoreboard.content_probe', {}) or _get(
        metrics, 'scoreboard.lift_over_raw.content_probe', {}
    )
    pooled = float(_get(metrics, 'sentence_retrieval.top1', float('nan')))
    return {
        'holdout': holdout,
        'n_words': _get(metrics, 'scoreboard.held_out_geometry.n_words'),
        'pooled_top1': round(pooled, 4),
        'held_out_top1': round(float(ho.get('top1', float('nan'))), 4),
        'held_out_chance': round(float(ho.get('chance_top1', float('nan'))), 4),
        'held_out_lift': round(float(ho.get('lift_top1', float('nan'))), 4),
        'held_out_rank_pct': round(float(ho.get('rank_percentile', float('nan'))), 3),
        'category_decode': round(float(cat.get('mean', float('nan'))), 3),
        'category_chance': round(float(cat.get('chance', float('nan'))), 3),
        'category_above_chance': bool(cat.get('above_chance', False)),
        'calibration_lift': round(float(_get(metrics, 'honesty.calibration.mean_lift', float('nan'))), 3),
        'who_variance': round(float(_get(metrics, 'neurons.who_variance', float('nan'))), 3),
        'same_word_gap': round(float(_get(metrics, 'emergence.cross_subject.same_word.gap', float('nan'))), 4),
        'content_probe_passes': bool(control.get('passes', False)) if control else None,
        'converged': pooled >= _CONVERGED_FLOOR,
        'collapsed': pooled < _COLLAPSED_CEIL,
    }


def fold_metrics(experiments: str | Path | Sequence[str | Path]) -> list[Path]:
    """Every fold's `metrics.json` under one or more roots, each of which may be a sweep root or a run directory.

    Note:
        Naming run directories one by one is how a sweep of one arm is summarised without averaging in its
        siblings: a sweep root holds every arm trained into it, and this command keys folds on the holdout
        alone, so a shared root would silently pool three alignment levels into one trend.

    Args:
        experiments (str | Path | Sequence[str | Path]): Sweep roots, run directories, or a mix.

    Returns:
        list[Path]: The `evaluation/metrics.json` paths found, deduplicated and sorted.
    """
    roots = [experiments] if isinstance(experiments, str | Path) else list(experiments)
    found: set[Path] = set()
    for entry in roots:
        root = Path(entry)
        if (direct := root / _METRICS).is_file():
            found.add(direct.resolve())
            continue

        found.update(path.resolve() for path in root.glob(f'*/{_METRICS}'))

    return sorted(found)


def collect_folds(experiments: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    """Reads every LOSO fold under one or more roots into honest rows, sorted by held-out subject."""
    rows: list[dict[str, Any]] = []
    for metrics_path in fold_metrics(experiments):
        try:
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning('Skipping %s: %r', metrics_path, exc)
            continue
        row = _fold_row(metrics)
        if row is not None:
            rows.append(row)
    return sorted(rows, key=lambda r: str(r['holdout']))


def _stats(values: list[float]) -> dict[str, float]:
    """Mean/std/min/max of a list, tolerating non-finite entries (dropped).

    Note:
        The spread is the n-1 sample sd: twelve folds are a sample of the subjects a clinical claim has to
        generalise to, and a population sd under-reports it by about 4% at that n.
    """
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return {'mean': float('nan'), 'std': float('nan'), 'min': float('nan'), 'max': float('nan')}
    return {
        'mean': statistics.mean(finite),
        'std': statistics.stdev(finite) if len(finite) > 1 else 0.0,
        'min': min(finite),
        'max': max(finite),
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates fold rows into the honest sweep-level verdict."""
    n = len(rows)
    held = [r['held_out_lift'] for r in rows]
    above = sum(1 for r in rows if r['held_out_top1'] > r['held_out_chance'])
    return {
        'n_folds': n,
        'held_out_lift': _stats(held),
        'pooled_top1': _stats([r['pooled_top1'] for r in rows]),
        'held_out_above_chance_folds': above,
        'category_decode': _stats([r['category_decode'] for r in rows]),
        'category_above_chance_folds': sum(1 for r in rows if r['category_above_chance']),
        'calibration_lift': _stats([r['calibration_lift'] for r in rows]),
        'converged_folds': sum(1 for r in rows if r['converged']),
        'collapsed_folds': sum(1 for r in rows if r['collapsed']),
        'content_probe_pass_folds': sum(1 for r in rows if r['content_probe_passes']),
    }


def render_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Renders the honest LOSO trend as a Markdown report."""
    hl = summary['held_out_lift']
    lines = [
        '# LOSO sweep — the honest trend',
        '',
        f'**{summary["n_folds"]} held-out subjects.** The headline is the held-out retrieval lift over '
        'chance (retrieval among the never-seen subject alone), NOT the pooled sentence retrieval, which '
        'is inflated by the training subjects.',
        '',
        f'- Held-out retrieval lift over chance: **{hl["mean"]:+.4f} ± {hl["std"]:.4f}** '
        f'(range {hl["min"]:+.4f} … {hl["max"]:+.4f})',
        f'- Folds beating chance: **{summary["held_out_above_chance_folds"]}/{summary["n_folds"]}**',
        f'- Held-out category decode: **{summary["category_decode"]["mean"]:.3f}** '
        f'({summary["category_above_chance_folds"]}/{summary["n_folds"]} folds above chance)',
        f'- Anchor-calibration lift: **{summary["calibration_lift"]["mean"]:+.3f}** '
        '(a new brain snapped into the shared frame from anchor words)',
        f'- Training convergence: **{summary["converged_folds"]} converged**, '
        f'**{summary["collapsed_folds"]} collapsed** (of {summary["n_folds"]}) — bimodal instability if both > 0.',
        f'- Content-probe positive control passing: **{summary["content_probe_pass_folds"]}/{summary["n_folds"]}**.',
        '',
        '| held-out | n_words | pooled Top-1 | held-out Top-1 | held-out lift | rank% | category | calib lift | who-var | conv? |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for r in rows:
        conv = 'yes' if r['converged'] else ('COLLAPSE' if r['collapsed'] else 'partial')
        lines.append(
            f'| {r["holdout"]} | {r["n_words"]} | {r["pooled_top1"]:.4f} | {r["held_out_top1"]:.4f} | '
            f'{r["held_out_lift"]:+.4f} | {r["held_out_rank_pct"]:.3f} | '
            f'{r["category_decode"]:.3f}{"✓" if r["category_above_chance"] else "·"} | '
            f'{r["calibration_lift"]:+.3f} | {r["who_variance"]:.3f} | {conv} |'
        )
    return '\n'.join(lines) + '\n'


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Writes the per-fold rows as CSV without a pandas dependency."""
    import csv

    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    """Parses the `zte-loso-summary` command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Aggregate a LOSO sweep into its honest held-out trend.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        type=str,
        default=['res/experiments/loso'],
        help='Sweep roots holding per-fold run folders, or the run folders themselves. Name them one by one to '
        'summarise a single arm: folds are keyed on the holdout alone, so a shared root pools every arm in it.',
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Markdown output path (default: <first --experiments>/LOSO_SUMMARY.md). A .csv is written alongside.',
    )
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def main() -> None:
    """Entry point for the `zte-loso-summary` console script."""
    args = parse_arguments()
    configure_logging(args.log_level)
    rows = collect_folds(args.experiments)
    if not rows:
        named = ', '.join(str(path) for path in args.experiments)
        raise SystemExit(f'No LOSO folds found under {named} (need {_METRICS}).')

    summary = summarise(rows)
    out = Path(args.out) if args.out else Path(args.experiments[0]) / 'LOSO_SUMMARY.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(rows, summary), encoding='utf-8')
    _write_csv(rows, out.with_suffix('.csv'))

    _LOG.info('LOSO summary (%d folds) written to %s', len(rows), out)
    print(render_markdown(rows, summary))


if __name__ == '__main__':
    main()
