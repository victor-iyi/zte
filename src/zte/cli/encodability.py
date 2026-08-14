"""`zte-encodability` -- which held-out brains a LOSO sweep encodes well, and what predicts it (docs/EVALUATION.md)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from zte.cli.loso_summary import _get
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.encodability')

# Spearman |rho| above this is ~p<0.05 (two-tailed) at n=12; below it, treat as noise.
_RHO_SIGNIFICANT_N12: float = 0.587

# Outcomes (higher = better encoded) and the properties that might explain them (docs/EVALUATION.md).
_OUTCOMES: tuple[tuple[str, str], ...] = (
    ('held_out_rank_pct', "where the held-out subject's correct match ranks (1=first)"),
    ('category_decode', 'held-out task/category decode accuracy'),
    ('calibration_lift', 'how much anchor calibration rescues the held-out subject'),
)
_PROPERTIES: tuple[tuple[str, str], ...] = (
    ('n_words', 'held-out subject word count (data quantity)'),
    ('omission_rate', 'fraction of words the subject skipped (data sparsity)'),
    ('who_variance', 'identity variance the run left in the space (run convergence)'),
    ('held_anisotropy', 'how collapsed the held-out embeddings are (cone = bad)'),
    ('task_variance', 'variance the held-out geometry spent on the task axis'),
)


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation of two equal-length lists, ignoring index pairs with a NaN.

    Uses average ranks for ties, so it is exact for the small subject counts a LOSO sweep produces.
    """
    pairs = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3:
        return float('nan')
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = math.sqrt(sum((v - mx) ** 2 for v in rx))
    vy = math.sqrt(sum((v - my) ** 2 for v in ry))
    return cov / (vx * vy) if vx * vy else float('nan')


def _subject_row(metrics: dict[str, Any], manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    """Builds one held-out subject's outcome+property row from its fold artefacts."""
    holdout = _get(metrics, 'honesty.loso_holdout')
    if not holdout:
        return None
    geom = _get(metrics, 'scoreboard.held_out_geometry', {}) or {}
    ho = _get(metrics, 'scoreboard.held_out_retrieval', {}) or {}
    cat = _get(metrics, 'honesty.cross_subject_decode.targets.category', {}) or {}
    omission = _get(manifest or {}, f'dataset.omission_rate_by_subject.{holdout}')
    return {
        'holdout': holdout,
        # outcomes
        'held_out_rank_pct': float(ho.get('rank_percentile', float('nan'))),
        'held_out_lift': float(ho.get('lift_top1', float('nan'))),
        'category_decode': float(cat.get('mean', float('nan'))),
        'calibration_lift': float(_get(metrics, 'honesty.calibration.mean_lift', float('nan'))),
        # properties
        'n_words': float(geom.get('n_words', float('nan'))),
        'omission_rate': float(omission) if omission is not None else float('nan'),
        'who_variance': float(_get(metrics, 'neurons.who_variance', float('nan'))),
        'held_anisotropy': float(geom.get('anisotropy', float('nan'))),
        'task_variance': float(geom.get('task_variance') or float('nan')),
    }


def collect_subjects(experiments: str | Path) -> list[dict[str, Any]]:
    """Reads every LOSO fold under a directory into per-held-out-subject rows.

    When several seeds share a held-out subject their rows are averaged, so a subject appears once with the
    mean of its outcomes -- the multi-seed way to ask "is this brain hard" rather than "was this run".
    """
    root = Path(experiments)
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for metrics_path in sorted(root.glob('*/evaluation/metrics.json')):
        try:
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning('Skipping %s: %r', metrics_path, exc)
            continue
        manifest_path = metrics_path.parent.parent / 'manifest.json'
        manifest = None
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            except OSError, json.JSONDecodeError:
                manifest = None
        row = _subject_row(metrics, manifest)
        if row is not None:
            by_subject.setdefault(row['holdout'], []).append(row)

    merged: list[dict[str, Any]] = []
    for holdout, rows in by_subject.items():
        keys = [k for k in rows[0] if k != 'holdout']
        avg = {'holdout': holdout, 'n_seeds': len(rows)}
        for k in keys:
            vals = [r[k] for r in rows if not math.isnan(r[k])]
            avg[k] = sum(vals) / len(vals) if vals else float('nan')
        merged.append(avg)
    return sorted(merged, key=lambda r: r.get('held_out_rank_pct', 0.0), reverse=True)


def correlate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Rank-correlates every outcome against every property across subjects."""
    table: dict[str, dict[str, float]] = {}
    for outcome, _ in _OUTCOMES:
        col = [r.get(outcome, float('nan')) for r in rows]
        table[outcome] = {prop: _spearman(col, [r.get(prop, float('nan')) for r in rows]) for prop, _ in _PROPERTIES}
    return table


def render_markdown(rows: list[dict[str, Any]], table: dict[str, dict[str, float]]) -> str:
    """Renders the encodability report: the per-subject ranking, then what predicts it."""
    multi = any(r.get('n_seeds', 1) > 1 for r in rows)
    lines = [
        '# What makes a brain easy to encode?',
        '',
        f'{len(rows)} held-out subjects'
        + (' (outcomes averaged over seeds)' if multi else ' (single seed — see caveat)')
        + '. Subjects are ordered by how well their held-out embeddings retrieve (rank-percentile).',
        '',
        '## Per-subject encodability',
        '',
        '| subject | rank% | category | calib lift | n_words | omission | who-var(run) | held-aniso |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for r in rows:
        lines.append(
            f'| {r["holdout"]} | {r.get("held_out_rank_pct", float("nan")):.3f} | '
            f'{r.get("category_decode", float("nan")):.3f} | {r.get("calibration_lift", float("nan")):+.3f} | '
            f'{r.get("n_words", float("nan")):.0f} | {r.get("omission_rate", float("nan")):.3f} | '
            f'{r.get("who_variance", float("nan")):.3f} | {r.get("held_anisotropy", float("nan")):.3f} |'
        )

    lines += [
        '',
        f'## What predicts encodability (Spearman rho; **bold** = |rho| > {_RHO_SIGNIFICANT_N12:.2f}, ~p<0.05 at n=12)',
        '',
        '| outcome ↓ / property → | ' + ' | '.join(p for p, _ in _PROPERTIES) + ' |',
        '| --- | ' + ' | '.join('---' for _ in _PROPERTIES) + ' |',
    ]
    for outcome, _ in _OUTCOMES:
        cells = []
        for prop, _ in _PROPERTIES:
            rho = table[outcome][prop]
            cell = 'n/a' if math.isnan(rho) else f'{rho:+.2f}'
            if not math.isnan(rho) and abs(rho) > _RHO_SIGNIFICANT_N12:
                cell = f'**{cell}**'
            cells.append(cell)
        lines.append(f'| {outcome} | ' + ' | '.join(cells) + ' |')

    lines += [
        '',
        '### Property glossary',
        *[f'- `{p}` — {desc}' for p, desc in _PROPERTIES],
        '',
        '> Caveat: at ~12 subjects and one seed, training instability and subject identity are '
        'confounded. Re-run the sweep at several seeds to tell a genuinely hard brain (hard at every '
        'seed) from an unlucky run (hard at some).',
    ]
    return '\n'.join(lines) + '\n'


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Writes the per-subject table as CSV without a pandas dependency."""
    import csv

    if not rows:
        return
    fields = ['holdout', 'n_seeds', *[k for k in rows[0] if k not in ('holdout', 'n_seeds')]]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    """Parses the `zte-encodability` command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyse which held-out brains a LOSO sweep encodes well, and what predicts it.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--experiments',
        type=str,
        default='res/experiments/loso',
        help='Directory holding the per-fold LOSO run folders.',
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Markdown output path (default: <experiments>/ENCODABILITY.md). A .csv is written alongside.',
    )
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def main() -> None:
    """Entry point for the `zte-encodability` console script."""
    args = parse_arguments()
    configure_logging(args.log_level)
    rows = collect_subjects(args.experiments)
    if len(rows) < 3:
        raise SystemExit(f'Need at least 3 held-out subjects under {args.experiments}; found {len(rows)}.')
    table = correlate(rows)
    out = Path(args.out) if args.out else Path(args.experiments) / 'ENCODABILITY.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(rows, table), encoding='utf-8')
    _write_csv(rows, out.with_suffix('.csv'))
    _LOG.info('Encodability analysis (%d subjects) written to %s', len(rows), out)
    print(render_markdown(rows, table))


if __name__ == '__main__':
    main()
