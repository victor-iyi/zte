"""Aggregation across seeds, folds and levers: the mean +/- sd tables a reviewer asks for."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from zte.evaluation.analysis.collect import Study
from zte.logging_utils import get_logger

_LOG = get_logger('evaluation.analysis.aggregate')

# The metrics every summary table carries, in the order a reader should meet them: the honest headline first,
# the confound-resistant version next, and the diagnostics after.
HEADLINE_METRICS: tuple[str, ...] = (
    'held_out_rank_percentile',
    'held_out_top1',
    'held_out_lift',
    'stratified_top1',
    'stratified_rank_percentile',
    'rescoring_top1',
    'rescoring_rank_percentile',
    'word_lift',
    'effective_rank_ratio',
    'generation_delta',
)

# The three comparisons the decoder ablation table exists to make, each a single lever with everything else held.
ABLATION_LEVERS: tuple[tuple[str, str], ...] = (
    ('frontend', 'raw conformer vs band-power MLP'),
    ('spatial_encoding', 'spherical harmonics vs standard channel indexing'),
    ('invariance', 'invariance recipe ON vs OFF'),
)


def multi_seed_table(
    study: Study,
    *,
    group: Sequence[str] = ('arm',),
    metrics: Sequence[str] = HEADLINE_METRICS,
    min_seeds: int = 1,
) -> pd.DataFrame:
    """Collapses repeated runs of one arm into mean, sd and a bootstrap interval over seeds.

    Note:
        Run-to-run drift on this project has been the size of the effect -- an arm that scored 4 hits in 700 scored 2
        on an identical re-run -- so a single-seed number is not a result. The interval here is over *seeds*, which
        is the axis that moved; the per-query bootstrap inside each run answers a different question and both are
        reported.

    Args:
        study (Study): The collected study.
        group (Sequence[str], optional): Columns identifying one arm. Defaults to ('arm',).
        metrics (Sequence[str], optional): Metrics to summarise. Defaults to `HEADLINE_METRICS`.
        min_seeds (int, optional): Arms with fewer runs are still reported, with `sd` as `nan` and `n_seeds`
            naming how thin the evidence is. Defaults to 1.

    Returns:
        pd.DataFrame: One row per arm with `<metric>_mean`, `<metric>_sd`, `<metric>_lo`, `<metric>_hi`
            and `n_seeds`.
    """
    runs = study.runs
    keys = [g for g in group if g in runs.columns]
    if runs.empty or not keys:
        return pd.DataFrame()

    present = [m for m in metrics if m in runs.columns]
    rows: list[dict[str, Any]] = []
    for name, block in runs.groupby(keys, dropna=False):
        row: dict[str, Any] = dict(zip(keys, name if isinstance(name, tuple) else (name,), strict=True))
        row['n_seeds'] = int(block['seed'].nunique()) if 'seed' in block else len(block)
        row['n_runs'] = int(len(block))
        row['seeds'] = sorted({int(s) for s in block['seed'].dropna()}) if 'seed' in block else []
        for metric in present:
            values = pd.to_numeric(block[metric], errors='coerce').dropna().to_numpy()
            row.update({f'{metric}_{k}': v for k, v in _summary(values).items()})
        if row['n_runs'] >= min_seeds:
            rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def loso_table(study: Study, *, metric: str = 'held_out_rank_percentile') -> pd.DataFrame:
    """Returns the per-fold trend for every arm: one row per (arm, held-out subject), averaged over seeds.

    Args:
        study (Study): The collected study.
        metric (str, optional): The fold metric to summarise. Defaults to 'held_out_rank_percentile'.

    Returns:
        pd.DataFrame: `arm`, `holdout`, `mean`, `sd`, `n_seeds` and the raw per-seed values.
    """
    folds = study.folds
    if folds.empty or metric not in folds.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (arm, holdout), block in folds.groupby(['arm', 'holdout'], dropna=False):
        values = pd.to_numeric(block[metric], errors='coerce').dropna().to_numpy()
        rows.append(
            {
                'arm': arm,
                'holdout': holdout,
                'metric': metric,
                'values': [float(v) for v in values],
                'n_seeds': int(values.size),
                **_summary(values),
            }
        )
    return pd.DataFrame(rows).sort_values(['arm', 'holdout']).reset_index(drop=True)


def feature_ablation_table(study: Study, *, metrics: Sequence[str] = HEADLINE_METRICS) -> pd.DataFrame:
    """The single-lever comparison table: what each design choice is worth, with everything else held.

    Note:
        The rows are matched only on the lever named in each block, so a level whose runs differ in some other
        respect will show that difference here as if it were the lever's. `n_runs` and `arms` are carried so that is
        visible; `zte-ablate` is the way to generate genuinely matched pairs.

    Args:
        study (Study): The collected study.
        metrics (Sequence[str], optional): Metrics to compare. Defaults to `HEADLINE_METRICS`.

    Returns:
        pd.DataFrame: One row per (lever, level) with the mean +/- sd of each metric and the run count, plus a
            `delta_vs_reference` column giving the lift of every level over the lever's first level.
    """
    runs = _with_invariance(study.runs)
    if runs.empty:
        return pd.DataFrame()

    present = [m for m in metrics if m in runs.columns]
    rows: list[dict[str, Any]] = []
    for lever, question in ABLATION_LEVERS:
        if lever not in runs.columns:
            continue
        block = runs[runs[lever].notna()]
        levels = list(dict.fromkeys(block[lever].astype(str)))
        if len(levels) < 2:
            continue
        reference: dict[str, float] = {}
        for i, level in enumerate(levels):
            cell = block[block[lever].astype(str) == level]
            row: dict[str, Any] = {
                'lever': lever,
                'question': question,
                'level': level,
                'is_reference': i == 0,
                'n_runs': int(len(cell)),
                'arms': sorted({str(a) for a in cell['arm']}),
            }
            for metric in present:
                values = pd.to_numeric(cell[metric], errors='coerce').dropna().to_numpy()
                stats = _summary(values)
                row.update({f'{metric}_{k}': v for k, v in stats.items()})
                if i == 0:
                    reference[metric] = stats['mean']
                row[f'{metric}_delta'] = _delta(stats['mean'], reference.get(metric))
            rows.append(row)
    return pd.DataFrame(rows)


def within_task_table(study: Study) -> pd.DataFrame:
    """Long-form within-task retrieval: one row per (run, task) with its own chance level.

    Note:
        Inside one reading task the passage set is fixed, so a lift that survives here cannot be a passage or task
        shortcut. The pool is smaller, so its chance level is higher and is reported beside every number.

    Args:
        study (Study): The collected study.

    Returns:
        pd.DataFrame: `run`, `arm`, `task`, `top1`, `chance`, `lift`, `rank_percentile`, `n_candidates`.
    """
    runs = study.runs
    if runs.empty:
        return pd.DataFrame()

    tasks = sorted({c.split('_')[1] for c in runs.columns if c.startswith('within_') and c.endswith('_top1')})
    rows: list[dict[str, Any]] = []
    for _, run in runs.iterrows():
        for task in tasks:
            top1 = run.get(f'within_{task}_top1')
            if top1 is None or (isinstance(top1, float) and np.isnan(top1)):
                continue
            chance = run.get(f'within_{task}_chance')
            rows.append(
                {
                    'run': run['run'],
                    'arm': run.get('arm'),
                    'seed': run.get('seed'),
                    'holdout': run.get('holdout'),
                    'task': task,
                    'top1': float(top1),
                    'chance': None if chance is None else float(chance),
                    'lift': None if chance is None else float(top1) - float(chance),
                    'rank_percentile': run.get(f'within_{task}_rank_percentile'),
                    'n_candidates': run.get(f'within_{task}_n_candidates'),
                }
            )
    return pd.DataFrame(rows)


def control_table(study: Study) -> pd.DataFrame:
    """Per-condition generation scores, averaged over sentences -- the headline against every control.

    Args:
        study (Study): The collected study.

    Returns:
        pd.DataFrame: One row per (run, condition) with the mean of each per-sentence metric and the paired delta
            against the real decode.
    """
    gens = study.generations
    if gens.empty or 'condition' not in gens.columns:
        return pd.DataFrame()

    score_columns = [c for c in gens.columns if c.startswith('score_')]
    grouped = gens.groupby(['run', 'condition'], dropna=False)[score_columns].mean().reset_index()
    real = grouped[grouped['condition'] == 'hypothesis'].set_index('run')

    # Signed hypothesis-minus-control, so a positive delta always means the brain-driven decode won -- including on
    # WER, where the raw metric runs the other way.
    for column in score_columns:
        if column not in real.columns:
            continue
        hyp = grouped['run'].map(real[column])
        grouped[f'{column}_delta'] = grouped[column] - hyp if column.endswith('wer') else hyp - grouped[column]
    return grouped


def summary_markdown(study: Study) -> str:
    """Renders the study's headline tables as Markdown, for the report beside the dashboard.

    Args:
        study (Study): The collected study.

    Returns:
        str: A Markdown document.
    """
    lines = ['# ZTE study summary', '']
    if study.is_empty:
        return '\n'.join([*lines, '_No evaluated run was found._'])

    lines += [
        f'Collected **{len(study.runs)}** run(s) from {", ".join(str(r) for r in study.roots)}.',
        '',
        _synthetic_warning(study),
        '',
        '## Headline, averaged over seeds',
        '',
    ]
    seeds = multi_seed_table(study)
    lines.append(
        _markdown_table(seeds, ['arm', 'n_seeds'], ('held_out_rank_percentile', 'held_out_top1', 'held_out_lift'))
    )

    lines += ['', '## Feature ablation -- one lever at a time', '']
    ablation = feature_ablation_table(study)
    if ablation.empty:
        lines.append('_No lever had two levels among the collected runs._')
    else:
        lines.append(
            _markdown_table(
                ablation,
                ['question', 'level', 'n_runs'],
                ('held_out_rank_percentile', 'held_out_top1', 'stratified_top1'),
            )
        )

    lines += ['', '## Within-task pools -- passage identity held fixed', '']
    within = within_task_table(study)
    if within.empty:
        lines.append('_No run reported a within-task pool._')
    else:
        agg = within.groupby(['arm', 'task'], dropna=False)[['top1', 'chance', 'lift']].mean().reset_index()
        lines.append('| arm | task | Top-1 | chance | lift |')
        lines.append('| --- | --- | --- | --- | --- |')
        for _, row in agg.iterrows():
            lines.append(
                f'| {row["arm"]} | {row["task"]} | {_fmt(row["top1"])} | {_fmt(row["chance"])} | {_fmt(row["lift"])} |'
            )
    return '\n'.join(lines) + '\n'


def _with_invariance(runs: pd.DataFrame) -> pd.DataFrame:
    """Adds the composite `invariance` lever: whether any of the three label-free identity steps was on."""
    if runs.empty:
        return runs
    out = runs.copy()
    parts = [c for c in ('raw_align', 'subject_adapter', 'identity_orthogonality') if c in out.columns]
    if not parts:
        return out

    def is_on(row: pd.Series) -> str:
        align = str(row.get('raw_align') or 'none') != 'none'
        adapter = bool(row.get('subject_adapter'))
        orth = float(row.get('identity_orthogonality') or 0.0) > 0.0
        return 'on' if (align or adapter or orth) else 'off'

    out['invariance'] = out.apply(is_on, axis=1)
    return out


def _summary(values: np.ndarray) -> dict[str, float]:
    """Mean, sd and a percentile bootstrap interval over a handful of seeds."""
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {'mean': float('nan'), 'sd': float('nan'), 'lo': float('nan'), 'hi': float('nan'), 'n': 0.0}
    if clean.size == 1:
        point = float(clean[0])
        return {'mean': point, 'sd': float('nan'), 'lo': point, 'hi': point, 'n': 1.0}

    rng = np.random.default_rng(0)
    draws = rng.choice(clean, size=(2000, clean.size), replace=True).mean(axis=1)
    return {
        'mean': float(clean.mean()),
        'sd': float(clean.std(ddof=1)),
        'lo': float(np.percentile(draws, 2.5)),
        'hi': float(np.percentile(draws, 97.5)),
        'n': float(clean.size),
    }


def _delta(value: Any, reference: Any) -> float | None:
    """`value - reference` as a float, or `None` when either side is missing."""
    try:
        left, right = float(value), float(reference)
    except TypeError, ValueError:
        return None
    return None if not (np.isfinite(left) and np.isfinite(right)) else left - right


def _synthetic_warning(study: Study) -> str:
    """States plainly whether any collected run was synthetic, since the two are not the same evidence."""
    if 'real_data' not in study.runs.columns:
        return ''
    synthetic = int((~study.runs['real_data'].astype(bool)).sum())
    if synthetic == 0:
        return '_Every run below is on real ZuCo._'
    if synthetic == len(study.runs):
        return '> **Every run below is SYNTHETIC.** Nothing here is a result; it is a wiring check.'
    return (
        f'> **{synthetic} of {len(study.runs)} runs below are SYNTHETIC** and are not results. '
        'Filter on `real_data` before reading any number as evidence.'
    )


def _markdown_table(frame: pd.DataFrame, keys: list[str], metrics: Sequence[str]) -> str:
    """Renders `mean +/- sd` cells for a few metrics of a summary frame."""
    if frame.empty:
        return '_(empty)_'
    columns = [k for k in keys if k in frame.columns]
    shown = [m for m in metrics if f'{m}_mean' in frame.columns]
    header = '| ' + ' | '.join([*columns, *shown]) + ' |'
    rule = '| ' + ' | '.join(['---'] * (len(columns) + len(shown))) + ' |'
    lines = [header, rule]
    for _, row in frame.iterrows():
        cells = [str(row[c]) for c in columns]
        for metric in shown:
            mean, sd = row.get(f'{metric}_mean'), row.get(f'{metric}_sd')
            cells.append(_fmt(mean) if sd is None or not np.isfinite(sd) else f'{_fmt(mean)} ± {_fmt(sd)}')
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def _fmt(value: Any) -> str:
    """Formats a metric for a Markdown cell, or an em dash when it is missing."""
    try:
        number = float(value)
    except TypeError, ValueError:
        return '—'
    return '—' if not np.isfinite(number) else f'{number:.4f}'
