"""Assembles the collected study into one self-contained, offline HTML analysis page."""

from __future__ import annotations

import html
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from zte.evaluation.analysis import figures as F
from zte.evaluation.analysis.aggregate import (
    feature_ablation_table,
    loso_table,
    multi_seed_table,
    summary_markdown,
    within_task_table,
)
from zte.evaluation.analysis.collect import Study
from zte.logging_utils import get_logger

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

_LOG = get_logger('evaluation.analysis.dashboard')

# Sections, in the order a reader should meet them: is it real, then how big, then why, then what it wrote.
_SECTIONS: tuple[tuple[str, str], ...] = (
    ('headline', 'The honest headline'),
    ('folds', 'Every fold, every seed'),
    ('ablation', 'What each lever is worth'),
    ('confound', 'The confounds'),
    ('decoder', 'What the decoder wrote'),
    ('geometry', 'The space itself'),
    ('training', 'Training'),
)


def build_dashboard(
    study: Study,
    out_path: str | Path,
    *,
    title: str = 'ZTE — study analysis',
    montage_csv: str | None = None,
) -> Path:
    """Renders every panel into one HTML file that opens with no network and no server.

    Args:
        study (Study): The collected study.
        out_path (str | Path): Where the page is written.
        title (str, optional): Page title. Defaults to 'ZTE — study analysis'.
        montage_csv (str | None, optional): Montage CSV for the 3-D electrode map. Defaults to None.

    Returns:
        Path: The written path.

    Note:
        Plotly's script is inlined rather than fetched from a CDN. The page has to open from a Drive mirror on a
        machine that may be offline, and a chart that silently fails to render is worse than no chart.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    panels: dict[str, list[str]] = {key: [] for key, _ in _SECTIONS}
    first = True
    for section, figure, caption in _panels(study, montage_csv):
        rendered = _render(figure, include_js=first)
        if rendered is None:
            continue
        first = False
        panels[section].append(_card(rendered, caption))

    panels['headline'].insert(0, _card(_table_html(_headline_table(study)), _HEADLINE_CAPTION))
    panels['ablation'].append(_card(_table_html(_ablation_view(study)), _ABLATION_CAPTION))
    panels['confound'].append(_card(_table_html(_within_view(study)), _WITHIN_CAPTION))
    panels['folds'].append(_card(_table_html(_fold_view(study)), _FOLD_CAPTION))

    body = '\n'.join(_section(key, label, panels[key]) for key, label in _SECTIONS if panels[key])
    out.write_text(_document(title, _banner(study), body), encoding='utf-8')
    _LOG.info('Analysis dashboard written to %s (%.1f MB).', out, out.stat().st_size / 1e6)
    return out


def write_summary(study: Study, out_path: str | Path) -> Path:
    """Writes the Markdown companion to the dashboard, for a reader who wants the numbers without a browser.

    Args:
        study (Study): The collected study.
        out_path (str | Path): Where the Markdown is written.

    Returns:
        Path: The written path.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary_markdown(study), encoding='utf-8')
    return out


def write_tables(study: Study, out_dir: str | Path) -> list[Path]:
    """Writes every tidy frame as CSV, so the analysis can be redone in any other tool.

    Args:
        study (Study): The collected study.
        out_dir (str | Path): Directory for the CSVs.

    Returns:
        list[Path]: The files written.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {
        'runs': study.runs,
        'folds': study.folds,
        'probes': study.probes,
        'subjects': study.subjects,
        'history': study.history,
        'generations': study.generations,
        'rebaseline': study.rebaseline,
        'multi_seed': multi_seed_table(study),
        'feature_ablation': feature_ablation_table(study),
        'within_task': within_task_table(study),
        'loso': loso_table(study),
    }
    written: list[Path] = []
    for name, frame in frames.items():
        if frame is None or frame.empty:
            continue
        path = directory / f'{name}.csv'
        frame.to_csv(path, index=False)
        written.append(path)
    _LOG.info('Wrote %d analysis table(s) to %s.', len(written), directory)
    return written


# ---- Panels ---- #

_HEADLINE_CAPTION = (
    'Read <code>held_out_rank_percentile</code> first: it uses every query rather than only the ones that landed, '
    'and it is the only cell that generalises over the subject and the stimulus at once. Top-1 on 700 queries at '
    'chance 1/700 expects exactly one hit, so a Top-1 of 0.006 is three hits and noise.'
)
_ABLATION_CAPTION = (
    'One lever per block, everything else as it fell. <code>n_runs</code> is carried because a level backed by one '
    'run is not an ablation; use <code>zte-ablate</code> for genuinely matched pairs.'
)
_WITHIN_CAPTION = (
    'Inside one reading task the passage set is fixed, so a lift here cannot be a passage or task shortcut. The pool '
    'is smaller, so its own chance level is printed beside every number.'
)
_FOLD_CAPTION = (
    'Every fold at every seed. Run-to-run drift on this project has been the size of the effect, so a cell backed by '
    'one seed is a measurement and not yet a result.'
)


def _panels(study: Study, montage_csv: str | None) -> Iterator[tuple[str, 'Figure | None', str]]:
    """Yields `(section, figure, caption)` for every chart, skipping the ones with no data behind them."""
    builders: tuple[tuple[str, Callable[[], Any], str], ...] = (
        (
            'headline',
            lambda: F.metric_explorer(study),
            'Every headline behind one selector — start here, then read the confound section before believing any of it.',
        ),
        (
            'headline',
            lambda: F.headline_bars(study, 'held_out_rank_percentile'),
            'Rank percentile per arm, mean ± sd over seeds, with each seed drawn on its bar.',
        ),
        (
            'headline',
            lambda: F.headline_bars(study, 'held_out_top1'),
            'Top-1 per arm. Quoted alone this is not evidence of decoding -- see the confound section.',
        ),
        ('headline', lambda: F.bit_budget_bars(study), 'What the conditioning channel delivered against its ceiling.'),
        ('headline', lambda: F.seed_histogram(study), 'Every run as one observation; a bimodal shape is the finding.'),
        ('headline', lambda: F.bit_budget_pie(study), 'The 9.45 bits of sentence identity, and who supplies them.'),
        ('folds', lambda: F.loso_heatmap(study), 'Which brains a recipe reaches, and which it never does.'),
        ('folds', lambda: F.fold_spread(study), 'The fold-to-fold distribution; a bimodal arm is visible here.'),
        ('folds', lambda: F.subject_difficulty(study), 'Per-subject difficulty, pooled across runs.'),
        ('ablation', lambda: F.ablation_bars(study, 'held_out_rank_percentile'), 'Each lever on the honest headline.'),
        (
            'ablation',
            lambda: F.ablation_bars(study, 'stratified_top1'),
            'Each lever inside the length-matched gallery.',
        ),
        ('ablation', lambda: F.mechanism_matrix(study), 'Which levers each run actually had on, read off its config.'),
        ('confound', lambda: F.length_confound_scatter(study), 'The encoder against a length-only oracle.'),
        ('confound', lambda: F.length_leakage_bars(study), 'Length leakage before and after the projection.'),
        ('confound', lambda: F.within_task_bars(study), 'Within-task pools against their own chance level.'),
        ('confound', lambda: F.length_vs_score(study), 'Sentence length against decode quality, per condition.'),
        ('decoder', lambda: F.control_ladder(study), 'The decode against every brain-independent control.'),
        ('decoder', lambda: F.score_distributions(study), 'The whole per-sentence distribution, not its mean.'),
        ('decoder', lambda: F.text_overlap_heatmap(study), 'Where the decode landed, sentence by sentence.'),
        ('decoder', lambda: F.word_frequency_bars(study), 'The words emitted against the words asked for.'),
        ('geometry', lambda: F.probe_heatmap(study), 'What each representation carries.'),
        ('geometry', lambda: F.variance_budget_pie(study), 'Who versus what.'),
        ('geometry', lambda: F.identity_vs_content(study), 'Identity against content, sized by effective rank.'),
        ('geometry', lambda: F.metric_correlations(study), 'Whether the headline metrics are saying one thing.'),
        ('geometry', lambda: F.scalp_3d(montage_csv=montage_csv), 'The electrode geometry the encoder reads.'),
        ('training', lambda: F.learning_curves(study), 'Train and validation loss per epoch.'),
        ('training', lambda: F.mechanism_curves(study), 'Did each mechanism engage, or was it merely configured?'),
    )
    for section, build, caption in builders:
        try:
            figure = build()
        except (KeyError, ValueError, TypeError) as exc:
            _LOG.warning('Panel %r skipped (%r).', caption, exc)
            continue
        yield section, figure, caption


def _render(figure: 'Figure | None', *, include_js: bool) -> str | None:
    """Renders one figure to an HTML fragment, inlining plotly.js on the first call only."""
    if figure is None:
        return None
    return str(
        figure.to_html(
            include_plotlyjs=True if include_js else False,
            full_html=False,
            config={'displaylogo': False, 'responsive': True},
        )
    )


# ---- Tables ---- #


def _headline_table(study: Study) -> pd.DataFrame:
    """The one table a reader should be able to stop at."""
    table = multi_seed_table(study)
    if table.empty:
        return table
    columns = ['arm', 'n_seeds', 'n_runs']
    for metric in ('held_out_rank_percentile', 'held_out_top1', 'held_out_lift', 'stratified_top1', 'word_lift'):
        if f'{metric}_mean' in table:
            table[metric] = [_cell(m, s) for m, s in zip(table[f'{metric}_mean'], table[f'{metric}_sd'], strict=True)]
            columns.append(metric)
    return table[columns]


def _ablation_view(study: Study) -> pd.DataFrame:
    """The feature-ablation table, formatted as mean ± sd cells."""
    table = feature_ablation_table(study)
    if table.empty:
        return table
    columns = ['question', 'level', 'n_runs']
    for metric in ('held_out_rank_percentile', 'held_out_top1', 'stratified_top1', 'rescoring_top1'):
        if f'{metric}_mean' in table:
            table[metric] = [_cell(m, s) for m, s in zip(table[f'{metric}_mean'], table[f'{metric}_sd'], strict=True)]
            columns.append(metric)
    return table[columns]


def _within_view(study: Study) -> pd.DataFrame:
    """Within-task retrieval, averaged over runs of one arm."""
    table = within_task_table(study)
    if table.empty:
        return table
    return (
        table.groupby(['arm', 'task'], dropna=False)[['top1', 'chance', 'lift', 'rank_percentile', 'n_candidates']]
        .mean()
        .round(4)
        .reset_index()
    )


def _fold_view(study: Study) -> pd.DataFrame:
    """The per-fold table, one row per (arm, held-out subject)."""
    table = loso_table(study)
    if table.empty:
        return table
    table['score'] = [_cell(m, s) for m, s in zip(table['mean'], table['sd'], strict=True)]
    return table[['arm', 'holdout', 'n_seeds', 'score']]


def _cell(mean: Any, sd: Any) -> str:
    """Formats one `mean ± sd` table cell."""
    try:
        point = float(mean)
    except TypeError, ValueError:
        return '—'
    if not np.isfinite(point):
        return '—'
    try:
        spread = float(sd)
    except TypeError, ValueError:
        return f'{point:.4f}'
    return f'{point:.4f}' if not np.isfinite(spread) else f'{point:.4f} ± {spread:.4f}'


def _table_html(frame: pd.DataFrame) -> str:
    """Renders a frame as a sortable HTML table, or a plain note when it is empty."""
    if frame is None or frame.empty:
        return '<p class="empty">No run in this study carried the numbers for this table.</p>'
    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in frame.columns)
    rows = ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(_text(v))}</td>' for v in row) + '</tr>'
        for row in frame.itertuples(index=False, name=None)
    )
    return f'<table class="grid"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'


def _text(value: Any) -> str:
    """Formats a table value compactly, rounding floats and joining lists."""
    if isinstance(value, float):
        return '—' if not np.isfinite(value) else f'{value:.4f}'
    if isinstance(value, (list, tuple)):
        return ', '.join(str(v) for v in value)
    return '—' if value is None else str(value)


# ---- Page ---- #


def _banner(study: Study) -> str:
    """The provenance strip: how many runs, from where, and whether any of them is synthetic."""
    if study.is_empty:
        return '<div class="banner warn">No evaluated run was found under the given roots.</div>'
    total = len(study.runs)
    synthetic = int((~study.runs['real_data'].astype(bool)).sum()) if 'real_data' in study.runs else 0
    roots = ', '.join(html.escape(str(r)) for r in study.roots)
    note = f'<div class="banner">{total} run(s) from {roots}.</div>'
    if synthetic:
        note += (
            f'<div class="banner warn"><strong>{synthetic} of {total} runs are SYNTHETIC.</strong> '
            'A synthetic run is a wiring check, never a result. Nothing drawn from those rows is evidence.</div>'
        )
    return note


def _card(content: str, caption: str) -> str:
    """Wraps one panel with its caption."""
    return f'<figure class="card">{content}<figcaption>{caption}</figcaption></figure>'


def _section(key: str, label: str, cards: list[str]) -> str:
    """Wraps a section's cards under a heading the nav links to."""
    return f'<section id="{key}"><h2>{html.escape(label)}</h2>{"".join(cards)}</section>'


def _document(title: str, banner: str, body: str) -> str:
    """Wraps the rendered panels in the page shell."""
    nav = ''.join(f'<a href="#{k}">{html.escape(v)}</a>' for k, v in _SECTIONS)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; --bg:#ffffff; --fg:#14181d; --muted:#5a6672; --line:#e2e8ee; --card:#ffffff; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#101418; --fg:#e8edf2; --muted:#9aa7b4; --line:#242c34; --card:#161b21; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
  header {{ position:sticky; top:0; z-index:10; background:var(--bg); border-bottom:1px solid var(--line); padding:14px 24px; }}
  header h1 {{ margin:0 0 8px; font-size:19px; letter-spacing:-0.01em; }}
  nav a {{ color:var(--muted); text-decoration:none; margin-right:18px; font-size:13px; }}
  nav a:hover {{ color:var(--fg); }}
  main {{ padding:0 24px 64px; max-width:1400px; margin:0 auto; }}
  section {{ padding-top:28px; }}
  section h2 {{ font-size:16px; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); border-bottom:1px solid var(--line); padding-bottom:8px; }}
  .card {{ margin:20px 0 30px; padding:12px; border:1px solid var(--line); border-radius:10px; background:var(--card); overflow-x:auto; }}
  .card figcaption {{ margin-top:10px; color:var(--muted); font-size:13px; }}
  .banner {{ margin:12px 0; padding:10px 14px; border-left:4px solid #0072b2; background:rgba(0,114,178,0.07); border-radius:0 6px 6px 0; }}
  .banner.warn {{ border-left-color:#c1121f; background:rgba(193,18,31,0.09); }}
  table.grid {{ border-collapse:collapse; width:100%; font-size:13px; }}
  table.grid th, table.grid td {{ border-bottom:1px solid var(--line); padding:7px 10px; text-align:left; white-space:nowrap; }}
  table.grid th {{ color:var(--muted); font-weight:600; text-transform:uppercase; font-size:11px; letter-spacing:0.06em; }}
  table.grid tbody tr:hover {{ background:rgba(127,127,127,0.08); }}
  code {{ background:rgba(127,127,127,0.14); padding:1px 5px; border-radius:4px; font-size:12px; }}
  .empty {{ color:var(--muted); font-style:italic; }}
</style></head>
<body>
<header><h1>{html.escape(title)}</h1><nav>{nav}</nav></header>
<main>{banner}{body}</main>
</body></html>
"""
