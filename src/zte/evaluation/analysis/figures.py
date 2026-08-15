"""Every chart the study dashboard draws, as self-contained Plotly figures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from zte.evaluation.analysis.aggregate import (
    HEADLINE_METRICS,
    control_table,
    feature_ablation_table,
    loso_table,
    multi_seed_table,
    within_task_table,
)
from zte.evaluation.analysis.collect import Study
from zte.logging_utils import get_logger

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

_LOG = get_logger('evaluation.analysis.figures')

# One qualitative ramp for conditions and one diverging ramp for lifts, so colour means the same thing on every
# chart in the dashboard: warm is the brain-driven decode, grey-blue are the controls it must beat.
CONDITION_COLOURS: dict[str, str] = {
    'hypothesis': '#e4572e',
    'oracle': '#17bebb',
    'mean_prefix': '#76949f',
    'null_prefix': '#8896ab',
    'phase': '#9a8c98',
    'noise': '#6d6875',
    'shuffled_z': '#5c677d',
    'length_only': '#b08968',
    'mismatch': '#4a5759',
}
SEQUENTIAL = 'Viridis'
DIVERGING = 'RdBu'

# Colour-blind-safe qualitative ramp for arms; deliberately not the condition ramp, which carries its own meaning.
ARM_COLOURS: tuple[str, ...] = (
    '#0072b2',
    '#d55e00',
    '#009e73',
    '#cc79a7',
    '#e69f00',
    '#56b4e9',
    '#7a5195',
    '#bc5090',
)

# Training-history columns the mechanism panel offers; every encoder-v3 term writes one of these prefixes.
_MECHANISM_PREFIXES: tuple[str, ...] = (
    'train_consensus',
    'train_gallery',
    'train_residual',
    'train_lexical',
    'train_clip',
)

# The levers the mechanism matrix reads, in the order the arms were designed to be compared.
_MECHANISM_LEVERS: tuple[str, ...] = (
    'residual_coding',
    'consensus_weight',
    'consensus_gallery_weight',
    'consensus_word_weight',
    'gallery_weight',
    'gallery_length_band',
    'length_projection',
    'lexical_weight',
    'lexical_reader_weight',
)

# H(identity) - H(identity | n_words) on the real 700-stimulus SR+NR gallery: what word count gives away for free.
_LENGTH_BITS: float = 5.1422


def _go() -> Any:
    """Imports `plotly.graph_objects`, raising a message that names the dependency group when it is missing."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - the viz group is a default group
        raise RuntimeError('The analysis dashboard needs plotly: `uv sync --group viz`.') from exc
    return go


def _layout(fig: 'Figure', title: str, *, height: int = 420, **kwargs: Any) -> 'Figure':
    """Applies the dashboard's shared layout so every panel reads as one document."""
    fig.update_layout(
        title=title,
        height=height,
        template='plotly_white',
        margin={'l': 60, 'r': 30, 't': 60, 'b': 60},
        font={'family': 'system-ui, -apple-system, Segoe UI, sans-serif', 'size': 12},
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0},
        **kwargs,
    )
    return fig


def _arm_colour(index: int) -> str:
    """Returns a stable colour for the `index`-th arm."""
    return ARM_COLOURS[index % len(ARM_COLOURS)]


def _lever_strength(value: Any) -> float:
    """Maps a lever's configured value onto a number the mechanism heatmap can colour; unset reads as 0."""
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except TypeError, ValueError:
        return 0.0 if value in (None, '', 'none') else 1.0


def headline_bars(study: Study, metric: str = 'held_out_rank_percentile') -> 'Figure | None':
    """Mean +/- sd of one headline metric per arm, with every seed drawn on top of its bar.

    Args:
        study (Study): The collected study.
        metric (str, optional): Metric to plot. Defaults to 'held_out_rank_percentile'.

    Returns:
        Figure | None: The bar chart, or `None` when nothing was collected.

    Note:
        The individual seeds are drawn as points because the bar alone hides the failure this project has already
        hit: a bimodal sweep where some seeds train a healthy code and others collapse. Two runs averaging to a good
        number is a different finding from four runs clustered on it.
    """
    go = _go()
    table = multi_seed_table(study, metrics=(metric,))
    if table.empty or f'{metric}_mean' not in table:
        return None

    fig = go.Figure()
    fig.add_bar(
        x=table['arm'],
        y=table[f'{metric}_mean'],
        error_y={'type': 'data', 'array': table[f'{metric}_sd'].fillna(0.0), 'visible': True},
        marker_color=[_arm_colour(i) for i in range(len(table))],
        hovertemplate='%{x}<br>mean %{y:.4f}<extra></extra>',
        name='mean ± sd',
    )
    runs = study.runs
    if metric in runs.columns:
        fig.add_scatter(
            x=runs['arm'],
            y=pd.to_numeric(runs[metric], errors='coerce'),
            mode='markers',
            marker={'color': '#222', 'size': 7, 'symbol': 'circle-open', 'line': {'width': 1.5}},
            name='per seed',
            hovertemplate='%{customdata}<br>%{y:.4f}<extra></extra>',
            customdata=runs['run'],
        )
    return _layout(fig, f'{metric.replace("_", " ")} by arm — mean ± sd over seeds', yaxis_title=metric)


def loso_heatmap(study: Study, metric: str = 'held_out_rank_percentile') -> 'Figure | None':
    """Arm-by-held-out-subject heatmap: which brains a recipe reaches and which it never does.

    Args:
        study (Study): The collected study.
        metric (str, optional): Fold metric. Defaults to 'held_out_rank_percentile'.

    Returns:
        Figure | None: The heatmap, or `None` when no fold was collected.
    """
    go = _go()
    table = loso_table(study, metric=metric)
    if table.empty:
        return None

    grid = table.pivot_table(index='arm', columns='holdout', values='mean')
    counts = table.pivot_table(index='arm', columns='holdout', values='n_seeds').reindex_like(grid)
    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=[str(c) for c in grid.columns],
            y=[str(i) for i in grid.index],
            colorscale=SEQUENTIAL,
            colorbar={'title': metric},
            customdata=counts.to_numpy(),
            hovertemplate='%{y}<br>held out %{x}<br>%{z:.4f} over %{customdata:.0f} seed(s)<extra></extra>',
        )
    )
    return _layout(
        fig,
        f'LOSO trend — {metric.replace("_", " ")} per held-out subject',
        height=90 + 34 * max(len(grid.index), 2),
        xaxis_title='held-out subject',
    )


def fold_spread(study: Study, metric: str = 'held_out_rank_percentile') -> 'Figure | None':
    """Per-fold distribution as a box plus every point, so a bimodal sweep cannot hide behind a mean.

    Args:
        study (Study): The collected study.
        metric (str, optional): Fold metric. Defaults to 'held_out_rank_percentile'.

    Returns:
        Figure | None: The box plot, or `None` when no fold was collected.
    """
    go = _go()
    folds = study.folds
    if folds.empty or metric not in folds.columns:
        return None

    fig = go.Figure()
    for i, (arm, block) in enumerate(folds.groupby('arm', dropna=False)):
        fig.add_box(
            y=pd.to_numeric(block[metric], errors='coerce'),
            name=str(arm),
            boxpoints='all',
            jitter=0.4,
            pointpos=0,
            marker_color=_arm_colour(i),
            text=block['holdout'],
            hovertemplate='%{text}<br>%{y:.4f}<extra></extra>',
        )
    return _layout(fig, f'Fold-to-fold spread of {metric.replace("_", " ")}', yaxis_title=metric)


def ablation_bars(study: Study, metric: str = 'held_out_rank_percentile') -> 'Figure | None':
    """The single-lever comparison: raw vs band power, harmonics vs indexing, invariance on vs off.

    Args:
        study (Study): The collected study.
        metric (str, optional): Metric to compare. Defaults to 'held_out_rank_percentile'.

    Returns:
        Figure | None: A grouped bar chart, or `None` when no lever had two levels.
    """
    go = _go()
    table = feature_ablation_table(study, metrics=(metric,))
    if table.empty or f'{metric}_mean' not in table:
        return None

    fig = go.Figure()
    for i, (question, block) in enumerate(table.groupby('question', sort=False)):
        fig.add_bar(
            x=block['level'],
            y=block[f'{metric}_mean'],
            error_y={'type': 'data', 'array': block[f'{metric}_sd'].fillna(0.0), 'visible': True},
            name=str(question),
            marker_color=_arm_colour(i),
            customdata=np.stack([block['n_runs'], block[f'{metric}_n']], axis=-1),
            hovertemplate='%{x}<br>%{y:.4f}<br>%{customdata[0]:.0f} run(s)<extra></extra>',
        )
    return _layout(fig, f'Feature ablation — {metric.replace("_", " ")}', barmode='group', yaxis_title=metric)


def within_task_bars(study: Study) -> 'Figure | None':
    """Within-task Top-1 against each pool's own chance level, which is the sceptic's version of the headline.

    Args:
        study (Study): The collected study.

    Returns:
        Figure | None: The grouped bar chart, or `None` when no run reported a within-task pool.
    """
    go = _go()
    table = within_task_table(study)
    if table.empty:
        return None

    agg = table.groupby(['arm', 'task'], dropna=False)[['top1', 'chance']].mean().reset_index()
    fig = go.Figure()
    for i, (task, block) in enumerate(agg.groupby('task', sort=True)):
        fig.add_bar(x=block['arm'], y=block['top1'], name=f'{task} Top-1', marker_color=_arm_colour(i))
        fig.add_scatter(
            x=block['arm'],
            y=block['chance'],
            mode='markers',
            marker={'symbol': 'line-ew', 'size': 22, 'line': {'width': 3, 'color': '#c1121f'}},
            name=f'{task} chance',
            hovertemplate='chance %{y:.4f}<extra></extra>',
        )
    return _layout(fig, 'Within-task retrieval — passage identity held fixed', barmode='group', yaxis_title='Top-1')


def control_ladder(study: Study, metric: str = 'score_content_f1') -> 'Figure | None':
    """The decode against every brain-independent control, per run -- the only readable generation number.

    Args:
        study (Study): The collected study.
        metric (str, optional): Per-sentence metric column. Defaults to 'score_content_f1'.

    Returns:
        Figure | None: A grouped bar chart, or `None` when no generation was collected.

    Note:
        Absolute height means nothing here. What is readable is whether the warm bar clears every grey one: a decoder
        reciting the corpus scores the same as `mean_prefix`, and one exploiting sentence length scores the same as
        `length_only`.
    """
    go = _go()
    table = control_table(study)
    if table.empty or metric not in table.columns:
        return None

    order = [c for c in CONDITION_COLOURS if c in set(table['condition'])]
    fig = go.Figure()
    for condition in order:
        block = table[table['condition'] == condition]
        fig.add_bar(
            x=block['run'],
            y=block[metric],
            name=condition,
            marker_color=CONDITION_COLOURS.get(condition, '#888'),
            hovertemplate=f'{condition}<br>%{{x}}<br>%{{y:.4f}}<extra></extra>',
        )
    return _layout(
        fig,
        f'Generation against every control — {metric.replace("score_", "")}',
        barmode='group',
        yaxis_title=metric.replace('score_', ''),
    )


def score_distributions(study: Study, metric: str = 'score_content_f1') -> 'Figure | None':
    """Per-sentence score histograms by condition: the whole distribution, not the mean of it.

    Args:
        study (Study): The collected study.
        metric (str, optional): Per-sentence metric column. Defaults to 'score_content_f1'.

    Returns:
        Figure | None: Overlaid histograms, or `None` when no generation was collected.
    """
    go = _go()
    gens = study.generations
    if gens.empty or metric not in gens.columns:
        return None

    fig = go.Figure()
    for condition in [c for c in CONDITION_COLOURS if c in set(gens['condition'])]:
        values = pd.to_numeric(gens.loc[gens['condition'] == condition, metric], errors='coerce').dropna()
        if values.empty:
            continue
        fig.add_histogram(
            x=values,
            name=condition,
            opacity=0.55,
            nbinsx=40,
            marker_color=CONDITION_COLOURS.get(condition, '#888'),
        )
    return _layout(
        fig,
        f'Per-sentence {metric.replace("score_", "")} by condition',
        barmode='overlay',
        xaxis_title=metric.replace('score_', ''),
        yaxis_title='sentences',
    )


def text_overlap_heatmap(study: Study, *, max_sentences: int = 60) -> 'Figure | None':
    """Sentence-by-condition heatmap of content-word overlap -- where the decode actually landed.

    Args:
        study (Study): The collected study.
        max_sentences (int, optional): Rows shown, longest references first. Defaults to 60.

    Returns:
        Figure | None: The heatmap, or `None` when no generation was collected.

    Note:
        Reading down a column says whether a condition is uniformly mediocre or occasionally right, which the mean
        cannot distinguish and which decides whether a small average lift is a real effect on a few sentences.
    """
    go = _go()
    gens = study.generations
    if gens.empty or 'score_content_f1' not in gens.columns:
        return None

    frame = gens[gens['run'] == gens['run'].iloc[0]]
    grid = frame.pivot_table(index='index', columns='condition', values='score_content_f1', aggfunc='mean')
    if grid.empty:
        return None
    grid = grid.head(max_sentences)
    references = frame.drop_duplicates('index').set_index('index')['reference'].reindex(grid.index)

    columns = [c for c in CONDITION_COLOURS if c in grid.columns]
    fig = go.Figure(
        go.Heatmap(
            z=grid[columns].to_numpy(),
            x=columns,
            y=[str(i) for i in grid.index],
            colorscale=SEQUENTIAL,
            colorbar={'title': 'content-word F1'},
            customdata=np.stack([references.fillna('').to_numpy()] * len(columns), axis=-1),
            hovertemplate='sentence %{y} — %{x}<br>F1 %{z:.3f}<br>%{customdata}<extra></extra>',
        )
    )
    return _layout(
        fig,
        f'Where the decode landed — {frame["run"].iloc[0]}',
        height=120 + 12 * max(len(grid.index), 6),
        xaxis_title='condition',
        yaxis_title='held-out sentence',
    )


def word_frequency_bars(study: Study, *, top_n: int = 25) -> 'Figure | None':
    """The words the decoder emits against the words it should have, which exposes a corpus-prior recital.

    Args:
        study (Study): The collected study.
        top_n (int, optional): Word types shown. Defaults to 25.

    Returns:
        Figure | None: A paired bar chart, or `None` when no generation was collected.
    """
    go = _go()
    gens = study.generations
    if gens.empty or 'text' not in gens.columns:
        return None
    from zte.evaluation.generation import content_words

    hypotheses = gens[gens['condition'] == 'hypothesis']
    if hypotheses.empty:
        return None
    emitted = pd.Series([w for t in hypotheses['text'].dropna() for w in content_words(str(t))]).value_counts()
    wanted = pd.Series(
        [w for t in hypotheses['reference'].drop_duplicates().dropna() for w in content_words(str(t))]
    ).value_counts()
    if emitted.empty:
        return None

    words = list(emitted.head(top_n).index)
    fig = go.Figure()
    fig.add_bar(x=words, y=[int(emitted.get(w, 0)) for w in words], name='emitted', marker_color='#e4572e')
    fig.add_bar(x=words, y=[int(wanted.get(w, 0)) for w in words], name='in the references', marker_color='#76949f')
    return _layout(
        fig,
        'Content words the decoder emits vs the words it was asked for',
        barmode='group',
        xaxis_title='content word',
        yaxis_title='occurrences',
    )


def length_confound_scatter(study: Study) -> 'Figure | None':
    """The encoder against the length-only oracle it has to beat, per tolerance.

    Args:
        study (Study): The collected study.

    Returns:
        Figure | None: The scatter, or `None` when no rebaseline audit was collected.

    Note:
        On the real 700-stimulus gallery, word count alone carries 5.14 bits of sentence identity and comes free from
        eye-tracking segmentation. A model point below the oracle curve is reproducing sentence length, whatever its
        Top-k says.
    """
    go = _go()
    audit = study.rebaseline
    if audit.empty:
        return None

    oracle = audit[audit['kind'] == 'oracle']
    model = audit[audit['kind'] == 'model']
    if oracle.empty and model.empty:
        return None

    fig = go.Figure()
    if not oracle.empty and 'rank_percentile' in oracle.columns:
        fig.add_scatter(
            x=oracle.get('tolerance', pd.Series(range(len(oracle)))),
            y=oracle['rank_percentile'],
            mode='lines+markers',
            name='length-only oracle',
            line={'color': '#c1121f', 'dash': 'dash', 'width': 3},
            hovertemplate='±%{x} words<br>rank percentile %{y:.4f}<extra></extra>',
        )
    if not model.empty and 'rank_percentile' in model.columns:
        fig.add_scatter(
            x=[0] * len(model),
            y=model['rank_percentile'],
            mode='markers',
            name='encoder',
            marker={'size': 12, 'color': '#0072b2'},
            text=model['run'] + ' — ' + model.get('postprocess', pd.Series(['?'] * len(model))).astype(str),
            hovertemplate='%{text}<br>rank percentile %{y:.4f}<extra></extra>',
        )
    return _layout(
        fig,
        'The length confound — encoder against a length-only oracle',
        xaxis_title='oracle word-count tolerance (±words)',
        yaxis_title='rank percentile',
    )


def bit_budget_bars(study: Study) -> 'Figure | None':
    """What the conditioning channel is architecturally allowed and what it actually delivered.

    Args:
        study (Study): The collected study.

    Returns:
        Figure | None: The bar chart, or `None` when no run used a rate ladder.

    Note:
        A 19.6-word English sentence needs roughly 190 bits. Sentence identity over 700 stimuli needs 9.45. This
        chart is against the second number, never the first, and it is why generation is reported as an expected null
        while retrieval is the powered readout.
    """
    go = _go()
    runs = study.runs
    if runs.empty or 'bit_mutual_information' not in runs.columns:
        return None
    block = runs[runs['bit_mutual_information'].notna()]
    if block.empty:
        return None

    fig = go.Figure()
    fig.add_bar(x=block['run'], y=block['bit_capacity'], name='architectural ceiling', marker_color='#dbe4ee')
    fig.add_bar(
        x=block['run'], y=block['bit_mutual_information'], name='delivered (upper bound)', marker_color='#0072b2'
    )
    if block['bit_residual_mi'].notna().any():
        fig.add_bar(
            x=block['run'],
            y=block['bit_residual_mi'],
            name='delivered beyond word count',
            marker_color='#e4572e',
        )
    fig.add_hline(
        y=9.4512,
        line={'color': '#c1121f', 'dash': 'dot'},
        annotation_text='9.45 bits — sentence identity over 700 stimuli',
    )
    fig.add_hline(y=5.1422, line={'color': '#b08968', 'dash': 'dot'}, annotation_text='5.14 bits — word count alone')
    return _layout(fig, 'The measured bit budget', barmode='group', yaxis_title='bits')


def probe_heatmap(study: Study) -> 'Figure | None':
    """Target-by-representation probe scores, which is the who-versus-what picture in one panel.

    Args:
        study (Study): The collected study.

    Returns:
        Figure | None: The heatmap, or `None` when no probe rows were collected.
    """
    go = _go()
    probes = study.probes
    if probes.empty:
        return None

    grid = probes.pivot_table(index='target', columns='representation', values='linear', aggfunc='mean')
    if grid.empty:
        return None
    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=[str(c) for c in grid.columns],
            y=[str(i) for i in grid.index],
            colorscale=DIVERGING,
            zmid=0.0,
            colorbar={'title': 'linear probe'},
            hovertemplate='%{y} from %{x}<br>%{z:.4f}<extra></extra>',
        )
    )
    return _layout(fig, 'Transfer probes — what each representation carries', height=90 + 40 * max(len(grid.index), 3))


def variance_budget_pie(study: Study) -> 'Figure | None':
    """How the embedding spends its variance: identity, content or neither.

    Args:
        study (Study): The collected study.

    Returns:
        Figure | None: A pie chart, or `None` when no run reported a neuron budget.
    """
    go = _go()
    runs = study.runs
    if runs.empty or 'who_vs_what' not in runs.columns:
        return None
    probes = study.probes
    if probes.empty:
        return None

    subject = probes[(probes['target'] == 'subject') & (probes['representation'] == 'ZTE')]['linear'].mean()
    content = probes[(probes['target'].isin({'word_len', 'log_freq'})) & (probes['representation'] == 'ZTE')][
        'linear'
    ].mean()
    if not np.isfinite(subject) and not np.isfinite(content):
        return None

    who = max(float(subject or 0.0), 0.0)
    what = max(float(content or 0.0), 0.0)
    fig = go.Figure(
        go.Pie(
            labels=['who (subject identity)', 'what (lexical content)', 'neither'],
            values=[who, what, max(1.0 - who - what, 0.0)],
            marker={'colors': ['#c1121f', '#0072b2', '#dbe4ee']},
            hole=0.45,
            hovertemplate='%{label}<br>%{value:.3f}<extra></extra>',
        )
    )
    return _layout(fig, 'Who versus what — the readable share of the space', height=380)


def learning_curves(study: Study) -> 'Figure | None':
    """Train and validation loss per epoch, which is where a bimodal sweep declares itself early.

    Args:
        study (Study): The collected study.

    Returns:
        Figure | None: The line chart, or `None` when no history was collected.
    """
    go = _go()
    history = study.history
    if history.empty or 'epoch' not in history.columns:
        return None

    fig = go.Figure()
    for i, (run, block) in enumerate(history.groupby('run', sort=True)):
        colour = _arm_colour(i)
        for column, dash in (('train_loss', 'solid'), ('val_loss', 'dot')):
            if column not in block.columns or block[column].isna().all():
                continue
            fig.add_scatter(
                x=block['epoch'],
                y=block[column],
                mode='lines',
                name=f'{run} {column.split("_")[0]}',
                line={'color': colour, 'dash': dash, 'width': 2},
            )
    return _layout(fig, 'Learning curves', xaxis_title='epoch', yaxis_title='loss', height=460)


def subject_difficulty(study: Study) -> 'Figure | None':
    """Per-subject retrieval, so "which brain is hard" is a chart rather than an impression.

    Args:
        study (Study): The collected study.

    Returns:
        Figure | None: The scatter, or `None` when no per-subject rows were collected.
    """
    go = _go()
    subjects = study.subjects
    if subjects.empty or 'subject' not in subjects.columns:
        return None
    metric = next((c for c in ('retrieval_top1', 'top1', 'r2(word_len)') if c in subjects.columns), None)
    if metric is None:
        return None

    fig = go.Figure()
    for i, (run, block) in enumerate(subjects.groupby('run', sort=True)):
        fig.add_scatter(
            x=block['subject'],
            y=pd.to_numeric(block[metric], errors='coerce'),
            mode='markers',
            name=str(run),
            marker={'size': 11, 'color': _arm_colour(i), 'opacity': 0.8},
        )
    return _layout(fig, f'Per-subject {metric}', xaxis_title='subject', yaxis_title=metric)


def length_vs_score(study: Study) -> 'Figure | None':
    """Sentence length against decode quality, per condition -- the length confound at sentence resolution.

    Args:
        study (Study): The collected study.

    Returns:
        Figure | None: The scatter, or `None` when no generation was collected.
    """
    go = _go()
    gens = study.generations
    if gens.empty or 'n_words' not in gens.columns or 'score_content_f1' not in gens.columns:
        return None

    fig = go.Figure()
    for condition in [
        c for c in ('hypothesis', 'length_only', 'mean_prefix', 'mismatch') if c in set(gens['condition'])
    ]:
        block = gens[gens['condition'] == condition]
        fig.add_scatter(
            x=pd.to_numeric(block['n_words'], errors='coerce'),
            y=pd.to_numeric(block['score_content_f1'], errors='coerce'),
            mode='markers',
            name=condition,
            marker={'size': 7, 'opacity': 0.55, 'color': CONDITION_COLOURS.get(condition, '#888')},
            text=block['reference'],
            hovertemplate='%{x} words<br>F1 %{y:.3f}<br>%{text}<extra></extra>',
        )
    return _layout(
        fig,
        'Sentence length against decode quality',
        xaxis_title='reference word count',
        yaxis_title='content-word F1',
    )


def metric_correlations(study: Study, metrics: Sequence[str] = HEADLINE_METRICS) -> 'Figure | None':
    """How the headline metrics move together across runs, which exposes a metric that is really another one.

    Args:
        study (Study): The collected study.
        metrics (Sequence[str], optional): Metrics to correlate. Defaults to `HEADLINE_METRICS`.

    Returns:
        Figure | None: The correlation heatmap, or `None` when fewer than three runs were collected.
    """
    go = _go()
    runs = study.runs
    present = [m for m in metrics if m in runs.columns and pd.to_numeric(runs[m], errors='coerce').notna().sum() > 2]
    if len(runs) < 3 or len(present) < 2:
        return None

    corr = runs[present].apply(pd.to_numeric, errors='coerce').corr(min_periods=3)
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=present,
            y=present,
            colorscale=DIVERGING,
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            colorbar={'title': 'r'},
            hovertemplate='%{y} vs %{x}<br>r = %{z:.2f}<extra></extra>',
        )
    )
    return _layout(fig, 'Do the headline metrics say the same thing?', height=120 + 34 * len(present))


def metric_explorer(study: Study, metrics: Sequence[str] = HEADLINE_METRICS) -> 'Figure | None':
    """Every headline metric, per arm, behind one dropdown -- the panel to open first and leave open.

    Args:
        study (Study): The collected study.
        metrics (Sequence[str], optional): Metrics to offer. Defaults to `HEADLINE_METRICS`.

    Returns:
        Figure | None: A bar chart with a metric selector, or `None` when nothing was collected.
    """
    go = _go()
    table = multi_seed_table(study, metrics=metrics)
    present = [m for m in metrics if f'{m}_mean' in table and table[f'{m}_mean'].notna().any()]
    if table.empty or not present:
        return None

    fig = go.Figure()
    for i, metric in enumerate(present):
        fig.add_bar(
            x=table['arm'],
            y=table[f'{metric}_mean'],
            error_y={'type': 'data', 'array': table[f'{metric}_sd'].fillna(0.0), 'visible': True},
            marker_color=[_arm_colour(j) for j in range(len(table))],
            visible=i == 0,
            hovertemplate='%{x}<br>%{y:.4f}<extra></extra>',
            name=metric,
        )

    buttons = [
        {
            'label': metric.replace('_', ' '),
            'method': 'update',
            'args': [{'visible': [j == i for j in range(len(present))]}, {'yaxis': {'title': metric}}],
        }
        for i, metric in enumerate(present)
    ]
    fig.update_layout(updatemenus=[{'buttons': buttons, 'x': 1.0, 'xanchor': 'right', 'y': 1.16}])
    return _layout(fig, 'Pick a headline — mean ± sd over seeds', yaxis_title=present[0], height=480)


def mechanism_curves(study: Study) -> 'Figure | None':
    """Per-epoch curves for the encoder mechanisms, one arm per line, behind a metric dropdown.

    Returns:
        Figure | None: The line chart, or `None` when no run recorded a mechanism metric.

    Note:
        A consensus term that never engaged, a gallery accuracy pinned at chance or a residual gate that trained to
        zero is visible here and nowhere else in the artifacts -- the final metrics cannot distinguish "the
        mechanism did nothing" from "the mechanism was never switched on".
    """
    go = _go()
    history = study.history
    if history.empty:
        return None

    tracked = [c for c in history.columns if c.startswith(_MECHANISM_PREFIXES) and history[c].notna().any()]
    if not tracked:
        return None

    fig, visibility = go.Figure(), []
    runs = sorted(history['run'].dropna().unique())
    for metric in tracked:
        for i, run in enumerate(runs):
            rows = history[history['run'] == run].sort_values('epoch')
            fig.add_scatter(
                x=rows['epoch'],
                y=pd.to_numeric(rows[metric], errors='coerce'),
                mode='lines+markers',
                name=run,
                line={'color': _arm_colour(i), 'width': 2},
                visible=metric == tracked[0],
                hovertemplate=f'{run}<br>epoch %{{x}}<br>%{{y:.4f}}<extra></extra>',
            )
        visibility.append(metric)

    buttons = [
        {
            'label': metric.removeprefix('train_').replace('_', ' '),
            'method': 'update',
            'args': [
                {'visible': [visibility[j // len(runs)] == metric for j in range(len(tracked) * len(runs))]},
                {'yaxis': {'title': metric}},
            ],
        }
        for metric in tracked
    ]
    fig.update_layout(updatemenus=[{'buttons': buttons, 'x': 1.0, 'xanchor': 'right', 'y': 1.16}])
    return _layout(fig, 'Did the mechanism engage? — per-epoch training metrics', xaxis_title='epoch', height=480)


def length_leakage_bars(study: Study) -> 'Figure | None':
    """How much of each arm's embedding variance word count explains, before and after the projection.

    Returns:
        Figure | None: The grouped bar chart, or `None` when no run ran the projection.

    Note:
        The residual after the projection is what a train-fitted basis failed to transfer, not length the encoder is
        free to use. A bar that barely moves means the fit did not generalise, which is a finding about the fit.
    """
    go = _go()
    runs = study.runs
    if runs.empty or 'length_leakage_before' not in runs.columns:
        return None

    rows = runs.dropna(subset=['length_leakage_before'])
    if rows.empty:
        return None

    fig = go.Figure()
    fig.add_bar(x=rows['run'], y=rows['length_leakage_before'], name='before', marker_color='#d55e00')
    fig.add_bar(x=rows['run'], y=rows['length_leakage_after'], name='after', marker_color='#0072b2')
    fig.update_layout(barmode='group')
    return _layout(
        fig,
        'Sentence-length leakage — variance word count explains, before and after the projection',
        yaxis_title='explained variance fraction',
    )


def identity_vs_content(study: Study) -> 'Figure | None':
    """Who versus what: the subject probe against the word-length probe, one point per run.

    Returns:
        Figure | None: The scatter, or `None` when neither probe was collected.

    Note:
        The bottom-right quadrant is the goal and the top-left is the ZTE v1 failure mode. A point that slides left
        *and* down has not become invariant, it has collapsed -- which is why the marker is sized by effective rank.
    """
    go = _go()
    runs = study.runs
    if runs.empty or not {'subject_probe', 'word_len_probe'} <= set(runs.columns):
        return None

    rows = runs.dropna(subset=['subject_probe', 'word_len_probe'])
    if rows.empty:
        return None

    rank = pd.to_numeric(rows.get('effective_rank_ratio'), errors='coerce').fillna(0.25)
    fig = go.Figure(
        go.Scatter(
            x=rows['subject_probe'],
            y=rows['word_len_probe'],
            mode='markers+text',
            text=rows['arm'],
            textposition='top center',
            marker={
                'size': 8 + 40 * rank,
                'color': rank,
                'colorscale': SEQUENTIAL,
                'colorbar': {'title': 'eff-rank ratio'},
                'line': {'width': 1, 'color': '#333'},
            },
            hovertemplate='%{text}<br>subject %{x:.3f}<br>word_len %{y:.3f}<extra></extra>',
        )
    )
    if 'subject_probe_raw' in rows and rows['subject_probe_raw'].notna().any():
        fig.add_vline(
            x=float(rows['subject_probe_raw'].dropna().mean()),
            line={'dash': 'dash', 'color': '#888'},
            annotation_text='raw features',
        )
    return _layout(
        fig,
        'Who vs what — subject probe (want low) against content probe (want high)',
        xaxis_title='subject probe accuracy',
        yaxis_title='word_len probe R²',
        height=520,
    )


def mechanism_matrix(study: Study) -> 'Figure | None':
    """Which mechanisms each arm actually had switched on, read straight off the resolved configs.

    Returns:
        Figure | None: The arm-by-lever heatmap, or `None` when no lever was recorded.

    Note:
        This is the panel that catches the most embarrassing failure in a sweep: an "ablation" whose config never
        differed from its baseline. Two identical rows mean two identical runs, whatever the file names say.
    """
    go = _go()
    runs = study.runs
    levers = [k for k in _MECHANISM_LEVERS if k in runs.columns]
    if runs.empty or not levers:
        return None

    values = np.array([[_lever_strength(v) for v in runs[lever]] for lever in levers], dtype=float)
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=runs['run'],
            y=[lever.replace('_', ' ') for lever in levers],
            colorscale=SEQUENTIAL,
            zmin=0.0,
            hovertemplate='%{x}<br>%{y} = %{z}<extra></extra>',
            colorbar={'title': 'setting'},
        )
    )
    return _layout(fig, 'Which levers were on — one column per run', height=140 + 34 * len(levers))


def bit_budget_pie(study: Study) -> 'Figure | None':
    """The 9.45 bits of ZuCo sentence identity, split into what length gives away and what the encoder adds.

    Returns:
        Figure | None: The pie, or `None` when no run reported a held-out rank percentile.

    Note:
        The encoder slice is derived from rank percentile, not from Top-1: with 700 queries at chance 1/700 the
        expected Top-1 hit count is one, so a Top-1-derived bit count would be mostly sampling noise.
    """
    go = _go()
    runs = study.runs
    if runs.empty or 'held_out_rank_percentile' not in runs.columns:
        return None

    best = pd.to_numeric(runs['held_out_rank_percentile'], errors='coerce').max()
    n_queries = pd.to_numeric(runs.get('held_out_n_queries'), errors='coerce').max()
    if not np.isfinite(best) or not np.isfinite(n_queries) or n_queries < 2:
        return None

    total = float(np.log2(n_queries))
    # Rank percentile p means the true sentence beats a fraction p of the gallery, so it survives (1-p) of it.
    encoder = max(total - float(np.log2(max((1.0 - best) * n_queries, 1.0))), 0.0)
    length = min(_LENGTH_BITS, total)
    fig = go.Figure(
        go.Pie(
            labels=['sentence length (free)', 'encoder, beyond length', 'still missing'],
            values=[length, max(encoder - length, 0.0), max(total - max(encoder, length), 0.0)],
            marker={'colors': ['#b08968', '#0072b2', '#dcdcdc']},
            hole=0.45,
            hovertemplate='%{label}<br>%{value:.2f} bits<extra></extra>',
        )
    )
    return _layout(fig, f'Bit budget — {total:.2f} bits needed to name one of {int(n_queries)} sentences', height=460)


def seed_histogram(study: Study, metric: str = 'held_out_top1') -> 'Figure | None':
    """Distribution of one metric over every run in the study, against its chance level.

    Args:
        study (Study): The collected study.
        metric (str, optional): Metric to bin. Defaults to 'held_out_top1'.

    Returns:
        Figure | None: The histogram, or `None` when the metric was not collected.

    Note:
        Bimodality here is the finding, not the noise: a sweep where some seeds train a healthy code and others
        collapse averages to a number that describes neither.
    """
    go = _go()
    runs = study.runs
    if runs.empty or metric not in runs.columns:
        return None

    values = pd.to_numeric(runs[metric], errors='coerce').dropna()
    if values.empty:
        return None

    fig = go.Figure(go.Histogram(x=values, nbinsx=max(6, min(24, len(values))), marker_color='#0072b2'))
    chance = pd.to_numeric(runs.get('held_out_chance'), errors='coerce').dropna()
    if metric.endswith('top1') and not chance.empty:
        fig.add_vline(x=float(chance.mean()), line={'dash': 'dash', 'color': '#d55e00'}, annotation_text='chance')
    return _layout(fig, f'{metric.replace("_", " ")} across every run', xaxis_title=metric, yaxis_title='runs')


def scalp_3d(
    values: np.ndarray | None = None, montage_csv: str | None = None, n_channels: int = 105
) -> 'Figure | None':
    """A 3-D electrode map on the scalp sphere, coloured by whatever per-channel quantity is supplied.

    Args:
        values (np.ndarray | None, optional): One value per electrode. Defaults to None, which colours by the
            anterior-posterior axis instead so the geometry is still readable.
        montage_csv (str | None, optional): A `channel,x,y,z` montage CSV. Defaults to None, which uses the
            coordinate-free fallback and says so in the title.
        n_channels (int, optional): Electrode count when `values` does not imply one. Defaults to 105.

    Returns:
        Figure | None: The 3-D scatter, or `None` when no geometry could be built.

    Note:
        The fallback geometry is a Fibonacci spiral on the sphere, not a real montage. It is labelled `approximate`
        in the title for the same reason `ScalpGeometry.approximate` exists: the maths downstream is exact either
        way, but the picture is only anatomy when the coordinates are.
    """
    go = _go()
    from zte.models.spatial import ScalpGeometry

    count = len(values) if values is not None else n_channels
    try:
        geometry = (
            ScalpGeometry.from_csv(montage_csv, count) if montage_csv else ScalpGeometry.fibonacci_fallback(count)
        )
    except (OSError, ValueError) as exc:
        _LOG.warning('No scalp geometry available for the 3-D map (%r).', exc)
        return None

    xyz = np.asarray(geometry.xyz, dtype=np.float64)
    labels = geometry.labels or tuple(f'ch{i:03d}' for i in range(len(xyz)))
    colour = np.asarray(values, dtype=np.float64) if values is not None else xyz[:, 1]
    fig = go.Figure(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode='markers',
            marker={
                'size': 6,
                'color': colour,
                'colorscale': SEQUENTIAL,
                'colorbar': {'title': 'value' if values is not None else 'anterior→posterior'},
                'opacity': 0.9,
            },
            text=list(labels),
            hovertemplate='%{text}<br>%{marker.color:.4f}<extra></extra>',
        )
    )
    fig.update_scenes(
        aspectmode='data',
        xaxis_title='x (right)',
        yaxis_title='y (front)',
        zaxis_title='z (up)',
    )
    suffix = ' (approximate geometry)' if getattr(geometry, 'approximate', False) else ''
    return _layout(fig, f'Electrode map{suffix}', height=560)
