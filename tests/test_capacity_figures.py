"""The decoder-capacity figures: the certifying gap, the bits ledger, the seed strip and the paired picture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from zte.evaluation import plots as P
from zte.evaluation.analysis import Study, collect_study, panel_builders
from zte.evaluation.analysis import figures as F
from zte.evaluation.audit.capacity import capacity_report, pooled_capacity

# Sixteen stimuli read once each by a training subject and once by the holdout: enough that a pool fills a
# 16-way menu and an exact sign test over unanimous wins clears any sane alpha.
GALLERY: int = 16
"""Distinct stimuli in every hand-built gallery below."""

# The sweep deliberately runs past what the pools can fill, because the real one does: exact word-count pools
# hold a median of eight candidates, so the largest sizes come back unreachable rather than failed.
SWEEP: tuple[int, ...] = (2, 4, 8, 16, 32)
"""Menu sizes every report in this module sweeps."""


def _report(model: np.ndarray, *, seed: int = 0, **overrides: Any) -> dict[str, Any]:
    """A capacity report over the standard synthetic gallery, on an honest split."""
    content = np.concatenate([np.arange(GALLERY), np.arange(GALLERY)])
    subjects = np.array(['A'] * GALLERY + ['B'] * GALLERY)
    arms = {
        'model': model,
        'length_only': np.zeros((GALLERY, GALLERY)),
        'shuffled_eeg': np.zeros((GALLERY, GALLERY)),
        'mismatch': np.zeros((GALLERY, GALLERY)),
    }
    kwargs: dict[str, Any] = {
        'tasks': np.array(['NR'] * (2 * GALLERY)),
        'ks': SWEEP,
        'n_perm': 200,
        'n_boot': 200,
        'seed': seed,
        'honest_split': True,
        'split_strategy': 'by_subject_and_stimulus',
        'split_cell': 'test',
    }
    kwargs.update(overrides)
    report = capacity_report(arms, content, subjects, 'B', np.full(2 * GALLERY, 9.0), **kwargs)
    assert report is not None

    return report


@pytest.fixture
def certifying() -> dict[str, Any]:
    """A decoder whose EEG prefix wins every query while every control loses it."""
    return _report(np.eye(GALLERY))


@pytest.fixture
def null_result() -> dict[str, Any]:
    """A decoder that separates nothing, so no menu size certifies."""
    return _report(np.zeros((GALLERY, GALLERY)), seed=3)


def _texts(figure: Figure) -> str:
    """Every string a figure draws: title, annotations, tick labels and legend entries."""
    axes: Axes = figure.axes[0]
    legend = axes.get_legend()
    pieces = [axes.get_title(), axes.get_xlabel(), axes.get_ylabel()]
    pieces += [text.get_text() for text in axes.texts]
    pieces += [label.get_text() for label in axes.get_xticklabels() + axes.get_yticklabels()]
    pieces += [] if legend is None else [text.get_text() for text in legend.get_texts()]

    return '\n'.join(pieces)


def _bars(figure: Figure) -> list[Rectangle]:
    """Every drawn bar of a figure."""
    return [patch for patch in figure.axes[0].patches if isinstance(patch, Rectangle)]


def _line(figure: Figure, label: str) -> Any:
    """The drawn line carrying one legend label."""
    matched = [line for line in figure.axes[0].get_lines() if line.get_label() == label]
    assert matched, f'no line labelled {label!r}'

    return matched[0]


def test_capacity_curve_draws_every_arm_against_a_moving_chance(certifying: dict[str, Any]) -> None:
    """Chance is a curve at 1/K rather than one horizontal line, and every conditioning arm is drawn."""
    figure = P.capacity_curve(certifying)
    drawn = {line.get_label() for line in figure.axes[0].get_lines()}

    assert {'model (EEG prefix)', 'length-only prefix', 'shuffled EEG (derangement)', 'mismatched stimulus'} <= drawn

    chance = _line(figure, 'chance = 1/K (moves with K)')
    feasible = [k for k in SWEEP if k <= GALLERY]

    assert list(chance.get_xdata()) == list(SWEEP)
    assert [round(float(y), 6) for y in chance.get_ydata()] == [round(1.0 / k, 6) for k in SWEEP]
    assert len(set(chance.get_ydata())) == len(SWEEP)
    assert figure.axes[0].get_xscale() == 'log'
    assert all(f'n={GALLERY}' in _texts(figure) for _ in feasible)


def test_capacity_curve_ribbons_only_the_model_and_the_length_control(certifying: dict[str, Any]) -> None:
    """Four intervals would hide the gap, so only the claim and the control deciding it carry a ribbon."""
    figure = P.capacity_curve(certifying)

    # The certifying gap plus one ribbon each for `model` and `length_only`; the other two arms carry none.
    assert len(figure.axes[0].collections) == 3
    assert 'model over length-only — the certifying gap' in _texts(figure)


def test_capacity_curve_greys_the_sizes_no_pool_could_fill(certifying: dict[str, Any]) -> None:
    """A menu no candidate pool can fill is labelled unreachable, never left to read as a failure."""
    block = certifying['scores']['pmi']['length_task_matched']

    assert block['ks_unreachable'] == [32]
    figure = P.capacity_curve(certifying)

    assert 'no pool could fill this menu' in _texts(figure)
    assert figure.axes[0].patches, 'the unreachable size is not shaded'
    assert 'n=0' in _texts(figure)


def test_capacity_curve_names_the_certified_size_and_the_paired_gap(certifying: dict[str, Any]) -> None:
    """The certified size and the paired delta over the length control are both written on the panel."""
    text = _texts(P.capacity_curve(certifying))

    assert 'certified K = 16' in text
    assert 'over length-only' in text
    assert 'sign-test p' in text


def test_capacity_curve_renders_an_uncertified_run_as_an_em_dash_with_its_clause(null_result: dict[str, Any]) -> None:
    """Nothing certified renders as an em dash naming a failing clause, never as a blank and never as a zero."""
    assert null_result['certified_k'] is None
    text = _texts(P.capacity_curve(null_result))

    assert 'certified K = —' in text
    assert 'above_chance' in text
    assert 'certified K = 0' not in text


def test_capacity_bits_ledger_credits_against_the_residual_not_the_total(certifying: dict[str, Any]) -> None:
    """The certified bits are stacked on top of what word count gives free, inside the 4.3090-bit residual."""
    figure = P.capacity_bits_ledger(certifying)
    bars = sorted(_bars(figure), key=lambda bar: bar.get_x())
    text = _texts(figure)

    assert [round(bar.get_x(), 4) for bar in bars] == [0.0, 5.1422, 9.1422]
    assert round(bars[0].get_width(), 4) == 5.1422
    assert round(bars[1].get_width(), 4) == 4.0
    assert '4.3090 bits of identity left once word count is known' in text
    assert '9.4512 — full stimulus identity' in text
    assert '92.8% of the residual' in text


def test_capacity_bits_ledger_hatches_and_dashes_a_run_that_certified_nothing(null_result: dict[str, Any]) -> None:
    """A run with no certified size keeps its ledger, with an em dash and every failing clause spelled out."""
    figure = P.capacity_bits_ledger(null_result)
    bars = sorted(_bars(figure), key=lambda bar: bar.get_x())
    text = _texts(figure)

    assert len(bars) == 2
    assert bars[1].get_hatch() == '//'
    assert round(bars[1].get_width(), 4) == 4.3090
    assert 'certified K = —' in text
    assert 'recovered nothing of the 4.3090-bit residual' in text
    assert 'permutation_significant' in text


def test_capacity_seed_strip_keeps_every_run_as_its_own_point(
    certifying: dict[str, Any], null_result: dict[str, Any]
) -> None:
    """Each run is a point with its own interval, over the interval the runs jointly support."""
    pooled = pooled_capacity([certifying, null_result])
    figure = P.capacity_seed_strip([certifying, null_result], pooled=pooled)
    points = sum(
        int(np.asarray(collection.get_offsets()).shape[0])
        for collection in figure.axes[0].collections
        if isinstance(collection, PathCollection)
    )
    text = _texts(figure)

    assert points == 2
    assert 'run certified a menu size' in text
    assert 'run certified nothing' in text
    assert 'pooled capacity is none' in text
    assert 'one seed is a measurement' not in text


def test_capacity_seed_strip_flags_a_single_seed(certifying: dict[str, Any]) -> None:
    """One run is labelled a measurement rather than a result, because drift here has been the size of the effect."""
    assert 'one seed is a measurement, not yet a result' in _texts(P.capacity_seed_strip([certifying]))


def test_capacity_vs_length_oracle_shows_the_pairing_not_the_average(certifying: dict[str, Any]) -> None:
    """Every menu size is drawn as its own wins, losses and ties, with the exact sign-test p beside it."""
    figure = P.capacity_vs_length_oracle(certifying)
    text = _texts(figure)

    # Three bars per scored size -- ties, model wins, control wins -- for the four sizes a pool could fill.
    assert len(figure.axes[0].patches) == 3 * 4
    assert 'K = 2 (certified)' in text
    assert f'{GALLERY} pairs' in text
    assert 'ties (count as losses)' in text
    assert 'clause holds' in text
    assert 'p = ' in text


def test_capacity_vs_length_oracle_marks_a_failing_clause(null_result: dict[str, Any]) -> None:
    """A control the model does not beat is written as a failing clause rather than quietly dropped."""
    text = _texts(P.capacity_vs_length_oracle(null_result))

    assert 'clause fails' in text
    assert 'clause holds' not in text


def test_every_capacity_figure_degrades_to_a_placeholder() -> None:
    """An absent report draws a centred message, never an empty frame a reader could mistake for a null result."""
    for figure, message in (
        (P.capacity_curve({}), 'no capacity report'),
        (P.capacity_bits_ledger({}), 'no bits ledger'),
        (P.capacity_seed_strip([]), 'no run scored a 2-way menu'),
        (P.capacity_vs_length_oracle({}), 'no paired comparison against length_only'),
    ):
        assert message in _texts(figure)
        assert not figure.axes[0].get_lines()


# ---- The tidy frame and the Plotly panels ---- #


def _write_run(root: Path, name: str, capacity: dict[str, Any], *, in_metrics: bool) -> None:
    """Fabricates one evaluated run directory carrying a capacity report."""
    run = root / name
    (run / 'evaluation').mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        'scoreboard': {'held_out_retrieval': {'rank_percentile': 0.7, 'top1': 0.01, 'chance_top1': 0.0014}},
        'verdict': {},
    }
    if in_metrics:
        metrics['decoder_capacity'] = capacity
    else:
        (run / 'evaluation' / 'capacity.json').write_text(json.dumps({'capacity': capacity}), encoding='utf-8')
    (run / 'evaluation' / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    (run / 'manifest.json').write_text(json.dumps({'synthetic': False, 'data_root': '/data'}), encoding='utf-8')


@pytest.fixture
def study(tmp_path: Path, certifying: dict[str, Any], null_result: dict[str, Any]) -> Study:
    """A two-run study: one run certifies, one does not, and only one wrote its report into `metrics.json`."""
    _write_run(tmp_path, 'run_strong', certifying, in_metrics=True)
    _write_run(tmp_path, 'run_weak', null_result, in_metrics=False)

    return collect_study(tmp_path)


def test_capacity_frame_carries_one_row_per_arm_and_menu_size(study: Study) -> None:
    """The tidy frame keeps the documented columns, and an unreachable size stays in it with no accuracy."""
    frame = study.capacity
    expected = {
        'run',
        'holdout',
        'seed',
        'score',
        'flavor',
        'headline',
        'alpha',
        'certifiable',
        'gamed',
        'certified_k',
        'subset',
        'k',
        'feasible',
        'arm',
        'accuracy',
        'ci_lo',
        'ci_hi',
        'chance',
        'n_queries',
        'perm_p',
        'perm_p_floor',
        'certified',
        'failed_clauses',
        'delta',
        'delta_lo',
        'delta_hi',
        'sign_test_p',
        'model_wins',
        'control_wins',
        'ties',
        'n_pairs',
    }

    assert expected <= set(frame.columns)
    assert set(frame['subset'].unique()) == {'per_k', 'common_subset'}

    headline = frame[frame['headline'] & (frame['subset'] == 'per_k') & (frame['run'] == 'run_strong')]

    assert set(headline[headline['k'] == 2]['arm']) == {'model', 'length_only', 'shuffled_eeg', 'mismatch'}
    assert float(headline[(headline['k'] == 2) & (headline['arm'] == 'model')]['accuracy'].iloc[0]) == 1.0
    assert float(headline[(headline['k'] == 2) & (headline['arm'] == 'length_only')]['delta'].iloc[0]) == 1.0

    unreachable = headline[headline['k'] == 32]

    assert not bool(unreachable['feasible'].iloc[0])
    assert bool(unreachable['accuracy'].isna().all())


def test_capacity_headlines_read_a_decode_only_run_from_its_own_json(study: Study) -> None:
    """A run that wrote only `evaluation/capacity.json` still lands in the run frame with its capacity headlines."""
    runs = study.runs.set_index('run')

    assert float(runs.loc['run_strong', 'capacity_k']) == 16.0
    assert float(runs.loc['run_strong', 'capacity_bits']) == 4.0
    assert bool(runs.loc['run_strong', 'capacity_certified'])
    assert runs.loc['run_weak', 'capacity_readout'] == 'menu selection'
    assert pd.isna(runs.loc['run_weak', 'capacity_k'])
    assert not bool(runs.loc['run_weak', 'capacity_certified'])


def test_plotly_capacity_panels_build_from_the_frame(study: Study) -> None:
    """Each dashboard panel draws the arms, the moving chance level and the denominator it credits against."""
    curve = F.capacity_curve(study)
    assert curve is not None
    names = {trace.name for trace in curve.data}

    assert 'model (EEG prefix)' in names
    assert 'length-only prefix' in names
    assert 'chance = 1/K (moves with K)' in names
    assert 'model over length-only — the certifying gap' in names
    assert curve.layout.xaxis.type == 'log'

    ledger = F.capacity_bits_ledger(study)
    assert ledger is not None

    assert {trace.name for trace in ledger.data} == {
        'word count, free — 5.1422 bits',
        'decoder certified (menu selection)',
        'unrecovered',
    }
    assert '—' in list(ledger.data[1].text)
    assert any('4.3090' in (annotation.text or '') for annotation in ledger.layout.annotations)
    assert any('nothing certified' in (annotation.text or '') for annotation in ledger.layout.annotations)

    strip = F.capacity_seed_strip(study)
    assert strip is not None

    assert sum(len(trace.x) for trace in strip.data) == 2


def test_plotly_capacity_panels_return_none_without_a_capacity_report() -> None:
    """No capacity report is not a blank chart: the panel declines to draw and the page leaves it out."""
    empty = Study()

    assert F.capacity_curve(empty) is None
    assert F.capacity_bits_ledger(empty) is None
    assert F.capacity_seed_strip(empty) is None


def test_the_dashboard_registers_the_capacity_panels(study: Study) -> None:
    """The three panels are on the page, in the decoder section, each with a caption saying what to read first."""
    panels = {panel.name: panel for panel in panel_builders(study)}

    assert {'capacity_curve', 'capacity_bits_ledger', 'capacity_seed_strip'} <= set(panels)
    assert all(panels[name].section == 'decoder' for name in ('capacity_curve', 'capacity_bits_ledger'))
    assert 'length-only trace' in panels['capacity_curve'].caption
    assert '4.31' in panels['capacity_bits_ledger'].caption
    assert all(panels[name].build() is not None for name in ('capacity_curve', 'capacity_seed_strip'))
