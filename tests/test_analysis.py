"""The study analysis: collecting many runs, aggregating over seeds, and drawing the whole picture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from zte.evaluation.analysis import (
    build_dashboard,
    collect_study,
    feature_ablation_table,
    loso_table,
    multi_seed_table,
    summary_markdown,
    within_task_table,
    write_summary,
    write_tables,
)
from zte.evaluation.analysis import figures as F


def _write_run(
    root: Path,
    name: str,
    *,
    rank_percentile: float,
    top1: float,
    frontend: str = 'raw_conformer',
    holdout: str = 'ZAB',
    seed: int = 42,
    real: bool = True,
    within: dict[str, dict[str, float]] | None = None,
) -> Path:
    """Fabricates one evaluated run directory in the layout `zte-run` writes."""
    run = root / name
    (run / 'evaluation').mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        'scoreboard': {
            'held_out_retrieval': {
                'rank_percentile': rank_percentile,
                'top1': top1,
                'top5': top1 * 4,
                'chance_top1': 0.0014,
                'n_queries': 700,
            },
            'held_out_retrieval_length_stratified': {'top1': top1 * 3, 'rank_percentile': rank_percentile - 0.04},
            'lift_over_raw': {
                'content_probe': {'raw_content_r2_best': 0.03, 'passes': True},
                'subject': {'zte_linear': 0.42, 'raw_linear': 0.81},
                'word_len': {'zte_linear': 0.025, 'raw_linear': -0.006},
            },
        },
        'length_projection': {
            'status': 'applied',
            'n_fit': 500,
            'length_leakage_before': 0.084,
            'length_leakage_after': 0.011,
        },
        'emergence': {
            'cross_subject': {'same_word': {'gap': 0.005}, 'same_meaning': {'gap': 0.028}},
            'neighbourhood': {'same_word_purity': 0.004, 'cross_subject_neighbour_fraction': 0.768},
        },
        'sentence_retrieval': {'top1': top1 / 2},
        'word_retrieval': {'top1': 0.004, 'chance_top1': 0.0031},
        'embedding_health': {'effective_rank_ratio': 0.25, 'anisotropy': 0.03, 'uniformity': -3.8},
        'probe_comparison': [
            {
                'target': 'subject',
                'representation': 'ZTE',
                'metric': 'accuracy',
                'linear_score': 0.42,
                'knn_score': 0.26,
            },
            {'target': 'word_len', 'representation': 'ZTE', 'metric': 'R2', 'linear_score': 0.028, 'knn_score': -0.06},
        ],
        'per_subject': [{'subject': holdout, 'retrieval_top1': top1, 'eff-rank ratio': 0.23}],
        'history': {
            'train_loss': [1.0, 0.6, 0.4],
            'val_loss': [1.1, 0.7, 0.65],
            'train_consensus_sentence_pull': [0.9, 0.5, 0.3],
            'train_gallery_top1': [0.001, 0.02, 0.05],
            'train_residual_context_explained': [0.1, 0.3, 0.35],
        },
    }
    (run / 'evaluation' / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    (run / 'manifest.json').write_text(
        json.dumps({'data_root': '/data/zuco' if real else '/tmp/synthetic_zuco', 'dataset': {'n_words': 160804}}),
        encoding='utf-8',
    )
    (run / 'config.yaml').write_text(
        yaml.safe_dump(
            {
                'model': {
                    'frontend': frontend,
                    'spatial_encoding': 'spherical_harmonics',
                    'residual_coding': True,
                },
                'objective': {
                    'name': 'clip',
                    'lexical_weight': 1.0,
                    'identity_orthogonality_weight': 1.0,
                    'consensus_weight': 1.0,
                    'gallery_weight': 1.0,
                    'gallery_length_band': 2,
                    'length_projection': True,
                },
                'dataset': {'raw_align': 'euclidean'},
                'train': {'seed': seed, 'loso_holdout_subject': holdout, 'split': 'by_subject_loso', 'mode': 'encoder'},
                'run_name': name,
            }
        ),
        encoding='utf-8',
    )
    if within is not None:
        (run / 'evaluation' / 'generation.json').write_text(
            json.dumps({'rescoring': {'top1': top1, 'within_task': within}, 'generation': {'applicable': True}}),
            encoding='utf-8',
        )
    return run


@pytest.fixture()
def study_root(tmp_path: Path) -> Path:
    """A tree of six fabricated runs: two arms, three seeds each, one arm on a different frontend."""
    root = tmp_path / 'experiments'
    for seed, rank in ((42, 0.967), (43, 0.960), (44, 0.972)):
        _write_run(root, f'exp14_lexical_loZAB_s{seed}', rank_percentile=rank, top1=0.011, seed=seed)
    for seed, rank in ((42, 0.930), (43, 0.925), (44, 0.940)):
        _write_run(
            root,
            f'exp14_bandpower_loZAB_s{seed}',
            rank_percentile=rank,
            top1=0.004,
            frontend='band_power_mlp',
            seed=seed,
        )
    return root


def test_a_study_collects_every_evaluated_run(study_root: Path) -> None:
    """A run without `evaluation/metrics.json` has not been evaluated and must not appear as a data point."""
    (study_root / 'half_finished' / 'checkpoints').mkdir(parents=True)

    study = collect_study(study_root)

    assert len(study.runs) == 6
    assert set(study.runs['arm']) == {'exp14_lexical', 'exp14_bandpower'}
    assert set(study.runs['seed']) == {42, 43, 44}
    assert set(study.runs['holdout']) == {'ZAB'}


def test_the_headline_is_reported_as_mean_plus_minus_sd_over_seeds(study_root: Path) -> None:
    """Run-to-run drift here has been the size of the effect, so a single-seed number is not a result."""
    table = multi_seed_table(collect_study(study_root))

    lexical = table[table['arm'] == 'exp14_lexical'].iloc[0]
    assert lexical['n_seeds'] == 3
    assert lexical['held_out_rank_percentile_mean'] == pytest.approx((0.967 + 0.960 + 0.972) / 3)
    assert lexical['held_out_rank_percentile_sd'] > 0.0
    assert lexical['held_out_rank_percentile_lo'] < lexical['held_out_rank_percentile_mean']


def test_a_single_seed_arm_reports_no_spread_rather_than_a_fake_one(tmp_path: Path) -> None:
    """One run cannot have a standard deviation, and inventing 0.0 would read as perfect stability."""
    root = tmp_path / 'one'
    _write_run(root, 'solo_loZAB_s42', rank_percentile=0.9, top1=0.01)

    row = multi_seed_table(collect_study(root)).iloc[0]

    assert row['n_seeds'] == 1
    assert row['held_out_rank_percentile_sd'] != row['held_out_rank_percentile_sd']  # nan


def test_the_feature_ablation_table_pivots_on_one_lever_at_a_time(study_root: Path) -> None:
    """The table the decoder chapter is built on: raw vs band power, with the run count carried."""
    table = feature_ablation_table(collect_study(study_root))

    frontend = table[table['lever'] == 'frontend']
    assert set(frontend['level']) == {'raw_conformer', 'band_power_mlp'}
    assert set(frontend['n_runs']) == {3}
    raw = frontend[frontend['level'] == 'raw_conformer'].iloc[0]
    assert raw['held_out_rank_percentile_mean'] > 0.9


def test_a_lever_with_one_level_is_not_reported_as_an_ablation(study_root: Path) -> None:
    """Every collected run uses spherical harmonics, so there is no harmonics-vs-indexing comparison to make."""
    table = feature_ablation_table(collect_study(study_root))

    assert 'spatial_encoding' not in set(table['lever'])


def test_the_loso_table_is_one_row_per_arm_and_held_out_subject(study_root: Path) -> None:
    """The honest trend is per-fold; pooling the folds is what hid a bimodal sweep before."""
    table = loso_table(collect_study(study_root))

    assert len(table) == 2
    assert set(table['holdout']) == {'ZAB'}
    assert all(n == 3 for n in table['n_seeds'])


def test_within_task_pools_carry_their_own_chance_level(tmp_path: Path) -> None:
    """A smaller pool has a higher chance level, so a Top-1 quoted without it is unreadable."""
    root = tmp_path / 'within'
    _write_run(
        root,
        'arm_loZAB_s42',
        rank_percentile=0.95,
        top1=0.02,
        within={
            'SR': {'top1': 0.05, 'chance_top1': 0.0025, 'rank_percentile': 0.94, 'n_candidates': 400},
            'NR': {'top1': 0.03, 'chance_top1': 0.0033, 'rank_percentile': 0.92, 'n_candidates': 300},
        },
    )

    table = within_task_table(collect_study(root))

    assert set(table['task']) == {'SR', 'NR'}
    sr = table[table['task'] == 'SR'].iloc[0]
    assert sr['chance'] == pytest.approx(0.0025)
    assert sr['lift'] == pytest.approx(0.05 - 0.0025)


def test_a_synthetic_run_is_named_as_one_in_the_summary(tmp_path: Path) -> None:
    """Synthetic and real are not the same kind of evidence, and the distinction has to survive into the prose."""
    root = tmp_path / 'mixed'
    _write_run(root, 'real_loZAB_s42', rank_percentile=0.9, top1=0.01, real=True)
    _write_run(root, 'fake_loZAB_s43', rank_percentile=0.9, top1=0.01, real=False, seed=43)

    text = summary_markdown(collect_study(root))

    assert 'SYNTHETIC' in text
    assert '1 of 2' in text


def test_an_empty_tree_produces_an_analysis_that_says_so(tmp_path: Path) -> None:
    """A study run before anything finished must write a readable page, not crash the last stage of the script."""
    study = collect_study(tmp_path / 'nothing')

    assert study.is_empty
    page = build_dashboard(study, tmp_path / 'out' / 'ANALYSIS.html')

    assert page.is_file()
    assert 'No evaluated run was found' in page.read_text(encoding='utf-8')


def test_the_dashboard_is_self_contained(study_root: Path, tmp_path: Path) -> None:
    """It has to open from a Drive mirror on a machine with no network, so nothing may be fetched at view time.

    Note:
        The check is on the *tags*, not on the text. Plotly's bundle mentions its own CDN as a config default, so a
        substring search for the host name reports a fetch that never happens.
    """
    import re

    page = build_dashboard(collect_study(study_root), tmp_path / 'ANALYSIS.html')
    html = page.read_text(encoding='utf-8')

    assert '<script' in html and 'Plotly' in html
    remote = re.findall(r'<(?:script|link|img|iframe)\b[^>]*\b(?:src|href)\s*=\s*["\'](https?:|//)', html, re.I)
    assert not remote, f'the page fetches {len(remote)} external resource(s) at view time'


def test_every_frame_is_written_as_a_table(study_root: Path, tmp_path: Path) -> None:
    """The analysis has to be redoable in another tool, so the tidy frames leave as CSV alongside the page."""
    written = write_tables(collect_study(study_root), tmp_path / 'tables')

    names = {p.stem for p in written}
    assert {'runs', 'multi_seed', 'feature_ablation', 'probes'} <= names
    assert all(p.is_file() and p.stat().st_size > 0 for p in written)


def test_the_summary_names_where_the_runs_came_from(study_root: Path, tmp_path: Path) -> None:
    """Provenance travels with the artifact: a table with no root is a table nobody can re-derive."""
    out = write_summary(collect_study(study_root), tmp_path / 'ANALYSIS.md')

    assert str(study_root) in out.read_text(encoding='utf-8')


# -- The exp16 mechanism panels ---------------------------------------------- #


def test_the_metric_explorer_offers_every_collected_headline(study_root: Path) -> None:
    """One selector over all headlines is the panel a reader opens first, so it must carry more than one trace."""
    figure = F.metric_explorer(collect_study(study_root))

    assert figure is not None
    assert len(figure.data) > 1
    assert figure.layout.updatemenus, 'without the dropdown it is just the first metric'


def test_the_mechanism_curves_read_the_training_history(study_root: Path) -> None:
    """A mechanism that never engaged is invisible in the final metrics and visible only here."""
    figure = F.mechanism_curves(collect_study(study_root))

    assert figure is not None
    labels = {button['label'] for button in figure.layout.updatemenus[0]['buttons']}
    assert {'consensus sentence pull', 'gallery top1', 'residual context explained'} <= labels


def test_the_leakage_panel_shows_before_beside_after(study_root: Path) -> None:
    """A single "after" bar cannot say whether the projection removed anything."""
    figure = F.length_leakage_bars(collect_study(study_root))

    assert figure is not None
    assert {trace.name for trace in figure.data} == {'before', 'after'}


def test_the_leakage_panel_is_absent_when_no_run_projected(tmp_path: Path) -> None:
    """A panel drawn from missing data would imply the de-confounding ran when it did not."""
    root = tmp_path / 'plain'
    run = _write_run(root, 'plain_loZAB_s42', rank_percentile=0.9, top1=0.01)
    metrics = json.loads((run / 'evaluation' / 'metrics.json').read_text())
    del metrics['length_projection']
    (run / 'evaluation' / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')

    assert F.length_leakage_bars(collect_study(root)) is None


def test_the_who_versus_what_panel_marks_the_raw_baseline(study_root: Path) -> None:
    """A low subject probe means nothing without the raw number beside it -- it can be collapse rather than removal."""
    figure = F.identity_vs_content(collect_study(study_root))

    assert figure is not None
    assert any('raw features' in str(annotation.text) for annotation in figure.layout.annotations)
    assert float(figure.data[0].x[0]) == pytest.approx(0.42)


def test_the_mechanism_matrix_shows_one_row_per_lever(study_root: Path) -> None:
    """Two identical columns mean two identical runs, whatever the file names claim."""
    figure = F.mechanism_matrix(collect_study(study_root))

    assert figure is not None
    assert 'residual coding' in set(figure.data[0].y)
    assert 'gallery length band' in set(figure.data[0].y)


def test_the_bit_budget_pie_splits_length_from_the_encoder(study_root: Path) -> None:
    """The point of the chart is that 5.14 of the 9.45 bits are free, so the slices must name that split."""
    figure = F.bit_budget_pie(collect_study(study_root))

    assert figure is not None
    assert 'sentence length (free)' in set(figure.data[0].labels)
    assert sum(figure.data[0].values) == pytest.approx(np.log2(700), rel=1e-6)


def test_the_seed_histogram_marks_chance(study_root: Path) -> None:
    """Six Top-1 values around 1% mean nothing until the reader can see where 1/700 sits."""
    figure = F.seed_histogram(collect_study(study_root))

    assert figure is not None
    assert any('chance' in str(shape) for shape in figure.layout.annotations)
