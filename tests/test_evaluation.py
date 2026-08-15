"""Tests for the nearest-neighbour index and the evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from zte.config import ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.evaluation import metrics as M
from zte.evaluation.report import _postprocess, _postprocess_fit, evaluate_representation
from zte.inference.embed import ZTEEmbedder
from zte.inference.retrieval import NearestNeighborIndex
from zte.training.pipeline import run_training


def _clustered(n_groups: int, per_group: int, dim: int, scale: float = 0.05) -> tuple:
    """Builds clustered embeddings: one centre per group + small noise."""
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(n_groups, dim)).astype(np.float32)
    groups = np.repeat(np.arange(n_groups), per_group)
    emb = centres[groups] + rng.normal(scale=scale, size=(n_groups * per_group, dim)).astype(np.float32)
    return emb, groups


def test_nn_index_query_predict_decode() -> None:
    """The index returns correctly shaped neighbours and sane predictions."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 8)).astype(np.float32)
    meta = pd.DataFrame(
        {
            'word': rng.choice(['cat', 'dog', 'sky'], size=60),
            'val': rng.normal(size=60),
            'cls': rng.integers(0, 2, size=60),
        }
    )
    index = NearestNeighborIndex(emb, meta)
    idx, sim = index.query(emb[:5], k=4)
    assert idx.shape == (5, 4) and sim.shape == (5, 4)

    reg = index.predict(emb[:5], 'val', k=5, task='regression')
    assert reg.shape == (5,) and np.isfinite(reg).all()
    clf = index.predict(emb[:5], 'cls', k=5, task='classification')
    assert set(np.unique(clf)).issubset({0, 1})
    assert len(index.decode(emb[:3], column='word')) == 3


def test_effective_rank_detects_collapse() -> None:
    """Effective rank is high for spread data and ~1 for collapsed (rank-1) variation."""
    rng = np.random.default_rng(0)
    spread = rng.normal(size=(300, 16)).astype(np.float32)
    # Collapse = all variation along a single direction (a line in embedding space).
    direction = rng.normal(size=(1, 16)).astype(np.float32)
    scores = rng.normal(size=(300, 1)).astype(np.float32)
    collapsed = scores @ direction
    assert M.effective_rank(spread) > 8.0
    assert M.effective_rank(collapsed) < 2.0


def test_embedding_health_keys() -> None:
    """Health report exposes the expected fields."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 12)).astype(np.float32)
    health = M.embedding_health(emb)
    for key in (
        'effective_rank',
        'effective_rank_ratio',
        'uniformity',
        'anisotropy',
        'mean_norm',
        'dead_dim_fraction',
    ):
        assert key in health


def test_content_retrieval_above_chance() -> None:
    """Clustered same-content points retrieve each other well above chance."""
    emb, groups = _clustered(n_groups=6, per_group=10, dim=8, scale=0.02)
    res = M.content_retrieval(emb, groups)
    assert res['top1'] > 0.8
    assert res['top1'] > res['chance_top1']


def test_bootstrap_ci_brackets_mean() -> None:
    """Bootstrap CI returns the mean as point and a lo <= point <= hi interval."""
    rng = np.random.default_rng(0)
    values = rng.normal(loc=1.5, scale=0.3, size=400)
    point, lo, hi = M.bootstrap_ci(values, n_boot=500, seed=0)
    assert abs(point - float(values.mean())) < 1e-9
    assert lo <= point <= hi
    # A tight interval around the true mean for a large, low-variance sample.
    assert lo < 1.5 < hi
    # Degenerate inputs are handled: constant -> zero-width, empty -> nan.
    p, clo, chi = M.bootstrap_ci(np.full(20, 2.0))
    assert p == clo == chi == 2.0
    assert all(np.isnan(v) for v in M.bootstrap_ci(np.array([])))


def test_query_weighted_chance() -> None:
    """Retrieval chance is query-weighted, differing from the type-weighted legacy."""
    rng = np.random.default_rng(0)
    # Group sizes 3, 2, 1 over n = 6 rows (the singleton cannot be a query).
    group_ids = np.array([0, 0, 0, 1, 1, 2])
    emb = rng.normal(size=(6, 4)).astype(np.float32)
    res = M.content_retrieval(emb, group_ids, return_hits=True)
    # sum_g g(g-1) / sum_g g / (n-1) over groups with g > 1 = (6 + 2) / 5 / 5 = 0.32.
    assert abs(res['chance_top1'] - 0.32) < 1e-9
    # Legacy mean((counts-1)/(n-1)) over all groups = mean(0.4, 0.2, 0.0) = 0.2.
    assert abs(res['chance_top1_typeweighted'] - 0.2) < 1e-9
    assert res['chance_top1'] != res['chance_top1_typeweighted']
    assert len(res['top1_hits']) == int(res['n_queries']) == 5


def test_task_transfer_disjoint_is_not_applicable() -> None:
    """Disjoint per-task stimuli make task-transfer not-applicable, not a bare NaN."""
    from zte.evaluation.analogy import analogy_report

    rng = np.random.default_rng(0)
    # Two subjects, two tasks; each task reads its own sentence indices (disjoint).
    rows = []
    for subject in ('A', 'B'):
        for task, sent_ids in (('SR', (0, 1)), ('NR', (2, 3))):
            for s in sent_ids:
                for w in range(3):
                    rows.append(
                        {
                            'subject': subject,
                            'task': task,
                            'sentence_idx': s,
                            'word_idx': w,
                            'word': f'{task}{s}{w}',
                        }
                    )
    meta = pd.DataFrame(rows)
    emb = rng.normal(size=(len(meta), 8)).astype(np.float32)
    report = analogy_report(emb, meta)
    tt = report['task_transfer']
    assert tt['reason'] == 'disjoint_stimuli'
    assert tt['applicable'] is False
    assert np.isnan(tt['top1'])
    # Subject transfer, by contrast, shares stimuli across subjects and runs.
    assert report['subject_transfer']['n_queries'] > 0


def test_representation_comparison_rows() -> None:
    """Comparison yields one row per (target, representation) with scores."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 10)).astype(np.float32)
    y = (x[:, 0] + 0.1 * rng.normal(size=120)).astype(np.float32)
    rows = M.representation_comparison(
        {'a': x, 'b': rng.normal(size=(120, 10)).astype(np.float32)},
        {'tgt': (y, 'regression')},
    )
    assert len(rows) == 2
    assert {r['representation'] for r in rows} == {'a', 'b'}
    assert all('linear_score' in r and 'knn_score' in r for r in rows)


def test_evaluate_representation_writes_artifacts(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """End-to-end: train, embed, evaluate -> metrics + report + figures on disk."""
    cfg = ZTEConfig()
    cfg.objective.name = 'skipgram'
    cfg.model.embed_dim = 48
    cfg.model.hidden_dim = 40
    cfg.model.n_layers = 2
    cfg.train.epochs = 2
    cfg.train.batch_size = 8
    cfg.train.device = 'cpu'
    cfg.train.precision = 'fp32'
    cfg.train.ckpt_dir = str(tmp_path / 'ckpt')
    run_training(cfg, small_dataset)

    from zte.cli.evaluate import collect_embeddings

    embedder = ZTEEmbedder.from_checkpoint(tmp_path / 'ckpt' / 'best.pt', small_dataset)
    word_emb, word_meta, raw_feats, sent_emb, sent_ids, sent_meta, word_bp = collect_embeddings(embedder, small_dataset)
    assert len(word_emb) == len(word_meta) == len(raw_feats)

    out = tmp_path / 'eval'
    metrics = evaluate_representation(
        word_emb,
        word_meta,
        raw_feats,
        sent_emb,
        sent_ids,
        out_dir=out,
        run_name='test',
        sent_meta=sent_meta,
        word_band_power=word_bp,
        config=cfg,
        interactive=True,
    )
    assert 'embedding_health' in metrics and 'verdict' in metrics
    # New analyses are present and catalogued.
    assert 'analogy' in metrics and 'breakdown_words' in metrics and 'region_importance' in metrics
    assert (out / 'report.md').is_file()
    assert (out / 'metrics.json').is_file()
    assert (out / 'comparison.csv').is_file()
    assert metrics['figures']  # at least one figure rendered


# --------------------------------------------------------------------------- #
# what the retrieval geometry was fitted on
# --------------------------------------------------------------------------- #
def _whitened_config(run_dir: Path) -> ZTEConfig:
    """Builds a fast band-power run whose evaluation post-processes the retrieval geometry."""
    cfg = ZTEConfig()
    cfg.objective.name = 'skipgram'
    cfg.objective.whiten = True
    cfg.model.frontend = 'band_power_mlp'
    cfg.model.embed_dim = 32
    cfg.model.hidden_dim = 32
    cfg.model.n_layers = 1
    cfg.dataset.representation = 'band_power'
    cfg.train.epochs = 1
    cfg.train.batch_size = 8
    cfg.train.device = 'cpu'
    cfg.train.precision = 'fp32'
    cfg.train.split = 'by_sentence'
    cfg.train.num_workers = 0
    cfg.train.ckpt_dir = str(run_dir / 'checkpoints')
    cfg.run_name = 'postprocess'
    return cfg


def test_a_real_evaluation_fits_its_geometry_on_the_training_split(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """The shipped evaluation reports `train split`, so the headline retrieval number is reproducible one row at a time.

    Whitening fitted on the scored rows is transductive: it reads every held-out sentence to decide how to
    transform each one, which a decoder handed a single reading could never redo. Both numbers are carried,
    and this label is the only thing telling a reader which of the two they are looking at.
    """
    import argparse

    from zte.cli.run import _evaluate  # noqa: PLC2701

    run_dir = tmp_path / 'run'
    config = _whitened_config(run_dir)
    run_training(config, small_dataset)
    _evaluate(
        config,
        small_dataset,
        run_dir,
        argparse.Namespace(no_tensorboard=True, no_interactive=True),
    )

    metrics = json.loads((run_dir / 'evaluation' / 'metrics.json').read_text(encoding='utf-8'))
    assert metrics['postprocess_fit'] == 'train split'
    assert 'top1' in (metrics['sentence_retrieval_transductive'] or {})


def test_the_post_processing_label_names_all_three_states() -> None:
    """`none`, `train split` and `transductive` are distinct claims and the label never collapses two of them."""
    assert _postprocess_fit(False, False) == 'none'
    assert _postprocess_fit(False, True) == 'none'
    assert _postprocess_fit(True, False) == 'transductive'
    assert _postprocess_fit(True, True) == 'train split'


def test_the_evaluate_command_supplies_the_training_split(
    small_dataset: ZuCoDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`zte-evaluate` reaches the report through the same train-fitted path `zte-run` does, not a transductive one."""
    from zte.cli import evaluate as E

    config = _whitened_config(tmp_path / 'run')
    run_training(config, small_dataset)
    bundle = small_dataset.save(tmp_path / 'bundle')

    captured: dict[str, Any] = {}

    def recorder(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured['sent_emb'] = args[3]
        captured.update(kwargs)
        return {'verdict': {}}

    monkeypatch.setattr(E, 'evaluate_representation', recorder)
    monkeypatch.setattr(
        'sys.argv',
        [
            'zte-evaluate',
            '--ckpt',
            str(tmp_path / 'run' / 'checkpoints' / 'best.pt'),
            '--bundle',
            str(bundle),
            '--out',
            str(tmp_path / 'out'),
            '--no-interactive',
        ],
    )
    E.main()

    train_emb = captured['train_sent_emb']
    assert train_emb is not None
    assert train_emb.shape[1] == captured['sent_emb'].shape[1]
    assert 0 < len(train_emb) < len(captured['sent_emb'])


def test_the_two_post_processing_fits_are_not_the_same_transform() -> None:
    """The label would be decorative if both fits produced one geometry, so they must actually differ."""
    rng = np.random.default_rng(0)
    train = rng.standard_normal((80, 6)).astype(np.float32)
    scored = rng.standard_normal((40, 6)).astype(np.float32) * 3.0 + 5.0
    on_train = _postprocess(scored, train, True, 1)
    transductive = _postprocess(scored, None, True, 1)
    assert on_train.shape == transductive.shape == scored.shape
    assert not np.allclose(on_train, transductive)


# --------------------------------------------------------------------------- #
# the sentence-level phase control
# --------------------------------------------------------------------------- #
def _raw_embedder(dataset: ZuCoDataset) -> ZTEEmbedder:
    """Builds an untrained raw-frontend embedder over a dataset's window shape."""
    from zte.device import resolve_device
    from zte.models.embedding import build_model

    assert dataset.raw_eeg is not None
    config = ZTEConfig()
    config.model.frontend = 'raw_conformer'
    config.model.embed_dim = 32
    config.model.hidden_dim = 32
    config.model.conformer_filters = 16
    raw_shape = (int(dataset.raw_eeg.shape[1]), int(dataset.raw_eeg.shape[2]))
    model = build_model(config.model, raw_shape=raw_shape)
    return ZTEEmbedder(model, config, resolve_device('cpu'))


def test_the_sentence_phase_control_lines_up_with_the_real_rows(
    small_dataset: ZuCoDataset,
) -> None:
    """The control is compared row against row, so an array of a different length or order compares nothing."""
    from zte.cli.evaluate import phase_shuffled_sent_emb

    embedder = _raw_embedder(small_dataset)
    real, meta = embedder.embed(small_dataset, level='sentence', batch_size=8)
    scrambled = phase_shuffled_sent_emb(embedder, small_dataset, batch_size=8)
    assert scrambled is not None
    assert scrambled.shape == real.shape == (len(meta), embedder.model.embed_dim)
    assert not np.allclose(scrambled, real)


def test_the_sentence_phase_control_is_absent_without_a_waveform(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """Band power is near phase-invariant, so no control is claimed rather than a flattering one reported."""
    from zte.cli.evaluate import phase_shuffled_sent_emb
    from zte.device import resolve_device
    from zte.models.embedding import build_model

    assert small_dataset.features is not None
    config = _whitened_config(tmp_path / 'run')
    model = build_model(config.model, in_dim=int(small_dataset.features.shape[1]))
    embedder = ZTEEmbedder(model, config, resolve_device('cpu'))
    assert phase_shuffled_sent_emb(embedder, small_dataset) is None


def test_the_training_split_embedding_holds_the_training_readings(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """The rows post-processing is fitted on are the training readings, not the ones about to be scored."""
    from zte.cli.decode import split_indices
    from zte.cli.evaluate import train_split_sent_emb

    config = _whitened_config(tmp_path / 'run')
    run_training(config, small_dataset)
    embedder = ZTEEmbedder.from_checkpoint(tmp_path / 'run' / 'checkpoints' / 'best.pt', small_dataset)
    train_idx = split_indices(small_dataset, config, 'train')
    assert train_idx is not None

    train_emb, train_n_words = train_split_sent_emb(embedder, small_dataset, config)
    expected, expected_meta = embedder.embed(small_dataset, level='sentence', indices=train_idx)
    all_emb, _ = embedder.embed(small_dataset, level='sentence')
    assert train_emb is not None
    assert train_emb.shape == expected.shape
    assert len(train_emb) < len(all_emb)
    assert np.allclose(train_emb, expected)

    # The word counts travel with those rows, because the length projection is fitted against the pair.
    assert train_n_words is not None
    assert np.array_equal(train_n_words, expected_meta['n_words'].to_numpy())
