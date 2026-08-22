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
from zte.evaluation.audit.capacity import CLAUSE_NAMES, capacity_report
from zte.evaluation.report import (
    _postprocess,
    _postprocess_fit,
    capacity_verdict,
    evaluate_representation,
    generation_verdict,
)
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


def _length_confounded_inputs(dim: int = 12) -> tuple[Any, ...]:
    """Builds scored and training sentence embeddings whose word count owns one planted direction."""
    rng = np.random.default_rng(11)
    subjects, n_stimuli, n_train = ('ZAB', 'ZDM', 'ZJM'), 24, 72
    direction = rng.standard_normal(dim).astype(np.float32)
    direction /= np.linalg.norm(direction)

    stim_words = rng.integers(5, 40, size=n_stimuli).astype(np.float32)
    centres = rng.standard_normal((n_stimuli, dim)).astype(np.float32)
    sent_emb = np.concatenate(
        [
            centres + np.outer(stim_words, direction) * 0.3 + rng.normal(scale=0.05, size=(n_stimuli, dim))
            for _ in subjects
        ]
    ).astype(np.float32)
    sent_ids = np.tile(np.arange(n_stimuli), len(subjects))
    sent_n_words = np.tile(stim_words, len(subjects))

    train_words = rng.integers(5, 40, size=n_train).astype(np.float32)
    train_sent_emb = (
        rng.standard_normal((n_train, dim))
        + np.outer(train_words, direction) * 0.3
        + rng.normal(scale=0.05, size=(n_train, dim))
    ).astype(np.float32)

    n_rows = len(sent_emb) * 2
    word_meta = pd.DataFrame(
        {
            'word': [f'w{i % 9}' for i in range(n_rows)],
            'word_len': rng.integers(2, 9, size=n_rows),
            'log_freq': rng.normal(size=n_rows),
            'subject': np.repeat(np.asarray(subjects), n_rows // len(subjects)),
            'task': ['SR'] * n_rows,
            'category': ['c'] * n_rows,
            'sentence_idx': np.repeat(np.arange(n_rows // 2), 2),
            'word_idx': [i % 2 for i in range(n_rows)],
        }
    )
    sent_meta = pd.DataFrame({'subject': np.repeat(np.asarray(subjects), n_stimuli), 'category': ['c'] * len(sent_emb)})

    return (
        rng.standard_normal((n_rows, 8)).astype(np.float32),
        word_meta,
        rng.standard_normal((n_rows, 6)).astype(np.float32),
        sent_emb,
        sent_ids,
        sent_meta,
        sent_n_words,
        train_sent_emb,
        train_words,
    )


def test_length_projector_is_fitted_in_the_scored_frame(tmp_path: Path) -> None:
    """Whitening moves the coordinates, so the length basis has to be fitted after it, not before.

    Note:
        A basis fitted on the raw training rows and subtracted from the whitened scored rows is a vector field in
        the wrong coordinates: it adds structure word count predicts rather than removing it, and
        `length_leakage_after` can come out above `length_leakage_before` while the report calls the number a
        de-confounded one.
    """
    word_emb, word_meta, raw_feats, sent_emb, sent_ids, sent_meta, sent_n_words, train_emb, train_words = (
        _length_confounded_inputs()
    )
    cfg = ZTEConfig()
    cfg.objective.whiten = True
    cfg.objective.all_but_top = 1
    cfg.objective.length_projection = True

    metrics = evaluate_representation(
        word_emb,
        word_meta,
        raw_feats,
        sent_emb,
        sent_ids,
        out_dir=tmp_path / 'eval',
        run_name='length_frame',
        sent_meta=sent_meta,
        config=cfg,
        tensorboard=False,
        interactive=False,
        train_sent_emb=train_emb,
        train_sent_n_words=train_words,
        sent_n_words=sent_n_words,
    )

    block = metrics['length_projection']
    assert block['status'] == 'applied'
    assert block['length_leakage_after'] < block['length_leakage_before']


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


# --------------------------------------------------------------------------- #
# the decoder menu capacity
# --------------------------------------------------------------------------- #
def _capacity_block(*, certified: bool, n_gallery: int = 16) -> dict[str, Any]:
    """Runs the real certification on a model that wins every menu, or one that ties its controls everywhere."""
    won = np.eye(n_gallery, dtype=np.float64)
    lost = np.zeros((n_gallery, n_gallery), dtype=np.float64)
    # K = 32 cannot be filled by a 16-sentence gallery, which is what makes it unreachable rather than failed.
    report = capacity_report(
        {'model': won if certified else lost, 'length_only': lost, 'shuffled_eeg': lost, 'mismatch': lost},
        np.concatenate([np.arange(n_gallery), np.arange(n_gallery)]),
        np.array(['A'] * n_gallery + ['B'] * n_gallery),
        'B',
        np.full(2 * n_gallery, 9.0),
        tasks=np.array(['NR'] * (2 * n_gallery)),
        ks=(2, 4, 8, 32),
        n_perm=200,
        n_boot=200,
        honest_split=True,
        split_strategy='by_subject_and_stimulus',
        split_cell='test',
    )
    assert report is not None

    return report


def _failing_generation() -> dict[str, Any]:
    """An honest-split generation block that beats nothing -- the case a certified menu must never rescue."""
    delta = {'point': -0.01, 'lo': -0.05, 'hi': 0.03, 'beats': False}

    return {
        'applicable': True,
        'primary_metric': 'content_f1',
        'split': 'test',
        'split_strategy': 'by_subject_and_stimulus',
        'n': 24,
        'n_candidate_sentences': None,
        'controls_requested': ['mean_prefix'],
        'deltas': {'mean_prefix': {'content_f1': delta}},
        'permutation': {'applicable': True, 'p_value': 0.4},
        'prefix_influence_kl': 0.2,
        'worst_control': 'mean_prefix',
        'worst_control_ci': {'point': -0.01, 'lo': -0.05, 'hi': 0.03},
    }


def _eval_with(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    """Runs a light evaluation over the synthetic sentences, with whatever decoder blocks the test supplies."""
    word_emb, word_meta, raw_feats, sent_emb, sent_ids, sent_meta, sent_n_words, _, _ = _length_confounded_inputs()

    return evaluate_representation(
        word_emb,
        word_meta,
        raw_feats,
        sent_emb,
        sent_ids,
        out_dir=tmp_path / 'eval',
        run_name='capacity',
        sent_meta=sent_meta,
        config=ZTEConfig(),
        tensorboard=False,
        interactive=False,
        sent_n_words=sent_n_words,
        **extra,
    )


def test_the_capacity_block_travels_into_metrics_under_its_own_key(tmp_path: Path) -> None:
    """The certified menu reaches `metrics.json` as `decoder_capacity`, never sharing `menu` with the cosine audit.

    Note:
        `rebaseline.json` already carries a `menu` block, and that one is the encoder's cosine menu. Two
        different readouts under one key would be read as one table and compared as if they were.
    """
    capacity = _capacity_block(certified=True)
    metrics = _eval_with(tmp_path, decoder_capacity=capacity)

    assert metrics['decoder_capacity'] == capacity
    assert 'menu' not in metrics

    on_disk = json.loads((tmp_path / 'eval' / 'metrics.json').read_text(encoding='utf-8'))
    assert on_disk['decoder_capacity']['readout'] == 'menu selection'
    assert on_disk['decoder_capacity']['certified_k'] == 8


def test_the_capacity_verdict_keys_are_merged_additively(tmp_path: Path) -> None:
    """The run verdict gains the capacity outcome under namespaced keys and loses none of its own."""
    without = _eval_with(tmp_path / 'plain')['verdict']
    with_capacity = _eval_with(tmp_path / 'certified', decoder_capacity=_capacity_block(certified=True))['verdict']

    assert with_capacity['capacity_certified'] is True
    assert with_capacity['capacity_k'] == 8
    assert with_capacity['capacity_bits'] == pytest.approx(3.0)
    assert with_capacity['capacity_readout'] == 'menu selection'
    assert set(with_capacity['capacity_clauses']) == set(CLAUSE_NAMES)

    # Additive: every pre-existing key survives, and only `capacity_*` keys are new.
    assert set(without) <= set(with_capacity)
    assert all(key.startswith('capacity_') for key in set(with_capacity) - set(without))


def test_the_capacity_verdict_writes_nothing_a_generation_clause_could_read() -> None:
    """A menu result is retrieval-shaped, so every key it contributes is namespaced away from the other gates."""
    assert all(key.startswith('capacity_') for key in capacity_verdict(_capacity_block(certified=True)))


def test_a_certified_menu_never_licenses_a_generation_headline(tmp_path: Path) -> None:
    """Selecting the read sentence out of eight candidates is not free generation and cannot pass its gate.

    Note:
        A K-way menu hands the decoder the answer among K-1 distractors; free generation asks for ~190 bits
        with nothing given. Letting the certified menu into the generation AND would license the exact
        overclaim the gate exists to refuse.
    """
    generation = _failing_generation()
    verdict = _eval_with(tmp_path, generation=generation, decoder_capacity=_capacity_block(certified=True))['verdict']

    assert verdict['capacity_certified'] is True
    assert verdict['generation_above_controls'] is False
    assert not any(key.startswith('capacity') for key in verdict['generation_clauses'])

    # The clauses are byte-identical to the ones the generation gate reaches on its own.
    assert verdict['generation_clauses'] == generation_verdict(generation, 0.05)['generation_clauses']


def test_the_capacity_section_names_the_menu_sizes_no_pool_could_fill(tmp_path: Path) -> None:
    """An uncertified capacity renders an em dash and the failing clauses, and says which sizes were never asked."""
    metrics = _eval_with(tmp_path, decoder_capacity=_capacity_block(certified=False))
    assert metrics['verdict']['capacity_certified'] is False

    report = (tmp_path / 'eval' / 'report.md').read_text(encoding='utf-8')
    assert '## Decoder menu capacity' in report
    assert 'menu selection' in report

    # Never a blank or a zero: nothing certified has to read as nothing certified.
    assert 'Certified menu size: **K = —**' in report
    assert 'above_chance' in report

    assert 'Menu sizes with queries: 2, 4, 8.' in report
    reach = next(line for line in report.splitlines() if 'Unreachable on this gallery' in line)
    assert reach.rstrip().endswith('32.')


def test_an_evaluation_without_a_capacity_block_is_unchanged(tmp_path: Path) -> None:
    """The capacity wiring is opt-in, so a run that never measured one reports no menu anywhere."""
    metrics = _eval_with(tmp_path)

    assert 'decoder_capacity' not in metrics
    assert not [key for key in metrics['verdict'] if key.startswith('capacity')]
    assert 'Decoder menu capacity' not in (tmp_path / 'eval' / 'report.md').read_text(encoding='utf-8')
