"""Tests for the nearest-neighbour index and the evaluation suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zte.config import ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.evaluation import metrics as M
from zte.evaluation.report import evaluate_representation
from zte.inference.embed import ZTEEmbedder
from zte.inference.retrieval import NearestNeighborIndex
from zte.training.pipeline import run_training


def _clustered(n_groups: int, per_group: int, dim: int, scale: float = 0.05) -> tuple:
    """Builds clustered embeddings: one centre per group + small noise."""
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(n_groups, dim)).astype(np.float32)
    groups = np.repeat(np.arange(n_groups), per_group)
    emb = centres[groups] + rng.normal(scale=scale, size=(n_groups * per_group, dim)).astype(
        np.float32
    )
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


def test_evaluate_representation_writes_artifacts(
    small_dataset: ZuCoDataset, tmp_path: Path
) -> None:
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
    word_emb, word_meta, raw_feats, sent_emb, sent_ids, sent_meta, word_bp = collect_embeddings(
        embedder, small_dataset
    )
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
