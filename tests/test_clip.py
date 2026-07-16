"""Tests for the sentence-level CLIP alignment objective (`objective.name='clip'`).

Covers the symmetric multi-positive InfoNCE, the frozen-text pipeline fallback, semantic-hard-negative
mining + the co-locating sampler, and the data plumbing (sentence_text_id through collate).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from zte.config import DatasetConfig, ModelConfig, ObjectiveConfig
from zte.data.dataset import ZuCoDataset
from zte.data.synthetic import generate_synthetic_zuco
from zte.data.text import build_sentence_text_matrix, mine_hard_negatives
from zte.data.torch_dataset import SemanticHardNegativeSampler, make_dataloader
from zte.models.embedding import build_model
from zte.models.objectives import SentenceClipObjective, _clip_direction, build_objective


# --------------------------------------------------------------------------- #
# text pipeline
# --------------------------------------------------------------------------- #
def test_build_sentence_text_matrix_none_source() -> None:
    """No source -> graceful (None, 0) so the caller can hash-fallback."""
    mat, dim = build_sentence_text_matrix(['a', 'b'], None)
    assert mat is None and dim == 0


def test_mine_hard_negatives_picks_surface_similar_meaning_distinct() -> None:
    """A hard negative is surface-similar (word overlap) but semantically distinct (low text-cosine)."""
    texts = [
        'the cat sat on the mat',  # 0
        'the cat sat on the mat too',  # 1 semantically ~identical to 0
        'the dog sat on the rug',  # 2 surface-similar to 0, different meaning
        'inflation rose sharply',  # 3 unrelated surface + meaning
    ]
    tm = np.eye(4, dtype=np.float32)
    tm[1] = tm[0]  # make 1 semantically identical to 0
    tm = tm / np.clip(np.linalg.norm(tm, axis=1, keepdims=True), 1e-8, None)
    hn = mine_hard_negatives(texts, tm, k=2)
    assert hn.shape == (4, 2)
    # For sentence 0, the surface-similar-but-distinct sentence 2 must rank as a hard negative,
    # and the semantically-identical sentence 1 must NOT.
    assert 2 in hn[0]
    assert 1 not in hn[0]


def test_hard_negative_sampler_covers_each_sentence_once() -> None:
    """The sampler yields every sentence exactly once per epoch."""
    text_ids = [0, 1, 2, 0, 1, 2]  # 3 texts, 2 readings each
    hn = np.array([[1, 2], [0, 2], [0, 1]])
    sampler = SemanticHardNegativeSampler(text_ids, hn, batch_size=4, shuffle=True, seed=0)
    covered = sorted(i for batch in sampler for i in batch)
    assert covered == list(range(6))


def test_clip_direction_rewards_alignment() -> None:
    """A near-perfectly-aligned similarity matrix has lower loss than a scrambled one."""
    n = 6
    valid = torch.ones(n, dtype=torch.bool)
    pos = torch.eye(n, dtype=torch.bool)
    aligned = torch.eye(n) * 10.0  # diagonal (positives) dominate
    scrambled = torch.randn(n, n)
    assert _clip_direction(aligned, pos, valid) < _clip_direction(scrambled, pos, valid)


# --------------------------------------------------------------------------- #
# objective
# --------------------------------------------------------------------------- #
def test_clip_objective_symmetric_multipositive() -> None:
    """CLIP compute: finite symmetric loss, multi-positive across subjects, grads to head + temperature."""
    obj = ObjectiveConfig(
        name='clip',
        clip_temperature=0.07,
        variance_weight=1.0,
        covariance_weight=1.0,
        subject_adversary_weight=0.1,
        cross_subject_positives=True,
    )
    mdl = ModelConfig(embed_dim=48, hidden_dim=24, n_layers=1, n_subjects=3, pool='attention')
    model = build_model(mdl, in_dim=40)
    o = build_objective(obj, model, feature_dim=40)
    assert isinstance(o, SentenceClipObjective)

    text_dim = 16
    tm = torch.nn.functional.normalize(torch.randn(3, text_dim), dim=-1)
    o.attach_text(tm)
    assert o.clip_head is not None and o.clip_head.out_features == text_dim

    b, length = 6, 5  # 3 texts read by 2 subjects
    text_id = torch.tensor([0, 1, 2, 0, 1, 2])
    batch = {
        'features': torch.randn(b, length, 40),
        'pad_mask': torch.ones(b, length, dtype=torch.bool),
        'presence': torch.ones(b, length, dtype=torch.bool),
        'subject': torch.tensor([0, 0, 0, 1, 1, 1]),
        'content_id': torch.zeros(b, length, dtype=torch.long),
        'word_id': torch.arange(b * length).reshape(b, length),
        'task_id': torch.zeros(b, dtype=torch.long),
        'sentence_text_id': text_id,
    }
    loss, m = o.compute(model, batch)
    assert torch.isfinite(loss)
    for k in ('clip_loss', 'clip_top1', 'logit_scale', 'n_valid', 'vicreg_var'):
        assert k in m and np.isfinite(m[k])
    assert m['n_valid'] == 6.0
    loss.backward()
    assert torch.isfinite(
        next(p.grad for p in o.clip_head.parameters() if p.grad is not None)
    ).all()
    assert o.logit_scale.grad is not None and torch.isfinite(o.logit_scale.grad)


def test_clip_objective_graceful_without_text() -> None:
    """With no text attached, CLIP degrades to the regulariser loss instead of crashing."""
    obj = ObjectiveConfig(name='clip', variance_weight=1.0)
    model = build_model(ModelConfig(embed_dim=32, hidden_dim=16, n_layers=1), in_dim=40)
    o = build_objective(obj, model)
    batch = {
        'features': torch.randn(2, 4, 40),
        'pad_mask': torch.ones(2, 4, dtype=torch.bool),
        'presence': torch.ones(2, 4, dtype=torch.bool),
        'subject': torch.zeros(2, dtype=torch.long),
        'sentence_text_id': torch.tensor([0, 1]),
    }
    loss, m = o.compute(model, batch)
    assert torch.isfinite(loss) and m['n_valid'] == 0.0


# --------------------------------------------------------------------------- #
# data plumbing
# --------------------------------------------------------------------------- #
@pytest.fixture(scope='module')
def synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small synthetic ZuCo tree with several subjects reading the same sentences."""
    out = tmp_path_factory.mktemp('zuco_clip')
    generate_synthetic_zuco(
        out, subjects=('ZAB', 'ZDM', 'ZPH'), tasks=('SR', 'NR'), n_sentences=8, show_progress=False
    )
    return out


def test_sentence_text_id_flows_through_collate(synth_dir: Path, tmp_path: Path) -> None:
    """`sentence_text_id` is carried per sentence and the hard-negative loader yields valid batches."""
    ds = ZuCoDataset(DatasetConfig(root=str(synth_dir), cache_dir=str(tmp_path / 'c'))).build(
        show_progress=False
    )
    td = ds.to_torch()
    assert len(td.text_vocab) > 0
    # every sentence has a text id in range
    assert all(0 <= t < len(td.text_vocab) for t in td._sentence_text_id)  # noqa: SLF001
    hn = np.full((len(td.text_vocab), 2), -1, dtype=np.int64)  # (empty hard-neg table is valid)
    loader = make_dataloader(td, batch_size=8, hard_negatives=hn, seed=0)
    batch = next(iter(loader))
    assert (
        'sentence_text_id' in batch
        and batch['sentence_text_id'].shape[0] == batch['features'].shape[0]
    )
    assert (batch['sentence_text_id'] >= 0).all()
