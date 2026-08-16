"""Tests for the task-matched contrastive denominators and the pooled-sentence anti-collapse guard.

Task and stimulus are fully confounded on ZuCo, so a cross-task negative is separable by task alone; these tests pin
that `within_task_negatives` removes such candidates from every sentence-level denominator, that the per-text task
buffer is an id join rather than a key parse, and that the sentence-level VICReg guard fires only when asked.
"""

import math
import os
from typing import Any

import pytest
import torch
import torch.nn.functional as F

# The decoder path is offline: `lm_source='tiny'` builds its LM and tokeniser locally.
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from zte.config import DecoderConfig, ModelConfig, ObjectiveConfig
from zte.data.dataset import ZuCoDataset
from zte.data.targets.tokens import build_target_tokens
from zte.data.torch_dataset import join_text_tasks
from zte.models.embedding import ZTEModel, build_model
from zte.models.encoder.consensus import ConsensusDistiller
from zte.models.encoder.gallery import GalleryContrast
from zte.models.objectives import PrefixDecodeObjective, SentenceClipObjective, _clip_direction, build_objective
from zte.models.objectives.decode import Conditioned


def _clip_objective(cfg: ObjectiveConfig, seed: int = 7) -> tuple[SentenceClipObjective, ZTEModel]:
    """Builds a small CLIP objective with an attached two-text hash gallery, deterministically."""
    torch.manual_seed(seed)
    model = build_model(ModelConfig(embed_dim=32, hidden_dim=16, n_layers=1, n_subjects=2), in_dim=20)
    objective = build_objective(cfg, model)
    assert isinstance(objective, SentenceClipObjective)
    objective.attach_text(F.normalize(torch.randn(2, 12), dim=-1))

    return objective, model


def _two_task_batch(seed: int = 11) -> dict[str, Any]:
    """A four-reading batch: text 0 under task 0 and text 1 under task 1, each read by two subjects."""
    torch.manual_seed(seed)

    return {
        'features': torch.randn(4, 3, 20),
        'pad_mask': torch.ones(4, 3, dtype=torch.bool),
        'presence': torch.ones(4, 3, dtype=torch.bool),
        'subject': torch.tensor([0, 1, 0, 1]),
        'task_id': torch.tensor([0, 0, 1, 1]),
        'sentence_text_id': torch.tensor([0, 0, 1, 1]),
    }


# --------------------------------------------------------------------------- #
# knobs off: the plain objective, unchanged
# --------------------------------------------------------------------------- #
def test_the_default_config_keeps_the_off_path_loss_and_metrics() -> None:
    """Every new knob at its default reproduces the exact loss and metric surface of the plain objective."""
    results: list[tuple[float, dict[str, float]]] = []
    for cfg in (
        ObjectiveConfig(name='clip', clip_temperature=1.0),
        ObjectiveConfig(
            name='clip',
            clip_temperature=1.0,
            within_task_negatives=False,
            sentence_variance_weight=0.0,
            sentence_covariance_weight=0.0,
        ),
    ):
        objective, model = _clip_objective(cfg)
        loss, metrics = objective.compute(model, _two_task_batch())
        results.append((float(loss.detach()), metrics))

    assert results[0][0] == results[1][0]
    assert set(results[0][1]) == set(results[1][1])
    assert 'sentence_vicreg_var' not in results[0][1]
    assert 'gallery_dropped' not in results[0][1]


def test_the_unmasked_direction_is_the_plain_multi_positive_infonce() -> None:
    """With no candidate mask the denominator ranges over every valid column, cross-task ones included."""
    torch.manual_seed(0)
    logits = torch.randn(4, 4)
    valid = torch.ones(4, dtype=torch.bool)
    pos = torch.eye(4, dtype=torch.bool)
    expected = (torch.logsumexp(logits, dim=1) - logits.diagonal()).mean()

    assert torch.allclose(_clip_direction(logits, pos, valid), expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# within_task_negatives: the denominator stays inside the anchor's task
# --------------------------------------------------------------------------- #
def test_the_candidate_mask_removes_a_strong_cross_task_negative() -> None:
    """A dominant cross-task logit leaves the denominator entirely once the pairwise mask is passed."""
    logits = torch.tensor([[2.0, 10.0], [0.0, 3.0]])
    pos = torch.eye(2, dtype=torch.bool)
    valid = torch.ones(2, dtype=torch.bool)
    cand = torch.eye(2, dtype=torch.bool)  # each anchor's task holds only itself

    unrestricted = _clip_direction(logits, pos, valid)
    restricted = _clip_direction(logits, pos, valid, cand)

    assert float(restricted) == pytest.approx(0.0, abs=1e-6)
    assert float(unrestricted) > float(restricted)


def test_within_task_negatives_empty_a_task_pure_clip_denominator() -> None:
    """When each anchor's same-task candidates are exactly its positives, the task-masked InfoNCE is satisfied."""
    off_objective, off_model = _clip_objective(ObjectiveConfig(name='clip', clip_temperature=1.0))
    on_objective, on_model = _clip_objective(
        ObjectiveConfig(name='clip', clip_temperature=1.0, within_task_negatives=True)
    )

    _, off_metrics = off_objective.compute(off_model, _two_task_batch())
    _, on_metrics = on_objective.compute(on_model, _two_task_batch())

    assert on_metrics['clip_loss'] == pytest.approx(0.0, abs=1e-5)
    assert off_metrics['clip_loss'] > 0.01


def test_within_task_negatives_refuse_a_batch_without_task_ids() -> None:
    """A missing `task_id` raises rather than silently widening back to cross-task candidates."""
    objective, model = _clip_objective(ObjectiveConfig(name='clip', within_task_negatives=True))
    batch = _two_task_batch()
    del batch['task_id']

    with pytest.raises(ValueError, match='task_id'):
        objective.compute(model, batch)


def test_the_gallery_denominator_stays_inside_the_anchor_task() -> None:
    """With tasks attached, an anchor's candidate row covers its own task's texts and nothing else."""
    contrast = GalleryContrast(band=0, within_task=True)
    contrast.attach_tasks(torch.tensor([0, 0, 1, 1]))

    mask = contrast.candidate_mask(torch.tensor([0, 2]), n_texts=4)

    assert mask[0].tolist() == [True, True, False, False]
    assert mask[1].tolist() == [False, False, True, True]


def test_a_stranded_within_task_anchor_is_dropped_not_widened() -> None:
    """The sparse-band fallback widens across the gallery only while task matching is off; on, the anchor drops."""
    lengths = torch.tensor([5, 50, 50, 5])

    off = GalleryContrast(band=1, min_candidates=3)
    off.attach_lengths(lengths)
    assert off.candidate_mask(torch.tensor([0]), n_texts=4)[0].all()

    on = GalleryContrast(band=1, min_candidates=3, within_task=True)
    on.attach_lengths(lengths)
    on.attach_tasks(torch.tensor([0, 0, 0, 1]))
    assert not on.candidate_mask(torch.tensor([0]), n_texts=4)[0].any()


def test_gallery_compute_reports_and_excludes_dropped_anchors() -> None:
    """An anchor with no same-task band contributes nothing to the loss and is counted as dropped."""
    contrast = GalleryContrast(band=1, min_candidates=3, within_task=True)
    contrast.attach_lengths(torch.tensor([5, 50, 50, 5]))
    contrast.attach_tasks(torch.tensor([0, 0, 0, 1]))

    gallery = F.normalize(torch.eye(4), dim=-1)
    loss, metrics = contrast.compute(torch.eye(4)[:1], gallery, torch.tensor([0]), torch.tensor(10.0))

    assert metrics['gallery_dropped'] == 1.0
    assert float(loss) == 0.0


def test_the_consensus_gallery_draws_same_task_distractors_only() -> None:
    """A near-duplicate prototype from the other task cannot enter the consensus denominator."""

    def gallery_loss(with_tasks: bool) -> float:
        distiller = ConsensusDistiller(n_keys=2, dim=3, n_subjects=2, min_readers=1, temperature=1.0)
        distiller.eval()
        if with_tasks:
            distiller.attach_tasks(torch.tensor([0, 1]))
        anchor = torch.tensor([[1.0, 0.0, 0.0]])
        distiller.bank.update(torch.tensor([0]), anchor, torch.tensor([0]))
        distiller.bank.update(torch.tensor([1]), torch.tensor([[0.9, 0.1, 0.0]]), torch.tensor([1]))
        loss, _ = distiller.compute(
            anchor, torch.tensor([0]), torch.tensor([0]), pull_weight=0.0, gallery_weight=1.0, prefix='c'
        )
        return float(loss.detach())

    assert gallery_loss(with_tasks=True) == pytest.approx(0.0, abs=1e-6)
    assert gallery_loss(with_tasks=False) > 1e-3


def _decode_objective() -> PrefixDecodeObjective:
    """Builds a tiny decode objective over four one-per-task reference texts."""
    torch.manual_seed(0)
    model = build_model(
        ModelConfig(embed_dim=16, hidden_dim=16, n_layers=1, n_heads=2, projection_hidden=16), in_dim=12
    )
    cfg = DecoderConfig(
        lm_source='tiny',
        tokenizer_source='tiny',
        max_target_tokens=16,
        prefix_slots=2,
        bottleneck=8,
        ground_negatives=2,
    )
    objective = build_objective(ObjectiveConfig(name='decode', within_task_negatives=True), model, decoder_config=cfg)
    assert isinstance(objective, PrefixDecodeObjective)
    targets = build_target_tokens(['aa bb', 'cc dd', 'ee ff', 'gg hh'], 'tiny', max_length=16)
    objective.attach_tokens(torch.from_numpy(targets.ids), torch.from_numpy(targets.mask))

    return objective


def test_grounding_negatives_are_drawn_same_task() -> None:
    """With one text per task no same-task negative exists, so the task-matched grounding term is silent."""
    objective = _decode_objective()
    assert objective.target_ids is not None and objective.target_mask is not None
    text_id = torch.tensor([0, 1, 2, 3])
    ids, mask = objective.target_ids[text_id], objective.target_mask[text_id]
    everyone = torch.ones(4, dtype=torch.bool)
    cond = Conditioned(z=torch.randn(4, 16), prefix=torch.randn(4, 2, objective.lm.hidden_dim))

    torch.manual_seed(1)
    on = objective._grounding(cond, ids, mask, text_id, everyone, everyone, torch.tensor([0, 1, 2, 3]))  # noqa: SLF001
    assert float(on) == 0.0

    # The same batch with the knob off draws those cross-task rows as negatives and pays a positive loss.
    objective.config.within_task_negatives = False
    torch.manual_seed(1)
    off = objective._grounding(cond, ids, mask, text_id, everyone, everyone, torch.tensor([0, 1, 2, 3]))  # noqa: SLF001
    assert float(off) > 0.0


def test_grounding_refuses_a_batch_without_task_ids() -> None:
    """Under `within_task_negatives` the grounding term raises when the batch carries no task ids."""
    objective = _decode_objective()
    assert objective.target_ids is not None and objective.target_mask is not None
    text_id = torch.tensor([0, 1, 2, 3])
    ids, mask = objective.target_ids[text_id], objective.target_mask[text_id]
    everyone = torch.ones(4, dtype=torch.bool)
    cond = Conditioned(z=torch.randn(4, 16), prefix=torch.randn(4, 2, objective.lm.hidden_dim))

    with pytest.raises(ValueError, match='task_id'):
        objective._grounding(cond, ids, mask, text_id, everyone, everyone, None)  # noqa: SLF001


# --------------------------------------------------------------------------- #
# the per-text task buffer: an id join, never a key parse
# --------------------------------------------------------------------------- #
def test_a_text_read_under_two_tasks_raises() -> None:
    """One text id under two tasks makes a same-task denominator ill-defined, so the join refuses it."""
    with pytest.raises(ValueError, match='two tasks'):
        join_text_tasks([0, 1, 0], [0, 1, 1], n_texts=2)


def test_the_join_marks_unread_texts_and_skips_missing_ids() -> None:
    """Texts no sentence reads stay `-1`, and `-1` sentence ids never enter the join."""
    assert join_text_tasks([0, 2, -1], [1, 0, 5], n_texts=4).tolist() == [1, -1, 0, -1]


def test_the_task_buffer_is_joined_from_per_sample_ids(small_dataset: ZuCoDataset) -> None:
    """The buffer agrees with every sample's own `(text_id, task_id)` pair; vocab keys carry no task to parse."""
    td = small_dataset.to_torch()
    buffer = td.text_task_ids()

    assert buffer.shape[0] == len(td.text_vocab)
    assert bool((buffer >= 0).any())
    for i in range(len(td)):
        sample = td[i]
        if sample.text_id >= 0:
            assert int(buffer[sample.text_id]) == sample.task_id


# --------------------------------------------------------------------------- #
# sentence-level VICReg: guard the tensor retrieval actually scores
# --------------------------------------------------------------------------- #
def test_a_collapsed_sentence_batch_pays_the_pinned_variance_penalty() -> None:
    """Identical pooled vectors have zero per-dimension std, so the hinge pays `gamma - sqrt(eps)` exactly."""
    objective, _ = _clip_objective(ObjectiveConfig(name='clip', sentence_variance_weight=1.0))

    loss, metrics = objective.sentence_regularize(torch.ones(8, 32))

    assert float(loss) == pytest.approx(1.0 - math.sqrt(1e-4), abs=1e-6)
    assert metrics['sentence_vicreg_var'] == pytest.approx(float(loss), abs=1e-6)


def test_a_well_spread_sentence_batch_pays_nothing() -> None:
    """Per-dimension stds far above the target leave the hinge at exactly zero."""
    objective, _ = _clip_objective(ObjectiveConfig(name='clip', sentence_variance_weight=1.0))
    torch.manual_seed(0)

    loss, _ = objective.sentence_regularize(torch.randn(256, 32) * 100.0)

    assert float(loss) == 0.0


def test_weight_zero_computes_no_sentence_vicreg_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both weights at zero return before the VICReg math ever runs -- a guard, not a multiply-by-zero."""
    objective, _ = _clip_objective(ObjectiveConfig(name='clip'))

    def boom(*args: object, **kwargs: object) -> tuple[torch.Tensor, dict[str, float]]:
        raise AssertionError('vicreg_terms must not run when both sentence weights are 0')

    monkeypatch.setattr('zte.models.objectives.base.vicreg_terms', boom)
    loss, metrics = objective.sentence_regularize(torch.randn(4, 32))

    assert float(loss) == 0.0 and metrics == {}


def test_sentence_vicreg_reaches_the_clip_loss_and_its_metrics() -> None:
    """With a positive weight the pooled-sentence guard contributes to the loss and reports its metrics."""
    objective, model = _clip_objective(
        ObjectiveConfig(name='clip', sentence_variance_weight=1.0, sentence_covariance_weight=1.0)
    )

    loss, metrics = objective.compute(model, _two_task_batch())

    assert 'sentence_vicreg_var' in metrics and 'sentence_vicreg_cov' in metrics
    assert torch.isfinite(loss)
