"""Tests for length- and piece-matched hard negatives, and for the denominator they narrow.

The point of a matched negative is that it cannot be rejected by counting: on the real gallery word count alone
carries 5.14 of the 9.45 bits needed to name a sentence, so an unmatched negative prices no semantics. These tests
pin the matching, the widening diagnostic, the surviving numerator, and -- most importantly -- that leaving
`hard_negative_in_loss` off reproduces the old loss value exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zte.config import ModelConfig, ObjectiveConfig
from zte.data.targets.text import MatchedNegatives, mine_hard_negatives, mine_matched_hard_negatives
from zte.models.embedding import build_model
from zte.models.encoder.gallery import GalleryContrast, build_gallery_contrast, text_word_counts
from zte.models.objectives import SentenceClipObjective, build_objective

# 0 and 1 are the same five words in a different order; 2 repeats 0's words but is far longer; 3 is a same-length
# stranger. Only 0 and 1 differ purely in meaning.
ROLE_TEXTS: list[str] = [
    'the dog bit the man',
    'the man bit the dog',
    'the dog bit the man in the street early this morning',
    'inflation rose sharply last year',
]


@pytest.fixture
def role_matrix() -> np.ndarray:
    """Text embeddings in which the reordered sentence is semantically identical to its anchor."""
    mat = np.eye(len(ROLE_TEXTS), dtype=np.float32)
    mat[1] = mat[0]  # same meaning as the anchor, so the surface miner scores it worthless
    return mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8, None)


def _word_pieces(rows: list[list[int]]) -> np.ndarray:
    """Pads per-word sub-word counts into the `(n_text, max_words)` int16 table `TokenAlignment` emits."""
    width = max(len(r) for r in rows)
    table = np.zeros((len(rows), width), dtype=np.int16)
    for i, row in enumerate(rows):
        table[i, : len(row)] = row
    return table


# --------------------------------------------------------------------------- #
# the matched miner
# --------------------------------------------------------------------------- #
def test_surface_miner_picks_a_negative_of_another_length(role_matrix: np.ndarray) -> None:
    """The unmatched miner prefers the eleven-word sentence, which is separable from the anchor by counting."""
    surface = mine_hard_negatives(ROLE_TEXTS, role_matrix, k=1)
    assert int(surface[0, 0]) == 2


def test_matched_miner_stays_inside_the_length_tolerance(role_matrix: np.ndarray) -> None:
    """Matched mining rejects that same negative and takes the reordered, same-length one instead."""
    mined = mine_matched_hard_negatives(ROLE_TEXTS, role_matrix, k=1, length_tol=1)
    assert int(mined.table[0, 0]) == 1
    assert 2 not in mined.table[0]

    # Every anchor that did not have to widen is matched; sentence 2 is eleven words long and stands alone, so it
    # widens and is flagged rather than quietly taking a five-word negative while claiming to be matched.
    lengths = np.array([len(t.split()) for t in ROLE_TEXTS])
    matched = ~mined.widened
    picked = (mined.table >= 0) & matched[:, None]
    anchors = np.broadcast_to(np.arange(len(ROLE_TEXTS))[:, None], mined.table.shape)
    assert np.all(np.abs(lengths[mined.table[picked]] - lengths[anchors[picked]]) <= 1)
    assert bool(mined.widened[2]) and not bool(mined.widened[0])


def test_matched_miner_honours_the_piece_tolerance(role_matrix: np.ndarray) -> None:
    """A same-length negative spelled in a different number of sub-word pieces is rejected when a piece table is given.

    Sentence 1 matches the anchor on words but is spelled in ten pieces against the anchor's five, so at `piece_tol=0`
    the only admissible candidate is the same-length, same-budget stranger.
    """
    pieces = _word_pieces([[1, 1, 1, 1, 1], [1, 1, 1, 1, 6], [1] * 11, [1, 1, 1, 1, 1]])

    without = mine_matched_hard_negatives(ROLE_TEXTS, role_matrix, k=1, length_tol=1)
    assert int(without.table[0, 0]) == 1
    assert without.piece_gap == -1

    with_pieces = mine_matched_hard_negatives(
        ROLE_TEXTS, role_matrix, k=1, length_tol=1, word_pieces=pieces, piece_tol=0
    )
    assert int(with_pieces.table[0, 0]) == 3
    assert not bool(with_pieces.widened[0])
    assert with_pieces.piece_gap >= 0  # a piece table was supplied, so the gap is measured rather than 'n/a'


def test_matched_miner_prefers_a_reordered_bag_over_an_unrelated_sentence(role_matrix: np.ndarray) -> None:
    """Given two admissible same-length candidates, the one built from the anchor's own words in another order wins."""
    mined = mine_matched_hard_negatives(ROLE_TEXTS, role_matrix, k=2, length_tol=1)
    assert list(mined.table[0]) == [1, 3]


def test_widening_is_recorded_rather_than_silently_shortening_the_pool() -> None:
    """A sentence with no in-tolerance candidate widens, fills its pool, and says so in the diagnostic."""
    texts = ['alpha', 'one two three four five', 'five four three two one', 'two three four five six']
    matrix = np.eye(len(texts), dtype=np.float32)

    mined = mine_matched_hard_negatives(texts, matrix, k=2, length_tol=1)
    assert bool(mined.widened[0])
    assert mined.n_widened == 1
    assert np.all(mined.table[0] >= 0)  # widened rather than left half-empty
    assert mined.length_gap > 1  # and the diagnostic admits the table is no longer matched

    # The five-word sentences find each other inside the tolerance and never widen.
    assert not bool(mined.widened[1])


def test_matched_negatives_never_include_the_anchor_itself(role_matrix: np.ndarray) -> None:
    """No sentence is its own hard negative, which would put the answer in the negative set."""
    mined = mine_matched_hard_negatives(ROLE_TEXTS, role_matrix, k=3, length_tol=4)
    for i, row in enumerate(mined.table):
        assert i not in row


def test_matched_negatives_reports_the_widened_count_as_a_property() -> None:
    """`n_widened` counts the flagged sentences, so a degraded table is one number away."""
    diag = MatchedNegatives(
        table=np.zeros((3, 2), dtype=np.int64),
        widened=np.array([True, False, True]),
        length_gap=4,
        piece_gap=-1,
    )
    assert diag.n_widened == 2


def test_matched_miner_rejects_a_mismatched_piece_table(role_matrix: np.ndarray) -> None:
    """A piece table with the wrong number of rows is an error, not a silently misaligned constraint."""
    with pytest.raises(ValueError, match='word_pieces'):
        mine_matched_hard_negatives(ROLE_TEXTS, role_matrix, word_pieces=np.zeros((2, 5), dtype=np.int16))


# --------------------------------------------------------------------------- #
# the narrowed denominator
# --------------------------------------------------------------------------- #
def _gallery(hard_only: bool, table: torch.Tensor | None, *, band: int = 0) -> GalleryContrast:
    """A gallery scorer with the hard-negative table optionally attached and optionally in force."""
    scorer = GalleryContrast(band=band, min_candidates=32, hard_negatives_only=hard_only)
    if table is not None:
        scorer.attach_hard_negatives(table)
    return scorer


def test_narrowed_denominator_holds_only_the_anchor_and_its_mined_negatives() -> None:
    """With the table in force an anchor is scored against its own text plus its mined negatives, nothing else."""
    table = torch.tensor([[2, 3], [4, 5], [0, 1], [1, -1], [0, -1], [0, -1]])
    scorer = _gallery(True, table)

    mask = scorer.candidate_mask(torch.tensor([0, 3]), n_texts=6)
    assert [int(i) for i in mask[0].nonzero().flatten()] == [0, 2, 3]
    assert [int(i) for i in mask[1].nonzero().flatten()] == [1, 3]  # -1 padding contributes nothing


def test_the_anchor_own_text_always_survives_the_narrowed_denominator() -> None:
    """Every anchor keeps its numerator, because a softmax with no target column is not a loss."""
    n_texts = 8
    table = torch.full((n_texts, 2), -1, dtype=torch.long)
    table[:, 0] = torch.arange(n_texts).roll(1)
    scorer = _gallery(True, table)

    text_id = torch.arange(n_texts)
    mask = scorer.candidate_mask(text_id, n_texts)
    assert bool(mask[torch.arange(n_texts), text_id].all())


def test_an_anchor_with_no_admissible_mined_negative_keeps_its_wider_denominator() -> None:
    """Narrowing to an empty negative set would zero that anchor's loss, so it falls back to the band instead."""
    n_texts = 6
    table = torch.full((n_texts, 1), -1, dtype=torch.long)  # nothing mined for anyone
    scorer = _gallery(True, table)
    scorer.restrict_to([0, 1, 2, 3, 4, 5], n_texts)

    mask = scorer.candidate_mask(torch.tensor([0, 1]), n_texts)
    assert int(mask.sum(dim=1).min()) == n_texts


def test_narrowing_respects_the_split_restriction() -> None:
    """A mined negative the training split must never see stays out of the denominator."""
    n_texts = 5
    table = torch.tensor([[1, 4], [0, 4], [0, 1], [0, 1], [0, 1]])
    scorer = _gallery(True, table)
    scorer.restrict_to([0, 1, 2, 3], n_texts)  # text 4 is a held-out stimulus

    mask = scorer.candidate_mask(torch.tensor([0]), n_texts)
    assert [int(i) for i in mask[0].nonzero().flatten()] == [0, 1]


def test_narrowing_shrinks_the_candidate_count_the_metrics_report() -> None:
    """The gallery term reports the narrowed denominator, so a run cannot hide how few distractors it faced."""
    torch.manual_seed(0)
    n_texts, dim = 12, 8
    gallery = torch.nn.functional.normalize(torch.randn(n_texts, dim), dim=-1)
    z_eeg = torch.nn.functional.normalize(torch.randn(4, dim), dim=-1)
    text_id = torch.arange(4)
    table = torch.stack([torch.tensor([(i + 1) % n_texts, (i + 2) % n_texts]) for i in range(n_texts)])
    scale = torch.tensor(10.0)

    wide = _gallery(False, table).compute(z_eeg, gallery, text_id, scale)[1]
    narrow = _gallery(True, table).compute(z_eeg, gallery, text_id, scale)[1]
    assert wide['gallery_candidates'] == float(n_texts)
    assert narrow['gallery_candidates'] == 3.0  # the anchor plus its two mined negatives


# --------------------------------------------------------------------------- #
# the byte-identical guarantee
# --------------------------------------------------------------------------- #
def test_hard_negatives_off_leaves_the_gallery_loss_bitwise_identical() -> None:
    """Attaching a table without `hard_negative_in_loss` must not move the loss by a single bit."""
    torch.manual_seed(0)
    n_texts, dim = 16, 8
    gallery = torch.nn.functional.normalize(torch.randn(n_texts, dim), dim=-1)
    z_eeg = torch.nn.functional.normalize(torch.randn(6, dim), dim=-1)
    text_id = torch.arange(6)
    table = torch.stack([torch.tensor([(i + 1) % n_texts, (i + 3) % n_texts]) for i in range(n_texts)])
    lengths = torch.arange(n_texts) % 5 + 3
    scale = torch.tensor(12.0)

    baseline = GalleryContrast(band=2, min_candidates=4)
    baseline.attach_lengths(lengths)
    attached = GalleryContrast(band=2, min_candidates=4, hard_negatives_only=False)
    attached.attach_lengths(lengths)
    attached.attach_hard_negatives(table)

    before = baseline.compute(z_eeg, gallery, text_id, scale)
    after = attached.compute(z_eeg, gallery, text_id, scale)
    assert torch.equal(before[0], after[0])
    assert before[1] == after[1]


def test_hard_negatives_off_leaves_the_clip_loss_bitwise_identical() -> None:
    """End to end: the whole CLIP objective is unmoved by a table the configuration does not switch on."""
    texts = [f'sentence number {i} reads like this' for i in range(6)]
    table = torch.stack([torch.tensor([(i + 1) % 6, (i + 2) % 6]) for i in range(6)])

    def _run(hard_negatives: torch.Tensor | None, in_loss: bool) -> torch.Tensor:
        torch.manual_seed(7)
        cfg = ObjectiveConfig(
            name='clip',
            gallery_weight=1.0,
            gallery_length_band=2,
            gallery_min_candidates=3,
            hard_negative_in_loss=in_loss,
            variance_weight=0.0,
            covariance_weight=0.0,
        )
        model = build_model(ModelConfig(embed_dim=32, hidden_dim=16, n_layers=1, n_subjects=2), in_dim=10)
        objective = build_objective(cfg, model, feature_dim=10)
        assert isinstance(objective, SentenceClipObjective)
        matrix = torch.nn.functional.normalize(torch.randn(6, 12), dim=-1)
        objective.attach_text(matrix, text_word_counts(texts), list(range(6)), None, hard_negatives)

        batch = {
            'features': torch.randn(4, 5, 10),
            'pad_mask': torch.ones(4, 5, dtype=torch.bool),
            'presence': torch.ones(4, 5, dtype=torch.bool),
            'subject': torch.tensor([0, 0, 1, 1]),
            'content_id': torch.zeros(4, 5, dtype=torch.long),
            'word_id': torch.arange(20).reshape(4, 5),
            'task_id': torch.zeros(4, dtype=torch.long),
            'sentence_text_id': torch.tensor([0, 1, 2, 3]),
        }
        return objective.compute(model, batch)[0].detach()

    assert torch.equal(_run(None, False), _run(table, False))
    assert not torch.equal(_run(None, False), _run(table, True))


def test_build_gallery_contrast_defaults_the_narrowing_off() -> None:
    """A configuration that never heard of the knob builds the old scorer."""
    scorer = build_gallery_contrast(ObjectiveConfig(name='clip', gallery_weight=1.0), n_texts=32)
    assert scorer is not None
    assert scorer.hard_negatives_only is False
    assert scorer.hard_negatives is None
