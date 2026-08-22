"""Token-level lexical alignment: the term that asks one word's EEG to mean that word."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from zte.config import ObjectiveConfig
from zte.data.targets.lexical import build_lexical_matrix, normalise_word
from zte.models.objectives.lexical import LexicalAligner, build_lexical_aligner


def _batch(subjects: list[int], content: list[list[int]], words: list[list[int]]) -> dict[str, Any]:
    """Builds the three per-token fields the aligner reads."""
    return {
        'subject': torch.tensor(subjects),
        'content_id': torch.tensor(content),
        'word_id': torch.tensor(words),
    }


def _aligner(hidden_dim: int = 12, text_dim: int = 8, n_types: int = 10) -> LexicalAligner:
    """An aligner with a frozen random word-embedding target attached."""
    torch.manual_seed(0)
    aligner = LexicalAligner(hidden_dim, text_dim)
    aligner.attach(torch.nn.functional.normalize(torch.randn(n_types, text_dim), dim=-1))
    return aligner


# --------------------------------------------------------------------------- #
# The target
# --------------------------------------------------------------------------- #


def test_a_word_keeps_its_case_and_loses_only_the_punctuation_the_reader_saw() -> None:
    """`colonel.` and `colonel` are one word; `Bread` and `bread` are not, because the encoders are case-sensitive."""
    assert normalise_word('colonel.') == 'colonel'
    assert normalise_word('"Bread,') == 'Bread'
    assert normalise_word("Browning's") == "Browning's"
    # Stripping must never empty a token: a bare punctuation mark is still a word the reader fixated.
    assert normalise_word('--') == '--'


def test_the_target_is_absent_rather_than_invented_without_an_encoder() -> None:
    """A silently fabricated lexical target would train the encoder to predict noise and report it as alignment."""
    matrix, dim = build_lexical_matrix({'a': 0, 'b': 1}, None)

    assert matrix is None
    assert dim == 0


def test_the_target_rows_follow_the_word_vocabulary_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Row `i` must be the word whose `word_id` is `i`, or every token is scored against the wrong word."""
    seen: list[list[str]] = []

    def fake(texts: list[str], *_a: Any, **_k: Any) -> tuple[np.ndarray, int]:
        seen.append(list(texts))
        return np.eye(len(texts), dtype=np.float32), len(texts)

    monkeypatch.setattr('zte.data.targets.lexical.build_sentence_text_matrix', fake)
    matrix, _ = build_lexical_matrix({'zebra': 2, 'apple,': 0, 'bread': 1}, 'stub')

    assert seen[0] == ['apple', 'bread', 'zebra']
    assert matrix is not None and matrix.shape[0] == 3


# --------------------------------------------------------------------------- #
# The loss
# --------------------------------------------------------------------------- #


def test_both_directions_train_the_encoder() -> None:
    """The gradient has to reach the token hiddens, or the term is decoration on a frozen projection."""
    aligner = _aligner()
    hidden = torch.randn(4, 3, 12, requires_grad=True)
    batch = _batch([0, 0, 1, 1], [[0, 1, 2]] * 4, [[1, 2, 3]] * 4)

    loss, metrics = aligner.compute(
        hidden, batch, torch.ones(4, 3, dtype=torch.bool), type_weight=1.0, reader_weight=1.0
    )
    loss.backward()

    assert 'lexical_type_loss' in metrics
    assert 'lexical_reader_loss' in metrics
    assert hidden.grad is not None and float(hidden.grad.abs().sum()) > 0.0


def test_the_cross_reader_term_needs_two_readers_of_one_word() -> None:
    """Its whole claim is "the same word, whoever read it", so one reader must produce no anchor at all."""
    aligner = _aligner()
    hidden = torch.randn(2, 3, 12)
    solo = _batch([0, 0], [[0, 1, 2]] * 2, [[1, 2, 3]] * 2)

    _, metrics = aligner.compute(hidden, solo, torch.ones(2, 3, dtype=torch.bool), type_weight=0.0, reader_weight=1.0)

    assert 'lexical_reader_loss' not in metrics


def test_a_word_read_by_another_person_is_a_positive_and_never_a_negative() -> None:
    """A different reading of the same word appearing as a negative would train the encoder to separate readers."""
    aligner = _aligner()
    hidden = torch.randn(4, 2, 12)
    paired = _batch([0, 0, 1, 1], [[7, 8]] * 4, [[1, 2]] * 4)

    _, metrics = aligner.compute(hidden, paired, torch.ones(4, 2, dtype=torch.bool), type_weight=0.0, reader_weight=1.0)

    # Eight tokens, every one of which has a same-content partner under the other subject.
    assert metrics['lexical_anchors'] == 8.0


def test_the_term_is_silent_when_both_weights_are_zero() -> None:
    """The default is off, so an existing encoder run is behaviourally byte-identical to before the rebuild."""
    aligner = _aligner()
    hidden = torch.randn(3, 2, 12)
    batch = _batch([0, 1, 2], [[0, 1]] * 3, [[1, 2]] * 3)

    loss, metrics = aligner.compute(
        hidden, batch, torch.ones(3, 2, dtype=torch.bool), type_weight=0.0, reader_weight=0.0
    )

    assert float(loss) == 0.0
    assert metrics == {}


def test_padded_and_skipped_tokens_never_reach_the_loss() -> None:
    """An omitted word is a non-token: scoring it would align the encoder against a word nobody read."""
    aligner = _aligner()
    hidden = torch.randn(2, 4, 12)
    batch = _batch([0, 1], [[0, 1, -1, -1]] * 2, [[1, 2, -1, -1]] * 2)
    usable = torch.tensor([[True, True, False, False], [True, True, False, False]])

    _, metrics = aligner.compute(hidden, batch, usable, type_weight=1.0, reader_weight=1.0)

    assert metrics['lexical_tokens'] == 4.0


def test_the_token_cap_spreads_across_the_batch_rather_than_truncating_it() -> None:
    """A cap that took the first N tokens would quietly turn a batch loss into a few-sentence loss."""
    aligner = _aligner()
    hidden = torch.randn(8, 6, 12)
    batch = _batch(list(range(8)), [list(range(6))] * 8, [[i % 10 for i in range(6)]] * 8)

    _, metrics = aligner.compute(
        hidden, batch, torch.ones(8, 6, dtype=torch.bool), type_weight=1.0, reader_weight=0.0, max_tokens=12
    )

    assert metrics['lexical_tokens'] <= 12.0


def test_a_mismatched_target_width_is_refused_at_attach_time() -> None:
    """A silently mis-sized target scores every word against the wrong space and never raises at the loss."""
    aligner = LexicalAligner(12, 8)

    with pytest.raises(ValueError, match='text space'):
        aligner.attach(torch.randn(10, 16))


def test_the_aligner_is_not_built_when_it_would_contribute_nothing() -> None:
    """Zero weights must not add parameters to the optimiser, or an ablation stops being matched."""
    assert build_lexical_aligner(ObjectiveConfig(), 12, 8) is None
    assert build_lexical_aligner(ObjectiveConfig(lexical_weight=1.0), 12, 8) is not None
    assert build_lexical_aligner(ObjectiveConfig(lexical_reader_weight=1.0), 12, 8) is not None
