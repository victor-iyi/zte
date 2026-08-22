"""Tests for sub-word token alignment, its frozen target, and the piece-profile oracle that gates it."""

from pathlib import Path

import numpy as np
import pytest
import torch

from zte.config import ModelConfig
from zte.data.targets.tokens import (
    TokenAlignment,
    _offsets,
    _slot_words,
    _word_start_offsets,
    build_subword_matrix,
    build_token_alignment,
)
from zte.evaluation.audit.rebaseline import piece_profile_report, piece_signatures, signature_oracle
from zte.models.frontends.raw_conformer import RawConformer
from zte.models.objectives.token import TokenAligner


class _ByteLevelFastTokenizer:
    """A fast tokeniser that folds the preceding space into each word-initial token, as Qwen's and GPT-2's do."""

    is_fast = True
    name_or_path = 'byte-level-fast'

    def __call__(
        self, texts: list[str], return_offsets_mapping: bool = False, add_special_tokens: bool = True
    ) -> dict[str, list[list[tuple[int, int]]]]:
        """One span per whitespace-delimited piece, each reaching back over the space in front of it."""
        rows = []
        for text in texts:
            spans: list[tuple[int, int]] = []
            cursor = 0
            for index, piece in enumerate(text.split(' ')):
                begin = cursor - 1 if index else cursor
                spans.append((begin, cursor + len(piece)))
                cursor += len(piece) + 1
            rows.append(spans)
        return {'offset_mapping': rows}


class _BosFastTokenizer:
    """A fast tokeniser that prepends a special token, which is what exposes an offset/id misalignment."""

    is_fast = True
    name_or_path = 'bos-fast'

    def __call__(
        self, texts: list[str], return_offsets_mapping: bool = False, add_special_tokens: bool = True
    ) -> dict[str, list[list[tuple[int, int]]]]:
        """Returns one span per whitespace-delimited piece, optionally behind a zero-width special."""
        rows = []
        for text in texts:
            spans: list[tuple[int, int]] = [(0, 0)] if add_special_tokens else []
            cursor = 0
            for piece in text.split(' '):
                spans.append((cursor, cursor + len(piece)))
                cursor += len(piece) + 1
            rows.append(spans)
        return {'offset_mapping': rows}


# ---- The word-to-sub-word map ---- #


def test_every_word_owns_the_punctuation_that_follows_it(tmp_path: str) -> None:
    """A trailing comma belongs to the word before it, or a sixth of ZuCo's sub-words carry no target."""
    alignment = build_token_alignment(
        ['He was born in Fillmore, Utah.'],
        [['He', 'was', 'born', 'in', 'Fillmore,', 'Utah.']],
        'tiny',
        max_length=64,
        cache_dir=str(tmp_path),
    )

    # 'Fillmore,' plus its following space is ten characters, and the tiny tokeniser is one token per character.
    assert int(alignment.word_pieces[0, 4]) == 10
    assert alignment.coverage == 1.0


def test_a_special_token_is_attributed_to_no_word(tmp_path: str) -> None:
    """Offsets and ids must be tokenised under the same special-token rule or every slot shifts one word left."""
    text = 'He was born in Fillmore'
    words = ['He', 'was', 'born', 'in', 'Fillmore']
    offsets = _offsets(_BosFastTokenizer(), [text], 64)[0]
    owners = _slot_words(offsets, _word_start_offsets(text, words), len(words))

    assert int(owners[0]) == -1, 'the leading special must own no word'
    assert [int(o) for o in owners[1:]] == [0, 1, 2, 3, 4], 'every word must own exactly its own slot'


def test_a_word_initial_token_that_reaches_back_over_its_space_still_belongs_to_its_own_word() -> None:
    """A byte-level BPE folds the preceding space into the word-initial token.

    Note:
        Its span therefore *begins* one character inside the previous word. Placing a slot by where it starts
        attributes every word to the one before it -- silently, at coverage 1.0, and the arm then measures how well
        EEG predicts the next word's spelling. The slot is placed by its last character for exactly this reason.
    """
    text = 'Emperor Hirohito relinquished sovereignty'
    words = text.split()
    offsets = _offsets(_ByteLevelFastTokenizer(), [text], 64)[0]
    owners = _slot_words(offsets, _word_start_offsets(text, words), len(words))

    assert [int(o) for o in owners] == [0, 1, 2, 3], "each word must own its own token, not its successor's"


def test_a_word_the_reference_does_not_spell_is_counted_against_coverage(tmp_path: str) -> None:
    """An unmatched word must not shift the words after it; it is reported, not silently absorbed."""
    alignment = build_token_alignment(
        ['alpha beta gamma'],
        [['alpha', 'DELTA', 'gamma']],
        'tiny',
        max_length=64,
        cache_dir=str(tmp_path),
    )

    assert alignment.coverage < 1.0
    assert int(alignment.word_pieces[0, 1]) == 0, 'the unmatched word carries no pieces'
    assert int(alignment.word_pieces[0, 2]) > 0, 'the word after it still aligns'


def test_the_alignment_round_trips_through_its_cache(tmp_path: str) -> None:
    """A cached table must return the same map, or a resumed run trains against a different target."""
    args = (['one two three'], [['one', 'two', 'three']], 'tiny')
    first = build_token_alignment(*args, max_length=32, cache_dir=str(tmp_path))
    second = build_token_alignment(*args, max_length=32, cache_dir=str(tmp_path))

    assert np.array_equal(first.token_word, second.token_word)
    assert np.array_equal(first.word_pieces, second.word_pieces)
    assert first.fingerprint == second.fingerprint


def test_pieces_per_word_refuses_indices_it_does_not_carry() -> None:
    """An out-of-range join must read zero rather than an arbitrary neighbouring word's count."""
    alignment = TokenAlignment(
        token_word=np.zeros((2, 4), dtype=np.int32),
        piece_index=np.zeros((2, 4), dtype=np.int16),
        word_pieces=np.array([[2, 3, 0], [1, 1, 1]], dtype=np.int16),
        fingerprint='x',
        coverage=1.0,
    )
    got = alignment.pieces_per_word(np.array([0, 1, 0, 9]), np.array([1, 2, 99, 0]))

    assert [int(v) for v in got] == [3, 1, 0, 0]


# ---- The frozen sub-word target ---- #


def test_the_subword_target_carries_only_the_types_the_corpus_uses() -> None:
    """Qwen's table is 151,936 rows; carrying it whole is 544 MB of frozen buffer for a few thousand pieces."""
    target = build_subword_matrix(np.array([[4, 5, 4, -1], [9, 5, 0, -1]]), 'tiny', dim=8)

    assert set(target.rows) == {0, 4, 5, 9}
    assert target.matrix.shape == (4, 8)
    assert np.allclose(np.linalg.norm(target.matrix, axis=1), 1.0, atol=1e-5)
    assert int(target.compact(np.array([12345]))[0]) == -1, 'an unseen type maps to -1, never to row 0'


# ---- The intra-word EEG path ---- #


def _frontend() -> RawConformer:
    config = ModelConfig(frontend='raw_conformer', hidden_dim=16, n_heads=2, n_layers=2, conformer_filters=8)
    return RawConformer(4, 24, config).eval()


def test_one_sub_token_reproduces_the_pooled_forward_pass() -> None:
    """The sub-token path must degenerate exactly to today's mean pool, or the level comparison is not matched."""
    torch.manual_seed(0)
    model = _frontend()
    x = torch.randn(2, 3, 4, 24)
    with torch.no_grad():
        pooled, single = model(x), model.sub_tokens(x, 1)

    assert torch.allclose(single[..., 0, :], pooled, atol=1e-6)


def test_the_sub_token_count_is_a_constant_and_not_read_from_the_text() -> None:
    """A word encoded by how many pieces its reference spells makes retrieval a piece-profile match test."""
    torch.manual_seed(0)
    model = _frontend()
    x = torch.randn(2, 3, 4, 24)
    with torch.no_grad():
        got = model.sub_tokens(x, 5)

    assert got.shape == (2, 3, 5, 16)
    with pytest.raises(ValueError, match='must be positive'):
        model.sub_tokens(x, 0)


def test_the_frontend_still_returns_one_vector_per_word() -> None:
    """`token_hidden` broadcasts FiLM and residual coding over a rank-3 tensor and mis-broadcasts silently on rank 4."""
    torch.manual_seed(0)
    with torch.no_grad():
        out = _frontend()(torch.randn(2, 3, 4, 24))

    assert out.ndim == 3


# ---- The oracle that gates every token-level number ---- #


def test_a_signature_shared_by_everyone_gives_away_nothing() -> None:
    """The empirical zero: an oracle resolving no sentence must score chance and report zero bits."""
    block = signature_oracle(['same'] * 20)

    assert block['top1'] == pytest.approx(1 / 20)
    assert block['information_bits'] == pytest.approx(0.0, abs=1e-9)


def test_a_unique_signature_identifies_every_sentence() -> None:
    """The other end: a signature unique per sentence resolves the gallery outright."""
    block = signature_oracle([f's{i}' for i in range(32)])

    assert block['top1'] == pytest.approx(1.0)
    assert block['information_bits'] == pytest.approx(np.log2(32))
    assert block['unique_fraction'] == pytest.approx(1.0)


def test_the_piece_profile_outranks_the_word_count_it_refines() -> None:
    """Word count is the documented 5.14-bit confound; the per-word profile is strictly finer and must score above it."""
    rng = np.random.default_rng(0)
    pieces = np.zeros((700, 60), dtype=np.int64)
    for row in range(700):
        length = int(np.clip(rng.normal(19.6, 8.0), 4, 60))
        pieces[row, :length] = rng.choice([1, 2, 3], size=length, p=[0.68, 0.24, 0.08])

    report = piece_profile_report(pieces, observed_top1=26 / 700)
    scores = {kind: block['top1'] for kind, block in report['oracles'].items()}

    assert scores['words'] < scores['total'] < scores['multiset'] < scores['profile']
    assert scores['profile'] > 0.9, 'the ordered profile resolves almost every sentence on a 700-item gallery'
    assert report['beats_oracles'] is False, "the flagship's 26/700 sits far below a brain-free piece oracle"
    assert 'BELOW' in report['verdict']


def test_the_oracle_verdict_is_unmeasured_until_a_run_is_compared_against_it() -> None:
    """A missing observation must read as not measured, never as a pass."""
    report = piece_profile_report(np.array([[1, 2, 0], [2, 1, 1]], dtype=np.int64))

    assert report['beats_oracles'] is None
    assert report['verdict'] == 'not measured'


def test_every_piece_signature_is_named_explicitly() -> None:
    """A typo'd signature must fail loudly rather than silently scoring the wrong oracle."""
    pieces = np.array([[1, 2, 0]], dtype=np.int64)

    assert piece_signatures(pieces, 'profile') == [(1, 2)]
    assert piece_signatures(pieces, 'total') == [3]
    assert piece_signatures(pieces, 'words') == [2]
    with pytest.raises(ValueError, match='Unknown piece signature'):
        piece_signatures(pieces, 'nonsense')  # type: ignore[arg-type]


# ---- The token objective ---- #


def _aligner(n_sub: int = 3, hidden: int = 8, text: int = 6, n_content: int = 4) -> TokenAligner:
    torch.manual_seed(0)
    aligner = TokenAligner(hidden, text, n_sub)
    matrix = torch.nn.functional.normalize(torch.randn(5, text), dim=-1)
    target = torch.full((n_content, n_sub), -1, dtype=torch.long)
    rows = [[0, 1], [2], [3, 4, 0]]
    for content, pieces in enumerate(rows[:n_content]):
        kept = pieces[:n_sub]
        target[content, : len(kept)] = torch.tensor(kept)
    aligner.attach(matrix, target)
    return aligner


def _batch(n_sub: int = 3, hidden: int = 8) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    torch.manual_seed(1)
    sub_hidden = torch.randn(4, 2, n_sub, hidden, requires_grad=True)
    batch = {
        'content_id': torch.tensor([[0, 1], [0, 1], [2, 0], [2, 1]]),
        'subject': torch.tensor([0, 1, 0, 1]),
    }
    return sub_hidden, batch, torch.ones(4, 2, dtype=torch.bool)


def test_the_token_loss_reaches_the_projection_it_trains() -> None:
    """A level that produces a number but no gradient is wiring that looks like science."""
    aligner = _aligner()
    sub_hidden, batch, usable = _batch()
    loss, metrics = aligner.compute(sub_hidden, batch, usable, type_weight=1.0, reader_weight=1.0)
    loss.backward()

    assert metrics['token_type_top1'] >= 0.0
    assert aligner.head.weight.grad is not None
    assert float(aligner.head.weight.grad.norm()) > 0.0


def test_a_slot_past_the_word_s_own_piece_count_is_never_scored() -> None:
    """The sub-word count may mask the loss and nothing else; a slot with no piece has no target to pull toward."""
    aligner = _aligner()
    sub_hidden, batch, usable = _batch()
    _, metrics = aligner.compute(sub_hidden, batch, usable, type_weight=1.0, reader_weight=0.0)

    # Words 0/1/2 carry 2/1/3 pieces and the batch reads them 3/3/2 times, so 15 of the 24 slots have a target.
    assert metrics['token_sub_tokens_scored'] == float(2 * 3 + 1 * 3 + 3 * 2)


def test_a_word_s_own_slices_are_never_each_other_s_negatives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adjacent slices of one fixation are near-identical, so separating them means encoding a within-word clock.

    Note:
        Exercised by making two slices of one word the *only* thing that could be told apart. With the exclusion on
        there is no usable anchor at all and the loss is exactly zero; with it off the pair enters the denominator
        and the loss is positive. Flipping the constant therefore has to change the number.
    """
    import zte.models.objectives.token as token_module

    aligner = _aligner(n_sub=2, n_content=2)
    torch.manual_seed(3)
    # Two readers, one word each, and the slices of that word are the only other rows in the batch.
    sub_hidden = torch.randn(2, 1, 2, 8)
    batch = {'content_id': torch.tensor([[0], [0]]), 'subject': torch.tensor([0, 1])}
    usable = torch.ones(2, 1, dtype=torch.bool)

    excluded, _ = aligner.compute(sub_hidden, batch, usable, type_weight=0.0, reader_weight=1.0)
    monkeypatch.setattr(token_module, '_EXCLUDE_SAME_WORD', False)
    included, _ = aligner.compute(sub_hidden, batch, usable, type_weight=0.0, reader_weight=1.0)

    assert float(excluded) == 0.0, "a word's own slices must leave no anchor with a usable denominator"
    assert float(included) > 0.0, 'with the exclusion off the same-word pair enters the denominator'


def test_the_level_builds_no_parameter_at_all_when_it_is_switched_off() -> None:
    """A submodule built unconditionally breaks `strict=True` loading of every checkpoint written before it."""
    from zte.config import ObjectiveConfig
    from zte.models.objectives.token import build_token_aligner

    assert build_token_aligner(ObjectiveConfig(name='clip'), 8, 6) is None
    on = ObjectiveConfig(name='clip', token_weight=1.0)
    assert build_token_aligner(on, 8, 6) is not None


def test_the_target_must_agree_with_the_sub_token_count_the_frontend_emits() -> None:
    """A silent width mismatch would align slot k of the EEG against piece k of a different word."""
    aligner = TokenAligner(8, 6, n_sub=3)
    matrix = torch.nn.functional.normalize(torch.randn(5, 6), dim=-1)
    with pytest.raises(ValueError, match='sub-tokens per word'):
        aligner.attach(matrix, torch.zeros((4, 2), dtype=torch.long))
    with pytest.raises(ValueError, match='must name the same space'):
        aligner.attach(torch.randn(5, 7), torch.zeros((4, 3), dtype=torch.long))


# ---- The leak guard ---- #


def test_the_pooled_vector_carries_no_more_fixation_length_than_the_word_arm_would() -> None:
    """The instrument the structural guard cannot be: does the scored vector actually carry the length channel?

    Note:
        ZuCo's raw window is a variable-length fixation zero-padded to 350 samples and then z-scored, so the tail is
        an exactly constant value starting at sample L -- the fixation length, handed to the encoder for free by the
        substrate. The sub-word loss supervises slot k only for words with more than k pieces, and piece count moves
        with word length, so the cheapest way to satisfy it is to read that boundary. A structural assertion cannot
        see this; only a probe on the scored vector can, and this is the shape of that probe.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(0)
    lengths = rng.integers(60, 350, size=240)

    # A representation that has read the padding boundary, and one that has not.
    reads_edge = np.stack([np.concatenate([[float(n)], rng.standard_normal(15)]) for n in lengths])
    blind = rng.standard_normal((240, 16))

    def readable(matrix: np.ndarray) -> float:
        model = RidgeCV(alphas=np.logspace(-2, 6, 17))
        return float(np.mean(cross_val_score(model, matrix, lengths.astype(float), cv=3, scoring='r2')))

    assert readable(reads_edge) > 0.9, 'the probe must see the channel when it is present'
    assert readable(blind) < 0.05, 'and must not invent it when it is absent'


def test_nothing_that_produces_a_scored_number_reaches_the_intra_word_encoder() -> None:
    """The structural half of the guard above, and the one that survives the code growing around it.

    Note:
        `sentence_hidden` and `embed_sentence` are what retrieval is scored on. If a sub-token tensor ever reaches
        them the piece profile reaches them too, and no numerical assertion downstream would notice. So the rule is
        not a caller count -- it is that nothing under `inference/`, `evaluation/` or `parallax/` may call it, and
        that every caller elsewhere is named here with why it is allowed.
    """
    import zte

    root = Path(zte.__file__).resolve().parent
    callers = {
        path.relative_to(root).as_posix()
        for path in root.rglob('*.py')
        for line in path.read_text(encoding='utf-8').splitlines()
        if ('sub_token_hidden(' in line or '.sub_tokens(' in line) and not line.lstrip().startswith(('#', '"', "'"))
    }
    scored = sorted(c for c in callers if c.startswith(('inference/', 'evaluation/', 'parallax/')))

    assert not scored, f'the intra-word path is reachable from a scored path: {scored}'
    assert callers == {
        'models/embedding.py',  # the definition, and its own call into the frontend
        'models/objectives/base.py',  # the training-side token level, the one loss that may see sub-tokens
        'cli/visualize.py',  # the atlas, which renders a picture and scores nothing
    }, f'the intra-word path gained a caller: {sorted(callers)}'


def test_the_word_and_sub_token_paths_stay_separate_shapes() -> None:
    """A sub-token tensor escaping into the pooled path would carry the piece profile into retrieval."""
    from zte.config import ModelConfig
    from zte.models.embedding import build_model

    torch.manual_seed(0)
    config = ModelConfig(
        frontend='raw_conformer',
        embed_dim=16,
        hidden_dim=8,
        n_layers=2,
        n_heads=2,
        conformer_filters=8,
        factored=False,
        subject_adapter=False,
    )
    model = build_model(config, raw_shape=(4, 24)).eval()
    batch = {
        'raw': torch.randn(2, 3, 4, 24),
        'pad_mask': torch.ones(2, 3, dtype=torch.bool),
        'presence': torch.ones(2, 3, dtype=torch.bool),
        'subject': torch.tensor([0, 1]),
        'subject_signature': None,
    }
    with torch.no_grad():
        word = model.token_hidden(batch)
        sub = model.sub_token_hidden(batch, 3)

    assert word.shape == (2, 3, 8), 'the word path stays one hidden per word'
    assert sub.shape == (2, 3, 3, 8), 'the sub-token path is reached only when asked for explicitly'


# ---- The decoder's pointer rate ---- #


def test_the_evidence_pointer_rate_is_fitted_on_the_training_stimuli_only() -> None:
    """The gallery is whole-dataset, so an unrestricted rate is averaged over the held-out sentences as well.

    Note:
        One scalar is a small leak, but it is a transductive fit a decoder facing one sentence at a time cannot
        reproduce, and `CLAUDE.md` requires the fit to travel with the number rather than be assumed away.
    """
    from zte.config import DecoderConfig, ModelConfig, ObjectiveConfig
    from zte.models.embedding import build_model
    from zte.models.objectives.decode import PrefixDecodeObjective

    torch.manual_seed(0)
    model = build_model(ModelConfig(frontend='band_power_mlp', embed_dim=16, hidden_dim=8, n_layers=1, n_heads=2), 6)
    decoder = DecoderConfig(lm_source='tiny', evidence_schedule='linear', evidence_tokens_per_word=0.0)
    objective = PrefixDecodeObjective(ObjectiveConfig(name='decode'), model, decoder)
    if objective.evidence is None:
        pytest.skip('this build carries no evidence path to fit a rate for')

    # Rows 0-1 are the training split; row 2 is held out and is spelled with far more tokens per word.
    ids = torch.zeros((3, 12), dtype=torch.long)
    mask = torch.zeros((3, 12), dtype=torch.bool)
    mask[0, :4] = mask[1, :4] = True
    mask[2, :12] = True
    words = torch.tensor([4, 4, 4], dtype=torch.long)

    objective.attach_tokens(ids, mask, words, rate_text_ids=[0, 1])
    honest, fit = objective.evidence.pointer.tokens_per_word, objective.evidence_rate_fit
    objective.attach_tokens(ids, mask, words)

    assert fit == 'train split'
    assert objective.evidence_rate_fit == 'transductive'
    assert honest == pytest.approx(1.0), 'the training rows spell one token per word'
    assert objective.evidence.pointer.tokens_per_word > honest, 'the held-out row must have moved the rate'


def test_the_cli_gate_reads_the_run_s_own_top1_rather_than_a_key_that_does_not_exist() -> None:
    """A gate that silently reads `None` renders `not measured` for ever and licenses anything.

    Note:
        `rebaseline_report` has no `held_out` block -- that shape belongs to a parallax transfer cell. The honest
        cell is `grid.train_fitted.full`, the same one the length oracle is compared against, so both floors are
        read against one number.
    """
    from zte.cli.rebaseline import observed_held_out_top1

    honest = {'grid': {'train_fitted': {'full': {'top1': 0.0371}}}}
    assert observed_held_out_top1(honest) == pytest.approx(0.0371)

    # The parallax shape the bug reached for, and the empty cases, must all read as no observation at all.
    assert observed_held_out_top1({'held_out': {'full': {'top1': 0.0371}}}) is None
    assert observed_held_out_top1({'grid': {'train_fitted': {'full': None}}}) is None
    assert observed_held_out_top1({}) is None
