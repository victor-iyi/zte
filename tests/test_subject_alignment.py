"""Tests for exp12: raw Euclidean alignment, the inferred subject signature and its adapter."""

from __future__ import annotations

import numpy as np
import torch

from zte.config import ModelConfig, ObjectiveConfig
from zte.data.features.alignment import RawSubjectAligner
from zte.evaluation.audit.scoreboard import _binom_tail_p, _bootstrap_ci
from zte.models.embedding import build_model
from zte.models.objectives.losses import identity_orthogonality
from zte.models.subject import SubjectAdapter

N_CH, N_T = 24, 48


def _cohort(seed: int = 0, n: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Three subjects whose only difference is a per-subject linear channel mixing."""
    rng = np.random.default_rng(seed)
    windows, subjects = [], []
    for i, code in enumerate('ABC'):
        mixing = rng.normal(size=(N_CH, N_CH)) * (0.5 + i)
        windows.append(np.einsum('cd,ndt->nct', mixing, rng.normal(size=(n, N_CH, N_T))))
        subjects += [code] * n
    return np.concatenate(windows).astype(np.float32), np.array(subjects)


def _identity_gap(windows: np.ndarray) -> float:
    """Mean absolute deviation of the trace-normalised channel covariance from the identity."""
    cov = np.einsum('nct,ndt->cd', windows, windows) / (len(windows) * N_T)
    cov = cov / (np.trace(cov) / N_CH)
    return float(np.abs(cov - np.eye(N_CH)).mean())


def test_alignment_whitens_every_subject_toward_identity() -> None:
    """Each subject's own covariance goes to the identity, so the forward-model difference is cancelled."""
    raw, subjects = _cohort()
    before = [_identity_gap(raw[subjects == c]) for c in 'ABC']

    aligned = RawSubjectAligner().fit(raw, subjects).transform(raw.copy(), subjects)
    after = [_identity_gap(aligned[subjects == c]) for c in 'ABC']

    assert all(a < b / 2 for a, b in zip(after, before, strict=True)), (before, after)
    # All three land on the SAME residual -- the shrinkage floor, not a per-subject difference.
    assert max(after) - min(after) < 0.01, after


def test_alignment_is_label_free_and_covers_a_subject_absent_from_the_fit() -> None:
    """A subject withheld from `fit` still transforms, via the cohort fallback rather than an error."""
    raw, subjects = _cohort()
    fit_mask = subjects != 'C'

    aligner = RawSubjectAligner().fit(raw, subjects, present=fit_mask)
    assert set(aligner.references) == {'A', 'B'}

    aligned = aligner.transform(raw.copy(), subjects)
    assert np.isfinite(aligned).all()
    # The fallback is strictly worse than a subject's own map -- which is exactly what the ablation measures.
    assert _identity_gap(aligned[subjects == 'C']) > _identity_gap(aligned[subjects == 'A'])


def test_calibrating_a_new_brain_recovers_its_own_map() -> None:
    """The zero-shot path: an unlabelled baseline registers a stranger as well as a fitted subject."""
    raw, subjects = _cohort()
    aligner = RawSubjectAligner().fit(raw[subjects != 'C'], subjects[subjects != 'C'])

    stranger, codes = raw[subjects == 'C'], np.array(['C'] * (subjects == 'C').sum())
    uncalibrated = _identity_gap(aligner.transform(stranger.copy(), codes))

    aligner.calibrate_subject(stranger[:100], 'C')
    calibrated = _identity_gap(aligner.transform(stranger.copy(), codes))

    # A 100-trial baseline buys the stranger the same alignment a fitted subject gets.
    fitted = _identity_gap(
        RawSubjectAligner().fit(raw, subjects).transform(raw.copy(), subjects)[subjects == 'C']
    )
    assert calibrated < uncalibrated / 2, (calibrated, uncalibrated)
    assert abs(calibrated - fitted) < 0.01, (calibrated, fitted)


def test_signature_separates_subjects_and_is_stable_across_recordings() -> None:
    """The descriptor identifies a brain, and does not drift between two halves of the same recording."""
    raw, subjects = _cohort()
    aligner = RawSubjectAligner().fit(raw, subjects)
    sigs = {c: aligner.signature_for(c) for c in 'ABC'}

    # Halves of one subject agree far more closely than two different subjects do.
    a = raw[subjects == 'A']
    split = RawSubjectAligner().fit(
        np.concatenate([a[: len(a) // 2], a[len(a) // 2 :]]),
        np.array(['A1'] * (len(a) // 2) + ['A2'] * (len(a) - len(a) // 2)),
    )
    within = np.linalg.norm(split.signatures['A1'] - split.signatures['A2'])
    between = np.linalg.norm(sigs['A'] - sigs['B'])
    assert within < between


def test_signature_survives_a_checkpoint_round_trip() -> None:
    """`state`/`from_state` reproduce the exact maps, so inference matches training."""
    raw, subjects = _cohort()
    aligner = RawSubjectAligner().fit(raw, subjects)
    restored = RawSubjectAligner.from_state(aligner.state)

    assert np.allclose(aligner.signature_for('B'), restored.signature_for('B'))
    assert np.allclose(
        aligner.transform(raw.copy(), subjects), restored.transform(raw.copy(), subjects)
    )


def test_adapter_starts_as_a_no_op_but_can_learn() -> None:
    """Zero-init makes the untrained adapter the identity map, so it cannot destabilise early training."""
    adapter = SubjectAdapter(16, 8, n_channels=N_CH).eval()
    gain, gamma, beta = adapter(torch.randn(4, 16))

    assert torch.allclose(gamma, torch.zeros_like(gamma))
    assert torch.allclose(beta, torch.zeros_like(beta))
    assert gain is not None and torch.allclose(torch.exp(gain), torch.ones_like(gain))

    x = torch.randn(4, 3, N_CH, N_T)
    assert torch.allclose(adapter.apply_spatial(x, gain), x)


def test_adapter_conditions_on_the_signature_not_the_subject_id() -> None:
    """The whole point: two different signatures must produce two different encodings."""
    config = ModelConfig(
        frontend='raw_conformer',
        hidden_dim=32,
        embed_dim=64,
        n_layers=2,
        conformer_filters=8,
        subject_adapter=True,
        spatial_encoding='none',
    )
    model = build_model(config, raw_shape=(N_CH, N_T), n_channels=N_CH, signature_dim=16).eval()
    torch.nn.init.normal_(model.subject_adapter.head.weight, std=0.05)

    batch = {
        'raw': torch.randn(2, 3, N_CH, N_T),
        'pad_mask': torch.ones(2, 3, dtype=torch.bool),
        'presence': torch.ones(2, 3, dtype=torch.bool),
        'subject': torch.zeros(2, dtype=torch.long),  # identical ids ...
        'subject_signature': torch.randn(2, 16),
    }
    first = model.token_hidden(batch)

    batch['subject_signature'] = torch.randn(2, 16)  # ... different signatures
    assert not torch.allclose(first, model.token_hidden(batch), atol=1e-4)


def test_model_without_a_signature_is_unchanged() -> None:
    """Backwards compatibility: no signature means no adapter and the previous forward path."""
    config = ModelConfig(
        frontend='raw_conformer',
        hidden_dim=32,
        embed_dim=64,
        n_layers=2,
        conformer_filters=8,
        subject_adapter=True,
        spatial_encoding='none',
    )
    model = build_model(config, raw_shape=(N_CH, N_T), n_channels=N_CH, signature_dim=0)
    assert model.subject_adapter is None


def test_identity_orthogonality_separates_leaking_from_clean_content() -> None:
    """Content carrying a per-subject offset must score far above content that does not."""
    torch.manual_seed(0)
    n_subj, n = 11, 2048
    signatures = torch.randn(n_subj, 64)
    who = torch.randint(0, n_subj, (n,))
    sig = signatures[who]

    clean = identity_orthogonality(torch.randn(n, 128), sig)
    leaking = identity_orthogonality(torch.randn(n, 128) + torch.randn(n_subj, 128)[who] * 2.0, sig)

    assert clean < 0.1, clean
    assert leaking > 0.5, leaking


def test_identity_orthogonality_cannot_be_gamed_by_collapsing() -> None:
    """Unlike an adversary, shrinking the content space earns no credit -- the term is scale-free."""
    torch.manual_seed(0)
    who = torch.randint(0, 6, (1024,))
    sig = torch.randn(6, 32)[who]
    content = torch.randn(1024, 64) + torch.randn(6, 64)[who] * 2.0

    full = identity_orthogonality(content, sig)
    shrunk = identity_orthogonality(content * 1e-4, sig)
    assert abs(float(full) - float(shrunk)) < 1e-3


def test_objective_reports_the_orthogonality_term() -> None:
    """The penalty reaches the loss and the adapter receives gradient through it."""
    from zte.models.objectives import build_objective

    config = ModelConfig(
        frontend='raw_conformer',
        hidden_dim=32,
        embed_dim=64,
        n_layers=2,
        conformer_filters=8,
        subject_adapter=True,
        factored=True,
        content_dim=32,
        spatial_encoding='none',
        n_subjects=4,
    )
    model = build_model(config, raw_shape=(N_CH, N_T), n_channels=N_CH, signature_dim=16)
    objective = build_objective(
        ObjectiveConfig(name='clip', identity_orthogonality_weight=1.0), model
    )
    objective.attach_text(torch.nn.functional.normalize(torch.randn(4, 16), dim=-1))

    batch = {
        'raw': torch.randn(8, 3, N_CH, N_T),
        'pad_mask': torch.ones(8, 3, dtype=torch.bool),
        'presence': torch.ones(8, 3, dtype=torch.bool),
        'subject': torch.arange(8) % 4,
        'subject_signature': torch.randn(4, 16)[torch.arange(8) % 4],
        'sentence_text_id': torch.arange(8) % 4,
    }
    loss, metrics = objective.compute(model, batch)
    assert 'identity_orth' in metrics

    loss.backward()
    assert model.subject_adapter.head.weight.grad.abs().sum() > 0


def test_binomial_tail_calls_a_handful_of_hits_what_it_is() -> None:
    """Top-1 on 700 queries at 1/700 expects ONE hit; the tail must not treat three as a finding."""
    assert _binom_tail_p(1, 700, 1 / 700) > 0.5
    assert _binom_tail_p(3, 700, 1 / 700) > 0.05
    assert _binom_tail_p(10, 700, 1 / 700) < 1e-5
    assert _binom_tail_p(32, 700, 5 / 700) < 1e-10
    assert _binom_tail_p(0, 700, 1 / 700) == 1.0


def test_rank_percentile_ci_brackets_chance_under_the_null() -> None:
    """A null run must produce a CI containing 0.5, or the headline metric is not calibrated."""
    mean, lo, hi = _bootstrap_ci(np.random.default_rng(0).random(700))
    assert lo < 0.5 < hi
    assert lo < mean < hi
