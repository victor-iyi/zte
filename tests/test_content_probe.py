"""The content probe's positive control, proved by breaking the thing each repair claims to fix."""

from typing import Any, Final

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from zte.evaluation.audit.honesty import cross_subject_decode
from zte.evaluation.audit.probe import detectability_curve, detectability_verdict, plant_linear_target
from zte.evaluation.audit.scoreboard import raw_content_positive_control
from zte.training.metrics import linear_probe, residualise

# Enough permutations for an interval, few enough to keep the suite offline and quick.
_TEST_PERMUTATIONS: Final[int] = 6
"""Permutation draws the tests ask the empirical zero for."""

# The leaky and honest scores must be further apart than any seed wobble, or the mutation proof is not a proof.
_LEAK_MARGIN: Final[float] = 0.15
"""Minimum R2 gap between an ungrouped fold and a grouped one on the sentence-coded fixture."""


@pytest.fixture(scope='module')
def sentence_coded_words() -> tuple[np.ndarray, pd.DataFrame]:
    """Band power that codes which sentence is being read, where word length is a property of the sentence."""
    rng = np.random.default_rng(0)
    n_stimuli, n_words, n_bands, n_channels = 60, 5, 4, 20
    width = n_bands * n_channels
    code = rng.normal(size=(n_stimuli, n_words, width))
    lengths = rng.integers(2, 12, size=(n_stimuli, n_words)).astype(float)

    rows: list[dict[str, Any]] = []
    power: list[np.ndarray] = []
    for subject in range(4):
        offset = rng.normal(size=width) * 2.0
        for stimulus in range(n_stimuli):
            for word in range(n_words):
                rows.append(
                    {
                        'subject': f'S{subject}',
                        'stimulus_key': f'stim{stimulus}',
                        'word_len': float(lengths[stimulus, word]),
                        'log_freq': float(rng.normal()),
                    }
                )
                power.append(code[stimulus, word] + offset + rng.normal(scale=0.35, size=width))

    band_power = np.asarray(power, dtype=np.float32).reshape(-1, n_bands, n_channels)
    return band_power, pd.DataFrame(rows)


@pytest.fixture(scope='module')
def sentence_coded_control(sentence_coded_words: tuple[np.ndarray, pd.DataFrame]) -> dict[str, Any]:
    """The positive control over the sentence-coded fixture, calibration included."""
    band_power, meta = sentence_coded_words
    control = raw_content_positive_control(band_power, meta, n_perm=_TEST_PERMUTATIONS, seed=0)

    assert control is not None
    return control


def _flat(band_power: np.ndarray) -> np.ndarray:
    """The probe's own view of band power: one row per word."""
    return np.asarray(band_power, dtype=np.float32).reshape(len(band_power), -1)


def _probe(x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> float:
    """The probe's R2 as a plain float."""
    score = linear_probe(x, y, task='regression', groups=groups)['score']

    return float(score) if isinstance(score, (int, float)) else float('nan')


def test_ungrouped_folds_score_the_sentence_they_trained_on(
    sentence_coded_words: tuple[np.ndarray, pd.DataFrame],
) -> None:
    """A shuffled k-fold reads word length off band power that only codes sentence identity; a grouped one cannot."""
    band_power, meta = sentence_coded_words
    x, y = _flat(band_power), meta['word_len'].to_numpy(dtype=float)
    groups = meta['stimulus_key'].to_numpy()

    leaky = _probe(x, y)
    honest = _probe(x, y, groups)

    assert leaky > _LEAK_MARGIN
    assert honest < 0.0
    assert leaky - honest > _LEAK_MARGIN


def test_the_pooled_probe_groups_its_folds(
    sentence_coded_words: tuple[np.ndarray, pd.DataFrame], sentence_coded_control: dict[str, Any]
) -> None:
    """The reported pooled R2 is the grouped number, so no sentence is both trained on and scored."""
    band_power, meta = sentence_coded_words
    x, y = _flat(band_power), meta['word_len'].to_numpy(dtype=float)

    leaky = _probe(x, y)
    honest = _probe(x, y, meta['stimulus_key'].to_numpy())

    assert sentence_coded_control['cv_groups'] == 'stimulus_key'
    assert sentence_coded_control['per_target_r2']['word_len'] == pytest.approx(honest, abs=0.02)
    assert leaky - sentence_coded_control['per_target_r2']['word_len'] > _LEAK_MARGIN


def test_the_within_subject_probe_groups_its_folds_too(
    sentence_coded_words: tuple[np.ndarray, pd.DataFrame], sentence_coded_control: dict[str, Any]
) -> None:
    """Residualising the data is only half the question: the within-subject folds are grouped as well."""
    band_power, meta = sentence_coded_words
    x, y = _flat(band_power), meta['word_len'].to_numpy(dtype=float)
    readers = meta['subject'].to_numpy()

    centred_x, centred_y = residualise(x, readers), residualise(y, readers)
    leaky = _probe(centred_x, centred_y)
    honest = _probe(centred_x, centred_y, meta['stimulus_key'].to_numpy())

    assert sentence_coded_control['within_subject_r2']['word_len'] == pytest.approx(honest, abs=0.02)
    assert leaky - sentence_coded_control['within_subject_r2']['word_len'] > _LEAK_MARGIN


def test_the_headline_is_the_within_subject_number_not_the_pooled_maximum() -> None:
    """When the pooled probe scores on who is reading, the honest within-subject number is the one quoted."""
    rng = np.random.default_rng(1)
    width = 60
    rows: list[dict[str, Any]] = []
    power: list[np.ndarray] = []
    for subject in range(4):
        # Each reader gets their own band-power signature and their own word-length range, and nothing links the two.
        signature = rng.normal(size=width) * 3.0
        for stimulus in range(40):
            for _word in range(4):
                rows.append(
                    {
                        'subject': f'S{subject}',
                        'stimulus_key': f'stim{stimulus}',
                        'word_len': float(2 * subject + rng.integers(1, 4)),
                        'log_freq': float(rng.normal()),
                    }
                )
                power.append(signature + rng.normal(scale=0.5, size=width))

    band_power = np.asarray(power, dtype=np.float32).reshape(-1, 6, 10)
    control = raw_content_positive_control(band_power, pd.DataFrame(rows), n_perm=2, calibrate=False)

    assert control is not None
    assert control['raw_content_r2_pooled_best'] > 0.5
    assert control['raw_content_r2_within_best'] < 0.1
    assert control['raw_content_r2_best'] == control['raw_content_r2_within_best']
    assert control['best_is'] == 'within-subject'


def test_the_empirical_zero_is_a_distribution_with_an_interval(
    sentence_coded_words: tuple[np.ndarray, pd.DataFrame], sentence_coded_control: dict[str, Any]
) -> None:
    """The zero is `n_perm` permutations with an interval, not one draw at seed 0."""
    band_power, meta = sentence_coded_words
    x, y = _flat(band_power), meta['word_len'].to_numpy(dtype=float)

    null = sentence_coded_control['permutation_null']['word_len']
    assert null['n_perm'] == _TEST_PERMUTATIONS
    assert len(set(null['scores'])) > 1
    assert null['interval'][0] < null['mean'] < null['interval'][1]
    assert null['p_floor'] == pytest.approx(1.0 / (_TEST_PERMUTATIONS + 1), abs=1e-4)

    single = _probe(x, np.random.default_rng(0).permutation(y), meta['stimulus_key'].to_numpy())
    assert sentence_coded_control['shuffled_target_r2']['word_len'] == null['mean']
    assert sentence_coded_control['shuffled_target_r2']['word_len'] != pytest.approx(single, abs=1e-9)


def test_passes_needs_a_measurement_and_not_only_a_working_estimator() -> None:
    """A machinery check that passes cannot certify a run whose band power was never probed."""
    rng = np.random.default_rng(2)
    n = 400
    word_len = rng.integers(2, 12, n).astype(float)
    meta = pd.DataFrame(
        {
            'word_len': word_len,
            'subject': np.repeat(['A', 'B'], n // 2),
            'stimulus_key': np.tile([f'stim{i}' for i in range(n // 8)], 8),
            'TRT': word_len * 40 + rng.normal(0, 15, n),
            'GD': word_len * 25 + rng.normal(0, 12, n),
            'FFD': word_len * 10 + rng.normal(0, 8, n),
        }
    )

    control = raw_content_positive_control(None, meta, n_perm=2, calibrate=False)

    assert control is not None
    assert control['machinery']['passes'] is True
    assert control['per_target_r2'] == {}
    assert control['passes'] is False
    assert control['passes_clauses']['band_power_probed'] is False
    assert control['passes_clauses']['machinery_reads_word_length'] is True


def test_every_pass_carries_a_band_power_measurement(sentence_coded_control: dict[str, Any]) -> None:
    """A passing control always has something in `per_target_r2`; a blank is never reported as a zero."""
    assert sentence_coded_control['passes'] is True
    assert sentence_coded_control['per_target_r2']
    assert sentence_coded_control['passes_clauses']['band_power_probed'] is True


def test_the_held_out_decoder_searches_the_ridge_penalty() -> None:
    """A no-signal target on a wide design scores about zero, not the -p/n a fixed alpha=1.0 returns."""
    rng = np.random.default_rng(3)
    n_per, width = 40, 200
    subjects = np.repeat(['S0', 'S1', 'S2'], n_per)
    emb = rng.normal(size=(3 * n_per, width))
    meta = pd.DataFrame({'subject': subjects, 'word_len': rng.integers(2, 12, 3 * n_per).astype(float)})

    searched = cross_subject_decode(emb, meta, targets=('word_len',), min_subjects=3)['targets']['word_len']['mean']

    fixed: list[float] = []
    for held_out in np.unique(subjects):
        train, test = subjects != held_out, subjects == held_out
        scaler = StandardScaler().fit(emb[train])
        model = Ridge(alpha=1.0).fit(scaler.transform(emb[train]), meta['word_len'].to_numpy()[train])
        truth = meta['word_len'].to_numpy()[test]
        residual = float(np.sum((truth - model.predict(scaler.transform(emb[test]))) ** 2))
        fixed.append(1.0 - residual / float(np.sum((truth - truth.mean()) ** 2)))

    assert float(np.mean(fixed)) < -0.3
    assert searched > -0.2


def test_recovered_r2_rises_with_the_planted_signal() -> None:
    """The detectability curve is monotone in the injected SNR, so the floor it reports means something."""
    rng = np.random.default_rng(4)
    features = rng.normal(size=(600, 30))
    groups = np.repeat([f'stim{i}' for i in range(60)], 10)

    curve = detectability_curve(features, groups=groups, snrs=(0.0, 0.02, 0.1, 0.4), n_repeats=2, seed=0)
    means = [rung['mean'] for rung in curve['rungs']]
    above_floor = [rung['mean'] for rung in curve['rungs'] if rung['mean'] >= curve['floor_r2']]

    # Two rungs under the floor may land either way round -- that is what "below detectability" means -- so the
    # ordering is asserted within the null rung's own spread, and strictly only where the probe can see.
    assert all(later >= earlier - 0.01 for earlier, later in zip(means, means[1:]))
    assert above_floor == sorted(above_floor)
    assert means[-1] > means[0] + 0.2
    assert curve['established'] is True
    assert curve['null_r2'] < curve['floor_r2'] <= means[-1]
    assert curve['grouped'] is True and curve['n_groups'] == 60


def test_a_planted_target_carries_the_share_it_claims() -> None:
    """The planted target's linearly explainable variance is the SNR that was asked for."""
    rng = np.random.default_rng(5)
    features = rng.normal(size=(600, 10))
    design = np.c_[features, np.ones(len(features))]

    for snr in (0.05, 0.2, 0.6):
        planted = plant_linear_target(features, snr, seed=7)
        coefficients, *_ = np.linalg.lstsq(design, planted, rcond=None)
        residual = float(np.sum((planted - design @ coefficients) ** 2))
        explained = 1.0 - residual / float(np.sum((planted - planted.mean()) ** 2))

        assert planted.mean() == pytest.approx(0.0, abs=1e-9)
        assert planted.std() == pytest.approx(1.0, abs=1e-9)
        assert explained == pytest.approx(snr, abs=0.05)


def test_the_floor_separates_below_detectability_from_absent(sentence_coded_control: dict[str, Any]) -> None:
    """A near-zero band-power score is reported as below the measured floor, not as an unmeasured blank."""
    curve = sentence_coded_control['detectability']
    verdict = sentence_coded_control['detectability_verdict']

    assert curve['established'] is True
    assert verdict['verdict'] == 'below detectability floor'
    assert verdict['observed'] < verdict['floor_r2']
    assert 'no word length this probe can decode linearly' in sentence_coded_control['finding']
    assert detectability_verdict(1.0, curve)['verdict'] == 'above detectability floor'
    assert detectability_verdict(0.001, None)['verdict'] == 'floor not established'
