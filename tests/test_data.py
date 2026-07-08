"""Tests for synthetic generation, .mat parsing, the dataset, missing values."""
# pylint: disable=protected-access

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from zte.config import MissingConfig
from zte.data.dataset import ZuCoDataset
from zte.data.missing import MissingValueImputer
from zte.data.schema import BANDS, ET_MEASURES, N_CHANNELS, band_feature_name


def test_synthetic_matches_zuco_schema(synthetic_dir: Path) -> None:
    """The synthetic files load via scipy and expose ZuCo's struct fields."""
    files = sorted(synthetic_dir.glob('*.mat'))
    assert files, 'no synthetic files generated'
    mat = loadmat(files[0], squeeze_me=True, struct_as_record=False)
    sentences = np.atleast_1d(mat['sentenceData'])
    sent = sentences[0]
    assert isinstance(sent.content, str) and sent.content
    words = np.atleast_1d(sent.word)
    fields = set(words[0]._fieldnames)
    for measure in ET_MEASURES:
        assert measure in fields
    assert band_feature_name('TRT', 't1') in fields
    # mean_<band> sentence-level fields exist.
    assert all(f'mean_{b}' in set(sent._fieldnames) for b in BANDS)


def test_omitted_words_are_empty(synthetic_dir: Path) -> None:
    """At least some words are omitted and surface as empty arrays."""
    mat = loadmat(sorted(synthetic_dir.glob('*.mat'))[0], squeeze_me=True, struct_as_record=False)
    words = [w for s in np.atleast_1d(mat['sentenceData']) for w in np.atleast_1d(s.word)]
    empties = [np.asarray(w.TRT_t1).size == 0 for w in words]
    assert any(empties), 'expected some omitted (empty) words'


def test_dataset_builds_with_aligned_shapes(small_dataset: ZuCoDataset) -> None:
    """Band-power tensor, features and presence are row-aligned with the table."""
    ds = small_dataset
    n = len(ds.words)
    assert n > 0
    assert ds.band_power_raw.shape == (n, len(BANDS), N_CHANNELS)
    # Features are flattened band power plus the appended eye-tracking scalars
    # (included by default); the toggle governs how many extra columns appear.
    n_et = len(ds.config.eye_tracking_measures) if ds.config.include_eye_tracking else 0
    assert ds.features.shape == (n, len(BANDS) * N_CHANNELS + n_et)
    assert ds.presence.shape == (n,)
    assert ds.raw_eeg.shape[0] == n and ds.raw_eeg.shape[1] == N_CHANNELS
    assert not np.isnan(ds.features).any(), 'features must be finite after imputation'


def test_presence_matches_omission(small_dataset: ZuCoDataset) -> None:
    """Presence mask is the logical complement of the omission flag."""
    omitted = small_dataset.words['is_omitted'].to_numpy().astype(bool)
    assert np.array_equal(small_dataset.presence, ~omitted)


def test_analyze_and_feature_selection(small_dataset: ZuCoDataset) -> None:
    """Analysis returns sane counts and feature selection ranks channels."""
    summary = small_dataset.analyze()
    assert summary['n_words'] == len(small_dataset.words)
    assert 0.0 <= summary['omission_rate_overall'] <= 1.0
    result = small_dataset.select_features(target='log_freq', method='f_score', k=16)
    assert len(result.indices) == 16
    assert result.scores.shape[0] == small_dataset.features.shape[1]


def test_splits_are_leakage_aware(small_dataset: ZuCoDataset) -> None:
    """LOSO and by-sentence splits partition rows without overlap."""
    loso = small_dataset.split('by_subject_loso', holdout_subject='ZDM')
    assert len(np.intersect1d(loso['train'], loso['val'])) == 0
    val_subjects = set(small_dataset.words.iloc[loso['val']]['subject'])
    assert val_subjects == {'ZDM'}

    by_sent = small_dataset.split('by_sentence', val_fraction=0.3, seed=1)
    train_uids = set(small_dataset.words.iloc[by_sent['train']]['sentence_uid'])
    val_uids = set(small_dataset.words.iloc[by_sent['val']]['sentence_uid'])
    assert train_uids.isdisjoint(val_uids), 'sentences must not span splits'


def test_save_reload_roundtrip(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """A saved bundle reloads with identical arrays and feature names."""
    bundle = small_dataset.save(tmp_path / 'bundle')
    reloaded = ZuCoDataset.load(bundle)
    assert reloaded.features.shape == small_dataset.features.shape
    assert np.allclose(reloaded.features, small_dataset.features)
    assert reloaded.feature_names == small_dataset.feature_names
    assert len(reloaded.words) == len(small_dataset.words)


@pytest.mark.parametrize(
    'method',
    ['zero', 'row_mean', 'col_mean', 'global_mean', 'median', 'knn', 'iterative', 'mask_only'],
)
def test_missing_methods_fill_all_nans(method: str) -> None:
    """Every imputation method returns a finite matrix and a correct mask."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 12)).astype(np.float32)
    x[5:9] = np.nan  # fully-omitted rows
    x[0, 3] = np.nan  # scattered missing
    imputer = MissingValueImputer(MissingConfig(method=method, knn_neighbors=3))
    filled, presence = imputer.fit_transform(x)
    assert not np.isnan(filled).any()
    assert presence.shape == (40,)
    assert presence.sum() == 36  # rows 5..8 are all-NaN -> absent


def test_ffill_is_group_aware() -> None:
    """Forward-fill never carries values across group (sentence) boundaries."""
    x = np.array([[1.0], [np.nan], [np.nan], [5.0]], dtype=np.float32)
    groups = np.array([0, 0, 1, 1])
    imputer = MissingValueImputer(MissingConfig(method='ffill'))
    filled, _ = imputer.fit_transform(x, group_ids=groups)
    assert filled[1, 0] == 1.0  # filled within group 0
    assert filled[2, 0] == 5.0  # group 1 back-filled from its own value, not group 0
