"""Tests for synthetic generation, .mat parsing, the dataset, missing values."""
# pylint: disable=protected-access

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from zte.config import MissingConfig
from zte.data.dataset import RAW_ARRAY_FILE, ZuCoDataset
from zte.data.features.missing import MissingValueImputer
from zte.data.features.transforms import band_power_from_raw
from zte.data.io.mat_loader import _raw_window
from zte.data.schema import BANDS, ET_MEASURES, N_CHANNELS, SAMPLING_RATE_HZ, band_feature_name
from zte.data.torch_dataset import make_dataloader


def _obj_array(items: list) -> np.ndarray:
    """Builds a 1-D object ndarray (like scipy.io.loadmat returns for a MATLAB cell)."""
    arr = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        arr[i] = x
    return arr


def test_raw_window_handles_ragged_cell_arrays() -> None:
    """Raw EEG that arrives as a ragged cell of per-fixation segments must not crash extraction.

    Regression for the raw-conformer path: `scipy.io.loadmat` returns a multi-fixation `rawEEG` as an
    object ndarray of variable-length `(channels, time)` segments, which a naive `np.asarray(...,
    float32)` cannot convert ("setting an array element with a sequence"). `_raw_window` coerces it to
    the largest segment and pads to the fixed window instead of raising.
    """
    window = 128
    ragged = _obj_array(
        [np.ones((N_CHANNELS, 40), np.float32), np.full((N_CHANNELS, 90), 2.0, np.float32)]
    )
    out = _raw_window(ragged, N_CHANNELS, window)
    assert out.shape == (N_CHANNELS, window)
    # The larger (90-sample) segment is kept; the rest is zero-padded.
    assert (out[:, :90] == 2.0).all() and (out[:, 90:] == 0.0).all()

    # Degenerate inputs never crash and yield an all-zero window (callers use the presence mask).
    for junk in ([], _obj_array([None, 'x']), 5.0, _obj_array([np.zeros(0), np.zeros(0)])):
        z = _raw_window(junk, N_CHANNELS, window)
        assert z.shape == (N_CHANNELS, window) and not z.any()

    # Normal (channels, time) and transposed (time, channels) still work.
    assert _raw_window(np.random.randn(N_CHANNELS, 200), N_CHANNELS, window).shape == (
        N_CHANNELS,
        window,
    )
    assert _raw_window(np.random.randn(200, N_CHANNELS), N_CHANNELS, window).shape == (
        N_CHANNELS,
        window,
    )


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


def test_raw_windows_are_sanitised_at_source(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """`raw_eeg` is NaN-free and per-channel z-scored for every consumer, old bundles included.

    Regression: unlike band power (imputed + FeatureNormalizer-scaled), raw EEG carries NaN (rejected
    samples/channels) and unscaled microvolts. Untreated it made the contrastive loss NaN from step 1,
    and -- because the embedding export reads `raw_eeg` directly rather than via the loader -- produced
    NaN embeddings that surfaced as `LinAlgError: SVD did not converge` during evaluation. The treatment
    therefore belongs at the source, not at any single read site.
    """
    ds = small_dataset
    assert ds.raw_eeg is not None
    assert np.isfinite(ds.raw_eeg).all(), 'built raw_eeg must be finite'

    # The training loader (one consumer) sees finite, z-scored windows.
    loader = make_dataloader(ds.to_torch(representation='raw'), batch_size=8, num_workers=0, seed=0)
    raw = next(iter(loader))['raw']
    assert torch.isfinite(raw).all(), 'raw batch must be finite (no NaN/inf reaches the model)'

    # New bundles keep raw EEG in its own uncompressed .npy so `load` can memory-map it: a ~24 GB tensor
    # must never become resident just to read one window.
    bundle = ds.save(tmp_path / 'bundle')
    assert (bundle / RAW_ARRAY_FILE).is_file(), 'raw EEG must be saved outside the compressed npz'
    with np.load(bundle / 'arrays.npz') as handle:
        assert 'raw_eeg' not in handle, 'raw EEG must not be duplicated inside the npz'

    mapped = ZuCoDataset.load(bundle)
    assert isinstance(mapped.raw_eeg, np.memmap), 'load must memory-map raw EEG'
    assert np.isfinite(mapped.raw_eeg).all()

    # A pre-mmap bundle carries NaN/unscaled windows inside the npz; loading it must still clean them
    # in place of an expensive rebuild.
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    for name in ('meta.json', 'words.pkl', 'sentences.pkl'):
        (legacy / name).write_bytes((bundle / name).read_bytes())
    rng = np.random.default_rng(0)
    stale = np.asarray(ds.raw_eeg, dtype=np.float32) * 5e4  # unscaled microvolts
    stale[rng.random(stale.shape) < 0.1] = np.nan  # scattered rejected samples
    stale[0] = np.nan  # a fully-rejected word
    with np.load(bundle / 'arrays.npz') as handle:
        arrays = dict(handle)
    np.savez_compressed(legacy / 'arrays.npz', raw_eeg=stale, **arrays)

    reloaded = ZuCoDataset.load(legacy)
    assert reloaded.raw_eeg is not None
    assert np.isfinite(reloaded.raw_eeg).all(), 'load must sanitise pre-fix bundles'
    assert abs(float(reloaded.raw_eeg.mean())) < 0.5, 'raw must be per-epoch z-scored'
    assert float(reloaded.raw_eeg.std()) < 5.0, 'raw must be per-epoch z-scored'


def test_legacy_bundle_upgrades_by_streaming_not_loading(tmp_path: Path) -> None:
    """A pre-mmap bundle gains its `.npy` without ever materialising the array.

    Regression: raw bundles are ~24 GB inflated, so on a standard Colab runtime every raw run was killed
    by the OOM reaper while loading -- silently, since the notebook loops ignored exit codes. An `.npz`
    member is itself an `.npy`, so the upgrade is a byte copy of the decompressed stream.
    """
    from zte.data.dataset import _extract_raw_member

    rng = np.random.default_rng(0)
    raw = rng.standard_normal((64, 8, 32), dtype=np.float32)
    np.savez_compressed(tmp_path / 'arrays.npz', raw_eeg=raw, presence=np.ones(64, bool))

    assert _extract_raw_member(tmp_path) is True
    mapped = np.load(tmp_path / RAW_ARRAY_FILE, mmap_mode='r')
    assert isinstance(mapped, np.memmap)
    assert np.array_equal(np.asarray(mapped), raw), 'the streamed copy must be byte-identical'

    # A bundle with no raw member (band power only) is left alone rather than half-upgraded.
    other = tmp_path / 'bp'
    other.mkdir()
    np.savez_compressed(other / 'arrays.npz', features=np.zeros((4, 4), np.float32))
    assert _extract_raw_member(other) is False
    assert not (other / RAW_ARRAY_FILE).exists()


def test_band_power_from_raw_localises_frequency_and_is_probe_sized() -> None:
    """Band power from raw lands a tone in the right band and stays narrow enough to probe.

    Regression: the raw frontend's eval baseline used to be the flattened time-domain window
    (n_channels * time_steps = 36,750 dims), mislabelled 'raw band-power'. Ridge forms a `d x d` Gram
    from it (~10.8 GB) and MemoryErrors, so eval could never finish for a raw config.
    """
    rng = np.random.default_rng(0)
    n, t = 40, 350
    tone = 10.0  # Hz -> falls in a1 (8.5-10.0)
    raw = (rng.standard_normal((n, N_CHANNELS, t)) * 0.1).astype(np.float32)
    raw += (3.0 * np.sin(2 * np.pi * tone * (np.arange(t) / SAMPLING_RATE_HZ))).astype(np.float32)

    bp = band_power_from_raw(raw)
    assert bp.shape == (n, N_CHANNELS * len(BANDS)), (
        'must match the band-power representation width'
    )
    assert np.isfinite(bp).all()
    per_band = bp.reshape(n, N_CHANNELS, len(BANDS)).mean(axis=(0, 1))
    assert BANDS[int(np.argmax(per_band))] == 'a1', 'a 10 Hz tone must dominate the a1 band'

    # A window too short to resolve the low bands reports zero power, never NaN from an empty mean.
    short = band_power_from_raw(rng.standard_normal((3, N_CHANNELS, 32)).astype(np.float32))
    assert np.isfinite(short).all()
    assert float(short.reshape(3, N_CHANNELS, len(BANDS))[..., 0].sum()) == 0.0


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


def test_load_into_populates_the_given_instance(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """`load(into=ds)` must populate that instance, even though a fresh dataset is falsy.

    Regression: `ds = into or cls(config)` discarded a freshly-constructed `into` (empty `__len__` is
    falsy) and populated a throwaway, so `build()`'s cache hit (`self.load(hit, into=self); return self`)
    returned an empty `self` and the next `analyze()` raised `KeyError: 'sentence_uid'`.
    """
    from zte.config import DatasetConfig

    bundle = small_dataset.save(tmp_path / 'bundle')
    target = ZuCoDataset(DatasetConfig())
    assert len(target) == 0  # a fresh dataset is falsy — the trigger for the bug
    returned = target.load(bundle, into=target)
    assert returned is target  # the caller's instance, not a throwaway
    assert 'sentence_uid' in target.words.columns
    assert len(target.words) == len(small_dataset.words)


def test_save_reload_roundtrip(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """A saved bundle reloads with identical arrays and feature names."""
    bundle = small_dataset.save(tmp_path / 'bundle')
    reloaded = ZuCoDataset.load(bundle)
    assert reloaded.features.shape == small_dataset.features.shape
    assert np.allclose(reloaded.features, small_dataset.features)
    assert reloaded.feature_names == small_dataset.feature_names
    assert len(reloaded.words) == len(small_dataset.words)


def test_stale_bundle_backfills_derived_columns(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """A cached bundle predating a derived column is repaired on load, not left to crash later.

    Regression: an old raw bundle on Drive lacked `sentence_uid` (added in `_process`), so a cache hit
    loaded a words table without it and `analyze()` raised KeyError after skipping the rebuild.
    """
    import pandas as pd

    bundle = small_dataset.save(tmp_path / 'bundle')
    derived = [
        'sentence_uid',
        'word_len',
        'log_freq',
        'is_omitted',
        'rel_pos',
        'category',
        'category_scheme',
        'length_band',
        'stimulus_key',
    ]
    words = pd.read_pickle(bundle / 'words.pkl')
    stripped = words.drop(columns=[c for c in derived if c in words.columns])
    assert 'sentence_uid' not in stripped.columns
    stripped.to_pickle(bundle / 'words.pkl')

    reloaded = ZuCoDataset.load(bundle)
    assert 'sentence_uid' in reloaded.words.columns
    assert 'category' in reloaded.words.columns
    # The exact call that crashed on Colab must now succeed.
    summary = reloaded.analyze()
    assert summary['n_sentences'] > 0
    assert reloaded.split()['train'].size > 0


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
