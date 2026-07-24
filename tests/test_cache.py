"""Tests for the layered, two-level dataset cache: build once, reuse everywhere, forever."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zte.config import DatasetConfig, MissingConfig
from zte.data.cache import REMOTE_ENV_VAR, BundleStore
from zte.data.dataset import ZuCoDataset


def _entry(directory: Path, payload: str = '{}') -> Path:
    """Creates a minimal cache entry (a directory carrying a `meta.json`)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'meta.json').write_text(payload, encoding='utf-8')
    return directory


def test_store_prefers_local_then_falls_back_to_remote(tmp_path: Path) -> None:
    """Test that lookups hit the fast local copy first and the persistent store second."""
    store = BundleStore(local=tmp_path / 'local', remote=tmp_path / 'drive')
    assert store.find('k') is None

    _entry(tmp_path / 'drive' / 'k', '{"from": "remote"}')
    found = store.find('k')
    assert found is not None
    assert found == tmp_path / 'local' / 'k'  # staged down, so later reads are local
    assert json.loads((found / 'meta.json').read_text(encoding='utf-8'))['from'] == 'remote'

    assert store.find('k') == tmp_path / 'local' / 'k'


def test_store_publishes_immediately_and_treats_entries_as_immutable(tmp_path: Path) -> None:
    """Test that a built entry reaches the persistent store, and an existing one is left alone."""
    store = BundleStore(local=tmp_path / 'local', remote=tmp_path / 'drive')
    _entry(store.reserve('k'), '{"v": 1}')
    store.publish('k')
    assert (tmp_path / 'drive' / 'k' / 'meta.json').read_text(encoding='utf-8') == '{"v": 1}'

    # Content-addressed entries never change, so publishing again must not rewrite the remote copy.
    (tmp_path / 'local' / 'k' / 'meta.json').write_text('{"v": 2}', encoding='utf-8')
    store.publish('k')
    assert (tmp_path / 'drive' / 'k' / 'meta.json').read_text(encoding='utf-8') == '{"v": 1}'


def test_store_without_a_remote_is_a_plain_local_cache(tmp_path: Path) -> None:
    """Test that omitting the persistent store degrades to local-only behaviour."""
    store = BundleStore(local=tmp_path / 'local', remote=None)
    _entry(store.reserve('k'))
    store.publish('k')  # must not raise
    assert store.find('k') == tmp_path / 'local' / 'k'


def test_store_reads_the_remote_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that `ZTE_CACHE_REMOTE` configures the persistent store for every command."""
    monkeypatch.setenv(REMOTE_ENV_VAR, str(tmp_path / 'drive'))
    assert BundleStore.create(tmp_path / 'local').remote == tmp_path / 'drive'

    monkeypatch.delenv(REMOTE_ENV_VAR)
    assert BundleStore.create(tmp_path / 'local').remote is None
    # An explicit argument always wins over the environment.
    monkeypatch.setenv(REMOTE_ENV_VAR, str(tmp_path / 'env'))
    assert BundleStore.create(tmp_path / 'local', tmp_path / 'x').remote == tmp_path / 'x'


def test_extract_key_ignores_processing_settings() -> None:
    """Test that the `.mat` extraction is shared by configs that differ only in processing.

    This is what makes a new dataset config cheap: normalisation, imputation, eye-tracking and length
    filters all re-derive from a cached extraction instead of re-parsing every `.mat` file.
    """
    base = DatasetConfig(representation='band_power')
    processing_only = [
        dataclasses.replace(base, normalize='zscore_subject'),
        dataclasses.replace(base, include_eye_tracking=not base.include_eye_tracking),
        dataclasses.replace(base, min_words=5),
        dataclasses.replace(base, max_words=20),
        dataclasses.replace(base, include_omitted=False),
        dataclasses.replace(base, missing=MissingConfig(method='median')),
    ]
    for variant in processing_only:
        assert ZuCoDataset(variant)._extract_key() == ZuCoDataset(base)._extract_key()
        assert ZuCoDataset(variant)._cache_key() != ZuCoDataset(base)._cache_key()


def test_extract_key_separates_genuinely_different_extractions() -> None:
    """Test that settings the `.mat` parse depends on do produce distinct extractions."""
    base = DatasetConfig(representation='band_power')
    for variant in (
        dataclasses.replace(base, representation='raw'),
        dataclasses.replace(base, raw_window=350),
        dataclasses.replace(base, tasks=('SR',)),
        dataclasses.replace(base, subjects=('ZAB',)),
        dataclasses.replace(base, bands=('t1', 't2')),
        dataclasses.replace(base, band_power_measures=('FFD',)),
    ):
        assert ZuCoDataset(variant)._extract_key() != ZuCoDataset(base)._extract_key()


def test_cache_location_settings_never_change_the_key(tmp_path: Path) -> None:
    """Test that where (and whether) we cache cannot invalidate an existing bundle.

    Adding these fields must not orphan the bundles already built on Drive.
    """
    base = DatasetConfig()
    for variant in (
        dataclasses.replace(base, cache_dir=str(tmp_path)),
        dataclasses.replace(base, cache_remote=str(tmp_path / 'drive')),
        dataclasses.replace(base, cache_extracts=False),
        dataclasses.replace(base, montage_csv='res/montage_gsn105.csv'),
    ):
        assert ZuCoDataset(variant)._cache_key() == ZuCoDataset(base)._cache_key()
        assert ZuCoDataset(variant)._extract_key() == ZuCoDataset(base)._extract_key()


def test_extract_round_trip_keeps_the_requested_config(tmp_path: Path) -> None:
    """Test that loading an extraction does not adopt the config that happened to build it.

    An extraction exists to serve a *different* config, so `_load_extract` must leave `config` alone
    (unlike `load`, which restores a bundle wholesale).
    """
    built = ZuCoDataset(DatasetConfig(normalize='zscore_channel', representation='band_power'))
    built.words = pd.DataFrame({'word': ['a', 'b'], 'subject': ['ZAB', 'ZAB']})
    built.sentences = pd.DataFrame({'sentence_idx': [0]})
    built.band_power_raw = np.ones((2, 8, 105), dtype=np.float32)
    built.bp_feature_names = ['f']
    built._save_extract(tmp_path / 'extract')

    wanted = DatasetConfig(normalize='riemannian', representation='band_power', min_words=4)
    restored = ZuCoDataset(wanted)
    restored._load_extract(tmp_path / 'extract')

    assert restored.config.normalize == 'riemannian'  # not the extraction's 'zscore_channel'
    assert restored.config.min_words == 4
    assert restored.band_power_raw is not None and restored.band_power_raw.shape == (2, 8, 105)
    # The processed, config-specific state is deliberately absent -- `_process` re-derives it.
    assert restored.features is None
    assert restored.presence is None
    assert restored.normalizer is None
