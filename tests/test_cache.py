"""Tests for the layered, two-level dataset cache: build once, reuse everywhere, forever."""

from __future__ import annotations

import argparse
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


def _prepare_args(tmp_path: Path, config: str, check: bool = False) -> argparse.Namespace:
    """Builds the `zte-prepare --configs` namespace against a scratch local/drive pair."""
    return argparse.Namespace(
        configs=[config],
        synthetic=False,
        synthetic_out='res/data/synthetic_zuco',
        cache_dir=str(tmp_path / 'local'),
        cache_remote=str(tmp_path / 'drive'),
        no_extract_cache=False,
        check=check,
    )


def test_has_reports_the_layer_without_staging(tmp_path: Path) -> None:
    """Test that presence checks never copy, so gating a session on them is free."""
    store = BundleStore(local=tmp_path / 'local', remote=tmp_path / 'drive')
    _entry(tmp_path / 'drive' / 'k')

    assert store.has('k') == 'persistent'
    assert not (tmp_path / 'local' / 'k').exists()  # the whole point: nothing was pulled down

    _entry(tmp_path / 'local' / 'k')
    assert store.has('k') == 'local'
    assert store.has('missing') is None


def test_prepare_skips_everything_when_the_persistent_store_is_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test the Colab case: local cache wiped, Drive warm -> no rebuild and no raw-data resolution.

    Guards the regression where a fresh runtime re-prepared every dataset because the local cache, and the
    sentinel that used to gate it, live on a disk Colab throws away.
    """
    from zte.cli import prepare as prepare_cli
    from zte.cli.support.sources import PENDING_ROOT
    from zte.config import ZTEConfig

    config = 'experiments/flagship/zte_raw_aligned.yaml'
    cfg = ZTEConfig.from_yaml(config).dataset
    cfg.root, cfg.cache_dir, cfg.cache_remote = PENDING_ROOT, tmp_path / 'local', None
    _entry(tmp_path / 'drive' / ZuCoDataset(cfg)._cache_key())

    # Resolving the raw root is the expensive step; a warm store must never reach it.
    def _boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError('touched the raw data despite a warm persistent store')

    monkeypatch.setattr(prepare_cli, '_resolve_build_root', _boom)
    monkeypatch.setattr(prepare_cli.ZuCoDataset, 'build', _boom)

    prepare_cli._prepare_configs(_prepare_args(tmp_path, config))
    assert not (tmp_path / 'local').exists()  # nothing staged either


def test_prepare_check_never_builds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that `--check` reports a cold cache without building or resolving anything."""
    from zte.cli import prepare as prepare_cli

    def _boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError('--check must not build')

    monkeypatch.setattr(prepare_cli, '_resolve_build_root', _boom)
    monkeypatch.setattr(prepare_cli.ZuCoDataset, 'build', _boom)

    prepare_cli._prepare_configs(
        _prepare_args(tmp_path, 'experiments/flagship/zte_raw_aligned.yaml', check=True)
    )


def test_artifacts_survive_a_wiped_local_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that frozen encoder matrices layer onto the persistent store like bundles do.

    The BERT meaning matrix and the E5 sentence embeddings cost minutes to build; cached only on the
    Colab VM's disk they were rebuilt every session.
    """
    from zte.data.cache import fetch_artifact, publish_artifact

    monkeypatch.setenv(REMOTE_ENV_VAR, str(tmp_path / 'drive'))
    local = tmp_path / 'local' / 'text_deadbeef.npy'
    local.parent.mkdir(parents=True)
    local.write_bytes(b'matrix')

    publish_artifact(local)
    assert (tmp_path / 'drive' / '_artifacts' / 'text_deadbeef.npy').is_file()

    local.unlink()  # the runtime reset
    assert fetch_artifact(local) is True
    assert local.read_bytes() == b'matrix'


def test_artifacts_are_a_no_op_without_a_persistent_store(tmp_path: Path) -> None:
    """Test that artifact layering stays inert when no remote is configured."""
    from zte.data.cache import fetch_artifact, publish_artifact

    missing = tmp_path / 'text_x.npy'
    publish_artifact(missing)  # must not raise
    assert fetch_artifact(missing) is False


def test_prepare_keys_are_independent_of_the_data_root(tmp_path: Path) -> None:
    """Test the assumption the deferred resolution rests on: `root` never reaches the cache key.

    If it did, keying with a placeholder would miss every bundle and re-prepare the whole project.
    """
    from zte.cli.support.sources import PENDING_ROOT
    from zte.config import ZTEConfig

    keys = set()
    for root in ('/local/zuco', '/gdrive/My Drive/ZuCo', PENDING_ROOT, None):
        cfg = ZTEConfig.from_yaml('experiments/flagship/zte_raw_aligned.yaml').dataset
        cfg.root, cfg.cache_dir = root, tmp_path
        keys.add(ZuCoDataset(cfg)._cache_key())
    assert len(keys) == 1, keys

    # Synthetic must still key separately, or a smoke run would poison the real bundle.
    cfg = ZTEConfig.from_yaml('experiments/flagship/zte_raw_aligned.yaml').dataset
    cfg.root, cfg.cache_dir = 'res/data/synthetic_zuco', tmp_path
    assert ZuCoDataset(cfg)._cache_key() not in keys


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
