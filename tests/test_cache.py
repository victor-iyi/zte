"""Tests for the layered, two-level dataset cache: build once, reuse everywhere, forever."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zte.config import DatasetConfig, MissingConfig
from zte.data import cache
from zte.data.cache import REMOTE_ENV_VAR, REQUIRED_ENTRY_FILES, BundleStore
from zte.data.dataset import ZuCoDataset


def _entry(directory: Path, payload: str = '{}') -> Path:
    """Creates a minimal COMPLETE cache entry: every required file present, `meta.json` carrying the payload."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_ENTRY_FILES:
        (directory / name).write_text(payload if name == 'meta.json' else 'x', encoding='utf-8')
    return directory


def _torn_entry(directory: Path, payload: str = '{}') -> Path:
    """Creates a torn cache entry: `meta.json` landed, the pickles it describes did not."""
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
    """Presence checks never copy, so gating a session on them is free."""
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
    """`--check` reports a cold cache without building or resolving anything."""
    from zte.cli import prepare as prepare_cli

    def _boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError('--check must not build')

    monkeypatch.setattr(prepare_cli, '_resolve_build_root', _boom)
    monkeypatch.setattr(prepare_cli.ZuCoDataset, 'build', _boom)

    prepare_cli._prepare_configs(_prepare_args(tmp_path, 'experiments/flagship/zte_raw_aligned.yaml', check=True))


def test_artifacts_survive_a_wiped_local_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Frozen encoder matrices layer onto the persistent store like bundles do.

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
    """Artifact layering stays inert when no remote is configured."""
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
    """Lookups hit the fast local copy first and the persistent store second."""
    store = BundleStore(local=tmp_path / 'local', remote=tmp_path / 'drive')
    assert store.find('k') is None

    _entry(tmp_path / 'drive' / 'k', '{"from": "remote"}')
    found = store.find('k')
    assert found is not None
    assert found == tmp_path / 'local' / 'k'  # staged down, so later reads are local
    assert json.loads((found / 'meta.json').read_text(encoding='utf-8'))['from'] == 'remote'

    assert store.find('k') == tmp_path / 'local' / 'k'


def test_store_publishes_immediately_and_treats_entries_as_immutable(tmp_path: Path) -> None:
    """A built entry reaches the persistent store, and an existing one is left alone."""
    store = BundleStore(local=tmp_path / 'local', remote=tmp_path / 'drive')
    _entry(store.reserve('k'), '{"v": 1}')
    store.publish('k')
    assert (tmp_path / 'drive' / 'k' / 'meta.json').read_text(encoding='utf-8') == '{"v": 1}'

    # Content-addressed entries never change, so publishing again must not rewrite the remote copy.
    (tmp_path / 'local' / 'k' / 'meta.json').write_text('{"v": 2}', encoding='utf-8')
    store.publish('k')
    assert (tmp_path / 'drive' / 'k' / 'meta.json').read_text(encoding='utf-8') == '{"v": 1}'


def test_store_without_a_remote_is_a_plain_local_cache(tmp_path: Path) -> None:
    """Omitting the persistent store degrades to local-only behaviour."""
    store = BundleStore(local=tmp_path / 'local', remote=None)
    _entry(store.reserve('k'))
    store.publish('k')  # must not raise
    assert store.find('k') == tmp_path / 'local' / 'k'


def test_store_reads_the_remote_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`ZTE_CACHE_REMOTE` configures the persistent store for every command."""
    monkeypatch.setenv(REMOTE_ENV_VAR, str(tmp_path / 'drive'))
    assert BundleStore.create(tmp_path / 'local').remote == tmp_path / 'drive'

    monkeypatch.delenv(REMOTE_ENV_VAR)
    assert BundleStore.create(tmp_path / 'local').remote is None
    # An explicit argument always wins over the environment.
    monkeypatch.setenv(REMOTE_ENV_VAR, str(tmp_path / 'env'))
    assert BundleStore.create(tmp_path / 'local', tmp_path / 'x').remote == tmp_path / 'x'


def test_extract_key_ignores_processing_settings() -> None:
    """The `.mat` extraction is shared by configs that differ only in processing.

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
    """Settings the `.mat` parse depends on do produce distinct extractions."""
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
    """Where (and whether) we cache cannot invalidate an existing bundle.

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
    """Loading an extraction does not adopt the config that happened to build it.

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


# --------------------------------------------------------------------------- #
# torn entries: an interrupted copy must cost a rebuild, never the run
# --------------------------------------------------------------------------- #
def test_a_torn_local_entry_is_a_miss_and_is_cleared(tmp_path: Path) -> None:
    """A local directory with `meta.json` but no pickles is a torn copy: refused and removed, never loaded."""
    store = BundleStore(local=tmp_path / 'local', remote=None)
    torn = _torn_entry(tmp_path / 'local' / 'k')

    assert store.find('k') is None
    assert not torn.exists()
    assert store.has('k') is None


def test_a_torn_persistent_entry_is_refused_and_never_staged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A torn entry on the persistent store is reported loudly and treated as a miss, not copied down."""
    import logging

    store = BundleStore(local=tmp_path / 'local', remote=tmp_path / 'drive')
    _torn_entry(tmp_path / 'drive' / 'k')

    with caplog.at_level(logging.WARNING, logger='zte.data.cache'):
        assert store.find('k') is None

    assert not (tmp_path / 'local' / 'k').exists()
    assert any('torn publish' in record.message for record in caplog.records)
    assert store.has('k') is None


def test_publish_repairs_an_incomplete_persistent_entry(tmp_path: Path) -> None:
    """A torn remote entry is completed by the next publish rather than frozen forever behind `meta.json`."""
    store = BundleStore(local=tmp_path / 'local', remote=tmp_path / 'drive')
    _entry(tmp_path / 'local' / 'k', payload='{"v": 1}')
    _torn_entry(tmp_path / 'drive' / 'k', payload='{"v": 1}')

    store.publish('k')

    for name in REQUIRED_ENTRY_FILES:
        assert (tmp_path / 'drive' / 'k' / name).is_file(), name
    assert store.has('k') == 'local'


def test_build_falls_back_past_an_unreadable_bundle(synthetic_dir: Path, tmp_path: Path) -> None:
    """A complete-but-corrupt cache entry is discarded and rebuilt, checkpoint-style, instead of crashing."""
    config = DatasetConfig(
        root=str(synthetic_dir),
        tasks=('SR',),
        representation='band_power',
        cache_dir=str(tmp_path / 'cache'),
    )
    first = ZuCoDataset(dataclasses.replace(config)).build(show_progress=False)
    n_sentences = len(first.sentences)

    entries = [p for p in (tmp_path / 'cache').iterdir() if p.is_dir() and not p.name.startswith('_')]
    assert len(entries) == 1
    (entries[0] / 'sentences.pkl').write_bytes(b'not a pickle')

    rebuilt = ZuCoDataset(dataclasses.replace(config)).build(show_progress=False)

    assert len(rebuilt.sentences) == n_sentences
    reread = ZuCoDataset(dataclasses.replace(config)).build(show_progress=False)
    assert len(reread.sentences) == n_sentences


# ---- Disk budget: a twelve-fold sweep must not fill the volume ---- #


def _seed_entry(root: Path, key: str, payload: bytes = b'x') -> Path:
    """A complete cache entry, so the store counts it as present."""
    entry = root / key
    entry.mkdir(parents=True, exist_ok=True)
    for name in cache.REQUIRED_ENTRY_FILES:
        (entry / name).write_bytes(payload)
    return entry


def test_staging_evicts_the_least_recently_used_bundle_rather_than_filling_the_disk(tmp_path: Path) -> None:
    """A ZuCo raw bundle is 11 GB for one task and 24 for SR+NR, and a campaign needs four task sets.

    Note:
        Nothing evicted them, so a long sweep filled the volume and every later run died on a full disk rather
        than on a bad number.
    """
    remote = tmp_path / 'remote'
    store = cache.BundleStore(local=tmp_path / 'local', remote=remote)
    for key in ('oldest', 'middle', 'newest'):
        _seed_entry(store.local, key)
        _seed_entry(remote, key)
    for age, key in enumerate(('newest', 'middle', 'oldest')):
        stamp = time.time() - (age + 1) * 100
        os.utime(store.local / key / 'meta.json', (stamp, stamp))

    assert [p.name for p in store.staged()] == ['oldest', 'middle', 'newest']

    # Ask for more than the volume can spare, so the budget has to bite.
    need = shutil.disk_usage(store.local).free / 1e9 + 1.0
    removed = store.make_room(need, keep='newest')

    assert removed == ['oldest', 'middle'], 'eviction must run least-recently-used first'
    assert [p.name for p in store.staged()] == ['newest'], 'the entry in use is never evicted'


def test_an_entry_the_store_cannot_restage_is_never_evicted(tmp_path: Path) -> None:
    """A local entry is rebuildable in principle, but rebuilding a ZuCo bundle is a multi-GB extraction."""
    remote = tmp_path / 'remote'
    store = cache.BundleStore(local=tmp_path / 'local', remote=remote)
    _seed_entry(store.local, 'only_local')
    _seed_entry(store.local, 'also_remote')
    _seed_entry(remote, 'also_remote')

    removed = store.make_room(shutil.disk_usage(store.local).free / 1e9 + 1.0)

    assert removed == ['also_remote']
    assert (store.local / 'only_local').is_dir(), 'an entry with no complete remote copy must survive'


def test_the_headroom_is_configurable_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine with a different disk needs a different budget, and an unreadable value must not crash a run."""
    monkeypatch.setenv(cache.FREE_SPACE_ENV_VAR, '3.5')
    assert cache.min_free_gb() == pytest.approx(3.5)

    monkeypatch.setenv(cache.FREE_SPACE_ENV_VAR, 'not-a-number')
    assert cache.min_free_gb() == pytest.approx(cache.MIN_FREE_GB)

    monkeypatch.delenv(cache.FREE_SPACE_ENV_VAR)
    assert cache.min_free_gb() == pytest.approx(cache.MIN_FREE_GB)


def test_staging_a_bundle_makes_room_before_it_copies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The eviction has to run on the staging path, not merely exist -- that is where the disk fills."""
    remote = tmp_path / 'remote'
    store = cache.BundleStore(local=tmp_path / 'local', remote=remote)
    _seed_entry(remote, 'wanted')

    asked: list[tuple[float, str | None]] = []
    original = cache.BundleStore.make_room

    def spy(self: cache.BundleStore, need_gb: float, *, keep: str | None = None) -> list[str]:
        asked.append((need_gb, keep))
        return original(self, need_gb, keep=keep)

    monkeypatch.setattr(cache.BundleStore, 'make_room', spy)
    staged = store.find('wanted')

    assert staged is not None and staged.is_dir(), 'the bundle must still be staged'
    assert asked, 'staging must ask for room before copying gigabytes onto the volume'
    assert asked[0][1] == 'wanted', 'the entry being staged must never be the one evicted'


# ---- Choosing the volume the bundle is staged on ---- #


def test_the_bundle_cache_moves_to_a_roomier_volume_when_one_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle is 11-24 GB and the checkout sits on the boot volume, which is not always the largest disk here."""
    roomy = tmp_path / 'scratch'
    roomy.mkdir()
    default = tmp_path / 'res' / 'cache' / 'prepared'

    monkeypatch.setattr(cache, 'SCRATCH_CANDIDATES', (str(roomy),))
    monkeypatch.setattr(
        cache, '_free_gb', lambda path: 400.0 if roomy in Path(path).parents or Path(path) == roomy else 30.0
    )

    chosen = cache.scratch_root(default)

    assert chosen == roomy / 'zte-cache'
    assert chosen.is_dir(), 'only the chosen directory is created'


def test_a_volume_that_is_barely_roomier_is_not_worth_moving_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging is a multi-GB copy, so a few spare gigabytes elsewhere does not justify paying for it."""
    other = tmp_path / 'other'
    other.mkdir()
    default = tmp_path / 'res' / 'cache' / 'prepared'

    monkeypatch.setattr(cache, 'SCRATCH_CANDIDATES', (str(other),))
    monkeypatch.setattr(cache, '_free_gb', lambda path: 35.0 if Path(path) == other else 30.0)

    assert cache.scratch_root(default) == default


def test_probing_candidates_creates_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A laptop must not collect `zte-cache` directories on every volume the scan happens to look at."""
    looked_at = tmp_path / 'looked-at'
    looked_at.mkdir()
    default = tmp_path / 'res' / 'cache' / 'prepared'

    monkeypatch.setattr(cache, 'SCRATCH_CANDIDATES', (str(looked_at), '/definitely/not/here'))
    monkeypatch.setattr(cache, '_free_gb', lambda path: 30.0)

    assert cache.scratch_root(default) == default
    assert list(looked_at.iterdir()) == [], 'a candidate that lost must be left untouched'


def test_the_scratch_directory_can_be_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine whose layout the scan cannot guess needs one env var, not a code change."""
    pinned = tmp_path / 'pinned'
    monkeypatch.setenv(cache.SCRATCH_ENV_VAR, str(pinned))

    chosen = cache.scratch_root(tmp_path / 'res' / 'cache' / 'prepared')

    assert chosen == pinned
    assert chosen.is_dir()


def test_the_headroom_never_claims_most_of_a_small_scratch_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Colab scratch disk is fixed and can be far smaller than the boot volume it was sized against.

    Note:
        A flat reserve big enough for a 200 GB disk would make a 40 GB one unusable rather than safe, so the
        headroom is capped at a share of the volume it is actually reserving on.
    """
    monkeypatch.delenv(cache.FREE_SPACE_ENV_VAR, raising=False)
    monkeypatch.setattr(cache, '_total_gb', lambda path: 40.0)

    scaled = cache.min_free_gb(tmp_path)

    assert scaled == pytest.approx(40.0 * cache.MAX_HEADROOM_SHARE)
    assert scaled < cache.MIN_FREE_GB, 'a small volume must reserve less than the flat figure'

    monkeypatch.setattr(cache, '_total_gb', lambda path: 400.0)
    assert cache.min_free_gb(tmp_path) == pytest.approx(cache.MIN_FREE_GB), 'a large volume keeps the flat figure'
    assert cache.min_free_gb() == pytest.approx(cache.MIN_FREE_GB), 'naming no volume returns it unscaled'
