"""Tests for the turn-key `--spatial` / `--meaning` provisioning (see `zte.cli.provision`).

These cover the config-wiring for each choice. The two choices that build heavy artifacts (`--spatial
exact`, `--meaning static`) are exercised with the builders monkeypatched, so the tests need neither
`mne` nor network access while still proving the artifact path is built and wired.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zte.cli import provision
from zte.config import ZTEConfig


def test_apply_spatial_keep_is_noop() -> None:
    """`keep` leaves the electrode config untouched."""
    cfg = ZTEConfig()
    before = (cfg.model.spatial_encoding, cfg.dataset.montage_csv)
    provision.apply_spatial(cfg, 'keep')
    assert (cfg.model.spatial_encoding, cfg.dataset.montage_csv) == before


def test_apply_spatial_off_and_approx() -> None:
    """`off` disables the encoding; `approx` turns on the coordinate-free spherical-harmonic cap."""
    cfg = ZTEConfig()
    cfg.dataset.montage_csv = 'stale.csv'
    provision.apply_spatial(cfg, 'off')
    assert cfg.model.spatial_encoding == 'none' and cfg.dataset.montage_csv is None

    cfg2 = ZTEConfig()
    cfg2.dataset.montage_csv = 'stale.csv'
    provision.apply_spatial(cfg2, 'approx')
    assert cfg2.model.spatial_encoding == 'spherical_harmonics'
    assert cfg2.dataset.montage_csv is None  # approx = no exact coordinates


def test_apply_spatial_exact_builds_and_wires(monkeypatch: pytest.MonkeyPatch) -> None:
    """`exact` builds the montage CSV and wires both the encoding and the montage path."""
    calls: dict[str, object] = {}

    def fake_build(out: str, *, montage: str, zuco105: bool) -> Path:
        calls['out'], calls['montage'], calls['zuco105'] = out, montage, zuco105
        return Path(out)

    monkeypatch.setattr('zte.data.montage.build_montage_csv', fake_build)
    cfg = ZTEConfig()
    provision.apply_spatial(cfg, 'exact', montage_out='res/m/x.csv')
    assert cfg.model.spatial_encoding == 'spherical_harmonics'
    assert cfg.dataset.montage_csv == 'res/m/x.csv'
    assert calls['zuco105'] is True


def test_apply_spatial_attention_uses_learned_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    """`attention` selects the learned spatial-attention scheme on the exact montage."""
    monkeypatch.setattr('zte.data.montage.build_montage_csv', lambda out, **_: Path(out))
    cfg = ZTEConfig()
    provision.apply_spatial(cfg, 'attention', montage_out='m.csv')
    assert cfg.model.spatial_encoding == 'spatial_attention'
    assert cfg.dataset.montage_csv == 'm.csv'


def test_apply_spatial_exact_without_mne_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing `mne` degrades `exact` to the approximate cap rather than crashing the run."""

    def boom(*_a: object, **_k: object) -> Path:
        raise ImportError('no mne')

    monkeypatch.setattr('zte.data.montage.build_montage_csv', boom)
    cfg = ZTEConfig()
    provision.apply_spatial(cfg, 'exact')
    assert cfg.model.spatial_encoding == 'spherical_harmonics'  # still on...
    assert cfg.dataset.montage_csv is None  # ...but coordinate-free


def test_apply_meaning_hash_and_keep() -> None:
    """`hash` picks the mechanism-only target; `keep` changes nothing."""
    cfg = ZTEConfig()
    cfg.objective.meaning_contextual = 'bert-base-uncased'
    provision.apply_meaning(cfg, 'hash', weight=0.5)
    assert cfg.objective.meaning_source == 'hash' and cfg.objective.meaning_contextual is None

    cfg2 = ZTEConfig()
    cfg2.objective.meaning_source = 'keepme'
    provision.apply_meaning(cfg2, 'keep')
    assert cfg2.objective.meaning_source == 'keepme'


def test_apply_meaning_contextual_wires_model_layer_dim() -> None:
    """`contextual` wires the model id, layer and (informational) dim."""
    cfg = ZTEConfig()
    provision.apply_meaning(cfg, 'contextual', model='bert-base-uncased', layer=8, weight=1.0)
    assert cfg.objective.meaning_contextual == 'bert-base-uncased'
    assert cfg.objective.meaning_context_layer == 8
    assert cfg.objective.meaning_dim == provision.DEFAULT_CONTEXTUAL_DIM
    assert cfg.objective.meaning_distill_weight == 1.0


def test_apply_meaning_static_builds_and_wires(monkeypatch: pytest.MonkeyPatch) -> None:
    """`static` builds the GloVe file, restricting to the given vocab, and wires source + dim."""
    seen: dict[str, object] = {}

    def fake_provision(out: str, *, vocab: object, model: str) -> tuple[Path, int]:
        seen['vocab'], seen['model'] = vocab, model
        return Path(out), 300

    monkeypatch.setattr('zte.data.glove.provision_glove', fake_provision)
    cfg = ZTEConfig()
    provision.apply_meaning(
        cfg, 'static', meaning_out='res/v/g.txt', weight=0.5, vocab={'apple', 'brain'}
    )
    assert cfg.objective.meaning_source == 'res/v/g.txt'
    assert cfg.objective.meaning_dim == 300
    assert cfg.objective.meaning_contextual is None
    assert seen['vocab'] == {'apple', 'brain'}


def test_provision_glove_reuses_existing_file(tmp_path: Path) -> None:
    """An existing GloVe file is reused (no download / re-filter), and its dim is read back."""
    from zte.data.glove import provision_glove

    f = tmp_path / 'g.txt'
    f.write_text('the 0.1 0.2 0.3\nbrain 0.4 0.5 0.6\n', encoding='utf-8')
    path, dim = provision_glove(f)  # overwrite=False (default) -> cache hit, no network
    assert path == f and dim == 3


def test_build_montage_csv_reuses_existing_file(tmp_path: Path) -> None:
    """An existing montage CSV is reused on the cache-hit path -- and needs no `mne`."""
    from zte.data.montage import build_montage_csv

    f = tmp_path / 'm.csv'
    f.write_text('channel,x,y,z,label,region\n0,0,0,1,E1,frontopolar\n', encoding='utf-8')
    assert build_montage_csv(f) == f  # returns immediately, no ScalpGeometry.from_mne


def test_contextual_meaning_matrix_uses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-seeded contextual-meaning cache is loaded without importing `transformers`."""
    import numpy as np
    import pandas as pd

    from zte.data import meaning

    words = pd.DataFrame(
        {'word': ['the', 'brain'], 'word_idx': [0, 1], 'stimulus_key': ['s1', 's1']}
    )
    # Reproduce the exact key the function computes, then seed a matrix there.
    skey = words['stimulus_key'].fillna('').astype(str).to_numpy()
    widx = words['word_idx'].to_numpy().astype(int)
    warr = words['word'].fillna('').astype(str).to_numpy()
    cache = meaning._hf_cache_path('bert-base-uncased', 8, skey, widx, warr, str(tmp_path))
    cache.parent.mkdir(parents=True, exist_ok=True)
    mat = np.arange(10, dtype=np.float32).reshape(2, 5)
    np.save(cache, mat)

    # Poison the transformers import so a cache miss would raise rather than silently pass.
    import builtins

    real_import = builtins.__import__

    def guard(name: str, *a: object, **k: object):
        if name == 'transformers':
            raise AssertionError('cache miss: transformers should not be imported on a hit')
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', guard)
    out, dim = meaning.build_meaning_matrix_hf(
        words, 'bert-base-uncased', layer=8, cache_dir=str(tmp_path)
    )
    assert dim == 5 and out.shape == (2, 5) and np.allclose(out, mat)


def test_provision_from_args_dispatches_both(monkeypatch: pytest.MonkeyPatch) -> None:
    """`provision_from_args` applies both flags from a parsed namespace."""
    import argparse

    monkeypatch.setattr('zte.data.montage.build_montage_csv', lambda out, **_: Path(out))
    cfg = ZTEConfig()
    args = argparse.Namespace(
        spatial='exact',
        meaning='contextual',
        montage_out='m.csv',
        montage_name=None,
        meaning_model='bert-base-uncased',
        meaning_layer=9,
        meaning_out=provision.DEFAULT_MEANING_OUT,
        meaning_weight=1.0,
    )
    provision.provision_from_args(cfg, args)
    assert cfg.dataset.montage_csv == 'm.csv'
    assert cfg.objective.meaning_contextual == 'bert-base-uncased'
    assert cfg.objective.meaning_context_layer == 9
