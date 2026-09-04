"""The electrode geometry a checkpoint carries: the packaged montage, a flag that follows the basis, and the run guards."""

import argparse
import csv
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from zte.cli.run import (
    _electrode_geometry,
    _refuse_placeholder_resume,
    _refuse_placeholder_training,
    _refuse_unprovisioned_montage,
)
from zte.config import ModelConfig, ZTEConfig
from zte.data.cache import ARTIFACT_SUBDIR
from zte.data.montage.montage import PACKAGED_MONTAGE_NAME, build_montage_csv, packaged_montage_csv
from zte.lens.montage import resolve_checkpoint_montage
from zte.models.embedding import ZTEModel, build_model
from zte.models.spatial import ScalpGeometry, SpatialChannelMixer, SphericalHarmonicEncoding, harmonics_match

_N_CHANNELS = 16
_WINDOW = 24


def _config() -> ModelConfig:
    """A small raw conformer with the harmonic electrode code on."""
    return ModelConfig(
        frontend='raw_conformer',
        embed_dim=32,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        conformer_filters=8,
        factored=False,
        subject_adapter=False,
        spatial_encoding='spherical_harmonics',
        spatial_harmonic_degree=4,
    )


def _write_montage(path: Path, xyz: np.ndarray) -> Path:
    """Writes `xyz` as a `channel,x,y,z,label,region` montage."""
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['channel', 'x', 'y', 'z', 'label', 'region'])
        for c, (x, y, z) in enumerate(xyz):
            writer.writerow([c, f'{x:.6f}', f'{y:.6f}', f'{z:.6f}', f'E{c + 1}', 'frontal'])

    return path


def _model(montage_csv: str | None) -> ZTEModel:
    """A seeded model built with (or without) a montage on this machine."""
    torch.manual_seed(0)

    return build_model(_config(), raw_shape=(_N_CHANNELS, _WINDOW), n_channels=_N_CHANNELS, montage_csv=montage_csv)


def _mixer(model: ZTEModel) -> SpatialChannelMixer:
    """The electrode mixer inside a built model, typed so its flag and encoding are checkable."""
    return cast('SpatialChannelMixer', model.frontend.spatial_mixer)


def _encoding(model: ZTEModel) -> SphericalHarmonicEncoding:
    """The harmonic encoding inside a built model."""
    return _mixer(model).pos


@pytest.fixture()
def cap(tmp_path: Path) -> Path:
    """A montage of sixteen random unit-sphere electrodes."""
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((_N_CHANNELS, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)

    return _write_montage(tmp_path / 'cap.csv', xyz)


@pytest.fixture()
def rotated(cap: Path, tmp_path: Path) -> Path:
    """The same electrodes on a head turned a quarter turn: a real file describing the wrong positions."""
    xyz = ScalpGeometry.from_csv(cap, _N_CHANNELS).xyz[:, [1, 0, 2]]

    return _write_montage(tmp_path / 'rotated.csv', xyz)


# ---- The packaged montage ---- #


def test_the_packaged_montage_is_the_exact_zuco_105_cap() -> None:
    """The package ships the 105 retained GSN-HydroCel electrodes, labelled, on the unit sphere, in eight regions."""
    path = packaged_montage_csv()
    assert path.name == PACKAGED_MONTAGE_NAME and path.is_file()

    geometry = ScalpGeometry.from_csv(path, 105)
    assert geometry.approximate is False and geometry.n_channels == 105
    assert geometry.labels is not None and len(set(geometry.labels)) == 105
    assert all(label.startswith('E') for label in geometry.labels)
    assert np.allclose(np.linalg.norm(geometry.xyz, axis=1), 1.0, atol=1e-6)

    with path.open(encoding='utf-8') as fh:
        regions = {row['region'] for row in csv.DictReader(fh)}
    assert len(regions) == 8


def test_build_montage_csv_ships_the_packaged_copy_when_mne_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `mne` the default ZuCo-105 request is served from the package; any other montage still needs it."""

    def no_mne(*_args: object, **_kwargs: object) -> object:
        raise ImportError('no mne')

    monkeypatch.setattr('zte.data.montage.montage.zuco105_labels', no_mne)
    monkeypatch.setattr('zte.models.spatial.ScalpGeometry.from_mne', no_mne)

    out = build_montage_csv(tmp_path / 'montage.csv')
    assert out.read_bytes() == packaged_montage_csv().read_bytes()

    with pytest.raises(ImportError):
        build_montage_csv(tmp_path / 'other_net.csv', montage='biosemi64')
    with pytest.raises(ImportError):
        build_montage_csv(tmp_path / 'full_net.csv', zuco105=False)


# ---- The flag and the basis ---- #


def test_the_mixer_flag_follows_the_loaded_basis_not_the_build_machine(cap: Path) -> None:
    """A checkpoint trained on a montage stays exact wherever it is loaded, and a placeholder one stays approximate."""
    exact, placeholder = _model(str(cap)), _model(None)
    assert _mixer(exact).approximate_geometry is False
    assert _mixer(placeholder).approximate_geometry is True

    loaded_without_csv = _model(None)
    loaded_without_csv.load_state_dict(exact.state_dict())
    assert _mixer(loaded_without_csv).approximate_geometry is False

    loaded_with_csv = _model(str(cap))
    loaded_with_csv.load_state_dict(placeholder.state_dict())
    assert _mixer(loaded_with_csv).approximate_geometry is True


def test_a_basis_verifies_only_the_coordinates_it_was_built_from(cap: Path, rotated: Path) -> None:
    """Rebuilding the harmonics from the training montage reproduces the buffer; any other head does not."""
    encoding = _encoding(_model(str(cap)))

    assert encoding.matches_geometry(ScalpGeometry.from_csv(cap, _N_CHANNELS))
    assert not encoding.matches_geometry(ScalpGeometry.from_csv(rotated, _N_CHANNELS))
    assert not encoding.matches_geometry(ScalpGeometry.fibonacci_fallback(_N_CHANNELS))

    basis = encoding.harmonics.numpy()
    assert not harmonics_match(basis, ScalpGeometry.from_csv(cap, _N_CHANNELS), encoding.l_max + 1)


def test_resolve_checkpoint_montage_verifies_the_named_file_and_refuses_a_stranger(
    cap: Path, rotated: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config's CSV is used once it rebuilds the basis, a store copy stands in for a missing one, and nothing else."""
    monkeypatch.delenv('ZTE_CACHE_REMOTE', raising=False)
    encoding = _encoding(_model(str(cap)))
    basis, l_max = encoding.harmonics.numpy(), encoding.l_max

    found, reason = resolve_checkpoint_montage(basis, l_max, False, str(cap), _N_CHANNELS)
    assert found is not None and reason is None
    assert found.source == 'config' and found.montage.path == cap and found.montage.labels[0] == 'E1'

    none, why = resolve_checkpoint_montage(basis, l_max, True, str(cap), _N_CHANNELS)
    assert none is None and why is not None and 'approximate' in why

    refused, why = resolve_checkpoint_montage(basis, l_max, False, str(rotated), _N_CHANNELS)
    assert refused is None and why is not None and 'reproduces' in why

    # The persistent store stands in for a file that this machine no longer has.
    store = tmp_path / 'store' / ARTIFACT_SUBDIR
    store.mkdir(parents=True)
    shutil.copyfile(cap, store / 'gone.csv')
    monkeypatch.setenv('ZTE_CACHE_REMOTE', str(tmp_path / 'store'))
    gone = tmp_path / 'elsewhere' / 'gone.csv'

    staged, reason = resolve_checkpoint_montage(basis, l_max, False, str(gone), _N_CHANNELS)
    assert staged is not None and staged.source == 'store' and gone.is_file() and reason is None


# ---- The run guards ---- #


def _args(**overrides: object) -> argparse.Namespace:
    """The `zte-run` flags the guards read, defaulting to no resume and `--spatial keep`."""
    fields: dict[str, object] = {'resume': False, 'spatial': 'keep'} | overrides

    return argparse.Namespace(**fields)


def test_the_run_guards_refuse_the_placeholder_only_when_exact_coordinates_were_asked_for(
    cap: Path, tmp_path: Path
) -> None:
    """`--spatial exact` fails loudly on a placeholder encoder, before or after training; `keep` and decoder runs pass."""
    config = ZTEConfig(model=_config())
    config.dataset.montage_csv = str(cap)
    exact, placeholder = _model(str(cap)), _model(None)

    geometry = _electrode_geometry(placeholder, config)
    assert geometry == {'spatial_encoding': 'spherical_harmonics', 'montage_csv': str(cap), 'approximate': True}
    assert _electrode_geometry(exact, config)['approximate'] is False

    _refuse_placeholder_training(geometry, config, _args(spatial='keep'))
    _refuse_placeholder_training(_electrode_geometry(exact, config), config, _args(spatial='exact'))
    with pytest.raises(SystemExit, match='approximate'):
        _refuse_placeholder_training(geometry, config, _args(spatial='exact'))

    decoder = ZTEConfig(model=_config())
    decoder.train.mode = 'decoder'
    _refuse_placeholder_training(geometry, decoder, _args(spatial='exact'))

    unprovisioned = ZTEConfig(model=_config())
    unprovisioned.dataset.montage_csv = None
    _refuse_unprovisioned_montage(unprovisioned, _args(spatial='keep'))
    with pytest.raises(SystemExit, match='could not provision'):
        _refuse_unprovisioned_montage(unprovisioned, _args(spatial='exact'))


def test_resuming_a_placeholder_run_under_exact_coordinates_is_refused_before_anything_loads(
    cap: Path, tmp_path: Path
) -> None:
    """A run whose `last.pt` was trained on the cap cannot be resumed as exact; a new name is the only honest fix."""
    config = ZTEConfig(model=_config())
    run_dir = tmp_path / 'run'
    last = run_dir / 'checkpoints' / 'last.pt'
    last.parent.mkdir(parents=True)

    torch.save({'config': config.to_dict(), 'model': _model(None).state_dict(), 'extra': {}}, last)
    with pytest.raises(SystemExit, match='new --name'):
        _refuse_placeholder_resume(run_dir, config, _args(resume=True, spatial='exact'))
    _refuse_placeholder_resume(run_dir, config, _args(resume=True, spatial='keep'))
    _refuse_placeholder_resume(run_dir, config, _args(resume=False, spatial='exact'))

    torch.save({'config': config.to_dict(), 'model': _model(str(cap)).state_dict(), 'extra': {}}, last)
    _refuse_placeholder_resume(run_dir, config, _args(resume=True, spatial='exact'))
