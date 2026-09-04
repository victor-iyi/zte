"""The publication schematics: every data-free figure renders, and the artifact-driven ones refuse what they must."""

import json
from pathlib import Path

import numpy as np
import pytest

from zte.data.montage.montage import packaged_montage_csv
from zte.evaluation.schematics import (
    SCHEMATICS,
    attention_temporal_figure,
    attention_topomap_figure,
    build_all,
    contact_sheet,
    save_figure,
    transfer_heatmap_figure,
)
from zte.lens.montage import azimuthal_xy, load_montage_csv

_WINDOW = 350


def _attention_payload(verified: bool = True, approximate: bool = False) -> dict[str, object]:
    """A synthetic `attention.json` over the packaged montage, with a temporal block and three scalp groups."""
    montage = load_montage_csv(packaged_montage_csv(), 105)
    assert montage is not None
    rng = np.random.default_rng(0)
    n = montage.xyz.shape[0]

    def scalp(tilt: float) -> dict[str, object]:
        mean = 1.0 + tilt * montage.xyz[:, 1] + 0.02 * rng.standard_normal(n)
        mean /= mean.sum()
        return {'n_readings': 10, 'mean': mean.tolist(), 'ci_low': mean.tolist(), 'ci_high': mean.tolist()}

    curve = np.full(_WINDOW, 1.0 / _WINDOW)
    layer = {'mean': curve.tolist(), 'ci_low': (0.9 * curve).tolist(), 'ci_high': (1.1 * curve).tolist()}

    return {
        'subject': 'ZAB',
        'temporal': {
            'n_layers': 2,
            'headline_layer': 1,
            'times_ms': [1000.0 * (t + 0.5) / 500.0 for t in range(_WINDOW)],
            'n400_window_ms': [300.0, 500.0],
            'uniform': 1.0 / _WINDOW,
            'groups': {'all': {'n_readings': 10, 'n_words': 100, 'layers': [layer, layer]}},
        },
        'spatial': {
            'n_channels': n,
            'labels': montage.labels,
            'regions': montage.regions,
            'xyz': montage.xyz.tolist(),
            'xy': azimuthal_xy(montage.xyz).tolist(),
            'approximate_geometry': approximate,
            'montage_verified': verified,
            'montage_source': 'packaged' if verified else None,
            'montage_reason': None if verified else 'no montage on this machine reproduces the checkpoint basis',
            'groups': {'all': scalp(0.0), 'correct': scalp(0.08), 'incorrect': scalp(-0.06)},
        },
    }


@pytest.mark.parametrize('name', sorted(SCHEMATICS))
def test_every_data_free_schematic_renders(name: str, tmp_path: Path) -> None:
    """Each named schematic writes a non-empty file in every requested format."""
    rendered = build_all(tmp_path, [name], formats=('png', 'svg'))

    assert [r.name for r in rendered] == [name]
    for path in rendered[0].paths:
        assert path.exists() and path.stat().st_size > 0


def test_build_all_refuses_an_unknown_name(tmp_path: Path) -> None:
    """A typo in a schematic name is an error, not a silently empty directory."""
    with pytest.raises(KeyError, match='unknown'):
        build_all(tmp_path, ['encoder_pipelin'])


def test_the_contact_sheet_tiles_what_was_rendered(tmp_path: Path) -> None:
    """The overview page is written from the PNGs and names each panel."""
    rendered = build_all(tmp_path, ['loso_ring', 'word_window'], formats=('png',))

    sheet = contact_sheet(rendered, tmp_path)

    assert sheet == tmp_path / 'contact_sheet.png' and sheet.stat().st_size > 0


def test_the_attention_figures_draw_a_verified_montage_and_refuse_an_unverified_one(tmp_path: Path) -> None:
    """The scalp figure is drawn only from a profile whose coordinates were proven against the checkpoint basis."""
    good = tmp_path / 'attention.json'
    good.write_text(json.dumps(_attention_payload()), encoding='utf-8')

    rendered = save_figure(attention_topomap_figure(good), tmp_path, 'topomap', ('png',))
    assert rendered.paths[0].stat().st_size > 0
    rendered = save_figure(attention_temporal_figure(good), tmp_path, 'temporal', ('png',))
    assert rendered.paths[0].stat().st_size > 0

    unverified = tmp_path / 'unverified.json'
    unverified.write_text(json.dumps(_attention_payload(verified=False)), encoding='utf-8')
    with pytest.raises(ValueError, match='not verified'):
        attention_topomap_figure(unverified)

    placeholder = tmp_path / 'placeholder.json'
    placeholder.write_text(json.dumps(_attention_payload(approximate=True)), encoding='utf-8')
    with pytest.raises(ValueError, match='not verified'):
        attention_topomap_figure(placeholder)

    with pytest.raises(ValueError, match='no temporal block'):
        attention_temporal_figure(good, group='correct')


def test_the_transfer_heatmap_reads_the_parallax_cells(tmp_path: Path) -> None:
    """The matrix is drawn from `cells[train][eval]`, averaged over seeds, and an empty artifact is refused."""
    tasks = ['NR', 'SR', 'TSR']
    cells = {
        train: {
            evaluated: [{'rank_percentile': 0.9 + 0.01 * (i + j), 'novel_stimuli': train != evaluated}]
            for j, evaluated in enumerate(tasks)
        }
        for i, train in enumerate(tasks)
    }
    parallax = tmp_path / 'PARALLAX.json'
    parallax.write_text(json.dumps({'tasks': tasks, 'cells': cells}), encoding='utf-8')

    rendered = save_figure(transfer_heatmap_figure(parallax), tmp_path, 'transfer', ('png',))
    assert rendered.paths[0].stat().st_size > 0

    empty = tmp_path / 'empty.json'
    empty.write_text(json.dumps({'tasks': tasks, 'cells': {}}), encoding='utf-8')
    with pytest.raises(ValueError, match='no transfer cells'):
        transfer_heatmap_figure(empty)
