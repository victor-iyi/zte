"""The attention read-out: hooks that capture the real weights, received mass that sums to one, and honest grouping."""

import builtins
import csv
import json
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pytest
import torch
from torch import nn

from zte.cli.lens import parse_arguments, run_attention
from zte.config import DatasetConfig, MissingConfig, ModelConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.montage.montage import regions_from_geometry
from zte.data.torch_dataset import ZuCoTorchDataset, build_subject_vocab, collate_sentences
from zte.device import resolve_device
from zte.inference.embed import ZTEEmbedder
from zte.lens.attention import (
    ATTENTION_CAVEAT,
    DISCLAIMER,
    AttentionRecorder,
    _force_weights,
    attention_modules,
    attention_profile,
    held_out_ranks,
    record_attention,
    render_markdown,
    write_figures,
)
from zte.models.embedding import ZTEModel, build_model
from zte.models.spatial import ScalpGeometry

# 350 samples at 500 Hz is the 700 ms window every live raw config uses, so the 300-500 ms band is samples 150-250.
_RAW_WINDOW: Final[int] = 350
"""Raw window the fixtures are built over."""

_N400_SAMPLES: Final[tuple[int, int]] = (150, 250)
"""Where the conventional N400 band falls in that window."""


@pytest.fixture()
def raw_dataset(synthetic_dir: Path, tmp_path: Path) -> ZuCoDataset:
    """A raw-representation dataset over the synthetic tree, with a montage CSV so the scalp side has geometry.

    Args:
        synthetic_dir (Path): The synthetic `.mat` directory.
        tmp_path (Path): Per-test temporary directory for the cache and the montage.

    Returns:
        ZuCoDataset: A built dataset carrying `(n_channels, 350)` raw windows and an exact montage.
    """
    config = DatasetConfig(
        root=str(synthetic_dir),
        tasks=('SR',),
        representation='raw',
        raw_window=_RAW_WINDOW,
        missing=MissingConfig(method='mask_only'),
        cache_dir=str(tmp_path / 'cache'),
    )
    dataset = ZuCoDataset(config).build(show_progress=False)
    assert dataset.raw_eeg is not None
    dataset.config.montage_csv = str(_montage_csv(tmp_path / 'montage.csv', int(dataset.raw_eeg.shape[1])))

    return dataset


def _montage_csv(path: Path, n_channels: int) -> Path:
    """Writes an upper-hemisphere montage of `n_channels` electrodes, so mne's topomap has a head to draw on."""
    xyz = ScalpGeometry.fibonacci_fallback(n_channels).xyz.copy()
    xyz[:, 2] = np.abs(xyz[:, 2])
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    regions = regions_from_geometry(xyz)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['channel', 'x', 'y', 'z', 'label', 'region'])
        for c in range(n_channels):
            writer.writerow([c, f'{xyz[c, 0]:.6f}', f'{xyz[c, 1]:.6f}', f'{xyz[c, 2]:.6f}', f'E{c + 1}', regions[c]])

    return path


def _model_config(spatial: bool = True, frontend: str = 'raw_conformer') -> ModelConfig:
    """A tiny raw model: two transformer layers of two heads over eight filters, with or without the mixer."""
    return ModelConfig(
        frontend=frontend,  # type: ignore[arg-type]
        embed_dim=16,
        hidden_dim=16,
        n_layers=4,
        n_heads=2,
        conformer_filters=8,
        factored=False,
        subject_adapter=False,
        grad_checkpoint=False,
        spatial_encoding='spherical_harmonics' if spatial else 'none',
        spatial_harmonic_degree=2,
    )


def _embedder(dataset: ZuCoDataset, spatial: bool = True, frontend: str = 'raw_conformer') -> ZTEEmbedder:
    """A seeded, eval-mode embedder over the dataset's raw shape, naming ZAB as its holdout."""
    assert dataset.raw_eeg is not None
    torch.manual_seed(0)
    c, t = int(dataset.raw_eeg.shape[1]), int(dataset.raw_eeg.shape[2])
    config = ZTEConfig(run_name='attention_test', model=_model_config(spatial, frontend))
    config.train.loso_holdout_subject = 'ZAB'
    model = build_model(config.model, raw_shape=(c, t), n_channels=c, montage_csv=dataset.config.montage_csv)

    return ZTEEmbedder(model, config, resolve_device('cpu'))


def _batch(dataset: ZuCoDataset, positions: list[int]) -> dict[str, Any]:
    """Collates the readings at `positions` of the dataset's deterministic sentence order."""
    torch_ds = ZuCoTorchDataset(dataset, subject_vocab=build_subject_vocab(dataset))

    return collate_sentences([torch_ds[p] for p in positions])


# ---- The hooks ---- #


def test_hooks_capture_per_layer_and_channel_weights_on_the_eval_path(raw_dataset: ZuCoDataset) -> None:
    """Under `eval()` and `no_grad`, every transformer layer and the channel mixer report one block per forward pass."""
    embedder = _embedder(raw_dataset)
    batch = _batch(raw_dataset, [0, 1])
    b, length = batch['pad_mask'].shape
    assert raw_dataset.raw_eeg is not None
    c, t = raw_dataset.raw_eeg.shape[1:]

    with record_attention(embedder.model) as recorder, torch.no_grad():
        embedder.model.embed_sentence(batch)

    assert sorted(recorder.temporal) == [0, 1]
    for chunks in recorder.temporal.values():
        assert len(chunks) == 1
        assert chunks[0].shape == (b * length, t)
        np.testing.assert_allclose(chunks[0].sum(axis=1), 1.0, atol=1e-5)
    assert len(recorder.spatial) == 1
    assert recorder.spatial[0].shape == (b * length, c)
    np.testing.assert_allclose(recorder.spatial[0].sum(axis=1), 1.0, atol=1e-5)
    assert recorder.n_heads == {'temporal': 2, 'spatial': 2}


def test_hooks_are_removed_and_the_fast_path_restored_after_the_block(raw_dataset: ZuCoDataset) -> None:
    """Leaving the block detaches every hook and hands torch's fast-path switch back as it was found."""
    embedder = _embedder(raw_dataset)
    before = torch.backends.mha.get_fastpath_enabled()
    with record_attention(embedder.model) as recorder:
        assert torch.backends.mha.get_fastpath_enabled() is False
    assert torch.backends.mha.get_fastpath_enabled() == before

    with torch.no_grad():
        embedder.model.embed_sentence(_batch(raw_dataset, [0]))
    assert recorder.temporal == {} and recorder.spatial == []

    spatial, temporal, _ = attention_modules(embedder.model)
    for module in [spatial, *temporal]:
        assert module is not None
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_received_attention_equals_the_weights_a_direct_call_returns() -> None:
    """The hook reads exactly `nn.MultiheadAttention`'s per-head weights, averaged over heads and queries."""
    torch.manual_seed(1)
    mha = nn.MultiheadAttention(8, 2, batch_first=True).eval()
    x = torch.randn(3, 5, 8)
    recorder = AttentionRecorder()
    mha.register_forward_pre_hook(_force_weights, with_kwargs=True)
    mha.register_forward_hook(recorder.sink('temporal', 0), with_kwargs=True)

    with torch.no_grad():
        hooked, _ = mha(x, x, x, need_weights=False)
        direct, weights = mha(x, x, x, need_weights=True, average_attn_weights=False)

    torch.testing.assert_close(hooked, direct)
    expected = weights.mean(dim=(1, 2)).numpy()
    np.testing.assert_allclose(recorder.temporal[0][0], expected, atol=1e-6)
    assert recorder.temporal[0][0].shape == (3, 5)


def test_attention_modules_say_why_a_kind_is_absent(raw_dataset: ZuCoDataset) -> None:
    """A model without a mixer, or without an intra-word transformer, names the reason rather than returning nothing."""
    spatial, temporal, reasons = attention_modules(_embedder(raw_dataset, spatial=False).model)
    assert spatial is None and len(temporal) == 2
    assert reasons['temporal'] is None
    assert 'spatial_encoding' in str(reasons['spatial'])

    spatial, temporal, reasons = attention_modules(_embedder(raw_dataset, frontend='eegnet').model)
    assert temporal == [] and spatial is not None
    assert 'no intra-word transformer' in str(reasons['temporal'])


# ---- Which readings count ---- #


def test_held_out_ranks_score_against_other_subjects_only() -> None:
    """A query's own subject is not in its gallery, and a sentence nobody else read cannot be scored."""
    subjects = np.array(['ZAB', 'ZAB', 'ZAB', 'ZDM', 'ZDM'])
    keys = np.array(['s1', 's2', 'only-zab', 's1', 's2'])
    gallery = np.array(
        [
            [1.0, 0.0, 0.0],  # ZAB s1: nearest is ZAB s2 (excluded), then ZDM s2, then ZDM s1 -> rank 2
            [1.0, 0.4, 0.0],  # ZAB s2: nearest other-subject reading is ZDM s2 -> rank 1
            [0.0, 0.0, 1.0],  # ZAB only-zab: no other subject read it -> unscorable
            [1.0, -0.5, 0.0],  # ZDM s1
            [1.0, 0.3, 0.0],  # ZDM s2
        ]
    )

    queries, ranks = held_out_ranks(gallery, subjects, keys, 'ZAB')

    assert queries.tolist() == [0, 1, 2]
    assert ranks.tolist() == [2, 1, -1]


def test_held_out_ranks_with_no_other_subject_is_all_unscorable() -> None:
    """One subject alone has no gallery, so every reading is unscorable rather than trivially correct."""
    queries, ranks = held_out_ranks(np.eye(2), np.array(['ZAB', 'ZAB']), np.array(['a', 'b']), 'ZAB')

    assert queries.tolist() == [0, 1]
    assert ranks.tolist() == [-1, -1]


# ---- The profile ---- #


def _plant_gallery(
    embedder: ZTEEmbedder, dataset: ZuCoDataset, correct: set[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rigs the gallery embed so exactly the ZAB readings at `correct` retrieve their sentence at Top-1.

    Every ZDM reading embeds as the one-hot of its sentence. A correct ZAB reading embeds as that same one-hot; an
    incorrect one embeds as the one-hot of a *different* sentence ZDM read, so its nearest stranger is wrong.
    """
    torch_ds = ZuCoTorchDataset(dataset, subject_vocab=build_subject_vocab(dataset))
    keys = list(torch_ds.stimulus_keys)
    index = {key: i for i, key in enumerate(dict.fromkeys(keys))}
    subjects = [str(dataset.words.iloc[int(rows[0])]['subject']) for rows in torch_ds.sequences]
    zdm_keys = sorted({key for key, subject in zip(keys, subjects, strict=True) if subject == 'ZDM'})
    assert len(zdm_keys) >= 2

    gallery = np.zeros((len(keys), len(index)), dtype=np.float32)
    for pos, (key, subject) in enumerate(zip(keys, subjects, strict=True)):
        if subject == 'ZAB' and pos not in correct:
            wrong = next(k for k in zdm_keys if k != key)
            gallery[pos, index[wrong]] = 1.0
        else:
            gallery[pos, index[key]] = 1.0

    import pandas as pd

    meta = pd.DataFrame({'subject': subjects})
    monkeypatch.setattr(embedder, 'embed', lambda *args, **kwargs: (gallery, meta))


def _zab_positions(dataset: ZuCoDataset) -> list[int]:
    """Positions of ZAB's readings in the dataset's deterministic sentence order."""
    torch_ds = ZuCoTorchDataset(dataset, subject_vocab=build_subject_vocab(dataset))

    return [p for p, rows in enumerate(torch_ds.sequences) if str(dataset.words.iloc[int(rows[0])]['subject']) == 'ZAB']


def test_profile_groups_readings_by_retrieval_and_conserves_mass(
    raw_dataset: ZuCoDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retrieved readings form the `correct` group, the rest `incorrect`, and every curve is a distribution."""
    embedder = _embedder(raw_dataset)
    zab = _zab_positions(raw_dataset)
    _plant_gallery(embedder, raw_dataset, set(zab[:2]), monkeypatch)

    report = attention_profile(embedder, raw_dataset, 'ZAB', batch_size=2)

    assert report is not None
    sel = report['selection']
    assert sel['criterion'] == 'held_out_top1' and sel['postprocess_fit'] == 'none'
    assert sel['n_queries'] == len(zab) and sel['n_correct'] == 2 and sel['n_incorrect'] == len(zab) - 2
    assert report['is_holdout'] is True and report['absent'] == {}

    temporal = report['temporal']
    assert temporal['n_layers'] == 2 and temporal['n_heads'] == 2
    assert len(temporal['times_ms']) == _RAW_WINDOW and temporal['times_ms'][0] == 1.0
    assert temporal['n400_window_samples'] == list(_N400_SAMPLES)
    groups = temporal['groups']
    assert groups['correct']['n_readings'] == 2 and groups['incorrect']['n_readings'] == len(zab) - 2
    assert groups['all']['n_readings'] == len(zab)
    for block in groups.values():
        assert len(block['layers']) == 2
        for layer in block['layers']:
            assert len(layer['mean']) == _RAW_WINDOW
            assert abs(sum(layer['mean']) - 1.0) < 1e-4
            assert all(
                lo <= m <= hi + 1e-9 for lo, m, hi in zip(layer['ci_low'], layer['mean'], layer['ci_high'], strict=True)
            )
        assert block['n400_mass_uniform'] == pytest.approx(100 / 350)
        assert 0.0 <= block['n400_mass'] <= 1.0
        assert block['n400_mass_ci'][0] <= block['n400_mass'] <= block['n400_mass_ci'][1]
        assert block['peak_in_n400_window'] == (300.0 <= block['peak_ms'] <= 500.0)
    assert temporal['contrast']['n400_mass_difference'] == pytest.approx(
        groups['correct']['n400_mass'] - groups['incorrect']['n400_mass']
    )

    spatial = report['spatial']
    assert spatial['approximate_geometry'] is False and spatial['has_time_axis'] is False
    assert spatial['labels'][0] == 'E1' and len(spatial['xy']) == spatial['n_channels']
    for block in spatial['groups'].values():
        assert abs(sum(block['mean']) - 1.0) < 1e-4
        assert len(block['top_channels']) == 10 and block['top_channels'][0]['label'].startswith('E')
        assert sum(block['region_mass'].values()) == pytest.approx(1.0, abs=1e-4)

    assert report['caveat'] == ATTENTION_CAVEAT and report['disclaimer'] == DISCLAIMER


def test_correct_group_is_the_mean_over_exactly_the_retrieved_readings(
    raw_dataset: ZuCoDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUTATION stand-in: the group curve is the per-reading average of those readings' own captured attention."""
    embedder = _embedder(raw_dataset)
    zab = _zab_positions(raw_dataset)
    chosen = [zab[1], zab[3]]
    _plant_gallery(embedder, raw_dataset, set(chosen), monkeypatch)

    report = attention_profile(embedder, raw_dataset, 'ZAB', batch_size=3)
    assert report is not None

    # Recompute the two readings by hand, one at a time, with no grouping machinery in the way.
    curves = []
    for pos in chosen:
        batch = _batch(raw_dataset, [pos])
        with record_attention(embedder.model) as recorder, torch.no_grad():
            embedder.model.embed_sentence(batch)
        valid = (batch['pad_mask'] & batch['presence']).numpy()[0]
        curves.append(recorder.temporal[1][0][valid].mean(axis=0))

    expected = np.mean(curves, axis=0)
    np.testing.assert_allclose(report['temporal']['groups']['correct']['layers'][1]['mean'], expected, atol=1e-5)


def test_max_readings_keeps_the_retrieved_readings_first(
    raw_dataset: ZuCoDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cap never drops a retrieved reading while an unretrieved one is kept."""
    embedder = _embedder(raw_dataset)
    zab = _zab_positions(raw_dataset)
    _plant_gallery(embedder, raw_dataset, set(zab[-2:]), monkeypatch)

    report = attention_profile(embedder, raw_dataset, 'ZAB', batch_size=2, max_readings=3)

    assert report is not None
    assert report['selection']['n_profiled'] == 3
    assert report['temporal']['groups']['correct']['n_readings'] == 2
    assert report['temporal']['groups']['incorrect']['n_readings'] == 1


def test_profile_declines_a_model_with_no_time_axis(raw_dataset: ZuCoDataset) -> None:
    """A band-power checkpoint has no window to attend over, so the answer is `None`, never an empty profile."""

    class _BandPower(nn.Module):
        uses_raw = False

    embedder = ZTEEmbedder(cast('ZTEModel', _BandPower()), ZTEConfig(run_name='bp'), resolve_device('cpu'))

    assert attention_profile(embedder, raw_dataset, 'ZAB') is None


def test_profile_refuses_bad_knobs_and_an_unknown_subject(
    raw_dataset: ZuCoDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-positive knobs and a subject with no readings fail loudly."""
    embedder = _embedder(raw_dataset)
    with pytest.raises(ValueError, match='positive'):
        attention_profile(embedder, raw_dataset, 'ZAB', correct_top_k=0)
    with pytest.raises(ValueError, match='positive'):
        attention_profile(embedder, raw_dataset, 'ZAB', batch_size=0)

    _plant_gallery(embedder, raw_dataset, set(), monkeypatch)
    with pytest.raises(ValueError, match='no reading'):
        attention_profile(embedder, raw_dataset, 'ZZZ')


def test_a_model_without_a_mixer_reports_the_scalp_side_absent(
    raw_dataset: ZuCoDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No mixer means no scalp map and a named reason, while the temporal side still profiles."""
    embedder = _embedder(raw_dataset, spatial=False)
    _plant_gallery(embedder, raw_dataset, set(), monkeypatch)

    report = attention_profile(embedder, raw_dataset, 'ZAB', batch_size=2)

    assert report is not None
    assert report['spatial'] is None and report['temporal'] is not None
    assert 'spatial_encoding' in report['absent']['spatial']
    assert 'correct' not in report['temporal']['groups']
    assert report['temporal']['contrast']['n400_mass_difference'] is None


# ---- Prose ---- #


def test_markdown_carries_the_disclaimer_the_caveat_and_the_groups(
    raw_dataset: ZuCoDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rendered report names every group's N400 mass, the most attended electrodes, and what it must not be read as."""
    embedder = _embedder(raw_dataset)
    zab = _zab_positions(raw_dataset)
    _plant_gallery(embedder, raw_dataset, set(zab[:1]), monkeypatch)
    report = attention_profile(embedder, raw_dataset, 'ZAB', batch_size=2)
    assert report is not None

    text = render_markdown(report, {'temporal': 'a.png', 'topomap': None, 'topomap_reason': 'no reason to'})

    assert DISCLAIMER in text and ATTENTION_CAVEAT in text
    assert '| correct | 1 |' in text and '| incorrect |' in text and '| all |' in text
    assert 'Correct minus incorrect N400 mass' in text
    assert 'E' in text and 'by region' in text
    assert '`a.png`' in text and 'scalp map not drawn: no reason to' in text


# ---- The CLI ---- #


def _checkpoint(embedder: ZTEEmbedder, dataset: ZuCoDataset, path: Path) -> Path:
    """Saves the embedder's model as a real checkpoint the CLI can rebuild from."""
    assert dataset.raw_eeg is not None
    c, t = int(dataset.raw_eeg.shape[1]), int(dataset.raw_eeg.shape[2])
    torch.save(
        {
            'config': embedder.config.to_dict(),
            'model': embedder.model.state_dict(),
            'epoch': 0,
            'step': 0,
            'extra': {'raw_shape': (c, t), 'n_channels': c, 'montage_csv': dataset.config.montage_csv},
        },
        path,
    )

    return path


def test_attention_parser_defaults_match_the_contract() -> None:
    """`attention` parses with exactly the contract defaults when only the required flags are given."""
    args = parse_arguments(['attention', '--ckpt', 'best.pt', '--out', 'att', '--synthetic'])

    assert args.command == 'attention'
    assert args.ckpt == 'best.pt' and args.out == Path('att') and args.synthetic is True
    assert args.subject is None and args.correct_top_k == 1 and args.batch_size == 4
    assert args.max_readings == 0 and args.seed == 0 and args.device == 'auto' and args.force is False


def test_cli_writes_the_profile_its_prose_and_its_figures(raw_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """`zte-lens attention` writes attention.json, attention.md and both PNGs, then skips an identical re-run."""
    embedder = _embedder(raw_dataset)
    ckpt = _checkpoint(embedder, raw_dataset, tmp_path / 'best.pt')
    bundle = tmp_path / 'bundle'
    raw_dataset.save(bundle)
    out = tmp_path / 'attention'
    args = parse_arguments(
        [
            'attention',
            '--ckpt',
            str(ckpt),
            '--out',
            str(out),
            '--bundle',
            str(bundle),
            '--batch-size',
            '2',
            '--device',
            'cpu',
        ]
    )

    path = run_attention(args)

    target = out / 'attention_test_ZAB_attention'
    assert path == target / 'attention.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['disclaimer'] == DISCLAIMER
    assert report['provenance']['run_name'] == 'attention_test'
    assert len(report['provenance']['ckpt_sha256']) == 64
    assert report['figures']['temporal'] == str(target / 'attention_temporal.png')
    assert report['figures']['topomap'] == str(target / 'attention_topomap.png')
    assert (target / 'attention_temporal.png').stat().st_size > 0
    assert (target / 'attention_topomap.png').stat().st_size > 0
    assert DISCLAIMER in (target / 'attention.md').read_text(encoding='utf-8')

    stamp = path.stat().st_mtime_ns
    assert run_attention(args) == path
    assert path.stat().st_mtime_ns == stamp


def test_cli_refuses_a_checkpoint_with_no_time_axis(
    raw_dataset: ZuCoDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A band-power checkpoint exits with a reason rather than leaving an empty artifact behind."""
    embedder = _embedder(raw_dataset)
    ckpt = _checkpoint(embedder, raw_dataset, tmp_path / 'best.pt')
    bundle = tmp_path / 'bundle'
    raw_dataset.save(bundle)
    monkeypatch.setattr('zte.lens.attention.attention_profile', lambda *a, **k: None)
    args = parse_arguments(['attention', '--ckpt', str(ckpt), '--out', str(tmp_path / 'att'), '--bundle', str(bundle)])

    with pytest.raises(SystemExit, match='no raw time axis'):
        run_attention(args)
    assert not (tmp_path / 'att').exists()


# ---- The figures ---- #


def test_scalp_map_is_declined_on_the_approximate_geometry(
    raw_dataset: ZuCoDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the coordinate-free cap a topoplot would show array indices, so none is drawn and the reason is recorded."""
    embedder = _embedder(raw_dataset)
    _plant_gallery(embedder, raw_dataset, set(), monkeypatch)
    report = attention_profile(embedder, raw_dataset, 'ZAB', batch_size=3)
    assert report is not None
    report['spatial']['approximate_geometry'] = True

    written = write_figures(report, tmp_path / 'figs')

    assert written['temporal'] and Path(written['temporal']).stat().st_size > 0
    assert written['topomap'] is None
    assert 'approximate' in str(written['topomap_reason'])
    assert not (tmp_path / 'figs' / 'attention_topomap.png').exists()


def test_scalp_map_falls_back_to_the_in_house_projection_without_mne(
    raw_dataset: ZuCoDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `mne` the scalp maps are still drawn, through the project's own azimuthal projection."""
    embedder = _embedder(raw_dataset)
    zab = _zab_positions(raw_dataset)
    _plant_gallery(embedder, raw_dataset, set(zab[:1]), monkeypatch)
    report = attention_profile(embedder, raw_dataset, 'ZAB', batch_size=3)
    assert report is not None

    real_import = builtins.__import__

    def without_mne(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'mne' or name.startswith('mne.'):
            raise ImportError('mne is not installed')

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_mne)
    written = write_figures(report, tmp_path / 'figs')

    assert written['topomap'] == str(tmp_path / 'figs' / 'attention_topomap.png')
    assert Path(written['topomap']).stat().st_size > 0
    assert written['topomap_reason'] is None
