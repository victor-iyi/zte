"""The temporal latency profile: the occluded span that moves the embedding, its null floor, and the ms axis."""

import json
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pytest
import torch
from torch import nn

from zte.cli.lens import parse_arguments, temporal_readings, write_temporal
from zte.config import DatasetConfig, MissingConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.device import resolve_device
from zte.inference.embed import ZTEEmbedder
from zte.lens.saliency import DISCLAIMER, Reading, select_reading
from zte.lens.temporal import CAVEAT, N400_WINDOW_MS, render_markdown, temporal_saliency
from zte.models.embedding import ZTEModel

# 350 samples at 500 Hz is the 700 ms window every live raw config uses, and 14 bins divide it into exact 50 ms steps.
_RAW_WINDOW: Final[int] = 350
"""Raw window the profile fixtures are built over."""

_N_BINS: Final[int] = 14
"""Bins the fixtures split that window into, one per 25 samples."""

_SIGNAL_SPAN: Final[tuple[int, int]] = (175, 200)
"""Sample span the planted signal lives in -- bin 7, 350-400 ms after word onset."""

_EMBED_DIM: Final[int] = 8
"""Width of the stub embedding: the summed raw trace of the first eight channels."""


class _RawSumModel(nn.Module):
    """Embeds a sentence as the masked sum of its raw samples per channel, so every occlusion effect is exact."""

    uses_raw = True

    def __init__(self, embed_dim: int = _EMBED_DIM) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self._unused = nn.Parameter(torch.zeros(1), requires_grad=False)

    def embed_sentence(self, batch: dict[str, Any], objective: object = None) -> torch.Tensor:
        """Sums the raw window over time and over valid words, truncated to `embed_dim` channels."""
        raw = batch['raw']
        mask = (batch['pad_mask'] & batch['presence']).to(raw.dtype)[:, :, None, None]

        return (raw * mask).sum(dim=(1, 3))[:, : self.embed_dim]


class _BandPowerModel(nn.Module):
    """A band-power stub: it has no time axis, which is the case the profile must decline rather than fake."""

    uses_raw = False

    def __init__(self) -> None:
        super().__init__()
        self._unused = nn.Parameter(torch.zeros(1), requires_grad=False)

    def embed_sentence(self, batch: dict[str, Any], objective: object = None) -> torch.Tensor:
        """Zeros of the batch's width, never reached by a profile that declines up front."""
        return torch.zeros(int(batch['pad_mask'].shape[0]), _EMBED_DIM)


def _embedder(model: nn.Module) -> ZTEEmbedder:
    """An embedder over one of the stubs above; the profile only calls `embed_sentence` and `uses_raw`."""
    return ZTEEmbedder(cast('ZTEModel', model), ZTEConfig(run_name='temporal_test'), resolve_device('cpu'))


def _plant_signal(dataset: ZuCoDataset, span: tuple[int, int]) -> None:
    """Rigs every raw window so channel 0 carries a burst inside `span` and channel 1 a flat baseline.

    Note:
        Channel 1's flat baseline is what makes the measurement possible: a burst alone would only change the
        embedding's magnitude, and cosine drop is scale-free, so the direction has to move for a bin to score.
    """
    assert dataset.raw_eeg is not None and dataset.presence is not None

    # A dataset loaded from a bundle is memory-mapped read-only, so planting needs its own writable copy.
    raw = np.zeros(np.asarray(dataset.raw_eeg).shape, dtype=np.float32)
    raw[:, 1, :] = 0.05
    raw[:, 0, span[0] : span[1]] = 20.0

    dataset.raw_eeg = raw
    dataset.presence = np.ones_like(np.asarray(dataset.presence), dtype=bool)


def _plant_noise(dataset: ZuCoDataset, seed: int = 7) -> None:
    """Fills every raw window with noise that is time-locked to nothing, the case the null band has to absorb."""
    assert dataset.raw_eeg is not None and dataset.presence is not None
    rng = np.random.default_rng(seed)

    dataset.raw_eeg = rng.normal(0.0, 1.0, size=np.asarray(dataset.raw_eeg).shape).astype(np.float32)
    dataset.presence = np.ones_like(np.asarray(dataset.presence), dtype=bool)


def _readings(dataset: ZuCoDataset, subject: str = 'ZAB', n: int = 3) -> list[Reading]:
    """The subject's first `n` readings, in the dataset's deterministic order."""
    return [select_reading(dataset, subject, index=i) for i in range(n)]


@pytest.fixture()
def raw_dataset(synthetic_dir: Path, tmp_path: Path) -> ZuCoDataset:
    """A raw-representation dataset over the synthetic tree, windowed to the 700 ms every live raw config uses.

    Args:
        synthetic_dir (Path): The synthetic `.mat` directory.
        tmp_path (Path): Per-test temporary directory for the cache.

    Returns:
        ZuCoDataset: A built dataset carrying `(n_channels, 350)` raw windows.
    """
    config = DatasetConfig(
        root=str(synthetic_dir),
        tasks=('SR',),
        representation='raw',
        raw_window=_RAW_WINDOW,
        missing=MissingConfig(method='mask_only'),
        cache_dir=str(tmp_path / 'cache'),
    )

    return ZuCoDataset(config).build(show_progress=False)


# ---- The profile ---- #


def test_profile_peaks_in_the_bin_that_carries_the_signal(raw_dataset: ZuCoDataset) -> None:
    """The occluded span holding the planted burst moves the embedding strictly further than any other span."""
    _plant_signal(raw_dataset, _SIGNAL_SPAN)

    report = temporal_saliency(_embedder(_RawSumModel()), raw_dataset, _readings(raw_dataset), n_bins=_N_BINS)

    assert report is not None
    assert report['method'] == 'temporal_occlusion_cosine_drop'
    drops = [block['mean_drop'] for block in report['bins']]
    peak = int(np.argmax(drops))
    assert report['peak']['bin'] == peak
    assert report['bins'][peak]['start_sample'] <= _SIGNAL_SPAN[0]
    assert report['bins'][peak]['end_sample'] >= _SIGNAL_SPAN[1]
    assert drops[peak] > max(d for b, d in enumerate(drops) if b != peak)


def test_profile_covers_every_bin_over_every_reading(raw_dataset: ZuCoDataset) -> None:
    """Each bin is aggregated over all the readings given, with a bootstrap interval that brackets its mean."""
    _plant_signal(raw_dataset, _SIGNAL_SPAN)
    readings = _readings(raw_dataset)

    report = temporal_saliency(_embedder(_RawSumModel()), raw_dataset, readings, n_bins=_N_BINS)

    assert report is not None
    assert report['n_readings'] == len(readings)
    assert report['n_words'] == sum(r.n_words for r in readings)
    assert len(report['bins']) == _N_BINS
    for block in report['bins']:
        assert block['n'] == len(readings)
        assert block['ci_low'] <= block['mean_drop'] <= block['ci_high']

    # The bins tile the window exactly: no sample is scored twice and none is left unscored.
    assert report['bins'][0]['start_sample'] == 0
    assert report['bins'][-1]['end_sample'] == _RAW_WINDOW
    assert all(a['end_sample'] == b['start_sample'] for a, b in zip(report['bins'], report['bins'][1:], strict=False))


def test_bins_are_reported_in_milliseconds_from_word_onset(raw_dataset: ZuCoDataset) -> None:
    """At 500 Hz a 350-sample window is 700 ms, so 14 bins step 50 ms each from onset."""
    _plant_signal(raw_dataset, _SIGNAL_SPAN)

    report = temporal_saliency(_embedder(_RawSumModel()), raw_dataset, _readings(raw_dataset, n=1), n_bins=_N_BINS)

    assert report is not None
    assert report['sampling_rate_hz'] == 500.0
    assert report['raw_window_samples'] == _RAW_WINDOW
    assert report['window_ms'] == 700.0
    for index, block in enumerate(report['bins']):
        assert block['start_ms'] == pytest.approx(50.0 * index)
        assert block['end_ms'] == pytest.approx(50.0 * (index + 1))
        assert block['center_ms'] == pytest.approx(50.0 * index + 25.0)

    # Samples 175-200 are exactly the 350-400 ms bin, which is where the burst was planted.
    signal_bin = report['bins'][7]
    assert (signal_bin['start_sample'], signal_bin['end_sample']) == _SIGNAL_SPAN
    assert (signal_bin['start_ms'], signal_bin['end_ms']) == (350.0, 400.0)


# ---- The null floor ---- #


def test_null_band_is_a_floor_the_planted_peak_clears(raw_dataset: ZuCoDataset) -> None:
    """The same width occluded at a random offset per word scores far below the time-locked peak."""
    _plant_signal(raw_dataset, _SIGNAL_SPAN)

    report = temporal_saliency(_embedder(_RawSumModel()), raw_dataset, _readings(raw_dataset), n_bins=_N_BINS)

    assert report is not None
    null = report['null_band']
    assert null['width_samples'] == _RAW_WINDOW // _N_BINS
    assert null['width_ms'] == 50.0
    assert null['n_draws'] == report['n_readings']
    assert report['peak']['mean_drop'] > null['ci_high']
    assert report['peak']['above_null'] is True

    # A bin holding nothing but the flat baseline cannot clear a floor built from the same amount of removed signal.
    assert report['bins'][0]['above_null'] is False


def test_no_bin_clears_the_null_when_nothing_is_time_locked(raw_dataset: ZuCoDataset) -> None:
    """Windows of pure noise move the embedding in every bin, and the null absorbs all of it -- an honest flat read."""
    _plant_noise(raw_dataset)

    report = temporal_saliency(_embedder(_RawSumModel()), raw_dataset, _readings(raw_dataset), n_bins=_N_BINS)

    assert report is not None
    # Every bin scores something, which is exactly why a bare bar chart of drops would look like a profile.
    assert min(block['mean_drop'] for block in report['bins']) > 0.0
    assert report['null_band']['mean_drop'] > 0.0
    assert report['peak']['above_null'] is False
    assert not any(block['above_null'] for block in report['bins'])


# ---- What the profile refuses to claim ---- #


def test_peak_in_n400_window_tracks_where_the_peak_actually_fell(raw_dataset: ZuCoDataset) -> None:
    """The flag is set by the peak's millisecond centre and by nothing else: an early burst reads False."""
    embedder = _embedder(_RawSumModel())

    _plant_signal(raw_dataset, _SIGNAL_SPAN)
    late = temporal_saliency(embedder, raw_dataset, _readings(raw_dataset), n_bins=_N_BINS)

    _plant_signal(raw_dataset, (0, 25))
    early = temporal_saliency(embedder, raw_dataset, _readings(raw_dataset), n_bins=_N_BINS)

    assert late is not None and early is not None
    assert list(N400_WINDOW_MS) == [300.0, 500.0]
    assert late['peak']['center_ms'] == 375.0
    assert late['peak_in_n400_window'] is True
    assert early['peak']['bin'] == 0
    assert early['peak']['center_ms'] == 25.0
    assert early['peak_in_n400_window'] is False


def test_profile_declines_a_model_with_no_time_axis(raw_dataset: ZuCoDataset) -> None:
    """A band-power checkpoint has no window to occlude, so the profile returns `None` instead of a fake one."""
    assert temporal_saliency(_embedder(_BandPowerModel()), raw_dataset, _readings(raw_dataset, n=1)) is None


def test_profile_refuses_to_run_over_no_readings(raw_dataset: ZuCoDataset) -> None:
    """One reading is a quirk and none is nothing: the aggregate needs readings to aggregate."""
    with pytest.raises(ValueError, match='at least one reading'):
        temporal_saliency(_embedder(_RawSumModel()), raw_dataset, [])


def test_profile_rejects_more_bins_than_the_window_has_samples(raw_dataset: ZuCoDataset) -> None:
    """A bin narrower than a sample is not a bin."""
    with pytest.raises(ValueError, match='n_bins'):
        temporal_saliency(_embedder(_RawSumModel()), raw_dataset, _readings(raw_dataset, n=1), n_bins=_RAW_WINDOW + 1)


def test_pass_budget_caps_the_readings_rather_than_the_bins(raw_dataset: ZuCoDataset) -> None:
    """A budget too small for every reading profiles fewer of them, and says how many it used."""
    _plant_signal(raw_dataset, _SIGNAL_SPAN)

    report = temporal_saliency(
        _embedder(_RawSumModel()),
        raw_dataset,
        _readings(raw_dataset),
        n_bins=_N_BINS,
        n_null=2,
        max_passes=_N_BINS + 3,
    )

    assert report is not None
    assert report['n_readings'] == 1
    assert len(report['bins']) == _N_BINS


# ---- The rendered artifact ---- #


def test_markdown_carries_the_disclaimer_the_caveat_and_the_peak(raw_dataset: ZuCoDataset) -> None:
    """The rendered profile states the latency in ms, and refuses to let the N400 band be read as a component."""
    _plant_signal(raw_dataset, _SIGNAL_SPAN)
    report = temporal_saliency(_embedder(_RawSumModel()), raw_dataset, _readings(raw_dataset), n_bins=_N_BINS)

    assert report is not None
    markdown = render_markdown(report)

    assert DISCLAIMER in markdown
    assert CAVEAT in markdown
    assert 'eye-tracking-segmented' in markdown
    assert 'is not evidence of one' in markdown
    assert '350.0' in markdown and '400.0' in markdown
    assert markdown.count('\n|') >= _N_BINS
    assert report['caveat'] == CAVEAT and report['disclaimer'] == DISCLAIMER


# ---- The CLI surface ---- #


def test_temporal_flags_default_to_off(raw_dataset: ZuCoDataset) -> None:
    """The profile is opt-in, so an existing `zte-lens` invocation writes exactly what it wrote before."""
    args = parse_arguments(['encode', '--ckpt', 'best.pt', '--out', 'lens', '--synthetic'])

    assert args.temporal is False
    assert args.temporal_bins == 14
    assert args.temporal_sentences == 12


def test_temporal_readings_stop_at_the_end_of_the_subject(raw_dataset: ZuCoDataset) -> None:
    """Asking for more readings than the subject has returns the ones that exist, not an error."""
    readings = temporal_readings(raw_dataset, 'ZAB', 0, None, limit=999)

    assert len(readings) == 6
    assert all(reading.subject == 'ZAB' for reading in readings)
    assert [r.position for r in readings] == sorted(r.position for r in readings)


def test_cli_writes_the_profile_beside_the_lens_artifacts(raw_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """`--temporal` writes temporal.json and temporal.md into the lens directory, provenance and disclaimer included."""
    _plant_signal(raw_dataset, _SIGNAL_SPAN)
    ckpt = tmp_path / 'best.pt'
    ckpt.write_bytes(b'stand-in checkpoint bytes; only their digest reaches the artifact')
    target = tmp_path / 'lens' / 'run_ZAB_0'
    target.mkdir(parents=True)
    flags = ['--temporal', '--temporal-bins', '7', '--temporal-sentences', '2']
    args = parse_arguments(['encode', '--ckpt', str(ckpt), '--out', str(tmp_path / 'lens'), '--synthetic', *flags])

    path = write_temporal(_embedder(_RawSumModel()), raw_dataset, 'ZAB', args, target)

    assert path == target / 'temporal.json'
    profile = json.loads(path.read_text(encoding='utf-8'))
    assert len(profile['bins']) == 7
    assert profile['n_readings'] == 2
    assert profile['disclaimer'] == DISCLAIMER
    assert profile['provenance']['ckpt'] == str(ckpt)
    assert len(profile['provenance']['ckpt_sha256']) == 64
    assert DISCLAIMER in (target / 'temporal.md').read_text(encoding='utf-8')


def test_cli_writes_nothing_for_a_checkpoint_with_no_time_axis(raw_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """A band-power checkpoint leaves no temporal artifact behind rather than an empty one to be quoted."""
    target = tmp_path / 'lens' / 'run_ZAB_0'
    target.mkdir(parents=True)
    args = parse_arguments(
        ['encode', '--ckpt', 'best.pt', '--out', str(tmp_path / 'lens'), '--synthetic', '--temporal']
    )

    assert write_temporal(_embedder(_BandPowerModel()), raw_dataset, 'ZAB', args, target) is None
    assert not (target / 'temporal.json').exists()
    assert not (target / 'temporal.md').exists()
