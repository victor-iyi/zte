"""Tests for the `zte-decode` and `zte-rebaseline` command-line surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from zte.cli.decode import (
    CONTROLS,
    DecodeOptions,
    candidate_set_size,
    mismatch_partners,
    noise_transform,
    options_from_args,
    parse_arguments,
    phase_transform,
    split_indices,
)
from zte.cli.rebaseline import default_out_dir, resolve_holdout
from zte.config import DecoderConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset

# --------------------------------------------------------------------------- #
# the mismatch control
# --------------------------------------------------------------------------- #


def _lengths_and_ids(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns a word-count and stimulus-id array with several readings per stimulus."""
    rng = np.random.default_rng(seed)
    content = np.repeat(np.arange(n // 3), 3)[:n]
    lengths = np.repeat(rng.integers(4, 30, size=n // 3), 3)[:n]
    return lengths.astype(np.int64), content.astype(np.int64)


def test_mismatch_partner_is_never_the_reading_itself() -> None:
    """Every row is paired with a different row, so the control cannot decode its own evidence."""
    lengths, content = _lengths_and_ids(60)
    partner = mismatch_partners(lengths, content, length_tol=1, seed=0)
    assert partner.shape == (60,)
    assert not np.any(partner == np.arange(60))


def test_mismatch_partner_is_a_different_stimulus() -> None:
    """The partner reads another sentence, so the control answers "which brain", not "any brain"."""
    lengths, content = _lengths_and_ids(60)
    partner = mismatch_partners(lengths, content, length_tol=1, seed=0)
    assert not np.any(content[partner] == content)


def test_mismatch_partner_is_length_matched() -> None:
    """Pairing inside word-count strata keeps the 5-bit length confound out of the control."""
    lengths, content = _lengths_and_ids(90, seed=1)
    partner = mismatch_partners(lengths, content, length_tol=1, seed=0)
    gap = np.abs(lengths[partner] - lengths)
    scrambled = np.abs(lengths[np.random.default_rng(0).permutation(90)] - lengths)
    assert gap.mean() < scrambled.mean()


def test_mismatch_partner_is_deterministic() -> None:
    """A fixed seed reproduces the pairing, so a reported delta can be recomputed."""
    lengths, content = _lengths_and_ids(45, seed=2)
    left = mismatch_partners(lengths, content, seed=7)
    right = mismatch_partners(lengths, content, seed=7)
    assert np.array_equal(left, right)


def test_mismatch_partner_handles_a_single_reading() -> None:
    """One reading has nobody to swap with, and the identity is returned rather than an error."""
    partner = mismatch_partners(np.array([12]), np.array([3]))
    assert partner.tolist() == [0]


# --------------------------------------------------------------------------- #
# the signal-destroying surrogates
# --------------------------------------------------------------------------- #


def test_phase_transform_keeps_the_power_spectrum_and_destroys_the_waveform() -> None:
    """The phase control has to leave every band's power exactly where it was, or it is a power control too."""
    import torch

    rng = np.random.default_rng(0)
    raw = torch.from_numpy(rng.standard_normal((4, 6, 8, 32)).astype(np.float32))
    out = phase_transform(seed=0)({'raw': raw})
    assert out['raw'].shape == raw.shape
    assert not torch.equal(out['raw'], raw)
    before = np.abs(np.fft.rfft(raw.numpy(), axis=-1))
    after = np.abs(np.fft.rfft(out['raw'].numpy(), axis=-1))
    assert np.allclose(before, after, atol=1e-3)


def test_phase_transform_leaves_a_batch_without_raw_signal_untouched() -> None:
    """A band-power batch carries no waveform to scramble, and the surrogate says so by changing nothing."""
    import torch

    batch = {'features': torch.zeros(2, 3, 4)}
    assert phase_transform(seed=0)(batch) is batch


def test_noise_transform_matches_moments_and_destroys_the_signal() -> None:
    """The noise control keeps the batch's first two moments and none of its structure."""
    import torch

    rng = np.random.default_rng(0)
    raw = torch.from_numpy(rng.standard_normal((4, 6, 8, 16)).astype(np.float32) * 3.0 + 1.0)
    out = noise_transform(seed=0)({'raw': raw})
    assert out['raw'].shape == raw.shape
    assert not torch.equal(out['raw'], raw)
    assert abs(float(out['raw'].mean()) - float(raw.mean())) < 0.2
    assert abs(float(out['raw'].std()) - float(raw.std())) < 0.3


# --------------------------------------------------------------------------- #
# the candidate set the verdict refuses to headline
# --------------------------------------------------------------------------- #

_GALLERY: list[str] = [
    'the market closed higher on friday',
    'she walked to the station in the rain',
    'they built a bridge across the river',
]


def test_a_free_decode_reports_no_candidate_set() -> None:
    """Text the gallery does not contain cannot have been picked from it, so the decode is free."""
    assert candidate_set_size(['market the friday closed on higher'], _GALLERY) is None
    assert candidate_set_size(['the market closed higher on friday today'], _GALLERY) is None


def test_a_decode_confined_to_the_gallery_is_read_back_as_a_candidate_set() -> None:
    """Every hypothesis being a gallery sentence is forced choice, whether or not the decode declared one."""
    hypotheses = ['The market closed higher on Friday!', 'they built a bridge across the river']
    assert candidate_set_size(hypotheses, _GALLERY) == 3


def test_one_hypothesis_off_the_gallery_makes_the_whole_decode_free() -> None:
    """A constrained decode emits only candidates, so a single novel sentence settles it."""
    hypotheses = ['the market closed higher on friday', 'a sentence nobody read']
    assert candidate_set_size(hypotheses, _GALLERY) is None


def test_candidate_set_size_needs_both_a_decode_and_a_gallery() -> None:
    """With nothing to compare there is no evidence of a candidate set, and none is claimed."""
    assert candidate_set_size([], _GALLERY) is None
    assert candidate_set_size(['the market closed higher on friday'], []) is None
    assert candidate_set_size([''], ['', '  ']) is None


# --------------------------------------------------------------------------- #
# option and split resolution
# --------------------------------------------------------------------------- #


def _args(**overrides: object) -> argparse.Namespace:
    """Builds a `zte-decode` namespace with every flag left at its parser default."""
    defaults: dict[str, object] = {
        'controls': None,
        'oracle': True,
        'beams': None,
        'max_new_tokens': None,
        'batch_size': 8,
        'n_perm': None,
        'n_boot': 2000,
        'rescore': None,
        'length_tol': None,
        'mean_prefix_readings': 512,
        'seed': 0,
    }
    return argparse.Namespace(**(defaults | overrides))


def test_options_fall_back_to_the_checkpoint_decoder_config() -> None:
    """An unset flag reads the value the run was trained under, not a CLI constant."""
    config = ZTEConfig(
        decoder=DecoderConfig(n_permutations=321, length_tol=3, rescore_gallery=False)
    )
    options = options_from_args(_args(), config)
    assert options.n_perm == 321
    assert options.length_tol == 3
    assert options.rescore is False
    assert options.controls == tuple(config.decoder.generation_controls)


def test_options_reject_an_unknown_control() -> None:
    """A typo in `--controls` fails loudly rather than silently dropping a control from the gate."""
    with pytest.raises(ValueError, match='unknown control'):
        options_from_args(_args(controls='mean_prefix,teleport'), ZTEConfig())


def test_every_default_control_is_known() -> None:
    """The shipped `generation_controls` are exactly the ones the CLI can decode."""
    assert set(DecoderConfig().generation_controls) <= set(CONTROLS)
    assert DecodeOptions().controls == CONTROLS


def test_split_indices_returns_disjoint_cells(small_dataset: ZuCoDataset) -> None:
    """The honest headline cell shares no reading with the cell the bridge trained on."""
    config = ZTEConfig()
    config.train.split = 'by_subject_and_stimulus'
    config.train.val_fraction = 0.15
    config.train.test_fraction = 0.2
    config.train.loso_holdout_subject = str(small_dataset.words['subject'].iloc[0])
    train = split_indices(small_dataset, config, 'train')
    test = split_indices(small_dataset, config, 'test')
    assert train is not None and test is not None
    assert not set(train.tolist()) & set(test.tolist())


def test_split_indices_reports_a_missing_cell(small_dataset: ZuCoDataset) -> None:
    """A strategy without the requested cell returns `None` instead of an empty decode."""
    config = ZTEConfig()
    config.train.split = 'random'
    config.train.test_fraction = 0.0
    assert split_indices(small_dataset, config, 'test_seen_stim') is None


def test_decode_parser_accepts_the_documented_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flags the decoder documentation promises all parse."""
    monkeypatch.setattr(
        'sys.argv',
        [
            'zte-decode',
            '--ckpt',
            'best.pt',
            '--synthetic',
            '--split',
            'test_seen_stim',
            '--controls',
            'mean_prefix,null_prefix',
            '--no-oracle',
            '--beams',
            '2',
            '--max-new-tokens',
            '12',
            '--n-perm',
            '64',
            '--no-rescore',
            '--length-tol',
            '2',
        ],
    )
    args = parse_arguments()
    options = options_from_args(args, ZTEConfig())
    assert args.split == 'test_seen_stim'
    assert options.controls == ('mean_prefix', 'null_prefix')
    assert options.oracle is False
    assert options.rescore is False
    assert (options.beams, options.max_new_tokens, options.n_perm, options.length_tol) == (
        2,
        12,
        64,
        2,
    )


# --------------------------------------------------------------------------- #
# the rebaseline surface
# --------------------------------------------------------------------------- #


def test_rebaseline_writes_beside_the_run() -> None:
    """The audit lands in the run directory, not next to the checkpoint file."""
    assert default_out_dir('res/experiments/run/checkpoints/best.pt') == (
        Path('res/experiments/run').resolve() / 'rebaseline'
    )


def test_rebaseline_holdout_prefers_the_cli_then_the_run_config() -> None:
    """An explicit `--holdout` wins; otherwise the run's own LOSO subject is audited."""
    config = ZTEConfig()
    config.train.loso_holdout_subject = 'ZAB'
    subjects = np.array(['ZAB', 'ZDM', 'ZKB'])
    assert resolve_holdout(config, 'ZKB', subjects) == 'ZKB'
    assert resolve_holdout(config, None, subjects) == 'ZAB'


def test_rebaseline_holdout_is_none_for_a_single_subject() -> None:
    """One subject means no cross-subject query set, and the audit says so rather than inventing one."""
    config = ZTEConfig()
    config.train.loso_holdout_subject = None
    assert resolve_holdout(config, None, np.array(['ZAB', 'ZAB'])) is None
