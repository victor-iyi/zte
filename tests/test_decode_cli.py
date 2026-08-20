"""Tests for the `zte-decode` and `zte-rebaseline` command-line surfaces."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

# The whole decoder test path is offline: `lm_source='tiny'` builds its LM and tokeniser locally.
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import pytest
import torch

from zte.cli.decode import (
    CONTROLS,
    DecodeOptions,
    candidate_set_size,
    decode_evaluation,
    mismatch_partners,
    noise_transform,
    options_from_args,
    parse_arguments,
    phase_transform,
    split_indices,
)
from zte.cli.rebaseline import default_out_dir, resolve_holdout
from zte.config import DecoderConfig, ModelConfig, ObjectiveConfig, TrainConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.torch_dataset import ZuCoTorchDataset, build_subject_vocab
from zte.device import resolve_device
from zte.inference.capacity import gallery_scores
from zte.inference.decode import ZTEDecoder
from zte.models.decoder import GapCorrector, build_bridge, build_lm
from zte.models.embedding import build_model

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
        'within_task': None,
        'capacity': None,
        'capacity_ks': None,
        'capacity_alpha': None,
        'capacity_n_perm': None,
        'seeds': None,
        'seed': 0,
    }
    return argparse.Namespace(**(defaults | overrides))


def test_options_fall_back_to_the_checkpoint_decoder_config() -> None:
    """An unset flag reads the value the run was trained under, not a CLI constant."""
    config = ZTEConfig(decoder=DecoderConfig(n_permutations=321, length_tol=3, rescore_gallery=False))
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
# the menu-capacity options
# --------------------------------------------------------------------------- #


def test_capacity_is_off_unless_something_asks_for_it() -> None:
    """The audit costs a length-matched gallery pass, so an existing run stays behaviourally identical."""
    assert DecodeOptions().capacity is False
    assert ZTEConfig().objective.eval_capacity is False
    assert options_from_args(_args(), ZTEConfig()).capacity is False


def test_capacity_follows_the_objective_and_the_flag_overrides_it() -> None:
    """`objective.eval_capacity` turns the audit on; an explicit flag wins over it in both directions."""
    on = ZTEConfig(objective=ObjectiveConfig(eval_capacity=True))
    assert options_from_args(_args(), on).capacity is True
    assert options_from_args(_args(capacity=False), on).capacity is False
    assert options_from_args(_args(capacity=True), ZTEConfig()).capacity is True


def test_capacity_settings_fall_back_to_the_checkpoint_decoder_config() -> None:
    """An unset flag reads the value the run was configured with, not a CLI constant."""
    config = ZTEConfig(
        decoder=DecoderConfig(capacity_ks=(2, 3, 5), capacity_alpha=0.01, capacity_n_perm=77, capacity_score='both')
    )
    options = options_from_args(_args(), config)
    assert options.capacity_ks == (2, 3, 5)
    assert options.capacity_alpha == 0.01
    assert options.capacity_n_perm == 77
    assert options.capacity_score == 'both'


def test_capacity_flags_override_the_checkpoint_decoder_config() -> None:
    """A sweep re-reads an old checkpoint under new sizes and a new alpha without retraining it."""
    config = ZTEConfig(decoder=DecoderConfig(capacity_ks=(2, 4), capacity_alpha=0.05, capacity_n_perm=2000))
    options = options_from_args(
        _args(capacity_ks='2, 4,8', capacity_alpha=0.001, capacity_n_perm=64),
        config,
    )
    assert options.capacity_ks == (2, 4, 8)
    assert options.capacity_alpha == 0.001
    assert options.capacity_n_perm == 64


def test_decode_parser_accepts_the_capacity_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """The four capacity flags parse and reach `DecodeOptions` intact."""
    monkeypatch.setattr(
        'sys.argv',
        [
            'zte-decode',
            '--ckpt',
            'best.pt',
            '--synthetic',
            '--capacity',
            '--capacity-ks',
            '2,4,8,16',
            '--capacity-alpha',
            '0.01',
            '--capacity-n-perm',
            '500',
        ],
    )
    options = options_from_args(parse_arguments(), ZTEConfig())
    assert options.capacity is True
    assert options.capacity_ks == (2, 4, 8, 16)
    assert (options.capacity_alpha, options.capacity_n_perm) == (0.01, 500)


def test_decode_parser_can_switch_the_capacity_audit_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--no-capacity` beats a config that asked for the audit, so a smoke run can skip its cost."""
    monkeypatch.setattr('sys.argv', ['zte-decode', '--ckpt', 'best.pt', '--synthetic', '--no-capacity'])
    config = ZTEConfig(objective=ObjectiveConfig(eval_capacity=True))
    assert options_from_args(parse_arguments(), config).capacity is False


# --------------------------------------------------------------------------- #
# the shared gallery pass
# --------------------------------------------------------------------------- #

_Z_DIM: Final[int] = 32
"""Width of the conditioning vectors the tiny bridge reads."""

_CAPACITY_OPTIONS: Final[dict[str, Any]] = {
    'controls': (),
    'oracle': False,
    'batch_size': 4,
    'n_perm': 8,
    'n_boot': 32,
    'capacity_ks': (2, 4, 8),
    'capacity_n_perm': 32,
}
"""Decode options small enough to run offline, with generation's own machinery turned down to nothing."""


def _capacity_config(dataset: ZuCoDataset, *, rescore_pmi: bool = False) -> ZTEConfig:
    """A held-out-subject-and-stimulus decoder run over the synthetic tree, with the offline tiny LM."""
    return ZTEConfig(
        dataset=dataset.config,
        model=ModelConfig(embed_dim=_Z_DIM, hidden_dim=_Z_DIM, n_layers=1, n_heads=2, projection_hidden=_Z_DIM),
        objective=ObjectiveConfig(name='decode'),
        train=TrainConfig(
            split='by_subject_and_stimulus',
            val_fraction=0.15,
            test_fraction=0.2,
            loso_holdout_subject='ZDM',
            seed=42,
            mode='decoder',
        ),
        decoder=DecoderConfig(
            lm_source='tiny',
            tokenizer_source='tiny',
            max_target_tokens=24,
            max_new_tokens=4,
            prefix_slots=2,
            bottleneck=8,
            rescore_chunk=4,
            rescore_pmi=rescore_pmi,
        ),
    )


def _tiny_decoder(dataset: ZuCoDataset, config: ZTEConfig) -> ZTEDecoder:
    """An untrained decoder over the offline tiny LM: the wiring under test needs shapes, not learned weights."""
    assert dataset.features is not None
    torch.manual_seed(0)
    model = build_model(config.model, in_dim=int(dataset.features.shape[1]))
    lm = build_lm(config.decoder, encoder=model)
    bridge, _ = build_bridge(config.decoder, _Z_DIM, _Z_DIM, lm.hidden_dim)

    return ZTEDecoder(
        model=model,
        config=config,
        decoder_config=config.decoder,
        bridge=bridge,
        lm=lm,
        gap=GapCorrector(_Z_DIM, mode='none'),
        device=resolve_device('cpu'),
    )


def _decode(dataset: ZuCoDataset, config: ZTEConfig, out_dir: Path | None, **overrides: Any) -> dict[str, Any]:
    """Runs one whole decode evaluation on the held-out cell."""
    return decode_evaluation(
        _tiny_decoder(dataset, config),
        dataset,
        split_indices(dataset, config, 'test'),
        split='test',
        config=config,
        options=DecodeOptions(**(_CAPACITY_OPTIONS | overrides)),
        out_dir=out_dir,
    )


def test_the_shared_gallery_pass_reproduces_the_rescoring_scores(small_dataset: ZuCoDataset) -> None:
    """Retrieval and the capacity audit read one LM pass, so that pass has to equal the rescoring call it replaced.

    Note:
        This is the whole cost argument for the audit -- a menu is a column slice of the rescoring matrix, not a
        rescoring run of its own -- and it is also what keeps the published retrieval numbers from moving.
    """
    config = _capacity_config(small_dataset)
    decoder = _tiny_decoder(small_dataset, config)
    indices = split_indices(small_dataset, config, 'test')
    readings = decoder.conditioning(small_dataset, indices, 4)
    gallery = ZuCoTorchDataset(small_dataset, subject_vocab=build_subject_vocab(small_dataset))
    texts = gallery.ordered_texts()

    bundle = gallery_scores(decoder, readings, texts, gallery_n_words=np.ones(len(texts)), batch_size=4)

    assert np.array_equal(bundle.raw, decoder.rescore(readings, texts, batch_size=4, pmi=False))
    assert np.array_equal(bundle.pmi, bundle.raw - decoder.null_rescore(texts)[None, :])


def test_retrieval_still_ranks_with_the_family_the_checkpoint_asked_for(small_dataset: ZuCoDataset) -> None:
    """The shared pass carries both families, so retrieval has to keep picking the one `rescore_pmi` names."""
    raw = _decode(small_dataset, _capacity_config(small_dataset), None)['rescoring']
    pmi = _decode(small_dataset, _capacity_config(small_dataset, rescore_pmi=True), None)['rescoring']

    assert raw.get('score') != 'pmi'
    assert 'pmi_vs_raw' not in raw
    assert pmi['score'] == 'pmi'

    # Ranking with the raw matrix under both labels would make this comparison a comparison of a matrix with
    # itself, so a zero delta is exactly the failure mode this pins.
    comparison = pmi['pmi_vs_raw']
    assert comparison['pmi_rank_percentile'] != comparison['raw_rank_percentile']
    assert comparison['rank_percentile_delta']['point'] != 0.0


def test_the_capacity_audit_does_not_move_the_retrieval_numbers(small_dataset: ZuCoDataset) -> None:
    """Turning the audit on must change no published number, or the two readouts are not reading one pass."""
    config = _capacity_config(small_dataset)
    without = _decode(small_dataset, config, None, capacity=False)
    with_audit = _decode(small_dataset, config, None, capacity=True)

    assert without['capacity'] is None
    assert with_audit['capacity'] is not None
    assert json.dumps(without['rescoring'], sort_keys=True, default=str) == json.dumps(
        with_audit['rescoring'], sort_keys=True, default=str
    )


def test_a_capacity_run_writes_its_block_and_its_artifact(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """The audit returns a report the run can embed and drops `capacity.json` beside the generation artifacts."""
    config = _capacity_config(small_dataset)
    out_dir = tmp_path / 'evaluation'

    result = _decode(small_dataset, config, out_dir, capacity=True)

    capacity = result['capacity']
    assert capacity['readout'] == 'menu selection'
    assert capacity['tie_policy'] == 'ties lose'
    assert (capacity['split_strategy'], capacity['split_cell']) == ('by_subject_and_stimulus', 'test')

    written = json.loads((out_dir / 'capacity.json').read_text(encoding='utf-8'))
    assert written['capacity'] == json.loads(json.dumps(capacity, default=str))
    assert written['provenance']['split'] == 'test'


def test_a_capacity_run_without_it_writes_no_artifact(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """No audit means no `capacity.json`, so a stale file can never be read as this run's certification."""
    out_dir = tmp_path / 'evaluation'

    _decode(small_dataset, _capacity_config(small_dataset), out_dir, capacity=False)

    assert (out_dir / 'generation.json').exists()
    assert not (out_dir / 'capacity.json').exists()


def test_the_capacity_audit_builds_the_length_control_from_the_training_split(small_dataset: ZuCoDataset) -> None:
    """`length_only` is the arm the 5.14-bit length confound demands; without it nothing may certify."""
    config = _capacity_config(small_dataset)

    capacity = _decode(small_dataset, config, None, capacity=True)['capacity']

    assert 'length_only' in capacity['provenance']['arms_present']
    assert {'model', 'shuffled_eeg', 'mismatch'} <= set(capacity['provenance']['arms_present'])


def test_an_uncertified_capacity_names_the_clauses_it_failed(small_dataset: ZuCoDataset) -> None:
    """An untrained bridge certifies nothing, and the report has to say so with the failing clauses named."""
    config = _capacity_config(small_dataset)

    capacity = _decode(small_dataset, config, None, capacity=True)['capacity']

    assert capacity['certified_k'] is None
    assert capacity['verdict']['capacity_certified'] is False
    assert capacity['verdict']['capacity_bits'] is None
    assert capacity['verdict']['reason']


def test_the_capacity_report_names_the_menu_sizes_the_gallery_cannot_fill(small_dataset: ZuCoDataset) -> None:
    """Exact word-count pools run out of candidates long before K = 64, and a dropped size would read as a pass."""
    config = _capacity_config(small_dataset)

    capacity = _decode(small_dataset, config, None, capacity=True, capacity_ks=(2, 4, 8, 16, 32, 64))['capacity']

    headline = capacity['headline']
    block = capacity['scores'][headline['score']][headline['flavor']]
    assert sorted(block['ks_feasible'] + block['ks_unreachable']) == [2, 4, 8, 16, 32, 64]
    assert block['ks_unreachable'], 'a 12-stimulus gallery cannot fill a 64-way exact-length menu'


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


def test_a_per_seed_options_clone_keeps_every_other_setting() -> None:
    """The seed sweep re-runs the control layer at each seed, so only the seed may differ between passes.

    Note:
        `DecodeOptions` is a slots dataclass and therefore has no `__dict__`; cloning it by unpacking one would
        raise at the first extra seed, long after the expensive decode had already run.
    """
    base = DecodeOptions(controls=('mean_prefix', 'length_only'), n_perm=17, n_boot=23, length_tol=3, seeds=(1, 2))

    clone = replace(base, seed=7, seeds=())

    assert (clone.seed, clone.seeds) == (7, ())
    assert clone.controls == base.controls
    assert (clone.n_perm, clone.n_boot, clone.length_tol) == (17, 23, 3)
    assert clone.within_task_pools == base.within_task_pools


def test_the_new_controls_are_accepted_by_name() -> None:
    """`shuffled_z` and `length_only` are pre-registered controls, not free-text, so a typo has to fail loudly."""
    assert {'shuffled_z', 'length_only'} <= set(CONTROLS)

    config = ZTEConfig(decoder=DecoderConfig(generation_controls=('shuffled_z', 'length_only')))
    options = options_from_args(_args(), config)
    assert options.controls == ('shuffled_z', 'length_only')

    with pytest.raises(ValueError, match='unknown control'):
        options_from_args(_args(controls='length_only,not_a_control'), config)
