"""Tests for the decoder config surface: the round trip, path coercion and the committed experiment YAMLs."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, get_args

import pytest
import yaml

from zte.config import (
    Conditioning,
    DecoderConfig,
    GapCorrection,
    ObjectiveConfig,
    ObjectiveName,
    SplitStrategy,
    TrainMode,
    ZTEConfig,
)
from zte.data.dataset import ZuCoDataset

_EXPERIMENTS: Path = Path(__file__).resolve().parents[1] / 'experiments'


def _committed_configs() -> list[Path]:
    """Returns every committed experiment YAML, so a schema change cannot silently orphan one."""
    return sorted(p for p in _EXPERIMENTS.rglob('*.yaml') if p.is_file())


# --------------------------------------------------------------------------- #
# the round trip
# --------------------------------------------------------------------------- #
def test_every_decoder_field_survives_a_yaml_round_trip(tmp_path: Path) -> None:
    """`ZTEConfig.from_dict` must rebuild the decoder section, or a run reloads under someone else's settings."""
    decoder = DecoderConfig(
        lm_source='tiny',
        lm_revision='abc123',
        tokenizer_source='tiny',
        lm_cache_dir='res/cache/lm',
        conditioning='pooled_plus_words',
        prefix_slots=4,
        word_slots=6,
        bottleneck=32,
        gap_correction='whiten',
        null_prefix_prob=0.25,
        cfg_weight=1.0,
        ground_weight=0.75,
        ground_negatives=5,
        max_target_tokens=48,
        max_new_tokens=24,
        beams=3,
        stage0_epochs=7,
        min_prefix_kl=0.11,
        clip_aux_weight=0.5,
        cache_embeddings=False,
        prompt_template='\nText: ',
        generation_controls=('null_prefix', 'mismatch'),
        n_permutations=321,
        rescore_gallery=False,
        length_tol=3,
    )
    path = ZTEConfig(decoder=decoder, run_name='round-trip').to_yaml(tmp_path / 'config.yaml')
    restored = ZTEConfig.from_yaml(path).decoder

    assert restored == decoder
    for field in dataclasses.fields(DecoderConfig):
        assert getattr(restored, field.name) == getattr(decoder, field.name), field.name


def test_generation_controls_come_back_as_a_tuple(tmp_path: Path) -> None:
    """YAML has no tuple, so the list it parses must be coerced back or the control set stops comparing equal."""
    path = ZTEConfig().to_yaml(tmp_path / 'config.yaml')
    assert isinstance(yaml.safe_load(path.read_text())['decoder']['generation_controls'], list)
    assert isinstance(ZTEConfig.from_yaml(path).decoder.generation_controls, tuple)


def test_a_config_written_without_a_decoder_block_keeps_the_defaults(tmp_path: Path) -> None:
    """A YAML that names no decoder must load, whether the section is absent altogether or written with no keys."""
    (tmp_path / 'absent.yaml').write_text('run_name: legacy\n', encoding='utf-8')
    (tmp_path / 'empty.yaml').write_text('run_name: legacy\ndecoder:\n', encoding='utf-8')
    for name in ('absent.yaml', 'empty.yaml'):
        config = ZTEConfig.from_yaml(tmp_path / name)
        assert config.decoder == DecoderConfig()
        assert config.run_name == 'legacy'


def test_every_section_of_the_config_round_trips(tmp_path: Path) -> None:
    """The section list is derived from the dataclass fields, so no section can be dropped by omission."""
    config = ZTEConfig()
    config.train.mode = 'joint'
    config.train.encoder_ckpt = 'res/experiments/src/checkpoints/best.pt'
    config.objective.eval_generation = True
    config.dataset.tasks = ('SR', 'NR')
    restored = ZTEConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()
    assert restored.dataset.tasks == ('SR', 'NR')


# --------------------------------------------------------------------------- #
# path coercion
# --------------------------------------------------------------------------- #
def test_path_fields_are_stored_as_strings() -> None:
    """A `Path` assigned after construction would break YAML, JSON and the checkpoint payload alike."""
    config = ZTEConfig()
    # The annotation says `str`; argparse `type=Path` and `resolve_data_root` hand it a `Path` anyway.
    config.decoder.lm_cache_dir = Path('res/cache/lm')  # type: ignore[assignment]
    config.train.encoder_ckpt = Path('res/experiments/src/checkpoints/best.pt')  # type: ignore[assignment]
    assert config.decoder.lm_cache_dir == 'res/cache/lm'
    assert config.train.encoder_ckpt == 'res/experiments/src/checkpoints/best.pt'
    assert yaml.safe_dump(config.to_dict())  # would raise on a PosixPath


def test_the_declared_path_fields_are_the_ones_that_hold_paths() -> None:
    """A path field left out of `_PATH_FIELDS` is coerced nowhere, so the two lists must agree."""
    assert DecoderConfig._PATH_FIELDS == ('lm_cache_dir',)  # noqa: SLF001
    assert 'encoder_ckpt' in ZTEConfig().train._PATH_FIELDS  # noqa: SLF001


# --------------------------------------------------------------------------- #
# the type surface
# --------------------------------------------------------------------------- #
def test_the_new_literals_carry_the_decoder_options() -> None:
    """The config surface the CLI and the objective registry branch on is a closed set, checked here."""
    assert get_args(TrainMode.__value__) == ('encoder', 'decoder', 'joint')
    assert get_args(Conditioning.__value__) == ('pooled', 'pooled_plus_words')
    assert get_args(GapCorrection.__value__) == ('none', 'mean_scale', 'whiten')
    assert 'decode' in get_args(ObjectiveName.__value__)
    assert 'by_subject_and_stimulus' in get_args(SplitStrategy.__value__)


def test_the_evaluation_flags_default_to_the_honest_readouts() -> None:
    """Rescoring and length stratification are on by default; free generation is opt-in because it is the weak one."""
    objective = ObjectiveConfig()
    assert objective.eval_generation is False
    assert objective.eval_rescoring is True
    assert objective.eval_length_stratified is True


def test_a_config_missing_the_flags_reads_their_defaults(tmp_path: Path) -> None:
    """A checkpoint written before a key existed carries none, so every entry point must default rather than fail."""
    payload: dict[str, Any] = ZTEConfig().to_dict()
    for key in ('eval_generation', 'eval_rescoring', 'eval_length_stratified'):
        payload['objective'].pop(key)
    payload.pop('decoder')
    restored = ZTEConfig.from_dict(payload)
    assert getattr(restored.objective, 'eval_rescoring', True) is True
    assert restored.decoder.lm_source == DecoderConfig().lm_source


# --------------------------------------------------------------------------- #
# the committed experiments
# --------------------------------------------------------------------------- #
def test_every_committed_experiment_loads() -> None:
    """A schema edit that orphans a shipped YAML is a broken experiment, not a config detail."""
    paths = _committed_configs()
    assert len(paths) > 8, paths
    for path in paths:
        config = ZTEConfig.from_yaml(path)
        assert config.run_name
        assert ZTEConfig.from_dict(config.to_dict()).to_dict() == config.to_dict(), path


def test_the_decoder_arms_name_their_source_encoder() -> None:
    """A decoder or joint run without `train.encoder_ckpt` fails at `run_training`, so the YAMLs must carry one."""
    for path in _committed_configs():
        config = ZTEConfig.from_yaml(path)
        if config.train.mode == 'encoder':
            continue
        assert config.train.encoder_ckpt, path
        assert config.objective.name == 'decode', path
        if config.train.mode == 'joint':
            assert config.decoder.cache_embeddings is False, path


def test_no_committed_experiment_asks_for_guidance() -> None:
    """`cfg_weight != 1.0` is rejected at generation time, so shipping one would be an unrunnable arm."""
    for path in _committed_configs():
        assert ZTEConfig.from_yaml(path).decoder.cfg_weight == 1.0, path


@pytest.mark.parametrize('fraction', [0.0, -1.0])
def test_the_four_cell_split_needs_a_test_fraction(
    fraction: float, small_dataset: ZuCoDataset
) -> None:
    """The unseen-subject x unseen-stimulus cell is the point of the strategy; an empty one is refused loudly."""
    with pytest.raises(ValueError, match='test_fraction'):
        small_dataset.split(
            'by_subject_and_stimulus',
            val_fraction=0.1,
            test_fraction=fraction,
            holdout_subject=str(small_dataset.words['subject'].iloc[0]),
            seed=0,
        )
