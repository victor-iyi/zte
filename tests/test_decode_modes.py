"""Tests for the prefix-decode objective and the encoder / decoder / joint training modes."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

# The whole decoder test path is offline: `lm_source='tiny'` builds its LM and tokeniser locally.
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from zte.cli.decode import CONTROLS, DecodeOptions, decode_evaluation, split_indices
from zte.cli.run import warn_on_split_override
from zte.config import (
    DatasetConfig,
    DecoderConfig,
    MissingConfig,
    ModelConfig,
    ObjectiveConfig,
    SplitStrategy,
    TrainConfig,
    TrainMode,
    ZTEConfig,
)
from zte.data.dataset import ZuCoDataset
from zte.data.targets.tokens import build_target_tokens
from zte.data.torch_dataset import build_subject_vocab, make_dataloader
from zte.evaluation.report import _verdict
from zte.inference.decode import ZTEDecoder
from zte.models.embedding import build_model
from zte.models.objectives import build_objective
from zte.training import stages
from zte.training.checkpoint import CheckpointManager
from zte.training.init import file_sha256, load_encoder
from zte.training.pipeline import run_training
from zte.training.scheduler import build_scheduler
from zte.training.trainer import Trainer

_HOLDOUT = 'ZDM'
_SPLIT: dict[str, Any] = {
    'val_fraction': 0.15,
    'test_fraction': 0.2,
    'holdout_subject': _HOLDOUT,
    'seed': 42,
}

# The shipped arm cuts the encoder run and the decoder stage differently, so their train cells differ.
_ENCODER_STRATEGY: SplitStrategy = 'by_subject_loso'
_ENCODER_SPLIT: dict[str, Any] = {
    'val_fraction': 0.1,
    'test_fraction': 0.1,
    'holdout_subject': _HOLDOUT,
    'seed': 42,
}


@dataclass(slots=True)
class _Run:
    """The artifacts of one encoder run followed by one decoder run over it."""

    dataset_config: DatasetConfig
    decoder_config: ZTEConfig
    encoder_ckpt: Path
    decoder_ckpt: Path
    encoder_extra: dict[str, Any]
    encoder_weights: dict[str, torch.Tensor]


def _same_state(left: Any, right: Any) -> bool:
    """Compares two fitted-transform states, which nest dicts, lists of floats and scalars."""
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same_state(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_state(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, float) and isinstance(right, float):
        return bool(np.isclose(left, right, equal_nan=True))
    return bool(left == right)


def _weights(module: nn.Module) -> dict[str, torch.Tensor]:
    """Returns a detached copy of every parameter in `module`, keyed by name."""
    return {name: param.detach().clone() for name, param in module.named_parameters()}


def _any_moved(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    """Returns whether any parameter differs between two snapshots of one module."""
    return any(not torch.equal(before[name], after[name]) for name in before)


def _in_dim(dataset: ZuCoDataset) -> int:
    """Returns the band-power feature width of a built dataset."""
    assert dataset.features is not None
    return int(dataset.features.shape[1])


def _text_matrix_stub(dim: int) -> Callable[..., tuple[np.ndarray, int]]:
    """Returns a stand-in for the frozen text encoder, whose weights this offline suite cannot download.

    Args:
        dim (int): Width the oracle's embeddings must have to reach the bridge.

    Returns:
        Callable[..., tuple[np.ndarray, int]]: A `build_sentence_text_matrix` replacement.
    """

    def build(texts: list[str], source: str | None, **_: Any) -> tuple[np.ndarray, int]:
        matrix = np.random.default_rng(0).standard_normal((len(texts), dim)).astype(np.float32)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True), dim

    return build


def _dataset_config(synthetic_dir: Path, cache_dir: Path) -> DatasetConfig:
    """Returns a band-power dataset configuration over the synthetic tree."""
    return DatasetConfig(
        root=str(synthetic_dir),
        tasks=('SR', 'NR'),
        representation='band_power',
        missing=MissingConfig(method='mask_only'),
        cache_dir=str(cache_dir),
    )


def _config(
    dataset_config: DatasetConfig,
    ckpt_dir: Path,
    mode: TrainMode,
    encoder_ckpt: str | None = None,
    *,
    strategy: SplitStrategy = 'by_subject_and_stimulus',
    split: dict[str, Any] | None = None,
) -> ZTEConfig:
    """Builds a one-epoch run configuration in `mode`, cut with `split` under `strategy`.

    Args:
        dataset_config (DatasetConfig): The dataset the run reads.
        ckpt_dir (Path): Where the run writes its checkpoints.
        mode (TrainMode): Which of the three training modes to configure.
        encoder_ckpt (str | None, optional): Source encoder for a decoder or joint run. Defaults to None.
        strategy (SplitStrategy, optional): Split strategy. Defaults to 'by_subject_and_stimulus'.
        split (dict[str, Any] | None, optional): Split keyword arguments. Defaults to None, which uses
            the decoder stage's cut.

    Returns:
        ZTEConfig: The run configuration.
    """
    cut = split or _SPLIT
    return ZTEConfig(
        dataset=dataset_config,
        model=ModelConfig(embed_dim=32, hidden_dim=32, n_layers=1, n_heads=2, projection_hidden=32),
        objective=ObjectiveConfig(name='clip' if mode == 'encoder' else 'decode'),
        train=TrainConfig(
            epochs=1,
            batch_size=4,
            split=strategy,
            val_fraction=float(cut['val_fraction']),
            test_fraction=float(cut['test_fraction']),
            loso_holdout_subject=str(cut['holdout_subject']),
            ckpt_dir=str(ckpt_dir),
            seed=int(cut['seed']),
            num_workers=0,
            mode=mode,
            encoder_ckpt=encoder_ckpt,
        ),
        decoder=DecoderConfig(
            lm_source='tiny',
            tokenizer_source='tiny',
            max_target_tokens=32,
            max_new_tokens=6,
            prefix_slots=4,
            bottleneck=16,
            ground_negatives=2,
            stage0_epochs=1,
        ),
        run_name=mode,
    )


@pytest.fixture(scope='module')
def decoder_run(synthetic_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> _Run:
    """Trains a tiny encoder and then a decoder over it, once for the whole module."""
    root = tmp_path_factory.mktemp('decode')
    dataset_config = _dataset_config(synthetic_dir, root / 'cache')

    encoder_config = _config(dataset_config, root / 'enc', 'encoder', strategy=_ENCODER_STRATEGY, split=_ENCODER_SPLIT)
    run_training(encoder_config, ZuCoDataset(dataset_config).build(show_progress=False))
    encoder_ckpt = root / 'enc' / 'best.pt'
    payload = CheckpointManager.load(encoder_ckpt)

    decoder_config = _config(dataset_config, root / 'dec', 'decoder', str(encoder_ckpt))
    run_training(decoder_config, ZuCoDataset(dataset_config).build(show_progress=False))
    return _Run(
        dataset_config=dataset_config,
        decoder_config=decoder_config,
        encoder_ckpt=encoder_ckpt,
        decoder_ckpt=root / 'dec' / 'best.pt',
        encoder_extra=payload['extra'],
        encoder_weights={k: v.clone() for k, v in payload['model'].items()},
    )


@pytest.fixture(scope='module')
def decoder(decoder_run: _Run) -> ZTEDecoder:
    """Loads the trained decoder back from its checkpoint."""
    dataset = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    return ZTEDecoder.from_checkpoint(decoder_run.decoder_ckpt, dataset=dataset)


# --------------------------------------------------------------------------- #
# objective registry
# --------------------------------------------------------------------------- #
def test_build_objective_keeps_its_positional_signature(small_dataset: ZuCoDataset) -> None:
    """The two existing positional call shapes still build, so no test call site breaks."""
    model = build_model(ModelConfig(), in_dim=_in_dim(small_dataset))
    assert build_objective(ObjectiveConfig(name='skipgram'), model) is not None
    assert build_objective(ObjectiveConfig(name='masked'), model, feature_dim=40) is not None


def test_build_objective_requires_a_decoder_config(small_dataset: ZuCoDataset) -> None:
    """`decode` without a `DecoderConfig` fails loudly rather than building an unusable objective."""
    model = build_model(ModelConfig(), in_dim=_in_dim(small_dataset))
    with pytest.raises(ValueError, match='decoder_config'):
        build_objective(ObjectiveConfig(name='decode'), model)


# --------------------------------------------------------------------------- #
# the objective itself
# --------------------------------------------------------------------------- #
def _standalone_objective(dataset: ZuCoDataset, *, inherit_head: bool = True, **overrides: Any) -> tuple[Any, Any, Any]:
    """Builds a fully attached decode objective over a frozen encoder, mirroring the decoder-mode wiring."""
    model = build_model(
        ModelConfig(embed_dim=16, hidden_dim=16, n_layers=1, n_heads=2, projection_hidden=16),
        in_dim=_in_dim(dataset),
    )
    config = DecoderConfig(
        lm_source='tiny',
        tokenizer_source='tiny',
        max_target_tokens=24,
        prefix_slots=4,
        bottleneck=16,
        ground_negatives=2,
        **overrides,
    )
    objective: Any = build_objective(ObjectiveConfig(name='decode'), model, decoder_config=config)
    torch_ds = dataset.to_torch(subject_vocab=build_subject_vocab(dataset))
    texts = torch_ds.ordered_texts()
    targets = build_target_tokens(texts, 'tiny', max_length=24)
    objective.attach_tokens(torch.from_numpy(targets.ids), torch.from_numpy(targets.mask))
    rng = np.random.default_rng(0)
    if inherit_head:
        objective.attach_clip_head(torch.from_numpy(rng.standard_normal((24, 16)).astype(np.float32)))
    matrix = rng.standard_normal((len(texts), 24)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    objective.attach_text(torch.from_numpy(matrix))
    model.requires_grad_(False)
    model.eval()
    return model, objective, torch_ds


def test_compute_returns_finite_float_metrics(small_dataset: ZuCoDataset) -> None:
    """Every metric the trainer logs is a plain float, `prefix_kl` included."""
    model, objective, torch_ds = _standalone_objective(small_dataset)
    loader = make_dataloader(torch_ds, batch_size=4, shuffle=False, drop_last=True)
    loss, metrics = objective.compute(model, next(iter(loader)))
    assert torch.isfinite(loss)
    assert all(type(v) is float for v in metrics.values()), metrics
    for key in ('loss', 'ce', 'ground', 'prefix_kl', 'null_frac', 'n_valid'):
        assert key in metrics and np.isfinite(metrics[key]), key


def test_the_step_metric_reads_dependence_on_the_conditioning_vector(
    small_dataset: ZuCoDataset,
) -> None:
    """With the bridge unable to read `z` every prompt is the same prompt, which `prefix_kl` must report as 0."""
    model, objective, torch_ds = _standalone_objective(small_dataset)
    loader = make_dataloader(torch_ds, batch_size=4, shuffle=False, drop_last=True)
    with torch.no_grad():
        objective.bridge.to_bottleneck.weight.zero_()
    _, metrics = objective.compute(model, next(iter(loader)))
    assert metrics['prefix_kl'] == pytest.approx(0.0, abs=1e-9)
    assert metrics['null_kl'] > 0.0


def test_the_step_metric_stays_above_zero_while_the_loss_falls(
    small_dataset: ZuCoDataset,
) -> None:
    """A prompt built from a batch statistic still trains, so `prefix_kl` must stay above 0 at every step."""
    torch.manual_seed(0)
    model, objective, torch_ds = _standalone_objective(small_dataset)
    loader = make_dataloader(torch_ds, batch_size=8, shuffle=False, drop_last=True)
    batch = next(iter(loader))
    optimizer = torch.optim.AdamW(objective.bridge.parameters(), lr=1e-2)
    objective.train()

    ce: list[float] = []
    kl: list[float] = []
    for _ in range(20):
        loss, metrics = objective.compute(model, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ce.append(metrics['ce'])
        kl.append(metrics['prefix_kl'])

    # A prompt that reads one batch-wide vector drives the divergence to exactly 0 while the loss keeps falling.
    assert ce[-1] < ce[0]
    assert min(kl) > 0.0


def test_a_reading_gets_the_same_prompt_alone_as_it_does_beside_seven_others(
    small_dataset: ZuCoDataset,
) -> None:
    """A prompt read off a batch statistic moves when the rows around it change, which this refuses."""
    torch.manual_seed(0)
    model, objective, torch_ds = _standalone_objective(small_dataset)
    loader = make_dataloader(torch_ds, batch_size=8, shuffle=False, drop_last=True)
    batch = next(iter(loader))
    objective.eval()

    with torch.no_grad():
        _, prefix = objective.conditioning(model, batch)
        for row in range(int(prefix.shape[0])):
            single = {k: (v[row : row + 1] if torch.is_tensor(v) else v) for k, v in batch.items()}
            _, alone = objective.conditioning(model, single)
            assert torch.allclose(prefix[row], alone[0], atol=1e-5), row


def test_null_prefix_dropout_makes_the_loss_independent_of_the_brain(
    small_dataset: ZuCoDataset,
) -> None:
    """At `null_prefix_prob=1` every prefix is the learned null one, so the EEG cannot move the loss."""
    model, objective, torch_ds = _standalone_objective(small_dataset, null_prefix_prob=1.0)
    objective.train()
    loader = make_dataloader(torch_ds, batch_size=4, shuffle=False, drop_last=True)
    batch = next(iter(loader))
    other = dict(batch)
    other['features'] = torch.ones_like(batch['features'])
    torch.manual_seed(0)
    first, _ = objective.compute(model, batch)
    torch.manual_seed(0)
    second, _ = objective.compute(model, other)
    assert float(first.detach()) == pytest.approx(float(second.detach()), abs=1e-9)


def test_stage0_refuses_a_held_out_stimulus(small_dataset: ZuCoDataset) -> None:
    """Text-only pretraining rejects held-out references instead of quietly filtering them."""
    _, objective, _ = _standalone_objective(small_dataset)
    with pytest.raises(ValueError, match='held-out'):
        objective.pretrain_text([0, 1, 2], holdout_text_ids=[2], epochs=1)


def test_embedding_cache_refuses_joint_mode(small_dataset: ZuCoDataset) -> None:
    """A cached sentence vector is stale the moment the encoder starts moving, so joint mode refuses it."""
    _, objective, torch_ds = _standalone_objective(small_dataset)
    with pytest.raises(ValueError, match='cache_embeddings'):
        objective.attach_cache(torch_ds.n_readings, mode='joint')


def test_embedding_cache_serves_the_second_epoch(small_dataset: ZuCoDataset) -> None:
    """A frozen encoder is a pure function of the reading, so the second pass is all cache hits."""
    model, objective, torch_ds = _standalone_objective(small_dataset)
    objective.attach_cache(torch_ds.n_readings, mode='decoder')
    loader = make_dataloader(torch_ds, batch_size=4, shuffle=False, drop_last=True)
    batch = next(iter(loader))
    _, first = objective.compute(model, batch)
    _, second = objective.compute(model, batch)
    assert first['cache_hits'] == 0.0
    assert second['cache_hits'] == float(len(batch['reading_id']))


def test_a_learned_text_projection_trains_and_forbids_the_cache(small_dataset: ZuCoDataset) -> None:
    """Without a source projection the decoder learns its own, which must receive gradient and stay uncached."""
    model, objective, torch_ds = _standalone_objective(small_dataset, inherit_head=False)
    objective.attach_cache(torch_ds.n_readings, mode='decoder')
    assert objective.cache_z is None
    loader = make_dataloader(torch_ds, batch_size=4, shuffle=False, drop_last=True)
    loss, _ = objective.compute(model, next(iter(loader)))
    loss.backward()
    assert objective.clip_head.weight.grad is not None
    assert float(objective.clip_head.weight.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# the four-cell split
# --------------------------------------------------------------------------- #
def _cells(dataset: ZuCoDataset) -> dict[str, Any]:
    """Returns the four `by_subject_and_stimulus` cells as (subject set, stimulus set, index set) triples."""
    splits = dataset.split('by_subject_and_stimulus', **_SPLIT)
    words = dataset.words
    return {
        name: (
            set(words['subject'].to_numpy()[rows]),
            set(words['stimulus_key'].fillna('').to_numpy()[rows]),
            set(rows.tolist()),
        )
        for name, rows in splits.items()
    }


def test_the_honest_cell_shares_neither_a_subject_nor_a_stimulus(
    small_dataset: ZuCoDataset,
) -> None:
    """`test` is unseen subject reading unseen text; anything it shared with `train` would be the headline's leak."""
    cells = _cells(small_dataset)
    assert set(cells) == {'train', 'val', 'test', 'test_seen_stim'}
    train_subj, train_stim, train_rows = cells['train']
    test_subj, test_stim, test_rows = cells['test']
    assert not train_subj & test_subj
    assert not train_stim & test_stim
    assert not train_rows & test_rows


def test_the_diagnostic_cell_names_the_axis_it_does_not_generalise_over(
    small_dataset: ZuCoDataset,
) -> None:
    """`test_seen_stim` generalises over the brain only, so it must share stimuli with `train` and no subject."""
    cells = _cells(small_dataset)
    train_subj, train_stim, train_rows = cells['train']
    seen_subj, seen_stim, seen_rows = cells['test_seen_stim']
    assert not train_subj & seen_subj
    assert seen_stim <= train_stim
    assert not train_rows & seen_rows
    assert not seen_rows & cells['test'][2]


def test_the_model_selection_cell_keeps_the_training_subjects(small_dataset: ZuCoDataset) -> None:
    """`val` generalises over language alone, which is what makes it usable for model selection."""
    cells = _cells(small_dataset)
    assert cells['val'][0] <= cells['train'][0]
    assert not cells['val'][1] & cells['train'][1]


def test_the_stimulus_partition_does_not_depend_on_which_subject_is_held_out(
    small_dataset: ZuCoDataset,
) -> None:
    """Every fold must hold out the same texts, or the twelve folds cannot be pooled into one analysis."""
    first = small_dataset.split('by_subject_and_stimulus', **(_SPLIT | {'holdout_subject': 'ZAB'}))
    second = small_dataset.split('by_subject_and_stimulus', **(_SPLIT | {'holdout_subject': 'ZDM'}))
    keys = small_dataset.words['stimulus_key'].fillna('').to_numpy()
    assert set(keys[first['test']]) == set(keys[second['test']])


# --------------------------------------------------------------------------- #
# loading a frozen encoder
# --------------------------------------------------------------------------- #
def test_load_encoder_names_the_tensor_that_does_not_fit(decoder_run: _Run, tmp_path: Path) -> None:
    """A shape conflict is a wiring mistake; the message must say which tensor, not just that loading failed."""
    payload = CheckpointManager.load(decoder_run.encoder_ckpt)
    key = next(iter(payload['model']))
    payload['model'][key] = torch.zeros(1)
    path = tmp_path / 'bad.pt'
    torch.save(payload, path)

    config = _config(decoder_run.dataset_config, tmp_path / 'x', 'decoder', str(path))
    with pytest.raises(ValueError, match=key):
        load_encoder(path, config)


def test_load_encoder_strips_a_compiled_prefix(decoder_run: _Run, tmp_path: Path) -> None:
    """`torch.compile` renames every key, so a compiled run's checkpoint must still load into a plain model."""
    payload = CheckpointManager.load(decoder_run.encoder_ckpt)
    payload['model'] = {f'_orig_mod.{k}': v for k, v in payload['model'].items()}
    path = tmp_path / 'compiled.pt'
    torch.save(payload, path)

    config = _config(decoder_run.dataset_config, tmp_path / 'x', 'decoder', str(path))
    source = load_encoder(path, config)
    assert source.frozen is True
    assert not any(p.requires_grad for p in source.model.parameters())
    assert source.sha256 == file_sha256(path)
    assert source.provenance()['path'] == str(path)


def test_a_decoder_run_without_a_source_encoder_is_refused(decoder_run: _Run, tmp_path: Path) -> None:
    """Decoder mode is defined over an encoder trained elsewhere, so an unset checkpoint fails before any training."""
    config = _config(decoder_run.dataset_config, tmp_path / 'x', 'decoder', None)
    dataset = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    with pytest.raises(ValueError, match='encoder_ckpt'):
        run_training(config, dataset)


# --------------------------------------------------------------------------- #
# training modes
# --------------------------------------------------------------------------- #
def test_encoder_mode_is_reproducible_under_a_fixed_seed(synthetic_dir: Path, tmp_path: Path) -> None:
    """Two identical encoder runs at seed 42 must produce identical history, or no result can be reproduced.

    `Trainer` seeds itself, but weight initialisation happens in `run_training` before it is constructed, so the
    pipeline seeds too. Without that, `train.seed` never reaches the encoder's initial weights.
    """
    dataset_config = _dataset_config(synthetic_dir, tmp_path / 'cache')
    histories = []
    for run in ('a', 'b'):
        config = _config(dataset_config, tmp_path / run, 'encoder')
        config.train = replace(config.train, epochs=2)
        dataset = ZuCoDataset(dataset_config).build(show_progress=False)
        histories.append(run_training(config, dataset).history)

    assert histories[0] == histories[1]
    assert set(histories[0]) == {'lr', 'train_loss', 'val_loss'}
    assert len(histories[0]['train_loss']) == 2


def test_encoder_mode_builds_one_optimiser_group(small_dataset: ZuCoDataset) -> None:
    """With no bridge the optimiser holds a single group over model + objective at `train.lr`, and no stage flips."""
    config = _config(_dataset_config(Path('.'), Path('.')), Path('.'), 'encoder')
    model = build_model(ModelConfig(), in_dim=_in_dim(small_dataset))
    objective = build_objective(ObjectiveConfig(name='masked'), model, feature_dim=40)

    groups = stages.parameter_groups(model, objective, config)
    assert len(groups) == 1
    assert groups[0]['name'] == 'encoder'
    assert groups[0]['lr'] == config.train.lr
    assert len(groups[0]['params']) == len(list(model.parameters())) + len(list(objective.parameters()))
    assert stages.apply_stage(1, model, objective, config) is False


def test_encoder_extra_holds_subject_codes(decoder_run: _Run) -> None:
    """`extra['subject_vocab']` is the subject map, never the sentence-text map."""
    vocab = decoder_run.encoder_extra['subject_vocab']
    assert set(vocab) == {'ZAB', 'ZDM'}
    assert len(decoder_run.encoder_extra['text_vocab']) > len(vocab)


def test_decoder_run_leaves_the_encoder_byte_identical(decoder_run: _Run) -> None:
    """A frozen encoder must come out of a decoder run exactly as it went in."""
    payload = CheckpointManager.load(decoder_run.decoder_ckpt)
    for key, value in decoder_run.encoder_weights.items():
        assert torch.equal(value, payload['model'][key]), key


def test_decoder_run_restores_the_source_normalizer(decoder_run: _Run) -> None:
    """The decoder stage reuses the source run's fitted statistics rather than refitting on its own split.

    Refitting here does not fail, it just feeds the frozen encoder mis-scaled inputs, so the equality is the only
    thing standing between a working decoder and a silently worse one. The second half of the test is what makes
    the first half say anything: the two runs are cut differently, so a refit lands somewhere else.
    """
    payload = CheckpointManager.load(decoder_run.decoder_ckpt)
    assert _same_state(payload['extra']['normalizer'], decoder_run.encoder_extra['normalizer'])
    assert _same_state(payload['extra']['aligner'], decoder_run.encoder_extra['aligner'])

    refit = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    train = split_indices(refit, decoder_run.decoder_config, 'train')
    assert train is not None
    refit.refit_normalizer(train)
    assert refit.normalizer is not None
    assert not _same_state(refit.normalizer.state, decoder_run.encoder_extra['normalizer'])


def test_decoder_checkpoint_carries_its_payload(decoder_run: _Run) -> None:
    """Everything a `ZTEDecoder` needs travels in the checkpoint, and the frozen LM does not."""
    payload = CheckpointManager.load(decoder_run.decoder_ckpt)
    extra = payload['extra']
    for key in (
        'decoder_state',
        'decoder_config',
        'gap_correction',
        'encoder_source',
        'lm_provenance',
    ):
        assert key in extra, key
    assert not any(k.startswith('lm.') for k in extra['objective_state'])
    assert not any(k.startswith('lm.') for k in extra['decoder_state'])
    assert decoder_run.decoder_ckpt.stat().st_size < 1_000_000


def test_the_checkpoint_pins_the_frozen_language_model(decoder_run: _Run, decoder: ZTEDecoder) -> None:
    """A decode is reproducible only if the checkpoint names the weights and the tokeniser that wrote it.

    The record has to come off the LM object: a configuration names a source and a revision, but it cannot count
    parameters or fingerprint a tokeniser, so a config-only record pins nothing that could actually drift. The last
    assertion is the round trip -- what the checkpoint says matches what the LM a decode runs on reports.
    """
    record = CheckpointManager.load(decoder_run.decoder_ckpt)['extra']['lm_provenance']
    assert record['source'] == 'tiny' and record['revision'] is None
    assert record['vocab_size'] == 64 and record['hidden_size'] == 32
    assert record['n_parameters'] == 22_688
    assert record['tokenizer'] == 'tiny' and len(record['tokenizer_fingerprint']) == 16
    assert 'name_or_path' not in record
    assert record == decoder.lm.provenance()


def test_joint_mode_unfreezes_the_encoder_after_stage_a(decoder_run: _Run, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage A moves the bridge and nothing else; stage B adds the encoder at its own, smaller rate.

    The curriculum is only real if the weights obey it, so each epoch is snapshotted around the loop that
    applies the stage. `requires_grad` and the two learning rates are what the schedule is built out of, but
    neither says whether an epoch actually left the encoder alone.
    """
    config = _config(
        decoder_run.dataset_config,
        decoder_run.decoder_ckpt.parent.parent / 'joint',
        'joint',
        str(decoder_run.encoder_ckpt),
    )
    config.train = replace(config.train, epochs=2, stage_a_epochs=1, freeze_encoder=False)
    config.decoder = replace(config.decoder, cache_embeddings=False, stage0_epochs=0)

    moved: list[tuple[bool, bool]] = []
    real = Trainer._train_one_epoch  # noqa: SLF001 -- wrapped to snapshot each epoch's parameters

    def spy(trainer: Trainer, epoch: int) -> float:
        objective: Any = trainer.objective
        encoder_before = _weights(trainer.model)
        bridge_before = _weights(objective.bridge)
        loss = float(real(trainer, epoch))
        moved.append(
            (
                _any_moved(encoder_before, _weights(trainer.model)),
                _any_moved(bridge_before, _weights(objective.bridge)),
            )
        )
        return loss

    monkeypatch.setattr(Trainer, '_train_one_epoch', spy)
    dataset = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    artifacts = run_training(config, dataset)

    # One `(encoder moved, bridge moved)` per epoch: stage A, then stage B.
    assert moved == [(False, True), (True, True)]
    assert 'lr_bridge' in artifacts.history and 'lr_encoder' in artifacts.history
    assert artifacts.history['lr_bridge'][0] > artifacts.history['lr_encoder'][0]
    assert any(p.requires_grad for p in artifacts.trainer.model.parameters())


def test_a_joint_run_that_freezes_its_encoder_is_refused(decoder_run: _Run, tmp_path: Path) -> None:
    """`freeze_encoder` would hold the encoder frozen through the stage B a joint run exists to reach."""
    config = _config(decoder_run.dataset_config, tmp_path / 'x', 'joint', str(decoder_run.encoder_ckpt))
    assert config.train.freeze_encoder is True
    dataset = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    with pytest.raises(ValueError, match='freeze_encoder'):
        run_training(config, dataset)


def test_a_frozen_encoder_stays_in_eval_while_the_bridge_trains(decoder_run: _Run) -> None:
    """Dropout left live in the encoder would make the conditioning vector random even with no gradient reaching it."""
    config = _config(
        decoder_run.dataset_config,
        decoder_run.decoder_ckpt.parent.parent / 'evalmode',
        'decoder',
        str(decoder_run.encoder_ckpt),
    )
    dataset = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    artifacts = run_training(config, dataset)
    trainer = artifacts.trainer

    trainer._set_train_mode()  # noqa: SLF001
    objective: Any = trainer.objective
    assert trainer.model.training is False
    assert objective.lm.training is False
    assert objective.bridge.training is True


def test_the_shared_schedule_keeps_the_group_ratio(decoder_run: _Run, small_dataset: ZuCoDataset) -> None:
    """One `lr_lambda` is applied against each group's own `initial_lr`, so the bridge stays ten times the encoder."""
    config = _config(decoder_run.dataset_config, Path('.'), 'joint', str(decoder_run.encoder_ckpt))
    config.train = replace(config.train, freeze_encoder=False)
    model = build_model(ModelConfig(), in_dim=_in_dim(small_dataset))
    _, objective, _ = _standalone_objective(small_dataset)

    optimizer = torch.optim.AdamW(stages.parameter_groups(model, objective, config))
    names = [g['name'] for g in optimizer.param_groups]
    assert names == ['bridge', 'encoder']

    warmup = int(100 * config.train.warmup_ratio)
    scheduler = build_scheduler(optimizer, 100, warmup, config.train.scheduler)
    ratios = []
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        bridge_lr, encoder_lr = scheduler.get_last_lr()
        ratios.append(bridge_lr / max(encoder_lr, 1e-30))
    assert ratios == pytest.approx([1.0 / config.train.encoder_lr_scale] * 3)


def test_a_resumed_decoder_run_restores_its_bridge(decoder_run: _Run) -> None:
    """`FrozenLM.state_dict()` is empty, so only a non-strict load fits a decoder objective back together."""
    root = decoder_run.decoder_ckpt.parent.parent / 'resumed'
    config = _config(decoder_run.dataset_config, root, 'decoder', str(decoder_run.encoder_ckpt))
    dataset = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    first = run_training(config, dataset)
    trained: Any = first.trainer.objective
    saved = {k: v.clone() for k, v in trained.bridge.state_dict().items()}

    config.train = replace(config.train, epochs=2)
    resumed = run_training(config, ZuCoDataset(decoder_run.dataset_config).build(show_progress=False), resume=True)
    reloaded: Any = resumed.trainer.objective
    restored = reloaded.bridge.state_dict()
    assert set(restored) == set(saved)
    assert resumed.history['train_loss'][0] == first.history['train_loss'][0]
    assert len(resumed.history['train_loss']) == 2


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def test_conditioning_names_every_reading(decoder: ZTEDecoder, decoder_run: _Run) -> None:
    """The conditioning frame carries the reference, the subject and the word count each analysis needs."""
    dataset = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    splits = dataset.split('by_subject_and_stimulus', **_SPLIT)
    z, meta = decoder.conditioning(dataset, indices=splits['test'], batch_size=4)
    assert len(z) == len(meta) > 0
    assert set(meta['subject']) == {_HOLDOUT}
    for column in ('n_words', 'reading_id', 'text_id', 'stimulus_key', 'text'):
        assert column in meta.columns


def test_generation_rejects_guidance(decoder: ZTEDecoder) -> None:
    """Any guidance weight but 1.0 would make the headline decode and the null-prefix control different paths."""
    z = np.zeros((2, decoder.z_dim), dtype=np.float32)
    decoder.decoder_config = replace(decoder.decoder_config, cfg_weight=1.5)
    try:
        with pytest.raises(ValueError, match='cfg_weight'):
            decoder.generate(z)
    finally:
        decoder.decoder_config = replace(decoder.decoder_config, cfg_weight=1.0)


def test_controls_share_decode_path(decoder: ZTEDecoder, decoder_run: _Run, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every control, the oracle and the headline reach one `generate_from_prefix` with identical arguments.

    The controls are the only floor free generation has, so a control that ran through a different decode -- a
    different cap, a different beam width, a different scaffold -- would make the paired delta meaningless. The
    evaluation is driven end to end, which is what makes the assertion cover the path `zte-decode` takes.
    """
    dataset = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    indices = split_indices(dataset, decoder_run.decoder_config, 'test_seen_stim')
    assert indices is not None
    monkeypatch.setattr('zte.data.targets.text.build_sentence_text_matrix', _text_matrix_stub(decoder.z_dim))

    calls: list[tuple[Any, ...]] = []
    real = decoder.lm.generate_from_prefix

    def spy(
        prefix: torch.Tensor,
        scaffold_ids: torch.Tensor | None = None,
        max_new_tokens: int = 96,
        beams: int = 1,
    ) -> list[str]:
        calls.append((scaffold_ids, max_new_tokens, beams))
        return real(prefix, scaffold_ids, max_new_tokens, beams)

    monkeypatch.setattr(decoder.lm, 'generate_from_prefix', spy)
    result = decode_evaluation(
        decoder,
        dataset,
        indices,
        split='test_seen_stim',
        config=decoder_run.decoder_config,
        options=DecodeOptions(controls=CONTROLS, batch_size=16, n_perm=20, n_boot=50, rescore=False),
    )

    block = result['generation']
    ran = set(CONTROLS) - set(result['controls_unavailable'])
    assert result['controls_unavailable'] == {'phase': 'the encoder consumes no raw signal to destroy'}
    assert set(block['deltas']) == ran
    assert 'oracle' in block['absolute']

    # The headline, one decode per control that ran, and the oracle, each a single batch.
    assert len(calls) == len(ran) + 2
    assert len(set(calls)) == 1, calls
    assert calls[0] == (None, decoder.decoder_config.max_new_tokens, decoder.decoder_config.beams)


def test_rescore_ranks_the_whole_gallery(decoder: ZTEDecoder, decoder_run: _Run) -> None:
    """Gallery rescoring is retrieval: one finite score per (query, candidate) pair."""
    dataset = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    splits = dataset.split('by_subject_and_stimulus', **_SPLIT)
    z, meta = decoder.conditioning(dataset, indices=splits['test'], batch_size=4)
    gallery = sorted(set(meta['text']))
    scores = decoder.rescore(z, gallery, batch_size=4)
    assert scores.shape == (len(z), len(gallery))
    assert bool(np.isfinite(scores).all())


def test_prefix_influence_kl_is_finite_per_reading(decoder: ZTEDecoder, decoder_run: _Run) -> None:
    """The bridge-collapse detector reports one non-negative divergence per reading."""
    dataset = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    splits = dataset.split('by_subject_and_stimulus', **_SPLIT)
    z, _ = decoder.conditioning(dataset, indices=splits['test'], batch_size=4)
    kl = decoder.prefix_influence_kl(z)
    assert kl.shape == (len(z),)
    assert bool(np.isfinite(kl).all()) and float(kl.min()) >= -1e-6
    assert decoder.prefix_influence_kl(z[:1]).tolist() == [0.0]


def test_generate_conditions_the_language_model_on_z(decoder: ZTEDecoder, monkeypatch: pytest.MonkeyPatch) -> None:
    """The headline decode reaches the frozen LM carrying `bridge(z)` and nothing else.

    Every published hypothesis and every control decode comes out of `generate`, so a `generate` that quietly dropped
    its argument would make the brain optional while leaving every score, delta and permutation intact. The prefix the
    LM actually receives is compared tensor-for-tensor against the bridge's own output, which fails for any
    substitution regardless of whether the tiny LM happens to decode two prompts to the same string.
    """
    rng = np.random.default_rng(0)
    z = rng.standard_normal((5, decoder.z_dim)).astype(np.float32)
    seen: list[torch.Tensor] = []
    real = decoder.lm.generate_from_prefix

    def spy(
        prefix: torch.Tensor,
        scaffold_ids: torch.Tensor | None = None,
        max_new_tokens: int = 96,
        beams: int = 1,
    ) -> list[str]:
        seen.append(prefix.detach().clone())
        return real(prefix, scaffold_ids, max_new_tokens, beams)

    monkeypatch.setattr(decoder.lm, 'generate_from_prefix', spy)
    decoder.generate(z, batch_size=len(z))

    expected = decoder.prefix_from_z(z)
    assert len(seen) == 1
    assert torch.equal(seen[0], expected)
    # A bridge that reads z separates the rows, so the tensor comparison above has something to fail on.
    assert not torch.equal(expected, expected[:1].expand_as(expected))


def test_prefix_influence_kl_catches_a_bridge_that_cannot_read_z(decoder: ZTEDecoder) -> None:
    """The clause the generation verdict gates on, against the failure it exists to catch.

    A bridge whose input weight is zeroed emits one prompt for every reading, which is bridge collapse in its purest
    form. The KL against another reading's prefix is then exactly 0 and no floor can be met; the KL against the null
    prefix stays well above 0, because the null prefix is a free parameter that has nothing to do with the brain.
    """
    rng = np.random.default_rng(0)
    z = rng.standard_normal((8, decoder.z_dim)).astype(np.float32)
    saved = decoder.bridge.to_bottleneck.weight.detach().clone()
    try:
        with torch.no_grad():
            decoder.bridge.to_bottleneck.weight.zero_()
        prefix = decoder.prefix_from_z(z)
        assert torch.equal(prefix, prefix[:1].expand_as(prefix))
        assert float(decoder.prefix_influence_kl(z).max()) == pytest.approx(0.0, abs=1e-9)
        assert float(decoder.null_prefix_kl(z).mean()) > 0.0
    finally:
        with torch.no_grad():
            decoder.bridge.to_bottleneck.weight.copy_(saved)


def test_a_collapsed_bridge_is_refused_by_the_generation_gate(decoder: ZTEDecoder, decoder_run: _Run) -> None:
    """The collapse detector driven end to end: `zte-decode`'s own block, through the pre-registered gate.

    A bridge that cannot read `z` still decodes, still scores and still produces deltas, so the clause is the only
    thing standing between it and a headline. It is checked here against the block `decode_evaluation` writes,
    which is what makes the check cover the plumbing between the two as well as the clause itself.
    """
    dataset = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    indices = split_indices(dataset, decoder_run.decoder_config, 'test_seen_stim')
    assert indices is not None
    options = DecodeOptions(controls=('null_prefix',), oracle=False, batch_size=16, n_perm=20, n_boot=50, rescore=False)

    def decode() -> dict[str, Any]:
        block: dict[str, Any] = decode_evaluation(
            decoder,
            dataset,
            indices,
            split='test_seen_stim',
            config=decoder_run.decoder_config,
            options=options,
        )['generation']
        return block

    healthy = decode()
    floor = float(healthy['prefix_influence_kl']) / 2.0
    assert floor > 0.0
    passing = _verdict([], {}, {}, generation=healthy, min_prefix_kl=floor)
    assert passing['generation_clauses']['prefix_influences_output'] is True

    saved = decoder.bridge.to_bottleneck.weight.detach().clone()
    try:
        with torch.no_grad():
            decoder.bridge.to_bottleneck.weight.zero_()
        collapsed = decode()
    finally:
        with torch.no_grad():
            decoder.bridge.to_bottleneck.weight.copy_(saved)

    assert collapsed['applicable'] is True
    assert collapsed['prefix_influence_kl'] == pytest.approx(0.0, abs=1e-9)
    refused = _verdict([], {}, {}, generation=collapsed, min_prefix_kl=floor)
    assert refused['generation_clauses']['prefix_influences_output'] is False
    assert refused['generation_above_controls'] is False


def test_the_configured_prefix_kl_floor_refuses_a_real_collapsed_forward_pass(
    decoder: ZTEDecoder, decoder_run: _Run
) -> None:
    """`decoder.min_prefix_kl` against a divergence the bridge and the frozen LM really produced.

    The clause is otherwise only ever met by a number written for it or by a floor derived from the very run
    being judged, either of which passes whatever the decoder does. Here the floor is the one the run
    configured and the divergence came off a `ZTEDecoder` forward pass over real conditioning vectors.
    """
    dataset = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    indices = split_indices(dataset, decoder_run.decoder_config, 'test_seen_stim')
    assert indices is not None
    floor = decoder_run.decoder_config.decoder.min_prefix_kl
    assert floor == pytest.approx(DecoderConfig().min_prefix_kl)
    assert floor == pytest.approx(0.05)

    z, _ = decoder.conditioning(dataset, indices=indices, batch_size=4)
    assert float(decoder.prefix_influence_kl(z).mean()) > 0.0

    saved = decoder.bridge.to_bottleneck.weight.detach().clone()
    try:
        with torch.no_grad():
            decoder.bridge.to_bottleneck.weight.zero_()
        block: dict[str, Any] = decode_evaluation(
            decoder,
            dataset,
            indices,
            split='test_seen_stim',
            config=decoder_run.decoder_config,
            options=DecodeOptions(
                controls=('null_prefix',),
                oracle=False,
                batch_size=16,
                n_perm=20,
                n_boot=50,
                rescore=False,
            ),
        )['generation']
    finally:
        with torch.no_grad():
            decoder.bridge.to_bottleneck.weight.copy_(saved)

    measured = float(block['prefix_influence_kl'])
    assert measured == pytest.approx(0.0, abs=1e-12)
    assert measured < floor
    verdict = _verdict([], {}, {}, generation=block, min_prefix_kl=floor)
    assert verdict['generation_prefix_kl'] == pytest.approx(measured)
    assert verdict['generation_min_prefix_kl'] == pytest.approx(floor)
    assert verdict['generation_clauses']['prefix_influences_output'] is False
    assert verdict['generation_above_controls'] is False


def test_conditioning_puts_a_dataset_on_the_training_statistics(decoder: ZTEDecoder, decoder_run: _Run) -> None:
    """A freshly built dataset normalises itself, and a frozen encoder handed those rows quietly underperforms."""
    dataset = ZuCoDataset(decoder_run.dataset_config).build(show_progress=False)
    assert dataset.normalizer is not None
    self_fitted = dataset.normalizer.state
    trained = CheckpointManager.load(decoder_run.decoder_ckpt)['extra']['normalizer']
    assert not _same_state(self_fitted, trained)

    splits = dataset.split('by_subject_and_stimulus', **_SPLIT)
    decoder.conditioning(dataset, indices=splits['test'], batch_size=4)
    assert _same_state(dataset.normalizer.state, trained)


# --------------------------------------------------------------------------- #
# --loso-holdout must not silently replace the decoder's honest split
# --------------------------------------------------------------------------- #


def _guard(mode: TrainMode, requested: SplitStrategy, applied: SplitStrategy) -> tuple[bool, str]:
    """Runs the split guard and returns whether it fired and what it logged."""
    config = ZTEConfig()
    config.train.mode = mode
    config.train.split = applied

    # The `zte` logger does not propagate, so caplog never sees it; attach a handler for the call.
    messages: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())  # type: ignore[method-assign]
    logger = logging.getLogger('zte.cli.run')
    logger.addHandler(handler)
    try:
        fired = warn_on_split_override(config, requested)
    finally:
        logger.removeHandler(handler)

    return fired, ' '.join(messages)


def test_loso_holdout_over_the_honest_split_warns_a_decoder_run() -> None:
    """Swapping in `by_subject_loso` makes every held-out sentence a training sentence, and the run says so."""
    fired, message = _guard('decoder', 'by_subject_and_stimulus', 'by_subject_loso')
    assert fired
    assert 'honest_split' in message
    assert 'by_subject_and_stimulus' in message


def test_loso_holdout_is_silent_for_an_encoder_run() -> None:
    """`by_subject_loso` is the encoder's own north-star split, so the swap costs it nothing."""
    assert _guard('encoder', 'by_subject_and_stimulus', 'by_subject_loso') == (False, '')


def test_a_decoder_run_is_not_warned_about_the_split_it_asked_for() -> None:
    """The guard reports a split the flag replaced, never one the config deliberately named."""
    assert _guard('decoder', 'by_subject_loso', 'by_subject_loso') == (False, '')
