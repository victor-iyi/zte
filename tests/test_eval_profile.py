"""Evaluation profiles, and the per-block progress that lets a reclaimed evaluation resume where it stopped."""

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from zte.config import DatasetConfig, ZTEConfig
from zte.config.train import EvalProfile, TrainConfig
from zte.data.dataset import ZuCoDataset
from zte.data.synthetic import generate_synthetic_zuco
from zte.evaluation import metrics as M
from zte.evaluation import report as report_mod
from zte.evaluation.report import PARTIAL_FILE, SWEEP_SKIPPED, _EvalStages, evaluate_representation


@dataclass(slots=True, frozen=True, kw_only=True)
class EvalInputs:
    """The row-aligned arrays and metadata `evaluate_representation` reads."""

    word_emb: np.ndarray
    word_meta: pd.DataFrame
    raw_feats: np.ndarray
    sent_emb: np.ndarray
    sent_content_ids: np.ndarray
    sent_meta: pd.DataFrame


def _boom(*args: Any, **kwargs: Any) -> Any:
    """Stands in for a block that must not be reached."""
    raise RuntimeError('this block was recomputed')


def _same(left: Any, right: Any) -> bool:
    """Whether two metric blocks hold identical numbers, NaN included -- which `==` never matches."""
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(right, sort_keys=True, default=str)


@pytest.fixture(scope='module')
def inputs(tmp_path_factory: pytest.TempPathFactory) -> EvalInputs:
    """Synthetic ZuCo behind a fixed random projection, so every test in this module evaluates the same numbers."""
    root = tmp_path_factory.mktemp('zuco_eval_profile')
    generate_synthetic_zuco(root, subjects=('ZAB', 'ZDM'), tasks=('SR', 'NR'), n_sentences=6, show_progress=False)
    config = DatasetConfig(
        root=str(root),
        tasks=('SR', 'NR'),
        representation='band_power',
        cache_dir=str(root / 'cache'),
    )
    dataset = ZuCoDataset(config).build(show_progress=False)

    present = np.ones(len(dataset.words), dtype=bool) if dataset.presence is None else dataset.presence
    word_meta = dataset.words.loc[present].reset_index(drop=True)
    raw_feats = np.asarray(dataset.features, dtype=np.float32)[present]

    # A fixed projection stands in for a trained encoder: a profile decides which blocks run, not what was learned.
    rng = np.random.default_rng(0)
    word_emb = (raw_feats @ rng.normal(size=(raw_feats.shape[1], 16)).astype(np.float32)).astype(np.float32)

    keys = ['subject', 'task', 'sentence_idx']
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for (subject, task, sentence_idx), group in word_meta.groupby(keys, sort=False):
        rows.append({'subject': subject, 'task': task, 'sentence_idx': sentence_idx})
        vectors.append(word_emb[group.index.to_numpy()].mean(axis=0))

    columns = [*keys, 'text', *[c for c in ('category',) if c in dataset.sentences]]
    sent_meta = pd.DataFrame(rows).merge(dataset.sentences[columns], on=keys, how='left')

    return EvalInputs(
        word_emb=word_emb,
        word_meta=word_meta,
        raw_feats=raw_feats,
        sent_emb=np.stack(vectors).astype(np.float32),
        sent_content_ids=pd.factorize(sent_meta['text'])[0],
        sent_meta=sent_meta,
    )


def _config(profile: EvalProfile) -> ZTEConfig:
    """A config whose optional word-retrieval blocks are on, so a profile that drops one is visible."""
    config = ZTEConfig()
    config.train.eval_profile = profile
    config.objective.eval_seen_novel = True
    config.objective.eval_freq_matched = True

    return config


def _evaluate(inputs: EvalInputs, out: Path, profile: EvalProfile) -> dict[str, Any]:
    """Runs the evaluation over the fixture arrays under one profile."""
    return evaluate_representation(
        inputs.word_emb,
        inputs.word_meta,
        inputs.raw_feats,
        inputs.sent_emb,
        inputs.sent_content_ids,
        out_dir=out,
        run_name='eval-profile',
        sent_meta=inputs.sent_meta,
        config=_config(profile),
        train_vocab=set(inputs.word_meta['word'].astype(str).iloc[::2]),
    )


def _interrupted(inputs: EvalInputs, out: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Evaluates until the honesty block, the way a reclaimed machine stops one, and returns the partial file."""
    monkeypatch.setattr(report_mod, '_honesty_block', _boom)
    with pytest.raises(RuntimeError):
        _evaluate(inputs, out, 'sweep')
    monkeypatch.undo()

    return json.loads((out / PARTIAL_FILE).read_text(encoding='utf-8'))


def test_full_is_the_default_profile_and_computes_every_block(inputs: EvalInputs, tmp_path: Path) -> None:
    """A run that names no profile evaluates everything and says so in its metrics."""
    assert TrainConfig().eval_profile == 'full'
    assert ZTEConfig().train.eval_profile == 'full'

    out = tmp_path / 'full'
    metrics = _evaluate(inputs, out, 'full')

    assert metrics['eval_profile'] == 'full'
    assert 'eval_skipped' not in metrics
    assert metrics['analogy'] and metrics['neurons'] and metrics['emergence']
    assert metrics['word_retrieval_by_novelty'] and metrics['word_retrieval_freq_matched']
    assert metrics['figures'] and 'interactive' in metrics
    assert (out / 'metrics.json').is_file() and (out / 'neurons.json').is_file()
    assert (out / 'report.md').is_file() and (out / 'comparison.csv').is_file()
    # The progress file exists only while an evaluation is in flight.
    assert not (out / PARTIAL_FILE).exists()


def test_sweep_drops_the_expensive_blocks_and_declares_which(inputs: EvalInputs, tmp_path: Path) -> None:
    """The sweep profile omits everything no headline is read from, and records that it did."""
    out = tmp_path / 'sweep'
    full = _evaluate(inputs, tmp_path / 'full', 'full')
    sweep = _evaluate(inputs, out, 'sweep')

    assert sweep['eval_profile'] == 'sweep'
    assert sweep['eval_skipped'] == list(SWEEP_SKIPPED)
    assert sweep['analogy'] == {} and sweep['neurons'] == {} and sweep['emergence'] == {}
    assert sweep['word_retrieval_by_novelty'] == {} and sweep['word_retrieval_freq_matched'] is None
    assert sweep['figures'] == []
    assert 'interactive' not in sweep and 'neuron_atlas' not in sweep
    assert not (out / 'neurons.json').exists() and not (out / 'interactive').exists()
    assert not list((out / 'figures').iterdir())

    # What is kept is not a cheaper approximation of the headline: it is the number the full profile computed.
    assert _same(sweep['embedding_health'], full['embedding_health'])
    assert _same(sweep['sentence_retrieval'], full['sentence_retrieval'])
    assert sweep['scoreboard'] is not None and _same(sweep['scoreboard'], full['scoreboard'])
    assert _same(sweep['honesty']['retrieval_permutation'], full['honesty']['retrieval_permutation'])

    stored = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    assert stored['eval_profile'] == 'sweep' and stored['eval_skipped'] == list(SWEEP_SKIPPED)
    assert 'Sweep evaluation profile' in (out / 'report.md').read_text(encoding='utf-8')


def test_a_recorded_block_is_read_back_instead_of_recomputed(
    inputs: EvalInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-entering an interrupted evaluation costs the block that was in flight, not the ones already recorded."""
    out = tmp_path / 'run'
    partial = _interrupted(inputs, out, monkeypatch)

    assert {'probe_comparison', 'embedding_health', 'sentence_retrieval', 'breakdown_words'} <= set(partial['blocks'])
    assert not (out / 'metrics.json').exists()

    # Recomputing either block is made impossible, so finishing proves neither was recomputed.
    monkeypatch.setattr(report_mod, '_adjacency_pairs', _boom)
    monkeypatch.setattr(report_mod, 'stratified_report', _boom)
    metrics = _evaluate(inputs, out, 'sweep')

    assert _same(metrics['embedding_health'], partial['blocks']['embedding_health'])
    assert _same(metrics['breakdown_words'], partial['blocks']['breakdown_words'])
    assert (out / 'metrics.json').is_file()
    assert not (out / PARTIAL_FILE).exists()


def test_deleting_the_partial_file_forces_a_full_recompute(
    inputs: EvalInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the progress file every block runs again, so a suspect evaluation can always be redone from scratch."""
    out = tmp_path / 'run'
    _interrupted(inputs, out, monkeypatch)
    (out / PARTIAL_FILE).unlink()

    monkeypatch.setattr(report_mod, '_adjacency_pairs', _boom)
    with pytest.raises(RuntimeError):
        _evaluate(inputs, out, 'sweep')


def test_a_partial_file_from_other_embeddings_is_never_reused(
    inputs: EvalInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded block is keyed to what produced it, so a re-run on new embeddings cannot inherit a stale number."""
    out = tmp_path / 'run'
    _interrupted(inputs, out, monkeypatch)

    monkeypatch.setattr(report_mod, '_adjacency_pairs', _boom)
    with pytest.raises(RuntimeError):
        _evaluate(replace(inputs, word_emb=inputs.word_emb + 1.0), out, 'sweep')


def test_an_unreadable_partial_file_is_discarded_rather_than_trusted(
    inputs: EvalInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A torn write costs the evaluation its progress, never its correctness."""
    out = tmp_path / 'run'
    _interrupted(inputs, out, monkeypatch)
    (out / PARTIAL_FILE).write_text('{"fingerprint": "abc", "blocks":', encoding='utf-8')

    monkeypatch.setattr(M, 'embedding_health', _boom)
    with pytest.raises(RuntimeError):
        _evaluate(inputs, out, 'sweep')


def test_the_block_progress_file_survives_the_machine_it_was_written_on(tmp_path: Path) -> None:
    """Evaluation is two thirds of a run and the run directory mirrors only once it has already returned.

    Note:
        So the one stage long enough to be interrupted is the one stage whose progress would die with a reclaimed
        Colab VM. A partial file that only ever exists on that VM protects nothing.
    """
    local, drive = tmp_path / 'run' / 'evaluation', tmp_path / 'drive' / 'run' / 'evaluation'
    local.mkdir(parents=True)

    stages = _EvalStages(local / PARTIAL_FILE, 'fp', 'sweep', drive / PARTIAL_FILE)
    stages.run('retrieval', lambda: {'top1': 0.25})

    assert (drive / PARTIAL_FILE).is_file(), 'the partial file never reached the durable copy'

    # The VM is reclaimed: the local disk is gone, the Drive copy is not.
    shutil.rmtree(tmp_path / 'run')
    local.mkdir(parents=True)
    resumed = _EvalStages(local / PARTIAL_FILE, 'fp', 'sweep', drive / PARTIAL_FILE)

    assert resumed.run('retrieval', lambda: pytest.fail('recomputed a block the mirror already carried')) == {
        'top1': 0.25
    }

    resumed.clear()
    assert not (drive / PARTIAL_FILE).exists(), 'a finished evaluation must leave no partial file behind'
