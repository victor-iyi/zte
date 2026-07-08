"""Tests for the new ZTE capabilities: regions, analogy, positional encodings,
eye-tracking toggle, categories, sources, interactive viz and TensorBoard."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from zte.config import DatasetConfig, MissingConfig, ModelConfig
from zte.data.categories import corpus_frequencies, length_band, sentence_categories
from zte.data.dataset import ZuCoDataset
from zte.data.regions import RegionMap, default_region_map, region_importance
from zte.data.schema import BANDS, N_CHANNELS
from zte.data.torch_dataset import SentenceSample, collate_sentences
from zte.evaluation.analogy import analogy_report, transfer_analogy
from zte.models.embedding import build_model

# --------------------------------------------------------------------------- #
# Positional encodings
# --------------------------------------------------------------------------- #


def _fake_batch(b: int = 4, max_len: int = 6, dim: int = 40) -> dict:
    """Builds a padded batch of band-power token sequences with an omitted word."""
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(b):
        length = int(rng.integers(3, max_len + 1))
        presence = torch.ones(length, dtype=torch.bool)
        presence[0] = False  # an omitted word
        samples.append(
            SentenceSample(
                features=torch.from_numpy(rng.standard_normal((length, dim)).astype('float32')),
                raw=None,
                presence=presence,
                subject=0,
                length=length,
            )
        )
    return collate_sentences(samples)


@pytest.mark.parametrize('pos', ['rope', 'sinusoidal', 'learned', 'alibi', 'none'])
def test_positional_encoding_forward(pos: str) -> None:
    """Every positional scheme runs contextual + causal forward without NaNs."""
    cfg = ModelConfig(
        frontend='band_power_mlp',
        embed_dim=32,
        hidden_dim=48,
        n_layers=2,
        n_heads=6,
        pos_encoding=pos,
    )
    model = build_model(cfg, in_dim=40).eval()
    batch = _fake_batch(dim=40)
    with torch.no_grad():
        for kwargs in (
            {'contextual': False},
            {'contextual': True},
            {'contextual': True, 'causal': True},
        ):
            out = model(batch, **kwargs)
            assert out.shape[-1] == 32
            assert not torch.isnan(out).any()
        assert not torch.isnan(model.embed_sentence(batch)).any()


def test_learned_encoding_has_position_table() -> None:
    """The learned scheme adds a position table; rope/alibi/none do not."""
    learned = build_model(
        ModelConfig(pos_encoding='learned', hidden_dim=32, embed_dim=32), in_dim=40
    )
    rope = build_model(ModelConfig(pos_encoding='rope', hidden_dim=32, embed_dim=32), in_dim=40)
    assert learned.pos_emb is not None
    assert rope.pos_emb is None


# --------------------------------------------------------------------------- #
# Brain regions
# --------------------------------------------------------------------------- #


def test_region_map_covers_all_channels() -> None:
    """The default map assigns every channel to exactly one region."""
    rmap = default_region_map(N_CHANNELS)
    assert rmap.n_channels == N_CHANNELS
    assert sum(rmap.region_sizes().values()) == N_CHANNELS
    assert set(np.unique(rmap.channel_region)) == set(range(rmap.n_regions))
    assert rmap.channels_in(rmap.names[0]).size > 0


def test_region_reduce_and_importance() -> None:
    """Region reduction has the right shape and importance normalises per target."""
    rng = np.random.default_rng(0)
    band_power = rng.standard_normal((200, len(BANDS), N_CHANNELS)).astype('float32')
    rmap = default_region_map(N_CHANNELS)
    reduced = rmap.reduce(band_power)
    assert reduced.shape == (200, len(BANDS), rmap.n_regions)

    targets = {'y': (rng.integers(0, 3, size=200), 'classification')}
    rows = region_importance(band_power, targets, region_map=rmap)  # type: ignore[arg-type]
    frame = pd.DataFrame(rows)
    assert set(frame['region']) == set(rmap.names)
    assert abs(frame[frame['target'] == 'y']['importance'].sum() - 1.0) < 1e-6


def test_region_map_from_csv(tmp_path: Path) -> None:
    """An exact montage CSV builds a non-approximate map."""
    csv = tmp_path / 'montage.csv'
    rows = ['channel,region']
    rows += [f'{c},{"left" if c < 50 else "right"}' for c in range(N_CHANNELS)]
    csv.write_text('\n'.join(rows), encoding='utf-8')
    rmap = RegionMap.from_csv(csv, N_CHANNELS)
    assert rmap.approximate is False
    assert set(rmap.names) == {'left', 'right'}


# --------------------------------------------------------------------------- #
# Vector arithmetic / analogy
# --------------------------------------------------------------------------- #


def test_transfer_analogy_recovers_additive_offset() -> None:
    """With embeddings = content + group offset, transfer arithmetic is near-perfect."""
    rng = np.random.default_rng(0)
    n_content, n_groups, dim = 40, 3, 16
    content = rng.standard_normal((n_content, dim)).astype('float32')
    offsets = rng.standard_normal((n_groups, dim)).astype('float32') * 3.0
    emb, groups, contents = [], [], []
    for g in range(n_groups):
        emb.append(content + offsets[g])
        groups.append(np.full(n_content, g))
        contents.append(np.arange(n_content))
    emb = np.concatenate(emb).astype('float32')
    groups = np.concatenate(groups)
    contents = np.concatenate(contents)

    out = transfer_analogy(emb, groups, contents, ks=(1, 5))
    assert out['n_queries'] > 0
    assert out['top1'] > 0.9  # offset is perfectly cancellable
    assert out['top1'] > out['chance_top1']


def test_analogy_report_structure() -> None:
    """analogy_report returns subject/task blocks and worked examples."""
    rng = np.random.default_rng(1)
    n_content, subjects = 15, ['ZAB', 'ZDM']
    content = rng.standard_normal((n_content, 12)).astype('float32')
    rows, meta_rows = [], []
    for s_i, subj in enumerate(subjects):
        rows.append(content + s_i)
        for w in range(n_content):
            meta_rows.append(
                {
                    'subject': subj,
                    'task': 'SR',
                    'sentence_idx': w // 3,
                    'word_idx': w % 3,
                    'word': f'w{w}',
                }
            )
    emb = np.concatenate(rows).astype('float32')
    report = analogy_report(emb, pd.DataFrame(meta_rows))
    assert 'subject_transfer' in report
    assert len(report['examples']) > 0


# --------------------------------------------------------------------------- #
# Eye-tracking toggle, categories, corpus frequency
# --------------------------------------------------------------------------- #


def test_eye_tracking_toggle_changes_width(synthetic_dir: Path, tmp_path: Path) -> None:
    """Including eye tracking appends exactly the configured gaze scalars."""

    def build(include: bool, cache: str) -> ZuCoDataset:
        cfg = DatasetConfig(
            root=str(synthetic_dir),
            tasks=('SR', 'NR'),
            representation='band_power',
            include_eye_tracking=include,
            missing=MissingConfig(method='mask_only'),
            cache_dir=str(tmp_path / cache),
        )
        return ZuCoDataset(cfg).build(show_progress=False)

    on, off = build(True, 'on'), build(False, 'off')
    n_et = len(on.config.eye_tracking_measures)
    assert on.features.shape[1] - off.features.shape[1] == n_et
    assert not np.isnan(on.features).any() and not np.isnan(off.features).any()
    assert any(name.startswith('ET::') for name in on.feature_names)


def test_categories_and_corpus_frequency() -> None:
    """Categories fall back to task; corpus frequency is in (0, 1]."""
    sentences = pd.DataFrame(
        {
            'subject': ['ZAB'] * 3,
            'task': ['SR', 'NR', 'SR'],
            'sentence_idx': [0, 1, 2],
            'n_words': [5, 12, 20],
            'text': ['a short one', 'a medium length sentence here now', 'x ' * 20],
        }
    )
    out = sentence_categories(sentences, root=None)
    assert list(out['length_band']) == ['short', 'medium', 'long']
    assert set(out['category']) <= {'SR', 'NR'}

    freq = corpus_frequencies(pd.Series(['the', 'the', 'rareword']))
    assert (freq > 0).all() and (freq <= 1).all()
    assert length_band(5) == 'short' and length_band(30) == 'long'


# --------------------------------------------------------------------------- #
# Source resolution
# --------------------------------------------------------------------------- #


def test_resolve_source_dir_and_zip(synthetic_dir: Path, tmp_path: Path) -> None:
    """resolve_source returns extracted dirs and unzips archives idempotently."""
    from zte.data.sources import resolve_source

    assert resolve_source(synthetic_dir) == synthetic_dir  # already extracted

    archive = shutil.make_archive(str(tmp_path / 'zuco'), 'zip', root_dir=str(synthetic_dir))
    extract_dir = tmp_path / 'extracted'
    resolved = resolve_source(archive, extract_dir=extract_dir)
    assert any(resolved.rglob('*.mat'))


# --------------------------------------------------------------------------- #
# Interactive viz + TensorBoard
# --------------------------------------------------------------------------- #


def test_interactive_explorer_writes_file(tmp_path: Path) -> None:
    """The explorer writes an HTML (Plotly) or a PNG fallback."""
    from zte.evaluation.interactive import embedding_explorer_html

    rng = np.random.default_rng(0)
    emb = rng.standard_normal((80, 16)).astype('float32')
    meta = pd.DataFrame(
        {
            'word': [f'w{i}' for i in range(80)],
            'subject': np.repeat(['ZAB', 'ZDM'], 40),
            'task': np.tile(['SR', 'NR'], 40),
        }
    )
    out = embedding_explorer_html(emb, meta, tmp_path / 'explorer.html')
    assert out.is_file() and out.suffix in {'.html', '.png'}


def test_tensorboard_reporter(tmp_path: Path) -> None:
    """The reporter logs scalars/embeddings and closes cleanly (no-op if unavailable)."""
    from zte.evaluation.tensorboard import TensorBoardReporter

    rng = np.random.default_rng(0)
    emb = rng.standard_normal((50, 12)).astype('float32')
    meta = pd.DataFrame({'subject': np.repeat(['ZAB', 'ZDM'], 25), 'word': ['x'] * 50})
    with TensorBoardReporter(tmp_path / 'tb') as tb:
        tb.log_scalars('m', {'a': 1.0, 'b': float('nan'), 'c': 'skip'})
        tb.log_embeddings(emb, meta)
        tb.log_embedding_stats(emb)
    if tb.enabled:
        assert any((tmp_path / 'tb').rglob('events*'))
