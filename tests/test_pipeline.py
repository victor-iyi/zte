"""End-to-end integration: train a tiny model, checkpoint, and extract embeddings."""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

from pathlib import Path

from zte.config import ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.inference.embed import ZTEEmbedder
from zte.training.metrics import linear_probe, noise_matched, retrieval_metrics
from zte.training.pipeline import run_training


def _tiny_config(ckpt_dir: Path) -> ZTEConfig:
    """Builds a fast CPU config for integration testing."""
    cfg = ZTEConfig()
    cfg.objective.name = 'skipgram'
    cfg.model.frontend = 'band_power_mlp'
    cfg.model.embed_dim = 48
    cfg.model.hidden_dim = 40
    cfg.model.n_layers = 2
    cfg.dataset.representation = 'band_power'
    cfg.train.epochs = 2
    cfg.train.batch_size = 8
    cfg.train.device = 'cpu'
    cfg.train.precision = 'fp32'
    cfg.train.split = 'by_sentence'
    cfg.train.ckpt_dir = str(ckpt_dir)
    cfg.train.log_every = 1
    return cfg


def test_train_then_extract(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """A short run produces checkpoints and usable word embeddings."""
    cfg = _tiny_config(tmp_path / 'ckpt')
    artifacts = run_training(cfg, small_dataset)
    assert len(artifacts.history['train_loss']) == 2
    assert (tmp_path / 'ckpt' / 'best.pt').is_file()
    assert (tmp_path / 'ckpt' / 'last.pt').is_file()

    embedder = ZTEEmbedder.from_checkpoint(tmp_path / 'ckpt' / 'best.pt', small_dataset)
    emb, meta = embedder.embed(small_dataset, level='word')
    assert emb.shape[0] == len(meta)
    assert emb.shape[1] == cfg.model.embed_dim
    # Every embedded word is a present (non-omitted) token.
    assert (meta['is_omitted'] == 0).all()

    out = embedder.export(emb, meta, tmp_path / 'emb.npz')
    assert out.is_file()


def test_sentence_embeddings_and_metrics(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """Sentence-level embedding + retrieval/probe metrics run end-to-end."""
    cfg = _tiny_config(tmp_path / 'ckpt2')
    artifacts = run_training(cfg, small_dataset)
    embedder = ZTEEmbedder(artifacts.trainer.model, cfg, artifacts.device)  # type: ignore[arg-type]
    emb, meta = embedder.embed(small_dataset, level='sentence')
    assert emb.shape[0] == len(meta) > 0

    metrics = retrieval_metrics(emb, emb)
    assert metrics['top1'] == 1.0  # self-retrieval is trivially perfect

    noise = noise_matched(emb)
    assert noise.shape == emb.shape
    probe = linear_probe(emb, meta['n_words'].to_numpy())
    assert 'score' in probe


def test_embed_new_signals_in_memory(synthetic_dir: Path, tmp_path: Path) -> None:
    """from_checkpoint (no dataset) restores the normaliser and embeds new EEG arrays.

    Brand-new EEG for an imagined-thought BCI carries no eye tracking, so the
    device-agnostic path trains and embeds EEG-only (band power without the gaze
    scalars).
    """
    import numpy as np

    from zte.config import DatasetConfig, MissingConfig
    from zte.data.features import flatten_band_power

    eeg_only = ZuCoDataset(
        DatasetConfig(
            root=str(synthetic_dir),
            tasks=('SR', 'NR'),
            representation='band_power',
            include_eye_tracking=False,
            missing=MissingConfig(method='mask_only'),
            cache_dir=str(tmp_path / 'eeg_only_cache'),
        )
    ).build(show_progress=False)

    cfg = _tiny_config(tmp_path / 'ckpt3')
    run_training(cfg, eeg_only)

    # Restore WITHOUT a dataset -> shapes and normaliser come from the checkpoint.
    embedder = ZTEEmbedder.from_checkpoint(tmp_path / 'ckpt3' / 'best.pt')
    assert embedder.normalizer is not None
    assert embedder.in_dim == len(eeg_only.feature_names)

    # New, un-normalised band-power token signals (as from a custom EEG pipeline).
    feats = flatten_band_power(eeg_only.band_power_raw)
    signals = feats[eeg_only.presence][:16]
    emb = embedder.embed_signals(band_power=signals, show_progress=False)
    assert emb.shape == (signals.shape[0], cfg.model.embed_dim)
    assert np.isfinite(emb).all()
