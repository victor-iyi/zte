# Training & inference guide

## Quick commands

```sh
# Skip-gram on a prepared bundle, 20 epochs, auto device.
uv run zte-train --bundle res/bundle --objective skipgram --epochs 20 --run-name sg

# Masked (data2vec) on raw EEG with a Conformer frontend, CUDA, bf16.
uv run zte-train --bundle res/bundle --objective masked \
    --frontend raw_conformer --representation raw --device cuda --precision bf16

# CPC with TensorBoard + Drive checkpoint backup.
uv run zte-train --bundle res/bundle --objective cpc --tensorboard \
    --drive-backup-dir /content/drive/MyDrive/ZTE/ckpts
```

## Config-driven runs

```yaml
# config.yaml
run_name: zte-skipgram-loso
dataset: { representation: band_power, normalize: zscore_channel, missing: { method: knn } }
model:   { frontend: band_power_mlp, embed_dim: 768, hidden_dim: 256, n_layers: 4, pool: attention }
objective: { name: skipgram, temperature: 0.07, context_window: 2 }
train:
  epochs: 50
  batch_size: 128
  lr: 0.0003
  split: by_subject_loso
  loso_holdout_subject: ZPH
  device: auto
  precision: auto
  tensorboard: true
```

```sh
uv run zte-train --bundle res/bundle --config config.yaml
```

CLI flags override YAML, so you can pin a base config and sweep one knob:
`uv run zte-train --config config.yaml --bundle b --objective cpc --lr 1e-4`.

## What a run produces

```sh
res/checkpoints/
├── best.pt            # lowest monitored loss
├── last.pt            # most recent
├── ckpt_epoch####.pt  # rotating (train.ckpt_keep_last)
├── config.yaml        # resolved config
├── training_curves.png
└── tb/<run_name>/     # TensorBoard (if enabled)
```

Each checkpoint embeds the model, optimiser, scheduler, AMP scaler, the full config, the fitted feature-normaliser state and the subject vocabulary, so inference reconstructs the exact pipeline.

## Inference

```python
from zte.inference.embed import ZTEEmbedder
embedder = ZTEEmbedder.from_checkpoint('res/checkpoints/best.pt', ds)

# Word-level "thought embeddings" (present words only) with aligned metadata.
emb, meta = embedder.embed(ds, level='word')
embedder.export(emb, meta, 'res/embeddings/word_embeddings.npz')

# Sentence-level (pooled) embeddings.
semb, smeta = embedder.embed(ds, level='sentence')

# Qualitative probe: nearest neighbours of a token in embedding space.
embedder.nearest_neighbors(emb, query_idx=0, k=5)
```

```sh
uv run zte-extract --ckpt res/checkpoints/best.pt --bundle res/bundle \
    --level word --probe-target log_freq --out res/embeddings/embeddings.npz
```

### Embedding brand-new EEG signals

`from_checkpoint` reads the input shapes and the fitted normaliser from the
checkpoint itself, so **no dataset is required** to embed new signals. Use
`embed` for new recordings packaged as `.mat` files, or `embed_signals` for EEG
already in memory (any `(N, F*C)` band-power array, or `(N, C, T)` raw windows for
a raw-Conformer model). Band-power inputs are normalised exactly as in training.

```python
from zte.inference.embed import ZTEEmbedder

embedder = ZTEEmbedder.from_checkpoint('res/checkpoints/best.pt')  # no dataset needed

# (a) New recordings as .mat files -> build a dataset, then embed.
from zte.config import DatasetConfig
from zte.data.dataset import ZuCoDataset
new_ds = ZuCoDataset(DatasetConfig(root='res/data/new_subject',
                                   representation=embedder.config.dataset.representation)).build()
word_emb, word_meta = embedder.embed(new_ds, level='word')

# (b) New signals already in memory (e.g. from a custom pipeline).
import numpy as np
signals = np.load('my_band_power.npy')          # shape (N, F*C), un-normalised
emb = embedder.embed_signals(band_power=signals)  # -> (N, embed_dim); normaliser applied
```

A complete runnable version of both flows is in `examples/embed_new_signals.py`:

```sh
uv run python examples/embed_new_signals.py            # trains a tiny model first
uv run python examples/embed_new_signals.py --ckpt res/checkpoints/best.pt
```

## Evaluating honestly

```python
from zte.training.metrics import linear_probe, retrieval_metrics, noise_matched

linear_probe(emb, meta['log_freq'].to_numpy())   # does the embedding encode frequency?
noise = noise_matched(ds.features)                # Gaussian floor (mean/var matched)
# Train a second model on `noise`; a real encoder must beat it with non-overlapping CIs.
```

## Performance tips

- **GPU**: `--device cuda --precision bf16` on Ampere+; raise `--batch-size`; set `compile_model: true` in YAML.
- **Apple silicon**: `--device mps` (fp32). Great for development-scale runs.
- **Throughput**: increase `train.num_workers`; `pin_memory` is auto-enabled on CUDA.
- **Large data**: cache once (`build()` writes a bundle), then train from the bundle repeatedly.
- **Contrastive objectives** benefit from larger batches (more in-batch negatives) — `skipgram`/`cbow`/`cpc` drop the last short batch automatically.
