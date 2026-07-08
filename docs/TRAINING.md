# Training & inference guide

Pretrain a self-supervised EEG **thought embedding**, then use the checkpoint to
embed words, sentences, or brand-new signals. This guide covers the one-command
path, the individual steps, every config knob, and inference.

> Related: [ARCHITECTURE.md](ARCHITECTURE.md) (model & objectives),
> [DATASET.md](DATASET.md) (building a bundle), [EVALUATION.md](EVALUATION.md)
> (proving the embedding is good), [RESULTS.md](RESULTS.md) (validated numbers).

## One command: `zte-run`

The recommended entry point. It resolves the data source, prepares + caches the
dataset, trains, evaluates and explores, cataloguing everything under
`res/experiments/<run_name>/`:

```sh
# No data needed — full pipeline on a synthetic ZuCo tree (great smoke test).
uv run zte-run --config experiments/exp1_skipgram_rope_et.yaml --synthetic --epochs 5

# Real data from a local folder (extracted .mat files, or a folder of task .zip archives).
uv run zte-run --config experiments/exp1_skipgram_rope_et.yaml --root res/data/zuco_extracted

# Real data straight from Google Drive (folder id or shareable URL; needs the `drive` group).
uv run zte-run --config experiments/exp2_masked_rope_eegonly.yaml --drive <folder-id-or-url>
```

Handy overrides (all optional): `--name`, `--subjects ZAB,ZDM`, `--tasks SR,NR`,
`--epochs N`, `--device auto|cpu|cuda|mps`, `--out-root`, `--extract-dir`,
`--no-tensorboard`, `--no-interactive`, `--skip-eval`, `--skip-explore`.

Each run writes `config.yaml`, `bundle/`, `checkpoints/`, `figures/`,
`evaluation/`, `exploration/`, `tb/`, `manifest.json`, `README.md`, plus a row in
`res/experiments/INDEX.md`. See [`experiments/README.md`](../experiments/README.md)
for the five flagship configs and their rationale.

## Prefer the individual steps? `zte-train`

```sh
# Skip-gram on a prepared bundle, 20 epochs, auto device.
uv run zte-train --bundle res/bundle --objective skipgram --epochs 20 --run-name sg

# Masked (data2vec) on raw EEG with a Conformer frontend, CUDA, bf16.
uv run zte-train --bundle res/bundle --objective masked \
    --frontend raw_conformer --representation raw --device cuda --precision bf16

# CPC with TensorBoard + Drive checkpoint backup.
uv run zte-train --bundle res/bundle --objective cpc --tensorboard \
    --drive-backup-dir /content/drive/MyDrive/ZTE/ckpts

# No bundle yet? Train straight from .mat files or synthetic data.
uv run zte-train --root res/data/zuco_extracted --objective skipgram
uv run zte-train --synthetic --objective masked --epochs 5
```

`zte-train` flags (a data source — `--bundle` / `--root` / `--synthetic` — is
required; everything else overrides the YAML/defaults):

| Flag                                               | Choices / type                                     | Overrides                                          |
| -------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- |
| `--config`                                         | path                                               | base YAML (`ZTEConfig`)                            |
| `--objective`                                      | `skipgram`,`cbow`,`masked`,`cpc`                   | `objective.name`                                   |
| `--frontend`                                       | `band_power_mlp`,`raw_conformer`                   | `model.frontend`                                   |
| `--representation`                                 | `band_power`,`raw`,`both`                          | `dataset.representation` (ignored with `--bundle`) |
| `--embed-dim`                                      | int                                                | `model.embed_dim`                                  |
| `--epochs` / `--batch-size` / `--lr`               | int/int/float                                      | `train.*`                                          |
| `--split`                                          | `random`,`by_sentence`,`by_subject_loso`,`by_task` | `train.split`                                      |
| `--device`                                         | `auto`,`cpu`,`cuda`,`mps`                          | `train.device`                                     |
| `--precision`                                      | `auto`,`fp32`,`fp16`,`bf16`                        | `train.precision`                                  |
| `--tensorboard`                                    | flag                                               | `train.tensorboard`                                |
| `--drive-backup-dir` / `--ckpt-dir` / `--run-name` | path/path/str                                      | `train.*` / `run_name`                             |

## Config-driven runs

Every knob is a field on a typed, YAML-serialisable `ZTEConfig`
(`src/zte/config.py`). A YAML file only needs to set what differs from the
defaults:

```yaml
# config.yaml
run_name: zte-skipgram-loso
dataset:
  representation: band_power
  normalize: zscore_channel
  include_eye_tracking: true
  missing: { method: knn }
model:
  frontend: band_power_mlp
  embed_dim: 768
  hidden_dim: 256
  n_layers: 4
  pos_encoding: rope          # rope | sinusoidal | learned | alibi | none
  pool: attention
objective:
  name: skipgram              # skipgram | cbow | masked | cpc
  temperature: 0.07
  context_window: 2
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

### Objective hyper-parameters (`ObjectiveConfig`)

| Field                   | Default    | Applies to    | Meaning                              |
| ----------------------- | ---------- | ------------- | ------------------------------------ |
| `name`                  | `skipgram` | —             | `skipgram`,`cbow`,`masked`,`cpc`     |
| `temperature`           | `0.07`     | contrastive   | InfoNCE softmax temperature          |
| `context_window`        | `2`        | skipgram/cbow | neighbouring words per side          |
| `n_negatives`           | `20`       | contrastive   | negatives per positive               |
| `mask_ratio`            | `0.5`      | masked        | fraction of tokens masked            |
| `masked_target`         | `latent`   | masked        | `latent` (data2vec) or `reconstruct` |
| `ema_decay`             | `0.999`    | masked/latent | teacher EMA decay                    |
| `cpc_steps`             | `4`        | cpc           | future steps predicted               |
| `reduce_omitted_weight` | `0.0`      | all           | down-weight omitted-word tokens      |

### Model knobs (`ModelConfig`)

`frontend`, `embed_dim` (768 by default → LLM-compatible), `hidden_dim`,
`n_layers`, `n_heads`, `dropout`, `pos_encoding`, `max_positions`, `pool`
(`mean`/`attention`/`cls`), `subject_conditioning`, `n_subjects`. Raw-Conformer
adds `conformer_filters` and `conformer_temporal_kernel`.

## Device & precision

Auto-detected by `zte.device.resolve_device`:

| Backend | Selected when            | Mixed precision                    | Notes                                     |
| ------- | ------------------------ | ---------------------------------- | ----------------------------------------- |
| `cuda`  | Nvidia GPU present       | bf16 (Ampere+) / fp16 + GradScaler | `--device cuda`; `compile_model` optional |
| `mps`   | Apple-silicon (M-series) | fp32 (autocast still maturing)     | `--device mps`                            |
| `cpu`   | otherwise                | fp32                               | fine for smoke-tests & synthetic data     |

```python
cfg.train.device = 'auto'      # or 'cuda' | 'mps' | 'cpu'
cfg.train.precision = 'auto'   # or 'bf16' | 'fp16' | 'fp32'
```

## The training loop

```mermaid
flowchart TD
    start([epoch loop]) --> batch[next batch → device]
    batch --> ac[autocast if AMP-safe]
    ac --> loss[objective.compute]
    loss --> bw[backward · grad-accum]
    bw --> step{accum step?}
    step -->|no| batch
    step -->|yes| clip[grad clip → optim → sched]
    clip --> ema[EMA teacher update<br/>data2vec only]
    ema --> log[log: rich progress + file + TensorBoard]
    log --> batch
    batch --> ee{epoch end}
    ee --> val[validate]
    val --> ckpt[save best/last · rotate · Drive backup]
```

## What a run produces

```text
res/checkpoints/
├── best.pt            # lowest monitored (val) loss
├── last.pt            # most recent
├── ckpt_epoch####.pt  # rotating (train.ckpt_keep_last)
├── config.yaml        # resolved config
└── training_curves.png
```

Each checkpoint embeds the model, optimiser, scheduler, AMP scaler, the full
config, the fitted feature-normaliser state and the subject vocabulary, so
inference reconstructs the exact pipeline — and can even run with no dataset.

## Reproducibility

Every config fixes `train.seed`; set `train.deterministic: true` for deterministic
cuDNN kernels on CUDA. Checkpoints embed the config, fitted normaliser and subject
vocabulary, so inference is exact. `zte-benchmark` runs a fixed-seed grid over
objective × positional-encoding × eye-tracking and writes each cell's `config.yaml`
(see [EVALUATION.md](EVALUATION.md)).

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

Or from the CLI, with an optional linear-probe sanity check + Drive upload:

```sh
uv run zte-extract --ckpt res/checkpoints/best.pt --bundle res/bundle \
    --level word --probe-target log_freq --out res/embeddings/embeddings.npz
```

### Embedding brand-new EEG signals

`from_checkpoint` reads the input shapes and the fitted normaliser from the
checkpoint itself, so **no dataset is required** to embed new signals. Use
`embed` for new recordings packaged as `.mat` files, or `embed_signals` for EEG
already in memory (any `(N, F·C)` band-power array, or `(N, C, T)` raw windows
for a raw-Conformer model). Band-power inputs are normalised exactly as in
training.

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
signals = np.load('my_band_power.npy')            # shape (N, F*C), un-normalised
emb = embedder.embed_signals(band_power=signals)  # -> (N, embed_dim); normaliser applied
```

A complete runnable version of both flows is in `examples/embed_new_signals.py`:

```sh
uv run python examples/embed_new_signals.py            # trains a tiny model first
uv run python examples/embed_new_signals.py --ckpt res/checkpoints/best.pt
```

## Performance tips

- **GPU**: `--device cuda --precision bf16` on Ampere+; raise `--batch-size`; set
  `compile_model: true` in YAML.
- **Apple silicon**: `--device mps` (fp32). Great for development-scale runs.
- **Throughput**: increase `train.num_workers`; `pin_memory` is auto-enabled on CUDA.
- **Large data**: cache once (`zte-prepare`/`build()` writes a bundle), then train
  from the bundle repeatedly.
- **Contrastive objectives** benefit from larger batches (more in-batch negatives)
  — `skipgram`/`cbow`/`cpc` drop the last short batch automatically.
