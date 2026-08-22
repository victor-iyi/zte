# Training & inference guide

Pretrain a self-supervised EEG **thought embedding**, then use the checkpoint to embed words, sentences, or brand-new signals. This guide covers the one-command path, the individual steps, every config knob, and inference.

> Related: [ARCHITECTURE.md] (model & objectives), [DATASET.md] (building a bundle), [EVALUATION.md] (proving the embedding is good), [RESULTS.md] (validated numbers).

## One command: `zte-run`

The recommended entry point. It resolves the data source, prepares + caches the dataset, trains, evaluates and explores, cataloguing everything under `res/experiments/<run_name>/`:

```sh
# No data needed — full pipeline on a synthetic ZuCo tree (great smoke test).
uv run zte-run --config experiments/flagship/zte_raw_aligned.yaml --synthetic --epochs 5

# Real data from a local folder (extracted .mat files, or a folder of task .zip archives).
uv run zte-run --config experiments/flagship/zte_raw_aligned.yaml --root res/data/zuco_extracted

# Real data straight from Google Drive (folder id or shareable URL; needs the `drive` group).
uv run zte-run --config experiments/flagship/clip_e5_raw.yaml --drive <folder-id-or-url>
# Same flag on any individual step: zte-prepare, zte-train, zte-evaluate, zte-explore, …
```

Handy overrides (all optional): `--name`, `--subjects ZAB,ZDM`, `--tasks SR,NR`, `--epochs N`, `--device auto|cpu|cuda|mps`, `--out-root`, `--extract-dir` (default `res/data/zuco_extracted`), `--no-tensorboard`, `--no-interactive`, `--skip-eval`, `--skip-explore`.

Each run writes `config.yaml`, `bundle/`, `checkpoints/`, `figures/`, `evaluation/`, `exploration/`, `tb/`, `manifest.json`, `README.md`, plus a row in `res/experiments/INDEX.md`. The `<run_name>` comes from inside the config, not from its file path, so a tiered config still writes to its historical directory (`experiments/flagship/zte_raw_aligned.yaml` -> `res/experiments/exp8_clip_e5/`). See [`experiments/README.md`](../experiments/README.md) for the config tiers (`flagship/`, `decoder/`, `benchmark/`, `ablation/`, `archive/`) and their rationale.

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

# No bundle yet? Train straight from .mat files, Google Drive, or synthetic data.
uv run zte-train --root res/data/zuco_extracted --objective skipgram
uv run zte-train --drive 13EYW1h6dHD5E4YoEWNsKe6ZBHmMU_kFQ --objective skipgram
uv run zte-train --synthetic --objective masked --epochs 5
```

`zte-train` flags (a data source — `--bundle` / `--root` / `--drive` / `--synthetic` — is required; everything else overrides the YAML/defaults):

| Flag                                               | Choices / type                                                                             | Overrides                                          |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------- |
| `--config`                                         | path                                                                                       | base YAML (`ZTEConfig`)                            |
| `--root` / `--drive`                               | path / Drive id or URL                                                                     | local `.mat` dir, zip(s), or Drive folder          |
| `--extract-dir`                                    | path (`res/data/zuco_extracted`)                                                           | where Drive/zips are unzipped                      |
| `--objective`                                      | `skipgram`,`cbow`,`masked`,`cpc`,`clip`,`decode`                                           | `objective.name`                                   |
| `--frontend`                                       | `band_power_mlp`,`raw_conformer`                                                           | `model.frontend`                                   |
| `--representation`                                 | `band_power`,`raw`,`both`                                                                  | `dataset.representation` (ignored with `--bundle`) |
| `--embed-dim`                                      | int                                                                                        | `model.embed_dim`                                  |
| `--epochs` / `--batch-size` / `--lr`               | int/int/float                                                                              | `train.*`                                          |
| `--split`                                          | `random`,`by_sentence`,`by_stimulus`,`by_task`,`by_subject_loso`,`by_subject_and_stimulus` | `train.split`                                      |
| `--mode` / `--encoder-ckpt`                        | `encoder`,`decoder`,`joint` / path                                                         | `train.mode` / `train.encoder_ckpt`                |
| `--device`                                         | `auto`,`cpu`,`cuda`,`mps`                                                                  | `train.device`                                     |
| `--precision`                                      | `auto`,`fp32`,`fp16`,`bf16`                                                                | `train.precision`                                  |
| `--tensorboard`                                    | flag                                                                                       | `train.tensorboard`                                |
| `--drive-backup-dir` / `--ckpt-dir` / `--run-name` | path/path/str                                                                              | `train.*` / `run_name`                             |
| `--resume`                                         | flag                                                                                       | continue from `last.pt`                            |

## Config-driven runs

Every knob is a field on a typed, YAML-serialisable `ZTEConfig` (`src/zte/config/`). A YAML file only needs to set what differs from the defaults:

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

CLI flags override YAML, so you can pin a base config and sweep one knob: `uv run zte-train --config config.yaml --bundle b --objective cpc --lr 1e-4`.

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

The contrastive objectives score pairs by cosine similarity of the unit-normalised embeddings, $s_{ij} = \hat z_i^\top \hat z_j$ with $\hat z_i = z_i / \lVert z_i \rVert$, and divide the logits by the `temperature` $\tau$. Skip-gram treats every context word of an anchor as a positive, so it minimises a multi-positive InfoNCE over anchors $A$, positives $P(i)$ and candidate keys $\mathcal{C}(i)$ (the positives plus `n_negatives` sampled negatives); that is,

$$
\mathcal{L}_{\text{SG}} = -\frac{1}{\lvert A \rvert}\sum_{i \in A} \log \frac{\sum_{p \in P(i)} \exp(s_{ip}/\tau)}{\sum_{k \in \mathcal{C}(i)} \exp(s_{ik}/\tau)}
$$

CBOW and CPC instead have a single positive $i^{+}$ per anchor (the pooled context, or the true future step), giving the standard single-positive form

$$
\mathcal{L} = -\frac{1}{N}\sum_{i} \log \frac{\exp(s_{i,i^{+}}/\tau)}{\sum_{k} \exp(s_{ik}/\tau)}
$$

Lowering $\tau$ sharpens the softmax and increasingly penalises hard negatives.

The masked objective (`masked_target: latent`, i.e. data2vec) regresses a student prediction onto an EMA **teacher**. Before the loss, the teacher targets are normalised across tokens per feature $j$, $\tilde t_{:,j} = (t_{:,j} - \mu_j) / \max(\sigma_j,\ \sigma_{\min})$, and the loss is a smooth-L1 (Huber) between the student prediction and $\tilde t$ over the masked positions (`mask_ratio` sets the masked fraction).

### Model knobs (`ModelConfig`)

`frontend`, `embed_dim` (768 by default -> LLM-compatible), `hidden_dim`, `n_layers`, `n_heads`, `dropout`, `pos_encoding`, `max_positions`, `pool` (`mean`/`attention`/`cls`), `subject_conditioning`, `n_subjects`. Raw-Conformer adds `conformer_filters` and `conformer_temporal_kernel`.

## Device & precision

Auto-detected by `zte.device.resolve_device`:

| Backend | Selected when            | Mixed precision                    | Notes                                     |
| ------- | ------------------------ | ---------------------------------- | ----------------------------------------- |
| `cuda`  | Nvidia GPU present       | bf16 (Ampere+) / fp16 + GradScaler | `--device cuda`; `compile_model` optional |
| `mps`   | Apple-silicon (M-series) | fp32 (autocast still maturing)     | `--device mps`                            |
| `cpu`   | otherwise                | fp32                               | fine for smoke-tests & synthetic data     |

```python
cfg.train.device = 'auto'  # or 'cuda' | 'mps' | 'cpu'
cfg.train.precision = 'auto'  # or 'bf16' | 'fp16' | 'fp32'
```

## The training loop

```mermaid
flowchart TD
    start([epoch loop]) --> batch[next batch -> device]
    batch --> ac[autocast if AMP-safe]
    ac --> loss[objective.compute]
    loss --> bw[backward · grad-accum]
    bw --> step{accum step?}
    step -->|no| batch
    step -->|yes| clip[grad clip -> optim -> sched]
    clip --> ema[EMA teacher update<br/>data2vec only]
    ema --> log[log: rich progress + file + TensorBoard]
    log --> batch
    batch --> ee{epoch end}
    ee --> val[validate]
    val --> ckpt[save best/last · rotate · Drive backup]
```

For the data2vec / masked objective, the `EMA teacher update` step above is an exponential moving average of the student parameters $\phi$ into the teacher parameters $\theta$, that is, $\theta \leftarrow \rho\,\theta + (1-\rho)\,\phi$, where the decay $\rho$ is the `ema_decay` field (typically ramped $\rho_0 \to \rho_1$ over training).

## Three modes: `encoder`, `decoder`, `joint`

`train.mode` selects what a run trains. The language model is frozen in all three — `joint` refers to the encoder and
the bridge, never to the LM.

| Mode                | Trains                                                                         | Encoder                                                  | Needs `train.encoder_ckpt` |
| ------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------- | -------------------------- |
| `encoder` (default) | encoder + objective, one group at `train.lr`                                   | from scratch                                             | no                         |
| `decoder`           | the prefix bridge at `train.bridge_lr`                                         | loaded, frozen, kept in `.eval()`                        | yes                        |
| `joint`             | stage A the bridge; stage B also the encoder at `bridge_lr × encoder_lr_scale` | loaded, frozen for `train.stage_a_epochs`, then unfrozen | yes                        |

`train.freeze_encoder` means the encoder never trains, so `joint` requires `freeze_encoder: false` and raises if it is
`true`; there, `train.stage_a_epochs` alone decides when the encoder joins in.

`encoder` is the pre-decoder pipeline unchanged: `stages.parameter_groups` returns the single AdamW group it always
returned, and the decoder wiring keys off the objective rather than the mode, so an `objective.name` other than
`decode` never builds a bridge or loads an LM. In the other two modes the source
checkpoint's fitted **normaliser and aligner are restored, not refitted** — refitting does not fail a frozen encoder,
it silently feeds it a scale it never trained on. Parameter groups are structural rather than conditional on
`requires_grad`, so a resume whose freeze state differs cannot break the optimiser state, and `LambdaLR` decays each
group against its own `initial_lr`.

`train.early_stop_patience` (0 disables) stops after that many epochs without an improvement in the monitored metric,
and the counter survives a resume. Every real run on record bottoms out its validation loss at epoch 5–6 of 40.

**The best-checkpoint monitor is stage-comparable.** In `joint` mode the auxiliaries that enter the loss when the
encoder unfreezes jump the monitored validation scalar by several units at the A→B boundary, so a lifetime
best-value comparison would lock `best.pt` into stage A and let patience kill the run before stage B was ever
measured. The trainer therefore forgets the best value and zeroes the patience counter at every stage transition,
and each checkpoint payload records the stage that produced it, so a resume re-detects the boundary. Values are
compared only within a stage; the auxiliaries themselves are never rescaled to flatter the monitor. A `decoder`-mode
run with `freeze_encoder: false` is refused loudly — that combination trains the encoder from epoch 1 while claiming
it is frozen; `joint` is the mode that means that.

Full method: [DECODER.md](DECODER.md).

## What a run produces

```text
res/checkpoints/
├── best.pt            # lowest monitored (val) loss
├── last.pt            # most recent
├── ckpt_epoch####.pt  # rotating (train.ckpt_keep_last)
├── config.yaml        # resolved config
└── training_curves.png
```

Each checkpoint embeds the model, optimiser, scheduler, AMP scaler, the full config, the fitted feature-normaliser state and the subject vocabulary, so inference reconstructs the exact pipeline — and can even run with no dataset.

## Reproducibility

Every config fixes `train.seed`; set `train.deterministic: true` for deterministic cuDNN kernels on CUDA. Checkpoints embed the config, fitted normaliser and subject vocabulary, so inference is exact. `zte-benchmark` runs a fixed-seed grid over objective × positional-encoding × eye-tracking and writes each cell's `config.yaml` (see [EVALUATION.md]).

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
# Or embed straight from Drive / local .mat files:
uv run zte-extract --ckpt res/checkpoints/best.pt --drive <folder-id-or-url> \
    --level word --out res/embeddings/embeddings.npz
```

### Embedding brand-new EEG signals

`from_checkpoint` reads the input shapes and the fitted normaliser from the checkpoint itself, so **no dataset is required** to embed new signals. Use `embed` for new recordings packaged as `.mat` files, or `embed_signals` for EEG already in memory (any `(N, F·C)` band-power array, or `(N, C, T)` raw windows for a raw-Conformer model). Band-power inputs are normalised exactly as in training.

```python
from zte.inference.embed import ZTEEmbedder

embedder = ZTEEmbedder.from_checkpoint('res/checkpoints/best.pt')  # no dataset needed

# (a) New recordings as .mat files -> build a dataset, then embed.
from zte.config import DatasetConfig
from zte.data.dataset import ZuCoDataset

new_ds = ZuCoDataset(
    DatasetConfig(root='res/data/new_subject', representation=embedder.config.dataset.representation)
).build()
word_emb, word_meta = embedder.embed(new_ds, level='word')

# (b) New signals already in memory (e.g. from a custom pipeline).
import numpy as np

signals = np.load('my_band_power.npy')  # shape (N, F*C), un-normalised
emb = embedder.embed_signals(band_power=signals)  # -> (N, embed_dim); normaliser applied
```

A complete runnable version of both flows is in `examples/embed_new_signals.py`:

```sh
uv run python examples/embed_new_signals.py            # trains a tiny model first
uv run python examples/embed_new_signals.py --ckpt res/checkpoints/best.pt
```

## Performance tips

- **GPU**: `--device cuda --precision bf16` on Ampere+; raise `--batch-size`; set `compile_model: true` in YAML.
- **Apple silicon**: `--device mps` (fp32). Great for development-scale runs.
- **Throughput**: increase `train.num_workers`; `pin_memory` is auto-enabled on CUDA.
- **Large data**: cache once (`zte-prepare`/`build()` writes a bundle), then train from the bundle repeatedly.
- **Contrastive objectives** benefit from larger batches (more in-batch negatives)
  — `skipgram`/`cbow`/`cpc` drop the last short batch automatically.

[ARCHITECTURE.md]: ./ARCHITECTURE.md
[DATASET.md]: ./DATASET.md
[EVALUATION.md]: ./EVALUATION.md
[RESULTS.md]: ./RESULTS.md

## Running the whole study — `scripts/run_zte_study.sh`

One resumable command for everything a claim has to survive. Each stage answers one question, and each one is
skippable:

| Stage        | Question                                                                            |
| ------------ | ----------------------------------------------------------------------------------- |
| `audit`      | Is the dataset confounded, before any model is trained?                             |
| `encoder`    | Does the encoder reach a stranger's brain, at more than one seed?                   |
| `loso`       | ...for every one of the 12 subjects? (multi-hour, times `SEEDS`)                    |
| `decoder`    | Does the decoder read the brain, or recite the corpus?                              |
| `ablation`   | Which lever did it: raw vs band power, harmonics vs indexing, invariance on vs off? |
| `rebaseline` | How much of the number is sentence length?                                          |
| `analysis`   | All of it, as one offline page plus CSV tables.                                     |

```sh
# Real data, three seeds, everything except the 12-fold sweep.
SEEDS='42 43 44' bash scripts/run_zte_study.sh res/data/zuco_extracted

# Add the sweep; this is the expensive one.
STAGES='audit encoder loso decoder ablation rebaseline analysis' bash scripts/run_zte_study.sh

# Re-draw the analysis from what is already on disk, training nothing.
STAGES=analysis bash scripts/run_zte_study.sh

# Offline wiring check in minutes. Every config is rewritten offline-safe first; nothing here is a result.
SMOKE=1 bash scripts/run_zte_study.sh
```

**Pause and resume.** Every training command carries `--resume`, so Ctrl-C or a reclaimed Colab VM costs at most
the epoch in flight; re-running the identical command skips finished work instantly. With `DRIVE_BACKUP` set the
whole run directory is mirrored after every stage, and the study-level artifacts (`LOSO_SUMMARY.md`,
`confound_audit.md`, `analysis/`) are copied at the end, so nothing lives only on a VM disk.

**A decoder arm never takes `--loso-holdout`.** That flag forces `split=by_subject_loso`, which shares all 700
stimuli between train and val — the one configuration in which a decoder recites the corpus and scores well. The
script writes the held-out subject into a temporary config instead, so the honest four-cell split survives.

### The cost of the word-synchronous evidence path

With `decoder.evidence_schedule` on, the per-word hiddens are needed at every step, so the frozen-encoder cache is
unavailable and the encoder runs every epoch. Budget roughly an encoder run's cost plus the frozen LM.
`experiments/decoder/decode_v2_pooled.yaml` and `decode_v2_ladder_only.yaml` keep the cache and are several times
cheaper, which is also why the arms run at one seed while the headline gets the spread.

## Surviving a reclaimed machine

ZTE runs in two places and the durability rules differ. On Colab the VM can vanish without warning and only the
mounted Drive survives; on a workstation nothing is ephemeral and `res/` is the durable root. The contract below
holds in both: **nothing expensive is ever more than one epoch, or one stage, away from durable storage.**

| written | when | why then |
| --- | --- | --- |
| `best.pt` | the moment it improves | it is the result; losing it loses the run's whole point |
| `last.pt` | every epoch | `--resume` reads it, so a reclaimed VM costs one epoch |
| `ckpt_epoch*.pt` | never mirrored | rotation history -- `keep_last` extra copies of a large file that a fresh VM cannot use |
| the run directory | each stage | config, `history.json`, evaluation, figures, TensorBoard |
| evaluation · generation · analysis · studio | straight to the durable root | expensive, and none of it resumes |
| the prepared feature bundle | once, ever | content-addressed and not date-stamped, so every future session reuses it |

`CheckpointManager._backup_to_drive` runs on every `save()` and mirrors exactly two files. That is both safer and
cheaper than mirroring the whole checkpoint directory: `best.pt` and `last.pt` are on Drive within one epoch, and the
per-epoch traffic drops from `keep_last + 2` large files to at most two. `mirror_file` skips a file whose bytes
already match, so a `best.pt` that did not improve costs a `stat` rather than a copy.

### A failed mirror is a warning, a silently failed mirror is a lost run

Mirroring never raises -- a full Drive must not kill a multi-hour run. But `mirror_file` returning `False` looks
identical whether the file was unchanged or the mount stopped accepting writes, so `_note_mirror` checks that
`last.pt` actually landed and counts *consecutive* failures. At 1, 3, 10 and 30 it escalates to an error naming the
missing path and saying plainly that the run is not currently recoverable from Drive. The count resets the moment a
mirror succeeds, so one early hiccup does not shout for the rest of the run.

### Resuming from a restored directory

A run directory restored from Drive carries `best.pt` and `last.pt` and no rotation history. `load_latest` handles
that: it tries `last.pt`, then the epoch files newest-first, then `best.pt`. The last of those is what makes the
thin mirror safe -- `best.pt` is a full-state checkpoint like any other, so even a `last.pt` torn by the write that
was in flight when the machine went away costs the epochs since the last improvement rather than the run.

On Colab the notebook's `resolve_ckpt()` searches this session's Drive folder, then every earlier session
newest-first, then the local disk. A fresh runtime can therefore evaluate, decode or open the studio on a run
trained in a previous session with no manual restore step.

## The exp16 encoder mechanisms in the loop

Four mechanisms join the training step, and each enters at a different point. `docs/METHODS.md` §9-12 has the maths;
this is where they sit in the loop.

| mechanism | where it runs | who owns the loss |
| --- | --- | --- |
| predictive residual | inside `ZTEModel.token_hidden`, after subject conditioning | the `Trainer`, which drains it |
| cross-reader consensus (sentence) | `SentenceClipObjective.compute`, on the pooled vector | the objective |
| cross-reader consensus (word) | `_ObjectiveBase.regularize`, on the usable token embeddings | the objective |
| gallery contrast | `SentenceClipObjective.compute`, after the in-batch InfoNCE | the objective |

**Why the residual head's loss belongs to the trainer.** It belongs to no objective: it de-trends the encoder's
tokens for whatever loss comes next, and every objective gets the same treatment. `ZTEModel` stashes it,
`Trainer._residual_loss` drains it once per step, scales it by `model.residual_predict_weight` and merges its
metrics. The drain also runs in `evaluate()`, where the stash would otherwise pin one autograd graph per validation
batch for the length of the loop.

**The consensus bank is training-only state.** It is written under `self.training` and read before it is written, so
a validation pass leaves it exactly as training left it and a held-out subject never enters it. It lives on the
objective, not on the model, so nothing in the inference path can reach it.

**Per-epoch metrics travel in `history.json`.** Every numeric key an objective returns is averaged over the epoch's
steps and appended to `Trainer.history`, then written next to the manifest. `zte-analyze` plots them as the
**mechanism curves** panel. This is not decoration: the final `metrics.json` cannot tell "the consensus term did
nothing" from "the consensus bank never reached `consensus_min_readers` and contributed exactly zero", and those are
very different findings.

**Inheriting an encoder replaces the run's `model` config.** A decoder run started with `--encoder-ckpt` uses the
source encoder as its model, so the run's own `model` section is overwritten with the source's before anything is
saved. Without that, the checkpoint would store a description of an encoder that was never built and would fail to
reload — which is exactly how a decoder over a residual-coded encoder broke before this rule existed.
