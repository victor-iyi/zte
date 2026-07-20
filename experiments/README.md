# ZTE experiments

Every file here is a plain, editable [`ZTEConfig`](../src/zte/config/) YAML. Run any of them end to end with **one command**; everything lands under `res/experiments/<run_name>/` (config, checkpoints, `evaluation/report.md`, figures, the interactive dashboards, and a `manifest.json` verdict).

```sh
# Synthetic smoke test (no data) → a real run → straight from Google Drive:
uv run zte-run --config experiments/sota_loso.yaml --synthetic --epochs 3
uv run zte-run --config experiments/sota_loso.yaml --root res/data/zuco_extracted --loso-holdout ZAB
uv run zte-run --config experiments/sota_loso.yaml --drive <folder-id-or-url> --loso-holdout ZAB
```

`--drive` works on every CLI that loads raw ZuCo data (`zte-prepare`, `zte-train`, `zte-evaluate`, `zte-explore`, `zte-benchmark`, …). Install Drive support once: `uv sync --group drive`. Zips download to `res/data/_downloads` and extract into `--extract-dir` (default `res/data/zuco_extracted`).

Handy overrides (all optional): `--name`, `--loso-holdout ZAB`, `--subjects ZAB,ZDM`, `--tasks SR,NR`, `--epochs N`, `--seed N`, `--device auto|cpu|cuda|mps`, `--resume`, `--skip-eval`, `--skip-explore`, `--no-tensorboard`, `--no-interactive`, `--out-root <dir>`, `--drive-backup <dir>`.

**`--resume` is idempotent:** a completed run is skipped instantly, an interrupted one continues from its last checkpoint. Re-run any command freely — it never redoes finished work.

---

## What "good" means here

The one number that decides everything is **held-out cross-subject retrieval**: train on 11 people, hold out a 12th the model has never seen (the *leave-one-subject-out*, or LOSO, split), and ask whether a sentence read by the stranger retrieves the *same sentence* read by the people the model knows. Because single-word EEG is the hardest non-invasive setting in the field, the honest headline is the **retrieval rank-percentile** (how far left of a label-shuffled permutation null the true match sits) and the **content lift over raw band-power**, not a raw top-1 number. Every run reports these, and the recommended configs below are the ones built to move them.

---

## The recommended configs (start here)

These are the current best recipes. All are LOSO (held out on `ZAB` by default) and EEG-only (an honest "thought, not gaze" choice — eye-tracking is a reading artefact absent from imagined thought).

| Config                          | Objective                                   | Encoder                                                                                                                | What it adds / its purpose                                                                                                                     |
| ------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **`sota_loso`**                 | skip-gram + full invariance stack           | band-power MLP + spherical-harmonic spatial encoding on the real electrode montage                                     | **The flagship.** Everything that makes the space healthy *and* subject-agnostic, plus the eval-time geometry fix. Start here.                 |
| **`exp7_sota_geom_invariance`** | skip-gram + full stack                      | **learned spatial attention** over 2-D electrode coordinates + **FiLM** subject conditioning + shrunk content subspace | A/B on the spatial model: condition on subject identity (not only adversarially remove it), and let the encoder learn its own scalp attention. |
| **`exp8_clip_e5`**              | **sentence-level CLIP** (symmetric InfoNCE) | band-power MLP, word-pooled to one sentence vector                                                                     | The direct attack on *meaning*: align each sentence's EEG to a frozen **E5** sentence embedding of its text, with semantic-hard negatives.     |
| **`exp8_clip_qwen`**            | sentence-level CLIP                         | band-power MLP, word-pooled                                                                                            | The same alignment against a **Qwen** (mean-pooled decoder-LLM) text target — the second arm of the text-encoder A/B.                          |
| **`exp8_clip_e5_raw`**          | sentence-level CLIP                         | **raw-conformer** (temporal→spatial convolution over raw EEG, ~700 ms window)                                          | CLIP on a time-resolved encoder instead of band power — tests whether the raw temporal signal carries more recoverable content.                |

### What the flagship recipe is made of

`sota_loso` stacks these, each targeting a specific failure mode. The full derivations (with math) are in [`../docs/METHODS.md`](../docs/METHODS.md); the summary:

- **Per-subject Riemannian normalisation** (`dataset.normalize: riemannian`) — re-centres and whitens each
  person's feature covariance, removing the constant per-subject offset that otherwise makes *who is
  reading* the cheapest thing to encode.
- **Subject + stimulus adversaries** (`objective.subject_adversary_weight`, `stimulus_adversary_weight`),
  **rebalanced and ramped** — a gradient-reversal classifier trains the encoder to *hide* subject/task
  identity. The strength is small (≈0.1, matching the EEG invariance literature; Özdenizci 2020) and its
  reversal coefficient ramps from zero (`subject_adversary_warmup_ratio`; Ganin 2016), so invariance
  pressure never erases the content it should preserve.
- **Cross-subject positives** — the contrastive positives are the *same sentence read by other people*, so
  subject identity becomes a nuisance the loss must remove rather than a shortcut it can exploit.
- **VICReg anti-collapse** (`variance_weight`, `covariance_weight`) + an **anti-cone uniformity** term
  (`anisotropy_weight`; Wang & Isola 2020) — a variance hinge and covariance penalty stop the space
  collapsing into a few of its 768 dimensions or into a single-direction cone.
- **Sharpened contrastive terms** — an **alignment** penalty (the missing half of alignment+uniformity),
  **debiased** InfoNCE (`tau_plus`; Chuang 2020, so another trial of the same word is not punished as a
  false negative), and a frozen-target **collapse-insurance** head (`data2vec_aux_weight`; Baevski 2022)
  that fills the otherwise-idle nuisance dimensions.
- **Spherical-harmonic spatial encoding** on the **real electrode montage** (`model.spatial_encoding`,
  `dataset.montage_csv`) — encodes *where each electrode sits on the scalp* using the Laplace-Beltrami
  eigenfunctions of the sphere, so the model is told Oz is at the back and Fp1 at the front instead of
  memorising channel indices.
- **Factored embedding + auxiliaries** — a dedicated content subspace (`content_dim`), a frozen
  word-meaning distillation target, and reading-behaviour (fixation-difficulty) supervision.
- **Eval-time geometry fix** (`whiten`, `all_but_top`, `csls_neighbors`) — at evaluation, ZCA-whitening,
  removing the top few shared principal directions (Mu & Viswanath 2018), and CSLS retrieval
  (Conneau 2018) strip the anisotropy/hubness that makes an otherwise-healthy space retrieve below chance.

### What the CLIP configs add — sentence-level semantic alignment

`exp8_clip_*` change the *objective itself*. Instead of predicting EEG neighbours, each sentence's word-EEG tokens are pooled into one vector and aligned — with a **symmetric InfoNCE loss** (CLIP; Radford 2021; Défossez 2023) — to a **frozen sentence embedding of its ground-truth text**:

```text
S = (z_eeg @ z_text.T) · logit_scale        # (B, B): rows = EEG readings, cols = text vectors
loss = ½ · ( InfoNCE(S, positives) + InfoNCE(Sᵀ, positives) )
positives[i, j] = (text_id[i] == text_id[j])  # same sentence, ANY subject, is a positive
```

The only way to win is to encode *what the sentence means*. Because the same sentence read by several subjects shares a `text_id`, every reading is a positive for that text → subject-invariance falls out for free. **Semantic-hard negatives** (`semantic_hard_negatives`) make the in-batch distractors surface-similar but meaning-distinct, so the encoder cannot win on surface form. The two text encoders (E5 vs Qwen) and the two EEG encoders (word-pool vs raw-conformer) are meant to be compared. Full tensor shapes and config surface: [`../docs/CLIP_ALIGNMENT.md`](../docs/CLIP_ALIGNMENT.md). The frozen encoders need `uv sync --group meaning`; without them the target falls back to a hash and a warning, so the pipeline still runs.

---

## Baseline configs (kept for reference & reproducibility)

The earlier configs. They isolate one design question each and are useful controls, but they predate the invariance/geometry stack above — for the strongest embedding use `sota_loso`.

| Config                            | Objective            | Eye-tracking | Split           | Isolates                                                    |
| --------------------------------- | -------------------- | ------------ | --------------- | ----------------------------------------------------------- |
| `exp1_skipgram_rope_et`           | skip-gram            | included     | by_sentence     | The plain reading-evoked embedding (eye-tracking on).       |
| `exp2_masked_rope_eegonly`        | masked (data2vec)    | excluded     | by_sentence     | Masked latent prediction, EEG-only (imagined-thought path). |
| `exp3_cpc_rope_et`                | CPC (wav2vec/BENDR)  | included     | by_sentence     | Whether reading *order* carries transferable structure.     |
| `exp4_skipgram_loso`              | skip-gram            | included     | by_subject_loso | The first honest subject-generalisation test.               |
| `exp5_raw_conformer_masked`       | masked (reconstruct) | excluded     | by_sentence     | Raw EEG-Conformer vs band-power compression.                |
| `exp6_skipgram_eegonly_invariant` | skip-gram            | excluded     | by_stimulus     | The first invariance recipe (superseded by `sota_loso`).    |

**exp1 vs exp2** isolates the single most important question for a *thought* code: does the representation lean on eye-tracking (great for reading, absent for imagined thought)? Compare their `evaluation/report.md` and `exploration/eye_tracking_contribution.csv`. **exp1 vs exp3** compares a context-free objective against a causal, order-aware one. **exp5** swaps the entire input pathway (raw EEG + Conformer) to check the band-power compression is not leaving signal on the table.

## Ablation configs (single-variable controls)

Matched pairs that flip exactly one lever, so its contribution is measurable in isolation. Prefer the `zte-ablate` workflow (below) for new levers; these pre-built pairs cover the historically important ones.

| Config pair                                                     | The one lever it flips                                |
| --------------------------------------------------------------- | ----------------------------------------------------- |
| `study_invariance_baseline_loso` / `study_invariance_full_loso` | The whole invariance stack, off vs on, under LOSO.    |
| `study_vicreg_off` / `study_vicreg_on`                          | The VICReg variance+covariance anti-collapse penalty. |
| `study_anticone_off` / `study_anticone_on`                      | The anti-cone uniformity term.                        |
| `study_all_levers`                                              | Everything on (the maximal ablation reference).       |

---

## Running them collectively

### The curated suite — `scripts/run_suite.sh`

Runs the recommended configs (the flagship, the spatial A/B, and the CLIP A/B) at a fixed seed, held out on `ZAB`, then the optional full-cohort LOSO sweep. Every run is `--resume`-safe.

```sh
bash scripts/run_suite.sh /path/to/zuco_extracted        # real data
SMOKE=1 bash scripts/run_suite.sh                        # tiny synthetic sanity pass (CPU)
# Mirror each run's checkpoints to Drive every epoch (train local, keep a live copy):
DRIVE_BACKUP=/gdrive/.../experiments bash scripts/run_suite.sh /path/to/zuco_extracted
```

### The full LOSO sweep — `scripts/run_loso.sh`

Rotates the held-out subject over the whole 12-person cohort for one config, turning a single number into a generalisation trend (`COMPARE.html`).

```sh
FULL_CFG=experiments/sota_loso.yaml bash scripts/run_loso.sh /path/to/zuco_extracted
FULL_CFG=experiments/sota_loso.yaml SUBJECTS="ZAB ZDM" bash scripts/run_loso.sh /path/to/zuco_extracted  # a subset
```

### The objective benchmark — `zte-benchmark`

Compares the four self-supervised objectives × positional encodings × {EEG-only, +eye-tracking} at fixed seeds in one table (headline metrics only, no per-run figures):

```sh
uv run zte-benchmark --root res/data/zuco_extracted \
    --objectives skipgram,cbow,masked,cpc --pos-encodings rope --eye-tracking both \
    --seeds 42 --out res/benchmark
```

### Prove one lever in isolation — `zte-ablate`

Emits a config pair that changes exactly one dotted `section.field`, runs both arms, and diffs their held-out scoreboards — the discipline behind every claim that a lever helps:

```sh
uv run zte-ablate generate --config experiments/sota_loso.yaml \
    --knob objective.subject_adversary_weight --values 0,0.1,0.3 --out-dir experiments/ablate
for cfg in experiments/ablate/*.yaml; do
  uv run zte-run --config "$cfg" --root res/data/zuco_extracted --loso-holdout ZAB --resume
done
uv run zte-ablate diff --knob objective.subject_adversary_weight \
    --baseline res/experiments/<off>/evaluation/metrics.json \
    --variant  res/experiments/<on>/evaluation/metrics.json
```

Any `objective.*` or `model.*` field works with zero code change, e.g. `objective.all_but_top`, `objective.csls_neighbors`, `objective.alignment_weight`, `objective.tau_plus`, `model.spatial_encoding`, `model.subject_film`.

---

## Reproducibility

Every config fixes `train.seed` and sets `train.deterministic: true`. `zte-run` copies the fully-resolved `config.yaml` into the run directory, so any run reproduces exactly:

```sh
uv run zte-run --config res/experiments/<run_name>/config.yaml --root <data> --name <run_name>
# or: --drive <folder-id-or-url>
```

## Catalogue

`res/experiments/INDEX.md` accumulates one row per run (words, held-out retrieval, rank-percentile, effective-rank ratio) so runs are comparable at a glance. Each run's own `README.md`, `manifest.json`, `evaluation/report.md`, and interactive `evaluation/interactive/held_out_scoreboard.html` hold the full configuration, data source and verdict. Compare any set of runs with `uv run zte-compare --experiments res/experiments`.
