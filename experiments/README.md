# ZTE experiments

Every file here is a plain, editable [`ZTEConfig`](../src/zte/config/) YAML, sorted into four tiers by what it has actually achieved on real ZuCo. Run any of them end to end with **one command**; everything lands under `res/experiments/<run_name>/` (config, checkpoints, `evaluation/report.md`, figures, the interactive dashboards, and a `manifest.json` verdict).

```sh
# Synthetic smoke test (no data) → a real run → straight from Google Drive:
uv run zte-run --config experiments/flagship/clip_e5_bandpower.yaml --synthetic --epochs 3
uv run zte-run --config experiments/flagship/clip_e5_bandpower.yaml --root res/data/zuco_extracted --loso-holdout ZAB
uv run zte-run --config experiments/flagship/clip_e5_bandpower.yaml --drive <folder-id-or-url> --loso-holdout ZAB
```

| Tier         | What lives there                                                                         |
| ------------ | ---------------------------------------------------------------------------------------- |
| `flagship/`  | The recipes that have beaten chance on real ZuCo, plus the one hypothesis built on them. |
| `benchmark/` | The controls a flagship must beat to earn its place.                                     |
| `ablation/`  | Single-lever studies — matched pairs that flip exactly one knob.                         |
| `archive/`   | Superseded or failed arms, kept for the record and for reproducibility.                  |

> **File paths moved; `run_name` did not.** `experiments/flagship/clip_e5_bandpower.yaml` still trains a run called `exp8_clip_e5`, so every run already on Drive keeps matching and `--resume` still skips it.
> The name in the left column below is the *file*; the run directory uses the `run_name` in brackets.

---

## The evidence

Real ZuCo, leave-one-subject-out with `ZAB` held out, 12 subjects / 160,804 words / 8,400 sentences (session of 2026-07-16, on Drive under `Sharables/ZTE/2026-07-16`). Chance Top-1 is 0.0013.

| Config (run_name)                                  | Sentence Top-1 | Permutation *p* | Held-out Top-1 lift | Verdict        |
| -------------------------------------------------- | -------------- | --------------- | ------------------- | -------------- |
| `flagship/clip_e5_bandpower` (`exp8_clip_e5`)      | **0.0932**     | **0.002**       | +0.29pp             | ✓ beats chance |
| `flagship/clip_e5_raw` (`exp8_clip_e5_raw`)        | 0.0065         | **0.002**       | **+0.71pp**         | ✓ beats chance |
| `benchmark/clip_qwen_bandpower` (`exp8_clip_qwen`) | 0.0010         | 0.096           | +0.00pp             | ✗              |
| `benchmark/baseline_skipgram_loso` (`sota_loso`)   | 0.0004         | 0.986           | +0.29pp             | ✗              |
| `archive/exp7_sota_geom_invariance`                | 0.0000         | 1.000           | +0.00pp             | ✗              |

Read that table as three findings. **Sentence-level CLIP against a frozen E5 text embedding is the only objective that has ever cleared the retrieval gate here** — skip-gram, which used to be the flagship, is now a control. **The text encoder matters**: the same recipe against a Qwen target fails, so the win is not "any text embedding will do". And **the encoder A/B is unresolved**: band power wins in-sample by 14×, but the raw-conformer wins where it counts more — on the held-out subject — and is the only arm that made subjects *harder* to identify than raw band power (subject probe 0.419 vs 0.809 for raw).

### What is still wrong with even the best run

State these next to any result from this directory; they are in every `evaluation/report.md`.

- **The content-probe positive control fails in every run.** Raw band power reads lexical content at R² = −0.008 against a floor of 0.02, so the probe cannot recover content even from raw features. Until that is fixed, "content variance 0%" and the content lifts are not interpretable.
- **`what_variance` is 0.0 everywhere.** The champion spends 93.6% of its variance on "none", 3.5% on subject and 2.9% on task — nothing on content. `clip_e5_meaning` is the direct attack on this.
- **The held-out number is two orders of magnitude below the headline.** 0.093 is cross-subject retrieval pooled over all 12 subjects; on genuinely held-out `ZAB` it is 0.43% vs 0.14% chance. The held-out task-variance of 0.613 says the freed variance moved into the task axis.

---

## `flagship/` — start here

| Config              | Objective                                  | Encoder                                                      | Why it is here                                                                                                           |
| ------------------- | ------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `clip_e5_bandpower` | sentence-level CLIP (symmetric InfoNCE)    | band-power MLP + spherical-harmonic spatial encoding         | **The champion.** Best cross-subject retrieval ever measured here (Top-1 0.093, *p* = 0.002).                            |
| `clip_e5_raw`       | sentence-level CLIP                        | raw-conformer (temporal→spatial convolution, ~700 ms window) | Best **held-out** lift (+0.71pp) and the best held-out effective rank (0.417); the de-identification winner.             |
| `clip_e5_meaning`   | sentence-level CLIP + meaning distillation | band-power MLP + spherical harmonics                         | **Untested hypothesis:** the champion plus a per-occurrence contextual meaning target, aimed at the 0% content variance. |

All three are EEG-only (an honest "thought, not gaze" choice — eye-tracking is a reading artefact absent from imagined thought), LOSO, and Riemannian-normalised per subject.

### What the CLIP objective does

Each sentence's word-EEG tokens are pooled into one vector and aligned — with a symmetric InfoNCE loss — to a frozen sentence embedding of its ground-truth text:

```text
S = (z_eeg @ z_text.T) · logit_scale        # (B, B): rows = EEG readings, cols = text vectors
loss = ½ · ( InfoNCE(S, positives) + InfoNCE(Sᵀ, positives) )
positives[i, j] = (text_id[i] == text_id[j])  # same sentence, ANY subject, is a positive
```

The only way to win is to encode *what the sentence means*. Because the same sentence read by several subjects shares a `text_id`, every reading is a positive for that text, so subject-invariance falls out for free. **Semantic-hard negatives** make the in-batch distractors surface-similar but meaning-distinct, so the encoder cannot win on surface form. Full tensor shapes and config surface: [`../docs/CLIP_ALIGNMENT.md`](../docs/CLIP_ALIGNMENT.md). The frozen encoders need `uv sync --group meaning`; without them the target falls back to a hash and a warning, so the pipeline still runs — but the result is meaningless, so check the log.

### The rest of the stack (shared by every flagship arm)

Per-subject **Riemannian normalisation**; **subject + stimulus adversaries**, rebalanced (≈0.1) and ramped from zero; **cross-subject positives**; **VICReg anti-collapse** plus an anti-cone uniformity term; **alignment** and **debiased** (`tau_plus`) contrastive terms and a frozen-target data2vec head; **spherical-harmonic spatial encoding** on the real electrode montage; a **factored embedding** with a dedicated content subspace; and the **eval-time geometry fix** (`whiten`, `all_but_top`, `csls_neighbors`) that strips the anisotropy and hubness which otherwise push retrieval below chance.  Derivations are in [`../docs/METHODS.md`](../docs/METHODS.md).

## `benchmark/` — the controls

| Config                   | What it controls for                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| `baseline_skipgram_loso` | The previous SOTA recipe (skip-gram + the full invariance stack). Answers "did CLIP earn it?". |
| `clip_qwen_bandpower`    | The second arm of the text-encoder A/B (E5 vs Qwen), on an otherwise identical recipe.         |

## `ablation/` — one lever at a time

| Config pair                                                     | The one lever it flips                                |
| --------------------------------------------------------------- | ----------------------------------------------------- |
| `study_invariance_baseline_loso` / `study_invariance_full_loso` | The whole invariance stack, off vs on, under LOSO.    |
| `study_vicreg_off` / `study_vicreg_on`                          | The VICReg variance+covariance anti-collapse penalty. |
| `study_anticone_off` / `study_anticone_on`                      | The anti-cone uniformity term.                        |
| `study_all_levers`                                              | Everything on (the maximal ablation reference).       |

The four `study_invariance_*` / `study_vicreg_*` files are generated from `ZTEConfig` objects by
`scripts/make_study_configs.py`, so they cannot drift from the schema. For a *new* lever, prefer
`zte-ablate` (below) over hand-writing a pair.

## `archive/` — retired, kept for the record

`exp1_skipgram_rope_et`, `exp2_masked_rope_eegonly`, `exp3_cpc_rope_et`, `exp4_skipgram_loso`, `exp5_raw_conformer_masked`, `exp6_skipgram_eegonly_invariant` are the original objective/encoder studies. On the 2026-07-12 real-data sweep every one of them scored a sentence-retrieval Top-1 of 0.0 (permutation *p* ≈ 1.0) with a who-vs-what variance ratio up to 1.0 — the identity-encoding failure mode the invariance stack was built to fix. `exp7_sota_geom_invariance` (learned spatial attention + FiLM subject conditioning) is archived for the opposite reason: it was a serious contender and it failed hardest, retrieving nothing at all (Top-1 0.0, *p* = 1.0). Keep them for reproducing the history; do not start new work from them.

---

## Running them collectively

### The tiered suite — `scripts/run_suite.sh`

```sh
bash scripts/run_suite.sh /path/to/zuco_extracted             # audit + flagship + controls (the default)
SMOKE=1 bash scripts/run_suite.sh                             # tiny synthetic sanity pass (CPU, minutes)
STUDIES="audit flagship controls benchmark ablate" bash scripts/run_suite.sh /path/to/zuco_extracted
```

`STUDIES` selects what runs: `audit` (the model-free confound report — run it before believing any result), `flagship`, `controls`, `benchmark` (objective sweep on top of the champion), `ablate` (one-knob studies), `loso` (the full 12-subject sweep). A failing arm no longer aborts the suite; the run is reported at the end and retried on the next invocation.

### The full LOSO sweep — `scripts/run_loso.sh`

Rotates the held-out subject over the whole 12-person cohort for one config, turning a single number into a generalisation trend (`COMPARE.html`). Defaults to the champion.

```sh
bash scripts/run_loso.sh /path/to/zuco_extracted                          # champion, all 12 subjects
FULL_CFG=experiments/flagship/clip_e5_raw.yaml bash scripts/run_loso.sh   # a different arm
SUBJECTS="ZAB ZDM" bash scripts/run_loso.sh /path/to/zuco_extracted       # a subset
CONTROL=1 bash scripts/run_loso.sh /path/to/zuco_extracted                # also run the skip-gram control
```

### The objective benchmark — `zte-benchmark`

Sweeps objectives **on top of a base recipe**, so the only thing differing between rows is the axis under test rather than the whole model. Resumable: a finished cell is reused from its `metrics.json`.

```sh
uv run zte-benchmark --root res/data/zuco_extracted \
    --base-config experiments/flagship/clip_e5_bandpower.yaml --loso-holdout ZAB \
    --objectives clip,skipgram,masked,cpc --pos-encodings rope --eye-tracking off \
    --seeds 42 --out res/benchmark --resume
```

### Prove one lever in isolation — `zte-ablate`

```sh
uv run zte-ablate generate --config experiments/flagship/clip_e5_bandpower.yaml \
    --knob objective.meaning_distill_weight --values 0,0.1,1.0 --out-dir res/ablate_configs
for cfg in res/ablate_configs/*.yaml; do
  uv run zte-run --config "$cfg" --root res/data/zuco_extracted --loso-holdout ZAB --resume
done
uv run zte-ablate diff --knob objective.meaning_distill_weight \
    --baseline res/experiments/<off>/evaluation/metrics.json \
    --variant  res/experiments/<on>/evaluation/metrics.json
```

Any `objective.*` or `model.*` field works with zero code change, e.g. `objective.all_but_top`, `objective.csls_neighbors`, `objective.alignment_weight`, `model.spatial_encoding`, `model.subject_film`.

---

## Surviving a reclaimed Colab VM

Multi-hour runs assume the machine can vanish at any moment, so nothing important lives only in RAM or only on the VM's disk:

- **`--resume` is idempotent.** A completed run is skipped instantly; an interrupted one continues from its last epoch. Re-run any command freely — it never redoes finished work.
- **Checkpoint writes are atomic**, and resume falls back past a torn file. A VM killed mid-write costs the epoch in flight, not the run.
- **`--drive-backup <mounted path>` mirrors the whole run directory** — config, checkpoints, evaluation, figures, TensorBoard — after every stage, and checkpoints after every epoch. Only changed files move, so the cost stays flat as checkpoints grow.
- **`config.yaml` is written before training starts**, so a run killed at any point is reproducible from its own directory without reconstructing CLI flags by hand.
- **`--data-cache <mounted path>` builds the processed dataset bundle once, ever** — across every subject, arm and session. Synthetic and real data can never collide in that cache.

```sh
DRIVE_BACKUP="/content/drive/MyDrive/Sharables/ZTE/$(date +%F)/experiments" \
DATA_CACHE="/content/drive/MyDrive/Sharables/ZTE/prepared" \
bash scripts/run_loso.sh /content/zuco_extracted
```

If the VM is reclaimed: copy the Drive folder back to `OUT_ROOT` (or point `OUT_ROOT` straight at Drive) and re-run the identical command.

## Reproducibility

Every config fixes `train.seed` and sets `train.deterministic: true`. `zte-run` copies the fully-resolved `config.yaml` into the run directory, so any run reproduces exactly:

```sh
uv run zte-run --config res/experiments/<run_name>/config.yaml --root <data> --name <run_name>
```

## Catalogue

`res/experiments/INDEX.md` accumulates one row per run so runs are comparable at a glance. Each run's own `README.md`, `manifest.json`, `evaluation/report.md` and interactive `evaluation/interactive/held_out_scoreboard.html` hold the full configuration, data source and verdict.  Compare any set of runs with `uv run zte-compare --experiments res/experiments`.
