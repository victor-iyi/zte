# ZTE experiments

Every file here is a plain, editable [`ZTEConfig`](../src/zte/config/) YAML, sorted into four tiers by what it has actually achieved on real ZuCo. Run any of them end to end with **one command**; everything lands under `res/experiments/<run_name>/` (config, checkpoints, `evaluation/report.md`, figures, the interactive dashboards, and a `manifest.json` verdict).

```sh
# Synthetic smoke test (no data) → a real run → straight from Google Drive:
uv run zte-run --config experiments/flagship/clip_e5_bandpower.yaml --synthetic --epochs 3
uv run zte-run --config experiments/flagship/clip_e5_bandpower.yaml --root res/data/zuco_extracted --loso-holdout ZAB
uv run zte-run --config experiments/flagship/clip_e5_bandpower.yaml --drive <folder-id-or-url> --loso-holdout ZAB
```

| Tier             | What lives there                                                                               |
| ---------------- | ---------------------------------------------------------------------------------------------- |
| `flagship/`      | The recipes that have beaten chance on real ZuCo, plus the encoder arms built on the champion. |
| `text_encoders/` | Text-encoder A/B: the champion recipe with only the CLIP sentence target swapped.              |
| `benchmark/`     | The controls a flagship must beat to earn its place.                                           |
| `ablation/`      | Single-lever studies — matched pairs that flip exactly one knob.                               |
| `archive/`       | Superseded or failed arms, kept for the record and for reproducibility.                        |

> **File paths moved; `run_name` did not.** `experiments/flagship/clip_e5_bandpower.yaml` still trains a run called `exp8_clip_e5`, so every run already on Drive keeps matching and `--resume` still skips it.
> The name in the left column below is the *file*; the run directory uses the `run_name` in brackets.

---

## The evidence

Real ZuCo, leave-one-subject-out with `ZAB` held out, 12 subjects / 160,804 words / 8,400 sentences (session of 2026-07-24, on Drive under `Sharables/ZTE/2026-07-24`). The board is a clean 2×2 over {frontend} × {meaning distillation}. Chance Top-1 is ≈0.001.

| Config (run_name)                                       | Frontend      | Meaning | Sentence Top-1 | Eff-rank | Subject var | Verdict        |
| ------------------------------------------------------- | ------------- | ------- | -------------- | -------- | ----------- | -------------- |
| `flagship/clip_e5_bandpower` (`exp8_clip_e5`)           | band_power    | off     | 0.019          | 0.166    | 10.1%       | ✓ beats chance |
| `flagship/clip_e5_meaning` (`exp9_clip_e5_meaning`)     | band_power    | **ON**  | **0.043**      | 0.160    | **0.9%**    | ✓ **champion** |
| `flagship/clip_e5_raw` (`exp8_clip_e5_raw`)             | raw_conformer | off     | 0.010          | **0.264**| 6.9%        | ✓ beats chance |
| `benchmark/clip_qwen_bandpower` (`exp8_clip_qwen`)      | band_power    | qwen    | 0.002          | 0.165    | 27.5%       | ✗              |

Read that table as three findings. **Meaning distillation is the disentangler.** Turning it on under band power (exp8 → exp9) cut subject-variance ~10× (10.1% → 0.9%) and 2.25×'d retrieval (0.019 → 0.043), and it is the only change that made a new brain snap into the shared frame (anchor-calibration lift +0.084 vs ≈0). **The text encoder matters**: swapping E5 for a Qwen target collapses retrieval to 0.002 — the `text_encoders/` A/B asks whether that is E5 specifically or the retrieval-tuned family. And **the encoder is the open frontier**: the raw conformer already holds the richest space on the board (eff-rank 0.264, best content probes, best held-out category decode) but, with meaning distillation *off*, never disentangles subject — so its retrieval languishes. `clip_e5_meaning_raw` (exp10) fills that empty 2×2 cell; `clip_e5_meaning_raw_v2` pushes the encoder itself.

### The full 12-subject LOSO sweep (exp8, 2026-07-24) — the honest trend

A complete leave-one-subject-out sweep of `clip_e5_bandpower` (meaning off) over all 12 subjects makes one thing unavoidable: **the per-fold "sentence Top-1" in `INDEX.md` is the POOLED number, dominated by the 11 subjects the model trained on, and it is not the model's generalisation.** Read it with `zte-loso-summary`, which reports the honest held-out metric instead:

- **Pooled retrieval swings 0.0015 → 0.131 across folds** (mean 0.061 ± 0.052) — but this is mostly training instability, not generalisation. Convergence was **bimodal: 5/12 folds trained to a healthy subject-invariant code, 3/12 collapsed** (pooled < 0.01, subject identity never removed). A single seed per fold cannot separate "hard subject" from "unlucky seed" — hence the new `SEEDS="42 43 44"` option on `scripts/run_loso.sh`.
- **Held-out retrieval — the honest headline — is essentially chance.** On the genuinely never-seen subject, Top-1 lift over chance is **+0.0017 ± 0.0030** (6/12 folds at or below chance). The correct match does rank around the 91st percentile on average, so *weak* signal exists, but it is nowhere near Top-1. The model does **not** yet retrieve a stranger's sentence.
- **What *does* generalise honestly:** held-out **category decode** beats chance in 10/12 folds (0.64 vs 0.54), and **anchor calibration helps in 12/12** (cohesion lift +0.04 … +0.16) — a new brain can be snapped into the shared frame from ~12 anchor words without retraining. That anchor result is the most promising lever for the decoder roadmap.

### What is still open

State these next to any result from this directory; they are in every `evaluation/report.md`.

- **The content-probe positive control now probes genuinely-raw band power** (fixed 2026-07-24). It previously read the model's *normalised* input, and a whitening normaliser (riemannian/zscore_subject) strips the amplitude that word-length and frequency ride on — so it read R² ≈ −0.008 and falsely branded the whole content probe broken. It now probes the untouched `(bands × channels)` band power, so a passing control means "content 0%" is a real absence rather than a measurement artefact. Re-run any older eval to get the corrected control.
- **The held-out number is well below the pooled headline** (see the 12-fold trend above). A richer, subject-invariant encoder is the lever with the most headroom left; the exp10 arms target exactly this.
- **Analogy/vector arithmetic is still at chance.** Cancelling *who* produced a thought (`emb(t,A) − centroid(A) + centroid(B)`) does not yet retrieve the same token for another subject. A cleaner ZTE-space is the path there.

---

## `flagship/` — start here

| Config                   | Objective                                  | Encoder                                                                   | Why it is here                                                                                                                |
| ------------------------ | ------------------------------------------ | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `clip_e5_meaning`        | sentence-level CLIP + meaning distillation | band-power MLP + spherical harmonics                                      | **The champion** (exp9). Best cross-subject retrieval measured here (Top-1 0.043); meaning distillation cut subject-var 10×.  |
| `clip_e5_bandpower`      | sentence-level CLIP (symmetric InfoNCE)    | band-power MLP + spherical-harmonic spatial encoding                      | The pre-meaning champion (exp8); the meaning-distillation-off cell of the 2×2.                                                |
| `clip_e5_raw`            | sentence-level CLIP                        | raw-conformer (temporal→spatial convolution, ~700 ms window)              | Richest space on the board (eff-rank 0.264, best content probes) but subject-entangled without meaning distillation.          |
| `clip_e5_meaning_raw`    | sentence-level CLIP + meaning distillation | raw-conformer, ~700 ms window                                             | **exp10 — the missing 2×2 cell:** the raw frontend under the champion objective. Does richer content + invariance beat 0.043? |
| `clip_e5_meaning_raw_v2` | sentence-level CLIP + meaning distillation | raw-conformer + multiscale temporal bank + attentive temporal pool, wider | **exp10 — the encoder leap:** the same objective, a deliberately richer per-word encoder. Run *after* `clip_e5_meaning_raw`.  |

All are EEG-only (an honest "thought, not gaze" choice — eye-tracking is a reading artefact absent from imagined thought), LOSO, and Riemannian-normalised per subject. The two exp10 arms share one `run_name` prefix (`exp10_…`) so they group as one study; run the plain `clip_e5_meaning_raw` first so the frontend-swap effect and the architecture effect stay separable.

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

## `text_encoders/` — what makes E5 unique

Every arm here is the champion recipe (`clip_e5_meaning`) with **only** the CLIP sentence target changed, so any delta isolates the frozen text encoder. Run them together with `STUDIES=text_ab bash scripts/run_suite.sh` (adds E5 and Qwen from the tiers above for the full four-way).

| Config               | Text target                               | What it isolates                                                                       |
| -------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------- |
| `clip_bge_meaning`   | BAAI/bge-base-en-v1.5 (retrieval-tuned)   | The retrieval-contrastive *family* — if BGE ≈ E5, the win is the pretraining style.    |
| `clip_mpnet_meaning` | all-mpnet-base-v2 (NLI/paraphrase, 768-d) | A strong non-retrieval encoder — if E5/BGE beat it, retrieval pretraining is the edge. |

The champion (`flagship/clip_e5_meaning`, E5) and `benchmark/clip_qwen_bandpower` (Qwen) are the other two arms of the same A/B. To add another encoder, copy `clip_bge_meaning.yaml` and change only `objective.text_source` (and `text_query_prefix` if the model needs one — E5/BGE queries do, MPNet does not).

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

Rotates the held-out subject over the whole 12-person cohort for one config, turning a single number into a generalisation trend. On completion it writes **`LOSO_SUMMARY.md`** (the honest held-out headline + convergence spread, via `zte-loso-summary`) alongside `COMPARE.html`. Defaults to the champion.

```sh
bash scripts/run_loso.sh /path/to/zuco_extracted                          # champion, all 12 subjects
SEEDS="42 43 44" bash scripts/run_loso.sh /path/to/zuco_extracted         # 3 seeds/fold -> mean±std, exposes instability
FULL_CFG=experiments/flagship/clip_e5_raw.yaml bash scripts/run_loso.sh   # a different arm
SUBJECTS="ZAB ZDM" bash scripts/run_loso.sh /path/to/zuco_extracted       # a subset
CONTROL=1 bash scripts/run_loso.sh /path/to/zuco_extracted                # also run the skip-gram control
```

Aggregate any existing sweep on its own with `uv run zte-loso-summary --experiments res/experiments/loso` — it reads every fold's `metrics.json` and reports the held-out lift over chance (mean ± std), how many folds beat chance, the converged/collapsed split, and the anchor-calibration lift. This is the number to quote for LOSO, **not** the pooled `sentence Top-1` in `INDEX.md`.

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
- **The dataset is processed once, ever.** The cache is layered (a fast local copy backed by a persistent Drive one) and two-level (the expensive `.mat` extraction is cached separately from the cheap processing). A freshly built bundle is published to Drive *immediately*, so a reclaimed VM never re-processes; a new config that changes only normalisation, imputation, eye-tracking or length filters reuses the extraction and re-derives in seconds instead of re-parsing every `.mat` file. Point every command at the persistent store once with `ZTE_CACHE_REMOTE` (or `--data-cache-remote`); synthetic and real data can never collide in it.

```sh
# Process every dataset the shipped experiments need, once — then every run below starts warm:
uv run zte-prepare --root /content/zuco_extracted --configs \
    --cache-dir res/cache/prepared \
    --cache-remote "/content/drive/MyDrive/Sharables/ZTE/prepared"

export ZTE_CACHE_REMOTE="/content/drive/MyDrive/Sharables/ZTE/prepared"   # every command now reads/writes it
DRIVE_BACKUP="/content/drive/MyDrive/Sharables/ZTE/$(date +%F)/experiments" \
DATA_CACHE="res/cache/prepared" \
bash scripts/run_loso.sh /content/zuco_extracted
```

If the VM is reclaimed: copy the Drive folder back to `OUT_ROOT` (or point `OUT_ROOT` straight at Drive) and re-run the identical command. The processed dataset is already on Drive, so it is never rebuilt.

## Reproducibility

Every config fixes `train.seed` and sets `train.deterministic: true`. `zte-run` copies the fully-resolved `config.yaml` into the run directory, so any run reproduces exactly:

```sh
uv run zte-run --config res/experiments/<run_name>/config.yaml --root <data> --name <run_name>
```

## Catalogue

`res/experiments/INDEX.md` accumulates one row per run so runs are comparable at a glance. Each run's own `README.md`, `manifest.json`, `evaluation/report.md` and interactive `evaluation/interactive/held_out_scoreboard.html` hold the full configuration, data source and verdict.  Compare any set of runs with `uv run zte-compare --experiments res/experiments`.
