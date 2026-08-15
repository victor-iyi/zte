# ZTE experiments

Every file here is a plain, editable [`ZTEConfig`](../src/zte/config/) YAML, sorted into four tiers by what it has actually achieved on real ZuCo. Run any of them end to end with **one command**; everything lands under `res/experiments/<run_name>/` (config, checkpoints, `evaluation/report.md`, figures, the interactive dashboards, and a `manifest.json` verdict).

```sh
# Synthetic smoke test (no data) → a real run → straight from Google Drive:
uv run zte-run --config experiments/flagship/zte_raw_aligned.yaml --synthetic --epochs 3
uv run zte-run --config experiments/flagship/zte_raw_aligned.yaml --root res/data/zuco_extracted --loso-holdout ZAB
uv run zte-run --config experiments/flagship/zte_raw_aligned.yaml --drive <folder-id-or-url> --loso-holdout ZAB
```

| Tier             | What lives there                                                                               |
| ---------------- | ---------------------------------------------------------------------------------------------- |
| `flagship/`      | The recipes that have beaten chance on real ZuCo, plus the encoder arms built on the champion. |
| `decoder/`       | The frozen-LM prefix decoder over a trained encoder, its ablations and its length audit.       |
| `benchmark/`     | The controls a flagship must beat to earn its place.                                           |
| `ablation/`      | Single-lever studies — matched pairs that flip exactly one knob.                               |
| `archive/`       | Superseded or failed arms, kept for the record and for reproducibility.                        |

> **File paths and `run_name` are independent.** A config's run directory is named by the `run_name` inside the YAML, not by its file path, so moving a config between tiers never breaks `--resume` against runs already on Drive.
> The name in the left column below is the *file*; the run directory uses the `run_name` in brackets.

---

## The evidence

Real ZuCo, leave-one-subject-out with `ZAB` held out, 12 subjects / 160,804 words / 8,400 sentences (sessions of
2026-07-24 and 2026-07-25, on Drive under `Sharables/ZTE/`).

**Everything below is scored on the held-out subject only.** That distinction is the whole story. The earlier board
ranked arms by *pooled* retrieval, which is computed over the 11 training subjects as well as the held-out one — so
it rewards memorising the brains you have rather than reaching the one you do not. Re-scored honestly (700 queries,
chance 1/700, exact binomial tail):

| Config (run_name)                                        | Frontend      | Top-5 hits / 700 | *p*    | Eff-rank  | Subject probe (raw baseline) |
| -------------------------------------------------------- | ------------- | ---------------- | ------ | --------- | ---------------------------- |
| `flagship/clip_e5_raw` (`exp8_clip_e5_raw`)              | raw_conformer | **32**           | 7e-16  | 0.264     | 0.45 (0.81)                  |
| `flagship/clip_e5_meaning_raw` (`exp10_..._raw`)         | raw_conformer | **32**           | 7e-16  | 0.264     | 0.41 (0.81)                  |
| `archive/clip_e5_meaning_raw_v2` (`exp10_..._raw_v2`)    | raw_conformer | 19               | 1e-06  | **0.535** | 0.36 (0.81)                  |
| `archive/clip_e5_meaning` (`exp9_clip_e5_meaning`)       | band_power    | 10               | 3e-02  | 0.160     | 0.23 (0.16)                  |
| `archive/clip_bge_meaning`                               | band_power    | 9                | 7e-02  | 0.160     | 0.23 (0.16)                  |
| `archive/clip_qwen_bandpower` (`exp8_clip_qwen`)         | band_power    | 5                | 5.6e-01| 0.170     | 0.22 (0.16)                  |

Read that table as three findings.

**The frontend was the real variable.** The raw conformer beats band power by 4x on the same fold. `exp9`'s famous
Top-1 of 0.043 was pooled; held out it is 4 hits in 700, and an identical re-run gave 2 — run-to-run noise the size
of the effect.

**Band power's "disentanglement" was an artifact of collapse.** Its subject probe of 0.23 was read as invariance,
but the raw band-power features only score 0.16 to begin with: there was almost nothing there to remove. The
effective-rank ratio of 0.160 is the tell — the 768-d space was spanned by ~123 directions. Invariance had been
bought by destroying capacity, and the pooled metric was paying for it. The whole band-power family, and the
E5/Qwen/BGE/MPNet text-encoder A/B built on top of it, is now in [`archive/`](archive/README.md).

**Identity is still unsolved on the winning path, because it was never addressed there.** `dataset.normalize` only
ever applied to band power, so `normalize: riemannian` was a *silent no-op* for every raw run above: the winning arm
trains on unaligned voltages, and its subject probe is still 0.41 against a 0.81 raw-feature baseline. That gap is
what `flagship/zte_raw_aligned` (exp12) closes, with three label-free steps — Euclidean alignment, a subject adapter
driven by a hypernetwork over each person's covariance geometry rather than an ID lookup, and a rank-preserving
identity-orthogonality penalty. See [`docs/SUBJECT_ALIGNMENT.md`](../docs/SUBJECT_ALIGNMENT.md).

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

| Config (`run_name`)                         | Objective                                     | Encoder                                                                          | Why it is here                                                                                                                                         |
| ------------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `clip_e5_raw` (`exp8_clip_e5_raw`)          | sentence-level CLIP                           | raw-conformer, ~700 ms window, 40 filters                                        | **Co-best measured** — 32 Top-5 hits / 700, *p* 7e-16, eff-rank 0.264, best content probes. Subject-entangled: probe 0.45 against a 0.81 raw baseline. |
| `clip_e5_meaning_raw` (`exp10_…_raw`)       | sentence-level CLIP + meaning distillation    | raw-conformer, ~700 ms window, 40 filters                                        | **Co-best measured** — the same 32 hits / 700, with the subject probe pulled to 0.41 by meaning distillation alone.                                    |
| `zte_raw_aligned` (`exp12_zte_raw_aligned`) | + identity orthogonality, over CLIP + meaning | raw-conformer, 40 filters, Euclidean-aligned, subject adapter                    | Rank percentile **0.9672** (0.9635–0.9708), and stable to 0.0002 across seeds. The alignment stack itself is a measured no-op — see below.             |
| `zte_encoder_v3` (`exp16_zte_encoder_v3`) | + predictive residual, cross-reader consensus, length-matched gallery, length projection | as `zte_lexical_raw`, byte-identical | **Not yet measured on real ZuCo.** Four architectural mechanisms after thirteen lever arms landed within noise of each other (2 to 9 hits in 700). See `docs/METHODS.md` §9-12.  |
| `zte_lexical_raw` (`exp14_zte_lexical_raw`) | + token-level lexical alignment               | as `zte_raw_aligned`, byte-identical                                             | **Not yet measured on real ZuCo.** It is here because it is the encoder the v2 decoder is built over; promote or retire it on the next sweep.          |
| `decode_zte_v2` (`exp15_decode_zte_v2`)     | frozen-LM prefix decode, metered and steered  | frozen `exp14_zte_lexical_raw`; frozen `Qwen/Qwen2.5-0.5B`                       | **Not yet measured on real ZuCo.** The rebuilt decoder: a rate-limited conditioning channel and a word-synchronous pointer. See `docs/DECODER.md`.     |

> **Three rows here have no number yet** — `exp14_zte_lexical_raw`, `exp15_decode_zte_v2` and
> `exp16_zte_encoder_v3`. They sit in `flagship/` as the current best-*designed* recipe of each kind, not the
> best-measured one, and this table says so rather than implying a result. The tier rule on this project is measured
> performance; the next sweep either earns them the row or moves them out.

The encoder arms are all EEG-only — an honest "thought, not gaze" choice, since eye-tracking is a reading artefact absent from imagined thought — and all are scored leave-one-subject-out.

### What the 2026-08-13 session settled

That session re-ran the flagship set on one fold (held out on `ZAB`, 700 queries) and scored it on **rank percentile**
with a bootstrap CI. Three things came out of it, and the third is the one to carry forward.

**Rank percentile is the metric that can rank these arms; Top-1 is not.** Two seeds of the same `zte_raw_aligned`
config give rank percentile 0.9672 and 0.9670 — a difference of 0.0002 — while their Top-1 moves 9 hits to 8. Every
comparison on this board that was ever made on Top-1 was made on noise.

**The three retained arms are statistically tied.** `zte_raw_aligned` 0.9672 (0.9635–0.9708), `clip_e5_meaning_raw`
0.9667 (0.9629–0.9705), `clip_e5_raw` 0.9635 (0.9599–0.9673). Overlapping intervals, no champion. `clip_e5_raw` is
the only one whose *length-stratified* Top-1 clears *p* < 0.05 (0.0443, *p* 0.012), which is why it keeps its place
despite the lowest point estimate.

**The exp12 alignment stack does not do anything measurable.** `ablation/exp12_align_off` — the same config with
Euclidean alignment switched off — returns rank percentile 0.9670 against the full stack's 0.9672, effective rank
190.31 against 190.25, and a subject probe of 0.4179 against 0.4180. Agreement to four decimals on every metric is
not a small effect; it is no effect. The stack was built to close the identity gap and, on this fold, it does not.
`zte_raw_aligned` stays in `flagship/` because it is tied for best measured, **not** because its levers are earning
their place — and until an ablation shows one of them moving a number, the honest description of exp12 is
`clip_e5_meaning_raw` with extra machinery attached.

One older caveat still stands: **`dataset.normalize` only ever applied to band power**, so `normalize: riemannian` is
a silent no-op on the two non-exp12 rows, which train on unaligned voltages. That is what `raw_align: euclidean` was
introduced to fix — and what the ablation above shows it does not, in fact, fix anything measurable.

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

## `decoder/` — text out, with the controls that make it readable

The decoder does **not** replace the encoder arms; it consumes one. `train.mode: decoder` loads a trained encoder from
`train.encoder_ckpt`, freezes it, freezes a 0.5B LM (`Qwen/Qwen2.5-0.5B`), and trains only a 226,560-parameter prefix
bridge between them. Nothing else in the run can learn, which is what makes "the output is corpus recall" a checkable
claim rather than a matter of trust: 700 ZuCo sentences cannot be stored in weights that are not being updated.

| Config                          | What it is                                                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `decode_v2_pooled`              | **The baseline the rebuild must beat.** Every v2 knob at its no-op default, i.e. the pooled-prefix decoder.        |
| `decode_v2_ladder_only`         | The semantic rate ladder alone. How little it costs measures how few bits the continuous vector was using.         |
| `decode_v2_evidence_only`       | Word-synchronous evidence alone, over the continuous conditioning vector.                                          |
| `decode_v2_no_length_stage`     | Required companion to the headline: the ladder with no stage reserved for the 5.14-bit word count.                 |
| `decode_v2_bandpower`           | The frontend row of the feature-ablation table — the decoder over a band-power encoder.                            |
| `decode_frozen_e5raw`           | The v1 headline. Bridge only, over `exp8_clip_e5_raw`, on the honest four-cell `by_subject_and_stimulus` split.    |
| `decode_joint_e5raw`            | The encoder unfreezes after 3 stage-A epochs at a tenth of the bridge LR; the CLIP loss stays on as an anchor.     |
| `decode_encoder_only`           | The regression control: `mode: encoder` must still reproduce `exp8_clip_e5_raw`'s history under the same seed.     |
| `decode_nostage0_ablation`      | Required reported ablation — text-only bridge pretraining off. If the delta needs it, Stage 0 was doing the work.  |
| `decode_words_ablation`         | Registered ablation — `conditioning: pooled_plus_words` adds 8 resampled word slots to the 8 pooled ones.          |
| `rebaseline_e5raw`              | The length-confound audit arm: the encoder recipe on the decoder's own split, then scored by `zte-rebaseline`.     |
| `smoke/decode_tiny_mps`         | Wiring only (`lm_source: tiny`, batch 4, 2 epochs, `run_name: smoke_mps`). Always `--synthetic`; never a result.   |

```sh
# The decoder needs a trained encoder; --encoder-ckpt overrides the path in the YAML. Do NOT add
# --loso-holdout: these configs name train.loso_holdout_subject inside by_subject_and_stimulus, and the
# flag would swap in by_subject_loso, which shares every stimulus between train and val. zte-run warns.
uv run zte-run --config experiments/decoder/decode_frozen_e5raw.yaml --root res/data/zuco_extracted \
    --encoder-ckpt res/experiments/exp8_clip_e5_raw_loZAB/checkpoints/best.pt --resume
uv run zte-decode --ckpt res/experiments/exp13_decode_frozen_e5raw/checkpoints/best.pt \
    --root res/data/zuco_extracted --split test
```

**Read the length audit before any decoder number.** On the real 700-stimulus gallery, `H(identity) = 9.4512` bits and
`H(identity | n_words) = 4.3090`, so sentence length alone carries **5.1422 bits** — and ZuCo's word segmentation comes
from eye tracking, so the model gets the word count for free. A length-only oracle at ±2 words scores Top-1 0.0214 /
Top-5 0.0786 / Top-10 0.1371 against the best encoder's 0.0143 / 0.0457 / 0.0886. `zte-rebaseline` reports the whole
3×2 grid (post-processing × gallery) against that floor plus the bit budget; it trains nothing, runs against
checkpoints already on Drive, and gates nothing — it tells you how much of a number is length.

Free-running generation is the **secondary**, expected-null readout, decoded with no reference length and no candidate
set, against five brain-independent controls (`mean_prefix`, `null_prefix`, `phase`, `noise`, and a length-stratified
`mismatch` derangement) plus a true-text-embedding oracle. The **primary** readout is decoder-rescoring retrieval over
the 700-sentence gallery, which is ~9.5 bits of forced choice at 700 queries and is labelled retrieval, never
generation. Full method, verdict gate and the pre-registered expectations: [`../docs/DECODER.md`](../docs/DECODER.md).

## `benchmark/` — the controls

| Config                   | What it controls for                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| `baseline_skipgram_loso` | The previous SOTA recipe (skip-gram + the full invariance stack). Answers "did CLIP earn it?". |

The text-encoder A/B that used to sit here (E5 vs Qwen vs BGE vs MPNet) is in [`archive/`](archive/README.md): every arm was band-power and none reached *p* < 0.07 on the held-out board.

## `ablation/` — one lever at a time

| Config                                                          | The one lever it flips                                                    | Against                 |
| --------------------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------- |
| `exp12_align_off`                                               | `dataset.raw_align` — Euclidean whitening off                             | `exp12_zte_raw_aligned` |
| `exp12_align_fit_train`                                         | `raw_align_fit: train` — the strict no-holdout-calibration variant        | `exp12_zte_raw_aligned` |
| `exp12_adapter_off`                                             | `model.subject_adapter` — the covariance-signature hypernetwork off       | `exp12_zte_raw_aligned` |
| `exp12_orthogonality_off`                                       | `objective.identity_orthogonality_weight` — the decorrelation penalty off | `exp12_zte_raw_aligned` |
| `study_invariance_baseline_loso` / `study_invariance_full_loso` | The whole invariance stack, off vs on, under LOSO                         | each other              |
| `study_vicreg_off` / `study_vicreg_on`                          | The VICReg variance+covariance anti-collapse penalty                      | each other              |
| `exp16_residual_off`                                            | `model.residual_coding` -- no context de-trending                            | `exp16_zte_encoder_v3`  |
| `exp16_consensus_off`                                           | All three cross-reader consensus weights off                             | `exp16_zte_encoder_v3`  |
| `exp16_gallery_off`                                             | `objective.gallery_weight` -- in-batch denominator only                     | `exp16_zte_encoder_v3`  |
| `exp16_gallery_band_off`                                        | `objective.gallery_length_band` -- full gallery, no length matching        | `exp16_zte_encoder_v3`  |
| `exp16_length_projection_off`                                   | `objective.length_projection` -- changes the measurement, not the model   | `exp16_zte_encoder_v3`  |
| `exp14_lexical_off`                                             | Both token-level lexical weights off                                      | `exp14_zte_lexical_raw` |
| `exp14_lexical_reader_off`                                      | The same-word-different-reader half only                                  | `exp14_zte_lexical_raw` |
| `feature_bandpower_mlp`                                         | `model.frontend` — band power instead of raw waveforms                    | `exp14_zte_lexical_raw` |
| `feature_spatial_off`                                           | `model.spatial_encoding` — standard channel indexing, no scalp geometry   | `exp14_zte_lexical_raw` |
| `feature_invariance_off`                                        | All three label-free identity steps at once (the composite row)           | `exp14_zte_lexical_raw` |

The last three are the rows of the **feature-ablation table** `zte-analyze` builds: raw conformer vs band-power
MLP, spherical harmonics vs standard channel indexing, and the invariance recipe on vs off. `feature_invariance_off`
is deliberately the composite — the three `exp12_*` arms above already isolate its parts individually.

Each `exp12_*` arm is the flagship `zte_raw_aligned` recipe with exactly one of its three label-free changes
disabled, so the delta attributable to that change is readable directly. The four `study_invariance_*` /
`study_vicreg_*` files are generated from `ZTEConfig` objects by `scripts/make_study_configs.py`, so they cannot
drift from the schema. For a *new* lever, prefer `zte-ablate` (below) over hand-writing a pair.

## `archive/` — retired, kept for the record

`exp1_skipgram_rope_et`, `exp2_masked_rope_eegonly`, `exp3_cpc_rope_et`, `exp4_skipgram_loso`, `exp5_raw_conformer_masked`, `exp6_skipgram_eegonly_invariant` are the original objective/encoder studies. On the 2026-07-12 real-data sweep every one of them scored a sentence-retrieval Top-1 of 0.0 (permutation *p* ≈ 1.0) with a who-vs-what variance ratio up to 1.0 — the identity-encoding failure mode the invariance stack was built to fix. `exp7_sota_geom_invariance` (learned spatial attention + FiLM subject conditioning) is archived for the opposite reason: it was a serious contender and it failed hardest, retrieving nothing at all (Top-1 0.0, *p* = 1.0). Keep them for reproducing the history; do not start new work from them. The CLIP band-power family — `clip_e5_meaning` (the former "champion", exp9), `clip_e5_bandpower`, `clip_qwen_bandpower`, `clip_bge_meaning`, `clip_mpnet_meaning` — was retired on 2026-07-25 when every run was re-scored on the held-out subject; `archive/README.md` carries the number that retired each one, and `study_anticone_off` / `study_anticone_on` / `study_all_levers` went with them as pre-scoreboard studies.

---

## Running them collectively

### The whole study — `scripts/run_zte_study.sh`

The one command that runs everything a claim has to survive, resumably: the confound audit, the flagship encoder at
several seeds, the decoder and its one-knob arms over that encoder, the feature-ablation table, the length audit
against every checkpoint, and `zte-analyze` at the end.

```sh
SEEDS='42 43 44' bash scripts/run_zte_study.sh res/data/zuco_extracted   # everything but the 12-fold sweep
STAGES='audit encoder loso decoder ablation rebaseline analysis' bash scripts/run_zte_study.sh
STAGES=analysis bash scripts/run_zte_study.sh                            # re-draw from what is on disk
SMOKE=1 bash scripts/run_zte_study.sh                                    # offline wiring check, minutes
```

Every step carries `--resume`, so re-running the identical command after an interruption skips finished work.
See `docs/TRAINING.md` for the stage table and the Drive-mirroring behaviour.

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
    --base-config experiments/flagship/zte_raw_aligned.yaml --loso-holdout ZAB \
    --objectives clip,skipgram,masked,cpc --pos-encodings rope --eye-tracking off \
    --seeds 42 --out res/benchmark --resume
```

### Prove one lever in isolation — `zte-ablate`

```sh
uv run zte-ablate generate --config experiments/flagship/zte_raw_aligned.yaml \
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
