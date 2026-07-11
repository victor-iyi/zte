# ZTE experiment suite

A rigorous, **bias-controlled** set of studies you can run end-to-end on the real ZuCo data, plus a "how to read every output" guide so a newcomer can peek into what ZTE actually learns. Each study is an **A/B (or a sweep) with everything else held identical**, so any metric delta is attributable to the one lever under test — not a confound.

> Related: [EVALUATION.md](./EVALUATION.md) (what every metric means and how the verdict is computed) · [ARCHITECTURE.md](./ARCHITECTURE.md) · the shipped `experiments/exp1..exp6*.yaml` presets.
>
> **Design vs result.** Everything in the *Expected / interesting* column below is a **hypothesis**, not a measured result. The suite is designed and verified on synthetic data; the real-data numbers are what you produce by running it. Tiny synthetic runs will *legitimately fail* several verdict checks (there is little real cross-subject signal to find) — that is the point of running it at scale.

## What ZTE is trying to do

ZTE turns per-word EEG into a **frozen, re-purposable embedding space** without training a decoder. The north-star question is whether it encodes **WHAT** was read (lexical / semantic content that transfers across people) rather than **WHO** read it (subject identity — the cheap shortcut). The suite is built to answer that honestly.

## Anti-bias / anti-overfitting principles (baked into every study)

These are enforced in the config files (`scripts/make_study_configs.py`) and the runner (`scripts/run_suite.sh`), and they are the reason the numbers are trustworthy:

- **Leakage-aware splits.** In-distribution generalisation uses `split: by_stimulus` — the same sentence *text* never appears in both train and test. Subject generalisation uses `split: by_subject_loso` — a whole subject is held out. We never use `random`/`by_sentence` for headline claims.
- **Always held-out evaluation.** `test_fraction > 0` (0.1) for stimulus/sentence splits; for LOSO the held-out subject *is* the test set. No headline number is scored in-sample.
- **Normaliser fit on train only.** `normalizer_fit: train` — the z-score / imputer statistics are estimated on the training split, never on val / test / the held-out subject. No leakage through preprocessing.
- **Regularisation against collapse & overfit.** VICReg variance + covariance terms (`variance_weight`, `covariance_weight`) stop the InfoNCE objective from collapsing into ~15 of 768 dimensions; plus dropout (`model.dropout`), AdamW weight decay (`train.weight_decay`), and (for the masked objective) an EMA teacher. Every run reports train vs val vs test so you can *see* the gap.
- **Multiple seeds → confidence intervals.** Run each study at seeds **42, 43, 44**. The evaluator already computes bootstrap 95% CIs on the load-bearing quantities (`beats_noise_ci`, `retrieval_ci`, `subject_arithmetic_ci`); seeds turn a single-run point into a distribution so differences carry error bars, not noise.
- **The eye-tracking confound is controlled by making EEG-only the honest headline.** Gaze scalars (fixation durations, pupil size) trivially encode word length and frequency. Every invariance / collapse study sets `include_eye_tracking: false` so a "reading-from-EEG" claim can't be a gaze artefact; Study 1 measures the size of that artefact directly.

## The studies

| #   | Study                                              | Hypothesis                                                 | Config(s)                                                                                                                                                                                     | What to look at                                                                                                                                                                 | Expected / interesting (hypothesis)                                                                                                                                               |
| --- | -------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Purpose / eye-tracking confound**                | "Reading from EEG" survives without gaze features          | Matched A/B: `zte-benchmark --eye-tracking both` (everything else identical). Full-scale flagships: `exp1_skipgram_rope_et.yaml` (ET on) vs `exp6_skipgram_eegonly_invariant.yaml` (EEG-only) | `comparison.csv` word-length / log-freq linear probe (ZTE vs raw vs noise); `region_importance.csv` EEG-vs-eye-tracking share; sentence retrieval Top-1                         | EEG-only keeps lexical structure **above the noise floor** → not purely a gaze artefact. The ET→EEG-only drop *quantifies* how much gaze was carrying.                            |
| 2   | **Subject-invariance A/B (LOSO — the north star)** | The invariance stack makes the code subject-agnostic       | `study_invariance_baseline_loso.yaml` (no levers) vs `study_invariance_full_loso.yaml` (VICReg + per-subject norm + cross-subject positives + adversary), same held-out subject               | `metrics.json` `analogy.subject_transfer` lift; the **subject** row of `comparison.csv` (probe accuracy should fall toward chance); held-out cross-subject `sentence_retrieval` | **Discovery question:** does the stack push subject-decodability toward chance **and** lift held-out cross-subject retrieval? A win means WHO is removed without destroying WHAT. |
| 3   | **Anti-collapse ablation (VICReg OFF vs ON)**      | VICReg prevents dimensional collapse and revives content   | `study_vicreg_off.yaml` vs `study_vicreg_on.yaml` (identical except `variance_weight`/`covariance_weight` 0→1), `by_stimulus`                                                                 | `metrics.json` `embedding_health.effective_rank_ratio`; `neurons` block `n_active`/`n_dead`/`variance_budget`; word-length & log-freq probes in `comparison.csv`                | **Discovery question:** how many neurons "come alive" (effective rank, `n_active`) and does lexical content become decodable once collapse is prevented?                          |
| 4   | **Objective sweep**                                | Some SSL objective carries the most transferable structure | `zte-benchmark --objectives skipgram,cbow,masked,cpc --eye-tracking off --pos-encodings rope --seeds 42,43,44`                                                                                | `benchmark.csv` / `benchmark.md`, sorted by `subject_transfer_lift`; also `sent_retrieval_lift`, `eff_rank_ratio`, `beats_noise`                                                | Which of skip-gram / CBOW / masked / CPC gives the highest subject-agnostic transfer. No prior — this is exploratory.                                                             |
| 5   | **Representation (optional)**                      | A raw temporal frontend can rival hand-made band-power     | `exp5_raw_conformer_masked.yaml` (raw conformer) vs `exp2_masked_rope_eegonly.yaml` (band-power); both masked, EEG-only                                                                       | Sentence/word retrieval, `effective_rank_ratio`, probes in `comparison.csv`                                                                                                     | Does learning the band-pass (`raw_conformer`) beat fixed band-power, or is band-power a strong-enough summary?                                                                    |

### New config files created for this suite

Generated by `scripts/make_study_configs.py` (build `ZTEConfig` objects → `to_yaml`, so they are guaranteed valid against the current schema). Studies 1/4/5 need no new files — they reuse the shipped presets or `zte-benchmark`.

| File                                              | One-line purpose                                                                                                         |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `experiments/study_invariance_baseline_loso.yaml` | Study 2 (A): skip-gram, EEG-only, LOSO, **no** invariance levers — the control.                                          |
| `experiments/study_invariance_full_loso.yaml`     | Study 2 (B): same, **with** the full invariance stack (VICReg + per-subject norm + cross-subject positives + adversary). |
| `experiments/study_vicreg_off.yaml`               | Study 3 (A): skip-gram, EEG-only, `by_stimulus`, **VICReg off** (`variance_weight=covariance_weight=0`).                 |
| `experiments/study_vicreg_on.yaml`                | Study 3 (B): identical to (A) except **VICReg on** (`=1`).                                                               |

All four share: `subjects: null` (all 12 subjects), `tasks: [SR, NR]`, `representation: band_power`, `include_eye_tracking: false`,
`normalizer_fit: train`, `test_fraction: 0.1`, `deterministic: true`, `seed: 42`.  Only the levers named in the table differ between an A/B pair.

## Running it

All commands assume the repo root and the project venv (`.venv/bin/...`). The console scripts (`zte-run`, `zte-benchmark`, `zte-visualize`) are installed in the venv; you can equally call `.venv/bin/python -m zte.cli.run` etc.

The fixed, documented driver is `scripts/run_suite.sh` (seeds `42 43 44`):

```sh
bash scripts/run_suite.sh                       # real data at res/data/zuco_extracted
bash scripts/run_suite.sh /path/to/zuco_extracted   # a different data root
```

`zte-run --seed <N>` overrides `train.seed` and suffixes the run name with `_s<N>` (e.g.  `study_vicreg_on_s43`), so a multi-seed sweep is just a shell loop: `for s in 42 43 44; do zte-run --config <cfg> --root <root> --seed $s; done`. The `run_suite.sh` runner does exactly this; the individual commands below show the single-seed form.

### Pause & resume (stop and continue any run at any time)

Long runs are interruptible. **Stop** a run whenever you like — press `Ctrl-C`, or `kill <pid>` a backgrounded one — and it pauses cleanly, leaving a checkpoint at the last completed epoch. **Continue** by re-running the *same command* with `--resume`:

```sh
# start (optionally in the background)
nohup zte-run --config experiments/study_vicreg_on.yaml --root res/data/zuco_extracted --seed 42 > run.log 2>&1 &
# ...pause it any time...  kill %1   (or Ctrl-C if in the foreground)
# ...later, continue exactly where it left off:
zte-run --config experiments/study_vicreg_on.yaml --root res/data/zuco_extracted --seed 42 --resume
```

`--resume` is idempotent and safe to always pass: it reuses the cached dataset bundle (skips the slow prepare), continues training from the last checkpoint, and skips evaluation / exploration that are already up to date (a `--force` flag redoes them). Because `scripts/run_suite.sh` **already passes** `--resume` **on every run**, you can stop the whole suite at any point and just re-run `bash scripts/run_suite.sh` to pick up where it stopped — completed runs are skipped, the interrupted one resumes. (`zte-benchmark` sweeps are not individually resumable; re-running restarts that sweep.)

### Fast synthetic smoke (no data — proves a config trains + evaluates)

Every study config runs end-to-end on fabricated data in a couple of minutes.  Use this to sanity-check a config before committing a multi-hour real run:

```sh
# Study 2/3 configs:
.venv/bin/python -m zte.cli.run --config experiments/study_vicreg_on.yaml \
    --synthetic --epochs 2 --name smoke --device cpu
.venv/bin/python -m zte.cli.run --config experiments/study_invariance_full_loso.yaml \
    --synthetic --epochs 2 --name smoke --device cpu
# Study 4 sweep:
.venv/bin/zte-benchmark --synthetic --objectives skipgram,cpc --pos-encodings rope \
    --eye-tracking off --seeds 42 --epochs 2 --out res/benchmark_smoke
# Explorer:
.venv/bin/zte-visualize --synthetic --out res/explorer/smoke.html
```

### Study 1 — eye-tracking confound

```sh
# Clean matched A/B (same model, seeds, split; only include_eye_tracking flips):
.venv/bin/zte-benchmark --root res/data/zuco_extracted \
    --objectives skipgram --pos-encodings rope --eye-tracking both \
    --seeds 42,43,44 --out res/benchmark/et_confound
# Full-scale flagships (report both; note they differ in more than ET, so the
# benchmark above is the clean confound test and these are the headline runs):
.venv/bin/zte-run --config experiments/exp1_skipgram_rope_et.yaml --root res/data/zuco_extracted
.venv/bin/zte-run --config experiments/exp6_skipgram_eegonly_invariant.yaml --root res/data/zuco_extracted
```

### Study 2 — subject-invariance A/B (LOSO)

```sh
# Single held-out subject (ZAB by default in the config), one seed:
.venv/bin/zte-run --config experiments/study_invariance_baseline_loso.yaml --root res/data/zuco_extracted
.venv/bin/zte-run --config experiments/study_invariance_full_loso.yaml --root res/data/zuco_extracted
```

For a **full** leave-one-subject-out sweep, rotate `train.loso_holdout_subject` across all 12 subjects (ZAB ZDM ZDN ZGW ZJM ZJN ZJS ZKB ZKH ZKW ZMG ZPH) — the runner shows the loop. Compare the baseline vs full curves subject-by-subject.

### Study 3 — anti-collapse ablation

```sh
.venv/bin/zte-run --config experiments/study_vicreg_off.yaml --root res/data/zuco_extracted
.venv/bin/zte-run --config experiments/study_vicreg_on.yaml  --root res/data/zuco_extracted
```

### Study 4 — objective sweep

```sh
.venv/bin/zte-benchmark --root res/data/zuco_extracted \
    --objectives skipgram,cbow,masked,cpc --pos-encodings rope \
    --eye-tracking off --seeds 42,43,44 --out res/benchmark/objective_sweep
```

### Study 5 — representation (optional)

```sh
.venv/bin/zte-run --config experiments/exp5_raw_conformer_masked.yaml --root res/data/zuco_extracted
.venv/bin/zte-run --config experiments/exp2_masked_rope_eegonly.yaml  --root res/data/zuco_extracted
```

### Peeking into a finished run

```sh
# The flagship interactive Thought-Space Explorer over a run's real embeddings:
.venv/bin/zte-visualize --run res/experiments/study_vicreg_on --out res/explorer/study_vicreg_on.html
```

## How to read the outputs

Every `zte-run` catalogues one self-contained directory under `res/experiments/<run_name>/`. The evaluation artifacts (all under `evaluation/`) are the ones you read to answer the studies.

### `metrics.json` — the machine-readable everything

The full metric dump (also returned by `evaluate_representation`). Key blocks:

- `verdict` — the pass/fail booleans and their statistics. Each check is
backed by a **bootstrap 95% CI, not a sign**:
  - `beats_noise` + `beats_noise_ci` — the paired per-fold (ZTE − noise) probe gap must clear an **effect-size floor** (`effect_size_floor = 0.01`), not merely be positive.
  - `retrieval_above_chance` + `retrieval_ci` — CI on `(Top-1 − chance)` over the per-query hit vector must exclude zero. **Chance is query-weighted** (matches how hits are scored); the old type-weighted value is kept as `chance_top1_typeweighted` and typically *understated* chance by ~30×.
  - `subject_arithmetic_above_chance` + `subject_arithmetic_ci` — same idea for the subject vector-arithmetic test.
- `embedding_health` — geometry / collapse metrics: `effective_rank_ratio` (high ≈ healthy, near 1.0; low ≈ collapsed), `anisotropy` (low is good), uniformity, alignment, dead-dim fraction. **The headline for Study 3.**
- `sentence_retrieval` **/** `word_retrieval` — leave-one-out cross-subject retrieval: does the *same stimulus read by a different person* retrieve its counterparts? `top1`, `mrr`, and `chance_top1` (query-weighted). **The honest cross-subject test — headline for Studies 1 & 2.**
- `analogy` — vector arithmetic `emb(t, A) − centroid(A) + centroid(B)` should retrieve `emb(t, B)`. `subject_transfer` / `task_transfer` Top-1 vs chance. **The falsifiable subject-agnosticism test — headline for Study 2.**
- `probe_comparison` — the ZTE-vs-raw-vs-noise probe rows (mirrored to `comparison.csv`).
- `neurons` — the compact neuron-interpretability block (full detail in `neurons.json`); see the who-vs-what budget below.

### `report.md` — the human-readable version

A Markdown narrative of the same numbers ending in the verdict checklist, with the figures embedded. Read this first to get the story; drop to `metrics.json` for exact values and CIs.

### `comparison.csv` — the probe table

One row per (representation × target). Columns include `representation` (`ZTE` / `raw` / `noise`), `target` (`word_len`, `log_freq`, `subject`, `task`), `linear_score`, `knn_score`, and a baseline. **Reading it:** ZTE should beat the `noise` control and rival `raw` band-power in far fewer effective dimensions. In Study 2 you *want* the `subject` row's ZTE score to fall (identity becoming hard
to decode); in Study 3 you *want* `word_len` / `log_freq` to rise.
(The per-fold `linear_scores` arrays that back the CIs stay in `metrics.json`; the CSV keeps one scalar per cell.)

### `breakdown.csv` — stratified, so a good average can't hide a bad subject

The same metrics recomputed **per subject / per task / per category**. Scan for a subject or task that collapses while the global number looks fine — that is the kind of hidden failure the suite is designed to surface.

### `region_importance.csv` — where on the scalp the information lives

Each anterior→posterior scalp region's share of the decodable information, per target (reading targets: word length, frequency; cognitive targets: task, subject). **Caveat:** without a montage CSV the region map is an *approximate* coordinate-free proxy — the run flags `region_map_approximate: true` and labels are indicative. Supply `dataset.montage_csv` for exact channel→region mapping.
This is where Study 1's "EEG vs eye-tracking share" is quantified.

### `neurons.json` + the `neurons` block in `metrics.json` — the neuron atlas data

`neurons.json` holds the full per-dimension interpretability report; the compact summary is embedded in `metrics.json` under `neurons`. This is how you see *which* of the 768 neurons fire, *what* they encode, and which are negligible:

- `variance_budget` — how the embedding's total variance is split across what each active neuron is *most selective for* (`subject`, `task`, `word_len`, `log_freq`, `category`, or `none`). This is the **"budget" framing**: the space has a fixed variance budget and the budget tells you what it spent it on.
- `who_vs_what_ratio` (with `who_variance` / `what_variance`) — variance spent on **identity** (subject/task = *who*) divided by variance spent on **content** (word length/frequency/category = *what*). **Lower is better** — the north-star number in one scalar. Watch it fall from Study 2 baseline → full.
- `n_active` **/** `n_dead` (and `embed_dim`, `active_variance_share`) — how many neurons carry real spread vs are effectively constant. **The headline pair for Study 3:** VICReg should raise `n_active` and shrink `n_dead`.
- `neuron_budget` — the same split counted in *neurons* rather than variance.
- `top` (in the metrics block) / `top_neurons` (full, in `neurons.json`) — the most-important dimensions with their dominant attribute, variance share, and top-activating words, plus per-neuron activation histograms and (with band power) scalp/band attribution. Read these to name what a neuron "means".

#### What makes a neuron "fire", and *importance to what*?

A neuron here is one of the 768 output dimensions. It **"fires" when it *varies* across words** — a dimension that outputs nearly the same number for every word carries no information (it's *dead*). So the default **importance = the share of the embedding's total variance that dimension carries**:

```text
var_share[d] = std[d]² / Σⱼ std[j]²          # importance(d), sums to 1 over all dims
```

`rank 0` is the highest-variance neuron; the *active threshold* line separates neurons that fire from the near-constant tail. This importance is **not "to" any label** — it's how much the neuron is *used at all*. That's a deliberate first question ("is the space even being used?", i.e. the collapse story).

**Importance *to a specific attribute*** is a different, equally valid question, and it's exposed too. Selectivity is the *fraction of a neuron's variance explained* by each attribute — `r²` for continuous targets (word length, log-freq) and `eta²` for categorical ones (subject, task, category), both on a 0–1 scale (`neurons.json` → `selectivity.scores`). `neurons.json` → `importance.rankings` gives you the dimensions **re-ranked by each axis**: `variance` (the default), and `selectivity:subject`, `selectivity:word_len`, etc. So to ask *"which neurons matter most for subject identity?"* you read `rankings["selectivity:subject"]` (or, in the atlas, switch the **sort-by** control to that target). To make the atlas behave differently, change the sort/colour axis (variance ↔ a target), toggle *hide-dead*, or point `neuron_report(..., top_k=…, active_quantile=…)` at a different definition of "active".

### The interactive explorers (`evaluation/interactive/`)

- `thought_space_explorer.html` — the flagship, fully-offline 3-D explorer over word embeddings: one subject / many words, one word across many brains (with a cross-subject cosine statistic), thought arithmetic `emb(t,A) − centroid(A) + centroid(B)`, an EEG-only-vs-eye-tracking toggle, and live colour / subject / word / dimension controls. Open in any browser — no server. Rebuild a standalone copy over a catalogued run with `zte-visualize --run res/experiments/<name>`.
- `word_explorer.html` — the simpler PCA explorer (a lightweight fallback view of the same embeddings).

### The neuron atlas — which neurons fire, and what they encode

Every evaluation writes an interactive `interactive/neuron_atlas.html` (path recorded under `metrics.neuron_atlas`), and you can regenerate it standalone:

```sh
.venv/bin/zte-visualize --atlas --run res/experiments/<name> --out res/atlas/<name>.html
# or both the thought-space explorer and the atlas at once:
.venv/bin/zte-visualize --kind both --run res/experiments/<name> --out res/viz/<name>.html
```

Read it as: a **ranked-importance chart of every dimension** (the tall bars on the left are the neurons that fire; the flat tail past the dashed *active threshold* line are the negligible / dead ones), coloured by each neuron's **dominant attribute** — amber for *who* (subject), the cool hues for *what* (word length, log-frequency, task, category), grey for none. Click any bar to open its **detail panel**: a selectivity bar (how strongly it tracks each attribute), its activation histogram, the top / bottom-firing words, and a scalp band × region attribution. The header tiles give the headline **who-vs-what variance budget**. The underlying numbers are all in `neurons.json` and the compact `neurons` block of `metrics.json`, so the analysis is reproducible without the HTML.

### Emergent properties — do similar thoughts cluster across people?

This is the north-star property (the thing that would make ZTE "like word embeddings for brains"): the **same or related meaning read by different subjects should sit together**, and higher-order arithmetic (`emb(t,A) − centroid(A) + centroid(B) ≈ emb(t,B)`) should work. Every run measures it honestly under `metrics.json` → `emergence` (and a "Emergent properties" section in `report.md`):

- `cross_subject.same_word` — mean cosine of the *same word read by different subjects* vs a random cross-subject baseline. The `gap` (same − random) is the real signal; a collapsed/anisotropic space makes all raw cosines high, so ignore the absolute number and read the gap and `verdict`.
- `cross_subject.same_meaning` — the same test using the sentence **category** as a meaning proxy: are same-category cross-subject pairs closer than random? This is the "cat near dog because both are animals, across people" test, grounded in the labels the corpus actually has.
- `neighbourhood` — for a sample of tokens, the fraction of nearest neighbours that are the same word / same category (vs chance), and the fraction drawn from a **different subject**. A working thought code wants **positive category coherence and a high cross-subject neighbour fraction**.

**How to *see* it, without guessing:** open `thought_space_explorer.html`. Its landing banners state, in plain language, whether same/related thoughts cluster; its **auto-analogy leaderboard** finds the working `A→B` analogies for you (no need to pick a word or a person); and its **neighbourhood view** shows a word's nearest thoughts across subjects.

**How to *make* it emerge (and which config):** ZTE v1 largely does **not** have these properties yet — the space encodes *who*. The levers designed to produce them are all in `study_invariance_full_loso.yaml` / `exp6_skipgram_eegonly_invariant.yaml`: **cross-subject positives** (same stimulus across subjects pulled together), a **subject adversary** + **per-subject normalisation** (remove identity), and **VICReg** (stop collapse so there's room for content). Study 2 (baseline → full) is precisely the A/B that shows whether the `emergence` gaps *move*. Turning the knobs up: raise `objective.subject_adversary_weight`, keep `cross_subject_positives: true`, raise `variance_weight`/`covariance_weight`, and use `normalize: zscore_subject`. Ultimately the parent project's LLM alignment (EEG-OT-CLIP) is what maps ZTE into a genuinely semantic space — this suite tells you how good a starting point ZTE is.

## Avoiding bias & overfitting — the guarantees, made explicit

- **No text leakage** across train/test (`by_stimulus`) for in-distribution claims; **no subject leakage** (`by_subject_loso`) for cross-subject claims.
- **No preprocessing leakage** — normaliser and imputer are fit on train only (`normalizer_fit: train`).
- **No in-sample scoring** — `test_fraction > 0`; LOSO scores the untouched held-out subject.
- **No silent collapse** — VICReg + weight decay + dropout, and `effective_rank_ratio` / `n_dead` are reported so you can *see* the space is healthy, not just assume it.
- **No single-run luck** — seeds 42/43/44 and bootstrap CIs on every verdict check; a difference has to clear an effect-size floor and a CI, not a sign.
- **No gaze shortcut in the headline** — the invariance and collapse studies are EEG-only; Study 1 measures the gaze contribution rather than hiding it.
- **Matched A/Bs** — every comparison changes exactly one lever; the config generator enforces this by sharing a base config between each pair.

## The LOSO "new brain" experiment (the decisive test) — `scripts/run_loso.sh`

Study 2 above is a single held-out subject (ZAB). The **LOSO sweep** rotates the held-out subject over the *whole cohort* so one data point becomes a **trend** — the honest answer to "does the invariance recipe transfer to brains it has never seen?".

```sh
SMOKE=1 bash scripts/run_loso.sh                 # fast synthetic dry-run (CPU, no data)
bash scripts/run_loso.sh res/data/zuco_extracted # real sweep (auto-GPU, multi-hour)
CONTROL=1 bash scripts/run_loso.sh <root>        # add the no-recipe control arm (clean A/B per subject)
```

- **Portable & auto-GPU.** Runs unchanged on macOS (MPS), Linux (CUDA), and Google Colab; `--device auto` picks the accelerator. See [RUNNING.md](./RUNNING.md) and `[notebooks/zte_colab.ipynb](../notebooks/zte_colab.ipynb)`.
- **Fully resumable.** Every per-subject run carries `--resume`; stop with `Ctrl-C` and re-run the identical command to continue exactly where it stopped (finished subjects skipped, the interrupted one resumed from its last checkpoint).
- **One combined view.** The sweep ends by building `res/experiments/loso/COMPARE.html` (`zte-compare`) so every held-out subject sits side by side, scored against the same pass/fail rubric.

### The honesty layer (what each run now proves, not just claims)

Every run's `metrics.json` and `report.md` gain a `honesty` block whenever the embedding set spans ≥ 2 subjects (so it is populated for `by_stimulus` runs and, for a LOSO run, for the held-out subject specifically):

- **Permutation null** (`honesty.retrieval_permutation`) — cross-subject retrieval Top-1 against a *label-shuffled empirical null* -> a p-value, not merely an analytic chance line. Feeds `verdict.retrieval_above_chance_perm`.
- **Held-out cross-subject decoding** (`honesty.cross_subject_decode`) — a linear probe trained on N−1 subjects and scored on the held-out one, one fold per subject, per target (category / length band / word length / log-frequency), with a bootstrap CI vs an honest chance baseline. Content that decodes on an unseen brain is the real generalization signal.
- **Anchor calibration lift** (`honesty.calibration`) — fits an orthogonal Procrustes alignment from a few shared **anchor** words to snap a held-out subject into the shared frame, then measures whether same-word cross-subject cohesion improves on *held-out* words. A metrics-side preview of "can a stranger be calibrated in without retraining?", mirroring the explorer's **Calibrate** mode.

## §10 roadmap — implemented improvements

The report's §10 "what to fix next" is now wired into the config and objectives (all off by default; enabled in the flagship recipe configs `exp6_skipgram_eegonly_invariant.yaml` and `study_invariance_full_loso.yaml`).

### Fix the dimensional collapse / the "cone" (§10.1)

The LOSO space became a near-degenerate **cone** (anisotropy ~0.997): rank looked high but no dimension separated content. Two complementary levers:

- `objective.whiten` — ZCA-whitens the exported embeddings at evaluation (centre + decorrelate + equalise variance). Centring removes the shared direction that *is* the cone, so **anisotropy drops from ~0.998 to ~0.00**; because it is label-free, every downstream metric is recomputed on the whitened space, so the report honestly shows whether content survives (on synthetic, un-saturating the cosines lifts the same-word cross-subject gap from ~0 to positive).
- `objective.anisotropy_weight` — a Wang & Isola **uniformity** term that spreads the normalised embeddings over the sphere during training (a mean-direction penalty is a saddle at a perfect cone and cannot break it; pairwise repulsion can).
- Turn **VICReg** up (`variance_weight`, `covariance_weight`) to keep every dimension alive and decorrelated — the effective-rank half of collapse is a *training* matter and only shows on real data (synthetic already uses ~126/768 dims).

A ready-made A/B ablation: `study_anticone_off.yaml` (VICReg only) vs `study_anticone_on.yaml` (VICReg + whitening + uniformity). Compare `embedding_health.anisotropy` and `effective_rank_ratio`.

### Kill the stimulus shortcut, chase meaning (§10.2)

- `objective.meaning_positives` (skip-gram) — also draws positive pairs from the **same content word occurring in different sentences** (subject- and context-agnostic word identity), not only the same stimulus token, so same-*meaning* clustering has room to grow instead of memorising which passage a word came from.
- `objective.stimulus_adversary_weight` — a **second gradient-reversal referee** that predicts *which passage/task* a token came from; the reversed gradient removes the "which of the sentence-sets" shortcut. (Sized by `model.n_tasks`.)

### Report on truly held-out data (§10.3)

- The **LOSO sweep** (`scripts/run_loso.sh`) evaluates every held-out subject; the **honesty block** adds a permutation null, a held-out cross-subject decoder, and the anchor-calibration lift for the held-out subject (see above). The **raw-waveform path** for richer signal already exists as `exp5_raw_conformer_masked.yaml` (`frontend: raw_conformer`).
