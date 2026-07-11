# Changelog

## §10 roadmap: fix the collapse/cone, chase meaning, and measure it honestly

Implements the report's §10 "what to fix next", plus the interactive/experiment infrastructure to run and read it. All new knobs default off; the flagship recipe configs (`exp6_skipgram_eegonly_invariant`, `study_invariance_full_loso`) turn them on.

- **Dimensional collapse / the "cone" (§10.1).** `objective.whiten` ZCA-whitens the exported embeddings at evaluation — centring removes the shared direction that *is* the cone, dropping anisotropy from ~0.998 to ~0.00, and every downstream metric is honestly recomputed on the whitened space. `objective.anisotropy_weight` adds a Wang & Isola uniformity term (a mean-direction penalty is a saddle at a perfect cone and can't break it — pairwise repulsion can). New A/B: `study_anticone_off/on.yaml`. Covered by `tests/test_collapse.py`.
- **Kill the stimulus shortcut, chase meaning (§10.2).** `objective.meaning_positives` draws skip-gram positives from the *same content word in different sentences* (subject/context-agnostic word identity), not only the same stimulus token. `objective.stimulus_adversary_weight` adds a second gradient-reversal referee that predicts *which passage/task* a token came from (sized by `model.n_tasks`). The data layer gained per-token `word_id` and per-sentence `task_id`.
- **Report on truly held-out data (§10.3).** The LOSO sweep (`scripts/run_loso.sh`, `zte-run --loso-holdout`) evaluates every held-out subject; the evaluation `honesty` block adds a permutation null, a held-out cross-subject decoder, and an anchor-calibration lift for the held-out subject (`zte.evaluation.honesty`, `tests/test_honesty.py`). Portable + auto-GPU + resumable, with `docs/RUNNING.md` and `notebooks/zte_colab.ipynb` (Colab via `uv`).

## Interactive views: comparison dashboard + explorer overhaul

- **`zte-compare`** builds one offline HTML comparing every catalogued run (scorecard matrix, sortable CI table, per-run cards, transparent best-run rubric).
- The **Thought-Space Explorer** was redesigned (icon mode cards, capability strip, insight card, progressive disclosure) and gained **Sentence** (per-reader word-by-word path), **Meaning** (same meaning across everyone), and **Calibrate** (snap a new brain in from anchor words, live Procrustes) modes plus a "remove reader identity" morph; the **Neuron Atlas** gained a scalp head-map; the classic word explorer was restyled to match.

## Pause & resume for long runs

Any run is now interruptible and continuable:

- `Trainer` gains `resume=True`: on a pause it restores model / optimiser / scheduler / AMP-scaler / objective + EMA-teacher / best-metric / history / step from `last.pt` and continues at the next epoch.
- `Ctrl-C` (SIGINT) or `kill` (SIGTERM) pauses cleanly — the last completed epoch is already checkpointed — instead of crashing.
- `zte-run --resume` makes the whole pipeline idempotent: it reuses the cached dataset bundle (skips the slow prepare), resumes training from `last.pt`, and skips evaluation / exploration that are already up to date (re-evaluating automatically if training advanced). `--force` redoes completed stages. `scripts/run_suite.sh` now passes `--resume` on every run, so stopping and re-running the suite continues where it left off. Covered by `tests/test_resume.py` (continuation, no-op, and a real SIGTERM interrupt→resume).

## Interpretability & experiment suite — peering inside ZTE

A follow-on wave focused on *explaining* the representation and running it properly.

### Neuron-level interpretability (`zte.evaluation.neurons`)

Every evaluation now emits a per-dimension "neuron" report — which dimensions fire, what each one encodes, and which matter vs. are negligible:

- **Importance** — per-neuron std and its share of total embedding variance, ranked most-active to dead (near-constant), with an active/dead count. The collapse story at neuron resolution.
- **Selectivity** — for every neuron, |Pearson r| with word length / log-frequency and eta² with subject / task / category; each neuron's *dominant* attribute is its argmax.
- **Who-vs-what budget** (the headline) — the share of the space's *variance* whose dominant attribute is identity (subject) vs content, with a `who_vs_what_ratio`. Quantifies the "encodes who, not what" failure mode per neuron.
- **Exemplars & attribution** — the words that most/least activate each top neuron, plus a correlational band × scalp-region attribution tying a neuron back to the brain (and exposing gaze-driven neurons when eye-tracking is on).
- Artifacts: full `neurons.json`, a compact `neurons` block in `metrics.json`, a "Neurons — what the dimensions encode" section in `report.md`, and an interactive **`neuron_atlas.html`** (ranked importance chart coloured by dominant attribute with the active/dead threshold line; per-neuron selectivity, activation histogram, top-firing words, and scalp attribution) — auto-emitted per run and buildable with `zte-visualize --atlas`.

### Emergent-property metrics + explorer overhaul (`zte.evaluation.emergence`)

Answering "do similar thoughts cluster across people?" — the north-star property — as a measured number, not a picture:

- Every run reports `metrics.emergence`: cross-subject **same-word** and **same-meaning** (category) clustering (same-pair cosine vs random baseline, with the honest *gap* since collapsed spaces make all cosines high), and **neighbourhood coherence** (are a token's nearest neighbours the same word / category, and how many come from a different subject). A plain-language verdict (`clustered` / `weakly` / `not`) plus a `report.md` section.
- The **Thought-Space Explorer** was rebuilt for interpretability: a "What am I looking at?" guide, three headline **verdict banners** (now showing the authoritative full-space emergence numbers with the in-browser PCA figure as a live estimate), an **auto-analogy leaderboard** that finds the working `A→B` analogies for you (no more guessing a word/subject), and a **semantic-neighbourhood** view.
- Neuron importance is now explicit and adjustable: `neurons.json` documents the exact formula (`var_share[d] = std[d]² / Σstd²`) and adds per-target `importance.rankings` (`variance`, `selectivity:subject`, `selectivity:word_len`, …) so you can rank neurons by importance *to a chosen attribute*, not just by how much they fire.

### Rigorous experiment suite (`docs/EXPERIMENTS.md`, `scripts/run_suite.sh`)

A curated, bias-controlled set of studies that maximises the dataset (all 12 subjects, SR+NR), evaluates on held-out data, and isolates each lever:

1. purpose / eye-tracking confound,
2. a LOSO subject-invariance A/B (baseline vs the full invariance stack),
3. an anti-collapse VICReg ablation,
4. an objective sweep via `zte-benchmark`,
5. band-power vs raw-conformer. All runs use leakage-aware splits (`by_stimulus` / `by_subject_loso`), train-only normalisation, VICReg + dropout + weight-decay regularisation, and multiple seeds so differences carry bootstrap CIs. `docs/EXPERIMENTS.md` gives exact commands and a "how to read every output" guide.

## Performance-review response — turning the honest negative result into levers

This release implements every "Areas for improvement" item (P0–P6) from the *Honest Performance & Accuracy Review* plus the shortcomings surfaced by the review's self-evaluation, and adds an interactive Thought-Space Explorer. Each change is documented in code and covered by tests (`tests/test_improvements.py`, plus additions to `tests/test_evaluation.py`).

### A correction to the review (P0)

The review's #1 "blocker" claimed four modules use Python-2 `except A, B:` syntax that is a hard `SyntaxError` under Python ≥3.12, so a clean checkout "cannot import the package". **This is not true for this tree.** The project requires Python ≥3.14, and under Python 3.14's parser (PEP 758) `except A, B:` is accepted and catches *both* exception types correctly — the package imports and the shipped tests pass on a clean checkout. (In fact the project's own `ruff` formatter, targeting py314, *normalises* `except (A, B):` back to the unparenthesised `except A, B:` — the parenthesised form is the redundant one here, not the other way round.) The four sites are correct as written; the review's "blocker" is a false positive. Being open to critique cuts both ways: this static-analysis claim was over-stated, and no clean checkout ever failed to import because of it.

### P1 — Anti-collapse (the biggest metric mover)

- Added a **VICReg variance-hinge + covariance penalty** on the exported embeddings, wired through a shared `_ObjectiveBase` so every objective (skip-gram, CBOW, masked, CPC) gets it (`models/objectives.py::vicreg_terms`, `_ObjectiveBase.regularize`).
- Knobs: `objective.variance_weight`, `objective.covariance_weight`, `objective.variance_target`.
- The variance term keeps each of the 768 dimensions "alive"; the covariance term decorrelates them (raising effective rank). This directly targets the ~15-of-768 collapse the review measured.

### P2 — Stop the model learning "who"

- **Cross-subject positives** (`objective.cross_subject_positives`): skip-gram can draw positives from the *same stimulus read by different subjects* using a new subject-agnostic per-token `content_id`. A `StimulusBatchSampler` (`data/torch_dataset.py`) co-locates the same sentence across subjects in a batch so those positives actually exist; the training pipeline routes the train loader through it automatically when the flag is on.
- **Gradient-reversal subject adversary** (`objective.subject_adversary_weight`, `models/heads.py::SubjectAdversary`/`gradient_reverse`): an auxiliary head tries to read the subject from the token hiddens; the reversed gradient trains the encoder to hide subject identity.
- **Per-subject normalisation** (`dataset.normalize='zscore_subject'`, `data/transforms.py::FeatureNormalizer`): removes the constant per-subject offset; serialises per-subject statistics and falls back to global pooled stats for unseen subjects at inference.

### P3 — Fix the masked objective and the eval paths

- The exported 768-d projection head is now **trained** under the masked objective (both latent and reconstruct variants predict/reconstruct *through* `model.project`). Previously it received no gradient, so exp2 exported a random projection.
- The data2vec teacher target is normalised **across tokens with a variance floor** (`_normalize_across_tokens`, `objective.teacher_variance_floor`) instead of a per-token LayerNorm — this is what stops teacher/student co-collapsing to a constant (the exp2 cone).
- The teacher **EMA decay is ramped** from `objective.ema_decay` to `objective.ema_decay_end` across training (data2vec schedule), driven by the trainer passing the global step to `post_step`.
- **Objective-aware inference routing** (`models/embedding.py::embed_sentence`, `inference/embed.py`): sentence/word embeddings now follow each objective's *trained* path — skip-gram/CBOW skip the transformer, CPC uses a causal mask, masked uses the bidirectional context.

### P4 — Evaluate on held-out data; leakage-aware splits and normalisation

- `train.test_fraction` now defaults to **0.1** (held-out evaluation is the norm, not in-sample).
- New **`by_stimulus`** split groups by normalised sentence text across subjects, so the same sentence never spans train and test (unlike `by_sentence`).
- The normaliser (and imputer) are **fit on the train split only** via `ZuCoDataset.refit_normalizer`, called by the pipeline after the split — no val/test/held-out-subject leakage. `dataset.normalizer_fit='all'` restores the legacy whole-dataset fit.

### P5 — Input features

- Default eye-tracking scalars **drop `SFD`** (≈60% missing, equals FFD where present).
- FFD/GD-locked band power is available by setting `dataset.band_power_measures` (e.g.  `('FFD','GD','TRT')`) — the feature machinery is fully general over measures.
- The **raw-EEG Conformer path (exp5)** is verified to run end-to-end (masked reconstruction through the trained projection head).
- **EEG-only is the honest headline**: new `exp6_skipgram_eegonly_invariant` preset excludes eye-tracking and turns on every subject-invariance lever with a `by_stimulus` held-out split.

### P6 — Tighten the evaluation and clean the config

- **Bootstrap/permutation confidence intervals + effect-size floors** replace the sign-only `beats_noise` / retrieval / subject-arithmetic verdicts (`evaluation/metrics.py::bootstrap_ci`, `evaluation/report.py::_verdict`).
- **`task_transfer` NaN bug fixed**: the analogy content id no longer embeds the field being transferred across; genuinely disjoint SR/NR stimuli are reported as not-applicable rather than a bare NaN.
- **Query-weighted retrieval chance** so the "×chance" multiple is computed consistently with how hits are scored (the type-weighted value is retained under `chance_top1_typeweighted`).
- Probe cross-validation is now **shuffled and scaled** (`KFold`/`StratifiedKFold(shuffle=True)` + `StandardScaler`), so probe R² magnitudes are trustworthy (direction was already correct).
- **Electrode montage**: `dataset.montage_csv` loads a real montage for scalp-region importance; without one, region claims are flagged and softened as an "approximate region proxy".
- **Dead knobs removed**: `objective.n_negatives` and `objective.reduce_omitted_weight` (which misrepresented what ran) are deleted from the config. Old YAMLs carrying these keys still load (unknown keys are ignored).

### New — Interactive Thought-Space Explorer

- `evaluation/interactive.py::thought_space_explorer_html` and the `zte-visualize` CLI produce a single self-contained, offline Plotly HTML with live controls for: one subject / many words; many subjects / one word (with a cross-subject cosine stat); **thought arithmetic** (`emb(t,A) − centroid(A) + centroid(B) ≈ emb(t,B)`, drawn as an arrow with the nearest-neighbour hit); an **eye-tracking with/without** toggle; and real-time colour/subject/word/2D-3D/view switching.

### Experiment presets

`experiments/exp1..exp6` regenerated against the new schema: VICReg on every run; exp2 with the masked fixes; exp4 as the LOSO subject-invariance flagship (adversary + cross-subject positives + per-subject norm); exp5 raw-conformer; exp6 the EEG-only, everything-on, `by_stimulus` headline.
