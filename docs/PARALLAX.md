# Parallax — one encoder per task, and the transfer matrix that becomes measurable

ZuCo was recorded under three reading tasks — natural reading (NR), sentiment reading (SR) and task-specific reading (TSR) — and each task has its own stimulus set. The parallax study trains **three independent encoders, one per task**, each seeing only its own task's readings, and then evaluates every encoder on every task. The name is the method: three vantage points observing the same minds. What stays fixed across vantage points — measured by cross-task transfer and by representational similarity (CKA) between the models — is the task-invariant part of the code; what moves is register.

Code: `zte.parallax` (`study`, `transfer`, `report`, with `chamber` as the renderer). CLI: `zte-parallax`. Configs: [`experiments/parallax/`](../experiments/README.md). Notebook: `notebooks/zte_parallax.ipynb`. Metric definitions live in [`EVALUATION.md`](EVALUATION.md); this document owns the study design and its artifacts.

## Why per-task training — the confound is structural

On ZuCo, task *is* stimulus. No sentence appears under two tasks, so the association between the task label and sentence identity measures Cramér $V = 0.998$ — knowing which task a reading came from all but determines which stimulus set its sentence belongs to. For a contrastive objective trained across tasks this is an open door: an in-batch distractor from another task can be rejected on register alone — the sustained attentional and affective signature of *doing that task* — without encoding anything about the sentence. And the measured encoder walks through it: a linear task probe reads **0.918** on the embedding against **0.685** on the raw features. The encoder does not merely inherit the confound; it amplifies it, because register is an easy, reliable way to win the contrastive game.

`objective.within_task_negatives` treats the symptom inside the loss by making every contrastive denominator task-pure. Parallax removes it structurally: an encoder trained on a single task has a constant task register, so there is no register variance to encode, no cross-task distractor to reject on it, and no way for the objective to feed on the confound. Whatever such an encoder learns, it learned from within-task variation.

Removing the confound also creates the measurement. With one encoder trained across all tasks, "train on X, evaluate on Y" is not a cell — every task is in-distribution and the stimulus sets are all seen. With three per-task encoders, the cross-task cell exists for the first time: a model trained on task X, evaluated on task Y ≠ X, faces a **never-seen subject** (the LOSO holdout) reading **never-seen stimuli** (the tasks' sentence sets are disjoint by construction). That is the strongest generalization test this project can produce, and the study exists to measure it.

## The 3×3 transfer matrix

Each encoder is trained with the current best-measured recipe — the `exp17_base` configuration, with residual coding off and the gallery CE off, both falsified as levers on 2026-08-15 — on its own task's readings from the 11 non-holdout subjects, with `ZAB` held out. A cell of the matrix evaluates one trained encoder on one eval task's readings from the held-out subject, against that task's own sentence gallery:

- **The diagonal ($X \to X$)** is standard LOSO: a never-seen subject, reading stimuli the model has seen through other brains. `transfer.json` records `novel_stimuli: false` and the full `stimulus_overlap`.
- **The off-diagonal ($X \to Y$, $Y \neq X$)** adds never-seen stimuli: the holdout subject *and* a disjoint sentence set. The disjointness is asserted from the data (`stimulus_overlap: 0`), never assumed.

The readout is **closed-set retrieval** on the eval task's gallery, reported the way every honest number on this project is reported:

- **Stratified rank percentile with a bootstrap CI** is the headline statistic — it uses every query, not just the winners, and it is the one statistic run-to-run Top-1 noise does not swamp. The cell reuses `zte.evaluation.audit.rebaseline.stratified_retrieval`, once on the full per-task gallery and once length-matched (`held_out_length_stratified`).
- **Top-K as hit counts** out of `n_queries`, with exact binomial tails, never as bare rates.
- **The menu-capacity audit** (`zte.evaluation.audit.menu`) rides in every cell — see the menu-capacity section of [`EVALUATION.md`](EVALUATION.md) for the derivation and its guards. One simplification is specific to parallax: within a single eval task the task dimension of `length_task_matched` is constant, so the certified pools reduce to exact-length pools.
- **Unanswerable queries are excluded and counted**, never zero-scored — the exclusion convention from the scoreboard applies unchanged, because forced zeros over an at-chance remainder manufacture a below-chance number out of pure construction.
- **Post-processing is fitted on `non-holdout subjects, eval task`**, and the cell says so verbatim in `postprocess_fit`. Whitening and all-but-the-top are fitted on the eval task's readings from the 11 training subjects only. On the diagonal this is the ordinary train fit; on an off-diagonal cell the model has no training rows in the eval task at all, so fitting on other brains' readings of that task is the deployable analogue — and the held-out subject's readings never enter the fit in any cell.

**Chance differs per cell.** The gallery is per-task — roughly 300–400 distinct sentences rather than the pooled 700 — so chance is $1/N_Y$ where $N_Y$ is the eval task's gallery size, and $N_Y$ varies with $Y$. Cells that share an eval task (a *column* of the matrix) are directly comparable; cells in a *row* face different galleries and are never compared without saying so. Rank percentile, whose chance level is 0.5 regardless of gallery size, is the statistic that travels across the whole matrix.

## Triangulation — CKA between vantage points

Transfer asks whether one model's code *works* on another task. The complementary question is whether the three models converged on the same *geometry*. For a pair of models, both embed the identical set of readings, giving matrices $X, Y \in \mathbb{R}^{n \times d}$ — $n$ readings, one row per reading, embedded by each model in its own $d$-dimensional space. With $X_c$ and $Y_c$ the column-centered matrices (each column has its mean over the $n$ readings subtracted), linear CKA is

$$
\mathrm{CKA}(X, Y) \;=\; \frac{\lVert X_c^\top Y_c \rVert_F^2}{\lVert X_c^\top X_c \rVert_F \, \lVert Y_c^\top Y_c \rVert_F}
$$

where $\lVert \cdot \rVert_F$ is the Frobenius norm. The score lies in $[0, 1]$ and is invariant to orthogonal transforms and isotropic scaling of either space, so it compares the *shape* of the two representations — which readings sit near which — rather than their coordinates. `PARALLAX.json` records it per model pair (`NR|SR`, `NR|TSR`, `SR|TSR`), per seed, with the eval task the shared readings came from named in the artifact.

What convergent geometry would and would not prove. High CKA between encoders trained on disjoint tasks and disjoint stimulus sets would be evidence that the geometry is driven by structure the training sets share — the subjects' brains and whatever sentence-general code they carry — rather than by task register, which the models never shared. It would **not**, on its own, be evidence of semantic decoding: two models can agree because both encode sentence length, signal amplitude, or residual subject identity, all of which are shared low-level structure. CKA is read *beside* the transfer matrix, never in place of it — convergent geometry with null transfer means the shared structure is not the part retrieval needs.

## What to expect, honestly

Stated before the runs, so the reading of the results is pre-committed:

- **The diagonal should clear chance.** A per-task encoder on its own task under LOSO is the regime existing flagship arms already operate in; a diagonal at chance means the per-task training set (a third of the pooled data) is below the recipe's operating floor, and that is the finding.
- **The off-diagonal may be null, and a null is a publishable finding.** Its plain statement: *the code does not survive a change of stimulus distribution*. Given that the pooled encoder supplies ~4.7 bits of sentence identity on stimuli it has seen through other brains, transfer to never-seen stimuli in a never-seen brain may well carry none. Reporting that plainly is the point of building the cell.
- **Nothing in parallax is a generation claim.** The readout is closed-set retrieval and closed-set menu capacity, full stop. Free generation is not a parallax deliverable, and no parallax number is ever phrased as one.
- **Cells are not comparable across galleries.** Per-task galleries differ in size and length distribution, so chance and difficulty differ per column. Any cross-cell comparison names the galleries and leans on rank percentile.
- **TSR is the odd vantage point by design.** TSR carries an explicit relation-search instruction, so its EEG contains task-driven attention components the other two tasks lack. Asymmetries involving TSR — weaker transfer into or out of it, lower CKA against it — are expected and are read as register, not as a failure of the method. NR and SR are the closer pair of registers.
- **"Never-seen subject" carries one qualifier: checkpoint selection.** Under `by_subject_loso` the validation split *is* the holdout, and `best.pt` is the epoch with the lowest validation loss — so the model's parameters never saw the holdout, but the choice of *which epoch to keep* did (on the order of $\log_2(\text{epochs})$ bits). Every transfer cell records `checkpoint_selection` in its provenance for exactly this reason. Scoring `last.pt` through the same CLI is the selection-free sensitivity check; a claim that survives both is the one to quote.

## Measured (2026-08-16/17)

The first real run of the study — Colab sessions of 2026-08-16/17 on Drive, holdout `ZAB`, seeds 42/43/44 — measured the NR/SR half of the matrix and the TSR diagonal. Numbers as recorded, per-cell bootstrap CIs in `transfer.json` / `PARALLAX.json`.

**Cross-task transfer is real.** The strongest generalization cell this project can produce — a never-seen subject reading never-seen stimuli — clears chance decisively and does so at every seed:

| Cell | Rank percentile (seeds 42 / 43 / 44) |
| --- | --- |
| NR → SR | 0.9507 / 0.9647 / 0.9715 |
| SR → NR | 0.9515 / 0.9577 / 0.9591 |
| NR → NR (diagonal, pooled over seeds) | 0.9530 |
| SR → SR (diagonal, pooled over seeds) | 0.9575 |

Length-stratified, the off-diagonal cells hold at ~0.92–0.93 — the transfer is not the length confound. And the geometry healed with the exp17 recipe: effective-rank ratios sit at 0.41–0.46 where the v3 encoder measured 0.06–0.09.

**TSR in-task is a null, stated plainly as a finding.** At s44 the TSR diagonal scores held-out Top-1 0.00246 — exactly chance on its 407-sentence gallery — with lift −0.0003 and permutation *p* = 0.998, at a healthy effective rank of 0.33. The geometry is fine; there is no measurable content signal. This is the odd-vantage-point outcome the pre-registration anticipated, now with a number.

**The menu is at chance where the percentile is not, and the decomposition says why.** The certified prototype menu on exact-length pools reads K=2 accuracy 0.522 (CI 0.484–0.560, permutation *p* = 0.12) for NR s44 — chance — and the ±1/±2 tolerance sensitivity rows read 0.526 / 0.538, so tolerance is not the driver. The open menu reaches K=2 0.707 (permutation *p* = 0.002), but its built-in length oracle scores 0.971, so the block is stamped `gamed: true` and correctly self-disqualifies. The reconciliation: the retrieval percentile ranks the first of ~11 cross-subject *readings* of the true sentence, while the certified menu scores one *prototype centroid* inside an exact-length pool. The sentence-level discriminative signal lives in individual readings, not in centroids — which is exactly what the enrolled menu flavor (below) is built to score honestly.

**One operational note.** In the 2026-08-17 session the notebook's §3b derived the task list by globbing the raw dataset directory, and on a fresh runtime that check missed TSR — so the transfer loop ran 2×2 (all 15 TSR-involving cells absent) and the local `INDEX.md`, rebuilt without TSR rows, was mirrored over the Drive copy. All three TSR arms were in fact trained *and* evaluated on Drive; the matrix completes on re-run, and the catalogue mirror now merges by run name instead of overwriting, so a session can no longer erase another session's rows.

## Operations

Three configs, each byte-identical to `experiments/ablation/exp17_base.yaml` except for `dataset.tasks` and `run_name` — so the per-task encoders inherit the best-measured recipe exactly and differ from it, and from each other, by one field:

| Config                                   | `dataset.tasks` | `run_name`     |
| ---------------------------------------- | --------------- | -------------- |
| `experiments/parallax/parallax_nr.yaml`  | `[NR]`          | `parallax_nr`  |
| `experiments/parallax/parallax_sr.yaml`  | `[SR]`          | `parallax_sr`  |
| `experiments/parallax/parallax_tsr.yaml` | `[TSR]`         | `parallax_tsr` |

Training runs through `zte-run` as usual, and run directories on disk carry the runner's suffix: `parallax_nr_loZAB_s42`. The seed set is **42 / 43 / 44**, with 45 and 46 optional when three seeds leave a direction ambiguous. The study is driven from `notebooks/zte_parallax.ipynb` on Colab, and every long step is resumable and mirrors to Drive like every other run.

The `zte-parallax` CLI owns the study's own three steps:

```sh
# One cell of the matrix: a trained encoder scored on one eval task's held-out readings.
uv run zte-parallax transfer --ckpt res/experiments/parallax_nr_loZAB_s42/checkpoints/best.pt \
    --eval-task TSR --root res/data/zuco_extracted --out res/parallax/cells

# Aggregate every cell directory into the study artifacts.
uv run zte-parallax report --transfers res/parallax/cells --out res/parallax/report

# Render the chamber page from the report's data. It computes nothing.
uv run zte-parallax chamber --report-dir res/parallax/report --out res/parallax/chamber.html
```

`transfer` writes one directory per cell, named `<train_task>_to_<eval_task>_s<seed>/`, holding `transfer.json` and `embeddings.npz` (the held-out sentence embeddings with their content ids, subjects and word counts). `transfer.json` carries the cell whole: train and eval task, seed, holdout, `novel_stimuli` and `stimulus_overlap`, `n_queries`, the `held_out` and `held_out_length_stratified` retrieval blocks, the `menu` block, the `postprocess_fit` label, and provenance — checkpoint path and sha256, git commit, `run_name`, and the training tasks read from the checkpoint's own config, so a cell can never misstate which encoder produced it.

`report` writes three artifacts: **`PARALLAX.json`** (the full matrix — per-seed cell summaries under `cells[train][eval]`, the per-pair CKA, holdout, seeds, provenance), **`PARALLAX.md`** (the human-readable board), and **`CHAMBER_DATA.json`** — the render-ready payload: per eval task, at most 700 points (one per distinct sentence, the cross-subject prototype), each with its text, cluster, word count, per-model rank percentile, and a 3D coordinate per model view, PCA-reduced and Procrustes-aligned across the three models so the vantage points are visually comparable; plus the transfer matrix with CIs, the certified menu capacities, and the three CKA scores. `chamber` renders that payload to a single self-contained HTML page and computes nothing — every number on the page exists in the JSON first.

Two audit surfaces exist because the measured menu-vs-percentile gap demanded them:

- **The enrolled menu flavor** (`zte.evaluation.audit.menu`) scores each K-way option against the *enrolled individual readings* of its sentence — the non-holdout subjects' readings, each kept as its own reference — taking the option's best reading match rather than collapsing the readings into a prototype centroid first. It is certified like the other flavors: exact stimulus-level length matching in the distractor pool, ties losing, and the built-in length-oracle null. The oracle is live protection for the `open` pool only: inside an exact-length pool every candidate shares the word count, so the oracle ties to zero by construction and the real protection is the pool itself. A pool where the oracle escapes chance is stamped `gamed: true` and can certify no capacity. The rationale is measured, not aesthetic: the retrieval percentile ranks readings and clears chance while the prototype menu does not, so a menu that never leaves reading space is the honest test of whether that signal supports a forced choice.
- **The 2-way decomposition diagnostic** (`menu_decomposition` in `PARALLAX.json`, rendered in `PARALLAX.md`) re-scores every diagonal cell under the four combinations of {prototype, best reading} × {exact length, ±1 word}, so the gap between the menu and the percentile decomposes into named factors instead of being argued about. It is a diagnostic and gates nothing; no cell of it is a headline.

**No claim from this study enters [`RESULTS.md`](RESULTS.md) without directional consistency across seeds.** A cell whose seeds disagree on sign is reported as unresolved, not averaged into a headline — three seeds are the minimum to say anything, and the optional 45/46 exist to settle exactly this.
