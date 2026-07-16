# The road to state-of-the-art — methods & rationale

How ZTE went from *below-chance* cross-subject retrieval to a de-hubbed, invariance-tempered, meaning-targeted encoder — one config-gated lever at a time. Every move below is real and in the codebase now; each states its hypothesis, its neuroscience/engineering grounding with a citation, the exact config key that turns it on and where the code lives, and the metric that confirms it.

Companion reading: [METHODS.md](./METHODS.md) (the generative thesis and the modelling levers this layer is built on), [EVALUATION.md](./EVALUATION.md) (the four families of evidence and the verdict machinery), and [SPATIAL_ENCODING.md](./SPATIAL_ENCODING.md) (the electrode-geometry encoders).

---

## 0. The blocking result, and the diagnosis

ZTE turns the single-word EEG recorded while a person reads (ZuCo, 12 subjects, English) into a 768-dimensional "thought embedding". An earlier round of evaluation surfaced a blocking result:

> On the held-out subject **ZAB** (leave-one-subject-out, LOSO), cross-subject retrieval was **below chance** — the *away game* was lost — on a space that was otherwise **healthy**: high effective rank, low anisotropy, no dimensional collapse.

The diagnosis was *not* "no signal", but two concrete, mechanical faults:

1. **Hostile target geometry.** The exported space carried a dominant shared direction (a residual frequency / "hub" axis) and hubness — a few points that are everyone's nearest neighbour. That is the textbook cause of below-chance retrieval on an *otherwise isotropic* space: the health metrics look fine in aggregate while the nearest-neighbour graph is quietly broken.
2. **An over-aggressive subject adversary eroding content.** The gradient-reversal subject classifier ran at full strength (`subject_adversary_weight = 1.0`) from step 0. Because content is confounded with identity and task (the confound audit found task ≈ stimulus, Cramér's *V* ≈ 0.99), a hard early adversary deletes content *along with* identity — exactly the failure the disentanglement analysis predicted, where driving identity to 0% let task variance rise to 85%.

The methods below are the fix. They are organised into three groups: **(1) geometry & invariance**, **(2) sharpen the contrastive objective**, **(3) architecture & evaluation hardening**. Nothing here abandons the generative disentanglement thesis (see [METHODS.md](./METHODS.md)); this is the *retrieval-and-honesty* layer that sits on top of it.

---

## 1. How to read this document, and the win condition

**The honest framing (read this before any number).** EEG single-word retrieval is the *hardest* non-invasive setting there is. Défossez et al. (2023) — the strongest non-invasive speech-decoding result to date — land on the order of a few percent top-1 for **EEG** against tens of percent for **MEG** (EEG top-1 ≈ 5% vs MEG ≈ 38%); MEG's cleaner forward model and higher SNR do most of that work, and reading a *single word* from scalp EEG is harder still than decoding continuous perceived speech. A headline top-1 is therefore the wrong yardstick and invites the "BLEU-trap" the project explicitly guards against.

**The pre-registered win condition** is two things, both on the *held-out* subject and both stated as a *lift over the raw-band-power control*:

1. a **retrieval rank distribution that sits left of the permutation null** — the whole distribution of correct-match ranks shifts toward 1, even when top-1 is small; and
2. a **positive content-lift-over-raw** — the learned embedding reads lexical content *better than the un-learned input features* on a subject it never trained on.

Rank-percentile and median-rank (which degrade gracefully with gallery size, unlike top-1) are the reporting currency throughout. A method that improves a *home-game* number but not the *away game* has not earned its place.

---

## 2. Tier 1 — geometry & invariance

The first tier attacks the two mechanical causes of below-chance retrieval directly: the hostile geometry, and the over-aggressive adversary. These are the highest-leverage, lowest-risk moves because the first two are **label-free, monotone re-rankings** applied consistently at evaluation — they add no signal and cannot inflate a number, they only stop the geometry from hiding a real effect.

### 2.1 Retrieval-geometry fix — all-but-the-top + CSLS

**Hypothesis.** The below-chance retrieval is a *geometry* artefact, not an absence of content: stripping the shared/hub axes and de-hubbing the neighbour search will lift cross-subject retrieval without touching the encoder.

**Grounding.** Two established, label-free corrections, applied together:

- **All-but-the-top (ABTT)** — Mu & Viswanath (2018) showed that after centring, word embeddings share a few dominant PCA directions along which nearly every vector has a large common component; projecting those out removes the anisotropy/hub axis and measurably improves downstream similarity tasks. It is the correct fix for "a healthy-looking space with a broken nearest-neighbour graph".
- **CSLS (Cross-domain Similarity Local Scaling)** — Conneau et al. (2018), from the MUSE bilingual-lexicon work, corrects each cosine to `2·cos(x,y) − r_x − r_y`, where `r` is a point's mean cosine to its *k* nearest neighbours. It penalises hub-dense regions so a handful of points can no longer win every top-1. It is a monotone re-ranking: it adds no information, it only removes the hub advantage.

**Config & code.**

- `objective.all_but_top` (int; `1` in both flagship configs) — the number of leading directions to strip. Applied in the evaluation post-processing block of `src/zte/evaluation/report.py` (the `whiten` → ABTT order is deliberate: whiten equalises variance across dims, then ABTT strips the residual shared axes), calling `all_but_the_top` in `src/zte/evaluation/metrics.py`.
- `objective.csls_neighbors` (int *k*; `10` in both configs) — the CSLS neighbourhood, implemented in `NearestNeighborIndex` in `src/zte/inference/retrieval.py` (`query`), applied to the retrieval index **and its permutation null** so top-1 and its *p*-value stay coherent.

Both are label-free and applied to every representation identically (including the phase-shuffled control), so every metric downstream is honestly recomputed on the corrected space — the report shows whether *content survives the geometry fix*, not merely that anisotropy dropped.

**Confirms if.** Cross-subject held-out retrieval **Top-1 / rank-percentile lift over raw** turns positive on ZAB; the `geometry_before_after.png` figure shows anisotropy collapsing and neighbour structure emerging; the scoreboard's held-out anisotropy drops toward 0 while the effective-rank ratio stays high.

### 2.2 Rebalance and ramp the subject adversary

**Hypothesis.** The adversary was doing more harm than good because it was too strong, too early. A smaller steady weight plus a warm-up ramp lets the encoder learn content *before* invariance pressure is applied, so identity is removed without erasing the content it is confounded with.

**Grounding.**

- **Weight cut 1.0 → 0.1.** The EEG-invariance literature runs adversarial-invariance terms far weaker than this project's original 1.0 — Özdenizci et al. (2020), *Learning Invariant Representations From EEG via Adversarial Inference*, operate in the ~0.03–0.05 regime. The theoretical reason is Zhao et al. (2019), *On Learning Invariant Representations for Domain Adaptation*: perfect marginal-feature invariance provably **lower-bounds** the joint error when the label distribution shifts across domains — invariance traded too hard *must* cost content.
- **Ramp the gradient-reversal λ 0 → 1.** This is the standard DANN schedule of Ganin et al. (2016), *Domain-Adversarial Training of Neural Networks*: the reversal strength is annealed up so a cold adversary early does not invert gradients on a not-yet-formed representation.

**Config & code.**

- `objective.subject_adversary_weight` (`0.1` in both configs) — the flat loss weight.
- `objective.subject_adversary_warmup_ratio` (`0.3` in both configs) — the fraction of total optimiser steps over which λ ramps linearly 0 → 1, then holds at 1.

The ramp is threaded from the trainer via `_ObjectiveBase.set_progress(step, total_steps)` in `src/zte/models/objectives.py`, read back by `_adv_lambda()`, and applied inside `_ObjectiveBase.regularize` (the `subject_adversary(subj_adv_in, lambda_=adv_lambda)` call). When progress is unset (evaluation, legacy callers) or the ratio is 0, λ = 1 exactly, preserving the old behaviour. Note that when the model is `factored`, the adversary acts on the **content subspace** (pushing identity *out of content*) rather than the shared hidden — the disentanglement crux of the factored model (see [METHODS.md](./METHODS.md)).

**Confirms if.** Subject-probe accuracy falls toward chance **without** the content variance budget collapsing or task variance rising to absorb it (the exp6 failure mode); the logged `adv_lambda`, `adv_acc`, and `adv_loss` trace the ramp; held-out content-lift-over-raw stays positive. Provable in isolation with `zte-ablate` on `objective.subject_adversary_weight` and `objective.subject_adversary_warmup_ratio`.

### 2.3 Exact electrode montage

**Hypothesis.** The spatial encoders can only exploit scalp geometry if they run on *real* electrode coordinates; the approximate fallback silently degrades the single most effective identity-reducing lever found so far.

**Grounding.** Real coordinates matter because the approximate fallback silently degrades the encoder's strongest identity-reducing lever. ZuCo v1/v2 use the 129-channel EGI GSN-HydroCel net; the retained 105 scalp electrodes have well-defined 3-D positions. The spherical-harmonic mixer (the Laplace–Beltrami eigenfunctions of the sphere — the correct generalisation of sinusoidal position encoding from a line to the scalp surface) and the spatial-attention mixer both consume these coordinates. See [SPATIAL_ENCODING.md](./SPATIAL_ENCODING.md).

**Config & code.**

- `dataset.montage_csv: res/montage_gsn105.csv` (both configs) — the real `channel,x,y,z` table, produced by `scripts/export_montage.py`. Consumed by `ScalpGeometry` in `src/zte/models/spatial.py`; when it is `null` the code falls back to a Fibonacci coordinate approximation and flags every region claim `approximate=True`.

**Confirms if.** `ScalpGeometry.approximate` is `False` (exact geometry in force); the `scalp_topomap.png` figure and `region_importance.csv` become interpretable on true positions; and, as a lever, subject-identity decodability drops / content-lift rises under exact vs. fallback coordinates (`zte-ablate` on `dataset.montage_csv`).

---

## 3. Tier 2 — sharpen the contrastive objective

Tier 1 makes retrieval *readable*. Tier 2 makes the contrastive objective actually *put content in* — closing the theoretical gaps in the skip-gram loss and giving the meaning target a fair chance.

### 3.1 The alignment term — the missing half of align + uniformity

**Hypothesis.** The objective already spread embeddings over the sphere (uniformity) but never explicitly *pulled positives together* (alignment); adding the missing half tightens the same-word geometry that retrieval depends on.

**Grounding.** Wang & Isola (2020), *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, decompose a good contrastive space into two properties: **alignment** (positive pairs close) and **uniformity** (features spread out). ZTE's `anisotropy_weight` already supplied the uniformity half; this adds the alignment half — for unit vectors, `‖c_i − x_j‖² = 2 − 2·c_iᵀx_j`, so minimising it directly maximises positive-pair cosine.

**Config & code.**

- `objective.alignment_weight` (`0.1` in both configs). Implemented as `alignment_penalty` in `src/zte/models/objectives.py`, called from `SkipGramObjective.compute` over the final positive-pair mask using the already-L2-normalised center/context embeddings; the scalar `alignment_loss` is logged.

**Confirms if.** The label-free **alignment** health metric (mean positive-pair distance) falls; the logged `alignment_loss` decreases; cross-subject retrieval lift improves. Ablatable via `zte-ablate` on `objective.alignment_weight`.

### 3.2 Debiased contrastive — stop punishing correct answers

**Hypothesis.** In a word-level batch, another EEG trial of the *same word* sits among the InfoNCE "negatives"; plain InfoNCE shoves it away, actively fighting the meaning structure. Correcting for those false negatives lets semantically identical items stay together.

**Grounding.** Chuang et al. (2020), *Debiased Contrastive Learning* (NeurIPS): the negative expectation is corrected with a class-prior `τ⁺`, `E_neg = (mean_neg − τ⁺·mean_pos)/(1 − τ⁺)`, floored so the estimator stays positive. It subtracts an estimate of the positive mass leaking into the negative set — exactly the same-word false-negative problem, which is acute in low-SNR EEG batches.

**Config & code.**

- `objective.tau_plus` (`0.1` in both configs; small values `~0.05–0.1` are appropriate for low-SNR EEG). Implemented as `debiased_infonce` in `src/zte/models/objectives.py`, selected inside `SkipGramObjective.compute` (replaces the plain `logsumexp` denominator when `tau_plus > 0`), with a per-anchor max-shift for numerical stability.

**Confirms if.** Word-level (same-word) retrieval Top-*k* and rank-percentile improve, especially on the held-out subject; content-lift-over-raw rises. Ablatable via `zte-ablate` on `objective.tau_plus`.

### 3.3 Collapse-proof regression auxiliary — fill the idle nuisance dims

**Hypothesis.** The factored embedding reserves `embed_dim − content_dim` nuisance dimensions, but if nothing supervises them they receive gradient from nothing and can drift or collapse; a *frozen* regression target both fights collapse and gives those dims a real job.

**Grounding.** In the data2vec / HuBERT spirit (Baevski et al., 2022, *data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language*), but with the anti-collapse property made explicit: the nuisance subspace regresses toward a **fixed random projection of the token's own input features**. Because the target never moves, there is no teacher/student co-collapse (the exact failure mode a plain EMA teacher can hit), and the otherwise-idle nuisance room is genuinely used to reconstruct the input.

**Config & code.**

- `objective.data2vec_aux_weight` (`0.5` in both configs; requires `model.factored`). The frozen projection and the regression head are built in `_ObjectiveBase.attach_auxiliary` (`data2vec_proj.requires_grad_(False)`) and the cosine regression is applied in `_ObjectiveBase.regularize` over the `_nuisance_slice`, in `src/zte/models/objectives.py`; the scalar `data2vec_loss` is logged.

**Confirms if.** Effective-rank ratio stays high and the dead-dim fraction stays low with the nuisance dims *occupied* (visible in the `variance_budget_pie.png`); `data2vec_loss` decreases. Ablatable via `zte-ablate` on `objective.data2vec_aux_weight`.

### 3.4 Per-occurrence contextual meaning target

**Hypothesis.** A static, word-type-keyed meaning vector (GloVe/fastText) collapses polysemy — every occurrence of "bank" gets the same target. A *contextual*, per-occurrence target disambiguates sense and aligns better with brain activity.

**Grounding.** The brain-alignment literature is consistent that **contextual** representations, read from a **middle** layer, predict neural responses to language far better than static word vectors: Toneva & Wehbe (2019); Caucheteux & King (2022), *Brains and algorithms partially converge in natural language processing*; and Tang et al. (2023), *Semantic reconstruction of continuous language from non-invasive brain recordings*, which targets a middle layer. Because a word occurrence's linguistic content is subject-independent, the target is exactly the subject-invariant signal LOSO wants.

**Config & code.**

- `objective.meaning_contextual` (HF model id, e.g. `bert-base-uncased`; `null` in both flagship configs, which keep the static path) and `objective.meaning_context_layer` (`-1` = last hidden; `~7–9` aligns best with brain data). The target is built by `build_meaning_matrix_hf` in `src/zte/data/meaning.py` — the encoder runs once per unique sentence (`stimulus_key`) and the per-position vectors broadcast to every subject's reading of that word (a ~12× saving and the subject-invariant target LOSO wants). It is plumbed through the `SentenceSample`/collate/pipeline as a per-row `meaning_target`, preferred over the static matrix inside `_ObjectiveBase.regularize` when present.
- **Opt-in dependency.** The contextual path needs the `meaning` dependency group (`transformers`); when unavailable it **falls back gracefully** to the static word-type path with a warning. The static fastText ↔ GloVe swap remains config-only via `objective.meaning_source` (`res/vectors/glove.300d.txt`, `meaning_dim: 300` in both flagship configs).

The distillation itself (`meaning_distill_weight`, `1.0` in both configs) is a cosine pull of the **content subspace** toward the frozen vector, and is documented as a base modelling lever in [METHODS.md](./METHODS.md).

**Confirms if.** Content probe / cross-subject retrieval lift rises — and, with the seen-vs-novel word-type split (Section 4.2), the gain shows up on **novel** word types, not just memorised ones.  Ablatable via `zte-ablate` on `objective.meaning_contextual` (static vs contextual).

---

## 4. Tier 3 — architecture & evaluation hardening

### 4.1 FiLM subject conditioning + learned spatial attention

**Hypothesis (FiLM).** Deleting identity with an adversary is one-sided; you can also *condition on* identity — as long as the conditioning degrades to the identity transform for a subject the model has never seen, so it never injects noise into the held-out north-star.

**Grounding.** FiLM (feature-wise linear modulation; Perez et al., 2018) applies a per-condition affine `(1 + γ)·h + β`. Défossez et al. (2023) argue you should *condition on* subject identity, not only adversarially remove it — but naïvely, an unseen subject has no learned `(γ, β)`. The fix here makes it **honest for LOSO**: the per-subject FiLM table is **zero-initialised**, so `γ = 0, β = 0` is the identity, and any subject id never updated in training (the held-out subject) is a *no-op*, not an untrained random vector.

**Hypothesis (spatial attention).** A learned attention over 2-D scalp coordinates is an alternative geometry encoder to spherical harmonics; both are kept available for a clean A/B.

**Grounding.** Défossez et al. (2023) found a per-subject learned **spatial-attention** layer over electrode coordinates to be the single most important component of their non-invasive decoder. Each output electrode is a fixed, geometry-derived weighted combination of inputs, `out_o = in_o + Σ_c softmax_c(z_o(pos_c))·in_c`, where `z_o` reads a 2-D Fourier embedding of each electrode's scalp coordinate. The spherical-harmonic mixer remains the more *principled* geometry encoder (and the `sota_loso` default); spatial attention is the learned alternative.

**Config & code.**

- `model.subject_film` (`false` in `sota_loso`, `true` in `exp7`) — the zero-init per-subject affine in `ZTEModel.token_hidden` (`src/zte/models/embedding.py`; `nn.init.zeros_(self.subject_film.weight)`).
- `model.spatial_encoding: spatial_attention` (in `exp7`; `spherical_harmonics` in `sota_loso`) with `model.spatial_attn_freqs` (`8`) — `SpatialAttention` in `src/zte/models/spatial.py`.

**Confirms if.** Held-out retrieval is **not harmed** by FiLM relative to additive conditioning (the identity-degradation property holds); the `subject_similarity.png` heatmap shows the held-out subject sitting on the shared frame; and the `exp7` (spatial-attention + FiLM + shrunk `content_dim`) vs `sota_loso` (spherical-harmonics) A/B decides which geometry encoder wins on held-out content-lift.

### 4.2 Evaluation hardening

**Hypothesis.** A single below-chance-or-not gate is too weak and too easily fooled; the verdict must be an AND of independent controls, and retrieval must be reported as a *distribution* against a *permutation null*.

**Grounding & mechanisms.** Each is an established honesty control:

- **Permutation-*p* gate, now AND-ed (not ignored).** The retrieval verdict now requires the bootstrap CI *and* a permutation-null *p* < 0.05 to both pass — a single check can no longer carry it.  Implemented in `src/zte/evaluation/report.py` (the `retrieval_above_chance` is AND-ed with `perm['above_chance']`), backed by `retrieval_permutation_test` in `src/zte/evaluation/honesty.py` (`p = (1 + #{null ≥ observed}) / (n_perm + 1)`). When the permutation is inapplicable (too few items), the CI verdict stands alone.
- **Phase-scrambled-input control.** `objective.eval_phase_shuffle` runs the *same trained encoder* on FFT-phase-randomised EEG (power spectrum preserved, temporal/phase structure destroyed), via `phase_scramble` in `src/zte/data/transforms.py`. It proves the encoder invents no structure from spectrum alone. Honestly reported as a **no-op-by-construction for band power** (near phase-invariant) and **informative for raw** frontends. `false` in both band-power flagship configs.
- **Seen vs novel word-type split.** `objective.eval_seen_novel` (`true` in both configs) splits cross-subject retrieval into *seen* vs *novel* word types (novel = absent from the training split), so "zero-shot" means unseen *word types*, not only unseen subjects. Applied in `report.py`.
- **Frequency-matched distractor bank.** `objective.eval_freq_matched` (`true` in both configs) restricts each query's distractor bank to its own frequency/length bin via `matched_content_retrieval` in `src/zte/evaluation/metrics.py`, so a hit reflects *content*, not a rare word standing out among common ones (chance is recomputed *within* each bin).
- **Rank-percentile / median-rank everywhere.** `content_retrieval` (with `return_ranks=True`) reports `median_rank`, `mean_rank`, and `rank_percentile` alongside Top-*k* — the pre-registered success metric is a *rank distribution shifting left of the permutation null*, which top-1 alone hides.

**Confirms if.** The verdict passes only when CI **and** permutation agree; the phase-shuffle control reads content at R² ≈ 0 (band power); the seen/novel split localises where content lift comes from; and `retrieval_rank_distribution.png` shows the correct-match ranks piled left of the null.

---

## 5. The config surface

Every lever is an independent, config-gated knob. The table below is the complete surface; values shown are the `sota_loso` / `exp7` flagship settings.

| Config key                                 | Value (sota_loso · exp7)                    | Turns on                                         | Code — file · symbol                                             | Confirming metric                                                |
| ------------------------------------------ | ------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| `objective.all_but_top`                    | `1` · `1`                                   | Strip top-*n* PCA (hub) directions               | `evaluation/report.py` post-proc · `metrics.all_but_the_top`     | Held-out retrieval rank-percentile lift; `geometry_before_after` |
| `objective.csls_neighbors`                 | `10` · `10`                                 | CSLS hubness-corrected retrieval                 | `inference/retrieval.py` · `NearestNeighborIndex.query`          | Held-out Top-1 lift over raw                                     |
| `objective.subject_adversary_weight`       | `0.1` · `0.1`                               | Gradient-reversal subject adversary (rebalanced) | `models/objectives.py` · `_ObjectiveBase.regularize`             | `adv_acc` → chance w/o content drop                              |
| `objective.subject_adversary_warmup_ratio` | `0.3` · `0.3`                               | DANN λ ramp 0→1                                  | `models/objectives.py` · `set_progress` / `_adv_lambda`          | `adv_lambda` trace; content lift held                            |
| `dataset.montage_csv`                      | `res/montage_gsn105.csv`                    | Exact electrode coordinates                      | `models/spatial.py` · `ScalpGeometry`                            | `approximate == False`; region/topomap                           |
| `objective.alignment_weight`               | `0.1` · `0.1`                               | Align+uniformity: alignment half                 | `models/objectives.py` · `alignment_penalty`                     | `alignment` health metric ↓                                      |
| `objective.tau_plus`                       | `0.1` · `0.1`                               | Debiased InfoNCE                                 | `models/objectives.py` · `debiased_infonce`                      | Same-word retrieval / content lift                               |
| `objective.data2vec_aux_weight`            | `0.5` · `0.5`                               | Frozen-target nuisance regression                | `models/objectives.py` · `attach_auxiliary` / `regularize`       | Eff-rank ratio ↑; nuisance occupied                              |
| `objective.meaning_contextual`             | `null` · `null`                             | Per-occurrence contextual target                 | `data/meaning.py` · `build_meaning_matrix_hf`                    | Content lift on novel words                                      |
| `objective.meaning_context_layer`          | `-1` · `-1`                                 | Which contextual layer                           | `data/meaning.py` · `build_meaning_matrix_hf`                    | (as above)                                                       |
| `objective.meaning_source`                 | `glove.300d.txt`                            | Static fastText/GloVe swap                       | `data/meaning.py` · `build_meaning_matrix`                       | Content probe / retrieval lift                                   |
| `model.subject_film`                       | `false` · `true`                            | Zero-init per-subject FiLM affine                | `models/embedding.py` · `ZTEModel.token_hidden`                  | Held-out retrieval not harmed                                    |
| `model.spatial_encoding`                   | `spherical_harmonics` · `spatial_attention` | Geometry encoder A/B                             | `models/spatial.py` · `SpatialAttention` / `SpatialChannelMixer` | Held-out content lift A/B                                        |
| `model.spatial_attn_freqs`                 | `8` · `8`                                   | Fourier freqs for spatial attention              | `models/spatial.py` · `SpatialAttention`                         | (as above)                                                       |
| `objective.eval_phase_shuffle`             | `false` · `false`                           | Phase-scrambled-input control                    | `data/transforms.py` · `phase_scramble`                          | R² ≈ 0 (band power)                                              |
| `objective.eval_seen_novel`                | `true` · `true`                             | Seen vs novel word-type split                    | `evaluation/report.py`                                           | Lift localised to novel types                                    |
| `objective.eval_freq_matched`              | `true` · `true`                             | Frequency-matched distractors                    | `metrics.matched_content_retrieval`                              | Within-bin retrieval lift                                        |

Base modelling levers this layer builds on but does not introduce — `model.factored` (`true`), `model.content_dim` (`384` · `320`), `objective.whiten` (`true`), `objective.meaning_distill_weight` (`1.0`), `objective.hard_negatives` (`true`), `objective.behaviour_weight` (`0.5`), `dataset.normalize: riemannian` — are documented in [METHODS.md](./METHODS.md).

---

## 6. The two flagship configs

Both configs hold the LOSO harness fixed — `train.split: by_subject_loso`, `train.loso_holdout_subject: ZAB`, 40 epochs, batch 128, deterministic seed 42 — and the whole geometry/objective stack. They differ only in the architecture A/B:

- **`experiments/sota_loso.yaml`** — the **geometry-fixed spherical-harmonic SOTA**.  `spatial_encoding: spherical_harmonics`, `subject_film: false`, `content_dim: 384`. The principled default: the harmonic mixer is the mathematically correct geometry encoder, and FiLM is left off so identity handling is purely disentanglement + the tempered adversary.

- **`experiments/exp7_sota_geom_invariance.yaml`** — the **learned-geometry + conditioning arm**.  `spatial_encoding: spatial_attention`, `subject_film: true`, and a **shrunk `content_dim: 320`** (a smaller content subspace, more nuisance room). This is the Défossez-style A/B: learned spatial attention over coordinates plus honest per-subject FiLM, against the harmonic + adversary-only baseline.

Everything else — ABTT, CSLS, the tempered/ramped adversary, alignment, debiased InfoNCE, the nuisance data2vec auxiliary, meaning distillation, behaviour supervision, Riemannian alignment, and the hardened evaluation — is identical, so the two runs isolate *geometry encoder + conditioning* as the single comparison.

---

## 7. Proving each lever — `zte-ablate`

No lever ships on assertion. `zte-ablate` (`src/zte/cli/ablate.py`) emits a matched config pair that toggles **exactly one knob**, then diffs the two finished runs on the held-out LOSO scoreboard — a single-variable discipline applied uniformly to every lever above.

```sh
# 1) Emit a one-knob sweep from a flagship config:
uv run zte-ablate generate --config experiments/sota_loso.yaml \
    --knob objective.csls_neighbors --values 0,10 --out-dir experiments/ablate_csls

# 2) Run each emitted config with zte-run (train → evaluate), then isolate the knob's contribution:
uv run zte-ablate diff --knob objective.csls_neighbors \
    --baseline res/experiments/<base>/evaluation/metrics.json \
    --variant  res/experiments/<var>/evaluation/metrics.json
```

`generate` uses `single_variable_configs`; `diff` uses `diff_scoreboards` / `render_diff` (in `src/zte/evaluation/ablation.py`) so a metric delta is attributable to one knob. Suggested single-knob sweeps, one per lever above: `objective.all_but_top` (0,1), `objective.csls_neighbors` (0,10), `objective.subject_adversary_weight` (0,0.1,1.0), `objective.subject_adversary_warmup_ratio` (0,0.3), `dataset.montage_csv` (null, res/montage_gsn105.csv), `objective.alignment_weight` (0,0.1), `objective.tau_plus` (0,0.1), `objective.data2vec_aux_weight` (0,0.5), `model.subject_film` (false,true), `model.spatial_encoding` (spherical_harmonics, spatial_attention), `objective.eval_freq_matched` (false,true). A lever is kept only if it moves the north-star.

---

## 8. Evaluation figures and the interactive scoreboard

A set of figures aimed squarely at the geometry-and-honesty story is rendered by `src/zte/evaluation/report.py` into `res/evaluation/figures/` (each is skipped gracefully if its inputs are unavailable):

| Figure                            | What it shows                                                                                       |
| --------------------------------- | --------------------------------------------------------------------------------------------------- |
| `geometry_before_after.png`       | The embedding geometry before vs after whiten + ABTT — the anti-cone / anti-hub fix made visible    |
| `retrieval_rank_distribution.png` | The correct-match rank distribution against the permutation null — the pre-registered win condition |
| `variance_budget_pie.png`         | The variance budget across content / nuisance / identity — shows the nuisance dims are occupied     |
| `subject_similarity.png`          | Subject-similarity heatmap — whether the held-out subject sits on the shared frame                  |
| `neuron_selectivity.png`          | Per-dimension selectivity for the most content-selective units                                      |
| `scalp_topomap.png`               | Per-channel lexical-frequency importance on the real GSN-HydroCel montage                           |

The **honest scoreboard** (`src/zte/evaluation/scoreboard.py`, `render_markdown`) is rendered at the **top of every `report.md`**: the content-probe positive control (can the probe read content from raw band power *at all* — otherwise "content 0%" is meaningless), then the held-out-only geometry (effective-rank ratio, anisotropy, content variance budget), the held-out cross-subject retrieval Top-1 vs chance stated as a **lift**, and every probe as **ZTE − raw**. The interactive views come from `zte-visualize` (`src/zte/cli/visualize.py`) — the self-contained offline Thought-Space Explorer (one subject/many words, one word across many brains with a cross-subject cosine statistic) and the neuron atlas — and the `zte-dashboard` skill assembles these into a single interactive scoreboard page.

---

## 9. How to run

### Notebook (Colab)

`notebooks/zte_colab.ipynb` runs the full pipeline end to end — prepare → train → evaluate, catalogued to Drive, with the honest scoreboard at the top of `report.md`. Section 5 ("Train an experiment — the new SOTA model") trains `experiments/sota_loso.yaml` directly (it already includes electrode spatial encoding and every lever documented here); set `CONFIG` to any `experiments/*.yaml` and `HOLDOUT` to the subject to leave out. Section 6 rotates the held-out subject over the whole cohort.

### CLI

```sh
# Restrict GloVe-300 to the ZuCo vocabulary (the small file sota_loso.yaml expects):
uv run python scripts/build_meaning_vectors.py --out res/vectors/glove.300d.txt \
    --vocab-from experiments/sota_loso.yaml --root res/data/zuco_extracted

# Export the exact GSN-HydroCel-105 montage the spatial encoders need:
uv run python scripts/export_montage.py --zuco105 --out res/montage_gsn105.csv

# Train + evaluate the geometry-fixed spherical-harmonic SOTA (writes report.md with the scoreboard):
uv run zte-run --config experiments/sota_loso.yaml

# The learned-geometry + FiLM arm:
uv run zte-run --config experiments/exp7_sota_geom_invariance.yaml

# Re-evaluate a checkpoint on its own (scoreboard, figures, tables, interactive explorer):
uv run zte-evaluate --ckpt res/checkpoints/best.pt --bundle res/bundle --out res/evaluation

# Build the interactive Thought-Space Explorer:
uv run zte-visualize --run res/experiments/sota_loso --out res/explorer/thought_space_explorer.html
```

Optional dependency for the contextual meaning target: `uv sync --group meaning` (installs
`transformers`); without it the meaning target falls back to the static GloVe/fastText path.

---

## 10. References

- Baevski, A., Hsu, W.-N., Xu, Q., Babu, A., Gu, J., & Auli, M. (2022). *data2vec: A General Framework
  for Self-supervised Learning in Speech, Vision and Language.* ICML.
- Caucheteux, C., & King, J.-R. (2022). *Brains and algorithms partially converge in natural language
  processing.* Communications Biology, 5, 134.
- Chuang, C.-Y., Robinson, J., Lin, Y.-C., Torralba, A., & Jegelka, S. (2020). *Debiased Contrastive
  Learning.* NeurIPS.
- Conneau, A., Lample, G., Ranzato, M., Denoyer, L., & Jégou, H. (2018). *Word Translation Without
  Parallel Data* (MUSE; the CSLS criterion). ICLR.
- Défossez, A., Caucheteux, C., Rapin, J., Kabeli, O., & King, J.-R. (2023). *Decoding speech
  perception from non-invasive brain recordings.* Nature Machine Intelligence, 5, 1097–1107.
- Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., &
  Lempitsky, V. (2016). *Domain-Adversarial Training of Neural Networks.* JMLR, 17(59), 1–35.
- Kutas, M., & Hillyard, S. A. (1980). *Reading senseless sentences: brain potentials reflect semantic
  incongruity.* Science, 207(4427), 203–205.
- Mu, J., & Viswanath, P. (2018). *All-but-the-Top: Simple and Effective Postprocessing for Word
  Representations.* ICLR.
- Özdenizci, O., Wang, Y., Koike-Akino, T., & Erdoğmuş, D. (2020). *Learning Invariant Representations
  From EEG via Adversarial Inference.* IEEE Access, 8, 27074–27085.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). *FiLM: Visual Reasoning
  with a General Conditioning Layer.* AAAI.
- Tang, J., LeBel, A., Jain, S., & Huth, A. G. (2023). *Semantic reconstruction of continuous language
  from non-invasive brain recordings.* Nature Neuroscience, 26, 858–866.
- Toneva, M., & Wehbe, L. (2019). *Interpreting and improving natural-language processing (in machines)
  with natural language-processing (in the brain).* NeurIPS.
- Wang, T., & Isola, P. (2020). *Understanding Contrastive Representation Learning through Alignment and
  Uniformity on the Hypersphere.* ICML.
- Zhao, H., Combes, R. T. des, Zhang, K., & Gordon, G. J. (2019). *On Learning Invariant
  Representations for Domain Adaptation.* ICML.
