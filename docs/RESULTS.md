# Validated results

## Real ZuCo — the held-out board (2026-08-13, held out on `ZAB`, 700 queries)

Everything in this section is measured on real ZuCo. Everything below it is a synthetic smoke-run and is labelled as
such; the two are different kinds of evidence and must never be quoted together.

**Read rank percentile, not Top-1.** Two seeds of the same `zte_raw_aligned` config give rank percentile 0.9672 and
0.9670 — a gap of 0.0002 — while their Top-1 moves from 9 hits to 8. At chance 1/700 a Top-1 comparison between
these arms is a comparison between one or two lucky queries. Rank percentile uses all 700.

| Config (`run_name`) | Rank percentile (95% CI) | Top-1 / 700 | Top-5 / 700 | Length-stratified rank pct | Eff-rank |
| --- | --- | --- | --- | --- | --- |
| `flagship/zte_raw_aligned` (seed 41) | 0.9672 (0.9635–0.9708) | 9 | 39 | 0.9271 | 0.25 |
| `flagship/zte_raw_aligned` (seed 42) | 0.9670 (0.9632–0.9706) | 8 | 38 | 0.9269 | 0.25 |
| `ablation/exp12_align_off` | 0.9670 (0.9632–0.9706) | 8 | 39 | 0.9269 | 0.25 |
| `flagship/clip_e5_meaning_raw` | 0.9667 (0.9629–0.9705) | 7 | 37 | 0.9268 | 0.25 |
| `flagship/clip_e5_raw` | 0.9635 (0.9599–0.9673) | 9 | 38 | 0.9208 | 0.24 |
| `archive/zte_raw_aligned_wide` | 0.9523 (0.9479–0.9565) | 5 | 14 | 0.9212 | 0.45 |

Three findings, stated as they were measured.

**The three retained flagships are tied.** Their intervals all overlap. There is no champion, and any claim that one
of these recipes beats another on this fold is not supported. `clip_e5_raw` is the only arm whose *length-stratified*
Top-1 clears *p* < 0.05 (0.0443 against a 0.0285 stratified chance rate, *p* 0.012); the other two do not (*p* 0.150
and 0.206), which matters because length alone carries 5.14 bits of sentence identity here.

**The exp12 alignment stack is a measured no-op.** `exp12_align_off` switches Euclidean alignment off and returns
rank percentile 0.9670 against the full stack's 0.9672, effective rank 190.31 against 190.25, and a subject probe of
0.4179 against 0.4180. Agreement to four decimal places on every metric is not a small effect; it is no effect. The
stack was built to close the cross-subject identity gap, and on this fold it does not.

**Capacity retained is not capacity used.** `zte_raw_aligned_wide` and `clip_e5_meaning_raw_v2` have the two
healthiest geometries ever measured here (effective-rank ratios 0.45 and 0.53, against 0.24–0.25 for the retained
set) and the two worst retrieval scores. `wide` is the first arm this board separates with non-overlapping
intervals, and its length-stratified Top-1 of 0.0200 is *below* stratified chance. Both were retired to
`experiments/archive/` on 2026-08-14. The long-standing warning that a *low* effective rank can mean invariance
bought by destroying capacity now has a converse with a number behind it.

Nothing here is a decoding result. It is cross-subject sentence *retrieval* over a 700-sentence gallery, and the
honest summary of real ZuCo remains: decodable, and not yet subject-invariant.

## The thirteen-arm sweep (2026-07-25, held out on `ZAB`)

The full lever sweep, quoted as **held-out Top-1 hit counts out of 700** because that is what the numbers are. Chance
expects one hit.

| arm | hits / 700 | eff-rank ratio |
| --- | --- | --- |
| `exp12_align_off` | 9 | 0.29 |
| `exp8_clip_e5_raw` · `exp12_zte_raw_aligned` (s42) · `exp12_orthogonality_off` | 8 | 0.26 · 0.25 · 0.28 |
| `exp10_clip_e5_meaning_raw` · `exp12_zte_raw_aligned_wide` | 7 | 0.26 · 0.52 |
| `exp12_adapter_off` · `exp12_align_fit_train` | 6 | 0.26 · 0.24 |
| `exp12_zte_raw_aligned` | 5 | 0.23 |
| `exp10_clip_bge_meaning` | 4 | 0.16 |
| `exp10_clip_mpnet_meaning` | 3 | 0.16 |
| `exp9_clip_e5_meaning` · `exp10_clip_e5_meaning_raw_v2` · `exp8_clip_qwen` | 2 | 0.17 · 0.53 · 0.17 |

**The finding is the spread, not the ordering.** Thirteen arms flipping Euclidean alignment, the subject adapter,
identity orthogonality, the text encoder and the meaning target land between 2 and 9 hits. Turning alignment *off*
scores highest. Two seeds of one unchanged configuration have already produced 4 hits and then 2. Run-to-run noise is
the size of every effect in this table, so **no arm here is measurably better than another**, and the honest reading
is that the levers exposed at that point were exhausted. That is what motivated the exp16 architectural work rather
than a fourteenth lever.

## The decoder on real ZuCo (`exp13_decode_frozen_e5raw`, 2026-08-13)

| measurement | value |
| --- | --- |
| Held-out retrieval rank percentile | 0.9636 (0.9599–0.9674), 9 hits / 700 |
| Length-stratified rank percentile | 0.9211 (0.9154–0.9270) |
| Decoder-rescoring rank percentile | 0.7244 (0.6835–0.7654), 105 queries |
| Decoder-rescoring, **length-stratified** | **0.4349** (0.3743–0.4998) |
| Free-running generation vs its worst control | −0.0117 (−0.0239, −0.0001), permutation *p* = 0.96 |
| Prefix-influence KL | 0.4166 nats (floor 0.05) |
| Variance budget | 8.4% subject · 0.0% content · 91.6% neither |
| Same word across subjects | cosine 0.005 vs random −0.000 — *not clustered* |

Two things to state plainly. **Generation is an honest null**: it beats none of the five controls and its permutation
*p* is 0.96, which is exactly what the bit budget predicts — 1.5 encoder bits against the ~190 a 19.6-word sentence
needs. And **the decoder's rescoring contribution is entirely length**: rank percentile 0.7244 unstratified falls to
0.4349 once sentence length is held constant, which is *below* the 0.5 chance line. The prefix does influence the
output (KL 0.4166 clears the floor), so the mechanism is wired; what it carries is word count.

## The v2 decoder on real ZuCo (`decode_zte_v2_loZAB_s42`, 2026-08-15, over the v3 encoder)

| measurement | value |
| --- | --- |
| Held-out retrieval rank percentile | 0.9359 (0.9295–0.9419), 5 hits / 700 |
| Honest cell (train-fitted, length-stratified) | 0.8775 (0.8695–0.8855) vs length-oracle 0.9525 — `clears_floor: False` |
| Decoder-rescoring rank percentile | 0.6234 (0.5737–0.6736), 105 queries |
| Decoder-rescoring, **length-stratified** | 0.4325 (0.3767–0.4916) |
| Free-running generation | verdict **False** — beats 1 of 7 controls; `mean_prefix` beats the model; permutation *p* = 0.19 |
| Prefix-influence KL | 1.9943 nats (floor 0.05) |
| Encoder bit budget | 0.639 bits carried, 9.45 needed, 5.14 free from length |

The stratified-rescoring cells in both decoder tables were produced under a convention that hard-scored a query as
percentile 0.0 whenever its truth fell outside its own ±1-word stratum; that convention has been retired in the
audit code (unanswerable queries are now excluded and counted), and the cell is expected to move toward chance —
an honest null, not an anti-signal — when re-scored. Until that re-score lands, read these two rows as "the
rescoring adds nothing beyond length", not as "the LM anti-ranks the truth".

## The exp16 sweep on real ZuCo (2026-08-15, held out on `ZAB`, 700 queries)

| arm | held-out Top-1 | eff-rank ratio |
| --- | --- | --- |
| `zte_encoder_v3` s42 / s43 / s44 | 0.010 / 0.021 / 0.029 | 0.078 / 0.060 / 0.094 |
| `exp16_residual_off` s42 | **0.0371** | **0.289** |
| `exp16_gallery_off` s42 | 0.030 | 0.088 |
| `exp16_gallery_band_off` s42 | 0.027 | 0.098 |
| `exp16_consensus_off` s42 | 0.0057 | 0.071 |
| `exp16_length_projection_off` s42 | 0.0086 | 0.077 |

The sweep falsified two of the four v3 mechanisms. **The predictive residual is the collapse**: its expectation head
learns to subtract every within-sentence-predictable component of the token hiddens — which is exactly the
per-sentence-constant code retrieval scores — and it runs at inference with training-subject statistics; turning it
off nearly quadruples held-out Top-1 and more than triples effective rank. **The gallery CE hurts too**: a
single-positive classification onto frozen train-text anchors rewards any route to the anchor, subject-specific
features included, and buys pooled Top-1 at the expense of the cross-subject consistency the honest metric measures.
Consensus is the one mechanism whose removal hurts (0.0057), and the length projection is measurement-neutral.
The run's own length audit (`decode_zte_v2_loZAB_s42`): the encoder carries **0.639 bits** of sentence identity
against 9.45 needed; the honest cell (train-fitted, length-stratified) rank percentile is 0.8775 (0.8695–0.8855)
versus the ±1 length oracle's 0.9525 — `clears_floor: False`. The variance budget of the v3 space reads 41.1%
subject, 35.7% task, ~0% content — the encoder *amplifies* the task register (probe 0.918 vs 0.685 raw).

The repair family is `ablation/exp17_*` (residual off + gallery off as the base, then sentence-slice VICReg,
task-pure negatives, and the deployable alignment fit as matched pairs). The best-measured encoder arm today is
`exp16_residual_off` at one seed; seeds 43/44 are the first item on the run matrix.

**Stale pending re-measurement.** Every arm in this table except `exp16_length_projection_off` ran with
`length_projection: true`, which was fitted in the wrong frame — see [Measurement corrections](#measurement-corrections).

## The parallax transfer matrix (2026-08-16/17, held out on `ZAB`, seeds 42/43/44)

The per-task encoders (`experiments/parallax/`, the exp17 recipe on one task each) measured on real ZuCo. Full design
and per-cell CIs: [`PARALLAX.md`](PARALLAX.md).

| cell | rank percentile (s42 / s43 / s44) |
| --- | --- |
| NR → SR (never-seen subject × never-seen sentences) | 0.9507 / 0.9647 / 0.9715 |
| SR → NR (never-seen subject × never-seen sentences) | 0.9515 / 0.9577 / 0.9591 |
| NR → NR diagonal (pooled over seeds) | 0.9530 |
| SR → SR diagonal (pooled over seeds) | 0.9575 |

Length-stratified, the off-diagonal cells hold at ~0.92–0.93, and effective-rank ratios sit at 0.41–0.46 against the
v3 encoder's 0.06–0.09 — the geometry healed without giving the transfer back. The honest statement: **a
task-invariant, stimulus-set-invariant code reaches a never-seen subject at rank-percentile ~0.95–0.97,
length-stratified ~0.92; single-reference exact-length menus remain at chance; TSR carries no measurable content
signal in-task.**

The two findings behind the qualifiers, stated as measured:

- **TSR in-task is a null.** At s44: held-out Top-1 0.00246 — exactly chance on its 407-sentence gallery — lift
  −0.0003, permutation *p* = 0.998, effective rank 0.33. Healthy geometry, no content signal.
- **The certified menu is at chance on exact-length prototype pools.** K=2 accuracy 0.522 (CI 0.484–0.560,
  permutation *p* = 0.12) for NR s44; the ±1/±2 tolerance rows read 0.526 / 0.538, so tolerance is not the driver.
  The open menu's 0.707 (*p* = 0.002) is stamped `gamed: true` by its own 0.971 length oracle and self-disqualifies.
  The discriminative signal lives in individual readings, not centroids — the enrolled menu flavor exists to score
  exactly that.

**Stale pending re-measurement.** All three parallax encoders ran with `length_projection: true`, which was fitted
in the wrong frame — see [Measurement corrections](#measurement-corrections).

## The three alignment levels on real ZuCo (2026-08-22, twelve-fold LOSO, seed 42)

The granularity ablation: one contrastive term, moved between the pooled sentence vector, the single fixated word and
four fixed intra-word slices, with nothing else changed. Twelve folds each, `train_fitted` post-processing, scored on
the **length-stratified** gallery (chance Top-1 0.0285). Spread is the sample standard deviation across folds.

| level | folds | rank percentile (mean ± sd) | vs. length oracle | Top-1 (mean ± sd) | hits/700 | folds *p* < 0.05 | bits from EEG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sentence` | 12 | 0.9238 ± 0.0079 | −0.0287 | 0.0406 ± 0.0093 | 28 | 5/12 | 1.81 |
| `word` | 12 | 0.9203 ± 0.0143 | −0.0322 | 0.0417 ± 0.0100 | 29 | 7/12 | 1.74 |
| `token` | 12 | 0.9286 ± 0.0063 | −0.0239 | 0.0475 ± 0.0085 | 33 | 10/12 | 2.02 |

The length-only oracle at ±1 word reaches rank percentile **0.9525** on the same gallery. **No level clears it**, and
the ordering does not favour the level the design predicted: `token` is nominally highest on every column.

### The sub-word piece oracle, measured on ZuCo's own gallery

Four brain-free signatures scored as retrieval oracles over the real 700-sentence SR+NR gallery — no model, no
training, nothing but the reference spelling. Tokeniser `Qwen/Qwen2.5-0.5B`; word-to-text alignment coverage 0.99993,
so the piece counts are trustworthy. Naming one sentence out of 700 costs $\log_2 700 = 9.4512$ bits.

| signature | what it is told | Top-1 | hits/700 | bits of the 9.4512 | unique fraction |
| --- | --- | --- | --- | --- | --- |
| `words` | the word count | 0.0757 | 53 | 5.145 | 0.016 |
| `total` | the total sub-word piece count | 0.1014 | 71 | 5.579 | 0.023 |
| `multiset` | the piece counts, order destroyed | 0.7386 | 517 | 8.816 | 0.583 |
| `profile` | the ordered per-word piece counts | 0.9443 | 661 | 9.309 | 0.914 |

`total` is the gate this repository's design can actually reach, because `objective.token_sub_tokens` is a fixed
constant per word and never the count of pieces the reference spells that word in. `profile` is the ceiling a design
that sized a word's EEG by its own piece count would have handed over, and it resolves 661 of 700 sentences on its
own.

### What this measures

**Every level is out-retrieved by a single integer.** The best of the three retrieves 33 of 700 length-stratified and
15 of 700 unstratified; the word count alone retrieves 53, and the total sub-token count 71. The encoders are not
competing with chance, they are competing with spelling, and they lose to it by a factor of two to five.

**The ordering is the confound signature, not a win.** `token` leads on rank percentile, on Top-1, on the number of
significant folds and on the bit budget — and the token level is precisely the arm with the most exposure to the
channel the oracle table shows is worth 9.309 of the 9.4512 bits. A granularity ablation in which the most
confound-adjacent arm wins is evidence about the confound, not about sub-word neural content.

**The pre-registered null lands, and it is broader than what was registered.** `docs/ALIGNMENT_LEVELS.md` pre-registered
a null for the token level against the word level. What was measured is a null across all three levels against a
brain-free floor. The honest statement is: *on ZuCo, cross-reader sentence identity recoverable from EEG does not
exceed what the reference spelling gives away for free, at any of the three granularities tried.*

**This does not say the EEG is empty.** The bit budget puts 1.74–2.02 bits of sentence identity in the embeddings
against the 9.4512 needed, and the parallax matrix above shows a code that reaches a never-seen subject reading
never-seen sentences at rank percentile ~0.95. It says the *readout* is not yet separable from length and spelling on
this gallery, which is a statement about measurement power as much as about the signal.

The anchor these arms were designed to be read against — `exp16_residual_off` at held-out Top-1 0.0371, restated
across the docs as "26 of 700" — is a back-conversion of a rounded rate, carries no hit count, no binomial tail and no
confidence interval, is not length-stratified, and is marked stale below. It sits **below the 53-hit word-count
oracle on this same page**. It should not be quoted as a comparand again.

## Measurement corrections

Corrections to the *measuring instrument*, not to the numbers. Nothing in the tables above has been edited; what
changes is which of them may still be quoted.

### The length projector was fitted in the wrong frame

`objective.length_projection` regresses each sentence embedding on a basis of its word count and subtracts the
fitted component. The projector was **fitted on the raw training rows and applied to the whitened, all-but-the-top
rows** — two different frames. A basis fitted before post-processing does not describe the space it is subtracted
from, so the subtraction removed a direction the scored rows did not have and left the one they did.

The real-data signature is that the de-confounder made the confound **worse**, not merely weaker than hoped. In
`exp16_residual_off`, word count explained 0.0206 of sentence-embedding variance before the projection and 0.3619
after it — leakage rising by more than an order of magnitude, in the one metric whose entire job is to fall. A
projection that increases length leakage is not a partially effective projection; it is a mis-specified one, and
its retrieval numbers were measured on a space with *more* length in it than the unprojected space had.

The projector is now fitted on the same post-processed training rows it is applied to.

**Every retrieval number measured with `length_projection: true` predates this fix and must be re-measured before
it is quoted again.** That is:

| Section above | Status |
| --- | --- |
| [The exp16 sweep on real ZuCo](#the-exp16-sweep-on-real-zuco-2026-08-15-held-out-on-zab-700-queries) | **Re-measure.** Every arm except `exp16_length_projection_off` ran with the knob on |
| [The parallax transfer matrix](#the-parallax-transfer-matrix-2026-08-1617-held-out-on-zab-seeds-424344) | **Re-measure.** `parallax_{nr,sr,tsr}` all ran with the knob on |
| `ablation/exp17_*` | Configs only; no number from them has ever been quoted |
| The held-out board, the thirteen-arm sweep, and every decoder section | Unaffected — none of those configs sets `length_projection` |

The direction of the correction is not predictable from the sign of the bug: the affected numbers may rise, fall or
hold. Until they are re-measured on Drive they are neither confirmed nor retracted — they are **stale**, and a
stale number is not a result.

## Not yet measured

`flagship/zte_lexical_raw` (exp14) remains unmeasured on real ZuCo, and the `exp17_*` repair family exists as
configs only. **No number from them appears anywhere in this document.**

## Synthetic smoke-runs

The real ZuCo archives are 17–23 GB each and cannot be pulled into this environment, so the numbers below come from **schema-faithful synthetic ZuCo** — the generator reproduces ZuCo's exact struct schema (105 channels × 8 bands × 5 eye-tracking measures, empty arrays for omitted words). The point is to show the whole pipeline runs end-to-end and behaves sensibly; on real ZuCo the same commands produce the same artifacts at scale. **A synthetic run is never a result.**

> How to regenerate everything on this page is at the [bottom](#reproduce).
> For methodology see [EVALUATION.md]; for the knobs see [TRAINING.md] and [DATASET.md].

## 1. Headline pretraining run

A skip-gram (InfoNCE) model trained end-to-end via `examples/run_demo.py` on a 3-subject × 2-task synthetic tree:

| Metric                       | Value                             |
| ---------------------------- | --------------------------------- |
| Words / sentences / subjects | 614 / 72 / 3                      |
| Overall omission rate        | 0.295                             |
| Objective                    | skip-gram (InfoNCE)               |
| Epochs / device              | 15 / CPU                          |
| Final train loss             | 3.472 (monotonically decreasing)  |
| Word embeddings extracted    | 433 (present words only)          |
| Embedding dim                | 128 (768 by default in real runs) |

Train loss decreases steadily; on this tiny synthetic split the validation curve is essentially flat — expected, since the smoke-run exercises the machinery rather than benchmarking generalisation.

## 2. All four objectives train

Each objective was trained for 3 quick epochs on the same synthetic tree (10 sentences × 3 subjects -> 504 words); all converge without error on both frontends.  Loss scales are not comparable *across* objectives (different losses), only the fact that each optimises cleanly:

| Objective         | Frontend       | Final train loss (3 epochs) |
| ----------------- | -------------- | --------------------------- |
| skip-gram         | band_power_mlp | 3.758                       |
| cbow              | band_power_mlp | 4.592                       |
| masked (data2vec) | band_power_mlp | 0.282                       |
| cpc               | band_power_mlp | 4.598                       |

The raw-Conformer frontend (`--frontend raw_conformer --representation raw`) trains the masked/CPC objectives the same way; see `examples/run_demo.py` and [TRAINING.md].

## 3. Dataset analysis (auto-generated by `zte.data.viz.save_overview`)

The synthetic generator injects the structure the real corpus exhibits, and the analysis tooling recovers it:

Short words are skipped far more often (omission ≈ 0.5 at length 1–2, falling with length) and total reading time rises with word length — exactly the lexical effects ZuCo is known for.

## 4. Evaluation of the learned space

Running the full evaluation suite (`examples/evaluate_zte.py`, a 4-subject skip-gram model, `embed_dim=96`) on **664 word / 112 sentence** embeddings gives the verdict:

| Check                                | Result                          | Detail                                                  |
| ------------------------------------ | ------------------------------- | ------------------------------------------------------- |
| Beats the noise control              | ✓ (word_len, log_freq, subject) | linear probe                                            |
| No representation collapse           | ✓                               | effective-rank ratio **0.57** (54.6 / 96), 0% dead dims |
| Cross-subject retrieval above chance | ✗                               | Top-1 0.054 vs chance 0.054                             |
| Subject arithmetic above chance      | ✓                               | Top-1 0.010 vs chance 0.006                             |

Effective rank quantifies how many dimensions the space meaningfully spans: from the singular values $\sigma_k$ of the centred $Z\in\mathbb{R}^{n\times d}$ with $p_k=\sigma_k/\sum_j\sigma_j$,

$$
\mathrm{erank}=\exp\!\Big(-\sum_k p_k\log p_k\Big),
$$

and the reported **0.57** is $\mathrm{erank}/d = 54.6/96$. Retrieval is scored by $\text{Recall@}k$ — the fraction of $Q$ queries whose correct match ranks in the top $k$ (Top-1 is $k=1$):

$$
\text{Recall@}k=\frac{1}{Q}\sum_q \mathbf{1}[\text{rank}_q\le k].
$$

The embedding is healthy (well-spread, no collapse) and carries lexical attributes above the noise floor. Cross-subject retrieval sits **at** chance here — expected, because synthetic subjects share no real neural structure to retrieve across. That is precisely the gap real ZuCo is meant to close, and why the suite reports it honestly rather than hiding it.

See [EVALUATION.md] for what every figure and metric means, plus the per-subject/per-task breakdowns, analogy transfer and scalp-region importance.

## Reproduce

```sh
# 1) Headline pretraining run + dataset figures + training curve.
uv run python examples/run_demo.py --objective skipgram --epochs 15 --sentences 12

# 2) Each objective (swap --objective for cbow | masked | cpc).
uv run python examples/run_demo.py --objective masked --epochs 3 --sentences 10

# 3) Full evaluation suite (figures, report.md, verdict, interactive HTML, TensorBoard).
uv run python examples/evaluate_zte.py --out res/eval_demo/evaluation

# 4) The test suite exercises the same code paths on every objective.
uv run pytest
```

Point the `--ckpt` / `--root` / `--drive` flags of `zte-evaluate` and `zte-extract` at real artifacts to reproduce these tables on genuine ZuCo recordings.

[DATASET.md]: ./DATASET.md
[EVALUATION.md]: ./EVALUATION.md
[TRAINING.md]: ./TRAINING.md
