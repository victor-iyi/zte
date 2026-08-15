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

## Not yet measured

`flagship/zte_encoder_v3` (exp16) and its five matched ablations exist as configs and pass the synthetic smoke path.
**No number from them appears anywhere in this document**, because none has been produced on real ZuCo. They sit in
`flagship/` as the best-designed recipe, not the best-measured one; `zte_raw_aligned` still holds that title.

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
