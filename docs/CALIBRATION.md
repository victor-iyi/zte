# Anchor calibration — what a new reader gains from a few labelled sentences

A deployed BCI meets a brain nobody trained on. The question this experiment answers is the deployment question:
**how many sentences must a new reader read, with the text known, before their retrieval improves — and does it
improve at all?**

Nothing here retrains. The encoder is frozen; only a per-subject map into the shared space is fitted, from the new
reader's own labelled readings. That is what makes it a deployment property rather than a training result.

Run it with [`zte-calibrate`](#running-it); the curve, its controls and its verdict land in `calibration.json` and
`calibration.md`.

## What is fitted

Let $A \subset \{1, \dots, 700\}$ be the **anchor stimuli** — the sentences the new reader read whose text is known.
For each $s \in A$, write $q_s \in \mathbb{R}^d$ for the held-out reader's embedding of $s$, and $r_s$ for the
**cross-subject prototype**: the mean embedding of $s$ over every *other* subject.

$$
r_s \;=\; \frac{1}{|S_s|}\sum_{u \in S_s} z_{u,s}
$$

where $S_s$ is the set of subjects other than the held-out one who read $s$. The prototype is the target because it
is the one a deployed system can actually construct: it needs the cohort, not the new person's labels beyond $A$.

Both map families are affine in one shape — centre on the anchor mean, apply a matrix, land on the cohort mean:

$$
f(z) \;=\; \big(z - \bar{q}\big)\,M \;+\; \bar{r}
$$

with $\bar{q} = \frac{1}{|A|}\sum_{s \in A} q_s$ and $\bar{r} = \frac{1}{|A|}\sum_{s \in A} r_s$. They differ only in
what $M$ is allowed to be:

| family | $M$ | why it is in the pair |
| --- | --- | --- |
| `procrustes` | the orthogonal matrix minimising $\sum_{s \in A} \lVert (q_s - \bar{q})M - (r_s - \bar{r})\rVert^2$, from the SVD of the cross-covariance | rotation only, so it cannot inflate a retrieval number by rescaling directions the metric happens to like |
| `ridge` | the closed-form ridge solution $\big(Q^\top Q + \alpha I\big)^{-1} Q^\top R$ | strictly more expressive, so the two together bracket what *any* affine calibration could buy |

If `procrustes` helps and `ridge` does not help more, the gain is a genuine rigid misalignment between the reader and
the cohort. If only `ridge` helps, the gain is at least partly the extra capacity — which is what the shuffled
control below is there to price.

## The three arms, on one gallery

At each anchor count $n$, three arms are scored **on the identical reduced gallery**:

| arm | what it is | what it controls for |
| --- | --- | --- |
| `uncalibrated` | the frozen embeddings, anchors removed | the gallery got smaller, so retrieval gets easier for free |
| `calibrated` | $f$ fitted on the true $(q_s, r_s)$ pairs | the measurement |
| `shuffled` | $f$ fitted on a **derangement** of those pairs | the transform's raw capacity |

The `shuffled` arm is the mandatory one and it is the reason this experiment can be believed. It fits the same map,
with the same number of parameters, on the same number of pairs — but each anchor reading is paired with the *wrong*
reference. Any lift it produces is what the transform buys by existing, not by being calibrated. **A curve that does
not beat its own shuffled control is not a calibration result**, and `verdict[family].beats_shuffled` records exactly
that.

### The exclusion that makes it honest

Every anchor stimulus is removed **from the query set and from the gallery**. Leaving an anchor in the gallery would
let the map place a sentence it was fitted on next to itself, which manufactures the entire effect. The payload
carries `n_queries` and `n_gallery` at every point so the shrinkage is visible: at $n = 200$ roughly 29% of a
700-sentence gallery is gone, and the comparison at that point is a genuinely easier problem than at $n = 0$.

That is why $n = 0$ is re-scored on each reduced gallery rather than once on the full one. The `uncalibrated` column
at anchor count $n$ is the *same* gallery as the `calibrated` column beside it. Comparing a calibrated 500-sentence
gallery against an uncalibrated 700-sentence one would show a lift that is pure arithmetic.

## Which number is the result

**`rank_percentile` on the length-stratified gallery, with its bootstrap interval over anchor draws.** Which
particular 10 sentences a reader happened to be given is a real source of variance, so every anchor count is repeated
over `--draws` seeded draws and the interval is taken across them. `lift` and `margin_over_shuffled_ci` are **paired**
per-draw differences, not differences of means.

Top-*k* travels as a hit count with an exact binomial tail computed on **one draw's worth of queries** — pooling
draws would count the same reader many times, and `p_basis` records that it did not.

## What is already known before running it

Two facts bound what this experiment can find, and both belong beside any result it produces.

**The existing cohesion measurement is not a retrieval measurement.** `zte.evaluation.audit.honesty` has carried
`anchor_calibration_lift` for some time: it fits an orthogonal Procrustes from ~12 anchor **words** and reports the
change in *cohesion* — the mean cosine between same-word cross-subject centroids. Its fitted map has never been
applied to a scored embedding; `_calibrate_one` discards it. A positive cohesion lift is therefore **not** evidence
that Top-*k* or rank percentile would move, and the two have never been connected in this repository. Any figure
quoting "+0.0628 lift from 12 shared words" is quoting that cohesion diagnostic, at word level, and it does not
transfer to this curve's claim.

**Some calibration is already in the baseline.** `dataset.raw_align_fit` defaults to `'all'`, so the per-subject
whitening reference for the LOSO holdout is already fitted on the holdout's own windows. The project treats that as
label-free calibration rather than leakage, and it travels as `held_out_retrieval['alignment_fit']`. The curve
therefore measures what **labelled** anchors add *on top of* unlabelled whitening that has already happened. A
comparison against a `raw_align_fit: 'train'` run is a different experiment and must say so.

## Running it

```sh
uv run zte-calibrate --ckpt <run>/checkpoints/best.pt --root <zuco> \
  --anchor-counts 0,10,25,50,100,200 --draws 5 --family both
```

| flag | default | what it changes |
| --- | --- | --- |
| `--anchor-counts` | `0,10,25,50,100,200` | the sweep; `0` is the gallery-matched control and is always scored |
| `--draws` | `5` | seeded anchor draws per point, which is what the interval is taken over |
| `--family` | `both` | `procrustes`, `ridge`, or both — the bracket described above |
| `--ridge-alpha` | `1.0` | ridge regularisation; a small $d$-dimensional fit on 10 anchors is underdetermined and is flagged |
| `--postprocess` | off | fit whitening and all-but-the-top on cohort rows only, before calibrating |
| `--length-tol` | `1` | the word-count tolerance defining the length-stratified gallery |
| `--holdout` | the run's own | which reader is the stranger |

Artifacts are `calibration.json` and `calibration.md` in `--out` (default `<run>/calibration`), guarded by the
`.zte-done` signature so a re-run against an unchanged checkpoint costs seconds.

### Reading the payload

`curve` is the flat plottable series — one record per family × anchor count, carrying `rank_percentile`,
`rank_percentile_ci`, `lift`, `shuffled_lift`, `margin_over_shuffled_ci`, `n_queries` and `n_gallery`.
`series[family][gallery]` is the same thing as parallel arrays. `detail` holds the full per-draw aggregation, and
`verdict[family]` carries `helps`, `beats_shuffled` and the sentence that combines them.

Three fields decide whether a point may be quoted at all:

- **`underdetermined`** — the fit had fewer anchors than dimensions. A `ridge` map at $n = 10$ on a 768-dimensional
  space is interpolating, and the flag says so.
- **`degraded_fits`** — how many draws at that point failed to fit and fell back. `fit_calibration` returns `None`
  and logs rather than silently returning an identity, so a degraded point is visible rather than a quiet no-op.
- **`saturated`** — the requested anchor count exceeded the stimuli the reader actually has, so the point is not the
  count it was asked for.

## What this experiment cannot show

- **It is not a generation claim.** The readout is closed-set retrieval over the sentence gallery.
- **It is not a within-subject result.** The map is fitted on the held-out reader and scored on that reader's *other*
  sentences; it says nothing about how the encoder would do on a reader in the training set.
- **It does not clear the length floor by itself.** The floor rules in [`EVALUATION.md`](EVALUATION.md) apply here
  unchanged: a calibrated rank percentile is read against the same ±1-word length oracle every other number is, and
  the [evidence board](EVALUATION.md) renders a calibration row that fails its shuffled control as failing.
