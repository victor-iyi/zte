# Subject alignment — cancel the brain, keep the meaning

How ZTE handles the fact that every skull is different, and why the usual answer cannot work for the one subject
that matters in a leave-one-subject-out test.

Config: [`experiments/flagship/zte_raw_aligned.yaml`](../experiments/flagship/zte_raw_aligned.yaml) (exp12).
Code: `zte.data.features.alignment`, `zte.models.subject`, `zte.models.objectives.losses.identity_orthogonality`.

## The problem, stated precisely

Two people reading the same sentence produce different voltages for two unrelated reasons. One is that they are
thinking slightly different things — that is signal. The other is that their skull thickness, cap placement and
electrode impedances differ — that is a linear nuisance that lives almost entirely in the channel covariance, and it
is far larger than the signal.

Cross-subject retrieval therefore fails in a specific way: the model learns *who* long before it learns *what*, and
on a held-out subject it has learned a mapping keyed to 11 brains that the 12th does not match.

## What was actually wrong here

Three things, found by re-scoring every run on Drive on 2026-07-25.

**1. The raw path had no alignment at all.** `dataset.normalize` is applied to `features` (band power) in
`ZuCoDataset._process`. `raw_eeg` receives `sanitize_raw_windows` and an optional band-pass, and nothing else. Every
raw-conformer config on the board carried `normalize: riemannian`, and for every one of them it was a **silent
no-op**. The best arm on the board was training on completely unaligned voltages.

**2. The subject-conditioning mechanism was inert exactly where it was needed.** `model.subject_film` is an
`nn.Embedding` indexed by subject id, zero-initialised so that an unseen id is the identity map. Under LOSO the
held-out subject *is* the unseen id. So the model applied a learned correction to all 11 training subjects and no
correction to the one being tested — the mechanism was guaranteed to be useless on the away game. This is not a bug
in the implementation; it is the structural limit of the standard per-subject layer (Défossez et al., 2023), which
is always trained and evaluated with every subject present.

**3. Invariance was being bought with capacity.** A gradient-reversal adversary asks that subject identity be
*unpredictable* from the representation. An encoder that stops representing anything satisfies that perfectly. The
band-power arms did exactly this: subject probe fell to 0.23, but their effective-rank ratio was 0.160 — the 768-d
space had collapsed to roughly 123 directions. The invariance metric improved because the representation died.

## The three changes

Each is **label-free**: it reads no text, no split and no label — only voltages the subject themselves produced.
That is what makes all three legitimate to apply to the held-out subject.

### 1. Euclidean alignment (`dataset.raw_align: euclidean`)

For each subject, estimate the mean per-trial channel covariance and whiten by its inverse square root, so every
subject arrives with the same second-order statistics (He & Wu, 2019). Trace-normalising each trial first stops a
few high-amplitude trials from owning the reference; Ledoit-Wolf shrinkage keeps the root well-conditioned for short
recordings.

`raw_align_fit: all` includes the held-out subject. This is deliberate and is not leakage — the map is a function of
that person's own covariance and nothing else. Withholding it would not make the number more honest; it would
simulate a BCI that refuses to calibrate its own cap. The strict alternative is available as
`raw_align_fit: train` and is run as an ablation (`exp12_align_fit_train`), so the choice is measured rather than
asserted.

### 2. A subject adapter keyed on inferred statistics, not identity (`model.subject_adapter`)

This is the part that is new.

Rather than looking a subject up, compute their **signature**: a fixed-width descriptor of their covariance
geometry, made of the per-channel log scale (impedance and cap contact) concatenated with the log-Euclidean tangent
vector of their region-averaged covariance (the coarse orientation of their forward model). For the ZuCo-105
montage that is 141 numbers. A small hypernetwork maps the signature to a per-electrode gain, applied to the raw
window before the frontend, and a FiLM affine applied to the token hiddens after it.

The consequence is the point: **a subject the model has never seen is not a missing table row, but a point in
signature space that the hypernetwork interpolates to.** A stranger puts on the cap, reads anything for thirty
seconds, and the network configures itself — `ZTEEmbedder.calibrate_subject(baseline_raw=..., subject_code=...)`
is the whole procedure. No labels, no fine-tuning, no retraining.

The signature is computed **before** whitening, because whitening drives every subject's covariance to the identity
and would flatten the descriptor to a constant. The division of labour is deliberate: alignment cancels the linear
part of the individual difference, and the adapter models the nonlinear residual that a single linear map cannot
reach.

The head is zero-initialised, so an untrained adapter is exactly the identity and cannot destabilise early
training; it learns adaptation as a correction to a working encoder.

### 3. Identity orthogonality (`objective.identity_orthogonality_weight`)

Instead of making identity unpredictable, require that content be *uncorrelated* with it: the normalised Frobenius
norm of the cross-covariance between the content subspace and the signature (a linear-CKA statistic). A full-rank
space that happens to be identity-free scores zero and pays nothing, so the collapse shortcut is closed. The term
is scale-free, so shrinking the representation earns no credit either.

In the operating regime (a batch of ~2500 usable tokens, a signature that is constant per subject and therefore
rank ≤ 11) the statistic separates cleanly: ~0.02 for independent content versus ~0.93 for content carrying a
per-subject offset. The subject adversary is kept at half its previous weight as a second referee rather than
removed, so the two are not fighting over the same gradient.

## Reading the result

The scoreboard now leads with **rank percentile and a 95% bootstrap CI**, because every query contributes to it,
and reports Top-K as **raw hit counts against the number expected by chance, with an exact binomial tail**.

This matters more than it sounds. Held-out retrieval on ZuCo has ~700 queries at 1/700 chance, so Top-1 expects
**one** hit. A headline of "0.006 vs 0.001 chance" is three hits — *p* ≈ 0.08, indistinguishable from noise, and it
is how the previous champion was crowned. Rates are not readable at that count; counts and tails are.

## Ablations

Each config is byte-identical to the flagship except for one knob
([`experiments/ablation/`](../experiments/ablation/)):

| Config | Isolates |
| --- | --- |
| `exp12_align_off` | Euclidean alignment — can the adapter cancel the forward model alone? |
| `exp12_adapter_off` | The hypernetwork — how much of the gap is nonlinear residual? |
| `exp12_orthogonality_off` | The rank-preserving penalty vs the adversary. **Watch effective rank, not retrieval.** |
| `exp12_align_fit_train` | Held-out calibration — what the number becomes if a new user may never calibrate. |

```bash
STUDIES="flagship ablate" bash scripts/run_suite.sh /path/to/zuco_extracted
```

## Cost and compatibility

Alignment is applied **after** the cached dataset bundle loads, not baked into it, so enabling it never invalidates
a prepared bundle — `raw_align`, `raw_align_fit` and `subject_signature` are excluded from the cache key. The
covariance work is a handful of 105×105 eigendecompositions, negligible against the `.mat` parse.

Defaults are off (`raw_align: none`, `subject_signature: false`, `subject_adapter: false`), so every existing config
behaves exactly as before; with no signature the adapter is not constructed and the forward path is unchanged.

The fitted maps and signatures are embedded in the checkpoint, so `ZTEEmbedder` reproduces the exact alignment at
inference without the training data.

## References

- He, H. & Wu, D. (2019). Transfer learning for brain-computer interfaces: a Euclidean space data alignment
  approach. *IEEE TBME*.
- Défossez, A. et al. (2023). Decoding speech perception from non-invasive brain recordings.
  *Nature Machine Intelligence*. — the per-subject layer this replaces.
- Barachant, A. et al. Riemannian geometry / tangent-space methods for EEG covariance.
- Kornblith, S. et al. (2019). Similarity of neural network representations revisited. — the CKA statistic.
- Ganin, Y. & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. — the adversary this
  demotes.
