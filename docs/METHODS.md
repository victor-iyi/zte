# ZTE methods — a meaning-oriented, subject-invariant thought code

The methods behind ZTE's second-generation thought embedding. The approach turns the EEG recorded while a person reads a word into a 768-dimensional code that depends on *what* was read rather than *who* read it, by treating the signal as a superposition of generative factors, disentangling them, and aligning them across brains.

Configuration: `experiments/flagship/zte_raw_aligned.yaml` (the champion recipe); the same levers in their skip-gram form are kept as the control, `experiments/benchmark/baseline_skipgram_loso.yaml`. Every method is an independent, config-gated lever, so each is validated in isolation with `zte-ablate` against the held-out-LOSO scoreboard (`zte.evaluation.audit.scoreboard`).

---

## 0. The generative thesis — the neural superposition

A word-evoked EEG response $x \in \mathbb{R}^{8 \times 105}$ (8 bands $\times$ 105 electrodes) is not *meaning + noise*. It is a sum of simultaneous, physically distinct sources:

$$
x \;=\; s_{\text{content}}(\text{word}) \;+\; s_{\text{subject}}(\text{anatomy}) \;+\; s_{\text{task}}(\text{attention}) \;+\; s_{\text{behaviour}}(\text{oculomotor}) \;+\; s_{\text{state}}(\text{arousal}) \;+\; \varepsilon
$$

- **content** — the N400 ($\sim$300–500 ms centro-parietal negativity; Kutas & Hillyard 1980), whose amplitude tracks lexical surprisal.
- **subject** — the EEG *forward model*: individual skull conductivity and cortical folding project cortical sources onto the scalp differently for each person. This is real biophysics, not noise, and it is the largest source of variance in reading EEG (a linear probe reads identity at $\sim$99%).
- **task / state / behaviour** — top-down attentional set, arousal, and the oculomotor dynamics of fixations and regressions.

The v1 failure — the space encoding identity, not meaning — follows directly: identity is the loudest term, and an unconstrained objective grabs the loudest correlate of the label. The premise here is that these terms should be **separated and modelled**, not deleted, because several of them (frequency drives both the N400 and fixation duration) are *correlated with meaning*, so blind deletion removes content with the nuisance.

---

## 1. Subspace Factorization (disentanglement)

**Idea.** Split the embedding into a content subspace and a nuisance subspace, $z = [\,z_c \mid z_n\,]$ with $z_c \in \mathbb{R}^{d_c}$ and $z_n \in \mathbb{R}^{d - d_c}$, and route each objective to the subspace it belongs to: the meaning target and retrieval read only $z_c$; the subject adversary acts on $z_c$ to push identity *out* of content (its gradient-reversed classifier makes $z_c$ un-decodable for subject). Identity then has room to live in $z_n$ instead of contaminating $z_c$.

**Why.** An adversary alone only pushes a nuisance down; it never pulls content up, and the freed variance is absorbed by the next-loudest correlate (empirically, task rose to 85% when identity was driven to 0%). Giving each factor its own subspace is the standard fix from speaker/content disentanglement in speech.

**Math.** $z_c = P_c\,z$, with $P_c$ the projection onto the first $d_c$ coordinates. The subject adversary $g$ is trained by gradient reversal to *maximise* the conditional entropy of subject given the content slice, $\max_{\theta}\, H\big(\text{subject} \mid z_c\big)$, i.e. to make $z_c$ un-informative about identity.

**Config:** `model.factored`, `model.content_dim`.

**Code:** `models/objectives/` (`_content_slice`, adversary routing), `evaluation/audit/scoreboard.py` (judges the content slice).

## 2. Lexical Meaning Distillation

**Idea.** Distil $z_c$ toward a frozen language-model vector $v(\text{word})$, so the objective has an explicit meaning target skip-gram never had.

**Why.** LM surprisal robustly predicts N400 amplitude (Frank et al. 2015; Heilbron et al. 2022, *Nat. Commun.*), so aligning the content subspace to a static LM embedding grounds the target in the best-established EEG$\leftrightarrow$meaning relationship. Random contrastive negatives are separable by identity/task, so they never force the token in; an explicit target does.

**Math.** cosine distillation

$$
\mathcal{L}_{\text{meaning}} \;=\; 1 - \cos\!\big(W z_c,\; v(\text{word})\big),
$$

with $W$ a learned projection sized to the teacher width.

**Config:** `objective.meaning_distill_weight`, `meaning_source` (GloVe/fastText/LM file; hash fallback for mechanism tests), `meaning_dim`.

**Code:** `data/targets/meaning.py`, `objectives.py`. Provision real vectors with `scripts/build_meaning_vectors.py`.

## 3. Confound-Matched Contrastive Negatives

**Idea.** In the skip-gram InfoNCE, restrict negatives to tokens that share the anchor's confound (same subject and task); positives stay eligible regardless.

**Why.** The confound audit (`zte-audit`) shows task is fully confounded with the stimulus (Cramér's $V \approx 0.99$). If negatives differ from the anchor in subject/task, the softmax can be won by reading identity/task off them; forcing negatives to *match* on those axes leaves the word as the only discriminative signal.

**Math.** for anchor $i$ and candidate $j$, the negative is admissible iff

$$
M_{ij} \;=\; \big(\text{subject}_i = \text{subject}_j \;\wedge\; \text{task}_i = \text{task}_j\big) \;\vee\; \text{pos}_{ij}.
$$

**Sentence level.** `objective.within_task_negatives` extends the same discipline to every sentence-level
contrastive denominator — the CLIP in-batch InfoNCE, the gallery CE (whose sparse-row fallback becomes
drop-the-anchor rather than widen-to-full-gallery), the consensus prototype gallery and the decoder grounding
negatives. Under $V \approx 0.99$ a cross-task distractor can be rejected on register alone, which is how an encoder
comes to *amplify* the task probe (0.918 measured, against 0.685 for raw features); a task-pure denominator leaves
content as the only discriminative signal. Per-text task labels are joined from `(sentence_text_id, task_id)` pairs,
never parsed from vocabulary keys, and a text appearing under two tasks is a loud error. Off by default. Expect the
full-gallery Top-1 to *fall* when this is on — losing the task subsidy is the success mode, and the within-task,
length-matched delta is the number that judges it.

**Config:** `objective.hard_negatives`, `hard_negative_keys` (skip-gram); `objective.within_task_negatives`
(sentence level).

**Code:** `objectives.py::SkipGramObjective`; `clip.py`, `encoder/gallery.py`, `encoder/consensus.py`,
`objectives/decode.py` for the sentence-level masks.

## 4. Riemannian Subject Alignment

**Idea.** Whiten each subject's feature covariance to a shared reference before encoding, so the second-order structure the forward model imposes is removed. Per subject $s$ with baseline mean $\mu_s$ and covariance $\Sigma_s$, transform

$$
x \;\mapsto\; \Sigma_s^{-1/2}\,(x - \mu_s).
$$

**Why.** The LOSO "cone" is the forward model reasserting itself on a stranger. Covariance carries the subject fingerprint (this is why Riemannian / tangent-space alignment is state-of-the-art for cross-subject EEG transfer; Barachant; Zanini et al. 2018). Per-channel z-scoring only removes the mean; recentring the covariance removes the whole second-order fingerprint.

**Math.** the SPD inverse square root comes from the symmetric eigendecomposition $\Sigma = V \Lambda V^{\top}$, giving $\Sigma^{-1/2} = V \Lambda^{-1/2} V^{\top}$, with Ledoit–Wolf shrinkage for conditioning:

$$
\Sigma \;\leftarrow\; (1-\gamma)\,\Sigma \;+\; \gamma\,\frac{\mathrm{tr}\,\Sigma}{d}\,I .
$$

**Config:** `dataset.normalize='riemannian'`.

**Code:** `data/features/transforms.py`.

## 5. Oculomotor Privileged Supervision

**Idea.** Train an auxiliary head to predict per-word reading difficulty (total reading time, regression time $\text{GPT} - \text{GD}$, skipping) from $z_c$.

**Why.** Eye-movement control in reading is lexically driven — readers fixate longer on low-frequency, less-predictable words (E-Z Reader, Reichle et al.; SWIFT, Engbert et al.), so fixation behaviour is a *proxy for lexical processing load* that is meaning-adjacent but not identity-correlated. Predicting it (learning-using-privileged-information; Vapnik) injects a content-shaped gradient. ZuCo's simultaneous eye-tracking makes this free.

**Math.** a masked multi-target loss over difficulty signals $y_k$ (regression, or BCE for the binary *skipped* target), summing only over cells present by design:

$$
\mathcal{L}_{\text{behaviour}} \;=\; \sum_{k} \mathbf{1}[\,y_k \text{ finite}\,]\;\ell_k\!\big(\text{head}(z_c),\, y_k\big).
$$

**Config:** `objective.behaviour_weight`, `behaviour_targets`.

**Code:** `data/targets/behaviour.py`, `objectives.py`.

## 6. Band-Family Routing

**Idea.** Encode theta/gamma ($t_1,t_2,g_1,g_2$) and alpha/beta ($a_1,a_2,b_1,b_2$) through separate pathways.

**Why.** The bands are functionally distinct: theta indexes lexical retrieval / working-memory load and gamma indexes semantic unification (Bastiaansen; Hagoort), whereas alpha/beta carry attention and arousal *state*. Routing them separately lets invariance pressure fall asymmetrically — protect the theta–gamma content path, scrub the alpha–beta state path — instead of mixing them in a flat 840-vector. (Mutually exclusive with spatial encoding in the current frontend; use one.)

**Config:** `model.band_routing`.

**Code:** `models/frontends/::BandRoutedMLP`.

## 7. Electrode Spatial Encoding

**Idea.** Add each electrode's scalp position to its token via the real spherical-harmonic basis $Y_l^m(\theta, \phi)$ — the Laplace–Beltrami eigenfunctions of the sphere, the correct generalisation of sinusoidal position encoding from a line to the scalp surface.

**Why.** Reading engages a left-lateralized network (occipitotemporal VWFA, posterior temporal, inferior frontal); a geometry-aware channel encoding lets the model exploit it. Empirically the single most effective identity-reducing lever found so far.

**Montage.** ZuCo v1/v2 use the 129-channel EGI HydroCel net (E1–E128 + Cz vertex reference, named E129) with standard channel ordering, so `scripts/export_montage.py --zuco105` reproduces the retained 105 scalp electrodes with real coordinates — no manual channel list needed.

**Config:** `model.spatial_encoding`, `dataset.montage_csv`.

**Code:** `models/spatial.py`; details in `docs/SPATIAL_ENCODING.md`.

## 8. Zero-Shot Subject Calibration (encoding a new brain)

**Idea.** The encoder takes no subject-ID; identity enters only at the normaliser. So a genuinely new person needs only a short *unlabelled* baseline to compute their own normalisation — per-subject mean/std, or the Riemannian whitening map $\Sigma_{\text{new}}^{-1/2}$ — then their words embed on the shared frame with no labels and no retraining.

**Why.** This is the cheapest new-subject path: adapt the *one* place identity enters, not the encoder. Riemannian calibration is the principled version — it recentres the new brain's covariance $\Sigma_{\text{new}}$ to the training reference.

**Code:** `FeatureNormalizer.calibrate_subject`, `ZTEEmbedder.calibrate_subject` + `embed_signals(subject_codes=...)`.

---

## Evaluation & proof

- **Scoreboard** (`zte.evaluation.audit.scoreboard`) — held-out-only geometry, cross-subject held-out retrieval (query = the stranger, gallery = trained-on people), every probe stated as $\text{ZTE} - \text{raw}$, and a content-probe positive control. Rendered at the top of every `report.md`.
- **Confound audit** (`zte-audit`) — the factor-entanglement table that motivates the confound-matched negatives (Section 3).
- **Ablation** (`zte-ablate`) — one-knob sweeps + scoreboard diff, so each method's held-out LOSO contribution is attributable. A method is kept only if it moves the north-star.

The north-star metric is **held-out-LOSO cross-subject retrieval**, not in-sample scores. A method that improves a home-game number but not the away game has not earned its place.

---

## The road to state-of-the-art

The generative thesis above disentangles the neural superposition into content and nuisance subspaces. It left one blocking result: on the held-out subject **ZAB**, cross-subject retrieval was **below chance** on an otherwise healthy (non-collapsed, isotropic) space. Evaluation diagnosed this not as a lack of signal but as **hostile target geometry** (anisotropy + hubness breaking the nearest-neighbour graph) plus an **over-aggressive subject adversary** (full strength from step 0) eroding content it is confounded with.

The **retrieval-and-honesty layer** is the implemented fix, layered on top of the factored model, in three groups:

- **Tier 1 — geometry & invariance:** all-but-the-top + CSLS retrieval correction
  (`objective.all_but_top`, `objective.csls_neighbors`), a rebalanced and DANN-ramped subject adversary
  (`objective.subject_adversary_weight: 0.1`, `objective.subject_adversary_warmup_ratio`), and the exact
  GSN-HydroCel montage (`dataset.montage_csv`).
- **Tier 2 — sharpen the contrastive objective:** the alignment half of align+uniformity
  (`objective.alignment_weight`), debiased InfoNCE (`objective.tau_plus`), a collapse-proof frozen-target
  nuisance regression (`objective.data2vec_aux_weight`), and a per-occurrence contextual meaning target
  (`objective.meaning_contextual`).
- **Tier 3 — architecture & evaluation hardening:** zero-init per-subject FiLM and learned spatial
  attention (`model.subject_film`, `model.spatial_encoding: spatial_attention`), and a hardened verdict
  (permutation-*p* AND-ed with the bootstrap CI; phase-shuffle, seen/novel, and frequency-matched
  controls; rank-percentile / median-rank reporting).

The stack is carried by the flagship CLIP configs, `experiments/flagship/zte_raw_aligned.yaml` (geometry-fixed spherical-harmonic recipe) and `experiments/flagship/clip_e5_raw.yaml` (the raw-conformer arm); `experiments/benchmark/baseline_skipgram_loso.yaml` runs the same levers under skip-gram as the control. The spatial-attention + FiLM + shrunk-`content_dim` A/B (`experiments/archive/exp7_sota_geom_invariance.yaml`) is archived — it scored 0.0 with permutation *p*=1.0 on held-out ZAB, so it is a recorded failed arm rather than an active comparison. The win condition stays honest: a **rank distribution left of the permutation null** and a **positive content-lift-over-raw on the held-out subject**, not a headline top-1 (EEG single-word retrieval is the hardest non-invasive setting).

## Lever: the standard EEG architectures as controls

**Idea.** Swap the raw conformer for **EEGNet** and **DeepConvNet** and run them through the identical InfoNCE
pipeline. The objective never sees the frontend, so only the encoder changes.

**Why.** An ablation against a band-power MLP does not answer whether the conformer is the right architecture. The
established EEG deep-learning baselines do, and if none of the three clears the length-oracle floor, the honest
conclusion is that the readout is confound-bound for all of them and architecture is not the binding constraint —
which is a more useful statement than a ranking inside the noise.

**Config:** `model.frontend: eegnet | deep_conv_net`, with `eegnet_f1` / `eegnet_depth` / `eegnet_kernel` /
`eegnet_dropout` and `deepconv_filters` / `deepconv_kernel`. Arms: `experiments/benchmark/eegnet_clip.yaml` and
`deepconvnet_clip.yaml`, each a one-lever change against `experiments/alignment/sentence/combined.yaml`.

**Code:** `models/frontends/eegnet.py::EEGNet`, `::DeepConvNet`; dispatched by the `match` in
`models/frontends/__init__.py::build_frontend`, which now raises on an unknown name instead of silently
constructing a conformer.

### How the baselines are adapted to a retrieval pipeline

Both nets are published as classifiers, so the same surgery is applied to each. The classification head — the dense
layer over the flattened feature map and its softmax — is removed. What remains is the convolutional trunk, and its
flattened output ($F_1 D \cdot \lfloor T/32 \rfloor = 16 \cdot 10 = 160$ for EEGNet-8,2 and $200 \cdot 2 = 400$ for
the four-block DeepConvNet at the 350-sample window) is lifted by one linear layer to the encoder's token width,
`hidden_dim = 256`: exactly where the conformer hands over its own token. From that point the path is shared and
unchanged — the electrode mixer in front of the trunk, the four-layer rotary contextualiser, the masked attention
pool, and the `Linear(256, 512) -> GELU -> Dropout -> Linear(512, 768)` projection whose unit-norm output the
objective scores. The baselines therefore replace the *token encoder* alone; the sentence-level machinery, the loss,
the seed and the step budget are byte-identical to the flagship arm, which is what makes Table VI a matched
comparison. Sub-word tokens are read the way the conformer's are: the linear lift is applied to the trunk's output
restricted to one of four contiguous spans, so every span is read by the same weights as the whole word.

### Four deviations from the published architectures

These are departures a reviewer will ask about, so they are recorded rather than buried.

**Group normalisation replaces batch normalisation.** Both published nets use `BatchNorm2d`. This pipeline's
frontend contract passes **no mask** — padded word slots arrive as all-zero windows and are discarded downstream —
so batch statistics fitted over them would make a word's embedding depend on which words shared its batch. Both
classes use per-token `GroupNorm(1, F)` instead. These are therefore EEGNet and DeepConvNet *with a normalisation
swap*, not bit-exact reimplementations; a literal `BatchNorm` would need a masked frontend path, not a one-line
change.

**DeepConvNet's temporal kernel defaults to 5, not 10.** At the live `raw_window: 350` a four-block stack with the
original kernel of 10 underflows before the last block, so 5 is what makes the published depth reachable at this
window length at all. `deepconv_kernel` exposes it.

**`eegnet_kernel` defaults to the published 64, which means something different here.** The original defines it at
128 Hz, where 64 samples is half a second. ZuCo raw is 500 Hz, so 64 samples is **128 ms**. The faithful
translation would be 250, at roughly four times the compute. The configs ship 64 and the knob is exposed; this is
a defensible choice, not an obviously correct one.

**DeepConvNet's sub-word path is weaker than the conformer's.** Its pooling stack leaves about two time steps at
the 350-sample window, so `sub_tokens(x, 4)` cannot return four genuinely distinct intra-word spans — some repeat.
That is a property of the architecture's aggressive pooling rather than a bug, and it means a token-level number
from DeepConvNet is not comparable to one from the conformer.

One more thing a benchmark run must not do silently: **EEGNet's depthwise `Conv2d(F1, F1·D, (C, 1))` is already a
spatial filter over electrodes.** Stacking `spatial_encoding: spherical_harmonics` in front of it double-counts the
geometry, so the constructor logs a warning when both are on. Treat that combination as an explicit ablation, never
a default.

## Lever: token-level lexical alignment

**The measurement that forced it.** On real ZuCo, cross-subject *word* retrieval sits at Top-1 0.0040 against a
query-weighted chance of 0.0031, and the held-out `word_len` probe is negative in 13 of 13 runs. Sentence-level
transfer is real — rank percentile 0.9670, 8 hits in 700 where chance expects 1 — but it is carried by
whole-sentence gestalt, and 5.14 of the 9.45 bits of sentence identity on this corpus are sentence length.

**Why it was never going to emerge on its own.** The CLIP objective pulls at the *pooled* sentence vector. No term
anywhere in the loss asks a single word's EEG to mean that word, so there is no gradient pushing lexical structure
into the token representations. Hoping for it was the mistake; the fix is to demand it.

Two directions, weighted separately because they are different claims. With $v_i$ the L2-normalised projection of
usable EEG token $i$, $t_i$ its word's frozen text embedding, $c_i$ its `(stimulus, word index)` and $s_i$ its
subject:

$$
\mathcal{L}_\text{type} = -\frac{1}{|A|}\sum_{i \in A} \log
  \frac{\exp(v_i \cdot t_i / \tau)}{\sum_{c \in T} \exp(v_i \cdot t_c / \tau)}
$$

over the distinct word types $T$ present in the batch — absolute lexical identity, learnable from one reader and
not required to transfer. And:

$$
\mathcal{L}_\text{reader} = -\frac{1}{|A|}\sum_{i \in A} \log
  \frac{\sum_{j:\, c_j = c_i,\ s_j \neq s_i} \exp(v_i \cdot v_j / \tau)}
       {\sum_{j \in \mathcal{C}(i)} \exp(v_i \cdot v_j / \tau)}
$$

— the same word position of the same sentence read by **someone else**. This is the property a cross-subject
decoder needs. Two constraints make it mean what it says: a different reading of the same word is never a negative
(it is a positive, whoever produced it), and with `lexical_same_subject_negatives` the denominator holds only the
anchor's own subject, so telling anchor from negative cannot be done on subject identity — the shortcut that makes
an easy contrastive loss worthless here.

**Where the target comes from.** `data.targets.lexical.build_lexical_matrix` embeds each word type with the *same*
frozen encoder that supplies the sentence-level CLIP target, so a word and the sentence containing it land in one
space. That is not a convenience: it is what lets the decoder's word-synchronous evidence path read per-word
vectors through the map it already learned for the pooled vector. Edge punctuation is stripped (ZuCo keeps the
punctuation the reader saw, so `colonel.` and `colonel` would otherwise be two words) and case is kept, because
the encoders are case-sensitive and a sentence-initial capital is something the reader saw.

Both weights default to 0. `experiments/flagship/zte_lexical_raw.yaml` turns them on;
`experiments/ablation/exp14_lexical_off.yaml` is the matched pair and
`experiments/ablation/exp14_lexical_reader_off.yaml` isolates the cross-reader half. If word retrieval does not
move between them, the honest reading is that ZuCo's per-word EEG carries no cross-reader lexical content — a
result, and one worth reporting plainly.

## Lever: sub-word alignment (the token level)

The lever above aligns whole words. This one aligns the pieces a word is spelled from: each word's 350-sample
fixation window is cut into **four fixed intra-word slices**, and each slice is scored against the frozen sub-word
embedding of the piece it reads, plus against the same piece of the same word read by *someone else*. Same two
directions as the word level — absolute identity, and the cross-reader property a decoder actually needs — one rung
finer. Both weights default to 0, and the level needs a raw-window frontend, because there is no sub-word structure
to slice out of a per-word band-power vector.

The three granularities are **exclusive, not cumulative**: `objective.token_*` and `objective.lexical_*` are never
on together, so sentence → word and sentence → token each flip exactly one lever. The arms live in
`experiments/alignment/{sentence,word,token}/`, each byte-identical to `experiments/ablation/exp16_residual_off.yaml`
except the weights named — which means the word arm *is* that config, the best-measured encoder on the board, under
a new name.

**The confound that shapes the design.** The natural way to build this lever is to give a word as many EEG
sub-tokens as the reference spells it word-pieces. That hands the model the sentence's piece profile, and on a
700-sentence gallery the profile is a brain-free key: measured with the real `Qwen/Qwen2.5-0.5B` tokeniser on a
corpus matched to ZuCo's statistics, the ordered per-word profile carries 9.44 of the gallery's 9.4512 bits and
retrieves 697/700 on its own — 673/700 even after ZuCo's 33% word omission. The *total* piece count, one integer per
sentence, retrieves 62/700, against this programme's best measured 26/700, itself stale pending re-measurement. So `objective.token_sub_tokens` is a
fixed 4 for every word whatever its text says: the piece count enters the loss's target mask and nothing the
encoder computes, and every token-level headline is gated on `zte-rebaseline --piece-oracle`, whose refusal is in
the constructor of `zte.alignment.compare.LevelRetrieval` rather than in a convention. This is to a token-level
number what the length oracle is to a sentence-level one, and it is the larger of the two.

The mathematics, the tensor shapes, the word-to-sub-word map and the campaign that measures the three levels are in
`docs/ALIGNMENT_LEVELS.md`. No token-level number exists yet: the code and its gate are built, the campaign has not
run.

---

## The exp16 encoder — four mechanisms, four measured failures

Everything above is a lever on one architecture. The four mechanisms here are a change to the architecture, and each
exists because a specific number said the old one could not get there. The evidence, from real ZuCo with `ZAB` held
out (2026-08-13) and the thirteen-arm sweep of 2026-07-25:

| measurement                                                        | value                                       |
| ------------------------------------------------------------------ | ------------------------------------------- |
| Held-out rank percentile                                           | 0.9636 [0.9599, 0.9674]                     |
| Length-stratified rank percentile                                  | 0.9211 [0.9154, 0.9270]                     |
| Variance budget                                                    | 8.4% subject · 0.0% content · 91.6% neither |
| Same word, different subject                                       | cosine gap +0.005 — *not clustered*         |
| Held-out `word_len` probe                                          | $R^2 = -0.060$                              |
| Decoder rescoring, length-stratified rank percentile               | 0.4349 — *below chance*                     |
| Spread across 13 arms flipping alignment / adapter / orthogonality | 2 to 9 hits in 700                          |

The last row is the one that matters for design: run-to-run noise was the size of every effect, so the exposed levers
are exhausted. Every mechanism below defaults to off and has a matched ablation that flips exactly one field.

**Measured on real ZuCo (2026-08-15, `ZAB` held out): two of the four are falsified.** `exp16_residual_off` scores
held-out Top-1 0.0371 against 0.010 with the residual on (effective-rank ratio 0.289 vs 0.078), and `gallery_off`
scores 0.030 — both mechanisms *hurt* the honest metric while buying the pooled one. The training telemetry shows
the failure mechanism directly: the expectation head ends training predicting 99.2% of the token hiddens' variance
(`residual_context_explained` 0.9918), so the per-sentence-constant code that pooling extracts is subtracted before
it can be scored, while the token-level VICReg terms sit satisfied (0.984) on a tensor no retrieval reads. Consensus
is the one mechanism whose removal hurts (0.0057), and the length projection is measurement-neutral. The repair
levers are §14 and the sentence-level `within_task_negatives` (§3), combined in the `exp17_*` family.

## 9. Predictive Residual Coding

**Idea.** Do not read word $w$ from $\mathrm{EEG}(w)$. Read it from what the preceding words failed to predict.

**Why.** Reading is predictive, and the largest language-related EEG deflection is a *surprisal* response — the N400
scales with how unexpected a word was given its context. Everything unsurprising about a moment of reading is
predictable from the last few seconds: the reader's tonic state, cap impedance, the 1/f background, slow drift.
Those are exactly the terms $s_{\text{subject}}$, $s_{\text{state}}$ and $\varepsilon$ of §0, and they cancel in a
context residual while the word-specific response does not. This is a different attack on the same problem the
adversary of §1 attacks, and it does not require deleting anything.

**Math.** With token hiddens $h_1, \dots, h_L$ and a causal head $f$ reading only the left context (position 1 sees a
learned BOS):

$$
\hat{h}_t \;=\; f\big(h_1, \dots, h_{t-1}\big), \qquad
\tilde{h}_t \;=\; h_t - \gamma \cdot \mathrm{sg}\big[\hat{h}_t\big]
$$

with $\gamma$ a learnable scalar and $\mathrm{sg}[\cdot]$ the stop-gradient. The head is trained by its own
regression against a detached target,

$$
\mathcal{L}_{\text{predict}} = \frac{1}{\lvert V \rvert d}\sum_{t \in V} \big\lVert f(\mathrm{sg}[h_{<t}]) - \mathrm{sg}[h_t] \big\rVert^2
$$

so no gradient from $\mathcal{L}_{\text{predict}}$ reaches the encoder. **That detachment is the whole guarantee.**
Attached, the encoder could cut this loss by making itself predictable, and a constant representation drives it to
zero — collapse wearing the disguise of a well-fit de-trender.

**What it predicts, and how to falsify it.** `residual_context_explained` reports the fraction of a token that the
context accounted for. If the mechanism works, the subject probe on the residual falls *below* the subject probe on
the token while the content probe rises. If both fall, the residual is noise and the mechanism failed. That is a
clean, pre-registered prediction with a matched pair to test it against.

**Config:** `model.residual_coding`, `residual_layers`, `residual_gate`, `residual_predict_weight`.
**Code:** `models/encoder/residual.py`; applied inside `ZTEModel.token_hidden`, its loss drained by the `Trainer`.
**Ablation:** `experiments/ablation/exp16_residual_off.yaml`.

## 10. Cross-Reader Consensus Distillation

**Idea.** ZuCo gives all twelve subjects the same 700 sentences, so every stimulus has twelve noisy measurements of
one latent content vector. Train each single reading against the *consensus* of the others rather than against the
text alone.

**Why.** §0 says a reading is content plus reader plus trial noise. The variance budget says the third term dominates:
91.6% of the space's variance has no measurable attribute at all. Averaging $n$ readings suppresses the reader and
trial terms as $1/\sqrt{n}$ and leaves the content term untouched, so the cross-reader mean is a strictly better
content estimate than any row the encoder can see. This is not augmentation-based self-distillation — the teacher
averages over *different brains reading the same text*, which is precisely the invariance we need and which no
augmentation of a single reading can produce.

**Math.** An EMA prototype bank holds one vector per stimulus key $k$, updated only while training and read before it
is written:

$$
p_k \;\leftarrow\; \rho\, p_k + (1 - \rho)\,\overline{z}_k^{\,\text{batch}}, \qquad
\mathcal{L}_{\text{pull}} \;=\; 1 - \cos\!\big(z_i,\; p_{k(i)}\big)
$$

served only once at least `consensus_min_readers` **distinct subjects** have contributed to $p_k$. Beside it, the
term that matters most:

$$
\mathcal{L}_{\text{gallery}} \;=\; -\log
  \frac{\exp\big(z_i \cdot p_{k(i)} / \tau\big)}{\sum_{k \in \mathcal{K}} \exp\big(z_i \cdot p_k / \tau\big)}
$$

over every prototype the bank serves. **That is the evaluation, moved into the loss.** The evaluation asks a
held-out reading to pick its sentence out of 700; this asks a training reading to pick its stimulus out of every
stimulus — and it scores EEG against EEG, so the modality gap cannot be what separates the answer from the
distractors.

**Honesty.** The bank is written only in `training` mode and never consulted at inference, so a held-out subject
neither enters it nor reads from it, and the exported embedding is unchanged by its existence. Read-then-write
ordering means this step's teacher was built from earlier steps only. One approximation, flagged: the anchor's own
earlier passes sit in its prototype with weight bounded by $1 - \rho$. That can weaken the teacher; it cannot
manufacture a held-out result.

**Config:** `objective.consensus_weight`, `consensus_gallery_weight`, `consensus_word_weight`, `consensus_decay`,
`consensus_min_readers`, `consensus_temperature`, `consensus_gallery_size`.
**Code:** `models/encoder/consensus.py`; wired through `_ObjectiveBase.attach_consensus` / `sentence_consensus`.
**Ablation:** `experiments/ablation/exp16_consensus_off.yaml`.

## 11. Length-Matched Gallery Contrast

**Idea.** Put all 700 texts in the InfoNCE denominator instead of the batch's fifteen, and restrict that denominator
to texts of the anchor's own word count.

**Why, part one.** Training asks the model to beat fifteen distractors; evaluation asks it to beat 699. The hardest
distractors — same length, same passage, same register — are almost never in a batch of sixteen. The frozen text
matrix is already resident, so widening the denominator costs one matrix product.

**Why, part two, and this is the real reason.** Word count carries 5.1422 of the 9.4512 bits of sentence identity,
and eye-tracking segmentation hands the model that count for free on every reading. A denominator of same-length
texts makes counting words worth **exactly nothing**, because every candidate has the same count. Whatever the loss
learns to separate them with is therefore not length. This is the training-time counterpart of the length-stratified
evaluation that has, until now, only been able to *measure* the confound after the fact.

**Math.** With $\mathcal{B}(i) = \lbrace c : \lvert n_c - n_i \rvert \le b \rbrace$ the anchor's length band,

$$
\mathcal{L}_{\text{gallery}} \;=\; -\log
  \frac{\exp\big(z_i \cdot t_{c(i)} / \tau\big)}{\sum_{c \in \mathcal{B}(i)} \exp\big(z_i \cdot t_c / \tau\big)} .
$$

Two guards make it a loss rather than a formality. An anchor whose band holds fewer than `gallery_min_candidates`
texts falls back to the full gallery — a band that strands an outlier-length sentence with two distractors makes its
loss *small*, not hard, which is the opposite of the point. And the anchor's own text is always in its own
denominator, because a cross-entropy whose target column is masked saturates at the float floor and stops reading
the model. `gallery_chance` reports the chance level the denominator actually implies, so the number is never quoted
against $1/700$ when the band made it $1/40$.

**Config:** `objective.gallery_weight`, `gallery_length_band`, `gallery_min_candidates`.
**Code:** `models/encoder/gallery.py`; used by `SentenceClipObjective.compute`.
**Ablations:** `experiments/ablation/exp16_gallery_off.yaml` (the term entirely) and
`exp16_gallery_band_off.yaml` (the full gallery without length matching, isolating the two halves).

## 12. Length Projection

**Idea.** Remove the sentence-length subspace from the exported embeddings, fitted on the training split, and then
measure what is left.

**Why.** Length-stratified retrieval answers "would this hit survive if length were held constant". It is a good
question and it has been the project's main defence. This asks the stronger one: make the representation itself
carry no length, and report retrieval on that. A length-only oracle at $\pm 2$ words currently beats the best
encoder on every top-k, so the difference between the two numbers is the entire claim.

**Math.** With $\phi(n) = [1,\; n,\; \log n,\; n^{-1},\; n^2]$ and $W$ the ridge solution of
$\min_W \lVert Z_{\text{train}} - \Phi_{\text{train}} W \rVert^2 + \lambda \lVert W \rVert^2$ fitted on training rows only:

$$
\tilde{z}_i \;=\; z_i - \big(\phi(n_i) - \overline{\phi}\big)\,W .
$$

The basis is small enough to fit on a few hundred sentences and rich enough to catch the saturating part — length
enters retrieval through more than one route (more tokens to pool, a longer eye-tracking trace, a wider pad mask), so
a straight line in $n$ does not span it.

**Provenance is the whole point.** Fitting on the rows about to be scored is transductive and drives the residual
leakage to essentially zero — a number that says nothing about the encoder and that a decoder scoring one sentence at
a time cannot reproduce. Fitted on train, `length_leakage_after` is *not* zero, and the residual is what the basis
failed to transfer. Both numbers travel with the metric, and the projection is refused with a stated reason rather
than silently skipped when the word counts needed to fit it are absent.

**Config:** `objective.length_projection` (an evaluation post-processing knob, alongside `whiten` and `all_but_top`).
**Code:** `models/encoder/nuisance.py`; applied in `evaluation/report.py` before any retrieval is computed.
**Ablation:** `experiments/ablation/exp16_length_projection_off.yaml` — it changes no gradient, only what the
evaluation reads, so the pair measures how much of the headline was word count.

## 13. What §10 and §11 do to the retrieval task itself

Both mechanisms range over the *stimulus set*, and the retrieval gallery **is** the stimulus set. That interaction
has to be stated before either number is quoted.

**There is no transductive leak.** The consensus bank is written only under `self.training`, so no held-out
reading enters it, and it is never consulted at inference. The gallery denominator is the frozen *text* matrix; no
held-out EEG touches it. Nothing fitted on a scored row reaches the score.

**But a subject-only split holds out people, not sentences.** Under `by_subject_loso` every one of the 700 gallery
sentences was in training. That was already true of the sentence-level CLIP target, and of every arm on the board.
What §10 and §11 change is the *sharpness*: an in-batch InfoNCE separated fifteen texts at a time and the model
never saw the 700-way problem, whereas a full-gallery denominator and a per-stimulus prototype bank make separating
these exact 700 items **the training objective**. The task quietly becomes closed-set identification over a known
sentence set for an unseen reader, rather than open-set retrieval of an unseen sentence.

Neither reading is wrong; they are different claims, and the narrower one is still clinically meaningful — a
communication board is a fixed phrase set. What is not acceptable is quoting the closed-set number beside an
open-set one. So `metrics['gallery_exposure']` records the split, which gallery terms were active and whether the
combination is closed-set, and `report.md` prints a **closed-set caveat** whenever it is. An arm carrying that
caveat is comparable only with other arms carrying it.

**The open-set claim needs `by_subject_and_stimulus`,** whose `test` cell is unseen subject × unseen text. There the
restriction below makes both mechanisms clean by construction.

**One consequence was a real bug.** `text_vocab` is deliberately whole-dataset, so an id means the same sentence in
every split — which meant the full-gallery denominator contained rows for sentences the split had held out. Training
against a held-out text as a *negative* still teaches the encoder where not to map, and that shapes the evaluation
geometry. `GalleryContrast.restrict_to` now masks the denominator down to the text ids the training split actually
reads, and the band and the sparse-anchor widening both stay inside that admissible set. Measured on a
stimulus-holding-out split, the denominator drops from every text to only the training texts and the held-out
stimuli appear as neither positive nor negative.

The consensus bank needs no such restriction and never did: it is sized whole-dataset but written only from training
rows, so a held-out stimulus has zero readers, sits below `consensus_min_readers`, and never enters `ready_keys`.
That is asserted directly rather than assumed.
