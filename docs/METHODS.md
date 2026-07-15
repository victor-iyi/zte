# MOSAIC — Meaning-Oriented, Subject-Aligned Invariant Code

The methods behind ZTE's second-generation thought embedding. **MOSAIC** turns the EEG recorded while a person reads a word into a 768-dimensional code that depends on *what* was read rather than *who* read it, by treating the signal as a superposition of generative factors, disentangling them, and aligning them across brains.

Configuration: `experiments/sota_loso.yaml`. Every method is an independent, config-gated lever, so each is validated in isolation with `zte-ablate` against the held-out-LOSO scoreboard (`zte.evaluation.scoreboard`).

---

## 0. The generative thesis — the neural superposition

A word-evoked EEG response $x \in \mathbb{R}^{8 \times 105}$ (8 bands $\times$ 105 electrodes) is not *meaning + noise*. It is a sum of simultaneous, physically distinct sources:

$$
x \;=\; s_{\text{content}}(\text{word}) \;+\; s_{\text{subject}}(\text{anatomy}) \;+\; s_{\text{task}}(\text{attention}) \;+\; s_{\text{behaviour}}(\text{oculomotor}) \;+\; s_{\text{state}}(\text{arousal}) \;+\; \varepsilon
$$

- **content** — the N400 ($\sim$300–500 ms centro-parietal negativity; Kutas & Hillyard 1980), whose amplitude tracks lexical surprisal.
- **subject** — the EEG *forward model*: individual skull conductivity and cortical folding project cortical sources onto the scalp differently for each person. This is real biophysics, not noise, and it is the largest source of variance in reading EEG (a linear probe reads identity at $\sim$99%).
- **task / state / behaviour** — top-down attentional set, arousal, and the oculomotor dynamics of fixations and regressions.

The v1 failure — the space encoding identity, not meaning — follows directly: identity is the loudest term, and an unconstrained objective grabs the loudest correlate of the label. MOSAIC's premise is that these terms should be **separated and modelled**, not deleted, because several of them (frequency drives both the N400 and fixation duration) are *correlated with meaning*, so blind deletion removes content with the nuisance.

---

## 1. Subspace Factorization (disentanglement)

**Idea.** Split the embedding into a content subspace and a nuisance subspace, $z = [\,z_c \mid z_n\,]$ with $z_c \in \mathbb{R}^{d_c}$ and $z_n \in \mathbb{R}^{d - d_c}$, and route each objective to the subspace it belongs to: the meaning target and retrieval read only $z_c$; the subject adversary acts on $z_c$ to push identity *out* of content (its gradient-reversed classifier makes $z_c$ un-decodable for subject). Identity then has room to live in $z_n$ instead of contaminating $z_c$.

**Why.** An adversary alone only pushes a nuisance down; it never pulls content up, and the freed variance is absorbed by the next-loudest correlate (empirically, task rose to 85% when identity was driven to 0%). Giving each factor its own subspace is the standard fix from speaker/content disentanglement in speech.

**Math.** $z_c = P_c\,z$, with $P_c$ the projection onto the first $d_c$ coordinates. The subject adversary $g$ is trained by gradient reversal to *maximise* the conditional entropy of subject given the content slice, $\max_{\theta}\, H\big(\text{subject} \mid z_c\big)$, i.e. to make $z_c$ un-informative about identity. **Config:** `model.factored`, `model.content_dim`. **Code:** `models/objectives.py` (`_content_slice`, adversary routing), `evaluation/scoreboard.py` (judges the content slice).

## 2. Lexical Meaning Distillation

**Idea.** Distil $z_c$ toward a frozen language-model vector $v(\text{word})$, so the objective has an explicit meaning target skip-gram never had.

**Why.** LM surprisal robustly predicts N400 amplitude (Frank et al. 2015; Heilbron et al. 2022, *Nat. Commun.*), so aligning the content subspace to a static LM embedding grounds the target in the best-established EEG$\leftrightarrow$meaning relationship. Random contrastive negatives are separable by identity/task, so they never force the token in; an explicit target does.

**Math.** cosine distillation

$$
\mathcal{L}_{\text{meaning}} \;=\; 1 - \cos\!\big(W z_c,\; v(\text{word})\big),
$$

with $W$ a learned projection sized to the teacher width. **Config:** `objective.meaning_distill_weight`, `meaning_source` (GloVe/fastText/LM file; hash fallback for mechanism tests), `meaning_dim`. **Code:** `data/meaning.py`, `objectives.py`. Provision real vectors with `scripts/build_meaning_vectors.py`.

## 3. Confound-Matched Contrastive Negatives

**Idea.** In the skip-gram InfoNCE, restrict negatives to tokens that share the anchor's confound (same subject and task); positives stay eligible regardless.

**Why.** The confound audit (`zte-audit`) shows task is fully confounded with the stimulus (Cramér's $V \approx 0.99$). If negatives differ from the anchor in subject/task, the softmax can be won by reading identity/task off them; forcing negatives to *match* on those axes leaves the word as the only discriminative signal.

**Math.** for anchor $i$ and candidate $j$, the negative is admissible iff

$$
M_{ij} \;=\; \big(\text{subject}_i = \text{subject}_j \;\wedge\; \text{task}_i = \text{task}_j\big) \;\vee\; \text{pos}_{ij}.
$$

**Config:** `objective.hard_negatives`, `hard_negative_keys`. **Code:** `objectives.py::SkipGramObjective`.

## 4. Riemannian Subject Alignment

**Idea.** Whiten each subject's feature covariance to a shared reference before encoding, so the second-order structure the forward model imposes is removed. Per subject $s$ with baseline mean $\mu_s$ and covariance $\Sigma_s$, transform

$$
x \;\mapsto\; \Sigma_s^{-1/2}\,(x - \mu_s).
$$

**Why.** The LOSO "cone" is the forward model reasserting itself on a stranger. Covariance carries the subject fingerprint (this is why Riemannian / tangent-space alignment is state-of-the-art for cross-subject EEG transfer; Barachant; Zanini et al. 2018). Per-channel z-scoring only removes the mean; recentring the covariance removes the whole second-order fingerprint.

**Math.** the SPD inverse square root comes from the symmetric eigendecomposition $\Sigma = V \Lambda V^{\top}$, giving $\Sigma^{-1/2} = V \Lambda^{-1/2} V^{\top}$, with Ledoit–Wolf shrinkage for conditioning:

$$
\Sigma \;\leftarrow\; (1-\gamma)\,\Sigma \;+\; \gamma\,\frac{\operatorname{tr}\Sigma}{d}\,I .
$$

**Config:** `dataset.normalize='riemannian'`. **Code:** `data/transforms.py`.

## 5. Oculomotor Privileged Supervision

**Idea.** Train an auxiliary head to predict per-word reading difficulty (total reading time, regression time $\text{GPT} - \text{GD}$, skipping) from $z_c$.

**Why.** Eye-movement control in reading is lexically driven — readers fixate longer on low-frequency, less-predictable words (E-Z Reader, Reichle et al.; SWIFT, Engbert et al.), so fixation behaviour is a *proxy for lexical processing load* that is meaning-adjacent but not identity-correlated. Predicting it (learning-using-privileged-information; Vapnik) injects a content-shaped gradient. ZuCo's simultaneous eye-tracking makes this free.

**Math.** a masked multi-target loss over difficulty signals $y_k$ (regression, or BCE for the binary *skipped* target), summing only over cells present by design:

$$
\mathcal{L}_{\text{behaviour}} \;=\; \sum_{k} \mathbb{1}[\,y_k \text{ finite}\,]\;\ell_k\!\big(\text{head}(z_c),\, y_k\big).
$$

**Config:** `objective.behaviour_weight`, `behaviour_targets`. **Code:** `data/behaviour.py`, `objectives.py`.

## 6. Band-Family Routing

**Idea.** Encode theta/gamma ($t_1,t_2,g_1,g_2$) and alpha/beta ($a_1,a_2,b_1,b_2$) through separate pathways.

**Why.** The bands are functionally distinct: theta indexes lexical retrieval / working-memory load and gamma indexes semantic unification (Bastiaansen; Hagoort), whereas alpha/beta carry attention and arousal *state*. Routing them separately lets invariance pressure fall asymmetrically — protect the theta–gamma content path, scrub the alpha–beta state path — instead of mixing them in a flat 840-vector. (Mutually exclusive with spatial encoding in the current frontend; use one.) **Config:** `model.band_routing`. **Code:** `models/frontends.py::BandRoutedMLP`.

## 7. Electrode Spatial Encoding

**Idea.** Add each electrode's scalp position to its token via the real spherical-harmonic basis $Y_l^m(\theta, \phi)$ — the Laplace–Beltrami eigenfunctions of the sphere, the correct generalisation of sinusoidal position encoding from a line to the scalp surface.

**Why.** Reading engages a left-lateralized network (occipitotemporal VWFA, posterior temporal, inferior frontal); a geometry-aware channel encoding lets the model exploit it. Empirically the single most effective identity-reducing lever found so far. **Config:** `model.spatial_encoding`, `dataset.montage_csv` (exact when a real montage is supplied). **Code:** `models/spatial.py`; details in `docs/SPATIAL_ENCODING.md`.

## 8. Zero-Shot Subject Calibration (encoding a new brain)

**Idea.** The encoder takes no subject-ID; identity enters only at the normaliser. So a genuinely new person needs only a short *unlabelled* baseline to compute their own normalisation — per-subject mean/std, or the Riemannian whitening map $\Sigma_{\text{new}}^{-1/2}$ — then their words embed on the shared frame with no labels and no retraining.

**Why.** This operationalises the report's cheapest new-subject path: adapt the *one* place identity enters, not the encoder. Riemannian calibration is the principled version — it recentres the new brain's covariance $\Sigma_{\text{new}}$ to the training reference. **Code:** `FeatureNormalizer.calibrate_subject`, `ZTEEmbedder.calibrate_subject` + `embed_signals(subject_codes=...)`.

---

## Evaluation & proof

- **Scoreboard** (`zte.evaluation.scoreboard`) — held-out-only geometry, cross-subject held-out retrieval (query = the stranger, gallery = trained-on people), every probe stated as $\text{ZTE} - \text{raw}$, and a content-probe positive control. Rendered at the top of every `report.md`.
- **Confound audit** (`zte-audit`) — the factor-entanglement table that motivates §3.
- **Ablation** (`zte-ablate`) — one-knob sweeps + scoreboard diff, so each method's held-out LOSO contribution is attributable. A method is kept only if it moves the north-star.

The north-star metric is **held-out-LOSO cross-subject retrieval**, not in-sample scores. A method that improves a home-game number but not the away game has not earned its place.
