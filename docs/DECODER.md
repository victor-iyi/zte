# The decoder — a frozen LM on a metered leash, steered word by word

How ZTE turns EEG into English, and — more importantly — how it keeps itself honest while doing it. The short version: **the language model is frozen, the encoder is frozen, the conditioning channel has an architectural bit ceiling, and the only things that learn are a small bridge, a codebook and a rank-64 map.** Everything else in this document exists to stop those, or their reader, from claiming more than the signal supports.

Configs: [`experiments/flagship/decode_zte_v2.yaml`](../experiments/flagship/decode_zte_v2.yaml) and [`experiments/decoder/`](../experiments/decoder/). Code: `zte.models.decoder` (`PrefixBridge`, `SemanticRateLadder`, `WordEvidence`, `MonotonicPointer`, `GapCorrector`, `FrozenLM`), `zte.models.objectives.decode.PrefixDecodeObjective`, `zte.models.objectives.lexical.LexicalAligner`, `zte.inference.decode.ZTEDecoder`, `zte.evaluation.generation`, `zte.evaluation.audit.rebaseline`.

CLIs: `zte-run --mode decoder`, `zte-decode`, `zte-rebaseline`, `zte-analyze`.

**Every mechanism added in v2 defaults to off.** `experiments/decoder/decode_v2_pooled.yaml` reproduces the pooled-prefix decoder exactly, so the difference between it and the flagship is attributable to the mechanisms rather than to the rebuild.

## Read this first: sentence length is worth more bits than the encoder

Measured on the real 700-stimulus SR+NR gallery (8,400 readings, 700 unique texts; length mean 19.60 words, sd 9.79, min 3, max 65):

$$
H(\text{identity}) = 9.4512\ \text{bits}, \qquad H(\text{identity} \mid n_{\text{words}}) = 4.3090\ \text{bits}
$$

so **sentence length alone carries 5.1422 bits of sentence identity**. Length is not a subtle leak: ZuCo's word segmentation comes from eye tracking, so the width of `pad_mask` *is* the word count, and the model receives it for free on every reading.

A length-only oracle — one that knows nothing but the word count and guesses uniformly inside the matching stratum — scores this against the same 700-sentence gallery:

| Oracle                                  | Top-1      | Top-5      | Top-10     | MRR        | rank percentile |
| --------------------------------------- | ---------- | ---------- | ---------- | ---------- | --------------- |
| exact length                            | 0.0729     | 0.3243     | 0.5129     | 0.2054     | 0.9847          |
| length ±1                               | 0.0286     | 0.1257     | 0.2243     | 0.0966     | 0.9494          |
| length ±2                               | 0.0214     | 0.0786     | 0.1371     | 0.0672     | 0.9177          |
| length ±4                               | 0.0100     | 0.0514     | 0.0986     | 0.0470     | 0.8670          |
| **encoder (raw conformer, 2026-07-24)** | **0.0143** | **0.0457** | **0.0886** | **0.0427** | **0.9617**      |

The encoder's entire Top-k/MRR profile is matched or beaten by knowing sentence length to ±2 to ±4 words. The one statistic that resists is `rank_percentile`: 0.9617 against 0.9494 for the ±1 oracle, exceeded only by the exact-length oracle. So there is plausibly a residual non-length signal — but **the headline Top-k is not evidence of it**, and no decoder number computed on an unstratified gallery is either.

This is why every retrieval readout below is reported twice, once on the full 700 gallery and once inside the `|Δn_words| ≤ length_tol` stratum (mean 67.4 candidates, mean per-query chance 0.0285), and why the `mismatch` control is a **length-stratified** derangement rather than a plain shuffle.

`zte-rebaseline` is the tool that measures all of this against any existing checkpoint, with no retraining. It is a first-class diagnostic and it blocks nothing: it tells you how much of a number is length, and the answer travels with the result.

### The bit budget

`rank_percentile` 0.9617 puts the true sentence at mean rank ≈ 27 of 700, i.e. at most $\log_2(700/27) = 4.7$ bits of sentence identity — and that was measured *after* transductive whitening, which a decoder cannot use (see below). A 19.6-word English sentence needs roughly 190 bits. **The encoder supplies about 2.5% of what free generation requires, and 5.14 of those bits are recoverable from word count alone.** `rebaseline.bit_budget` reports `{bits_needed, bits_from_length, bits_from_eeg, ratio}` so this arithmetic is in the artifact rather than in a reader's head.

### The published baseline is transductively contaminated

Handed no training rows, `report._postprocess` fits `whiten_features` and `all_but_the_top(1)` on the union of all 12 subjects' sentence embeddings — including the held-out one — and that array is what reaches `cross_subject_holdout_retrieval`. The transforms are label-free, so this is a soft leak rather than label leakage, but a decoder sees **one sentence at a time** and cannot reproduce a statistic fitted over the evaluation set. Handing `report.evaluate_representation` a `train_sent_emb` swaps in the train-fitted transform instead, and `metrics['postprocess_fit']` names which of `none` / `train split` / `transductive` produced the numbers, so the answer travels in the artifact rather than in a reader's assumption. Both `zte-run` and `zte-evaluate` supply those rows from `cli.evaluate.train_split_sent_emb`. `zte-rebaseline` reports all three conditions side by side, and the train-fitted column is the one the decoder actually inherits.

## Architecture

### 1. The conditioning vector

The bridge does not read ZTE's own 768-d space. It reads the **text-aligned** vector: the CLIP projection that the source encoder run already fitted against frozen sentence embeddings. That choice is what makes Stage 0 possible — the bridge's input space is a *text* space, so it can be pretrained without any EEG at all.

```plaintext
h_tok  = model.token_hidden(batch)                          (B, L, 256)
h_ctx  = model.contextualize(h_tok, valid)                  (B, L, 256)
h_sent = model._pool_tokens(h_ctx, valid)                   (B, 256)      = model.sentence_hidden(batch)
z_raw  = normalize(clip_head(model.project(h_sent)))        (B, 768)      text space, L2-normalised
z      = gap(z_raw)                                         (B, 768)      train-fitted modality correction
P      = bridge(z)                                          (B, 8, 896)   the soft prompt
```

`valid = pad_mask & presence`, with a fallback to `pad_mask` for a row where nothing is present (the attention pooler returns NaN on a fully masked row; the transformer guards, the pooler does not). That recipe lives in one place, `ZTEModel.pooling_mask`, and `ZTEModel.sentence_hidden` is the differentiable path — `embed_sentence` keeps its `@torch.no_grad()` decorator and is unusable inside a loss.

`clip_head` is restored from the source checkpoint's `extra['objective_state']` and frozen in every mode — including stage B of a joint run, where the encoder underneath it moves — so the target space stays exactly where the encoder run put it and the gap correction fitted against it stays valid. If the source run carries no `clip_head`, one is built and trained here — and the embedding cache is disabled while it moves, because a cached vector from a moving projection is stale.

### 2. The modality gap

EEG vectors do not land where text vectors live, even after alignment: the two clouds have different means and different scales. `GapCorrector` maps one onto the other with statistics fitted on the **train split only**:

$$
z = \frac{z_\text{raw} - \mu_\text{eeg}}{\sigma_\text{eeg}} \cdot \sigma_\text{txt} + \mu_\text{txt}
$$

`gap_correction: whiten` replaces the per-dimension scaling with a full ZCA: whiten by $\Sigma_\text{eeg}^{-1/2}$, then colour by $\Sigma_\text{txt}^{+1/2}$, both computed in float64 with eps-clipped eigenvalues. `none` is the identity.  The fitted state (including `n_fit`, the number of readings it saw) rides in the checkpoint under `extra['gap_correction']`, so a transductive fit would be visible in the artifact rather than merely forbidden by a docstring.

### 3. The bridge — the pooled trainable surface

With $d_z = 768$, $d_\text{lm} = 896$ (Qwen2.5-0.5B), $k = 8$ slots and bottleneck $r = 128$:

$$
u = W_\text{in}\,\mathrm{LN}(z) \in \mathbb{R}^{r}, \qquad
u_j = u \odot (1 + \gamma_j) + \beta_j, \qquad
P_j = \mathrm{LN}\big(W_\text{out}\, u_j\big) \in \mathbb{R}^{d_\text{lm}}
$$

for $j = 1 \dots k$, with per-slot FiLM parameters $\gamma, \beta \in \mathbb{R}^{k \times r}$. Counted exactly:
LayerNorm 1,536 + $W_\text{in}$ 98,432 + FiLM 2,048 + $W_\text{out}$ 115,584 + LayerNorm 1,792 + the learned null
prefix $P_\text{null} \in \mathbb{R}^{8 \times 896}$ 7,168 = **226,560 trainable parameters**.

FiLM is initialised from a truncated normal, not zeroed: zero-initialising both $\gamma$ and $\beta$ makes all eight slots the identical vector at step 0, which is an eight-position prompt one position wide.

Against roughly 120k supervised target tokens this is already generous. Anything larger memorises the corpus, and the size is the argument: with 700 sentences and no LM weights to move, there is no mechanism by which the corpus can be stored in the weights that produce text.

`conditioning: pooled_plus_words` (the registered ablation, `experiments/decoder/decode_words_ablation.yaml`) adds `WordResampler`: 8 learned latents at 256 cross-attending `h_ctx` under `valid`, 2 blocks, then `Linear(256, 896)` and its own 8×896 learned null — 769,664 parameters — concatenated to give a 16-slot prefix. It is an ablation for two measured reasons. Cross-subject word-level content is absent on ZuCo (`word_len` R² −0.0649, `log_freq` −0.0584, negative in 13/13 runs). And a length-L memory hands the decoder the word count directly, which is 5.14 bits of the answer.

### 4. The semantic rate ladder — the bit budget as a constraint, not an argument

`decoder.rate_ladder: rvq`. The problem this solves is that a continuous 768-d conditioning vector can in principle carry any number of bits, so "how much did the brain contribute" has to be argued after the fact from retrieval ranks. A residual vector quantiser replaces the argument with a constraint. With $S$ stages of $K$ codes:

$$
r_0 = z, \qquad c_s = \operatorname{argmin}_k \lVert r_s - e_{s,k} \rVert^2, \qquad r_{s+1} = r_s - e_{s,c_s}, \qquad \hat{z} = \sum_{s<S} e_{s,c_s}
$$

so the channel carries at most $S\log_2 K$ bits — 32 at the default $4 \times 256$, against the 9.4512 that sentence identity over 700 stimuli actually needs. The ladder is deliberately **not** the binding constraint; it is the instrument. `bit_report` measures what arrived: per-stage entropy, the joint code entropy, and the plug-in mutual information with sentence identity (an upper bound, biased upward at 700 queries, and labelled as one).

The gradient reaches the encoder straight through ($\hat{z} \leftarrow z + \mathrm{sg}[\hat{z} - z]$) and the codebooks are updated by EMA rather than by the decoding loss, so a stage cannot be dragged off the text manifold by the thing it is supposed to constrain. Codes are seeded by k-means on the **frozen text cloud**, fitted stage by stage to the residual the stages above left, so the ladder is coarse-to-fine and a code names a region the LM already writes fluent English from. A code unused for `rate_revive_after` steps is re-seeded onto the batch's worst-fitting vector: a dead code silently lowers the real rate below the ceiling the report quotes, which would make the measured budget a lie.

**The reserved length stage.** With `rate_length_stage: true`, stage 0 carries a linear head trained to predict the word count, and the remaining stages pay a penalty proportional to the normalised cross-covariance between their code vectors and length. `residual_mutual_information_bits` is then the code's information about sentence identity *with the reserved stage removed* — the part of the answer the brain supplied rather than the part eye-tracking word segmentation gave away. `experiments/decoder/decode_v2_no_length_stage.yaml` is the required companion arm: if the headline's advantage vanishes without the reserved stage, the reserved stage was doing the work, and that is a finding about the confound rather than about the brain.

### 5. Word-synchronous lexical evidence — the brain at every step, not only the first

`decoder.evidence_schedule: linear`. A soft prompt spends its influence in the first few generated tokens; by the tenth the LM is mostly reading its own output. ZuCo hands over, free and for every reading including a held-out one, which stretch of EEG belongs to which word — eye tracking is what defines the word boundaries. The evidence path walks it.

At generated token $t$ a Gaussian pointer sits over word $t / \bar{c}$, where $\bar{c}$ is the mean LM tokens per word **measured from the tokenised training corpus** rather than configured (a hand-set rate desynchronises the pointer whenever the tokeniser or the corpus changes; the measured value rides in the checkpoint). The pointed-at words are pooled, mapped into the LM's hidden space through a rank-$r$ map, gated, norm-capped, and **added to the LM's final hidden state**:

$$
\pi_t(j) \propto \exp\Big(-\frac{(j - t/\bar{c})^2}{2\sigma^2}\Big), \qquad
m_t = g \sum_j \pi_t(j)\, \mathrm{LN}(W_\text{up} W_\text{down} v_j), \qquad
\ell_t = \mathrm{head}(h_t + m_t)
$$

Because the frozen output head is linear, $\ell_t = \mathrm{head}(h_t) + \mathrm{head}(m_t)$: this is exactly a rank-limited additive bias on the token logits, with no new vocabulary parameters and no second decode path. The gate $g$ is zero-initialised, so a run begins as the pooled decoder and the evidence path only enters the output to the extent the loss pays for it; `evidence_gate` is logged per epoch. The norm cap exists because an uncapped bias wins the loss by saturating the distribution on a handful of tokens, which reads as decoding and is not.

**The schedule is content-free, and that is the load-bearing property.** $\pi_t$ depends on the step count and the word mask alone — never on what the words were — so *every* brain-independent control inherits the identical walk. Word count is worth 5.14 bits here, and a schedule the controls did not get would hand the headline those bits for free. `MonotonicPointer.forward` takes no content argument at all, which makes this structural rather than a convention.

`v_j` is the per-word vector from the **encoder's** lexical projection (`objective.lexical_weight`, see `docs/METHODS.md`), restored frozen from the source checkpoint exactly as `clip_head` is. Over an encoder that never trained one, the path degrades to the pooled decoder and says so at startup. The cost is that the per-word hiddens are needed at every step, so the frozen-encoder cache is unavailable and a decoder run costs roughly what an encoder run costs plus the frozen LM.

### 6. The language model — frozen in all three modes, always

`Qwen/Qwen2.5-0.5B` (Apache-2.0, hidden 896, 24 layers, vocab 151,936), pinned by `decoder.lm_revision`. **There is no LoRA and no fine-tuning of the LM in any mode.** That single constraint is what makes "the output is corpus recall" unarguable rather than a matter of trust.

The prompt is assembled from the frozen input embedding $E$:

```plaintext
inputs_embeds  = [ E(bos) | P (k slots) | E(scaffold) | E(target) ]     (B, 1+k+S+T, 896)
labels         = -100 everywhere except the target span; padding -> -100
attention_mask = ones, zeroed at target padding
```

`decoder.prompt_template` (`'\nSentence: '`) is the scaffold. No new embedding parameters are introduced anywhere.

`FrozenLM.state_dict()` returns `{}`. This is not tidiness: the trainer writes `objective.state_dict()` into every epoch checkpoint and byte-copies it to `last.pt` and `best.pt`, so without the override a 0.5B LM would add about a gigabyte per epoch, three times over, to a Drive tree of cloud-only placeholders. The matching consequence is that a decoder resume must load its objective state non-strictly — `nn.Module`'s load recursion reports the LM's keys as missing regardless of what `state_dict` returns.

`lm_source: 'tiny'` builds a 2-layer, 32-wide `LlamaForCausalLM` (22,688 parameters) locally with a 64-token byte tokeniser and no network at all. Every decoder test and the MPS smoke config use it, so the suite needs no downloads.

**The precision follows the encoder.** `decoder.lm_dtype` defaults to `auto`, which reads the dtype off the encoder the bridge is fed by, so the two halves of the pipeline are never at different precisions — an encoder at float32 gives a frozen LM at float32. Naming `float32`, `float16` or `bfloat16` pins it instead; that is a legitimate trade, since the LM is three orders of magnitude larger than the bridge, but it puts the halves out of step and is logged as a warning naming both.

What it is never taken from is the HuggingFace checkpoint. `transformers` defaults to loading at whatever dtype the weights were exported in — bf16 for Qwen2.5 — which would make every token log-probability a property of the uploader's export choice and of the installed library version, and would change the numbers silently under an upgrade. The bridge trains in float32 regardless; the soft prompt is cast to the frozen embedding's dtype at the one point the two meet, and log-probabilities are read back in float32, so a half-precision LM halves its memory without putting a half-precision dynamic range into the optimiser. Scores produced under different `lm_dtype` values are not comparable with each other, and the *resolved* value travels in `provenance()`.

`lm_dtype` pins the **weights**, not the arithmetic. `Trainer` wraps `objective.compute` in `torch.autocast`, and `train.precision: auto` selects bf16 on capable CUDA, so a training step runs the frozen LM's matmuls in bf16 whatever `lm_dtype` says; `manifest.json` records both `precision` and `autocast_dtype`. Evaluation does not autocast — `ZTEDecoder`, and therefore every reported generation and rescoring number, runs at `lm_dtype`. The consequence to know is narrow: the teacher-forced loss a run minimises is not bit-identical to the one `zte-decode` reports for the same checkpoint. Set `train.precision: fp32` if a run needs the two to be the same number.

## The three training modes

`train.mode` selects the stage. The LM is frozen in all three; `joint` refers to the encoder and the bridge.

| Mode      | Trains                                                              | Encoder                                            | Needs `encoder_ckpt` | Cache           |
| --------- | ------------------------------------------------------------------- | -------------------------------------------------- | -------------------- | --------------- |
| `encoder` | encoder + objective, one parameter group at `train.lr`              | trained from scratch                               | no                   | n/a             |
| `decoder` | bridge (+ resampler) at `train.bridge_lr`                           | loaded, frozen, `.eval()`                          | yes                  | on              |
| `joint`   | stage A bridge; stage B + encoder at `bridge_lr × encoder_lr_scale` | loaded, frozen for `stage_a_epochs`, then unfrozen | yes                  | **must be off** |

`train.freeze_encoder` is the hard freeze — with it the encoder receives no gradient in any epoch — so `joint` mode requires `freeze_encoder: false` and raises if it is `true`. There is no combination in which one of the two settings silently overrides the other: in `joint` mode `stage_a_epochs` alone decides when the encoder starts training, and `stage_a_epochs: 0` unfreezes it at epoch 1.

`mode: encoder` is the pre-decoder pipeline unchanged — `stages.parameter_groups` returns the single `AdamW(model.params + objective.params)` group it always returned. The decoder wiring keys off the *objective*, not the mode: `run_training` builds the objective either way and attaches the LM, the targets, the gap fit and Stage 0 only when that objective is `PrefixDecodeObjective`. Every `mode: encoder` config therefore ships `objective.name: clip`, and pairing `mode: encoder` with `objective.name: decode` would load the frozen LM and run Stage 0 as usual.
`experiments/decoder/decode_encoder_only.yaml` exists to keep it that way: its history must reproduce `exp8_clip_e5_raw`'s under the same seed.

In `decoder` and `joint` mode the source checkpoint's **normaliser and aligner states are restored, not refitted**.
This is the quietest hazard in the whole feature: refitting does not crash a frozen encoder, it just hands it inputs at a scale it was never trained on, and the run underperforms with no error.

**The embedding cache.** With the encoder frozen and in eval, $z$ is a pure function of the reading, so `reading_id` keys an `(n_readings, 768)` float32 cache (8,400 × 768 = 25.8 MB) filled on the warm-up pass and read thereafter. The raw conformer — the expensive part — then never runs again. This is what makes 12 folds × 3 seeds
affordable, and it is why `cache_embeddings` must be `false` in `joint` mode, where the encoder moves.

**Staging.** `stages.apply_stage` flips `model.requires_grad_` at the epoch boundary; the parameter groups themselves
are structural and never change, because `torch.optim.Optimizer.load_state_dict` rejects a resume whose group count differs and `LambdaLR` captures `base_lrs` at construction. A frozen member costs nothing — AdamW allocates no state for a parameter whose `.grad` is `None`. The frozen LM is excluded from every group outright.

`train.early_stop_patience` is not decoration: every real run on record bottoms out its validation loss at epoch 5–6 of 40 (the winner reached 3.1868 at epoch 5 and 4.3827 by epoch 40), and a 227k-parameter bridge will bottom out sooner.

## The loss

$$
\mathcal{L} = \mathcal{L}_{\text{CE}}
+ \lambda_{\text{ground}} \mathcal{L}_{\text{ground}}
+ \lambda_{\text{clip}} \mathcal{L}_{\text{CLIP}} + \mathcal{L}_{\text{reg}}
$$

The last two terms are **stage B only**: both are switched on by the encoder's own `requires_grad`, so they are absent
from every stage-A epoch and from a `decoder` run entirely.

**$\mathcal{L}_\text{CE}$** is teacher-forced cross-entropy over the target span. Teacher forcing is legitimate for *training*; the trap it is famous for is an evaluation trap, and the only teacher-forced number computed at evaluation time is quarantined as a diagnostic that the verdict provably cannot read.

**Null-prefix dropout.** With probability `null_prefix_prob` (0.1) the whole prefix is replaced by the learned $P_\text{null}$, which trains the unconditional branch the `null_prefix` control decodes. At $p = 1.0$ the loss is exactly independent of $z$ — `tests/test_decode_modes.py::test_null_prefix_dropout_covers_every_slot_the_prefix_occupies` holds it to 1e-9 on both conditioning arms, which is what makes the control meaningful rather than approximate. *Whole* prefix is the operative word: under `pooled_plus_words` the resampler carries its own learned null for its slots, because a null spanning only the pooled half would leave the brain — and the word count — inside the branch that is supposed to be independent of both.

**$\mathcal{L}_\text{ground}$** is the term that punishes a bridge for ignoring the brain. For each item, its own reference plus $M = 3$ in-batch references from *other* sentences are scored under **its own prefix** by length-normalised log-likelihood, and the softmax cross-entropy over the $M+1$ candidates at temperature 0.1 is added:

$$
\mathcal{L}_\text{ground} = -\frac{1}{|A|}\sum_{i \in A}
  \log \frac{\exp\big(s(y_i \mid P_i)/\tau\big)}{\sum_{c \in \mathcal{C}(i)} \exp\big(s(y_c \mid P_i)/\tau\big)},
  \qquad s(y \mid P) = \frac{1}{|y|}\log p_\text{LM}(y \mid P), \quad \tau = 0.1
$$

Cross-entropy alone is content with a constant prefix, since the corpus prior explains most of the tokens. This term is not: a constant prefix scores every candidate identically and pays the full $\log(M+1)$. Rows whose prefix was replaced by the null one are excluded from the anchor set $A$ — asking the unconditional branch to prefer one particular reference is the opposite of what it is for.

**$\mathcal{L}_\text{CLIP}$** (stage B) keeps the unfrozen encoder anchored in the text space the bridge was fitted against; without it the decoding gradient can drag the encoder away and orphan the bridge. It is computed inline from the same $z$ the bridge reads, reusing `clip.py`'s direction term, so the encoder is not forward-passed twice per step.
`regularize(...)` (VICReg, ramped subject adversary, stimulus adversary) is inherited unchanged and, like the CLIP anchor, is switched on by the encoder's own `requires_grad` — so both are absent from every stage-A epoch and from a `decoder` run entirely.
Every metric the objective emits is a plain float, including `prefix_kl`, which is logged on **every** step — training and validation. Bridge collapse to a constant prefix is the most likely training failure here, and it has to be visible per epoch, not discovered at the end. The step metric partners each row with a row of a *different* stimulus rather than with its batch neighbour, so it measures the same quantity as verdict clause 5: hard-negative batching seeds a batch from one sentence and fills it with that sentence's own readings, where a healthy bridge is supposed to score near zero.

## Stage 0 — text-only bridge pretraining

Before the EEG loop, `decoder.stage0_epochs` (20) epochs train the bridge on `(text embedding → text)` pairs: the frozen sentence encoder embeds a training sentence, the bridge turns that into a prefix, and the frozen LM is asked to produce the sentence. No EEG is involved, so "text vector → English" is learned where data is not the constraint, leaving only the EEG-to-text-space residual for the ~5,775 training readings.

Leakage is prevented by construction, not by convention. `pretrain_text(train_ids, holdout_text_ids=...)` takes **both** sets and raises `ValueError` on any intersection, so a caller that forgets to filter fails loudly instead of quietly pretraining on the sentences it will later be scored on.

Stage 0 also produces the **text oracle**: the identical bridge fed the true sentence embedding at evaluation time.
It bounds what the head can achieve and localises failure — an oracle that decodes well while the EEG path does not says the bottleneck is the representation, not the decoder.

The cost of Stage 0 is that it teaches the bridge to emit fluent English from *anything* on the text manifold, so the EEG contribution is a perturbation on a strong learned prior. The `mean_prefix` control absorbs that prior by construction, and `decode_nostage0_ablation.yaml` is a required reported arm: if the delta vanishes without Stage 0, the honest reading is that Stage 0 was doing the work.

## Split policy — `by_subject_and_stimulus`

The 700 stimulus keys are partitioned under a fixed seed and intersected with the LOSO subject mask. The stimulus permutation depends on the seed alone and not on the subject mask, so all 12 folds hold out the same texts and pool cleanly.

| Cell                                             | Readings         | Generalises over | Status                      |
| ------------------------------------------------ | ---------------- | ---------------- | --------------------------- |
| `train`                                          | 11 × 525 = 5,775 | —                | —                           |
| `val` (seen subject, unseen stimulus)            | 11 × 70 = 770    | language only    | model selection             |
| `test` (unseen subject, unseen stimulus)         | 105              | **both**         | **the only headline cell**  |
| `test_seen_stim` (unseen subject, seen stimulus) | 525              | the brain only   | diagnostic, labelled as one |

`val_fraction: 0.10` and `test_fraction: 0.15` reproduce that table exactly. A `test_fraction ≤ 0` raises rather than returning an empty `test` cell, because the unseen-subject × unseen-stimulus cell is the entire point of the strategy.

Every cell is reported with its generalisation axis named and none is collapsed into another. Because $n = 105$ per fold, the pre-registered primary generation analysis pools 12 LOSO folds against one fixed stimulus partition (1,260 generations) with fold-level bootstrap.

A headline on `by_subject_loso` is forbidden by the verdict gate: that split shares all 700 texts between train and val, which is exactly the configuration in which a decoder recites the corpus and scores well.

## Evaluation

### Primary (powered) readout — decoder-rescoring retrieval

`ZTEDecoder.rescore` scores every gallery sentence by length-normalised $\log p(\text{text}_j \mid z_i)$, and `scoreboard.decoder_rescoring_retrieval` reports Top-1/5/10, `rank_percentile`, an exact binomial tail and a bootstrap
CI — directly comparable to `scoreboard.cross_subject_holdout_retrieval` (the encoder's [0.9558, 0.9639]). Reported on the full 700 gallery **and** length-stratified.

This is ~9.5 bits of forced choice at 700 queries, against a generation delta at $n = 105$. It is the readout most likely to carry a real, honest improvement — and it is **retrieval**, labelled as retrieval everywhere it appears.

`decoder.rescore_pmi` (default off) replaces the score handed downstream with a PMI score, per-token means under the query's prefix and under the learned null prefix:

$$
s(q, c) \;=\; \tfrac{1}{|c|}\log p(c \mid P_q) \;-\; \tfrac{1}{|c|}\log p(c \mid P_\text{null})
$$

Every trainable part of the decoder — the Stage-0 bridge, the train-fitted gap correction, the RVQ codebooks, the grounding loss — is fitted on train-cell reference texts only, so a train-cell gallery candidate collects a familiarity bonus the held-out truth cannot receive; the subtraction cancels any candidate-side constant, familiarity and fluency alike, and leaves only what the query's prefix added. The unconditional branch is the one `null_prefix_prob` already trains. Its gallery pass is query-independent — no word-synchronous evidence, one pass per gallery under the same `rescore_chunk` memory bound — and with the knob on, `rescoring['score'] = 'pmi'` marks the block while `rescoring['pmi_vs_raw']` carries the paired per-query comparison: rank percentiles under both scores and the delta (PMI minus raw) with a percentile-bootstrap CI, so the correction's effect is itself a measured number.

### Secondary (expected-null) readout — free-running generation

`ZTEDecoder.generate` only: BOS, feed back the model's own greedy tokens, stop at EOS or `max_new_tokens`, **no reference length, no candidate set**. Stopping is a trained behaviour, not a free one: every reference that fits inside `max_target_tokens` ends in a supervised EOS inside the loss mask, on the HuggingFace path as well as the offline `tiny` one. Without that the bridge has no gradient toward stopping and every hypothesis runs the full 96 tokens against a 19.6-word reference, which makes WER exceed 1 by construction and collapses every precision-based metric. `cfg_weight` is asserted `== 1.0` and `beams` is asserted `== 1`, so the headline decode and every control run byte-identical code; guidance and beam search are both refused precisely so that no control branch can become a second code path, and at this signal level beam search raises the language prior rather than the brain signal. The loop lives in `FrozenLM.generate_from_prefix` rather than in `transformers.generate`, because the evidence nudge has to reach the output state at every step and no generation hook exposes it. `generation.json` records `teacher_forced: false` and `decode_strategy: greedy` so the contract travels in the artifact. A decode constrained to a candidate set is retrieval and is reported as retrieval, never as generation — `NearestNeighborIndex.decode` is exactly that in disguise, and `n_candidate_sentences` is recorded on every block so the distinction is machine-checkable rather than a matter of wording.

Metrics are implemented in pure stdlib + numpy in `zte.evaluation.generation` (BLEU-1..4 with brevity penalty, ROUGE-1/2/L, WER, and content-word F1 over the distinct content words of each sentence). No metric package is a dependency, so nothing about the score depends on which BLEU implementation happened to be installed.

### The seven controls, and why each exists

All seven decode through the **identical** path — same weights, same greedy loop, same detokenisation, same `max_new_tokens`. Only $z$ changes.

| Control       | What it substitutes                                                                                 | What it rules out                                                                                                                       |
| ------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `mean_prefix` | the train-split mean $z$                                                                            | **the decisive one.** It absorbs the Stage-0 text prior; a decoder reciting ZuCo regardless of input scores exactly here                |
| `null_prefix` | $P = P_\text{null}$                                                                                 | the pure LM prior with no bridge at all                                                                                                 |
| `phase`       | phase-scrambled raw windows through the same frozen encoder                                         | spectral-envelope artefacts that survive scrambling                                                                                     |
| `noise`       | mean/variance-matched Gaussian through the identical frozen encoder                                 | anything explainable by the signal's first two moments                                                                                  |
| `shuffled_z`  | another reading's whole conditioning bundle, by **unstratified** derangement                        | the bridge and the LM working without the encoder — the "feed it shuffled EEG embeddings" control, stated directly                      |
| `length_only` | the **length-conditional** mean $z$, with the pointer schedule kept and the per-word content zeroed | **the one the ZuCo arithmetic demands.** It has the word count and nothing else, so a headline that beats it beat it on lexical content |
| `mismatch`    | another held-out reading's bundle, by **length-stratified derangement**                             | dependence on *which* brain — and it neutralises the 5.14-bit length confound                                                           |

Plus one positive control, `oracle`: the true sentence embedding through the identical bridge and LM. It bounds the achievable score. It will look good (expect BLEU-4 15–45) and it says **nothing** about EEG.

### The verdict gate

`_verdict['generation_above_controls']` is an AND over five clauses, each reported with its numbers and an explicit `False` when it fails:

1. `honest_split` — the readings come from the `test` cell of `by_subject_and_stimulus` (`report.HONEST_SPLIT`), the only cell that generalises over the subject and the stimulus at once. A `val` or `test_seen_stim` block fails it.
2. `no_candidate_set` — `n_candidate_sentences is None`, i.e. this was free generation.
3. `beats_every_control` — the paired per-sentence delta's bootstrap CI lower bound (n_boot 2000, seed 0) is above zero against **every one** of the seven controls, not the mean of them. A control that was requested but could not be decoded counts as not beaten, so losing one can never promote the verdict. A block that records no `controls_requested` cannot show which controls it pre-registered, so the absent ledger itself counts as not beaten and the clause fails.
4. `permutation_significant` — `honesty.generation_permutation_test` with the hypotheses held fixed and only the pairing permuted, $p = (1 + N_{\text{null} \ge \text{obs}})/(n_{\text{perm}}+1) < 0.05$.
5. `prefix_influences_output` — mean $\mathrm{KL}\big(p(\cdot \mid P_i) \parallel p(\cdot \mid P_j)\big) \ge$ `decoder.min_prefix_kl` (0.05 nats), where $P_j$ is **another reading's** prefix under a seeded derangement, $j \neq i$. Below that the prompt does not depend on which brain produced it and no delta is meaningful.

   It has to be another reading's prefix and not $P_\text{null}$. A bridge collapsed to one constant prompt for every reading still sits some distance from the *learned unconditional* prompt, which is a free parameter, so it can clear a floor stated against $P_\text{null}$ while ignoring the brain entirely — the exact failure this clause exists to catch. The KL also reads only the first generated token's distribution, which makes it a necessary condition and not a sufficient one; the length-stratified `mismatch` control is what closes the remaining gap.

Named floors, since free generation has no analytic chance level: the seven controls, the length-only oracle, and the **retrieval upper bound** — what the frozen encoder already achieves with cosine kNN (Top-1 10/700, `rank_percentile` 0.9617). If generation does not beat the last of those, the report says so.

### Within-task pools

No ZuCo stimulus appears under more than one task — the confound audit measures Cramér's V(task, stimulus) at 0.998 — so a model can score on the full 700-sentence gallery by telling SR sentences from NR ones, which is a property of the passage set and not a reading of the brain. `decoder.within_task_pools` re-ranks every query inside its own task, where the passage set is fixed. The pool is smaller, so its own chance level, hit counts and bootstrap interval are reported beside every number, and a lift that survives here is a lift on sentence content. `scoreboard.within_task_retrieval` computes it and `rescoring['within_task']` carries it.

### Seeds, and why a single number is not a result

Run-to-run drift on this project has been the size of the effect: an arm that scored 4 hits in 700 scored 2 on an identical re-run, and two seeds of `zte_raw_aligned` give rank percentiles of 0.9672 and 0.9670 while their Top-1 moves 9 hits to 8. Every headline is therefore reported as **mean ± sd over seeds**, and the two intervals answer different questions — the per-query bootstrap inside a run, and the across-seed interval `zte-analyze` computes. `scripts/run_zte_study.sh` sweeps `SEEDS` and `decoder.eval_seeds` re-runs the control layer at extra decode seeds, which puts an error bar on the comparison the verdict reads.

### Quarantined diagnostics

`teacher_forced_ppl_DIAGNOSTIC` is computed, stored and **provably unread** by the verdict: `strip_quarantined` removes any key matching `*_DIAGNOSTIC` or `*_RETRIEVAL` at any depth, `_verdict` re-applies it to whatever it is handed rather than trusting the caller, so a deliberately excellent teacher-forced perplexity cannot reach a clause.

The `*_RETRIEVAL` half of the suffix rule is a standing contract rather than a live key: no forced-choice number is emitted today, and any that is added is quarantined the moment it is named. The powered retrieval readout is *not* quarantined — it is `metrics['rescoring']`, kept readable on purpose and labelled retrieval everywhere it appears.

### Artifacts

`evaluation/generation.jsonl` carries, per held-out sentence, the reference, the free-running hypothesis, the same row for each control and the oracle, and per-sentence scores. `evaluation/generation.json` holds absolute scores, paired
deltas with bootstrap CIs, the permutation p, `prefix_influence_kl` and `n_candidate_sentences`.
`evaluation/interactive/generation.html` renders the side-by-side with no external dependencies — that page is the single most persuasive artifact against the teacher-forcing trap, because a reader can see the `mean_prefix` row saying
almost the same thing as the hypothesis. Provenance (git SHA, resolved device, wall time, batch size, precision, torch/transformers versions, `lm_revision`, tokenizer fingerprint, and the source checkpoint's path, epoch, step and sha256) travels in `manifest.json` and in every checkpoint's `extra['provenance']`.

## Config surface

| Key                                                                       | Meaning                                                                                           |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `train.mode`                                                              | `encoder` \| `decoder` \| `joint`. `encoder` is the pre-decoder pipeline unchanged.               |
| `train.encoder_ckpt`                                                      | Source checkpoint for `decoder`/`joint`; its shapes, normaliser and aligner are reused verbatim.  |
| `train.freeze_encoder`                                                    | Never let the loaded encoder train: frozen and in `.eval()` for every epoch. Rejected by `joint`. |
| `train.bridge_lr` / `train.encoder_lr_scale`                              | Bridge LR, and the encoder's LR as a fraction of it once it unfreezes.                            |
| `train.stage_a_epochs`                                                    | Bridge-only epochs before the encoder unfreezes; read in `joint` mode only.                       |
| `train.early_stop_patience`                                               | Epochs without improvement before stopping (`0` disables).                                        |
| `decoder.lm_source` / `lm_revision` / `lm_cache_dir`                      | The frozen LM, its pinned commit and its local snapshot. `'tiny'` is offline.                     |
| `decoder.lm_dtype`                                                        | `auto` (default, inherits the encoder's) \| `float32` \| `float16` \| `bfloat16`.                 |
| `decoder.conditioning`                                                    | `pooled` (headline) or `pooled_plus_words` (registered ablation).                                 |
| `decoder.rate_ladder`                                                     | `none` (default, continuous) \| `rvq` (the metered channel).                                      |
| `decoder.rate_stages` / `rate_codes`                                      | The ceiling: `stages x log2(codes)` bits. Defaults 4 x 256 = 32.                                  |
| `decoder.rate_commit_weight` / `rate_decay` / `rate_revive_after`         | Commitment loss, EMA decay, and how long a dead code survives before re-seeding.                  |
| `decoder.rate_length_stage` / `rate_length_weight`                        | Reserve stage 0 for word count and penalise the others for carrying it.                           |
| `decoder.evidence_schedule`                                               | `none` (default) \| `linear` \| `fixation`. The word-synchronous path.                            |
| `decoder.evidence_rank` / `evidence_width` / `evidence_max_bias`          | Rank of the text-to-LM map, pointer window in words, and the logit-bias cap.                      |
| `decoder.evidence_tokens_per_word`                                        | `0` measures the walking rate from the training corpus, which is the honest default.              |
| `decoder.evidence_gate_init`                                              | Initial gate; `0` starts the run as the pooled decoder.                                           |
| `decoder.bridge_depth`                                                    | Residual blocks in the bottleneck; `1` is the plain linear map.                                   |
| `decoder.ground_hard_length`                                              | Draw the grounding negatives from references of a similar word count.                             |
| `decoder.within_task_pools`                                               | Tasks whose candidate pool is also reported alone (`SR`, `NR`).                                   |
| `decoder.rescore_chunk`                                                   | Candidate rows per frozen-LM forward pass; bounds memory, not work.                               |
| `decoder.rescore_pmi`                                                     | Subtract each candidate's null-prefix log-likelihood, cancelling candidate-side familiarity bias. |
| `decoder.eval_seeds`                                                      | Extra decode seeds for a mean ± sd headline on the control comparison.                            |
| `decoder.prefix_slots` / `word_slots` / `bottleneck`                      | Bridge geometry: $k$, the resampler's slots, and $r$.                                             |
| `decoder.gap_correction`                                                  | `none` \| `mean_scale` \| `whiten`. Fitted on the train split only.                               |
| `decoder.null_prefix_prob`                                                | Probability of substituting the learned null prefix during training.                              |
| `decoder.cfg_weight`                                                      | Asserted `1.0`; any other value is rejected at generation time.                                   |
| `decoder.ground_weight` / `ground_negatives`                              | Weight and negative count of the in-batch grounding loss.                                         |
| `decoder.stage0_epochs`                                                   | Text-only pretraining epochs (`0` is the reported ablation).                                      |
| `decoder.min_prefix_kl`                                                   | Verdict floor in nats for prefix influence on the output distribution.                            |
| `decoder.cache_embeddings`                                                | Cache the frozen encoder's sentence vectors by `reading_id`. Must be `false` in `joint`.          |
| `decoder.generation_controls`                                             | The brain-independent controls decoded through the identical path.                                |
| `decoder.rescore_gallery` / `length_tol`                                  | The primary retrieval readout, and the word-count tolerance for strata.                           |
| `objective.eval_generation` / `eval_rescoring` / `eval_length_stratified` | Which evaluations `zte-run` performs.                                                             |

`objective.text_source`, `text_backend` and `text_query_prefix` must match the source encoder run exactly: the bridge reads a space that run's `clip_head` was fitted to, and rebuilding the target with a different text encoder silently moves it.

## The arms

| Config                                   | What it is                                                                                      |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `flagship/decode_zte_v2.yaml`            | **The headline.** Rate ladder + reserved length stage + word-synchronous evidence.              |
| `decoder/decode_v2_pooled.yaml`          | Every new knob off: the pooled-prefix decoder, and the baseline the other two must beat.        |
| `decoder/decode_v2_ladder_only.yaml`     | The rate ladder alone. How little it costs measures how few bits the continuous vector used.    |
| `decoder/decode_v2_evidence_only.yaml`   | Word-synchronous evidence alone, against the continuous conditioning vector.                    |
| `decoder/decode_v2_no_length_stage.yaml` | Required companion: the ladder with no stage reserved for word count.                           |
| `decoder/decode_v2_bandpower.yaml`       | The frontend row of the feature-ablation table: the decoder over a band-power encoder.          |
| `decode_frozen_e5raw.yaml`               | The v1 headline, kept for continuity with the runs already on Drive.                            |
| `decode_joint_e5raw.yaml`                | The same with the encoder unfrozen after 3 stage-A epochs, CLIP kept on as an anchor.           |
| `decode_encoder_only.yaml`               | The control proving `mode: encoder` is unchanged; its history must reproduce exp8's.            |
| `decode_nostage0_ablation.yaml`          | Required reported ablation: `stage0_epochs: 0`.                                                 |
| `decode_words_ablation.yaml`             | Registered ablation: `conditioning: pooled_plus_words` (16-slot prefix).                        |
| `rebaseline_e5raw.yaml`                  | The length-confound audit arm: the encoder recipe on the decoder's own split.                   |
| `decoder/decode_parallax_nr.yaml`        | Phase 3: the v2 decoder over the parallax NR encoder, PMI rescoring on, NR gallery.             |
| `decoder/decode_parallax_nr_joint.yaml`  | Phase 3: the same with `mode: joint` — the encoder unfreezes after 3 stage-A epochs.            |
| `smoke/decode_tiny_mps.yaml`             | Wiring only. `lm_source: tiny`, batch 4, 2 epochs, `run_name: smoke_mps`, always `--synthetic`. |

### Phase 3 — the parallax decoder arms

The parallax study measured a cross-task, cross-stimulus code that reaches the held-out subject at rank percentile
~0.95–0.97 (length-stratified ~0.92), so the decoder now gets its first encoder worth rescoring with.
`decode_parallax_nr.yaml` is `decode_zte_v2` with exactly one lever moved: the encoder and its task gallery —
`train.encoder_ckpt` names the parallax NR winner (`parallax_nr_loZAB_s44`; the notebook overrides per seed via
`--encoder-ckpt`), `dataset.tasks: [NR]` scores it against its own task's gallery, and `decoder.rescore_pmi: true`
subtracts each candidate's null-prefix likelihood so candidate-side familiarity cancels.
`decode_parallax_nr_joint.yaml` flips one further lever — `train.mode: joint` with `freeze_encoder: false` — so
stage B fine-tuning of the encoder becomes measurable against the frozen arm.

The pre-registered expectations, stated before the runs: the PMI-rescored **length-stratified** rank-percentile CI
sits above 0.5, where the raw conditional scorer measured 0.43; free-running generation remains the expected null
under the full verdict gate; and the joint arm must produce a `best.pt` from stage B — the stage-comparable monitor
exists so that stage A cannot lock it out. Rescoring stays labelled retrieval, never generation.

## How to run

```sh
# 0) The audit. Runs against any existing checkpoint, trains nothing, gates nothing.
uv run zte-rebaseline --ckpt res/experiments/exp8_clip_e5_raw_loZAB/checkpoints/best.pt \
    --root res/data/zuco_extracted --holdout ZAB --length-tol 1 --oracle-tol 0,1,2,4

# 1) Train the encoder the evidence path reads. Its per-word projection is what the decoder inherits.
uv run zte-run --config experiments/flagship/zte_lexical_raw.yaml --root res/data/zuco_extracted \
    --loso-holdout ZAB --resume

# 2) Train the decoder over that frozen encoder.
uv run zte-run --config experiments/flagship/decode_zte_v2.yaml --root res/data/zuco_extracted \
    --encoder-ckpt res/experiments/exp14_zte_lexical_raw_loZAB/checkpoints/best.pt --resume

# 3) Decode the held-out cell with all seven controls, the oracle, the permutation null and the bit report.
uv run zte-decode --ckpt res/experiments/exp15_decode_zte_v2/checkpoints/best.pt \
    --root res/data/zuco_extracted --split test --rescore

# Or all of it, resumably, at three seeds, with the analysis at the end:
SEEDS='42 43 44' bash scripts/run_zte_study.sh res/data/zuco_extracted

# Wiring check on synthetic data with no LM download. Nothing from it is a result.
uv run zte-run --config experiments/decoder/smoke/decode_tiny_mps.yaml --synthetic --mode encoder \
    --name smoke_mps_encoder
uv run zte-run --config experiments/decoder/smoke/decode_tiny_mps.yaml --synthetic
```

## The honest expectation, stated in advance

Written before the runs, so the result cannot be graded on a moving target.

- **Rescoring retrieval, full 700 gallery:** Top-1 0.008–0.025 (6–18 hits), `rank_percentile` 0.94–0.97.
  Length-stratified (mean 67.4 candidates, chance 0.0285): Top-1 0.03–0.06. Odds the stratified CI clears the matched length-oracle floor: **~45%**. Odds it is statistically distinguishable from the 10/700 CLIP baseline: **~25%**.
- **Free generation, one fold ($n = 105$):** BLEU-4 absolute 0.0–1.0 with the paired-delta CI containing zero.
  ROUGE-1 0.10–0.18 absolute, delta +0.00 to +0.02. Content-word F1 absolute 0.01–0.04, delta +0.005 to +0.015.
  WER 0.95–1.05. Odds of clearing the full verdict gate on one fold: **under 10%**.
- **Pooled over 12 folds ($n = 1{,}260$):** content-word F1 clearing all five controls with permutation $p < 0.05$ at roughly **35%**; BLEU-4 under **15%**. Verbatim reproduction of any unseen held-out sentence: **under 3%**.
- **The oracle** will reach BLEU-4 15–45 and ROUGE-1 0.45–0.70. That is a positive control for the head, not evidence about EEG, and it is stored apart from the hypothesis under `generation['absolute']['oracle']` for exactly that reason.

**The most likely honest headline:** on unseen subjects reading unseen sentences, a frozen 0.5B LM driven by a 227k prefix bridge produces free-running text statistically indistinguishable from its phase-scrambled, noise, shuffled, length-only, mismatched, mean-prefix and null-prefix controls on BLEU-4, with at most a small content-word-F1 margin when pooled over 12 folds; the same bridge fed the true sentence embedding reaches BLEU-4 ~30, so the bottleneck is the EEG representation rather than the decoder. Separately, the field's headline ZuCo retrieval metric is substantially reproducible from eye-tracking-derived sentence length.

**The outcome most likely to be mistaken for success:** train this under `by_subject_loso` — which shares all 700 texts between train and val — with an unfrozen LM, and it emits fluent ZuCo sentences at BLEU-4 in the 10–30 range. Every point is corpus memorisation, and the gate is built to say so: `mean_prefix` scores just as high, the paired delta is zero to numerical precision, and the permutation null gives $p \approx 1.0$. A decoder that ignores its conditioning entirely and emits the corpus's most frequent sentence for every query reaches a respectable absolute BLEU-1 and is
still rejected.

## What this can and cannot claim

It **can** claim: that a frozen LM conditioned on a sentence-level EEG embedding does or does not produce text distinguishable from five brain-independent controls on an unseen subject reading unseen sentences; how much of any ZuCo retrieval number is sentence length; and where the bottleneck sits, via the text oracle.

It **cannot** claim sentence reconstruction from EEG. At ~4.7 bits of measured sentence identity against ~190 bits of English sentence, and with 5.14 of those bits attributable to word count, the arithmetic forbids it — and a system that appeared to do it at this scale would be reciting the corpus. That is the failure this whole design is built to detect rather than to avoid mentioning.

## References

- Li, X. L. & Liang, P. (2021). Prefix-tuning: optimizing continuous prompts for generation. *ACL*. — the soft-prompt
  formulation the bridge emits.
- Lester, B., Al-Rfou, R. & Constant, N. (2021). The power of scale for parameter-efficient prompt tuning. *EMNLP*.
- Mokady, R., Hertz, A. & Bermano, A. (2021). ClipCap: CLIP prefix for image captioning. — a frozen LM driven by a
  mapped embedding, the closest architectural analogue.
- Merullo, J. et al. (2023). Linearly mapping from image to text space. *ICLR*. — evidence that a small linear map into
  a frozen LM's input space is sufficient when the source representation carries the content.
- Radford, A. et al. (2021). Learning transferable visual models from natural language supervision. — CLIP, the
  alignment the conditioning vector inherits.
- Wang, L. et al. (2022). Text embeddings by weakly-supervised contrastive pre-training. — E5, the frozen text space.
- Yang, A. et al. (2024). Qwen2.5 technical report. — the frozen decoder.
- Wang, Z. & Ji, H. (2022). Open vocabulary electroencephalography-to-text decoding. *AAAI*. — the result this design
  is built not to reproduce accidentally.
- Jo, H. et al. (2024). Are EEG-to-text models working? — teacher-forced EEG-to-text scores survive replacing the EEG
  with noise; the reason free-running decode is the only headline path here.
- Défossez, A. et al. (2023). Decoding speech perception from non-invasive brain recordings.
  *Nature Machine Intelligence*.
- Tang, J. et al. (2023). Semantic reconstruction of continuous language from non-invasive brain recordings.
  *Nature Neuroscience*. — sentence-level gist against a frozen LM as the honest bar.
- Papineni, K. et al. (2002). BLEU. *ACL*; Lin, C.-Y. (2004). ROUGE. *Text Summarization Branches Out*.
- Ho, J. & Salimans, T. (2022). Classifier-free diffusion guidance. — the mechanism `cfg_weight` would enable, kept at
  1.0 in v1 so the headline and the null-prefix control share one code path.

---

## The decode studio (`zte-studio`)

`zte-decode` answers *did it beat its controls*. The studio answers *what did it actually do*, for one reading at a
time, and it exists because a paired delta with a confidence interval tells you nothing about whether the machinery
is wired up the way you think it is.

```sh
uv run zte-studio --ckpt <decoder best.pt> --root <data> --split test --rows 8 \
    --montage res/montage_gsn105.csv --out res/analysis/STUDIO.html
```

One self-contained HTML file, no server and no network. What it draws, and the real quantity behind each panel:

| panel | the quantity |
| --- | --- |
| Scalp field, 2-D cap and draggable 3-D head | per-word band power interpolated over the montage, at the word the pointer is on |
| Target sentence | the pointer's Gaussian window over the reading's words at the current step |
| Decoded text | tokens as emitted, shaded by probability; clicking one seeks to that step |
| Alternatives | the top-8 next-token distribution at that step |
| Evidence KL | the same hidden state with and without the nudge -- how hard the brain pushed on *this* token |
| Pointer walk | the whole `(step, word)` attention matrix |
| Rate-ladder codes | the codebook entry each stage selected for this reading |

### The trace cannot change the decode

`FrozenLM.generate_from_prefix` takes an optional `trace` sink. Passing one adds a second head application per step
-- the un-nudged logits, needed for the evidence KL -- and appends a per-step record. Passing `None`, which every
headline, control and oracle decode does, leaves the loop byte-identical.

That is the load-bearing property and it is mutation-tested: break the loop so the trace perturbs the emitted token
and `test_tracing_a_decode_does_not_change_the_decode` goes red. A page built to inspect a decode showing a
*different* decode from the one the evaluation scored is the failure this guards against, and nothing downstream
would have caught it.

### What the evidence KL is, and what it is not

It is $\mathrm{KL}(p_{\text{nudged}} \parallel p_{\text{bare}})$ at one step, with the same cache and the same token
history -- the nudge is the only difference, so the comparison is matched by construction. It is **not** the
prefix-influence KL in the verdict, which compares whole prefixes; and it is not a comparison against a separately
decoded null prefix, which would diverge in what it emits after the first step and stop being matched at all.

### Reading it honestly

The page carries its own banner saying so, and the banner is asserted in the test suite:

- **A handful of readings is an anecdote.** The verdict needs the paired delta over every held-out reading, its
  bootstrap interval and the permutation null. Those live in `zte-decode` and the evaluation report.
- **Absolute scores mean nothing alone.** A frozen LM reaches ROUGE-1 of 0.10-0.18 against any English reference
  from function words. The controls beside each decode are the readable part.
- **The scalp colour scale is relative within one reading** -- log-scaled and quantised per reading -- so two maps
  that look alike are not alike in microvolts.
- **The pointer indexes EEG word slots**, and `target_words` is a text tokenisation. They usually agree; when the
  counts differ the page shows the slot index and declines to name the word.

The studio decodes `null_prefix`, `length_only` and `mismatch` for itself so one reading is readable on its own.
That is a subset of the pre-registered seven, chosen to fit a page, and it is not the gate.
