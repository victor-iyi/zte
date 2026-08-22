# Alignment levels — sentence, word and sub-word as one controlled contrast

Every ZTE encoder on the board is trained by a contrastive alignment loss, and every one of them has aligned the
*pooled sentence vector*. What that loss is applied to has never been the manipulated variable. The three-level
study makes it the only one: three arms of one recipe that differ in the granularity at which EEG is asked to mean
something, and in nothing else.

Configs: [`experiments/alignment/`](../experiments/alignment/) — `{sentence,word,token}/{combined,nr,sr,tsr}.yaml`.
Code: `zte.models.objectives.token`, `zte.data.targets.tokens`, `zte.alignment` (`atlas`, `contrastive`, `compare`).
Notebooks: `notebooks/alignments/zte_{sentence,word,token}.ipynb`. CLI: `zte-colab sweep`,
`zte-rebaseline --piece-oracle`. The sentence-level objective itself is [`CLIP_ALIGNMENT.md`](CLIP_ALIGNMENT.md);
metric definitions are [`EVALUATION.md`](EVALUATION.md). This document owns the study, the sub-word level's
mathematics, and the confound that gates it.

## The three levels

| Level      | One aligned unit is                                          | It is aligned against                                          | The levers that define it                    |
| ---------- | ------------------------------------------------------------ | -------------------------------------------------------------- | -------------------------------------------- |
| `sentence` | the pooled sentence vector — one per reading                 | a frozen E5 embedding of the ground-truth sentence             | every `lexical_*` and `token_*` weight at 0  |
| `word`     | one fixated word — one EEG token                             | that word type's frozen vector, and other readers of that word | `lexical_weight` / `lexical_reader_weight` 1 |
| `token`    | four fixed intra-word slices of the word's 350-sample window | the frozen sub-word embeddings of the pieces it spells         | `token_weight` / `token_reader_weight` 1     |

Every arm is byte-identical to [`experiments/ablation/exp16_residual_off.yaml`](../experiments/ablation/exp16_residual_off.yaml)
except the weights named above, `train.eval_profile` and `run_name`. The sentence-level CLIP term is present in all
three, so a level is **the CLIP objective plus at most one extra term** — the sentence arm adds nothing, the word
arm adds the lexical pair, the token arm adds the sub-word pair.

### Why exclusive rather than cumulative

Stacking the levels would be the natural reading of a hierarchy, and it is the wrong experiment. A cumulative token
arm would differ from the word arm in two loss terms and from the sentence arm in three, so a movement in held-out
retrieval could be attributed to granularity, to the number of terms competing for the gradient budget, or to the
magnitude of the total loss — three explanations, one measurement, no attribution. Exclusive levels keep the
project's one-variable-per-ablation rule: sentence → word flips exactly one lever, and sentence → token flips
exactly one other.

The design also inherits an anchor. `exp16_residual_off` is the best-measured encoder this project has produced —
held-out Top-1 0.0371, which on the 700-sentence gallery is **26 hits**, though `docs/RESULTS.md` marks the
whole exp16 sweep **stale** pending re-measurement after the length-projection fix — and the word arm *is* that recipe with a
new name. So the study is not three untested arms hoping one lands; it is one measured arm with a term subtracted
and the same term swapped, and the two new arms are read against a number that already exists.

## The sub-word path, in shapes

A batch of `B` sentence-readings, each `L` padded word slots, `K = objective.token_sub_tokens` slices per word
(fixed at 4), `H = model.hidden_dim`, and `Dt` the width of the frozen sub-word space (896 for Qwen2.5-0.5B).

```plaintext
EEG side (per batch):
  raw_eeg                (B, L, 105, 350)     350 samples ~ 700 ms of the fixation, 105 electrodes
  sub_tokens(x, K)       (B, L, K, H)         the conformer's time axis cut into K equal spans, mean-pooled
  select usable rows     (N, H)               real, fixated words with a target; N <= token_max_tokens (4096)
  head + L2 norm         u  (N, Dt)           the sub-word-space vector scored below

Target side (built once per run, frozen, cached):
  subword_matrix         (n_types, Dt)        one row per sub-word type the corpus spells, L2-normalised
  piece_target           (n_content, K)       row of subword_matrix for slot k of word content_id; -1 if absent

Type direction:
  types / inverse        (|T|,) / (N,)        the distinct pieces present in this batch, and each row's index
  gallery                (|T|, Dt)
  logits                 (N, |T|)             cross-entropy against `inverse`

Reader direction:
  logits                 (N, N)               67 MB at N = 4096 in float32 — which is why the cap exists
  positive[i, j]         same content_id AND same slot AND different subject
  candidate[i, j]        different content_id (AND same subject, by default)
```

The pooled sentence vector is produced by a *separate* forward pass (`ZTEModel.token_hidden`), not by re-pooling the
slices. Deriving it from the slices would be cheaper, but the spans do not divide 350 samples evenly, so the pooled
vector would stop being bit-identical to the sentence arm's — and the level comparison would stop being a matched
pair, which is the one thing it exists to be.

## The loss

Write $A$ for the batch's scored sub-tokens. Sub-token $i$ carries a unit vector $u_i \in \mathbb{R}^{D_t}$, the
`content_id` $c_i$ of the word it slices — which is the pair (stimulus, word index) — its slot
$k_i \in \{0,\dots,K-1\}$,
and its subject $s_i$. Let $\pi(c, k)$ be the sub-word type of slot $k$ of word $c$, $e_t$ that type's frozen
embedding, $T$ the distinct types present in the batch, and $1/\tau$ the learnable logit scale, clamped at 100.

The **type direction** asks whether this is the EEG of *this* word-piece — absolute lexical identity, learnable from
a single reader and not required to transfer:

$$
\mathcal{L}_\text{type} \;=\; -\frac{1}{|A|}\sum_{i \in A}
  \log \frac{\exp\big(u_i \cdot e_{\pi(c_i,\,k_i)} / \tau\big)}{\sum_{t \in T}\exp\big(u_i \cdot e_t / \tau\big)}
$$

The **reader direction** asks whether this is the same word-piece, whoever read it — the property a cross-subject
decoder actually needs. It is a multi-positive InfoNCE at sub-word resolution, over

$$
\begin{aligned}
P(i) &= \{\, j : c_j = c_i,\; k_j = k_i,\; s_j \neq s_i \,\} \\
N(i) &= \{\, j \neq i : c_j \neq c_i,\; s_j = s_i \,\}
\end{aligned}
$$

$$
\mathcal{L}_\text{reader} \;=\; -\frac{1}{|A'|}\sum_{i \in A'}
  \log \frac{\sum_{j \in P(i)} \exp\big(u_i \cdot u_j / \tau\big)}
            {\sum_{j \in P(i) \cup N(i)} \exp\big(u_i \cdot u_j / \tau\big)}
$$

where $A'$ is the subset of anchors that have both a positive and a candidate — an anchor with neither contributes
nothing rather than contributing a degenerate term. The two are added to the sentence objective with independent
weights, because they are different claims:

$$
\mathcal{L} \;=\; \mathcal{L}_\text{clip}
  \;+\; \lambda_\text{type}\,\mathcal{L}_\text{type}
  \;+\; \lambda_\text{reader}\,\mathcal{L}_\text{reader}
$$

**The positive is keyed on `(stimulus, word index, slot)`, never on a position in the reading's own sub-token
sequence.** This is the single most important line in the implementation. About a third of ZuCo's words are never
fixated and therefore produce no EEG at all, so two readers of the same sentence contribute different *subsets* of
its words. Under a positional key, reader A's $k$-th sub-token and reader B's $k$-th sub-token are pieces of
different words entirely, and the loss would spend its gradient insisting that two unrelated word-pieces are the
same thing. `content_id` already encodes (stimulus, word index), so keying on it is both correct and free: the
pairing survives any pattern of omission, any per-subject split, and any reordering of the batch.

The negatives are drawn from the anchor's own subject (`objective.token_same_subject_negatives`, default on), so a
negative can never be rejected on the grounds that it came from somebody else — the shortcut that makes an easy
contrastive loss worthless in a cross-subject setting.

## Why a word's own slices never negate each other

The four slices of one word are adjacent windows of a single 350-sample fixation. They are near-identical inputs by
construction. Left in each other's denominator they become hard negatives carrying different labels, and the loss
then has exactly two outcomes, both bad: it is unsatisfiable, or it is satisfiable only by encoding *position within
the fixation*. That second solution is a clock — and a clock, read off a word's slices, reconstructs how many pieces
that word was cut into, which is precisely the channel the next section is about. So the exclusion is not a
regulariser and not a tuning choice: `_EXCLUDE_SAME_WORD` in `zte.models.objectives.token` is a module constant with
no config knob behind it, because the alternative has no honest solution.

## The piece-profile confound

This is the section to read before quoting any token-level number.

The obvious way to build a sub-word objective is to give a word as many EEG sub-tokens as the reference spells it
word-pieces — three slices for a three-piece word, one for a one-piece word. It is the natural design, and it hands
the model the sentence's **piece profile**: the ordered vector of per-word piece counts. That vector is a
brain-free observable, and on a 700-sentence gallery it is very nearly a unique key.

Measured with the real `Qwen/Qwen2.5-0.5B` fast tokeniser on a 700-sentence corpus matched to ZuCo's statistics
(1.463 pieces per word against ZuCo's measured 1.4), against the identity ceiling
$H(\text{identity}) = \log_2 700 = 9.4512$ bits:

| Observable                                        | Bits of identity | Oracle Top-1 | Hits in 700 |
| ------------------------------------------------- | ---------------- | ------------ | ----------- |
| word count alone — the documented length confound | 4.96             | 4.71%        | 33          |
| total sub-word pieces — **one integer**           | 5.58             | 8.86%        | 62          |
| word count and total pieces jointly               | 8.18             | 51.29%       | 359         |
| the ordered per-word piece profile                | 9.44             | 99.57%       | 697         |
| that profile after ZuCo's 33% word omission       | 9.37             | 96.14%       | 673         |

An oracle here resolves a query to the set of gallery sentences sharing its signature, ranks that stratum uniformly
at random and everything else behind it. With $N$ gallery sentences, $\sigma_i$ the signature of sentence $i$ and
$m(\sigma)$ the number of sentences carrying signature $\sigma$, the scores are exact expectations over that
ordering — no seed, no Monte Carlo error:

$$
\text{Top-}k \;=\; \frac{1}{N}\sum_{i=1}^{N} \frac{\min\big(k,\, m(\sigma_i)\big)}{m(\sigma_i)}
\qquad
I(\text{identity};\, \sigma) \;=\; \log_2 N \;-\; \frac{1}{N}\sum_{i=1}^{N} \log_2 m(\sigma_i)
$$

The first row is the validity check on the stand-in corpus: 4.96 bits against the 5.1422 bits measured on the real
ZuCo gallery, so the matched corpus reproduces the documented length confound to within two tenths of a bit. It
fragments marginally *more* than ZuCo does, which makes its profiles slightly more discriminative than ZuCo's — the
gaps in the table are nowhere near close enough for that to change any ordering.

**What the table means.** The best encoder this project has ever measured retrieves 26 sentences out of 700 -- a stale figure, but not one
any plausible re-measurement moves by the order of magnitude this comparison turns on. The
*total number of sub-word pieces* — a single integer per sentence, computable from the text with no brain involved
— retrieves 62. The full profile retrieves 697, and still retrieves 673 when it is restricted to the roughly two
thirds of words a reader actually fixates, which is all the EEG a reading contains. A token-level objective built
the obvious way would not be measuring EEG decoding at all; it would be measuring how well a model can count, and
it would post a number several times the champion's while doing so.

### The structural answer, and where it is enforced

The fix is not a correction applied afterwards. **`objective.token_sub_tokens` is a fixed 4 for every word,
whatever its text says**, and the piece count is allowed to touch the loss's target mask and nothing else:

- `RawConformerFrontend.sub_tokens` cuts the contextualised time axis into `n_sub` equal spans with `torch.linspace`
  over the window. It never sees the reference text, so a word's EEG is encoded identically whatever it spells.
- `TokenAligner._select` drops a slot whose `piece_target` row is `-1` — a slot past that word's own piece count.
  That is a mask on which slots receive gradient, applied to the loss, and it is the only place in the whole level
  where the piece count acts.
- `TokenAligner.attach` refuses a target whose slot axis disagrees with `n_sub`, so a mismatch is a startup error
  rather than a silently mis-supervised run.
- Nothing scored at evaluation is computed from the count. Retrieval reads the pooled sentence vector from
  `token_hidden`, a forward path the token term never enters: the slices are a second pass, so the computation that
  produces the scored vector is the sentence arm's, unchanged.

### What the structure does *not* buy — the padding boundary, and why the gate is a gate

The argument above is about *computation*, and finishing it honestly is where a hostile reading lands.

The sub-token pass and the word pass share a trunk: both run `RawConformerFrontend._contextualise`, so the
sub-word loss backpropagates into the very weights `token_hidden` uses. A token level that succeeds therefore
makes the word representation carry sub-word information, and the pooled sentence vector inherits it. That much is
the mechanism working rather than a leak — provided the encoder can only get there by reading the brain.

**It cannot only get there by reading the brain, and this is the substrate's doing rather than the level's.**
ZuCo's per-word window is a variable-length fixation segment zero-padded to `raw_window` samples
(`zte.data.io.mat_loader._raw_window`) and then z-scored across the whole padded width
(`zte.data.features.transforms.sanitize_raw_windows`). The tail is therefore not merely zero: it is an *exactly
constant* per-channel value beginning at sample $L$, where $L$ is the fixation length. Measured directly:

| true fixation length $L$ | tail constant from sample | tail value |
| ------------------------ | ------------------------- | ---------- |
| 120                      | 120                       | $-0.1487$  |
| 200                      | 200                       | $-0.0982$  |
| 305                      | 305                       | $-0.1492$  |
| 350 (no padding)         | —                         | —          |

$L$ is an eye-tracking quantity. Word length is linearly readable off eye-tracking features — that is the whole
basis of the probe machinery control — so the padding boundary hands the encoder a per-word length estimate for
free, and Euclidean alignment is a linear channel mix that preserves a constant-in-time vector rather than erasing
it. This is a property of every raw-representation run on the board, not something the token level introduced.

What the token level adds is an *incentive*. Slot $k$ is supervised only for words with more than $k$ pieces, and
piece count moves with word length, so the supervision pattern is indexed by the same latent variable the padding
boundary gives away. The cheapest gradient path to "predict piece 3 of this word" may be "notice the signal does
not stop until sample 300".

**No assertion about tensor shapes distinguishes that from decoding.** The route carries no reference tensor, so
every structural guard in this change stays green while it runs. Only measurement separates them, and that is why
the oracle below is a gate on *reporting* rather than a diagnostic to consult when convenient. Three instruments
belong beside any token-level number, and none of them has been run:

- probe fixation length and piece count from the word hidden, and total piece count from the pooled sentence
  vector, token arm against the matched sentence arm — the delta is the answer;
- a gallery stratified on total piece count, as the length-stratified block is on word count;
- the direct ablation: noise-fill the padding tail so the boundary carries nothing, and see whether the token
  arm's lift survives it.

### The gate

Reporting is gated in code, not by convention:

- `zte-rebaseline --piece-oracle` builds the run's own alignment table and scores all four signatures through
  `zte.evaluation.audit.rebaseline.piece_profile_report`, returning a `clears` verdict for each.
- **The ordered profile is the ceiling, not the gate.** It is the floor for a design that sized a word's EEG by
  how many pieces its reference spells it in, and on a real gallery it resolves 99.6% of sentences — so gating on
  it would read `below the floor` whatever the encoder did, which is a column carrying no information rather than
  a check. This encoder is fixed-K, so what it can actually reach is the **total** piece count through the length
  channel above; `gate_signature` defaults to `total` and every signature's verdict is printed beside it, so the
  choice is visible rather than assumed.
- A run with no observed Top-1 beside it reads `not measured`, never as a pass.
- `zte.alignment.compare.LevelRetrieval` **refuses to construct** a token-level row without an `oracle_floor`, so a
  cross-level table cannot be built that quotes a token number with no floor beside it.

This is to a token-level headline exactly what `length_oracle` is to a sentence-level one, and it is the larger of
the two: word count carries 5.14 of the gallery's 9.45 bits, the piece profile carries 9.4 of them.

## Building the word-to-sub-word map

`zte.data.targets.tokens.build_token_alignment` answers one question per sentence: which sub-word slots belong to
which of its words. It is built from **real character offsets**, not from a heuristic re-tokenisation.

The fast tokeniser is asked for `return_offsets_mapping`, giving the character span of every slot. Both that call
and the id encoding beside it tokenise with the library's default `add_special_tokens=True`; if they disagreed, the
two sequences would differ by however many specials the tokeniser prepends and every slot would be attributed to
the word one place to its left. A special token carries a zero-width span and is dropped. A slow tokeniser reports
no offsets and is refused with an error naming the missing `tokenizers` install rather than guessing.

Word starts are found by scanning the reference text forward with a cursor, so a word repeated in one sentence gets
two distinct anchors instead of collapsing onto the first. **A word then owns the text from where it starts up to
where the next word starts**, resolved with a binary search over the anchors. That rule is what gives trailing
punctuation to the word before it: without it, every comma and full stop would be an unsupervised slot, which on
ZuCo is about a sixth of the sub-word sequence.

A word the scan cannot locate — ZuCo's word list sometimes normalises a character the reference spells differently
— leaves the cursor where it was, so every later word still aligns, and is counted against a reported **coverage**
figure. Coverage below 1 means words the loss can never reach, so the level is training on less than it appears to;
below 0.99 it logs a warning, which is loud enough to notice and not fatal, because a handful of unmatched words is
normal. `TokenAlignment` carries that coverage and a tokeniser fingerprint, so a silent tokeniser upgrade cannot
pass unnoticed, and the tables are cached under a key that includes the corpus, the tokeniser, the revision, the
width and the alignment-rule version.

The frozen target itself is restricted to the sub-word types the corpus actually uses: Qwen2.5's table is
151,936 × 896, which is 544 MB of frozen buffer to carry a corpus that spells 700 sentences with a few thousand
distinct pieces. Without `transformers` the target degrades to a deterministic hash so the pipeline still runs — and
logs that nothing lexical from such a run is meaningful, because it is not.

## Config surface

| Key                                      | Meaning                                                                                       |
| ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| `objective.token_weight`                 | Weight of the EEG-sub-token-to-frozen-sub-word direction; 0 disables the whole level.         |
| `objective.token_reader_weight`          | Weight of the same-piece-different-reader direction.                                          |
| `objective.token_sub_tokens`             | Slices per word. **A fixed constant, never a per-word piece count** — see the confound above. |
| `objective.token_source`                 | Model whose input embedding table supplies the target; `null` uses `decoder.lm_source`.       |
| `objective.token_temperature`            | Initial softmax temperature for both directions; the log-scale is learnable and clamped.      |
| `objective.token_max_tokens`             | Cap on sub-tokens scored per step — the reader direction is quadratic in them.                |
| `objective.token_same_subject_negatives` | Restrict the reader direction's negatives to the anchor's own subject.                        |
| `objective.token_max_length`             | Sub-word width the alignment table is built to, matching the decoder's own target width.      |

Both weights default to 0, so every existing config and every number already on the board is unaffected by the
level existing. The aligner is constructed at attach time rather than in `__init__`, so a run with the level off
builds no parameter at all and every checkpoint written before the level existed still loads under `strict=True`.

The level needs a raw-window frontend. `ZTEModel.sub_token_hidden` raises `NotImplementedError` against a frontend
with no intra-word path, which band power does not have — there is no sub-word structure to slice out of a
per-word band-power vector.

## Running the campaign

54 planned runs, 51 distinct after the shared arm, **108.8 GPU-hours** — the plan is generated, not maintained by
hand, by `zte-colab sweep`:

| Tier        | Runs | Hours | The question it answers                                                                                |
| ----------- | ---- | ----- | ------------------------------------------------------------------------------------------------------ |
| `mechanism` | 12   | 17.8  | Does the level move anything at all? Three levels × four arms (combined, NR, SR, TSR), `ZAB`, seed 42. |
| `power`     | 36   | 84.0  | Does the combined arm survive twelve folds? Three levels × the twelve LOSO holdouts.                   |
| `spread`    | 6    | 14.0  | Does it survive a reseed? Three levels × combined × `ZAB` × seeds 43/44.                               |

The level is the innermost loop everywhere, so *every prefix of the plan is a matched comparison across all three*
rather than one level finished and two untouched. Tier 1's `ZAB` fold at seed 42 is the arm tier 0 already trained,
one per level, which is why 54 planned runs cost 51 trainings. A run counts as done when its
`evaluation/metrics.json` exists — never when the session `INDEX.md` says so, because a run that died between
writing its metrics and its catalogue row is finished, and keying on the catalogue would spend its hours a second
time.

Every arm carries `train.eval_profile: sweep`, and the reason is measured rather than assumed: on this project's
Colab timings evaluation takes 61–75 minutes against 36 minutes of training, so **evaluation is 63–67% of a run**.
Across 51 trainings the full profile would therefore spend most of the campaign's wall-clock producing blocks no
cross-level claim reads. `sweep` keeps embedding health, sentence retrieval, the held-out scoreboard and the
permutation null — the only numbers allowed to be a headline — and drops the neuron, emergence, analogy,
seen-vs-novel and frequency-matched blocks, every figure and the interactive explorers. The profile that produced a
run is stamped into its `metrics.json`, so a `sweep` number can never be mistaken for a full one.

```sh
uv run zte-colab sweep plan                      # the whole campaign, ordered
uv run zte-colab sweep next  --out-root res/experiments   # the next run that still owes hours
uv run zte-colab sweep status                    # what has landed, what is left

uv run zte-run --config experiments/alignment/token/combined.yaml \
    --root "<ZuCo>" --loso-holdout ZAB --out-root res/experiments --resume
uv run zte-rebaseline --ckpt <ckpt> --root "<ZuCo>" --out <audit> --piece-oracle
```

Each level has its own front door — `notebooks/alignments/zte_sentence.ipynb`, `zte_word.ipynb`, `zte_token.ipynb`
— and the token notebook scores the piece oracle before it renders anything.

## What is not claimed

Stated before the runs, so the reading of the results is pre-committed.

- **No token-level number exists.** The objective, its target builder, its oracle and its refusal are built and
  tested offline against synthetic data. Not one arm of the campaign has trained on real ZuCo. Nothing in this
  document is a result.
- **The piece-oracle table is measured on a matched-statistics corpus, not on ZuCo's own gallery.** It establishes
  the size of the channel and the design constraint that follows from it. The ZuCo figure is produced by
  `zte-rebaseline --piece-oracle` against a real checkpoint, and that is the number a token-level headline is read
  against.
- **The 26 hits in 700 belong to the word level alone, and are themselves stale.** `docs/RESULTS.md` requires
  the exp16 sweep to be re-measured after the length-projection fix, so the anchor is provisional; the word arm's
  own combined run re-measures it. It is the anchor the other two levels are read against —
  a measured number the study did not produce, and not evidence about either of the arms that have yet to run.
- **A null is a finding, and a likely one.** If the token level does not beat the word level while clearing its
  piece oracle, the honest statement is that intra-word EEG slices carry no recoverable cross-reader sub-word
  content on ZuCo — a publishable negative, and one worth reporting plainly.
- **Nothing here is a generation claim.** The readout is closed-set retrieval over the sentence gallery, and the
  bit budget has not moved: an encoder supplying a handful of bits of sentence identity does not become a text
  generator by aligning at a finer grain.
