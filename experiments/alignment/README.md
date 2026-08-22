# The three-level alignment study

Three encoders that differ in **which unit the contrastive term pulls at**, and nothing else. Every arm here is
byte-identical to [`ablation/exp16_residual_off.yaml`](../ablation/exp16_residual_off.yaml) — the recipe that
measured the programme's best held-out retrieval, 26 of 700 on a subject no parameter had seen — except for the
two weights that switch its level on. That 26 is itself **stale**: `docs/RESULTS.md` requires the whole exp16
sweep to be re-measured after the length-projection fix, and the word level's own combined arm is what
re-measures it.

| Level      | The unit it aligns                       | Frozen target                | Config                   |
| ---------- | ---------------------------------------- | ---------------------------- | ------------------------ |
| `sentence` | the pooled sentence vector               | an E5 sentence embedding     | [`sentence/`](sentence/) |
| `word`     | one fixated word = one EEG token         | a frozen word vector         | [`word/`](word/)         |
| `token`    | four fixed intra-word slices of one word | the LM's sub-word embeddings | [`token/`](token/)       |

Each level has four arms: `combined.yaml` trains SR+NR together, and `nr.yaml` / `sr.yaml` / `tsr.yaml` train one
reading task alone so the parallax transfer matrix can ask whether the effect survives a passage set the encoder
has never seen.

## Reading the ladder

The levels are **exclusive, not cumulative**. Each is the sentence-level CLIP objective plus exactly one extra
term, so `sentence -> word` and `sentence -> token` each flip one lever and a difference between them is
attributable to it. A cumulative arm — all three terms at once — is a natural follow-up and is deliberately not
one of these twelve.

```sh
diff <(grep -E 'lexical_weight|token_weight' sentence/combined.yaml) \
     <(grep -E 'lexical_weight|token_weight' token/combined.yaml)
```

Reading the configuration diff rather than the run names is the point: the whole knob set is written out in every
arm, so the diff between two files names exactly the levers that move and nothing else.

`word/combined.yaml` is the published champion recipe. It is included as a level rather than referenced so the
three arms are a matched triple scored under one evaluation profile.

## Before you quote a token-level number

**The sub-word piece profile is a brain-free channel larger than the sentence identity it would be used to
recover.** Measured with the real `Qwen/Qwen2.5-0.5B` tokeniser on a 700-sentence corpus matched to ZuCo's
statistics (1.463 pieces per word against ZuCo's measured 1.4):

| What an oracle is told, and nothing else             |     Bits |      Top-1 | hits / 700 |
| ---------------------------------------------------- | -------: | ---------: | ---------: |
| `n_words` — the documented 5.14-bit length confound  |     4.96 |      4.71% |         33 |
| **total sub-word pieces — one integer per sentence** | **5.58** |  **8.86%** |     **62** |
| `(n_words, total pieces)` jointly                    |     8.18 |     51.29% |        359 |
| **the per-word piece profile**                       | **9.44** | **99.57%** |    **697** |
| the same profile after ZuCo's 33% word omission      |     9.37 |     96.14% |        673 |

$H(\text{identity}) = 9.4512$ bits on a 700-sentence gallery, so the ordered profile leaves 0.01 bits unresolved.
A single integer — the total piece count — retrieves 62 of 700 where the best encoder in this programme retrieves
26.

Two consequences, both enforced in code:

- **`token_sub_tokens` is a fixed 4 for every word, never the number of pieces its reference spells it in.** The
  piece count enters the loss's target mask and nothing the encoder computes, so a reading is encoded identically
  whatever sentence it turns out to be. `tests/test_token_alignment.py` holds a structural guard on this: the
  intra-word path is allowed exactly two callers, both training-side, and gaining a third fails the suite.
- **Every token-level headline is gated on the piece oracle.** `zte.evaluation.audit.rebaseline.piece_profile_report`
  scores all four signatures above and returns a `clears` verdict for each. The gate is the **total** piece count —
  what a fixed-K encoder can actually reach, through the length channel below — and the ordered profile is reported
  beside it as the ceiling a variable-K design would have given away. Gating on the profile would print *below the
  floor* whatever the encoder did, which is decoration rather than a check.

### A second channel, in the substrate rather than the level

The per-word window is a variable-length fixation zero-padded to 350 samples and then z-scored across the whole
padded width, so its tail is an *exactly constant* value beginning at sample $L$ — the fixation length. Measured:
$L = 120$ gives a tail constant from sample 120, $L = 305$ from sample 305. Fixation length is an eye-tracking
quantity that word length is readable from, so the padding boundary hands every raw-representation encoder a
per-word length estimate for free. That is true of every arm on this board, not only these twelve.

What the token level adds is an incentive to use it: slot $k$ is supervised only for words with more than $k$
pieces, and piece count moves with word length. No structural guard can see the difference between "read the
sub-words" and "found the boundary", because neither carries a reference tensor. Three instruments settle it, and
none has been run: probe fixation length off the word hidden and total piece count off the pooled vector, token
arm against the matched sentence arm; stratify the gallery on total piece count; and noise-fill the padding tail
and see whether the lift survives. Read `docs/ALIGNMENT_LEVELS.md` before quoting anything from `token/`.

## The campaign

`uv run zte-colab sweep plan` emits the whole thing as JSON. 54 runs, ~109 GPU-hours:

| Tier        | What it answers                                                                         | Runs |
| ----------- | --------------------------------------------------------------------------------------- | ---: |
| `mechanism` | 3 levels × 2 regimes, holdout ZAB, seed 42 — directly comparable to the published table |   12 |
| `power`     | 3 levels × combined × all 12 folds — the population estimate with a real CI             |   36 |
| `spread`    | 3 levels × combined × ZAB × seeds 42/43/44 — separates seed noise from fold noise       |    6 |

Every stopping point is a complete, reportable table. A run counts as done when its
`evaluation/metrics.json` exists on Drive, so a reclaimed VM re-plans and skips finished work.

The parallax regime stays at one fold on purpose: the transfer matrix is a mechanism demonstration, not a powered
estimate, and twelve folds of it would be 324 further runs that buy no CI.

## Why these arms carry `eval_profile: sweep`

Evaluation, not training, is two thirds of a run on this project's own measured Colab timings — 61 to 75 minutes
against 36 of training. The `sweep` profile keeps exactly the numbers `CLAUDE.md` §5 permits as a headline
(sentence retrieval, the held-out scoreboard, the length-stratified block, the permutation null) and drops the
figures, the interactive explorers and the frequency-matched word gallery. Run the full profile on a winning arm.
