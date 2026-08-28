# CLIP — sentence-level EEG↔text alignment (`objective.name='clip'`)

The pivot from *self-supervised structural mapping* (skip-gram over EEG neighbours) to **explicit semantic alignment**: pool each ZuCo sentence's word-EEG tokens into one sentence vector and align it, with a **symmetric InfoNCE loss**, to a **frozen** text embedding of the ground-truth sentence. This directly attacks the content gap — the loss now *forces* the encoder to represent meaning, because the only way to win is to sit close to the right sentence's text vector and far from the rest.

Grounding: CLIP (Radford et al. 2021); non-invasive brain↔stimulus contrastive decoding (Défossez et al.  2023 — the same symmetric-InfoNCE recipe that gives MEG segment-retrieval top-10 ≈ 70%). Sentence-level gist against a frozen LM is also the fMRI/MEG-honest bar (Tang et al. 2023), and it sidesteps the single-word-EEG SNR ceiling (Défossez: EEG word top-1 ≈ 5%).

What is **kept** from the base recipe, as *auxiliaries*: VICReg anti-collapse, the rebalanced subject / stimulus adversary, cross-subject positives, and the eval-time geometry fix (whiten + ABTT + CSLS).  CLIP supplies the content; those keep the geometry healthy and subject-agnostic. They compose.

## Tensor shapes (the walkthrough)

Batch of `B` sentence-readings; each is `L` padded word tokens.

```plaintext
EEG side  (per batch):
  features            (B, L, F)            F = flattened band-power (Stage 1) or raw is (B, L, 105, T)
  token_hidden        (B, L, H)            H = model.hidden_dim   — per-word frontend
  contextualize       (B, L, H)            bidirectional transformer over the sentence
  _pool_tokens        (B, H)               masked attention pool over present tokens -> 1 vec/sentence
  project             (B, D)               D = model.embed_dim (768)
  clip_head + L2norm  z_eeg (B, Dt)        Dt = text-embedding width (e.g. 768 E5-base, 896 Qwen2.5-0.5B)

Text side (precomputed once, frozen, cached):
  z_txt               (B, Dt)              F.embedding(sentence_text_id, text_matrix); already L2-normed
  text_matrix         (n_sentences, Dt)    one row per UNIQUE sentence text (shared across subjects)

Similarity + symmetric loss:
  logit_scale = clamp(exp(learnable log-temp), max=100)          # CLIP temperature
  S = (z_eeg @ z_txt.T) * logit_scale        (B, B)              # row = EEG reading, col = text
  pos[i, j] = (text_id[i] == text_id[j])     (B, B)  bool         # multi-positive: same sentence,
                                                                 # any subject, is a positive
  loss = 0.5 * ( multipos_infonce(S,   pos)      # EEG -> text
              + multipos_infonce(S.T, pos) )     # text -> EEG
  loss += VICReg + adversary + (optional behaviour/data2vec)     # auxiliaries, on the token embeddings
```

`multipos_infonce(S, pos)` per anchor row = `logsumexp(S over valid columns) − logsumexp(S over positive columns)` — standard InfoNCE generalised to ≥1 positives. Because the **same sentence read by different subjects shares a `text_id`**, every EEG reading of a text is a positive for that text, so subject identity is pushed out of the aligned space *for free* (this is why the cross-subject-positives sampler and the CLIP multi-positive mask reinforce each other).

Insertion points: `zte.models.objectives.SentenceClipObjective` (`_sentence_vectors`, `compute`, `_clip_direction`), `zte.data.targets.text.build_sentence_text_matrix`, `zte.data.torch_dataset` (`sentence_text_id` in collate; `SemanticHardNegativeSampler`), `zte.training.pipeline` (build + attach).

## Semantic-hard negatives

Random distractors let the encoder win on surface form. `zte.data.targets.text.mine_hard_negatives` ranks, per sentence, the others by `surface_overlap − semantic_cosine` — high word-token Jaccard (they *look* alike) but low frozen-text cosine (they *mean* different things) — and `SemanticHardNegativeSampler` co-locates each anchor with those hard negatives (and its cross-subject positives) in the same batch. Verified: for "the cat sat on the mat", the miner picks "the dog sat on the rug" and rejects a semantically-identical paraphrase. Toggle with `objective.semantic_hard_negatives`, `objective.hard_negative_pool`.

### Matching the negative on length and spelling

Ranking by `surface_overlap − semantic_cosine` is the right criterion over the wrong candidate set. It imposes **no
length constraint**, and on this corpus word count carries 5.14 of the 9.45 bits needed to name a sentence — so a
mined negative can still be told from its anchor by counting words, which teaches the encoder nothing it did not
already get for free.

`objective.hard_negative_strategy` selects the candidate set before that ranking runs:

| value | a candidate must … | when to use it |
| --- | --- | --- |
| `surface` | nothing — every other sentence is admissible | the original behaviour, and the default |
| `length_matched` | sit within `hard_negative_length_tol` words of the anchor | when only the word-count channel matters |
| `piece_matched` | also sit within `hard_negative_piece_tol` of the anchor's **total sub-word count** | the sub-word budget as well as the word count |

`mine_matched_hard_negatives` adds one more term to the ranking: multiset overlap scaled by positional
*disagreement*, so "the man bit the dog" scores high against "the dog bit the man" while an identical sentence
scores zero. That is the negative which isolates role and syntax from bag-of-content — same length, same piece
profile, same vocabulary, different meaning.

A sentence that cannot fill its pool inside the tolerance **widens rather than returning fewer**, and every widening
is counted in the returned diagnostic, so a table that quietly degraded to unmatched negatives is one number away
rather than invisible. Piece totals come from the run's own token alignment; if the tokeniser will not load, the
pipeline warns loudly that the sub-word budget is **not** controlled and falls back to length matching alone.

### Reaching the loss, not just the batch

`semantic_hard_negatives` alone is *batch composition*: it raises the chance the hard negative is present in the
denominator. `objective.hard_negative_in_loss` goes further and narrows the full-gallery denominator to the anchor's
own mined negatives.

The narrowing runs **after** the existing band / task / split masking, never instead of it, and three properties are
enforced: the anchor's own text always survives, because a softmax with no numerator is not a loss; a mined negative
that is out-of-band or held out never enters; and an anchor whose mined negatives are all inadmissible falls back to
the wider denominator rather than facing a softmax containing only its own answer. `gallery_min_candidates` does
*not* widen a mined pool — a mined pool is thin on purpose.

All four knobs default off, so every existing run stays byte-identical. The matched pair is
`experiments/alignment/sentence/hardneg.yaml` against `combined.yaml`, differing in exactly those four lines.

## Config surface

| Key                                                        | Meaning                                                                                                                                 |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `objective.name: clip`                                     | Select the symmetric sentence-alignment objective.                                                                                      |
| `objective.text_source`                                    | Frozen text encoder id (`intfloat/e5-base-v2`, `BAAI/bge-base-en-v1.5`, `Qwen/Qwen2.5-0.5B`, …); `null` → hash target (mechanism only). |
| `objective.text_backend`                                   | `sentence-transformers` (E5/BGE), `hf` (mean-pool a decoder LLM like Qwen), or `auto`.                                                  |
| `objective.text_query_prefix`                              | Instruction prefix (E5 wants `"query: "`; empty for BGE/Qwen).                                                                          |
| `objective.clip_temperature`                               | Initial CLIP temperature; the log-scale is learnable + clamped.                                                                         |
| `objective.semantic_hard_negatives` / `hard_negative_pool` | Surface-similar / meaning-distinct in-batch negatives.                                                                                  |
| `objective.hard_negative_strategy`                          | Candidate set for mining: `surface` / `length_matched` / `piece_matched`.                                                               |
| `objective.hard_negative_length_tol` / `hard_negative_piece_tol` | Word-count and total-sub-word-count tolerances a matched candidate must satisfy.                                                   |
| `objective.hard_negative_in_loss`                           | Narrow the full-gallery denominator to the mined negatives, rather than only seeding the batch.                                        |
| `objective.cross_subject_positives`                        | Keep `true` — same-sentence readings co-occur as multi-positives.                                                                       |

VICReg / adversary / whiten / all_but_top / csls / eval-hardening keys are inherited from the skip-gram baseline (`experiments/benchmark/baseline_skipgram_loso.yaml`) and stay on as auxiliaries.

## The A/B (both text encoders) and staged granularity

- `experiments/flagship/zte_raw_aligned.yaml` — **E5** sentence encoder (purpose-built sentence embeddings). On real ZuCo (held out ZAB, 2026-07-16) this is the only recipe that has ever beaten chance: sentence-retrieval Top-1 0.093 vs 0.0013 chance, permutation *p*=0.002.
- `experiments/archive/clip_qwen_bandpower.yaml` — **Qwen2.5-0.5B**, mean-pooled (local, fast iteration). It sits in `benchmark/` as the text-encoder control.
- `experiments/flagship/clip_e5_raw.yaml` — **Stage 2**: the raw-signal sentence encoder (the raw-conformer temporal-spatial-convolution encoder — `raw_conformer` frontend, `raw_window: 350` ≈ 700 ms to reach the N400). Ships after Stage 1 (word-pool) validates the objective; it took the best held-out lift (+0.71pp).

The `run_name` inside each config is unchanged by the tiering, so these still write to `res/experiments/exp8_clip_e5/`, `exp8_clip_qwen/` and `exp8_clip_e5_raw/`.

`transformers` + `sentence-transformers` are opt-in (`uv sync --group meaning`); without them the target falls back to a hash and a warning, so the pipeline never breaks.

## How to run

```sh
# provision E5/Qwen once
uv sync --group meaning
# Stage-1 A/B, held out on ZAB (each writes the scoreboard + interactive dashboard, --resume-safe)
uv run zte-run --config experiments/flagship/zte_raw_aligned.yaml  --root "<ZuCo>" --loso-holdout ZAB --out-root res/experiments --resume
uv run zte-run --config experiments/archive/clip_qwen_bandpower.yaml --root "<ZuCo>" --loso-holdout ZAB --out-root res/experiments --resume
# compare E5 vs Qwen vs the skip-gram control on the held-out north-star
uv run zte-compare --experiments res/experiments
```

In the Colab notebook, add the two configs to the `SPOTLIGHT` list in cell 5-iv and run 5-iv -> 5-v.

## The honest win condition

Grade on **held-out sentence-retrieval rank-percentile shifting left of the permutation null** and a positive **content-lift-over-raw** on the held-out subject — not word top-1 (the field's hardest number).  A CLIP target that does *not* beat the whitened static baseline is itself the scientific result that word/sentence-EEG lacks the recoverable semantic content — a publishable negative, not a failure. The decoder still comes last.
