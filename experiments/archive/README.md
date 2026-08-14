# Archive — configs that were measured and set aside

Nothing here is deleted, because a negative result is still a result: these are the arms that were run on real
ZuCo and did not earn a place in the suite. Each row gives the number that retired it.

## Retired 2026-08-14 — capacity that did not buy retrieval

The 2026-08-13 Drive session re-ran the flagship set on one fold (held out on `ZAB`, 700 queries) and scored it on
**rank percentile**, which uses every query rather than only the ones that land. That metric turns out to be
seed-stable where Top-1 is not — the same `zte_raw_aligned` config at two seeds gives 0.9672 and 0.9670 while its
Top-1 moves 9 hits to 8 — so it is what these two demotions rest on.

| config | rank percentile (95% CI) | Top-1 / 700 | Top-5 / 700 | eff-rank | verdict |
| --- | --- | --- | --- | --- | --- |
| `zte_raw_aligned_wide` | **0.9523** (0.9479–0.9565) | 5 | 14 | 0.45 | Significantly below the retained set. |
| `clip_e5_meaning_raw_v2` | never scored on this board | 2 | — | 0.53 | Its one honest number is chance. |

**`zte_raw_aligned_wide` is the first arm this board separates.** Its interval (0.9479–0.9565) does not overlap the
retained three (~0.963–0.971), so this is a real difference and not the run-to-run noise that has swallowed every
previous comparison here. Its length-stratified Top-1 of 0.0200 is *below* the 0.0285 stratified chance rate.

**`clip_e5_meaning_raw_v2` was never re-run**, so its report predates the rank-percentile scoreboard entirely. The
only held-out number it has is 2 hits in 700 where chance expects 1, and it is the sole arm on the board whose
same-category cross-subject cosine gap is **negative** (−0.003): related meanings sit no closer than random ones.

Both carry the lesson worth keeping. They have the two healthiest geometries ever measured here — effective-rank
ratios of 0.45 and 0.53 against the retained set's 0.24–0.25 — and the two worst retrieval scores. A high effective
rank is not evidence of a better representation; read alongside retrieval, it says the extra directions are not
carrying sentence identity. The existing warning in this file is that a *low* effective rank can mean invariance
bought by destroying capacity. The converse now also has a measurement: capacity retained is not capacity used.

## Retired 2026-07-25 — the honest-board re-scoring

Every run on Drive (`Sharables/ZTE/2026-07-24`, `2026-07-25`) was re-scored on the **held-out** subject rather than
the pooled set. Pooled retrieval includes the training subjects, so it rewards memorising the 11 brains you have
instead of generalising to the 12th. On the honest board, held out on ZAB with 700 queries and chance = 1/700:

| config | frontend | Top-5 hits / 700 | exact p | eff-rank | verdict |
| --- | --- | --- | --- | --- | --- |
| `clip_e5_meaning.yaml` (exp9) | band power | 10 | 0.03 | 0.160 | the *former* champion — see below |
| `clip_bge_meaning.yaml` | band power | 9 | 0.07 | 0.160 | indistinguishable from E5, and from noise |
| `clip_e5_bandpower.yaml` (exp8) | band power | — | — | 0.166 | dominated by every raw arm |
| `clip_qwen_bandpower.yaml` | band power | 5 | 0.56 | 0.17 | Qwen is a dead end; no sentence head |
| `clip_mpnet_meaning.yaml` | band power | 7 | 0.24 | 0.16 | no better than chance |

**Why the exp9 "champion" was not one.** Its headline Top-1 of 0.043 was the *pooled* figure. Held out, it scored
4 hits in 700 — and an identical re-run the next day scored 2. Run-to-run noise was the size of the effect.

**Why the whole band-power family went with it.** Its subject probe of 0.23 was read as strong disentanglement,
but the *raw band-power features* only score 0.16 to begin with: there was almost no identity there to remove.
The effective-rank ratio of 0.160 is the real tell — the 768-d space was spanned by roughly 123 directions.
Invariance had been bought by destroying capacity, and the pooled metric was paying for it.

The raw conformer scores 32/700 on the same fold (p≈1e-15) at eff-rank 0.264. That is the line of work that
continues, in `experiments/flagship/`.

## Retired earlier

`exp1`–`exp7` are the pre-CLIP objective sweep (skip-gram, masked, CPC), superseded by the CLIP alignment work.
`study_all_levers`, `study_anticone_on/off` predate the 2026-07-24 scoreboard and target knobs that the current
flagship no longer exposes the same way.

## Re-running one

They still work — nothing was removed from the code:

```bash
bash scripts/run_suite.sh --config experiments/archive/clip_e5_meaning.yaml
```
