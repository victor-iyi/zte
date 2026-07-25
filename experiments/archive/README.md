# Archive — configs that were measured and set aside

Nothing here is deleted, because a negative result is still a result: these are the arms that were run on real
ZuCo and did not earn a place in the suite. Each row gives the number that retired it.

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
