# Changelog

## Unreleased

### Added

- **`zte-lens attention` — the encoder's own attention, read through forward hooks.** A post-hoc pass over one
  subject's readings (the checkpoint's holdout by default) with `register_forward_pre_hook` /
  `register_forward_hook` on the electrode mixer's and the intra-word transformer's `nn.MultiheadAttention`, in
  `eval()` under `no_grad`. It writes `attention.json`, `attention.md`, a matplotlib temporal curve of attention
  received per time step over the 700 ms word window with the 300-500 ms band marked, and an `mne` scalp map of
  attention received per electrode -- each for the correctly retrieved readings, the rest, and all, with bootstrap
  intervals over readings and an interval on the correct - incorrect N400 mass. The scalp map is declined on the
  approximate geometry, the electrode weights are stated to carry no latency axis, and every artifact carries the
  lens disclaimer plus a caveat that a weight is not a counterfactual. Each figure is written as a PNG and a vector
  PDF. `notebooks/tbme/zte_attention.ipynb` drives it over the evidence suite's sentence-level folds, zips every
  artifact for a browser download, and `zte-colab audit --kind attention` reads the result.

- **`zte-lens` skips a lens it has already built.** It was the one expensive command with no `.zte-done` stamp, so
  every re-run repeated the occlusion passes -- channel saliency over ten electrode groups, and with `--temporal` a
  latency profile over every bin of every word of twelve readings. The stamp is decided before the dataset is built
  and covers the checkpoint's SHA-256, the dataset key and every flag, so adding `--temporal` or `--html` to a
  previous invocation correctly rebuilds while an identical one returns in the time it takes to hash a checkpoint.
- **`zte-evidence` and `zte-levels` fingerprint what they read, not just how they were called.** Both aggregate
  other commands' artifacts, so a stamp over flags alone would serve yesterday's board after a new audit landed.
  The record now covers the resolved artifacts' names and sizes -- never mtimes, which a Drive mirror resets -- so a
  new fold, a re-run audit or a newly written calibration rebuilds them on its own and neither ever needs `--force`.
- **`zte-loso-summary --experiments` takes several roots, and accepts run directories.** It previously took one
  directory and globbed `*/evaluation/metrics.json` under it, which meant a sweep root pooled every arm trained into
  it: folds are keyed on the holdout alone, so three alignment levels averaged into one trend. Naming the run
  directories of a single arm is now expressible, matching `zte-analyze`'s existing shape.
- **`zte-audit --config` and `--root` compose.** They were mutually exclusive though the body already handled both.
  `--config` decides which dataset is built -- representation, window, tasks, and so the bundle key -- while
  `--root` says where the files are; with `--root` alone the defaults built a band-power/128-sample dataset that
  keyed nowhere near a prepared raw bundle and re-parsed every archive.

- **The evidence board — `zte-evidence`.** One command assembles every measured claim beside the brain-free floor it
  has to clear, reading the artifact each audit already wrote and **recomputing nothing**, so the board cannot
  disagree with the runs it describes. Three rules are enforced in code and mutation-tested: a claim with no floor
  renders `not measured` and can never be a headline; the confidence *interval* must clear the floor, not the point
  estimate; and a missing artifact is named in `missing` rather than dropped, because a silently absent row reads as
  a claim nobody made. `Claim.headline_safe()` is the single predicate deciding whether a row may be quoted alone,
  and the rendered document says plainly when none may be. Top-*k* renders as a hit count over the queries actually
  scored, and a count too thin to resolve a difference carries that caveat whatever the verdict says.
- **The anchor-calibration curve — `zte-calibrate`.** The deployment measurement: held-out rank percentile against
  the number of labelled sentences a new reader has supplied, with the encoder frozen and nothing retrained. Two map
  families bracket what any affine calibration can buy (`procrustes`, rotation only; `ridge`, strictly more
  expressive), fitted from the reader's anchor readings onto the cross-subject prototype. Three arms are scored on
  one identical reduced gallery — `uncalibrated`, `calibrated` and `shuffled`, the last fitting the same map on a
  derangement of the anchor pairings — so the result is `calibrated − shuffled` as a *paired* per-draw difference,
  not a lift over zero. Every anchor stimulus leaves both the query set and the gallery. A thin fit returns `None`
  and logs rather than silently becoming an identity. See [`docs/CALIBRATION.md`](docs/CALIBRATION.md).
- **The granularity ablation as a command — `zte-levels`.** `cross_level_table` had no driver; this is it. It loads
  no model and re-scores no query, groups already-evaluated runs by alignment level, aggregates across LOSO folds
  with a **sample** (n−1) standard deviation and a bootstrap over fold means, and prints each level against its
  floor. A level with no measured floor renders as `floor not measured`, never as a pass.
- **`noise_prefix` — the decoder reality check.** A mean/variance-matched Gaussian **z** fed straight into the
  bridge, skipping the encoder entirely, so what it scores is the language model's prior plus the bridge. The
  existing `noise` control matches the *encoder input*; this one isolates the question the field's generation
  numbers actually turn on. Matched moments rather than a standard normal, because an off-manifold prefix is a
  trivially weak control. It is in `DecoderConfig.generation_controls` by default and named explicitly by every
  shipped decoder config, which **tightens** the verdict gate — an unavailable control fails its clause.
- **The temporal latency profile — `zte-lens encode --temporal`.** Occlusion within the word: a time span of the raw
  window is zeroed, the reading re-embedded, and the cosine displacement recorded, reported in milliseconds from
  word onset with a bootstrap interval per bin and a random-offset **null band**. Occlusion rather than attention
  because no trained checkpoint has an attentive temporal pool — `conformer_temporal_pool` is `mean` in every live
  config — and a counterfactual is the better instrument regardless. `peak_in_n400_window` is reported and gates
  nothing: ZuCo word windows are eye-tracking-segmented and overlap their neighbours, so a peak in that band is
  consistent with an N400 and is not proof of one.
- **`notebooks/tbme/zte_tbme.ipynb` — the evidence suite.** Eight experiments and the board, standalone on Colab,
  resumable, mirrored to Drive. Sections 7, 8, 9, 12 and 14 read checkpoints and cost minutes; the rest train.

- **The steps after training are not redone on a re-run.** `zte-audit`, `zte-decode`, `zte-rebaseline` and
  `zte-parallax transfer` record what each artifact was built from — every option, the checkpoint's SHA-256 and the
  dataset's bundle key — beside it as `.zte-done-<artifact>.json`, and skip when nothing has moved, so a notebook
  re-run top to bottom costs only its unfinished work. The decision comes before the bundle is staged and before a
  model loads, so a finished step re-runs in the time it takes to hash a checkpoint, and its headline is logged
  again from disk rather than left blank. Anything that would change the number rebuilds it: a checkpoint trained
  further, a changed option, different data, a deleted or half-written artifact, or an artifact carrying no record
  at all. The raw path is deliberately not part of the identity, so a re-mounted Drive does not invalidate a day of
  audits; `--force` on any of the four rebuilds regardless.
- **`write_mode: auto`, and it is now the default** (`zte-colab session --write-mode`). Runs are written straight to
  Drive when Drive is mounted, and to the local disk otherwise: a twelve-fold sweep is ~27 GB of checkpoints, which
  a Colab VM cannot hold beside an 11-24 GB dataset bundle, while a workstation wants the fast local disk. The
  resolved value is logged and carried in the session payload.
- **The bundle cache picks the roomiest local volume.** `res/cache/prepared` lives on whatever disk the checkout
  is on, which on a Colab GPU runtime is often not the largest attached; `zte.data.cache.scratch_root` scans
  `/content`, `/var/scratch`, `/scratch`, `/mnt/disks/local` and `/tmp` and moves the cache to any that beats the
  default by more than 20 GB. Probing creates nothing on the candidates it rejects, and `ZTE_SCRATCH_DIR` pins it.
- **The local bundle cache now evicts rather than accumulates.** Staging an entry first frees room for it, removing
  least-recently-used entries until the incoming one fits with `ZTE_MIN_FREE_GB` (default 12 GB) still free. Only an
  entry the persistent store holds complete is evictable, so nothing that would need rebuilding from the multi-GB
  extraction is ever deleted; a shortfall is reported instead. `BundleStore.staged()` and `make_room()` are public.

- **Decoder menu capacity — the readout that can be proved.** `zte.evaluation.audit.capacity` certifies the largest
  $K$-way menu a decoder serves: given the held-out reading and $K$ candidate sentences, does it score the one
  actually read above every distractor? Accuracy is the exact expectation over uniformly drawn distractors
  ($\binom{b}{K-1} / \binom{m}{K-1}$), so chance is exactly $1/K$, there is no sampling seed, and ties lose — a
  constant scorer gets 0.0, not chance. Seven clauses must hold at $K$ *and* at every smaller swept size *and* on
  the common subset of queries scoreable at every size: an honest `by_subject_and_stimulus`/`test` split, a
  length-matched (never `open`) pool, a bootstrap CI lower bound above $1/K$, paired wins over `length_only`,
  `shuffled_eeg` and `mismatch` on both a bootstrap CI and an exact sign test, and a permutation $p$ below alpha.
  Turned on by `objective.eval_capacity` (default `false`) or `zte-decode --capacity`; tuned by
  `decoder.capacity_ks` / `capacity_alpha` / `capacity_n_perm` / `capacity_score` and their CLI overrides
  `--capacity-ks` / `--capacity-alpha` / `--capacity-n-perm`. Written to `metrics['decoder_capacity']`, the sibling
  `evaluation/capacity.json`, and a Markdown block in `report.md`.
- **The conditioning arms it certifies against** (`zte.inference.capacity`): `gallery_scores` scores the whole
  gallery under every reading in one pass, and `capacity_arms` builds `model`, `length_only` (a word-count-matched
  training prefix at tolerance 0), `shuffled_eeg` (a derangement), `mismatch` (a different-stimulus, length-matched
  partner) and `null_prefix` — every one of them through the identical bridge, LM, scaffold and length
  normalisation, so only the conditioning differs. `null_prefix` is identically zero under PMI and is therefore
  reported under `raw` alone. An arm whose ingredients are missing is omitted rather than approximated, and its
  clause then fails.
- **`pooled_capacity`** turns a set of seeds or holdouts into one number: the smallest certified size across runs,
  and `None` if any run certified nothing.
- **`zte-colab capacity`** renders a decode run's certification as one JSON payload — per-K rows with the
  unreachable sizes named, the paired control comparisons, and the clause flags.
- **Capacity figures** in `zte.evaluation.plots`: `capacity_curve`, `capacity_bits_ledger`, `capacity_seed_strip`
  and `capacity_vs_length_oracle`. Each renders an honest placeholder, never an empty axis, when nothing certified.
- **`dataset.raw_align_amplitude`** (default `false`) divides each subject's own RMS voltage out along with their
  covariance shape. Euclidean alignment trace-normalises each trial when fitting the reference but applies the
  whitener to un-normalised windows, so it equalises covariance *shape* and leaves per-subject *gain* untouched:
  two subjects differing only by a 10× gain get whiteners identical to round-off and a post-transform power ratio
  of exactly 100.00. Excluded from the dataset cache key, so flipping it never invalidates a prepared bundle.
- **`--allow-closed-set`** on `zte-run`, the named opt-out of the refusal below.
- **The three-level alignment study.** Three encoders that differ in *which unit the contrastive term pulls at* and
  in nothing else: the pooled sentence vector against a frozen E5 sentence embedding, one fixated word (= one EEG
  token) against a frozen word vector, or four fixed intra-word slices against the LM's own sub-word embeddings. The
  levels are exclusive rather than cumulative — each is the sentence-level CLIP objective plus at most one extra
  term — so `sentence -> word` and `sentence -> token` each flip exactly one lever. Twelve configs at
  `experiments/alignment/{token,word,sentence}/{combined,nr,sr,tsr}.yaml`, every one byte-identical to
  `experiments/ablation/exp16_residual_off.yaml` apart from the weights that switch its level on;
  `word/combined.yaml` is the published champion recipe (26 of 700 held out, 3.714% — stale pending the
  length-projection re-measurement), included as a level rather
  than referenced so the three are a matched triple. Run from `notebooks/alignments/zte_token.ipynb`,
  `zte_word.ipynb` and `zte_sentence.ipynb`.
- **`objective.token_*` — the sub-word alignment level** (`zte.models.objectives.token`). `TokenAligner` scores a
  word's intra-word EEG sub-tokens against the frozen sub-word embeddings of the pieces it spells, and against the
  same piece read by *another person* — the property a cross-subject decoder needs and a single-reader loss will
  skip. `token_weight` and `token_reader_weight` default to 0, so the level is off unless asked for; `token_source`,
  `token_sub_tokens`, `token_temperature`, `token_max_tokens`, `token_max_length` and
  `token_same_subject_negatives` tune it. The frontend path is `RawConformer.sub_tokens` and
  `ZTEModel.sub_token_hidden`; `zte.data.targets.tokens` gains `build_token_alignment` (the word-to-sub-word map,
  built from real character offsets and keyed by `content_id`, so no collate change) and `build_subword_matrix` (the
  frozen embedding of every piece type the corpus actually spells, rather than a modern tokeniser's whole table).
- **The sub-word piece oracle** — `signature_oracle`, `piece_signatures` and `piece_profile_report` in
  `zte.evaluation.audit.rebaseline`, reachable as `zte-rebaseline --piece-oracle`. Measured with the real
  `Qwen/Qwen2.5-0.5B` tokeniser on a 700-sentence corpus matched to ZuCo's statistics (1.463 pieces per word against
  ZuCo's measured 1.4, $H(\text{identity}) = 9.4512$ bits): word count alone carries 4.96 bits and retrieves 33 of
  700; the *total* sub-word piece count — one integer per sentence — carries 5.58 bits and retrieves 62; the two
  jointly carry 8.18 bits and retrieve 359; the per-word piece profile carries 9.44 bits and retrieves 697, and 673
  even after ZuCo's 33% word omission. The best encoder this project has trained retrieves 26. Two consequences are
  enforced in code: **`token_sub_tokens` is a fixed 4 for every word**, so the piece count enters the loss's target
  mask and nothing the encoder computes, and every token-level headline is gated on `piece_profile_report`.
- **`zte-colab sweep plan|next|status`** — the campaign driver. `plan` prints the ordered run list, with the config
  each trains and the run directory each resolves to, on a machine with nothing trained and no Drive mounted;
  `status` and `next` add what has already landed. A run counts as **done** when its `evaluation/metrics.json`
  exists under a search root (the dated Drive sessions first, then the local run root) and never when its `INDEX.md`
  row does, so a run that died between writing its metrics and its catalogue row is not paid for twice. The plan is
  54 runs in three tiers — mechanism (12), power (36), spread (6) — 51 distinct trainings and ~109 GPU-hours, and
  every tier is a complete, reportable table on its own.
- **`train.eval_profile`** (`full` | `sweep`, default `full`). `sweep` keeps embedding health, sentence retrieval,
  the held-out scoreboard and the permutation null — the only numbers allowed to be a headline — and drops the
  neuron, emergence, analogy, seen-vs-novel and frequency-matched blocks, every figure and the interactive
  explorers. Evaluation is the larger half of a run here (61–75 minutes against 36 of training on this project's
  measured Colab timings), so the campaign's arms all carry it; the profile that produced a run is stamped into its
  `metrics.json`.
- **`src/zte/alignment/`** — the cross-level view, above the model stack and holding no `nn.Module`: `atlas` (one
  jointly fitted projection of all three levels, as 2D and 3D plotly figure JSON), `contrastive` (alignment,
  uniformity, effective rank and the positive/negative gap, per level) and `compare` (the cross-level table: hit
  counts, exact binomial tails, rank percentile and the oracle floor).

### Changed

- **`--loso-holdout` is refused on a decoder or joint run whose config named a different split**, rather than
  warned about. `by_subject_loso` shares all 700 stimuli between train and val, so every gallery sentence is also a
  training sentence and the generation verdict fails its `honest_split` clause — the run costs hours and can never
  produce a headline, which is too expensive for a warning that scrolls past in a Colab log. The remedy the message
  names is `train.loso_holdout_subject` in the config; `--allow-closed-set` runs it deliberately as a closed-set
  control.
- **`zte.evaluation.audit.menu` gained a public seam** — `MenuPool`, `menu_pools`, `beaten_in_pool`, `win_prob` and
  `score_menu_flavor` (renamed from `_win_prob` / `_score_flavor`) — so the decoder capacity builds its pools with
  the embedding-side audit's implementation instead of a second copy. Bodies and RNG consumption order are
  unchanged and every existing menu number is byte-identical.
- **One gallery LM pass now feeds both decoder readouts.** Scoring is per-(query, candidate), so every $K$-way menu
  at every $K$ is a column slice of the matrix rescoring already built. The only extra frozen-LM work the capacity
  audit adds is the `length_only` arm, and that is one pass per distinct word count, not per query.
- **The run verdict carries `capacity_certified`, `capacity_k`, `capacity_bits`, `capacity_clauses`,
  `capacity_readout` and `capacity_reason`**, merged additively. `capacity_certified` can never enter
  `generation_above_controls`: a forced choice among $K$ candidates does not license a free-generation headline.
- **`flagship/decode_zte_v2.yaml`, `decoder/decode_v2_pmi.yaml` and `decoder/decode_parallax_nr.yaml` set
  `objective.eval_capacity: true`**, so the three arms that should be certified are.
- Documentation: the capacity method, pool rule, arms and bits ledger in `docs/DECODER.md`; the
  `decoder_capacity.*` path registry and its quoting rules in `docs/EVALUATION.md`; the gain measurement and the
  new knob in `docs/SUBJECT_ALIGNMENT.md`; the capacity readout in `experiments/README.md`.

### Fixed

- **`zte-colab mirror` refused the write mode the session was opened with.** Its `--write-mode` predated `auto` and
  accepted only `local+mirror` and `drive`, so a notebook that opened its session with the new default and handed
  the same value to the mirror died on `argument --write-mode: invalid choice: 'auto'` — after the training cell had
  finished, which is the worst place to lose a cell. Both commands now read their choices from one `WRITE_MODES`
  tuple and default to `auto`. The mode also decides which side of the mirror is local: it is `session.out_root`
  rather than always `res/experiments`, so a session that wrote straight to Drive reports that source and
  destination are one directory instead of copying an empty local tree over the Drive catalogue.
- **The three-level atlas drew one subject, so its contrastive geometry had nothing to measure and the run died
  writing the figure.** `zte-visualize --kind levels` took the *first* `--max-points // 8` sentences of a
  subject-major dataset — on real ZuCo, 500 readings by one person, no two of them the same stimulus. Every level
  therefore had an anchor with no positive pair, reported `None` for its gap as designed, and `contrastive_figure`
  raised `TypeError: float() argument must be a string or a real number, not 'NoneType'` before the atlas JSON was
  written, taking the notebook cell that reads it down with it. Sentences are now drawn a whole stimulus at a time,
  so every subject's reading of a chosen sentence travels with it and the levels carry the cross-subject positives
  the geometry scores. A level that still cannot be scored is named under the figure's title instead of drawn as a
  bar at zero — a zero-length bar claims the term bought nothing, which is a different statement from nothing having
  been measured — and a report in which no level could be scored omits the figure, keeping the per-level report.
- **The three-level atlas embedded every sentence in one forward and died on any GPU.** `zte-visualize --kind
  levels` collated `--max-points // 8` sentences — 500 at the documented `--max-points 4000` — into a single pass,
  and the raw conformer self-attends over its 350-step window for every word token in that batch: a 118 GiB
  allocation against an 80 GB card. It now embeds in chunks of eight sentences, which puts the peak near 2 GiB and
  makes it independent of `--max-points`. Nothing crosses a sentence boundary, so the vectors are the ones a single
  forward produced; `--batch-size` raises the chunk for speed on a large GPU and changes no number.
- **The length projector was fitted in the wrong frame.** `objective.length_projection` fitted its word-count basis
  on the *raw* training rows and subtracted it from the *post-processed* (whitened, all-but-the-top) rows, so the
  basis did not describe the space it was removed from. The real-data signature was length leakage **rising**:
  0.0206 before the projection and 0.3619 after it in `exp16_residual_off`, in the one metric whose entire job is
  to fall. The projector is now fitted on the same post-processed training rows it is applied to. **Every retrieval
  number measured with `length_projection: true` predates this fix and must be re-measured before it is quoted
  again** — the exp16 sweep and the parallax transfer matrix in `docs/RESULTS.md` are marked stale accordingly.

## Parallax phase 3: the reading-level menu, the INDEX merge, and the decoder arms

The 2026-08-16/17 sessions measured real cross-task transfer (NR↔SR rank percentile 0.95–0.97 at every seed,
length-stratified ~0.92–0.93, on a never-seen subject reading never-seen sentences) while the certified exact-length
prototype menu stayed at chance — the discriminative signal lives in individual readings, not centroids. This phase
follows that finding through the stack:

- **The enrolled menu flavor** (`zte.evaluation.audit.menu`) scores each K-way option against the enrolled
  individual readings of its sentence (best reading match, never a centroid), certified with the same exact-length
  pools, losing ties and built-in length-oracle guard as the other flavors. Every enrolled block records the
  reading counts it drew from, and a `gamed` pool can certify no capacity: certification ANDs the oracle verdict,
  and the gamed state travels into `CHAMBER_DATA.json` where the chamber renders it as a server-side amber
  disqualification note. `experiments/decoder/decode_v2_pmi.yaml` joins the decoder tier as the PMI-only matched
  control, so the Phase-3 composite arm's delta decomposes into named factors.
- **The 2-way decomposition diagnostic** (`menu_decomposition` in `PARALLAX.json`) re-scores every diagonal cell
  under {prototype, best reading} × {exact length, ±1 word}, so the menu-vs-percentile gap decomposes into named
  factors. Diagnostic only; it gates nothing.
- **The chart-quality pass** over the study's rendered artifacts.
- **The notebook task-derivation fix**: `notebooks/zte_parallax.ipynb` §3b no longer derives the task list by
  globbing the raw dataset directory — on a fresh runtime that check missed TSR, silently shrinking the §5 transfer
  loop to 2×2 and dropping TSR from the §4 loop. The TSR-involving cells complete on re-run.
- **The experiments INDEX now merges instead of clobbering.** `zte-run`'s catalogue rewrote the local `INDEX.md`
  from local knowledge alone and the Drive mirror pushed it whole over the shared copy, so a fresh VM erased every
  row earlier sessions had written (observed: the overnight TSR rows vanished on 2026-08-17). The catalogue now
  takes the union of the mirrored and local rows keyed by `run_name` — a session can add or update its own runs,
  never erase another session's; an unreachable remote degrades to local-only.
- **The phase-3 decoder configs**: `experiments/decoder/decode_parallax_nr.yaml` (the v2 decoder over the parallax
  NR encoder, `decoder.rescore_pmi: true`, NR gallery) and `decode_parallax_nr_joint.yaml` (`mode: joint` one lever
  further), with the pre-registered expectations in `docs/DECODER.md`.

## Torn cache entries can no longer poison the store

A prepared-bundle cache entry now exists only when it is *complete*. Copies between the local cache and the
persistent Drive store walk files alphabetically, so an interrupted copy could land `meta.json` before the pickles
it describes; the old existence check then treated the torn directory as a warm hit — and, because publish
early-returned on `meta.json`, the poisoned entry was frozen into the store and crashed every later session with
`FileNotFoundError: .../sentences.pkl` (observed on the TSR bundle, 2026-08-17). Every layer now requires all four
required files before counting an entry as present: torn local copies are cleared and rebuilt, torn persistent
entries are reported loudly and repaired by the next publish, and a complete-but-unreadable bundle is discarded and
rebuilt instead of crashing — the checkpoint layer's fall-back-past-a-torn-file discipline, applied to the cache.

## The lens: a single-reading inspection surface

`src/zte/lens/` (`saliency`, `trace`, `page`) and the `zte-lens` CLI walk one reading — one subject reading one
sentence — through a trained checkpoint and show what the model did with it: the thought embedding, occlusion-based
word saliency (each word masked out of the pad mask in turn, scored by the cosine drop of the re-embedded sentence),
occlusion-based channel saliency grouped by montage region (`null`, with an honest note on the page, when the
checkpoint has no montage), the reading's top-k neighborhood in the gallery, and — for decoder checkpoints, via
`zte-lens decode` — the greedy generation with per-prefix-slot occlusion, the word-synchronous evidence weights when
the checkpoint uses them, and the null-prefix control side by side. The artifact is one `lens.json` per reading
(plus a self-contained `LENS.html` with `--html`, built by `zte.lens.page.build_lens_page`), with full provenance.
This is an inspection tool, never an evaluation: every artifact carries and every page renders the disclaimer
"inspection, not a result -- no number here is a headline", the neighbor gallery never contains the query reading
itself, and a non-holdout subject is rendered prominently as a training brain. Driven from §8 of
`notebooks/zte_parallax.ipynb`, with every artifact written to Drive; definitions and the honest-reading guidance:
`docs/LENS.md`.

## The parallax study: three per-task encoders and the cross-task transfer matrix

ZuCo's task is fully confounded with its stimulus set (Cramér V 0.998; no sentence appears under two tasks), and the
measured encoder amplifies the task probe (0.918 against 0.685 raw) — a cross-task contrastive run can win on register.
`src/zte/parallax/` (`study`, `transfer`, `report`, `chamber`) removes the confound structurally: three independent
encoders, one per task, each trained only on its own task's readings with the best-measured recipe
(`experiments/parallax/parallax_{nr,sr,tsr}.yaml`, byte-identical to `ablation/exp17_base.yaml` except `dataset.tasks`
and `run_name`). The prize is the 3×3 transfer matrix: an off-diagonal cell scores a never-seen subject (`ZAB` held
out) reading never-seen stimuli — the strongest generalization cell this project can produce, where a null is a
finding and is reported plainly. `zte-parallax transfer` writes one cell (`transfer.json` + embeddings — stratified
rank percentile with bootstrap CI, the menu-capacity audit, exclusion counts, `postprocess_fit: 'non-holdout
subjects, eval task'`, full provenance); `report` aggregates cells into `PARALLAX.json` / `PARALLAX.md` /
`CHAMBER_DATA.json`, including linear CKA between model pairs on shared readings; `chamber` renders the chamber page —
the three models' PCA-and-Procrustes-aligned views of the same sentences — from the report data and computes nothing.
Free generation is not a parallax deliverable, per-task galleries make chance differ per cell, and no claim enters
`docs/RESULTS.md` without directional consistency across seeds (42/43/44, optionally 45/46). Driven from
`notebooks/zte_parallax.ipynb`; design and artifacts: `docs/PARALLAX.md`.

## De-confounded objective knobs

Two new objective levers, both defaulting off so every existing run stays byte-identical. `within_task_negatives`
makes every sentence-level contrastive denominator task-pure — the CLIP in-batch InfoNCE, the gallery CE (whose
sparse-row fallback becomes drop-the-anchor rather than widen-to-full-gallery), the consensus prototype gallery and
the decoder grounding negatives — because task and stimulus are fully confounded on ZuCo (Cramér V 0.998) and a
cross-task distractor can be rejected on register alone; per-text task labels are joined from ids, never parsed
from vocabulary keys, and a text under two tasks is a loud error. `sentence_variance_weight` /
`sentence_covariance_weight` apply the VICReg variance and covariance terms to the content slice of the pooled
sentence embedding — the tensor retrieval actually scores, which no anti-collapse term previously guarded. The
`exp17_*` ablation family exercises both. Also: a loud warning when the data2vec auxiliary head is silently
disabled on a raw frontend.

## Evaluation-integrity repairs: exclusion, length units, provenance, verdict basis

Four repairs make the honest numbers legible on their own. (1) The scoreboard's retrieval blocks
(`held_out_retrieval`, `decoder_rescoring_retrieval`, `within_task_retrieval` and their `length_stratified` cells)
now **exclude and count** unanswerable queries in `excluded_no_positive` instead of zero-scoring them — forced zeros
over an at-chance remainder read as below chance by construction, which is an artifact, not a measurement. (2) Length
strata use **one unit on both sides**: queries stratify on their stimulus's median word count (the gallery's unit)
rather than the reading's own eye-tracking count, so a reading that skipped words can no longer lose its own truth to
the stratum (`stimulus_median_lengths`, `_lengths_in_gallery_units`). (3) **Provenance travels inside
`held_out_retrieval`**: `postprocess_fit`, `alignment_fit` (`dataset.raw_align_fit`) and `embedding_checksum` — a
short sha256 of the exact sentence-embedding matrix the block measured — are stamped into the block itself. (4) The
machine verdict's `retrieval_above_chance` is now judged on `scoreboard.held_out_retrieval` whenever the split holds
a subject out, with `verdict['retrieval_basis']` naming the basis; the pooled `sentence_retrieval` — which scores
the training subjects' brains alongside the stranger's — can no longer turn the clause green on a LOSO run. The CI
and permutation/phase-control demotion structure is unchanged. `tests/test_eval_integrity.py` pins each behaviour,
mutation-tested.

## Stage-comparable best-checkpoint monitor

In `joint` decoder training the auxiliaries that enter the loss when the encoder unfreezes jump the monitored
validation scalar at the stage A→B boundary; a lifetime best-value comparison therefore locked `best.pt` into a
stage-A epoch whose encoder weights were bit-for-bit the loaded checkpoint, and early-stop patience killed the run a
few epochs into stage B — the joint arm was measuring its own frozen input. The trainer now forgets the best value
and zeroes patience at every stage transition, checkpoints record the stage that produced them (older payloads load
unchanged), and a `decoder`-mode run with `freeze_encoder: false` is refused loudly. The auxiliaries are untouched.

## PMI gallery rescoring

`decoder.rescore_pmi` (default off; existing runs byte-identical) turns decoder-rescoring retrieval into a PMI
score: each candidate's per-token-mean log-likelihood under the query's prefix, minus the same quantity under the
learned null prefix. Every trainable decoder part — the Stage-0 bridge, the train-fitted gap correction, the RVQ
codebooks, the grounding loss — is fitted on train-cell reference texts only, so a train-cell gallery candidate
collects a familiarity bonus the held-out truth cannot receive; the subtraction cancels any candidate-side
constant. The null pass reuses the unconditional branch `null_prefix_prob` already trains, carries no
word-synchronous evidence, and being query-independent runs once per gallery under the `rescore_chunk` memory
bound (`ZTEDecoder.null_rescore` exposes it). With the knob on, the rescoring block records `score: 'pmi'` and a
`pmi_vs_raw` entry — per-query rank percentiles under both scores and the paired delta (PMI minus raw) with a
percentile-bootstrap CI — so the correction's effect is itself measured. `tests/test_pmi_rescoring.py` plants a
per-candidate familiarity bonus in a stub LM and verifies the raw ranking is distorted while PMI recovers it, with
the subtraction mutation-tested.

## Menu capacity: the honest 80% readout

`zte-rebaseline` now ends with the constructive twin of its length audit: **K-way closed-set accuracy** over
training-subject sentence prototypes, swept over K ∈ {2, 4, 8, 16, 32, 64}. The headline flavour is
**length_task_matched**: distractors share the query's task and its *exact* stimulus-level median word count —
exact matching is load-bearing, because at tolerance ±1 the true candidate is systematically the unique best length
match in its own stratum and a pure length code beats chance. Widened tolerances appear only as labelled
`sensitivity` rows no verdict may read; an `open` pool (length legitimately allowed, as in deployment) is reported
beside it. Each accuracy is an exact hypergeometric expectation with chance exactly 1/K, ties counted as losses,
bootstrap CIs throughout, post-processing train-fitted only. Three guards ride inside the block: a per-K
**permutation p** (true label reassigned within the candidate set), a built-in **length-oracle null** that stamps
`gamed: true` if word count alone escapes chance inside a certified pool, and **exclude-and-count** for queries that
cannot field a pool (the zero-scoring convention that once manufactured a below-chance stratified rescoring number
is retired across the audit). The headline is the **certified capacity**: the largest K with CI lower bound ≥ 0.80
and permutation p < 0.05. This turns "the decoder is right 80% of the time" into a pre-registered, confound-guarded
number — the clinical menu size the system can currently serve — and the tracked goal becomes growing it.
`zte.evaluation.audit.menu`, reported in `rebaseline.md`/`rebaseline.json` under `menu`.

## `zte-colab`: the notebook stops importing the package

Colab opens a notebook with its own interpreter, which is older than the `>=3.14` this package requires, so
`import zte` in a cell has never actually been safe. The notebooks did it anyway --- and each import dragged a second
copy of real logic into cells nothing tests: their own run-search order, their own checkpoint resolution, their own
`shutil.ignore_patterns` list deciding what a Drive backup leaves behind, their own re-derivation of a decode's
scores. A notebook cell is the worst place for any of that, because it is the one place a wrong answer is read as a
result.

**`zte-colab` is now the only route in.** Seven subcommands, one question each, every one printing a single JSON
object on stdout with its logs routed to stderr so the stream parses whole: `env` (interpreter, accelerator, device
plan, machine limits), `session` (the dated Drive layout and the environment it exports), `runs` (every run on Drive
and locally, with its checkpoints and its held-out headline), `arms` (the trainable configs, read live off
`experiments/` and labelled by their own header comments), `readings` (one decode's scored readings beside the gate
that judged them), `panels` (the study's charts as plotly figure JSON), and `mirror` (a session between the VM and
Drive). The kernel renders payloads and computes nothing.

The pieces they reach through are ordinary library functions, tested like everything else: `utils/session.py` for the
Drive layout and the run search, `device.device_plan` for what the machine will actually do with a batch,
`utils/env.env_defaults` for the environment a run wants --- returned as *data*, because a notebook's `!` subprocesses
inherit the kernel's environment and the kernel is where those defaults have to land --- `utils/mirror.mirror_tree`
now taking `exclude_files`, so the rotation checkpoints a fresh VM cannot use stay off Drive while `last.pt` always
travels, and `analysis/dashboard.panel_builders`, which the offline page and the notebook now share so the notebook
cannot draw a chart the page does not.

**Two changes are about honesty rather than plumbing.** `zte-colab readings` reads what `zte-decode --out` wrote
instead of decoding a sample of its own: the old cell re-ran `generate()` over twelve readings and tabulated the
result beside a verdict computed from a different, larger set, which is a number the gate never saw. And
`interactive/generation.generation_payload` --- the five-clause AND, promoted from a private helper --- now travels
with every rendering of a generation block, so a control that could not run still **fails** its clause wherever the
block is read. `zte-decode` records `min_prefix_kl` in its provenance for the same reason: the floor is recoverable
from nowhere else in the artifacts, and a clause that cannot be evaluated reads exactly like a clause that passed.

`tests/test_notebook_gateway.py` holds the boundary: no code cell imports `zte`, no `%%bash` cell runs an interpreter
it may not have provisioned yet, every `zte-*` command the notebooks name is a declared entry point, and every
`experiments/*.yaml` path they name exists on disk --- so a promoted config that moves tier breaks a test rather than
the front door.

## The encoder, rebuilt: four mechanisms after the levers ran out

Thirteen arms on 2026-07-25 flipped Euclidean alignment, the subject adapter, identity orthogonality, the text
encoder and the meaning target. They landed between **2 and 9 hits in 700**, with alignment *off* scoring highest,
and two seeds of one unchanged configuration had already produced 4 hits and then 2. Run-to-run noise was the size
of every effect. The exposed levers were exhausted, so this change is architectural.

Four mechanisms, each aimed at one number the 2026-08-13 board actually reported, each defaulting to off, and each
with a matched ablation that flips exactly one field. The maths is in `docs/METHODS.md` §9-12.

**Predictive residual coding (`model.residual_coding`) subtracts what the left context already predicted.** The
variance budget said 8.4% subject, 0.0% content and 91.6% *neither* -- nine tenths of the space spent on
single-trial variability. Reading is predictive and the large language-related EEG deflections are surprisal
responses, so everything unsurprising about a moment of reading -- tonic state, cap impedance, the 1/f background,
the drift of the last few seconds -- is predictable from the preceding words and cancels in a context residual,
while the word-specific response does not. A one-layer causal head predicts each token from its left context and
the token keeps the remainder. **The head regresses a detached target from a detached input**, so no gradient from
its loss reaches the encoder: attached, the encoder could cut that loss by making itself predictable, and a constant
representation drives it to zero. `residual_context_explained` reports how much of a token the context accounted
for, and the falsifiable prediction is that the subject probe falls while the content probe rises -- both falling is
collapse, and `exp16_residual_off` is the pair that decides it.

**Cross-reader consensus (`objective.consensus_*`) trains a reading against what the other eleven readers agreed
on.** ZuCo gives all twelve subjects the same 700 sentences, so every stimulus has twelve noisy measurements of one
latent content vector, and the cross-reader mean suppresses the reader and trial terms as `1/sqrt(n)` while leaving
content untouched. This is not augmentation-based self-distillation: the teacher averages over *different brains
reading the same text*, which is the invariance the project needs and which no augmentation of one reading can
produce. An EMA bank holds one prototype per stimulus, served only once `consensus_min_readers` **distinct
subjects** have contributed. Beside the pull term sits the one that matters more -- a cross-entropy over every
prototype the bank knows, which is **the evaluation moved into the loss**, scored EEG-to-EEG so the modality gap
cannot be what separates the answer from the distractors. The word-level bank is aimed straight at the measured
null: same word, different subject, cosine gap +0.005. The bank is written only in training mode and read before it
is written, so a held-out subject never enters it and never consults it; the one approximation, flagged, is that the
anchor's own earlier passes sit in its prototype with weight bounded by `1 - decay`.

**Length-matched gallery contrast (`objective.gallery_*`) makes counting words worth nothing.** A batch of sixteen
asks the model to beat fifteen distractors; the evaluation asks it to beat 699, and the hardest of those are almost
never in a batch. The frozen text matrix is already resident, so the full denominator costs one matrix product. The
band is the real change: word count carries 5.1422 of the 9.4512 bits of sentence identity and eye-tracking
segmentation hands it over free, so a denominator of same-length texts leaves nothing for it to buy. It is the
training-time counterpart of length-stratified evaluation, which until now could only measure the confound after the
fact. Two guards keep it a loss rather than a formality: an anchor whose band strands it below `gallery_min_candidates`
falls back to the full gallery, because a two-distractor softmax is *small*, not hard; and the anchor's own text is
always in its own denominator, because a cross-entropy with its target column masked saturates at the float floor and
stops reading the model. `gallery_chance` travels with `gallery_top1`, so a band of forty candidates is never quoted
against 1/700.

**Length projection (`objective.length_projection`) removes the length subspace and then measures.** A length-only
oracle at +/-2 words beats every encoder measured here on every top-k, and the decoder's rescoring rank percentile
falls from 0.7244 to **0.4349 -- below chance -- once length is held constant**. Fitted on the training split only,
over the basis `[1, n, log n, 1/n, n^2]`, and reported as `length_leakage_before` / `length_leakage_after` so the
projection has to show it removed length rather than shrinking the vectors. The residual is *not* expected to be
zero: zero would mean the fit saw the scored rows. A projection that cannot be fitted is refused with a reason that
reaches `report.md`, never silently skipped, because a report showing length-free numbers that are not length-free
is worse than no de-confounding at all.

### Two leaks, one closed and one labelled

Both new mechanisms range over the stimulus set, and the retrieval gallery *is* the stimulus set.

**Closed:** `text_vocab` is deliberately whole-dataset so an id means the same sentence in every split, which meant
the full-gallery denominator contained rows for sentences the split had held out. Training against a held-out text
as a *negative* still teaches the encoder where not to map. `GalleryContrast.restrict_to` masks the denominator to
the text ids the training split actually reads, and both the length band and the sparse-anchor widening stay inside
it. The consensus bank never had this problem -- written only from training rows, a held-out stimulus has zero
readers and never clears `consensus_min_readers` -- and that is now asserted rather than assumed.

**Labelled:** a subject-only split holds out people, not sentences, so under `by_subject_loso` every gallery
sentence was in training. That was already true of every arm on the board; what the new terms change is that
separating those exact 700 items becomes *the training objective*, turning the headline into closed-set
identification for an unseen reader rather than open-set retrieval of an unseen sentence. It is a narrower claim,
not a wrong one, and it is now recorded in `metrics['gallery_exposure']` and printed as a **closed-set caveat** in
`report.md`. The open-set claim needs `by_subject_and_stimulus` and its `test` cell.

### One real bug, found by the smoke path

A decoder run inheriting an encoder through `--encoder-ckpt` saved its *own* `model` config while running the
*source* encoder, so the checkpoint described an encoder that was never built and failed to reload. It surfaced the
moment an inherited encoder carried a residual coder the decoder config did not know about, and it would equally
have bitten any architectural field. The run's `model` section is now overwritten from the source before anything is
written.

### Persistence: nothing expensive is more than one epoch from durable storage

`CheckpointManager` stops mirroring the whole checkpoint directory every epoch and mirrors exactly two files
instead: `last.pt`, because `--resume` reads it, and `best.pt`, because it is the result. That is both safer and
cheaper -- the result is on Drive the moment it improves, and per-epoch traffic drops from `keep_last + 2` large
files to at most two, since `mirror_file` skips a file whose bytes already match. The rotation files ride the run
directory's stage mirror; they are history, and a fresh VM cannot use them.

Dropping them from the per-epoch mirror would have thinned the resume fallback, so `load_latest` gained `best.pt` as
its final candidate: `last.pt`, then the epoch files newest-first, then `best.pt`. On a directory restored from
Drive -- which carries no rotation history -- that is the whole safety net, and it turns a `last.pt` torn by the
write that was in flight when the machine went away from a lost run into a lost few epochs.

**A silently failing mirror is worse than a loud one.** `mirror_file` never raises, by design, so a mount that
stopped accepting writes at epoch 3 looked exactly like one that was working right up until the VM was reclaimed at
epoch 40. `_note_mirror` now checks that `last.pt` actually landed and counts *consecutive* failures, escalating at
1, 3, 10 and 30 to an error that names the missing path and says the run is not currently recoverable. Success
resets the count, so one hiccup does not shout for the rest of the run.

The notebook gained `resolve_ckpt()`, which searches this session's Drive folder, then every earlier session
newest-first, then the local disk. A fresh Colab runtime can evaluate, decode or open the studio on a run trained in
a previous session with no manual restore. `durable()` returns the right root for wherever the notebook is running
-- the Drive session folder on Colab, `res/` locally -- so expensive non-resumable outputs are written there rather
than written locally and copied afterwards.

### The decode studio

`zte-studio` writes one self-contained interactive page that answers a question `zte-decode` cannot: not *did it
beat its controls* but *what did it actually do*. A reading picker, a transport bar, and every panel backed by a
real array rather than an illustration -- the scalp field is per-word band power interpolated over the montage at
the word the pointer is on (2-D cap or draggable 3-D head, eight bands), the target sentence carries the pointer's
Gaussian window as it walks, the decoded text reveals token by token shaded by probability with the top-8
alternatives at each step, and the firing panel plots the evidence KL: the same hidden state with and without the
word-synchronous nudge, which is how hard the brain pushed on *that* token.

`FrozenLM.generate_from_prefix` grew an optional `trace` sink to make this possible. Passing one costs a second head
application per step; passing `None` -- which every headline, control and oracle decode does -- leaves the loop
byte-identical. **A visualisation must never be able to change the number it visualises**, and that is
mutation-tested rather than asserted: break the loop so the trace perturbs the emitted token and the paired test
goes red.

The page carries its own warning, and the suite asserts the warning is on it: a handful of readings chosen to look
at is an anecdote, absolute scores mean nothing beside a frozen LM's function-word floor, and the scalp scale is
relative within one reading. Notebook sections 8d and 8e drive it -- a target-beside-hypothesis-beside-controls
table with the paired deltas, then the studio embedded inline.

### Analysis and the notebook

`Trainer` now averages every objective metric over each epoch into `history.json`, and `zte-analyze` plots them as a
**mechanism curves** panel behind a dropdown. That panel is not decoration: `metrics.json` cannot distinguish "the
consensus term did nothing" from "its bank never reached `consensus_min_readers` and contributed exactly zero", and
those are very different findings. Seven panels join the dashboard -- a metric selector over every headline, the
mechanism curves, length leakage before and after, identity against content sized by effective rank, a lever matrix
that catches an "ablation" whose config never differed from its baseline, the bit budget as a pie, and a histogram
of every run against chance.

**`notebooks/zte_colab_v2.ipynb` is new and written from scratch.** Drive-first: evaluation, generation, the analysis
dashboard and the archives are written straight to Drive, while checkpoints go to the VM's fast disk and mirror after
every stage, because a Drive FUSE stall mid-`torch.save` is a torn checkpoint. `RESUME_DATE` points every cell at an
existing session; `WRITE_MODE = 'drive'` puts checkpoints there too. `ipywidgets` pickers choose the arm, the
held-out subject and the seed, and a run explorer walks every collected run's headline block and figures. The
original notebook stays for the benchmark suite, `zte-ablate` and the housekeeping cells.

`scripts/run_zte_study.sh` now defaults its encoder to `zte_encoder_v3` and its ablation set to the five exp16 arms
plus the existing seven.

## The decoder, rebuilt: a metered channel and a pointer that walks the words

The decoder is replaced rather than patched. Two mechanisms, both aimed at the arithmetic that has governed this
project from the start -- sentence identity over 700 ZuCo stimuli needs 9.4512 bits, word count gives away 5.1422
of them free through eye-tracking segmentation, and the encoder has been measured at 1.4965.

**The semantic rate ladder (`decoder.rate_ladder: rvq`) makes the bit budget a constraint instead of an argument.**
The conditioning vector passes through `rate_stages` residual codebooks of `rate_codes` entries, seeded by k-means
on the frozen *text* cloud so a code names a region the LM already writes fluent English from. The channel then
carries at most `stages x log2(codes)` bits by construction, and `bit_budget` in `generation.json` reports the
entropy and the mutual information it actually delivered against the 9.45 it would need. Dead codes are re-seeded
rather than left to silently shrink the real rate below the quoted ceiling. With `rate_length_stage: true`, stage 0
is trained to absorb the word count and the remaining stages are penalised for correlating with it, so
`residual_mutual_information_bits` is the part of the answer the brain supplied rather than the part eye tracking
gave away. `experiments/decoder/decode_v2_no_length_stage.yaml` is the required companion: if the headline's
advantage disappears without the reserved stage, the reserved stage was doing the work.

**Word-synchronous lexical evidence (`decoder.evidence_schedule`) re-injects the brain at every decoding step.** A
pooled prefix spends its influence in the first few generated tokens. A soft monotonic pointer now walks the
reading's word tokens as the LM decodes -- ZuCo tells us which stretch of EEG belongs to which word, for free, on
every reading including a held-out one -- and nudges the LM's final hidden state. Because the frozen output head is
linear, that is exactly a rank-limited additive bias on the token logits: no new vocabulary parameters, and one
decode path. The gate is zero-initialised, so a run begins as the pooled decoder and the evidence earns its way in.
The pointer's walking rate is measured from the tokenised training corpus rather than configured, and rides in the
checkpoint. **Every knob defaults to off**, so `decode_v2_pooled.yaml` reproduces the previous decoder and the pair
attributes the difference to the mechanisms rather than to the rebuild.

**The pointer schedule is content-free, and that is what makes the new control fair.** It depends only on the step
count and the word count, so every brain-independent control inherits it. `length_only` is the control that
follows: it keeps the schedule and the length-conditional mean prefix, and destroys everything else, so a headline
that beats it beat it on lexical content and nothing else. `shuffled_z` joins the ladder too -- an unstratified
derangement of the conditioning vectors after the encoder, which is the "feed it shuffled EEG embeddings" control
stated directly. The five original controls are unchanged.

**One decode path, and it is greedy.** The decode loop is written in `FrozenLM.generate_from_prefix` rather than
delegated to `transformers.generate`, for two reasons: the evidence nudge has to reach the output state at every
step, which no generation hook exposes, and every control must run byte-identical code. Beam search is now refused
outright with the reason, exactly as `cfg_weight != 1.0` already was -- it would be a second decode path, and at
this signal level it raises the language prior rather than the brain signal. `tests/test_decoder_v2_mutation.py`
breaks each of these guarantees on purpose and watches the paired assertion go red.

**Rescoring is memory-bounded.** `target_token_logprobs` now materialises the 151,936-wide vocabulary a block of
positions at a time and reduces immediately, instead of holding a full `(rows, n_target, vocab)` tensor.

## The encoder attacks the word-level content gap directly

Cross-subject word retrieval on real ZuCo sits at Top-1 0.0040 against a chance of 0.0031, and the held-out
`word_len` probe is negative in 13 of 13 runs. Sentence-level transfer is real; word-level content is absent. The
reason is mechanistic -- a sentence-level InfoNCE never asks a single word's EEG to mean that word -- so
`objective.lexical_weight` and `objective.lexical_reader_weight` now demand it. The first scores each word's EEG
token against that word's frozen text embedding; the second scores it against **the same word position read by
another person**, with the negatives restricted to the anchor's own subject so subject identity separates nothing.
The projection they train is the one the decoder's evidence path reads, which is what ties the two halves together:
`experiments/flagship/zte_lexical_raw.yaml` is the encoder for `experiments/flagship/decode_zte_v2.yaml`. Both
weights default to 0, and `experiments/ablation/exp14_lexical_off.yaml` is the matched pair.

## Fix: the content probe reported "no signal" for a target it could not have fitted

The scoreboard's positive control read `R2 = -0.005` for word length from raw band power and concluded the probe
could not detect content. It could not -- but not for the reason recorded. `linear_probe` used a fixed
`Ridge(alpha=1.0)`, which on a standardised design of `p` features and `n` rows is barely regularised at all, so a
target the representation genuinely does not carry returns an out-of-sample `R2` of about `-p/n`. At 525 band-power
features over 108k words that is -0.005: the number was the estimator's overfitting penalty, not a measurement.

`linear_probe` now searches the ridge penalty over a log grid, which puts the no-signal floor back at 0 -- verified
directly: on pure noise the old estimator returned -0.131 and the new one returns -0.002. It also accepts `groups`
for grouped cross-validation, and `residualise` removes a nuisance factor's per-group mean so lexical content can be
probed *within* subject, which matters here because subject identity is linearly readable from raw band power at
0.81 while word length is not, so a pooled ridge spends its capacity on who is reading.

The positive control now asks two questions instead of conflating them. **`machinery`** probes word length from the
eye-tracking features that carry it by construction, so a failure there is a fault in the probe; **`pooled`** and
**`within_subject`** probe band power, and a failure there with the machinery passing is a result about the signal.
A shuffled-target column gives the empirical zero. A raw-signal run has no band power to probe, and that is now
reported as such rather than falling through to an unreliable proxy.

## Within-task pools, and every headline as mean +/- sd

**Within-task retrieval** (`scoreboard.within_task_retrieval`, `decoder.within_task_pools`) re-ranks each query
inside its own reading task. No ZuCo stimulus appears under more than one task -- the confound audit puts Cramer's
V(task, stimulus) at 0.998 -- so the full gallery lets a model score by telling SR sentences from NR ones, which is
a passage-set property. Inside one task the passage set is fixed. The pool is smaller, so its own chance level and
interval are reported beside every number.

**`zte-analyze`** is a new CLI that walks one or more experiment trees and writes the study analysis: a
self-contained interactive HTML page with plotly inlined so it opens from a Drive mirror with no network, the tidy
CSV tables behind every panel, and a Markdown summary. It aggregates over seeds (mean, sd and a bootstrap interval),
over LOSO folds, and over single levers -- including the feature-ablation table of raw conformer vs band-power MLP,
spherical harmonics vs standard channel indexing, and the invariance recipe on vs off. Panels cover the headline,
the fold spread, the length-oracle confound, the within-task pools, the control ladder, where each decode landed
sentence by sentence, the words the decoder emits against the words it was asked for, the measured bit budget, the
probe heatmap, the who-versus-what split, the learning curves and a 3-D electrode map. A synthetic run is named as
synthetic on the page, in the summary and in the terminal.

**`scripts/run_zte_study.sh`** runs the whole thing in one resumable command: the confound audit, the flagship
encoder at several seeds, the decoder and its one-knob arms over that encoder, the feature-ablation table, the
length-confound audit against every checkpoint, and the analysis. `SMOKE=1` rewrites every config into an
offline-safe copy first, so the wiring check needs no download. Section 6d of the notebook drives it, and Section 9c
is the analysis.

## The 2026-08-13 board: two demotions, one no-op, and a metric that can actually rank arms

Re-scoring the flagship set against Drive (held out on `ZAB`, 700 queries) settled three things and unsettled one.

**Rank percentile is the only metric here that survives a seed change.** Two seeds of `zte_raw_aligned` give 0.9672
and 0.9670; their Top-1 moves 9 hits to 8. Every arm comparison this project has made on Top-1 was made on noise, and
the tier tables now quote rank percentile with its bootstrap CI.

**`flagship/` drops to three, and they are tied.** `zte_raw_aligned` 0.9672 (0.9635–0.9708), `clip_e5_meaning_raw`
0.9667 (0.9629–0.9705), `clip_e5_raw` 0.9635 (0.9599–0.9673) — overlapping intervals, no champion. `clip_e5_raw`
keeps its place on the strength of being the only arm whose length-stratified Top-1 clears *p* < 0.05.
`zte_raw_aligned_wide` (0.9523, interval disjoint from all three, stratified Top-1 *below* chance) and
`clip_e5_meaning_raw_v2` (never scored on this board; its one honest number is 2 hits in 700 where chance expects 1)
move to `archive/`, with the numbers that retired them recorded there.

**The exp12 alignment stack does not do anything measurable.** `ablation/exp12_align_off` matches the full stack to
four decimal places on rank percentile, effective rank and the subject probe. `zte_raw_aligned` stays in `flagship/`
because it is tied for best measured, not because its levers earn their place.

**Capacity retained is not capacity used.** The two retired arms hold the healthiest geometries on record
(effective-rank 0.45 and 0.53 against the retained set's 0.24–0.25) and the worst retrieval. The existing warning
that a *low* effective rank can mean invariance bought by destroying capacity now has its converse measured.

**The decoder gets a matched second arm rather than a swap.** `experiments/decoder/decode_frozen_aligned.yaml` is
`decode_frozen_e5raw.yaml` with one knob moved — `exp12_zte_raw_aligned` underneath instead of `exp8_clip_e5_raw` —
so a tie the encoder board cannot break is broken on decoder rescoring instead of by assertion. Three dataset/model
levers travel with that encoder (`raw_align: euclidean`, `subject_signature`, `model.subject_adapter`), and they are
named in the config header because a frozen encoder handed features built under the other recipe silently sees a
scale it never trained on. The notebook now resolves both source checkpoints and pairs each arm with its own.

## Fix: the frozen decoder LM inherited its precision from the checkpoint

`AutoModelForCausalLM.from_pretrained` defaults to `dtype='auto'`, which loads a checkpoint at whatever precision it was exported in. `Qwen/Qwen2.5-0.5B` ships bf16, so on `transformers` 5 the frozen LM came up in bf16 while the bridge emitted float32 prefixes, and the first decoder forward pass died with `mat1 and mat2 must have the same dtype, but got Float and BFloat16`. Under an older `transformers`, whose default was float32, the same code ran — so this was a library upgrade silently changing the numerics of a frozen component, and the crash was the lucky outcome.

**`decoder.lm_dtype` now follows the encoder.** It defaults to `auto`, which reads the precision off the encoder the bridge is fed by, so the two halves of the pipeline cannot end up at different precisions by configuration or by library default; `float32`, `float16` and `bfloat16` pin it instead, and a pinned value that disagrees with the encoder is honoured with a warning naming both. The *resolved* value travels in `FrozenLM.provenance()` alongside `lm_revision` and the tokenizer fingerprint, because a token log-probability produced at one precision is not comparable with one produced at another. `FrozenLM.assemble` casts the soft prompt to the frozen embedding's dtype — the one point at which the float32 bridge and the LM have to meet — and log-probabilities are still read back in float32, so a half-precision LM halves its memory without putting a half-precision dynamic range into the optimiser. The `meaning` dependency group requires `transformers>=4.56.0`, where `from_pretrained(dtype=...)` replaced the since-removed `torch_dtype`.

Every encoder on Drive is float32, so `auto` resolves to float32 and the decoder now matches it by construction rather than by coincidence.

`lm_dtype` pins the weights, not the arithmetic. `Trainer` autocasts, so a CUDA training step still runs the frozen LM's matmuls in bf16 under `train.precision: auto`, while evaluation does not autocast and runs at `lm_dtype`. `manifest.json` already records `precision` and `autocast_dtype`; [`docs/DECODER.md`](docs/DECODER.md) now states the consequence, which is that a run's teacher-forced training loss is not bit-identical to the one `zte-decode` reports for the same checkpoint.

**Nothing already on Drive is affected**, verified rather than argued. `experiments/decoder/decode_encoder_only.yaml` run synthetically before and after this changeset gives a byte-identical `metrics.json` and byte-identical weights across all 99 tensors of both `best.pt` and `last.pt`; the extraction and prepared-bundle cache keys of all six flagship and decoder configs are unchanged, because that key is derived from `DatasetConfig` alone and every change here is in `DecoderConfig` or below. The token-cache schema bump reaches only decoder runs — `build_target_tokens` is called solely from `_wire_decoder` and `ZTEDecoder`, both guarded on the decode objective. `WordResampler`'s new parameter and `ZTEDecoder.from_checkpoint`'s new refusal apply only to decoder checkpoints, of which Drive holds none.

## Fix: the decoder was never taught to stop

`_encode` appended no end-of-sequence token on the HuggingFace path, and Qwen2's tokeniser adds none of its own — so on `Qwen/Qwen2.5-0.5B`, the LM in every real decoder config, the last supervised token of `"He was a good man."` was `.`. Every EOS-valued cell in the matrix was padding, padding is masked out of the cross-entropy, and the bridge therefore received **exactly zero gradient toward emitting EOS**. Free-running decode ran the full `max_new_tokens: 96` on every row against a 19.6-word reference: WER exceeds 1 by construction, BLEU's brevity penalty never fires, and every precision-based metric collapses — while the decode itself costs about four times the tokens it should.

The offline suite could not see it. `TinyByteTokenizer.encode` *does* append EOS, and every decoder test runs `lm_source: tiny`, so the whole suite exercised a stop token production did not have — the §15 trap exactly. `_encode` now pads both tokenisers through one loop that terminates every row that fits, and a stub-tokeniser test covers the HuggingFace branch offline. **The token cache key gains a schema version**, because it is keyed by corpus and tokeniser and neither moves when the encoding rule does; without the bump the EOS-less `tokens_*.npz` already published to Drive would be reused under the new semantics.

## Fix: the `pooled_plus_words` ablation could not train at all

`compute` builds a `prefix_slots + word_slots` prefix and handed it to `dropout_null`, whose null spans `prefix_slots`. `experiments/decoder/decode_words_ablation.yaml` pairs `conditioning: pooled_plus_words` with `null_prefix_prob: 0.1` at batch 16, so the arm died inside the first step or two with a broadcast error. No test reached it: the suite constructed that bridge but never ran a step through it.

The shape was only half of it. Replacing the pooled slots alone would have left the word slots — and with them the brain, and the word count that carries 5.14 bits — inside the branch that is supposed to be independent of both, quietly falsifying the "$p = 1.0$ makes the loss independent of $z$" property the `null_prefix` control rests on. `WordResampler` now carries its own learned null (769,664 parameters, up from 762,496), `dropout_null` raises on a null narrower than its prefix rather than broadcasting, and the independence test is parametrised over both conditioning arms.

## Fix: three ways a refused generation verdict could still read as a win

None of these could move `verdict['generation_above_controls']` itself — mutation testing confirmed the gate's five-clause AND is closed. They are the places a reader meets the number *before* the gate.

- **The interactive generation page** — which `docs/DECODER.md` calls the most persuasive artifact a run produces — led with "Beats every brain-independent control" off `beats_all_controls` alone. It checked three clauses of five, and could not check the prefix-influence floor at all because `min_prefix_kl` was never in its payload. A block with permutation *p* = 0.42 and KL = 0.0001 rendered with the warning band off. The page now calls `generation_verdict` directly, leads with the full AND, and names a failing permutation or KL clause in the warning band.
- **`zte-decode`'s console summary** printed the raw `beats_all_controls`, which counts only controls that produced a delta — so a skipped control read as a win. It now reports the same composition the gate uses, warns when a control could not run, and says so at the point the control is dropped rather than staying silent.
- **The rescoring line quoted an unstratified Top-1** with no mention of stratification, against §5's standing rule. It now labels that number `UNSTRATIFIED`, prints the length-stratified cell beside it, and warns when no stratified cell was computed.

Alongside those: `generation_permutation_test` hard-coded the upper tail, so a lower-is-better `primary_metric` scored *p* = 1.0 on a perfect decode while `paired_delta` called the same evidence a win — the direction now follows the metric. `paired_delta` drops non-finite rows and could reach `beats: True` on a single surviving sentence, since a one-element bootstrap returns the point estimate as its own bound; it now needs the same `n >= 4` floor its block does. `ZTEDecoder.from_checkpoint` refuses a checkpoint whose `gap_correction` is not `none` but carries no fitted statistics, instead of silently installing a pass-through corrector while provenance claimed a correction had been applied. And `zte-decode`'s provenance now records `gap_fitted`, `gap_n_fit`, `postprocess_fit` and which train-fitted transforms were restored.

## Fix: decoder scoring, device and doc corrections

- **Rescoring materialised ~7 GiB of transient per chunk.** `target_token_logprobs` built a float32 `log_softmax` over the full 151,936-token vocabulary for 64 rows × 96 target positions; a fused `cross_entropy` computes the identical quantity (max abs difference 5e-7) without either large temporary. `next_token_logits` also projected all twelve prompt positions to keep one, four times per training step — it now asks for one.
- **`GapCorrector` whitening could not run on MPS**, which has neither float64 nor `_linalg_eigh`. The eigendecomposition moves to CPU and back; it is a once-per-run fit, so the round trip is free. Latent until now because every shipped config uses `mean_scale`.
- **The per-step `prefix_kl` was not the quantity it shares a name with.** It compared each row against its batch *neighbour*, and hard-negative batching seeds a batch from one sentence and fills it with that sentence's own readings — where a healthy bridge should score near zero. It now partners each row with a row of a different stimulus, which is what verdict clause 5 measures.
- **Stage 0's leak guard passed vacuously** on a split that shares stimuli between cells (`by_subject_loso`, `by_task`): the holdout set is empty, nothing intersects it, and the bridge memorises all 700 references. It now says so, loudly, naming it as the outcome most likely to be mistaken for success.
- `assemble` moves the scaffold buffer to the prefix's device rather than assuming co-location; `_build_causal_lm` catches the `TypeError` an older `transformers` raises on the `dtype` keyword; and the `ImportError` message named a `decoder` dependency group that does not exist (`transformers` lives in `meaning`).
- **Three doc claims contradicted the code.** `docs/DECODER.md` stated verdict clause 5 as the KL against $P_\text{null}$ — the version whose own docstring says a collapsed bridge can clear it while ignoring the brain entirely; the code has always used another reading's prefix. `docs/DECODER.md` and `docs/TRAINING.md` both said `run_training` "branches away before the objective is built" for `mode: encoder`; it does not, and the decoder wiring keys off the objective rather than the mode.

## Fix: `missing.method: iterative` silently imputed column means

`IterativeImputer` is experimental in scikit-learn and raises `ImportError` unless `sklearn.experimental.enable_iterative_imputer` is imported first. That import was present in [`data/features/missing.py`](src/zte/data/features/missing.py) but **commented out**, so `_fill_sklearn` took its `except ImportError` branch, logged *"scikit-learn unavailable; falling back to column mean"* on a machine where scikit-learn was installed and working, and imputed column means instead of running the model-based imputer.

No committed config selects `iterative` — every one uses `mask_only` or `linear` — so no result on the board is affected. `test_missing_methods_fill_all_nans` passed throughout because it asserts only that no NaNs remain, which the wrong fallback also satisfies; it is a test that would not have failed had the behaviour been removed.

## exp13 — text out, on a 227k-parameter leash (and the length confound that governs it)

The decoder stage. `train.mode: decoder` loads a trained encoder, freezes it, freezes `Qwen/Qwen2.5-0.5B`, and trains **only** a 226,560-parameter prefix bridge between them — LayerNorm, a rank-128 map, per-slot FiLM, and a learned null prefix. There is no LoRA and no LM fine-tuning in any mode, which is the point: 700 ZuCo sentences cannot be memorised into weights that are never updated, so "the output is corpus recall" becomes a checkable claim rather than a matter of trust. `mode: encoder` is the pre-decoder pipeline unchanged, and `experiments/decoder/decode_encoder_only.yaml` exists to keep it that way.

- **The finding that came first, and outranks the feature.** On the real 700-stimulus SR+NR gallery, `H(identity) = 9.4512` bits and `H(identity | n_words) = 4.3090`, so **sentence length alone carries 5.1422 bits of sentence identity** — more than the ~4.7 bits the published encoder is credited with. ZuCo's word segmentation comes from eye tracking, so the width of `pad_mask` *is* the word count and the model gets it free. A length-only oracle at ±2 words scores Top-1 0.0214 / Top-5 0.0786 / Top-10 0.1371 / MRR 0.0672 against the best encoder's 0.0143 / 0.0457 / 0.0886 / 0.0427: the encoder's whole Top-k profile is matched by knowing length to ±2 to ±4 words. Only `rank_percentile` resists (0.9617 vs 0.9494 at ±1). **`zte-rebaseline`** reports the 3×2 grid (post-processing in {none, train-fitted, transductive} × gallery in {full 700, length-stratified}) against that floor plus the bit budget, runs against checkpoints already on Drive with no retraining, and gates nothing.
- **The published 0.9617 was measured under transductive whitening.** `report.py` fits `whiten_features` and `all_but_the_top(1)` over all 12 subjects including the held-out one. Label-free, so a soft leak rather than label leakage — but a decoder sees one sentence at a time and cannot reproduce it, so the train-fitted column is what the decoder actually inherits. Both are now reported side by side.
- **An honest split, with every cell's axis named.** `by_subject_and_stimulus` partitions the 700 stimulus keys under a fixed seed and intersects that with the LOSO subject mask, returning four cells: `train` (5,775 readings), `val` (seen subject, unseen stimulus — model selection), `test` (**unseen subject × unseen stimulus, 105 readings — the only headline cell**) and `test_seen_stim` (unseen subject, seen stimulus — diagnostic). The stimulus permutation is seeded independently of the subject mask, so all 12 folds hold out the same texts and pool.
- **Free-running generation is the secondary, expected-null readout.** No reference length, no candidate set, `cfg_weight` asserted `1.0` so the headline and every control run byte-identical code. `zte-decode` scores it against five brain-independent controls — `mean_prefix` (absorbs the Stage-0 text prior), `null_prefix`, `phase`, `noise`, and a **length-stratified** `mismatch` derangement — plus a true-text-embedding oracle. `_verdict['generation_above_controls']` is an AND over: honest split, no candidate set, paired bootstrap CI above zero against *every* control, permutation *p* < 0.05, and mean prefix-influence KL ≥ 0.05 nats. Metrics (BLEU, ROUGE, WER, content-word F1) are pure stdlib + numpy — no metric package is a dependency.
- **The powered readout is retrieval, and is labelled retrieval.** Decoder rescoring over the 700-sentence gallery is ~9.5 bits of forced choice at 700 queries, against a generation delta at *n* = 105; it lands in `scoreboard.decoder_rescoring_retrieval` with a length-stratified sub-block. `teacher_forced_ppl_DIAGNOSTIC` and `forced_choice_RETRIEVAL` are computed, stored, and **provably unread** by the verdict — `strip_quarantined` removes any `*_DIAGNOSTIC` / `*_RETRIEVAL` key at any depth and `_verdict` re-applies it to whatever it is handed.
- **Silent-correctness fixes the decoder would otherwise have inherited.** `ZTEConfig.from_dict` derived its section list from `dataclasses.fields` instead of hard-enumerating four, so a `decoder:` block no longer round-trips to nothing. A frozen-encoder run **restores** the source checkpoint's normaliser and aligner rather than refitting them (refitting does not crash a frozen encoder; it just hands it a scale it never trained on). `extra['subject_vocab']` holds the subject map again rather than the 700-sentence text vocab. `FrozenLM.state_dict()` returns `{}`, so a 0.5B LM does not add ~1 GB to every epoch checkpoint. And `train.early_stop_patience` exists, because every run on record bottoms out its validation loss at epoch 5–6 of 40.
- **`train.seed` never reached the encoder's initial weights.** `Trainer.__init__` seeds, but `run_training` builds the model and the objective *before* constructing it, so initialisation drew from an unseeded global generator and two runs of one config gave different losses — verified on both this tree and the previous release. `run_training` now seeds first. Encoder-mode history is byte-identical across runs, and byte-identical to the pre-decoder pipeline once both are seeded, which is what `experiments/decoder/decode_encoder_only.yaml` and `test_encoder_mode_is_reproducible_under_a_fixed_seed` exist to hold.
- **Three modes, one trainer.** `parameter_groups` gives the bridge `train.bridge_lr` and the encoder `bridge_lr × encoder_lr_scale`; groups are structural so a resume whose freeze state differs cannot break the optimiser state. With the encoder frozen and in eval, `z` is a pure function of the reading, so a `(n_readings, 768)` cache keyed by `reading_id` skips the raw conformer entirely after the warm-up pass — 25.8 MB, and what makes 12 folds × 3 seeds affordable. `cache_embeddings` must be off in `joint` mode, where the encoder moves.
- **New configs** under [`experiments/decoder/`](experiments/decoder/): the headline frozen arm, the staged joint arm, the `mode: encoder` regression control, the Stage-0 and `pooled_plus_words` ablations, the rebaseline audit arm, and an offline `smoke/decode_tiny_mps.yaml` (`lm_source: tiny` builds a 22,688-parameter LM locally, `run_name: smoke_mps`, always `--synthetic`, never a result). Method, controls, verdict gate and the pre-registered expectations: [`docs/DECODER.md`](docs/DECODER.md).
- **The notebook can run the decoder.** [`notebooks/zte_colab.ipynb`](notebooks/zte_colab.ipynb) gains **Section 8**: the length audit against the source encoder (`zte-rebaseline` — trains nothing, gates nothing), the bridge training run over a frozen `--encoder-ckpt` resolved from Drive, a scorecard that leads with the verdict's five clauses and the decoder-rescoring *retrieval* before it shows any generation number, and the five arms of §8d. The former Sections 8–11 shift to 9–12.
- **`--loso-holdout` no longer silently un-does the decoder's split.** It forces `by_subject_loso`, which shares all 700 texts between train and val — the one configuration in which a decoder recites the corpus, and the one `_verdict` refuses to headline. `zte-run` now warns when the flag replaces a non-encoder run's split, and points at `train.loso_holdout_subject` instead. Encoder runs, whose north-star split it *is*, are unaffected.

## Fix: three notebook cells that could not have shown a correct number

All three predate the decoder and all three are in [`notebooks/zte_colab.ipynb`](notebooks/zte_colab.ipynb).

- **The spotlight comparison table read a scoreboard key that has never existed.** `scoreboard.cross_subject_holdout_retrieval` is the *function*'s name; the block it writes is `scoreboard.held_out_retrieval`. Every held-out column was therefore empty — while the pooled `sentence_retrieval` Top-1 sitting beside them, the number that crowned the wrong champion, was populated. The table now reports the held-out block only, with Top-5 as a hit count and its exact binomial tail.
- **The run picker resolved every run's path and then ignored it.** `run_dirs()` finds runs Drive-first, but `show_run_figures` rebuilt the path as `res/experiments/<run>` — empty on a fresh runtime, which is the same class of bug `run_dirs()` was introduced to fix.
- **The comparison bar chart raised `KeyError` on every run.** It still asked for `retrieval_top1` / `eff_rank_ratio`, the columns the scorecard dropped when it switched to the honest held-out headline. It now plots `rank_pct` and `eff_rank`.

## Fix: raw runs were being OOM-killed, silently

Every raw-conformer run on a standard Colab runtime died between `[1/4] Preparing dataset` and `[2/4] Training`, with no error — the notebook printed `done` having trained nothing. The raw bundle is **23.6 GB when materialised** (160k words x 105 channels x 350 samples), against ~12.7 GB of RAM, so the kernel OOM reaper killed the process; `!uv run …` swallows the exit code, so the loop moved on.

- **Raw EEG is now memory-mapped.** It is saved to its own uncompressed `raw_eeg.npy` beside `arrays.npz`, and `load` maps it — only the windows a batch touches become resident. A compressed `.npz` member cannot be mapped; it must be inflated in full before a single window can be read.
- **Existing bundles upgrade themselves, without a big-RAM session.** An `.npz` member *is* an `.npy`, so `_extract_raw_member` copies the decompressed stream byte-for-byte into place: peak allocation ~138 MB regardless of array size, and no re-processing.
- **The alignment pass no longer allocates the whole tensor.** `raw[mask][::stride]` materialised a full copy *before* subsampling (measured: 2205 MB for a 2.21 GB array — ~23.6 GB at real scale) on top of the resident array. Covariances now stream in chunks over pre-subsampled indices, and whitening writes into its own memmap: peak allocation ~590 MB (fit) and ~216 MB (transform), independent of dataset size.
- **Whitening runs on the GPU** when one is available (`torch.matmul` on CUDA/MPS, numpy otherwise) — a large batched matmul that was idling the accelerator while straining system RAM.
- **Failures are visible again.** The notebook's `run_zte` helper checks exit codes, names the runs that did not complete, and calls out exit 137 as an out-of-RAM kill with the fix. A new `show_resources()` prints RAM/GPU/disk up front.

## Fix: notebook explorer cells read the wrong location

Section 5 writes runs to `{DRIVE_DIR}/experiments`; Section 6b trains under `res/experiments` then mirrors. The scorecard, run picker, deep-dive and `zte-visualize` cells all hard-coded `res/experiments`, which is empty on a fresh runtime — so they showed nothing regardless of how many runs existed. A single `run_dirs()` helper now resolves both locations, Drive first, deduped. The scorecard also switched to the honest held-out headline (rank percentile + CI, Top-5 hit counts with an exact tail, effective rank) instead of the pooled Top-1 that crowned the wrong champion.

## Fix: a warm Drive cache no longer re-prepares (or re-unzips) on every Colab session

Every command re-did expensive work on a fresh runtime even when the dataset was already sitting in the persistent Drive store. The ZuCo folder on Drive holds the task **`.zip` archives** (~63 GB), so "resolving the data source" meant unpacking tens of gigabytes onto the VM — and every CLI did that *before* consulting the bundle cache.

- **The raw source is now resolved lazily, everywhere.** Cache keys exclude the data root, so `zte-run`, `zte-train`, `zte-explore`, `zte-benchmark` and `zte-prepare` all key their config, ask the store, and resolve (unzip / download / synthesise) only what is genuinely missing. A warm run logs `Processed bundle already persistent; skipping raw-data extraction.` and never touches the archives. Shared helpers: `resolve_root_if_needed` / `bundle_is_cached` in `cli.support.sources`.
- **`zte-run` set its cache location *after* resolving the root**, so the probe had nowhere to look; the two are now ordered correctly.
- **Frozen encoder artifacts layer onto the persistent store too** (`BundleStore` gains `fetch_artifact`/`publish_artifact`, mirroring to `<remote>/_artifacts/`). The contextual BERT meaning matrix, the E5/BGE sentence embeddings and the provisioned GloVe file are content-addressed but were cached only on the ephemeral disk, so every session re-ran BERT and E5 over the whole corpus.

Measured on a zip-only root with the local cache, extracted tree and run directory all wiped: **3.4 s end-to-end including training, with the archives untouched.**

### `zte-prepare --configs` specifically

Two further causes, both fixed:

- **It rebuilt what it had just found.** The loop computed `status = 'cached' if store.find(key) ...` and then called `ZuCoDataset(cfg).build()` **unconditionally** — staging each bundle down from Drive and loading it into RAM in full, only to discard it. Cached entries are now skipped outright; staging happens lazily, in the first run that actually needs that dataset.
- **It resolved the raw data root before checking anything.** Cache keys exclude the data root, so every config is now keyed *first* and the `.mat` tree is resolved (or downloaded, or synthesised) only for entries genuinely absent. On a warm store the command never touches the raw data at all and finishes in under a second.
- **`BundleStore.has()`** reports which layer holds an entry (`local` / `persistent` / `None`) without copying, so a presence check costs one `meta.json` stat rather than a multi-GB download. `zte-prepare --check` reports the whole board and builds nothing.
- **The notebook's "already prepared" sentinel is gone.** It lived at `res/cache/prepared/.prepared.done` — on the disk Colab wipes between sessions — so it never fired where it mattered; and a sentinel that *did* survive would silently skip preparing the dataset for any newly-added config. The command is now cheap and authoritative, so asking the store replaces guessing.

Covered by `tests/test_cache.py` (warm-store no-op, `--check` builds nothing, and cache keys proven independent of the data root — the assumption the deferred resolution rests on).

## exp12 — cancel the brain, keep the meaning (and re-score the board honestly)

Every run on Drive was re-scored on the **held-out subject** instead of the pooled set. Pooled retrieval includes the 11 training subjects, so it rewards memorising them rather than reaching the 12th — and it had crowned the wrong champion. The band-power arm `clip_e5_meaning` (headline Top-1 0.043, *pooled*) lands **4 hits in 700** held out, and an identical re-run landed 2. On the same fold the raw conformer lands **32 at Top-5, *p* ≈ 7e-16**. Band power's low subject probe (0.23) was never disentanglement: its raw features only score 0.16, and its effective-rank ratio of 0.160 shows the 768-d space had collapsed to ~123 directions. Invariance had been bought by destroying capacity.

- **The band-power family is retired.** `clip_e5_bandpower`, `clip_e5_meaning`, and the whole E5/Qwen/BGE/MPNet text-encoder A/B moved to `experiments/archive/`, each with the number that retired it (`experiments/archive/README.md`). `scripts/run_suite.sh` and `scripts/run_loso.sh` now default to the raw conformer; the `text_ab` study and the `text_encoders/` tier are gone.
- **Euclidean alignment for the raw path** (`dataset.raw_align`, `zte.data.features.alignment`). `dataset.normalize` only ever applied to band power, so `normalize: riemannian` was a **silent no-op** for every raw run on the board — the winning arm had no cross-subject alignment of any kind. Each subject is now whitened by the inverse square root of their own mean channel covariance (He & Wu, 2019). Label-free, so `raw_align_fit: all` legitimately covers the held-out subject; `train` is kept as the strict ablation.
- **A subject adapter driven by inferred statistics, not an ID lookup** (`model.subject_adapter`, `zte.models.subject`). `subject_film` indexes an `nn.Embedding` by subject id, so under LOSO the held-out subject has no row and its zero-init entry makes the adaptation the *identity map* for the only person under test — the mechanism was structurally inert where it mattered. The adapter instead reads that person's covariance-geometry **signature** (141-d for ZuCo-105: per-channel log scale + the log-Euclidean tangent vector of the region-averaged covariance) and a hypernetwork emits their per-electrode gain and FiLM affine from it. A stranger is an interpolation in signature space rather than a missing row, so `ZTEEmbedder.calibrate_subject(baseline_raw=...)` registers a brand-new brain from one short unlabelled recording — no labels, no retraining.
- **Identity orthogonality** (`objective.identity_orthogonality_weight`). A gradient-reversal adversary asks that identity be *unpredictable*, which a collapsing encoder achieves for free. This term instead decorrelates the content subspace from the signature (normalised cross-covariance / linear CKA), so a full-rank identity-free space pays nothing. Scale-free, so shrinking earns no credit. The adversary is halved to 0.05 as a second referee.
- **The honest headline is no longer a rate.** Held-out retrieval has ~700 queries at 1/700 chance, so Top-1 expects **one** hit — "0.006 vs 0.001" is three hits and noise. The scoreboard now leads with **rank percentile + a 95% bootstrap CI** (every query contributes) and reports Top-K as **hit counts against chance with an exact binomial tail**.
- **New configs**: `flagship/zte_raw_aligned.yaml` (exp12, the champion candidate — exp10's encoder byte-for-byte plus the stack) and `zte_raw_aligned_wide.yaml` (the same on the v2 encoder), with four one-knob ablations under `ablation/exp12_*.yaml`.
- **Cache-safe and backwards compatible.** Alignment is applied *after* the bundle loads, and the new fields are excluded from the cache key, so enabling it never invalidates a prepared bundle. All three knobs default off; with no signature the adapter is not constructed and the forward path is unchanged. Fitted maps ride in the checkpoint so inference reproduces training exactly. `docs/SUBJECT_ALIGNMENT.md`, `tests/test_subject_alignment.py` (13 tests).

## Fix the collapse/cone, chase meaning, and measure it honestly

Adds the anti-collapse, meaning-over-stimulus and honest-evaluation improvements, plus the interactive/experiment infrastructure to run and read them. All new knobs default off; the flagship recipe configs (`exp6_skipgram_eegonly_invariant`, `study_invariance_full_loso`) turn them on.

- **Dimensional collapse / the "cone".** `objective.whiten` ZCA-whitens the exported embeddings at evaluation — centring removes the shared direction that *is* the cone, dropping anisotropy from ~0.998 to ~0.00, and every downstream metric is honestly recomputed on the whitened space. `objective.anisotropy_weight` adds a Wang & Isola uniformity term (a mean-direction penalty is a saddle at a perfect cone and can't break it — pairwise repulsion can). New A/B: `study_anticone_off/on.yaml`. Covered by `tests/test_collapse.py`.
- **Kill the stimulus shortcut, chase meaning.** `objective.meaning_positives` draws skip-gram positives from the *same content word in different sentences* (subject/context-agnostic word identity), not only the same stimulus token. `objective.stimulus_adversary_weight` adds a second gradient-reversal referee that predicts *which passage/task* a token came from (sized by `model.n_tasks`). The data layer gained per-token `word_id` and per-sentence `task_id`.
- **Report on truly held-out data.** The LOSO sweep (`scripts/run_loso.sh`, `zte-run --loso-holdout`) evaluates every held-out subject; the evaluation `honesty` block adds a permutation null, a held-out cross-subject decoder, and an anchor-calibration lift for the held-out subject (`zte.evaluation.audit.honesty`, `tests/test_honesty.py`). Portable + auto-GPU + resumable, with `docs/RUNNING.md` and `notebooks/zte_colab.ipynb` (Colab via `uv`).

## Interactive views: comparison dashboard + explorer overhaul

- **`zte-compare`** builds one offline HTML comparing every catalogued run (scorecard matrix, sortable CI table, per-run cards, transparent best-run rubric).
- The **Thought-Space Explorer** was redesigned (icon mode cards, capability strip, insight card, progressive disclosure) and gained **Sentence** (per-reader word-by-word path), **Meaning** (same meaning across everyone), and **Calibrate** (snap a new brain in from anchor words, live Procrustes) modes plus a "remove reader identity" morph; the **Neuron Atlas** gained a scalp head-map; the classic word explorer was restyled to match.

## Pause & resume for long runs

Any run is now interruptible and continuable:

- `Trainer` gains `resume=True`: on a pause it restores model / optimiser / scheduler / AMP-scaler / objective + EMA-teacher / best-metric / history / step from `last.pt` and continues at the next epoch.
- `Ctrl-C` (SIGINT) or `kill` (SIGTERM) pauses cleanly — the last completed epoch is already checkpointed — instead of crashing.
- `zte-run --resume` makes the whole pipeline idempotent: it reuses the cached dataset bundle (skips the slow prepare), resumes training from `last.pt`, and skips evaluation / exploration that are already up to date (re-evaluating automatically if training advanced). `--force` redoes completed stages. `scripts/run_suite.sh` now passes `--resume` on every run, so stopping and re-running the suite continues where it left off. Covered by `tests/test_resume.py` (continuation, no-op, and a real SIGTERM interrupt->resume).

## Interpretability & experiment suite — peering inside ZTE

A follow-on wave focused on *explaining* the representation and running it properly.

### Neuron-level interpretability (`zte.evaluation.neurons`)

Every evaluation now emits a per-dimension "neuron" report — which dimensions fire, what each one encodes, and which matter vs. are negligible:

- **Importance** — per-neuron std and its share of total embedding variance, ranked most-active to dead (near-constant), with an active/dead count. The collapse story at neuron resolution.
- **Selectivity** — for every neuron, |Pearson r| with word length / log-frequency and eta² with subject / task / category; each neuron's *dominant* attribute is its argmax.
- **Who-vs-what budget** (the headline) — the share of the space's *variance* whose dominant attribute is identity (subject) vs content, with a `who_vs_what_ratio`. Quantifies the "encodes who, not what" failure mode per neuron.
- **Exemplars & attribution** — the words that most/least activate each top neuron, plus a correlational band × scalp-region attribution tying a neuron back to the brain (and exposing gaze-driven neurons when eye-tracking is on).
- Artifacts: full `neurons.json`, a compact `neurons` block in `metrics.json`, a "Neurons — what the dimensions encode" section in `report.md`, and an interactive **`neuron_atlas.html`** (ranked importance chart coloured by dominant attribute with the active/dead threshold line; per-neuron selectivity, activation histogram, top-firing words, and scalp attribution) — auto-emitted per run and buildable with `zte-visualize --atlas`.

### Emergent-property metrics + explorer overhaul (`zte.evaluation.emergence`)

Answering "do similar thoughts cluster across people?" — the north-star property — as a measured number, not a picture:

- Every run reports `metrics.emergence`: cross-subject **same-word** and **same-meaning** (category) clustering (same-pair cosine vs random baseline, with the honest *gap* since collapsed spaces make all cosines high), and **neighbourhood coherence** (are a token's nearest neighbours the same word / category, and how many come from a different subject). A plain-language verdict (`clustered` / `weakly` / `not`) plus a `report.md` section.
- The **Thought-Space Explorer** was rebuilt for interpretability: a "What am I looking at?" guide, three headline **verdict banners** (now showing the authoritative full-space emergence numbers with the in-browser PCA figure as a live estimate), an **auto-analogy leaderboard** that finds the working `A->B` analogies for you (no more guessing a word/subject), and a **semantic-neighbourhood** view.
- Neuron importance is now explicit and adjustable: `neurons.json` documents the exact formula (`var_share[d] = std[d]² / Σstd²`) and adds per-target `importance.rankings` (`variance`, `selectivity:subject`, `selectivity:word_len`, …) so you can rank neurons by importance *to a chosen attribute*, not just by how much they fire.

### Rigorous experiment suite (`docs/EXPERIMENTS.md`, `scripts/run_suite.sh`)

A curated, bias-controlled set of studies that maximises the dataset (all 12 subjects, SR+NR), evaluates on held-out data, and isolates each lever:

1. purpose / eye-tracking confound,
2. a LOSO subject-invariance A/B (baseline vs the full invariance stack),
3. an anti-collapse VICReg ablation,
4. an objective sweep via `zte-benchmark`,
5. band-power vs raw-conformer. All runs use leakage-aware splits (`by_stimulus` / `by_subject_loso`), train-only normalisation, VICReg + dropout + weight-decay regularisation, and multiple seeds so differences carry bootstrap CIs. `docs/EXPERIMENTS.md` gives exact commands and a "how to read every output" guide.

## Anti-collapse, subject-invariance and honest-evaluation levers

Turns the honest negative result (well-instrumented but dimensionally collapsed and subject-dominated) into concrete levers, and adds an interactive Thought-Space Explorer. Each change is documented in code and covered by tests (`tests/test_improvements.py`, plus additions to `tests/test_evaluation.py`).

### Anti-collapse (the biggest metric mover)

- Added a **VICReg variance-hinge + covariance penalty** on the exported embeddings, wired through a shared `_ObjectiveBase` so every objective (skip-gram, CBOW, masked, CPC) gets it (`models/objectives.py::vicreg_terms`, `_ObjectiveBase.regularize`).
- Knobs: `objective.variance_weight`, `objective.covariance_weight`, `objective.variance_target`.
- The variance term keeps each of the 768 dimensions "alive"; the covariance term decorrelates them (raising effective rank). This directly targets the ~15-of-768 dimensional collapse.

### Stop the model learning "who"

- **Cross-subject positives** (`objective.cross_subject_positives`): skip-gram can draw positives from the *same stimulus read by different subjects* using a new subject-agnostic per-token `content_id`. A `StimulusBatchSampler` (`data/torch_dataset.py`) co-locates the same sentence across subjects in a batch so those positives actually exist; the training pipeline routes the train loader through it automatically when the flag is on.
- **Gradient-reversal subject adversary** (`objective.subject_adversary_weight`, `models/heads.py::SubjectAdversary`/`gradient_reverse`): an auxiliary head tries to read the subject from the token hiddens; the reversed gradient trains the encoder to hide subject identity.
- **Per-subject normalisation** (`dataset.normalize='zscore_subject'`, `data/transforms.py::FeatureNormalizer`): removes the constant per-subject offset; serialises per-subject statistics and falls back to global pooled stats for unseen subjects at inference.

### Fix the masked objective and the eval paths

- The exported 768-d projection head is now **trained** under the masked objective (both latent and reconstruct variants predict/reconstruct *through* `model.project`). Previously it received no gradient, so exp2 exported a random projection.
- The data2vec teacher target is normalised **across tokens with a variance floor** (`_normalize_across_tokens`, `objective.teacher_variance_floor`) instead of a per-token LayerNorm — this is what stops teacher/student co-collapsing to a constant (the exp2 cone).
- The teacher **EMA decay is ramped** from `objective.ema_decay` to `objective.ema_decay_end` across training (data2vec schedule), driven by the trainer passing the global step to `post_step`.
- **Objective-aware inference routing** (`models/embedding.py::embed_sentence`, `inference/embed.py`): sentence/word embeddings now follow each objective's *trained* path — skip-gram/CBOW skip the transformer, CPC uses a causal mask, masked uses the bidirectional context.

### Evaluate on held-out data; leakage-aware splits and normalisation

- `train.test_fraction` now defaults to **0.1** (held-out evaluation is the norm, not in-sample).
- New **`by_stimulus`** split groups by normalised sentence text across subjects, so the same sentence never spans train and test (unlike `by_sentence`).
- The normaliser (and imputer) are **fit on the train split only** via `ZuCoDataset.refit_normalizer`, called by the pipeline after the split — no val/test/held-out-subject leakage. `dataset.normalizer_fit='all'` restores the legacy whole-dataset fit.

### Input features

- Default eye-tracking scalars **drop `SFD`** (≈60% missing, equals FFD where present).
- FFD/GD-locked band power is available by setting `dataset.band_power_measures` (e.g.  `('FFD','GD','TRT')`) — the feature machinery is fully general over measures.
- The **raw-EEG Conformer path (exp5)** is verified to run end-to-end (masked reconstruction through the trained projection head).
- **EEG-only is the honest headline**: new `exp6_skipgram_eegonly_invariant` preset excludes eye-tracking and turns on every subject-invariance lever with a `by_stimulus` held-out split.

### Tighten the evaluation and clean the config

- **Bootstrap/permutation confidence intervals + effect-size floors** replace the sign-only `beats_noise` / retrieval / subject-arithmetic verdicts (`evaluation/metrics.py::bootstrap_ci`, `evaluation/report.py::_verdict`).
- **`task_transfer` NaN bug fixed**: the analogy content id no longer embeds the field being transferred across; genuinely disjoint SR/NR stimuli are reported as not-applicable rather than a bare NaN.
- **Query-weighted retrieval chance** so the "×chance" multiple is computed consistently with how hits are scored (the type-weighted value is retained under `chance_top1_typeweighted`).
- Probe cross-validation is now **shuffled and scaled** (`KFold`/`StratifiedKFold(shuffle=True)` + `StandardScaler`), so probe R² magnitudes are trustworthy (direction was already correct).
- **Electrode montage**: `dataset.montage_csv` loads a real montage for scalp-region importance; without one, region claims are flagged and softened as an "approximate region proxy".
- **Dead knobs removed**: `objective.n_negatives` and `objective.reduce_omitted_weight` (which misrepresented what ran) are deleted from the config. Old YAMLs carrying these keys still load (unknown keys are ignored).

### New — Interactive Thought-Space Explorer

- `evaluation/interactive.py::thought_space_explorer_html` and the `zte-visualize` CLI produce a single self-contained, offline Plotly HTML with live controls for: one subject / many words; many subjects / one word (with a cross-subject cosine stat); **thought arithmetic** (`emb(t,A) − centroid(A) + centroid(B) ≈ emb(t,B)`, drawn as an arrow with the nearest-neighbour hit); an **eye-tracking with/without** toggle; and real-time colour/subject/word/2D-3D/view switching.

### Experiment presets

`experiments/exp1..exp6` regenerated against the new schema: VICReg on every run; exp2 with the masked fixes; exp4 as the LOSO subject-invariance flagship (adversary + cross-subject positives + per-subject norm); exp5 raw-conformer; exp6 the EEG-only, everything-on, `by_stimulus` headline.
