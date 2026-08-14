# Changelog

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
