# Running ZTE — locally or on Google Colab

Everything here works the same on **Apple Silicon (MPS)**, **Linux/servers (CUDA)**, and **Google Colab (GPU)**.
The accelerator is chosen automatically — `--device auto` is the default, and it picks **CUDA -> MPS -> CPU**.
Every long run is **resumable**: stop any time (`Ctrl-C`) and re-run the same command to continue exactly where it left off.

> **Python 3.14 is required.** The project is [`uv`](https://docs.astral.sh/uv/)-managed; `uv` provisions the right Python
> for you on any machine (including Colab, which ships an older Python).

---

## Quick start (local machine)

```sh
# from the repo root
uv sync --group all          # provisions Python 3.14 + installs torch, viz, TensorBoard, gdown

# 1) Smoke test on synthetic data — no dataset needed (a couple of minutes)
uv run zte-run --config experiments/flagship/zte_raw_aligned.yaml --synthetic --epochs 3

# 2) A real run from a local folder of extracted .mat files
uv run zte-run --config experiments/flagship/zte_raw_aligned.yaml --root res/data/zuco_extracted
```

Which device did it pick?

```sh
uv run python -c "from zte.device import resolve_device; print(resolve_device('auto').name)"
```

Each run writes a self-contained folder under `res/experiments/<run>/` with `metrics.json`, figures, a `report.md`, and interactive HTML (`evaluation/interactive/thought_space_explorer.html`, `neuron_atlas.html`). The run name comes from `run_name` inside the config rather than its file path, so `experiments/flagship/zte_raw_aligned.yaml` writes to `res/experiments/exp8_clip_e5/`.

---

## The LOSO “new brain” experiment

Leave-one-subject-out is the decisive generalization test: train on N−1 people, evaluate on a person the model has **never seen**. `scripts/run_loso.sh` rotates the held-out subject over the whole cohort so one data point becomes a **trend**, and it is fully resumable.

```sh
# Fast synthetic dry-run first (validates the whole sweep on CPU, no data)
SMOKE=1 bash scripts/run_loso.sh

# The real sweep (auto-GPU, multi-hour). Point it at your extracted .mat folder:
bash scripts/run_loso.sh res/data/zuco_extracted

# Also run the no-recipe control arm (a clean A/B per subject):
CONTROL=1 bash scripts/run_loso.sh res/data/zuco_extracted
```

**Resume / interrupt.** Press `Ctrl-C` any time. Re-run the *same* command: finished subjects are skipped and the interrupted one continues from its last checkpoint — nothing is recomputed. This is safe on a flaky connection or a preemptible/Colab runtime.

Useful knobs (environment variables):

| var                            | meaning                                                                  |
| ------------------------------ | ------------------------------------------------------------------------ |
| `SMOKE=1`                      | synthetic dry-run on CPU (2 epochs)                                      |
| `CONTROL=1`                    | also run the baseline (no-recipe) arm for each held-out subject          |
| `DEVICE=cuda` (or `mps`/`cpu`) | force a device instead of auto-selecting                                 |
| `SUBJECTS="ZAB ZDM"`           | restrict the held-out set                                                |
| `SEEDS="42 43 44"`             | repeat each fold at N seeds for a mean/std (averages out training noise) |
| `EPOCHS=40`                    | override training length                                                 |

When the sweep finishes it writes `res/experiments/loso/LOSO_SUMMARY.md` (the honest held-out trend, via `zte-loso-summary`) and `COMPARE.html`. **Read the held-out number, not the pooled `sentence Top-1` in `INDEX.md`** — see [`EVALUATION.md`](EVALUATION.md). Aggregate any existing sweep on its own, and ask which brains it encodes well and why:

```sh
uv run zte-loso-summary --experiments res/experiments/loso   # honest held-out trend + convergence spread
uv run zte-encodability  --experiments res/experiments/loso   # per-subject encodability + what predicts it
```

---

## Google Colab

Open **[`notebooks/zte_colab.ipynb`](../notebooks/zte_colab.ipynb)** in Colab (there's an *Open in Colab* badge at the top). Pick a **GPU runtime** (`Runtime -> Change runtime type -> GPU`) and run the cells top to bottom. The notebook: installs `uv` + Python 3.14, confirms the GPU, runs a synthetic smoke test, runs the LOSO sweep (synthetic by default; switch to real Drive data in one cell), and renders the comparison view inline. Because every step is resumable, a disconnect just means re-running the last cell.

**Section 8 is the decoder stage.** It runs `zte-rebaseline` against the source encoder first (the length-confound audit, which trains nothing), then `zte-run` over the frozen encoder with `--encoder-ckpt`, then tabulates the verdict, the decoder-rescoring retrieval and the generation delta against the five controls. Its configs already name `train.loso_holdout_subject` inside `by_subject_and_stimulus`, so **`--loso-holdout` must not be passed to them** — it forces `by_subject_loso`, which shares every stimulus between train and val, and `zte-run` warns when it does. See [`DECODER.md`](DECODER.md).

### Surviving a lost Colab runtime (continuous Drive backup + resume)

Colab runtimes are ephemeral, so a long real run needs its checkpoints on Drive. Two options, both fully `--resume`-safe (a completed run is skipped instantly; an interrupted one continues from its last checkpoint):

- **Train local + live Drive mirror (recommended — fast, reliable I/O).** `--drive-backup <mounted-drive-folder>` copies each run's `best.pt`/`last.pt` to Drive **after every epoch** (best-effort; a Drive hiccup won't crash training). In the suite: `DRIVE_BACKUP="/gdrive/.../experiments" bash scripts/run_suite.sh <data>`. After a reset, copy the Drive checkpoints back to local `res/experiments/` (the notebook's `restore_from_drive()` does this) and re-run with `--resume`.
- **Write straight to Drive (simplest).** Point `--out-root` (or `OUT_ROOT=`) at a mounted Drive folder; every checkpoint, config and metric lands on Drive as it's written. To resume, just re-run the same command with the same `--out-root` and `--resume` — no restore step needed.

---

## Re-running a finished step

`--resume` covers training. Everything *after* it — the confound audit, the decode, the length rebaseline, a
parallax transfer cell — skips itself on a re-run when the artifacts on disk were built from the same inputs, so a
notebook can be re-run top to bottom and only the unfinished work costs anything.

`zte-audit`, `zte-decode`, `zte-rebaseline` and `zte-parallax transfer` each write a hidden record beside their
first artifact (`.zte-done-<artifact>.json`) holding what it was built from: every option they were given, the
checkpoint's SHA-256, and the dataset's bundle key. The next invocation compares, and rebuilds unless everything
matches:

| What moved since the artifact was written              | On the next run                                    |
| ------------------------------------------------------ | -------------------------------------------------- |
| nothing                                                | skipped, and its headline is logged again from disk |
| the checkpoint — another epoch, another arm            | rebuilt                                            |
| any option — a seed, a control, a tolerance, a split   | rebuilt                                            |
| the data — different tasks, subjects, representation   | rebuilt                                            |
| an artifact deleted, or half-written by a killed cell  | rebuilt                                            |
| only the path the raw data was read from               | skipped                                            |
| `--force`                                              | rebuilt                                            |

Two of those rows carry the reasoning. **The raw path is deliberately not part of the identity**: the data is keyed
by its bundle key, which excludes location, so one recording keys identically whether it was read from
`/content/drive/...` or from a local extract, and a re-mounted Drive does not invalidate a day of audits. And **an
artifact with no record is rebuilt rather than trusted** — one written before this existed, or copied in from
somewhere else, cannot say what produced it, and serving a stale number as a fresh one is the failure this exists
to prevent.

The record mirrors to Drive with the artifacts it describes, so a fresh runtime reaches the same answer without
re-measuring anything. The guard decides before the dataset is staged and before any model is loaded, which is
where the minutes go: a skipped decode costs a checkpoint hash.

---

## Combine and compare all runs

```sh
uv run zte-compare                                   # scans res/experiments/, writes COMPARE.html
uv run zte-compare --experiments res/experiments/loso # just the LOSO sweep
```

`COMPARE.html` is one offline page: a pass/fail scorecard across every run, a sortable metric table with confidence intervals, links to each run's own explorer/atlas/report, and a transparent “best run” ranking.

---

## What the evaluation now reports (honesty layer)

Beyond bootstrap CIs against analytic chance, every run's `metrics.json` (and `report.md`) now includes a `honesty` block, computed whenever ≥ 2 subjects are present:

- **Permutation null** — cross-subject retrieval Top-1 vs a *label-shuffled* empirical null -> a p-value, not just an analytic chance line. Over $B$ shuffles with scores $s^{\ast}_b$ against the observed $s$, the p-value is $p=\dfrac{1+\lvert \lbrace b : s^{\ast}_b \ge s \rbrace \rvert}{1+B}$.

- **Held-out cross-subject decoding** — train a linear probe on N−1 subjects, score it on the held-out subject (one fold per subject); the honest test of whether content transfers to a new brain.
- **Anchor calibration lift** — fit an orthogonal Procrustes map from a few shared *anchor* words that aligns a held-out subject into the shared frame, then measure whether same-word cross-subject cohesion improves on held-out words. A metrics-side preview of “can we snap a new brain in without retraining?”.

For a LOSO run these are reported for the held-out subject specifically (the sweep sets the held-out subject via `--loso-holdout`, which `metrics.json` records under `honesty.loso_holdout`).

---

## Cloud -> local hand-off (`zte-pack`)

Train on a powerful cloud GPU, zip the finished runs, download, and run inference locally. Heavy `cache/`, `tb/` and `bundle/` folders are excluded by default — a checkpoint already embeds the input shapes and fitted normaliser, so **inference needs only the checkpoint**.

```sh
uv run zte-pack list                                   # runs with sizes + completeness
uv run zte-pack zip --all --best-only --out ~/zte.zip  # smallest archive: just best.pt per run (inference-only)

uv run zte-pack zip --all --out ~/zte.zip              # + last.pt/epoch checkpoints (also resumable)

uv run zte-pack zip exp6 --with-bundle                 # one run, incl. dataset bundle (to re-evaluate offline)

uv run zte-pack zip --all --move --out ~/zte.zip       # zip, then delete the local run dirs (free space)

uv run zte-pack unpack ~/zte.zip --dest res/experiments # on your machine
uv run zte-pack delete colab_exp6 --yes                # delete one run (omit --yes for a dry run)
uv run zte-pack clean experiments cache --yes          # wipe res/ subtrees to free space (or: clean all)
```

**On Colab**, mount Drive (`from google.colab import drive; drive.mount('/gdrive')`), read the dataset directly with `--root "/gdrive/My Drive/Sharables/ZuCo Dataset"` (no `zte-download` needed), and point `zte-pack zip --out` at a Drive folder to upload the archive straight to Drive — or set `OUT_ROOT=/gdrive/... bash scripts/run_loso.sh` to write runs to Drive as they finish. See **[`notebooks/zte_colab.ipynb`](../notebooks/zte_colab.ipynb)** for the full end-to-end flow.

**Extraction is selective (only what you need).** When `--root` points at a folder of `.zip` archives (e.g. on Drive), ZTE reads each zip's index and extracts **only the `.mat` files matching the run's `tasks` / `subjects`** — the task and subject are parsed straight from the `results<SUBJECT>_<TASK>.mat` filename. So a run with `tasks: [SR, NR]` extracts only `task1 - SR` and `task2 - NR` and never unpacks `task3 - TSR` (or unrelated archives like `scripts.zip`), on Colab **and** locally. Already-extracted files are reused, not re-unpacked, unless you pass `--overwrite`. A folder that already holds extracted `.mat` files is used in place.

**Housekeeping.** `zte-pack clean <targets> --yes` removes `res/` subtrees (`experiments data cache benchmark explorer embeddings`, or `all`); dry-run without `--yes`. To pull critical updates on Colab, delete the checkout and re-clone (`rm -rf zte && git clone …`) — your data and saved runs on Drive are untouched.

## GPUs and TPUs

`--device auto` (the default) selects **CUDA -> Cloud TPU (`torch_xla`) -> Apple MPS -> CPU**. Colab **GPU** is the primary, tested path (bf16/fp16 AMP on CUDA). **TPU** is best-effort: install `torch_xla` on a TPU runtime and `auto` will pick it (the trainer calls `xm.mark_step()` per step); without `torch_xla` a TPU runtime falls back to CPU rather than erroring.

## `zte-colab` — the notebook's bridge

Colab's kernel is an **older interpreter than the `>=3.14` ZTE requires**, so `import zte` in a notebook cell is a `SyntaxError`. It never needs to. `zte-colab` exposes every capability a notebook wants as a subcommand that prints **one JSON object on stdout**, with its logs routed to stderr so the whole stream parses with the standard library:

```sh
uv run zte-colab env                                        # interpreter, accelerator, device plan, machine limits
uv run zte-colab session --drive "/gdrive/My Drive/Sharables/ZTE"
uv run zte-colab runs    --drive <root> --experiments res/experiments --headline
uv run zte-colab arms    --kind encoder                     # trainable configs, read live off experiments/
uv run zte-colab readings --from <zte-decode --out dir>     # the scored readings and the verdict that gates them
uv run zte-colab panels  --experiments <roots> --out <dir>  # the study charts as plotly figure JSON
uv run zte-colab sweep   plan|next|status --drive <root>    # the campaign, what landed, and what to train next
uv run zte-colab mirror  --drive <root> --direction up      # move a session between the VM and Drive
```

| subcommand | answers                                                                               | the notebook uses it for                            |
| ---------- | ------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `env`      | which interpreter, accelerator and machine, and the environment every run wants       | §2 wiring the kernel, §3 the hardware report        |
| `session`  | where this dated session reads and writes on Drive                                    | §4, and the env vars every later `!uv run` inherits |
| `runs`     | every run on Drive and locally, its checkpoints and its held-out headline             | `find_runs()`, `resolve_ckpt()`, §10c               |
| `arms`     | which configs are trainable, labelled by their own header comment                     | the §7a and §8 dropdowns                            |
| `readings` | one decode's readings, scored, beside the five-clause verdict                         | §8d                                                 |
| `panels`   | the study's charts, drawn once, as figure JSON                                        | §10b                                                |
| `sweep`    | the campaign as an ordered plan, which runs already landed, and the next one to train | the alignment notebooks' §7                         |
| `mirror`   | what moved between the VM and Drive, and what was deliberately left                   | §11, `mirror_to_drive()` / `restore_from_drive()`   |

`env` returns the environment as **data** rather than applying it: a notebook kernel's `!` subprocesses inherit the kernel's environment, so the kernel is where those defaults have to land. Each one fixes a failure that is silent rather than loud — a `module://` matplotlib backend that crashes a headless subprocess, a block-buffered stdout that makes a multi-hour run look hung, a CUDA allocator that fragments on the few very large blocks a raw-EEG batch asks for.

`tests/test_notebook_gateway.py` enforces the boundary: no code cell imports `zte`, no `%%bash` cell runs an interpreter it may not have yet, every `zte-*` command named is a declared entry point, and every `experiments/*.yaml` path named exists on disk.

Outside a notebook, `zte.utils.bootstrap()` remains the in-process equivalent — it applies the same defaults, resolves the project root and creates the `res/` output directories.

## Analysing a study

```sh
# Everything under one or more trees -> one offline page, its CSV tables and a Markdown summary.
uv run zte-analyze --experiments res/experiments --out res/experiments/analysis \
    --montage res/montage_gsn105.csv

# A Drive mirror and the local tree together; a run present in both is read once.
uv run zte-analyze --experiments "/gdrive/My Drive/Sharables/ZTE/2026-08-14/experiments" res/experiments
```

`ANALYSIS.html` inlines plotly, so it opens on a machine with no network — which is the point, since it is meant
to be read from a Drive mirror. `tables/*.csv` carries every frame behind it. Section 9c of
`notebooks/zte_colab.ipynb` runs the same thing and renders the panels inline.

## The Colab notebooks

Two notebooks, and they are not versions of each other.

**[`notebooks/zte_colab_v2.ipynb`](../notebooks/zte_colab_v2.ipynb)**
([open in Colab](https://colab.research.google.com/github/victor-iyi/zte/blob/main/notebooks/zte_colab_v2.ipynb))
is the current front door: written around the exp16 encoder and the v2 decoder, Drive-first, with `ipywidgets`
pickers for the arm, the held-out subject and the seed. Everything durable is written to Drive — evaluation,
generation, the analysis dashboard, the studio page and the archives go straight there, while training checkpoints
go to the VM's fast disk and are mirrored after every stage, because a Drive FUSE stall mid-`torch.save` is a torn
checkpoint. Set `WRITE_MODE = 'drive'` to put checkpoints on Drive too. Set `RESUME_DATE` to an existing session
folder and every cell resumes that session instead of starting a new one.

Its `resolve_ckpt()` searches this session's Drive folder, then every earlier session newest-first, then the local
disk, so a fresh runtime can evaluate, decode or open the studio on a run trained in a previous session with no
manual restore. `durable()` returns the right root for wherever it is running — the Drive session folder on Colab,
`res/` locally.

**No cell imports `zte`.** Every capability arrives through the `colab()` helper defined in §2, which runs one
`zte-colab` subcommand and returns its JSON. A code cell may import only the standard library, `IPython.display`,
`google.colab`, and Colab's own `pandas` / `plotly` — enough to render a payload and nothing more. That keeps the
search order, the mirror exclusions and the verdict arithmetic inside the package, where they are tested, rather
than drifting in a notebook where they are not.

**[`notebooks/zte_colab.ipynb`](../notebooks/zte_colab.ipynb)**
([open in Colab](https://colab.research.google.com/github/victor-iyi/zte/blob/main/notebooks/zte_colab.ipynb))
is the original, kept for the sections v2 does not carry: the benchmark suite, `zte-ablate` grid generation, the
per-run explorer views and the housekeeping cells.

Both need a GPU runtime (`Runtime → Change runtime type → GPU`) and both provision Python 3.14 through `uv` in their
first cell, because Colab's system interpreter is older than ZTE requires.

## The three alignment notebooks

`notebooks/alignments/` holds one notebook per **alignment level** — the unit the contrastive term pulls at:
[`zte_token.ipynb`](https://colab.research.google.com/github/victor-iyi/zte/blob/main/notebooks/alignments/zte_token.ipynb),
[`zte_word.ipynb`](https://colab.research.google.com/github/victor-iyi/zte/blob/main/notebooks/alignments/zte_word.ipynb) and
[`zte_sentence.ipynb`](https://colab.research.google.com/github/victor-iyi/zte/blob/main/notebooks/alignments/zte_sentence.ipynb).
The levels are **exclusive, not cumulative**: each is the sentence-level CLIP objective plus at most one extra term,
so `sentence -> word` and `sentence -> token` each flip exactly one lever.

| Level      | The unit it aligns               | Frozen target                       | Configs                           |
| ---------- | -------------------------------- | ----------------------------------- | --------------------------------- |
| `sentence` | the pooled sentence vector       | an E5 sentence embedding            | `experiments/alignment/sentence/` |
| `word`     | one fixated word = one EEG token | a frozen word vector                | `experiments/alignment/word/`     |
| `token`    | four fixed intra-word slices     | the LM's frozen sub-word embeddings | `experiments/alignment/token/`    |

Each level has four arms: `combined.yaml` trains SR+NR together, and `nr.yaml` / `sr.yaml` / `tsr.yaml` train one
reading task alone so the parallax transfer matrix has its vantage points. All twelve are byte-identical to
`experiments/ablation/exp16_residual_off.yaml` apart from the weights that switch a level on
(`objective.lexical_weight` / `lexical_reader_weight` for `word`, `objective.token_weight` / `token_reader_weight`
for `token`, all four at zero for `sentence`), so a `diff` between two arms names the levers that move and nothing
else. `word/combined.yaml` is the published champion recipe, included as a level rather than referenced so the three
are a matched triple scored under one evaluation profile.

Nothing forces an order between the three notebooks, and they write into the same dated Drive session.

### One prepared bundle serves all three levels

`ZuCoDataset._cache_key` hashes the `dataset` block alone, minus the fields that say *where* or *whether* to cache
and the ones applied after a bundle loads (`montage_csv`, `raw_align*`, `subject_signature`). The `objective` block
never enters it, and the three levels differ only in `objective`. So the levels never build a second bundle: the
four bundles the study needs correspond to its four task sets — SR+NR, NR, SR and TSR — not to its three levels.
Build all four in one pass, on Drive, before training anything:

```sh
uv run zte-prepare --root "<ZuCo Dataset>" --configs experiments/alignment \
    --cache-dir res/cache --cache-remote "/gdrive/My Drive/Sharables/ZTE/prepared"
```

`--configs` takes a directory and processes each **distinct** dataset once, so pointing it at the whole study is
both the cheapest and the safest thing to run: an arm whose dataset block ever drifts from its siblings shows up
here as a fifth bundle instead of as a surprise multi-GB `.mat` parse in the middle of a training cell. §5 of
whichever notebook you open first pays that cost; on a cache hit the other two are about fifteen seconds.

### Driving the campaign — `zte-colab sweep`

The study is a campaign, not a cell you press once, and `sweep` is what makes it survivable:

```sh
uv run zte-colab sweep plan   --levels token
uv run zte-colab sweep status --levels token --drive "/gdrive/My Drive/Sharables/ZTE" --out-root res/experiments
uv run zte-colab sweep next   --levels token --drive "/gdrive/My Drive/Sharables/ZTE" --out-root res/experiments
```

`plan` needs nothing trained and no Drive mounted — it prints the ordered list of runs with the config each trains
and the run directory each resolves to. `status` and `next` add the one thing that cannot be planned: what has
already landed. **A run counts as done when its `evaluation/metrics.json` exists** under one of the search roots
(the dated Drive sessions first, then the local run root), and never when its `INDEX.md` row does — a run that died
between writing its metrics and its catalogue row is finished, and keying doneness on the catalogue would spend its
hours a second time. `next` names the first run with no metrics, so a reclaimed VM re-plans, skips what landed and
picks up where the last session stopped.

The plan's order is a contract. Tiers run `mechanism -> power -> spread`, and within a tier the alignment level
varies fastest, so a campaign interrupted anywhere has finished the earlier tiers outright and leaves a matched
three-level comparison rather than one level finished and two untouched.

### `--resume` on every training cell

Every training cell passes `--resume`, and it is idempotent: a finished run exits in seconds and an interrupted one
continues from its last epoch. Omitting it on a re-run is destructive rather than merely wasteful. `--drive-backup`
mirrors `best.pt` and `last.pt` after every epoch, and the pull in the other direction — staging those two files
back down onto a machine that has never seen the run — happens *inside* the resume path. Without `--resume`, a fresh
VM starts at epoch 1 with no restored `best_metric`, `save()` seeds a **new** best from an untrained model, and the
next per-epoch mirror writes that over the good `best.pt` on Drive. The hours already banked are gone, and nothing
in the log says so.

### What the campaign costs

54 planned runs across the three levels, 51 distinct trainings (the tiers share run directories, and a shared
directory is trained and charged once) and **~109 GPU-hours**; `sweep plan` prints all three numbers. Colab Pro+ is
roughly 42 A100-hours a month, so this is a multi-week campaign run in sessions — not an overnight one. That is why
every cell mirrors to Drive, why doneness is keyed on an artifact rather than on a notebook that stayed open, and
why each tier is designed to be a complete, reportable table on its own.

It is also why these arms set `train.eval_profile: sweep`. Evaluation, not training, is the larger half of a run
here — 61–75 minutes of it against 36 of training on this project's measured Colab timings, roughly two thirds of
the wall clock. The `sweep` profile keeps embedding health, sentence retrieval, the held-out scoreboard and the
permutation null, and drops the neuron, emergence, analogy, seen-vs-novel and frequency-matched blocks, every figure
and the interactive explorers. The profile that produced a run is stamped into its `metrics.json`; re-run a winning
arm under `full` when you want the figures.

One operational gate applies to the `token` level specifically. Its notebook runs
`uv run zte-rebaseline --piece-oracle` against the trained checkpoint and reads the result before any number is
quoted, exactly as the length oracle is read at the sentence level. A sub-word piece count is a brain-free
signature large enough to out-retrieve the best encoder this project has trained, so a token-level headline that
does not clear that floor is not evidence of decoding. See [`EVALUATION.md`](EVALUATION.md).

## The results audit — one notebook, no training

[`notebooks/audits/zte_results_audit.ipynb`](https://colab.research.google.com/github/victor-iyi/zte/blob/main/notebooks/audits/zte_results_audit.ipynb)
is standalone: upload it on its own, or open it from the badge, and it clones the repo itself. It trains nothing
and writes nothing into a run directory — it reads the corpus and the checkpoints already on Drive, and hands back
one zip to unpack into `res/audits/`.

Three parts, and the first is the one that decides something:

| Part | What it answers                                                                            | Needs                | Roughly     |
| ---- | ------------------------------------------------------------------------------------------ | -------------------- | ----------- |
| A    | How much of sentence identity does *spelling* give away, on the real 700-sentence gallery? | the corpus only      | minutes     |
| B    | Where does each trained encoder sit against that floor, and against sentence length?       | a checkpoint per run | ~5 min each |
| C    | Everything above, zipped and downloaded                                                    | —                    | seconds     |

Part A is `zte-audit --root <corpus> --piece-oracle`, and it needs **no checkpoint**: the piece oracle is a property
of the corpus, not of any model, so it can be run before a token-level arm exists. That is the point of it — if
spelling alone resolves the gallery, a sub-word alignment level cannot produce an interpretable retrieval number,
and a fortnight of A100 time is better spent elsewhere.

Part B adds each run's own held-out Top-1 through `zte-rebaseline --piece-oracle`, which turns the oracle's
`not measured` into a verdict.

Leave `RESUME_DATE` at `None`. It selects only where the audit *writes*: the corpus path is shared and never
date-stamped, and the run discovery walks every dated session newest-first regardless.

Read `alignment_coverage` before anything else in the block. It is the fraction of ZuCo words that matched their
own reference text, and below about 0.99 the piece counts are partly wrong and the bits are not trustworthy.

## Where things are stored, and why it differs by machine

A twelve-fold sweep does not fit on a Colab VM's disk if everything stays local. The numbers, measured from Drive:

| What                                                             |     Size | Times             |
| ---------------------------------------------------------------- | -------: | ----------------- |
| a prepared raw bundle, one task (`NR`)                           |  11.3 GB | once per task set |
| a prepared raw bundle, `SR+NR`                                   | ~23.6 GB | once              |
| all four task sets the campaign needs                            |   ~60 GB | —                 |
| one run's checkpoints (5 x 90.6 MB) plus figures and TensorBoard |  ~0.5 GB | x54 runs = ~27 GB |

Two mechanisms keep that inside the disk, and both pick their behaviour from the machine.

**Runs follow `write_mode`, which defaults to `auto`.** With Drive mounted — that is, on Colab — runs are written
straight to Drive and the VM's disk never holds them. Off Colab, `auto` resolves to `local+mirror` and the local
disk is primary, which is what a workstation wants. `zte-colab session --write-mode` still takes `local+mirror` or
`drive` explicitly, and the resolved value is printed and carried in the session payload as `write_mode`.

Writing checkpoints onto a FUSE mount needs care: Drive can refuse `os.replace`, and the fallback used to write
straight over the file it was replacing, so a kill mid-write destroyed the only good copy. It now moves the
previous checkpoint aside first, and a write that fails outright keeps that previous file and logs rather than
raising — losing an epoch on day nine of a campaign is recoverable, losing the run is not.

**The bundle is staged on the roomiest local volume, not necessarily the checkout's.** `res/cache/prepared` sits
on whatever disk the repo was cloned onto, and on a Colab GPU runtime that boot volume is often not the largest
one attached. `DriveSession.prepared_local` now scans `/content`, `/var/scratch`, `/scratch`, `/mnt/disks/local`
and `/tmp`, and moves the cache to whichever beats the default by more than one SR+NR bundle (20 GB) — a smaller
gain is not worth a multi-GB copy. Nothing is created on a candidate that loses, and `ZTE_SCRATCH_DIR` pins the
choice outright on a machine whose layout the scan cannot guess.

A scratch volume is **fixed in size and does not survive the machine**, and both are fine here. It only ever holds
a staging copy of the persistent store, so losing it costs a re-stage and never a result — nothing durable is
written there. Runs, checkpoints and evaluation go to Drive under `write_mode: auto`, never to scratch. Being
fixed is why the headroom scales: a flat 12 GB reserve sized for a boot volume would make a 40 GB scratch unusable
rather than safe, so the reserve is capped at 15% of the volume it is actually reserving on. If the chosen volume
has less free space than an SR+NR bundle needs, that is said at selection time rather than at 80% of a copy.

**The prepared bundle stays local, and is evicted rather than accumulated.** It is memory-mapped and read at
random per batch, so serving it over Drive would be unusably slow; it has to be on the fast disk. What changed is
that staging a bundle now frees room for it first, removing least-recently-used entries until the incoming one
fits with `ZTE_MIN_FREE_GB` (default 12 GB) still free. Only an entry the persistent store holds *complete* is
evictable — anything that would have to be rebuilt from the multi-GB extraction stays put, and the shortfall is
reported instead. In practice the disk holds the one bundle in use rather than all four.
