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

## Colab environment bootstrap

Colab leaves some env vars unset and starts in the wrong directory. `zte.utils.bootstrap()` sets the missing vars (headless matplotlib, a writable config/cache dir, quiet tokenizers), resolves the project root, and creates the `res/` output directories — so the CLIs never error on a fresh runtime. The notebook calls it in Section 2; it is idempotent and a no-op on a laptop.

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

**[`notebooks/zte_colab.ipynb`](../notebooks/zte_colab.ipynb)**
([open in Colab](https://colab.research.google.com/github/victor-iyi/zte/blob/main/notebooks/zte_colab.ipynb))
is the original, kept for the sections v2 does not carry: the benchmark suite, `zte-ablate` grid generation, the
per-run explorer views and the housekeeping cells.

Both need a GPU runtime (`Runtime → Change runtime type → GPU`) and both provision Python 3.14 through `uv` in their
first cell, because Colab's system interpreter is older than ZTE requires.
