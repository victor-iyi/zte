# ZTE experiments

Five ready-to-run experiment configs. Each is a plain, editable [`ZTEConfig`](../src/zte/config.py) YAML; run any of them end to end with **one command** and everything lands under `res/experiments/<run_name>/`.

```sh
# Synthetic smoke test (no data), then a real run, then straight from Google Drive:
uv run zte-run --config experiments/exp1_skipgram_rope_et.yaml --synthetic --epochs 5
uv run zte-run --config experiments/exp1_skipgram_rope_et.yaml --root res/data/zuco_extracted
uv run zte-run --config experiments/exp2_masked_rope_eegonly.yaml --drive <folder-id-or-url>
```

`--drive` works on every CLI that loads raw ZuCo data (`zte-prepare`, `zte-train`,
`zte-evaluate`, `zte-explore`, `zte-benchmark`, …) — not just `zte-run`. Install
Drive support once: `uv sync --group drive`. Zips download to `res/data/_downloads`
and extract into `--extract-dir` (default `res/data/zuco_extracted`).

Handy overrides (all optional): `--name`, `--subjects ZAB,ZDM`, `--tasks SR,NR`, `--epochs N`, `--device auto|cpu|cuda|mps`, `--extract-dir`, `--no-tensorboard`, `--no-interactive`, `--skip-eval`, `--skip-explore`.

## The five configs

| Config                        | Objective            | Positional | Eye-tracking | Split               | Purpose                                                                            |
| ----------------------------- | -------------------- | ---------- | ------------ | ------------------- | ---------------------------------------------------------------------------------- |
| **exp1_skipgram_rope_et**     | skip-gram            | RoPE       | included     | by_sentence         | Flagship default — the strongest general reading-evoked embedding. **Start here.** |
| **exp2_masked_rope_eegonly**  | masked (data2vec)    | RoPE       | **excluded** | by_sentence         | The imagined-thought / device-agnostic path: EEG only, no gaze behaviour.          |
| **exp3_cpc_rope_et**          | CPC (wav2vec/BENDR)  | RoPE       | included     | by_sentence         | Tests whether reading *order* carries transferable structure.                      |
| **exp4_skipgram_loso**        | skip-gram            | RoPE       | included     | **by_subject_loso** | Subject-invariance benchmark: validation is an unseen person (holds out `ZPH`).    |
| **exp5_raw_conformer_masked** | masked (reconstruct) | RoPE       | excluded     | by_sentence         | Raw temporal path: EEG-Conformer over raw windows instead of band power.           |

### Why these five

- **exp1 vs exp2** isolates the single most important design question for a *thought* code: does the representation lean on eye-tracking (great for reading, absent for imagined thought)? Compare their `evaluation/report.md` and the `exploration/eye_tracking_contribution.csv`.
- **exp1 vs exp3** compares a context-free word2vec objective against a causal, order-aware one.
- **exp4** is the honest subject-agnosticism test — the validation subject is never seen in training; watch the subject-transfer analogy and cross-subject retrieval.
- **exp5** swaps the entire input pathway (raw EEG + Conformer) to check the band-power compression is not leaving signal on the table.

## Reproducibility

Every config fixes `train.seed` and sets `train.deterministic: true`. `zte-run` copies the fully-resolved `config.yaml` into the run directory, so any run reproduces with:

```sh
uv run zte-run --config res/experiments/<run_name>/config.yaml --root <data> --name <run_name>
# or: --drive <folder-id-or-url>
```

## Sweeping the grid

To compare objectives × positional encodings × eye-tracking under fixed seeds in one shot (headline metrics only, no per-run figures):

```sh
uv run zte-benchmark --root res/data/zuco_extracted \
    --objectives skipgram,masked,cpc --pos-encodings rope,learned --eye-tracking both \
    --seeds 42,43 --out res/benchmark
# Or benchmark straight from Drive:
uv run zte-benchmark --drive <folder-id-or-url> --objectives skipgram,masked --out res/benchmark
```

## Catalogue

`res/experiments/INDEX.md` accumulates one row per run (words, cross-subject retrieval Top-1, subject-transfer Top-1, effective-rank ratio) so runs are comparable at a glance.  Each run's own `README.md` and `manifest.json` hold the full configuration, data source and verdict.
