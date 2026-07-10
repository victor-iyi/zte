# AGENTS.md

## Cursor Cloud specific instructions

`zte` (ZuCo Thought Embedding) is a single Python package (no web server, API, or
database) managed by [`uv`](https://docs.astral.sh/uv/). It ships a CLI + library that
pretrains word-level EEG "thought embeddings" and writes all artifacts under `res/`
(gitignored).

### Environment
- Requires **Python >= 3.14**; `uv` provisions this interpreter automatically, so the
  system `python3` (3.12) is not used. `uv` is preinstalled at `~/.local/bin` and is on
  `PATH` in login/interactive shells.
- The startup update script runs `uv sync`, which installs the `all` + `dev` dependency
  groups (see `[tool.uv] default-groups` in `pyproject.toml`). Nothing else needs to run
  at startup — there are no long-running services.

### Standard commands (run from repo root)
- Lint: `uv run ruff check .` (format: `uv run ruff format`)
- Tests: `uv run pytest` (fully self-contained; uses synthetic fixtures)
- End-to-end smoke run (no external data): `uv run zte-run --config experiments/exp1_skipgram_rope_et.yaml --synthetic --epochs 3 --name <run_name>`
- Other entry points are defined in `pyproject.toml [project.scripts]` (`zte-prepare`,
  `zte-train`, `zte-extract`, `zte-evaluate`, `zte-explore`, `zte-benchmark`,
  `zte-download`).

### Non-obvious notes
- No external services, secrets, or env vars are needed for development or testing. Real
  ZuCo data / Google Drive are optional; use `--synthetic` for offline end-to-end runs.
- Runs default to **CPU** in this environment (no GPU); this is expected and sufficient
  for the test suite and synthetic smoke runs. CUDA wheels are installed but unused.
- Each `zte-run` writes a self-contained folder at `res/experiments/<run_name>/`
  (config, bundle, checkpoints, figures, `evaluation/`, `exploration/`, and an
  interactive `evaluation/interactive/word_explorer.html`).
- On synthetic data, `sklearn` may print `LinAlgWarning: ill-conditioned matrix` during
  region-importance ridge regression — this is benign noise from tiny synthetic samples.
- The README's "Python 3.12+" note is stale; `pyproject.toml` pins `requires-python =
  '>=3.14'` and ruff `target-version = 'py314'`.
