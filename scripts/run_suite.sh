#!/usr/bin/env bash
#
# run_suite.sh -- the fixed-seed driver for the ZTE recommended-experiment suite.
#
# This runs only the current best-performing recipes (the flagship invariance stack, its spatial-model
# A/B, and the sentence-level CLIP alignment A/B) at a fixed seed. Full real-data runs are slow (hours),
# so read this first, then run the study you want by commenting the others out at the bottom -- or run a
# fast synthetic smoke via the SMOKE flag.
#
# Usage:
#   bash scripts/run_suite.sh                         # real data (default root)
#   bash scripts/run_suite.sh /path/to/zuco_extracted # a different data root
#   SMOKE=1 bash scripts/run_suite.sh                 # tiny synthetic sanity pass
#
# PAUSE / RESUME: every run is launched with `zte-run --resume`, so you can stop the suite at ANY time
# (Ctrl-C, or `kill` the process) and simply RE-RUN THE SAME COMMAND to continue exactly where you left
# off -- finished runs are skipped instantly, an interrupted run resumes from its last checkpoint, and the
# cached dataset bundle is reused.
#
# Anti-bias guarantees are baked into the configs (leakage-aware leave-one-subject-out splits, a
# train-only normaliser, a held-out test subject) and into this runner (fixed seeds -> bootstrap CIs).

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT="${1:-res/data/zuco_extracted}"      # data root (positional arg 1)
SEEDS="${SEEDS:-42}"                       # fixed seed(s); default single seed 42. Override e.g. SEEDS="42 43".
PY="${PY:-.venv/bin/python}"              # project venv interpreter
OUT_ROOT="${OUT_ROOT:-res/experiments}"   # where zte-run catalogues each run (set to a mounted Drive path to persist)
DRIVE_BACKUP="${DRIVE_BACKUP:-}"          # mounted Drive folder to mirror checkpoints to each epoch (train local, live Drive copy)
SMOKE="${SMOKE:-0}"                       # SMOKE=1 -> tiny synthetic run

# All 12 ZuCo v1 subjects, for the optional full leave-one-subject-out sweep in Study C.
LOSO_SUBJECTS="ZAB ZDM ZDN ZGW ZJM ZJN ZJS ZKB ZKH ZKW ZMG ZPH"

# Common flags: --synthetic --epochs 2 in SMOKE mode, else real data.
if [[ "${SMOKE}" == "1" ]]; then
  SRC_ARGS=(--synthetic --epochs 2 --device cpu)
  echo ">>> SMOKE mode: tiny synthetic runs, seed(s)=${SEEDS}."
else
  SRC_ARGS=(--root "${ROOT}")
  echo ">>> Real-data mode, root=${ROOT}, seeds=${SEEDS}."
fi

# --------------------------------------------------------------------------- #
# Helper: run one config at one seed.
#
# `zte-run --seed <N>` overrides train.seed and suffixes the run name with `_s<N>`, so seeds live side by
# side under res/experiments/<name>_s<N>/. The held-out subject is baked into each config
# (train.loso_holdout_subject: ZAB), so these are all LOSO "new brain" runs.
# --------------------------------------------------------------------------- #
run_seeded() {
  local config="$1" seed="$2"
  echo ">>> [seed ${seed}] $(basename "${config}" .yaml)_s${seed}"
  local backup=()
  [[ -n "${DRIVE_BACKUP}" ]] && backup=(--drive-backup "${DRIVE_BACKUP}")
  # --resume makes each run idempotent: finished runs are skipped, interrupted ones continue.
  "${PY}" -m zte.cli.run --config "${config}" "${SRC_ARGS[@]}" \
    --seed "${seed}" --out-root "${OUT_ROOT}" --resume "${backup[@]+"${backup[@]}"}"
}

# =========================================================================== #
# STUDY A -- The flagship invariance recipe and its spatial-model A/B.
#   sota_loso  : skip-gram + full invariance stack + spherical-harmonic spatial encoding (the headline).
#   exp7       : the same stack with learned spatial attention over 2-D coordinates + FiLM conditioning.
#   Both EEG-only, held out on ZAB. Compare their held-out retrieval and rank-percentile.
# =========================================================================== #
study_flagship() {
  echo "=== STUDY A: flagship SOTA + spatial-model A/B ==="
  for seed in ${SEEDS}; do
    run_seeded experiments/sota_loso.yaml "${seed}"
    run_seeded experiments/exp7_sota_geom_invariance.yaml "${seed}"
  done
}

# =========================================================================== #
# STUDY B -- Sentence-level CLIP alignment: does aligning EEG to frozen text encode meaning?
#   exp8_clip_e5   : symmetric InfoNCE against an E5 sentence-embedding target.
#   exp8_clip_qwen : the same against a Qwen (mean-pooled) target -- the text-encoder A/B.
#   Needs the frozen encoders (`uv sync --group meaning`); falls back to a hash target otherwise.
# =========================================================================== #
study_clip() {
  echo "=== STUDY B: sentence-level CLIP alignment (E5 vs Qwen) ==="
  for seed in ${SEEDS}; do
    run_seeded experiments/exp8_clip_e5.yaml "${seed}"
    run_seeded experiments/exp8_clip_qwen.yaml "${seed}"
  done
}

# =========================================================================== #
# STUDY C (optional, expensive) -- Full leave-one-subject-out sweep on the flagship.
#   Rotate the held-out subject across all 12, turning one number into a generalisation trend.
#   Uncomment to run (|LOSO_SUBJECTS| x |SEEDS| runs). Prefer scripts/run_loso.sh for this.
# =========================================================================== #
study_loso_sweep() {
  echo "=== STUDY C: full LOSO sweep on the flagship ==="
  for subj in ${LOSO_SUBJECTS}; do
    for seed in ${SEEDS}; do
      "${PY}" -m zte.cli.run --config experiments/sota_loso.yaml "${SRC_ARGS[@]}" \
        --loso-holdout "${subj}" --seed "${seed}" --out-root "${OUT_ROOT}" --resume \
        ${DRIVE_BACKUP:+--drive-backup "${DRIVE_BACKUP}"}
    done
  done
}

# --------------------------------------------------------------------------- #
# Run the studies. Comment out any you do not want; each is independent.
# --------------------------------------------------------------------------- #
study_flagship
study_clip
# study_loso_sweep   # <- uncomment for the full 12-subject sweep (multi-hour)

echo ">>> Suite complete. Compare the runs with:"
echo "    ${PY} -m zte.cli.compare --experiments ${OUT_ROOT} --out ${OUT_ROOT}/COMPARE.html"
echo "    ${PY} -m zte.cli.visualize --run ${OUT_ROOT}/sota_loso_s42 --kind both"
