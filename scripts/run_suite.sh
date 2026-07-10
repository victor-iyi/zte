#!/usr/bin/env bash
#
# run_suite.sh -- the fixed-seed driver for the ZTE bias-controlled experiment suite.
#
# This script is safe to READ as documentation even if you never execute it: it is
# just the exact commands from docs/EXPERIMENTS.md, wired into seed loops. Full
# real-data runs are slow (hours), so read it first, then run the study you want by
# commenting the others out -- or run a fast synthetic smoke via the SMOKE flag.
#
# Usage:
#   bash scripts/run_suite.sh                         # real data (default root)
#   bash scripts/run_suite.sh /path/to/zuco_extracted # a different data root
#   SMOKE=1 bash scripts/run_suite.sh                 # tiny synthetic sanity pass
#
# PAUSE / RESUME: every run is launched with `zte-run --resume`, so you can stop the
# suite at ANY time (Ctrl-C, or `kill` the process) and simply RE-RUN THE SAME COMMAND
# to continue exactly where you left off -- finished runs are skipped, an interrupted
# run resumes from its last checkpoint, and the cached dataset bundle is reused.
#
# Anti-bias guarantees are baked into the configs (leakage-aware splits, train-only
# normaliser, held-out test) and into this runner (fixed seeds -> bootstrap CIs).
# See docs/EXPERIMENTS.md for the full rationale and "how to read the outputs".

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT="${1:-res/data/zuco_extracted}"     # data root (positional arg 1)
SEEDS="42 "                              # fixed seeds -> differences carry CIs
PY=".venv/bin/python"                    # project venv interpreter
OUT_ROOT="res/experiments"               # where zte-run catalogues each run
BENCH_ROOT="res/benchmark"               # where zte-benchmark writes tables
SMOKE="${SMOKE:-0}"                      # SMOKE=1 -> tiny synthetic run

# All 12 ZuCo v1 subjects, for the full leave-one-subject-out sweep in Study 2.
LOSO_SUBJECTS="ZAB ZDM ZDN ZGW ZJM ZJN ZJS ZKB ZKH ZKW ZMG ZPH"

# Common flags: --synthetic --epochs 2 in SMOKE mode, else real data.
if [[ "${SMOKE}" == "1" ]]; then
  SRC_ARGS=(--synthetic --epochs 2 --device cpu)
  BENCH_SRC=(--synthetic --epochs 2)
  SEEDS="42"
  echo ">>> SMOKE mode: tiny synthetic runs, single seed."
else
  SRC_ARGS=(--root "${ROOT}")
  BENCH_SRC=(--root "${ROOT}")
  echo ">>> Real-data mode, root=${ROOT}, seeds=${SEEDS}."
fi

# --------------------------------------------------------------------------- #
# Helper: run one config at one seed.
#
# `zte-run --seed <N>` overrides train.seed and suffixes the run name with
# `_s<N>`, so seeds live side by side under res/experiments/<name>_s<N>/.
# --------------------------------------------------------------------------- #
run_seeded() {
  local config="$1" seed="$2"
  echo ">>> [seed ${seed}] $(basename "${config}" .yaml)_s${seed}"
  # --resume makes each run idempotent: finished runs are skipped, interrupted ones continue.
  "${PY}" -m zte.cli.run --config "${config}" "${SRC_ARGS[@]}" \
    --seed "${seed}" --out-root "${OUT_ROOT}" --resume
}

# zte-benchmark sweeps are NOT resumable, so skip one whose benchmark.csv already exists
# (set FORCE=1 to redo). This keeps a restarted suite from re-running finished sweeps.
bench_once() {
  local out="$1"; shift
  if [[ "${FORCE:-0}" != "1" && -f "${out}/benchmark.csv" ]]; then
    echo ">>> benchmark already done: ${out} (skipping; FORCE=1 to redo)."
    return 0
  fi
  "${PY}" -m zte.cli.benchmark "${BENCH_SRC[@]}" "$@" --out "${out}"
}

# =========================================================================== #
# STUDY 1 -- Purpose / eye-tracking confound.
#   Clean matched A/B: same model/seed/split, only include_eye_tracking flips.
#   Plus the two full-scale flagships (exp1 ET-on vs exp6 EEG-only).
# =========================================================================== #
study1() {
  echo "=== STUDY 1: eye-tracking confound ==="
  bench_once "${BENCH_ROOT}/et_confound" \
    --objectives skipgram --pos-encodings rope --eye-tracking both \
    --seeds "${SEEDS// /,}"
  # Flagship full runs (report both; the benchmark above is the clean confound test).
  for seed in ${SEEDS}; do
    run_seeded experiments/exp1_skipgram_rope_et.yaml "${seed}"
    run_seeded experiments/exp6_skipgram_eegonly_invariant.yaml "${seed}"
  done
}

# =========================================================================== #
# STUDY 2 -- Subject-invariance A/B under LOSO (the north star).
#   Baseline (no levers) vs full invariance stack, matched, EEG-only.
# =========================================================================== #
study2() {
  echo "=== STUDY 2: subject-invariance A/B (LOSO) ==="
  for seed in ${SEEDS}; do
    run_seeded experiments/study_invariance_baseline_loso.yaml "${seed}"
    run_seeded experiments/study_invariance_full_loso.yaml "${seed}"
  done
  # Full leave-one-subject-out sweep: rotate the held-out subject across all 12.
  # Uncomment to run (expensive: 2 configs x 12 subjects x |SEEDS| runs).
  # for subj in ${LOSO_SUBJECTS}; do
  #   for cfg in study_invariance_baseline_loso study_invariance_full_loso; do
  #     for seed in ${SEEDS}; do
  #       tmp="${OUT_ROOT}/_seeded/${cfg}_${subj}_s${seed}.yaml"
  #       mkdir -p "$(dirname "${tmp}")"
  #       "${PY}" - "experiments/${cfg}.yaml" "${subj}" "${seed}" "${tmp}" <<'PYEOF'
  # import sys
  # from zte.config import ZTEConfig
  # cfg = ZTEConfig.from_yaml(sys.argv[1])
  # cfg.train.loso_holdout_subject = sys.argv[2]
  # cfg.train.seed = int(sys.argv[3])
  # cfg.to_yaml(sys.argv[4])
  # PYEOF
  #       "${PY}" -m zte.cli.run --config "${tmp}" "${SRC_ARGS[@]}" \
  #         --name "${cfg}_${subj}_s${seed}" --out-root "${OUT_ROOT}"
  #     done
  #   done
  # done
}

# =========================================================================== #
# STUDY 3 -- Anti-collapse ablation (VICReg OFF vs ON), by_stimulus, EEG-only.
# =========================================================================== #
study3() {
  echo "=== STUDY 3: VICReg anti-collapse ablation ==="
  for seed in ${SEEDS}; do
    run_seeded experiments/study_vicreg_off.yaml "${seed}"
    run_seeded experiments/study_vicreg_on.yaml  "${seed}"
  done
}

# =========================================================================== #
# STUDY 4 -- Objective sweep (skipgram/cbow/masked/cpc), EEG-only, fixed seeds.
# =========================================================================== #
study4() {
  echo "=== STUDY 4: objective sweep ==="
  bench_once "${BENCH_ROOT}/objective_sweep" \
    --objectives skipgram,cbow,masked,cpc --pos-encodings rope \
    --eye-tracking off --seeds "${SEEDS// /,}"
}

# =========================================================================== #
# STUDY 5 (optional) -- Representation: raw conformer vs band-power (both masked).
# =========================================================================== #
study5() {
  echo "=== STUDY 5: representation (raw conformer vs band-power) ==="
  for seed in ${SEEDS}; do
    run_seeded experiments/exp5_raw_conformer_masked.yaml "${seed}"
    run_seeded experiments/exp2_masked_rope_eegonly.yaml  "${seed}"
  done
}

# --------------------------------------------------------------------------- #
# Run the studies. Comment out any you do not want; each is independent.
# --------------------------------------------------------------------------- #
# study1
# study2
study3
study4
study5

echo ">>> Suite complete. Explore a run with:"
echo "    ${PY} -m zte.cli.visualize --run ${OUT_ROOT}/study_vicreg_on_s42 --out res/explorer/study_vicreg_on.html"
echo ">>> Read docs/EXPERIMENTS.md -> 'How to read the outputs' for every artifact."
