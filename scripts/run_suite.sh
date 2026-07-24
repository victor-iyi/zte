#!/usr/bin/env bash
#
# run_suite.sh -- the fixed-seed driver for the ZTE experiment suite.
#
# Study order follows the evidence from the 2026-07-16 real-ZuCo LOSO session (held out ZAB):
#
#   | run                       | sentence Top-1 (chance 0.0013) | permutation p | held-out Top-1 lift |
#   | ------------------------- | ------------------------------ | ------------- | ------------------- |
#   | clip_e5_bandpower         | 0.0932  ✓ above chance         | 0.002         | +0.29pp             |
#   | clip_e5_raw               | 0.0065  ✓ above chance         | 0.002         | +0.71pp  (best)     |
#   | clip_qwen_bandpower       | 0.0010  ✗                      | 0.096         | +0.00pp             |
#   | baseline_skipgram_loso    | 0.0004  ✗                      | 0.986         | +0.29pp             |
#
# So: CLIP against an E5 sentence target is the only objective that has ever beaten chance here, and
# skip-gram is now a control rather than a contender. Everything below is built around that.
#
# Usage:
#   bash scripts/run_suite.sh                         # real data (default root)
#   bash scripts/run_suite.sh /path/to/zuco_extracted # a different data root
#   SMOKE=1 bash scripts/run_suite.sh                 # tiny synthetic sanity pass (CPU, minutes)
#   STUDIES="audit flagship" bash scripts/run_suite.sh   # run only some studies
#
# STUDIES (default: "audit flagship controls"):
#   audit     -- model-free confound audit of the dataset (run this before believing any result)
#   flagship  -- the three CLIP arms held out on ZAB: band-power, raw-conformer, +meaning distillation
#   controls  -- the skip-gram baseline and the Qwen text-encoder arm, for honest comparison
#   benchmark -- objective sweep on top of the champion recipe (zte-benchmark, resumable)
#   ablate    -- one-knob-at-a-time studies on the champion (zte-ablate generate + run + diff)
#   loso      -- the full 12-subject LOSO sweep on the champion (multi-hour; delegates to run_loso.sh)
#
# PAUSE / RESUME: every run is launched with `zte-run --resume`, so you can stop at ANY time (Ctrl-C,
# or a reclaimed Colab VM) and re-run the SAME command to continue exactly where you left off.
# Finished runs are skipped instantly, an interrupted run resumes from its last checkpoint, and the
# cached dataset bundle is reused. With DRIVE_BACKUP set, the whole run directory is mirrored to
# Drive after every stage, so nothing but the epoch in flight is ever lost.
#
# Anti-bias guarantees are baked into the configs (leakage-aware leave-one-subject-out splits, a
# train-only normaliser, a held-out test subject) and into this runner (fixed seeds -> bootstrap CIs).

set -uo pipefail
cd "$(dirname "$0")/.."

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT="${1:-res/data/zuco_extracted}"      # data root (positional arg 1)
SEEDS="${SEEDS:-42}"                       # fixed seed(s); override e.g. SEEDS="42 43"
PY="${PY:-.venv/bin/python}"              # project venv interpreter
[ -x "${PY}" ] || PY="python"             # Colab / system python fallback
OUT_ROOT="${OUT_ROOT:-res/experiments}"   # where zte-run catalogues each run (a Drive path persists everything)
DRIVE_BACKUP="${DRIVE_BACKUP:-}"          # mounted Drive folder; mirrors each whole run dir every stage
DATA_CACHE="${DATA_CACHE:-}"              # shared PROCESSED-bundle dir; build once, reuse across every run
SMOKE="${SMOKE:-0}"                       # SMOKE=1 -> tiny synthetic run
SPATIAL="${SPATIAL:-exact}"               # build + wire the true ZuCo-105 montage (needs `mne`; degrades gracefully)
MEANING="${MEANING:-keep}"                # leave each config's own meaning target alone
STUDIES="${STUDIES:-audit flagship controls}"
HOLDOUT="${HOLDOUT:-ZAB}"                 # held-out subject for the single-fold studies

# The configs, by tier (see experiments/README.md).
FLAGSHIP_CONFIGS="${FLAGSHIP_CONFIGS:-\
experiments/flagship/clip_e5_bandpower.yaml \
experiments/flagship/clip_e5_raw.yaml \
experiments/flagship/clip_e5_meaning.yaml}"
CONTROL_CONFIGS="${CONTROL_CONFIGS:-\
experiments/benchmark/baseline_skipgram_loso.yaml \
experiments/benchmark/clip_qwen_bandpower.yaml}"
CHAMPION="${CHAMPION:-experiments/flagship/clip_e5_bandpower.yaml}"
ABLATE_KNOBS="${ABLATE_KNOBS:-objective.meaning_distill_weight objective.subject_adversary_weight}"

# Turn-key provisioning shared by every run (built once, then reused from cache): --spatial exact
# builds + wires the exact montage; --data-cache stores the processed bundle once so the .mat load +
# processing is skipped on every later run and session.
PROVISION=()
[ -n "${SPATIAL}" ] && PROVISION+=(--spatial "${SPATIAL}")
[ -n "${MEANING}" ] && PROVISION+=(--meaning "${MEANING}")
[ -n "${DATA_CACHE}" ] && PROVISION+=(--data-cache "${DATA_CACHE}")

BACKUP=()
[ -n "${DRIVE_BACKUP}" ] && BACKUP=(--drive-backup "${DRIVE_BACKUP}")

# Common flags: --synthetic --epochs 2 in SMOKE mode, else real data.
if [ "${SMOKE}" = "1" ]; then
  SRC_ARGS=(--synthetic --epochs "${EPOCHS:-2}" --device cpu)
  echo ">>> SMOKE mode: tiny synthetic runs, seed(s)=${SEEDS}."
else
  SRC_ARGS=(--root "${ROOT}")
  [ -n "${EPOCHS:-}" ] && SRC_ARGS+=(--epochs "${EPOCHS}")
  echo ">>> Real-data mode, root=${ROOT}, seeds=${SEEDS}."
fi
[ -n "${DRIVE_BACKUP}" ] && echo ">>> Drive mirror: ${DRIVE_BACKUP} (whole run dir, every stage)."
[ -n "${DATA_CACHE}" ]   && echo ">>> Shared bundle cache: ${DATA_CACHE} (built once, reused)."

FAILED=""

# --------------------------------------------------------------------------- #
# Helper: run one config at one seed, held out on one subject.
#
# `zte-run --seed <N>` overrides train.seed and suffixes the run name with `_s<N>`; `--loso-holdout`
# suffixes it with `_lo<SUBJ>`, so every arm gets its own resumable run directory.
# --------------------------------------------------------------------------- #
run_seeded() {
  local config="$1" seed="$2"
  local name; name="$(basename "${config}" .yaml)"
  echo ">>> [seed ${seed}] ${name} (hold out ${HOLDOUT})"
  # --resume makes each run idempotent: finished runs are skipped, interrupted ones continue.
  "${PY}" -m zte.cli.run --config "${config}" "${SRC_ARGS[@]}" \
    --seed "${seed}" --loso-holdout "${HOLDOUT}" --out-root "${OUT_ROOT}" --resume \
    "${PROVISION[@]+"${PROVISION[@]}"}" "${BACKUP[@]+"${BACKUP[@]}"}"
  local code=$?
  if [ "${code}" = "130" ]; then
    echo "⏸  Paused during ${name}. Re-run this script to resume exactly here."
    exit 130
  elif [ "${code}" != "0" ]; then
    echo "✗ ${name} failed (exit ${code}); continuing with the rest."
    FAILED="${FAILED} ${name}"
  fi
  return 0
}

run_configs() {
  local seed
  for config in $1; do
    for seed in ${SEEDS}; do
      run_seeded "${config}" "${seed}"
    done
  done
}

# =========================================================================== #
# AUDIT -- the model-free confound check. Run it FIRST: if task and stimulus are
# fully confounded in the data, a "task-decoding" result is a stimulus result.
# =========================================================================== #
study_audit() {
  echo "=== AUDIT: dataset confound report ==="
  local src=(--root "${ROOT}")
  [ "${SMOKE}" = "1" ] && src=(--synthetic)
  "${PY}" -m zte.cli.audit "${src[@]}" --out "${OUT_ROOT}/confound_audit.md" || {
    echo "✗ audit failed; continuing."; FAILED="${FAILED} audit"; }
  [ -n "${DRIVE_BACKUP}" ] && [ -f "${OUT_ROOT}/confound_audit.md" ] && \
    mkdir -p "${DRIVE_BACKUP}" && cp -f "${OUT_ROOT}/confound_audit."* "${DRIVE_BACKUP}/" 2>/dev/null
  return 0
}

# =========================================================================== #
# FLAGSHIP -- the three CLIP arms, held out on one subject.
#   clip_e5_bandpower : the champion (best in-sample cross-subject retrieval).
#   clip_e5_raw       : raw-conformer encoder (best HELD-OUT lift, and the only arm that made
#                       subjects harder to identify than raw band power).
#   clip_e5_meaning   : champion + contextual word-meaning distillation -- the untested hypothesis
#                       aimed at the 0.0% content variance the champion still reports.
# =========================================================================== #
study_flagship() {
  echo "=== FLAGSHIP: CLIP arms (E5 band-power · raw-conformer · +meaning) ==="
  run_configs "${FLAGSHIP_CONFIGS}"
}

# =========================================================================== #
# CONTROLS -- what the flagship must beat to earn its place.
#   baseline_skipgram_loso : the previous SOTA recipe (skip-gram + full invariance stack).
#   clip_qwen_bandpower    : the second arm of the text-encoder A/B (E5 vs Qwen).
# =========================================================================== #
study_controls() {
  echo "=== CONTROLS: skip-gram baseline + Qwen text-encoder arm ==="
  run_configs "${CONTROL_CONFIGS}"
}

# =========================================================================== #
# BENCHMARK -- objective sweep ON TOP OF the champion recipe, so the only thing
# that differs between rows is the objective (not the encoder or the geometry fix).
# Resumable: finished cells are reused from their metrics.json.
# =========================================================================== #
study_benchmark() {
  echo "=== BENCHMARK: objective sweep on the champion recipe ==="
  local src=(--root "${ROOT}")
  [ "${SMOKE}" = "1" ] && src=(--synthetic)
  "${PY}" -m zte.cli.benchmark "${src[@]}" \
    --base-config "${CHAMPION}" --loso-holdout "${HOLDOUT}" \
    --objectives "${BENCH_OBJECTIVES:-clip,skipgram,masked,cpc}" --pos-encodings rope \
    --eye-tracking off --seeds "$(echo "${SEEDS}" | tr ' ' ',')" \
    --epochs "${BENCH_EPOCHS:-${EPOCHS:-10}}" --out "${OUT_ROOT}/../benchmark" --resume \
    "${BACKUP[@]+"${BACKUP[@]}"}" || { echo "✗ benchmark failed; continuing."; FAILED="${FAILED} benchmark"; }
  return 0
}

# =========================================================================== #
# ABLATE -- prove one lever at a time on the champion. Generates a config per
# value, runs each (resumable), then diffs the held-out scoreboards.
# =========================================================================== #
study_ablate() {
  echo "=== ABLATE: one-knob studies on the champion ==="
  local dir="${OUT_ROOT}/../ablate_configs"
  for knob in ${ABLATE_KNOBS}; do
    echo "--- knob: ${knob}"
    "${PY}" -m zte.cli.ablate generate --config "${CHAMPION}" \
      --knob "${knob}" --values "${ABLATE_VALUES:-0,0.1,1.0}" --out-dir "${dir}" || {
      echo "✗ ablate generate failed for ${knob}; continuing."; FAILED="${FAILED} ablate/${knob}"; continue; }
  done
  for cfg in "${dir}"/*.yaml; do
    [ -e "${cfg}" ] || continue
    run_seeded "${cfg}" "$(echo "${SEEDS}" | awk '{print $1}')"
  done
  return 0
}

# =========================================================================== #
# LOSO -- the full 12-subject sweep on the champion (multi-hour). Delegates to
# run_loso.sh, which is itself fully resumable and Drive-mirrored.
# =========================================================================== #
study_loso() {
  echo "=== LOSO: full 12-subject sweep on the champion ==="
  SMOKE="${SMOKE}" OUT_ROOT="${OUT_ROOT}/loso" DRIVE_BACKUP="${DRIVE_BACKUP}" \
    DATA_CACHE="${DATA_CACHE}" FULL_CFG="${CHAMPION}" SPATIAL="${SPATIAL}" MEANING="${MEANING}" \
    PY="${PY}" bash scripts/run_loso.sh "${ROOT}" || {
      echo "✗ loso sweep reported failures; see above."; FAILED="${FAILED} loso"; }
  return 0
}

# --------------------------------------------------------------------------- #
# Run the requested studies.
# --------------------------------------------------------------------------- #
for study in ${STUDIES}; do
  case "${study}" in
    audit)     study_audit ;;
    flagship)  study_flagship ;;
    controls)  study_controls ;;
    benchmark) study_benchmark ;;
    ablate)    study_ablate ;;
    loso)      study_loso ;;
    *) echo "Unknown study '${study}' (valid: audit flagship controls benchmark ablate loso)"; exit 2 ;;
  esac
done

echo "═══════════════════════════════════════════════════════════════"
if [ -n "${FAILED}" ]; then
  echo "⚠  Suite finished with failures:${FAILED}"
  echo "   Re-run the same command to retry only those; everything else is skipped instantly."
else
  echo "✓ Suite complete."
fi
echo ">>> Compare the runs with:"
echo "    ${PY} -m zte.cli.compare --experiments ${OUT_ROOT} --out ${OUT_ROOT}/COMPARE.html"
echo "    ${PY} -m zte.cli.visualize --run ${OUT_ROOT}/exp8_clip_e5_lo${HOLDOUT}_s42 --kind both"
