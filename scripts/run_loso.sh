#!/usr/bin/env bash
# =============================================================================
# ZTE -- full leave-one-subject-out (LOSO) sweep: the "new brain" generalization test.
#
# Trains one config once per held-out subject, rotating the held-out subject over the whole
# 12-person ZuCo cohort, so a single held-out number becomes a trend with error bars.
#
# DEFAULT CONFIG = experiments/flagship/clip_e5_bandpower.yaml -- the only recipe that has ever
# cleared the retrieval gate on real ZuCo (held-out ZAB, 2026-07-16: sentence Top-1 0.093 vs 0.001
# chance, permutation p=0.002). Override with FULL_CFG to sweep a different arm.
#
# SURVIVING A LOST COLAB VM
#   Every run carries --resume, and with DRIVE_BACKUP set the *entire* run directory (config,
#   checkpoints, evaluation, figures, TensorBoard) is mirrored to Drive after every stage, and
#   checkpoints after every epoch. If the VM is reclaimed:
#     1. copy the Drive folder back to OUT_ROOT (or point OUT_ROOT straight at Drive), and
#     2. re-run this exact command.
#   Finished subjects are skipped instantly; the interrupted one resumes from its last epoch.
#   Point DATA_CACHE at a Drive path so the processed dataset bundle is built ONCE, ever.
#
# USAGE
#   bash scripts/run_loso.sh                       # real data at res/data/zuco_extracted
#   bash scripts/run_loso.sh /path/to/zuco         # real data elsewhere
#   SMOKE=1 bash scripts/run_loso.sh               # fast synthetic dry-run (CPU, 3 subjects)
#   CONTROL=1 bash scripts/run_loso.sh             # also run the skip-gram control arm (A/B)
#   SUBJECTS="ZAB ZDM" bash scripts/run_loso.sh    # restrict the held-out set
#   FULL_CFG=experiments/flagship/clip_e5_raw.yaml bash scripts/run_loso.sh
#
# COLAB (recommended -- train on local disk, keep a live Drive copy of everything):
#   DRIVE_BACKUP="/content/drive/MyDrive/Sharables/ZTE/$(date +%F)/experiments" \
#   DATA_CACHE="/content/drive/MyDrive/Sharables/ZTE/prepared" \
#   bash scripts/run_loso.sh /content/zuco_extracted
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="${1:-res/data/zuco_extracted}"
PY="${PY:-.venv/bin/python}"
[ -x "${PY}" ] || PY="python"          # Colab / system python fallback
OUT_ROOT="${OUT_ROOT:-res/experiments/loso}"   # set to a mounted Drive path to write runs straight to Drive
DRIVE_BACKUP="${DRIVE_BACKUP:-}"               # mounted Drive folder; mirrors the whole run dir each stage
DATA_CACHE="${DATA_CACHE:-}"                   # shared PROCESSED-bundle dir; build once, reuse every subject
FULL_CFG="${FULL_CFG:-experiments/flagship/clip_e5_bandpower.yaml}"   # the champion (see header)
CTRL_CFG="${CTRL_CFG:-experiments/benchmark/baseline_skipgram_loso.yaml}"  # skip-gram control arm
SPATIAL="${SPATIAL:-exact}"   # build + wire the true ZuCo-105 electrode montage (needs `mne`; degrades gracefully)
MEANING="${MEANING:-keep}"    # leave each config's own meaning target alone

# Built once and reused from cache across every held-out subject: the montage, the meaning target and
# the processed bundle are all subject-independent, so only the first subject pays for them.
PROVISION=()
[ -n "${SPATIAL}" ] && PROVISION+=(--spatial "${SPATIAL}")
[ -n "${MEANING}" ] && PROVISION+=(--meaning "${MEANING}")
[ -n "${DATA_CACHE}" ] && PROVISION+=(--data-cache "${DATA_CACHE}")

# All 12 ZuCo v1 subjects; the synthetic generator only makes three.
ALL_SUBJECTS="ZAB ZDM ZDN ZGW ZJM ZJN ZJS ZKB ZKH ZKW ZMG ZPH"
SYNTH_SUBJECTS="ZAB ZDM ZJN"

if [ "${SMOKE:-0}" = "1" ]; then
  SRC=(--synthetic --epochs "${EPOCHS:-2}" --device "${DEVICE:-cpu}")
  SUBJECTS="${SUBJECTS:-$SYNTH_SUBJECTS}"
else
  SRC=(--root "${ROOT}")
  [ -n "${DEVICE:-}" ] && SRC+=(--device "${DEVICE}")   # else the config's 'auto' picks the accelerator
  [ -n "${EPOCHS:-}" ] && SRC+=(--epochs "${EPOCHS}")
  SUBJECTS="${SUBJECTS:-$ALL_SUBJECTS}"
fi

FAILED=""

run_one() {   # cfg, holdout
  local cfg="$1" holdout="$2"
  echo "───────────────────────────────────────────────────────────────"
  echo "▶ LOSO hold out ${holdout}  ($(basename "${cfg}" .yaml))"
  local backup=()
  [ -n "${DRIVE_BACKUP}" ] && backup=(--drive-backup "${DRIVE_BACKUP}")
  "${PY}" -m zte.cli.run --config "${cfg}" "${SRC[@]}" \
      --loso-holdout "${holdout}" --out-root "${OUT_ROOT}" --resume --skip-explore \
      "${PROVISION[@]+"${PROVISION[@]}"}" "${backup[@]+"${backup[@]}"}"
  local code=$?
  if [ "${code}" = "130" ]; then
    echo "⏸  Paused during ${holdout}. Re-run this script to resume exactly here."
    exit 130
  elif [ "${code}" != "0" ]; then
    echo "✗ ${holdout} failed (exit ${code}). Re-run to retry; other subjects are unaffected."
    FAILED="${FAILED} $(basename "${cfg}" .yaml)/${holdout}"
  fi
  return 0
}

echo "LOSO sweep · config $(basename "${FULL_CFG}") · out ${OUT_ROOT} · held-out: ${SUBJECTS}"
[ -n "${DRIVE_BACKUP}" ] && echo "Drive mirror: ${DRIVE_BACKUP} (whole run dir, every stage)"
[ -n "${DATA_CACHE}" ]   && echo "Shared bundle cache: ${DATA_CACHE} (built once, reused per subject)"

for s in ${SUBJECTS}; do
  [ "${CONTROL:-0}" = "1" ] && run_one "${CTRL_CFG}" "${s}"
  run_one "${FULL_CFG}" "${s}"
done

echo "═══════════════════════════════════════════════════════════════"
if [ -n "${FAILED}" ]; then
  echo "⚠  Completed with failures:${FAILED}"
  echo "   Re-run the same command to retry only those (everything else is skipped instantly)."
else
  echo "✓ LOSO sweep complete."
fi

echo "Building the combined comparison view ..."
"${PY}" -m zte.cli.compare --experiments "${OUT_ROOT}" \
    --out "${OUT_ROOT}/COMPARE.html" --title "ZTE — LOSO (new-brain) trend" || true
if [ -n "${DRIVE_BACKUP}" ] && [ -f "${OUT_ROOT}/COMPARE.html" ]; then
  mkdir -p "${DRIVE_BACKUP}" && cp -f "${OUT_ROOT}/COMPARE.html" "${DRIVE_BACKUP}/" 2>/dev/null || true
fi
echo "Open ${OUT_ROOT}/COMPARE.html to compare every held-out subject side by side."
