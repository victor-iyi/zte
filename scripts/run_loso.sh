#!/usr/bin/env bash
# =============================================================================
# ZTE — Leave-One-Subject-Out (LOSO) experiment: the "new brain" generalization test.
#
# Trains the full invariance recipe once per held-out subject (rotating the held-out
# subject over the whole cohort), so the single-subject LOSO result becomes a *trend*.
# Each per-subject run is a self-contained, resumable `zte-run`.
#
#   * Portable: runs on Apple Silicon (MPS), Linux, and Google Colab. The device is chosen
#     automatically (CUDA > MPS > CPU) — no flags needed. Set DEVICE=... to force one.
#   * Fully resumable: every run carries `--resume`. Stop any time (Ctrl-C) and re-run
#     the exact same command — finished subjects are skipped and the interrupted one
#     continues from its last checkpoint. Nothing is recomputed.
#
# USAGE
#   bash scripts/run_loso.sh                      # real data at res/data/zuco_extracted
#   bash scripts/run_loso.sh /path/to/zuco        # real data elsewhere
#   SMOKE=1 bash scripts/run_loso.sh              # fast synthetic dry-run (no data, CPU)
#   CONTROL=1 bash scripts/run_loso.sh            # also run the no-recipe control arm (A/B)
#   DEVICE=cuda bash scripts/run_loso.sh          # force a device (else auto)
#   SUBJECTS="ZAB ZDM" bash scripts/run_loso.sh   # restrict the held-out set
#   FULL_CFG=experiments/sota_loso.yaml bash scripts/run_loso.sh   # run the SOTA stack instead of the old recipe
#   OUT_ROOT="/gdrive/My Drive/zte/loso" bash scripts/run_loso.sh   # write ALL runs to Google Drive (persist everything)
#   DRIVE_BACKUP="/gdrive/My Drive/zte/loso" bash scripts/run_loso.sh  # train local (fast) + mirror checkpoints to Drive each epoch
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="${1:-res/data/zuco_extracted}"
PY="${PY:-.venv/bin/python}"
[ -x "${PY}" ] || PY="python"          # Colab / system python fallback
OUT_ROOT="${OUT_ROOT:-res/experiments/loso}"   # set OUT_ROOT to a mounted Drive path to persist runs
DRIVE_BACKUP="${DRIVE_BACKUP:-}"               # mounted Drive folder to mirror checkpoints to each epoch (train local, live Drive copy)
FULL_CFG="${FULL_CFG:-experiments/study_invariance_full_loso.yaml}"       # the invariance recipe (override: FULL_CFG=experiments/sota_loso.yaml)
CTRL_CFG="${CTRL_CFG:-experiments/study_invariance_baseline_loso.yaml}"   # no-recipe control
SPATIAL="${SPATIAL:-}"    # optional: provision spatial encoding per run (e.g. exact); empty = use each config as-is
MEANING="${MEANING:-}"    # optional: provision meaning target per run (e.g. static / contextual); empty = use each config as-is

# Built once and reused from cache across every held-out subject (montage + meaning are subject-independent).
PROVISION=()
[ -n "${SPATIAL}" ] && PROVISION+=(--spatial "${SPATIAL}")
[ -n "${MEANING}" ] && PROVISION+=(--meaning "${MEANING}")

# All 12 ZuCo v1 subjects; synthetic mode only has three.
ALL_SUBJECTS="ZAB ZDM ZDN ZGW ZJM ZJN ZJS ZKB ZKH ZKW ZMG ZPH"
SYNTH_SUBJECTS="ZAB ZDM ZJN"

if [ "${SMOKE:-0}" = "1" ]; then
  SRC=(--synthetic --epochs "${EPOCHS:-2}" --device "${DEVICE:-cpu}")
  SUBJECTS="${SUBJECTS:-$SYNTH_SUBJECTS}"
else
  SRC=(--root "${ROOT}")
  [ -n "${DEVICE:-}" ] && SRC+=(--device "${DEVICE}")   # else config's 'auto' picks the GPU
  [ -n "${EPOCHS:-}" ] && SRC+=(--epochs "${EPOCHS}")
  SUBJECTS="${SUBJECTS:-$ALL_SUBJECTS}"
fi

run_one() {   # cfg, holdout
  local cfg="$1" holdout="$2"
  echo "───────────────────────────────────────────────────────────────"
  echo "▶ LOSO hold out ${holdout}  ($(basename "${cfg}"))"
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
  fi
  return 0
}

echo "LOSO sweep · ${OUT_ROOT} · held-out: ${SUBJECTS}"
for s in ${SUBJECTS}; do
  [ "${CONTROL:-0}" = "1" ] && run_one "${CTRL_CFG}" "${s}"
  run_one "${FULL_CFG}" "${s}"
done

echo "═══════════════════════════════════════════════════════════════"
echo "✓ LOSO sweep complete. Building the combined comparison view ..."
"${PY}" -m zte.cli.compare --experiments "${OUT_ROOT}" \
    --out "${OUT_ROOT}/COMPARE.html" --title "ZTE — LOSO (new-brain) trend" || true
echo "Open ${OUT_ROOT}/COMPARE.html to compare every held-out subject side by side."
