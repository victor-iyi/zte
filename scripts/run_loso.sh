#!/usr/bin/env bash
# =============================================================================
# ZTE -- full leave-one-subject-out (LOSO) sweep: the "new brain" generalization test.
#
# Trains one config once per held-out subject, rotating the held-out subject over the whole
# 12-person ZuCo cohort, so a single held-out number becomes a trend with error bars.
#
# DEFAULT CONFIG = experiments/flagship/zte_raw_aligned.yaml -- the raw-conformer champion.
#
# WHY NOT BAND POWER ANY MORE (2026-07-25 re-scoring of every run on Drive). The old default,
# clip_e5_meaning.yaml, was picked on its POOLED Top-1 of 0.043 -- a number computed over training
# subjects as well as the held-out one. Re-scored on the held-out subject alone (700 queries,
# chance 1/700) it lands 4 hits, and an identical re-run landed 2. On the same fold the raw
# conformer lands 32 at Top-5 (p ~ 1e-15). Band power's low subject-probe score was never
# disentanglement: its effective-rank ratio of 0.160 says the space had collapsed to ~123 of 768
# directions, so there was nothing left to identify anyone by. The band-power arms are now in
# experiments/archive/ with their numbers; see that README.
#
#   arm (held out ZAB)                 frontend        Top-5 hits/700   p        eff-rank
#   exp8_clip_e5_raw                   raw_conformer        32        7e-16      0.264
#   exp10_clip_e5_meaning_raw          raw_conformer        32        7e-16      0.264
#   exp10_clip_e5_meaning_raw_v2       raw_conformer        19        1e-06      0.535
#   exp9_clip_e5_meaning (retired)     band_power           10        3e-02      0.160
#
# WHAT THE DEFAULT ADDS ON TOP. The raw path had never had cross-subject alignment of any kind --
# `dataset.normalize` only ever applied to band power, so `normalize: riemannian` was a silent
# no-op for every raw run above. exp12 adds three label-free steps: Euclidean alignment of the raw
# windows, a subject adapter driven by a hypernetwork over each person's covariance geometry
# (rather than an id lookup, which is inert for the held-out subject by construction), and a
# rank-preserving identity-orthogonality penalty. See the config header for the full argument.
#
# READING THE RESULT. LOSO_SUMMARY.md now leads with rank percentile (every query contributes) and
# reports Top-K as raw hit counts against the handful expected by chance, with an exact binomial
# tail. Top-1 on 700 queries at 1/700 expects ONE hit; "0.006 vs 0.001" is three hits and noise.
#
# SURVIVING A LOST COLAB VM
#   Every run carries --resume, and with DRIVE_BACKUP set the *entire* run directory (config,
#   checkpoints, evaluation, figures, TensorBoard) is mirrored to Drive after every stage, and
#   checkpoints after every epoch. If the VM is reclaimed:
#     1. copy the Drive folder back to OUT_ROOT (or point OUT_ROOT straight at Drive), and
#     2. re-run this exact command.
#   Finished subjects are skipped instantly; the interrupted one resumes from its last epoch.
#   Point DATA_CACHE at a Drive path so the processed dataset bundle is built ONCE, ever.
#   Alignment is applied AFTER the bundle loads, so turning it on does not rebuild the cache.
#
# USAGE
#   bash scripts/run_loso.sh                       # real data at res/data/zuco_extracted
#   bash scripts/run_loso.sh /path/to/zuco         # real data elsewhere
#   SMOKE=1 bash scripts/run_loso.sh               # fast synthetic dry-run (CPU, 3 subjects)
#   CONTROL=1 bash scripts/run_loso.sh             # also run the skip-gram control arm (A/B)
#   SUBJECTS="ZAB ZDM" bash scripts/run_loso.sh    # restrict the held-out set
#   SEEDS="42 43 44" bash scripts/run_loso.sh      # repeat each fold at N seeds -> mean±std
#   FULL_CFG=experiments/flagship/zte_raw_aligned_wide.yaml bash scripts/run_loso.sh  # v2 encoder
#   FULL_CFG=experiments/flagship/clip_e5_meaning_raw.yaml bash scripts/run_loso.sh   # the 32/700 baseline
#
# WHY MULTI-SEED. The 2026-07-24 single-seed sweep converged bimodally: 5/12 folds trained to a healthy
# subject-invariant code, 3/12 collapsed (pooled retrieval < 0.01, identity not removed). A single seed
# per fold cannot tell "this subject is hard" from "this seed was unlucky". SEEDS="42 43 44" reruns each
# fold at three seeds so the honest summary can report mean±std and flag genuine vs seed-driven failure.
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
DATA_CACHE="${DATA_CACHE:-}"                   # shared local PROCESSED-bundle dir; build once, reuse every subject
# Persistent cache dir (e.g. a mounted Drive folder) layered behind DATA_CACHE: a bundle found there is
# copied down once, a freshly built one is published there immediately. Also honoured via $ZTE_CACHE_REMOTE.
CACHE_REMOTE="${CACHE_REMOTE:-${ZTE_CACHE_REMOTE:-}}"
export ZTE_CACHE_REMOTE="${CACHE_REMOTE}"
FULL_CFG="${FULL_CFG:-experiments/flagship/zte_raw_aligned.yaml}"   # the raw-conformer champion (see header)
CTRL_CFG="${CTRL_CFG:-experiments/benchmark/baseline_skipgram_loso.yaml}"  # skip-gram control arm
SPATIAL="${SPATIAL:-exact}"   # build + wire the true ZuCo-105 electrode montage (needs `mne`; degrades gracefully)
MEANING="${MEANING:-keep}"    # leave each config's own meaning target alone
SEEDS="${SEEDS:-42}"          # seed(s) per held-out subject; e.g. "42 43 44" to average out training instability

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

run_one() {   # cfg, holdout, seed
  local cfg="$1" holdout="$2" seed="$3"
  echo "───────────────────────────────────────────────────────────────"
  echo "▶ LOSO hold out ${holdout}  seed ${seed}  ($(basename "${cfg}" .yaml))"
  local backup=()
  [ -n "${DRIVE_BACKUP}" ] && backup=(--drive-backup "${DRIVE_BACKUP}")
  "${PY}" -m zte.cli.run --config "${cfg}" "${SRC[@]}" \
      --loso-holdout "${holdout}" --seed "${seed}" --out-root "${OUT_ROOT}" --resume --skip-explore \
      "${PROVISION[@]+"${PROVISION[@]}"}" "${backup[@]+"${backup[@]}"}"
  local code=$?
  if [ "${code}" = "130" ]; then
    echo "⏸  Paused during ${holdout} (seed ${seed}). Re-run this script to resume exactly here."
    exit 130
  elif [ "${code}" != "0" ]; then
    echo "✗ ${holdout}/s${seed} failed (exit ${code}). Re-run to retry; other runs are unaffected."
    FAILED="${FAILED} $(basename "${cfg}" .yaml)/${holdout}/s${seed}"
  fi
  return 0
}

echo "LOSO sweep · config $(basename "${FULL_CFG}") · out ${OUT_ROOT} · held-out: ${SUBJECTS} · seeds: ${SEEDS}"
[ -n "${DRIVE_BACKUP}" ] && echo "Drive mirror: ${DRIVE_BACKUP} (whole run dir, every stage)"
[ -n "${DATA_CACHE}" ]   && echo "Shared bundle cache: ${DATA_CACHE} (built once, reused per subject)"

for s in ${SUBJECTS}; do
  for seed in ${SEEDS}; do
    [ "${CONTROL:-0}" = "1" ] && run_one "${CTRL_CFG}" "${s}" "${seed}"
    run_one "${FULL_CFG}" "${s}" "${seed}"
  done
done

echo "═══════════════════════════════════════════════════════════════"
if [ -n "${FAILED}" ]; then
  echo "⚠  Completed with failures:${FAILED}"
  echo "   Re-run the same command to retry only those (everything else is skipped instantly)."
else
  echo "✓ LOSO sweep complete."
fi

echo "Aggregating the HONEST held-out trend (not the inflated pooled retrieval) ..."
"${PY}" -m zte.cli.loso_summary --experiments "${OUT_ROOT}" \
    --out "${OUT_ROOT}/LOSO_SUMMARY.md" || true

echo "Building the combined comparison view ..."
"${PY}" -m zte.cli.compare --experiments "${OUT_ROOT}" \
    --out "${OUT_ROOT}/COMPARE.html" --title "ZTE — LOSO (new-brain) trend" || true
if [ -n "${DRIVE_BACKUP}" ]; then
  mkdir -p "${DRIVE_BACKUP}"
  for f in COMPARE.html LOSO_SUMMARY.md LOSO_SUMMARY.csv; do
    [ -f "${OUT_ROOT}/${f}" ] && cp -f "${OUT_ROOT}/${f}" "${DRIVE_BACKUP}/" 2>/dev/null || true
  done
fi
echo "Honest trend  -> ${OUT_ROOT}/LOSO_SUMMARY.md   (the held-out headline + convergence spread)"
echo "Side-by-side  -> ${OUT_ROOT}/COMPARE.html"
