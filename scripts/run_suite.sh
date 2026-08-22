#!/usr/bin/env bash
#
# run_suite.sh -- the fixed-seed driver for the ZTE experiment suite.
#
# THE BOARD THIS RUNS AGAINST. On 2026-07-25 every run on Drive was re-scored on the HELD-OUT subject
# instead of the pooled set. Pooled retrieval includes the 11 training brains, so it rewards memorising
# them rather than reaching the 12th; it is what made band power look like the champion. Held out on ZAB
# (700 queries, chance 1/700, exact binomial tail):
#
#   | run                          | frontend      | Top-5 hits/700 | p     | eff-rank | subj probe (raw) |
#   | ---------------------------- | ------------- | -------------- | ----- | -------- | ---------------- |
#   | exp8_clip_e5_raw             | raw_conformer |       32       | 7e-16 |  0.264   |  0.45  (0.81)    |
#   | exp10_clip_e5_meaning_raw    | raw_conformer |       32       | 7e-16 |  0.264   |  0.41  (0.81)    |
#   | exp10_..._raw_v2             | raw_conformer |       19       | 1e-06 |  0.535   |  0.36  (0.81)    |
#   | exp9_clip_e5_meaning RETIRED | band_power    |       10       | 3e-02 |  0.160   |  0.23  (0.16)    |
#
# The raw conformer wins by 4x on the same fold. Band power's 0.23 subject probe was never
# disentanglement -- its raw features only score 0.16, so there was nothing to remove, and its
# effective-rank ratio of 0.160 shows the space had collapsed to ~123 of 768 directions. Every
# band-power arm, plus the Qwen/BGE/MPNet text-encoder A/B (all p >= 0.07), is now in
# experiments/archive/ with the number that retired it.
#
# WHAT THE SUITE RUNS NOW: the two measured raw arms, and the exp12 alignment stack built on top of
# them. The raw path had never had cross-subject alignment of ANY kind -- `dataset.normalize` only
# applied to band power, so `normalize: riemannian` was a silent no-op for every raw run above, and
# the winning arm is training on unaligned voltages with a subject probe still at 0.41 of 0.81. exp12
# closes that with three label-free steps (Euclidean alignment, a signature-driven subject adapter,
# and a rank-preserving identity penalty); see experiments/flagship/zte_raw_aligned.yaml.
#
# Usage:
#   bash scripts/run_suite.sh                         # real data (default root)
#   bash scripts/run_suite.sh /path/to/zuco_extracted # a different data root
#   SMOKE=1 bash scripts/run_suite.sh                 # tiny synthetic sanity pass (CPU, minutes)
#   STUDIES="audit flagship" bash scripts/run_suite.sh   # run only some studies
#
# STUDIES (default: "audit flagship ablate"):
#   audit     -- model-free confound audit of the dataset (run this before believing any result)
#   flagship  -- the arms held out on ZAB: the exp12 alignment stack (narrow + wide encoder) against
#                the two raw arms it must beat (exp10 raw+meaning at 32/700, exp8 raw at 32/700)
#   ablate    -- the exp12 one-knob studies: alignment off, adapter off, orthogonality off, and
#                alignment fit on train-only. Each isolates one lever of the new stack.
#   controls  -- the skip-gram baseline, kept as the honest floor (a control, not a contender)
#   benchmark -- objective sweep on top of the champion recipe (zte-benchmark, resumable)
#   loso      -- the full 12-subject LOSO sweep on the champion (multi-hour; delegates to run_loso.sh)
#
# PAUSE / RESUME: every run is launched with `zte-run --resume`, so you can stop at ANY time (Ctrl-C,
# or a reclaimed Colab VM) and re-run the SAME command to continue exactly where you left off.
# Finished runs are skipped instantly, an interrupted run resumes from its last checkpoint, and the
# cached dataset bundle is reused. Alignment is applied AFTER the bundle loads, so enabling it never
# invalidates a prepared bundle. With DRIVE_BACKUP set, the whole run directory is mirrored to Drive
# after every stage, so nothing but the epoch in flight is ever lost.
#
# Anti-bias guarantees are baked into the configs (leakage-aware leave-one-subject-out splits, a
# train-only normaliser, a held-out test subject) and into this runner (fixed seeds -> bootstrap CIs).
set -uo pipefail
cd "$(dirname "$0")/.."

# Stream ZTE's logs live: Python block-buffers stdout when it is not a terminal, so without
# this a long run looks frozen and then dumps everything at once.
export PYTHONUNBUFFERED=1

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT="${1:-res/data/zuco_extracted}"      # data root (positional arg 1)
SEEDS="${SEEDS:-42 2 10 95}"              # default seeds; override e.g. SEEDS="42" (single seed) or SEEDS="42 43 44" (multiple seeds)
PY="${PY:-.venv/bin/python}"              # project venv interpreter
[ -x "${PY}" ] || PY="python"             # Colab / system python fallback
OUT_ROOT="${OUT_ROOT:-res/experiments}"   # where zte-run catalogues each run (a Drive path persists everything)
DRIVE_BACKUP="${DRIVE_BACKUP:-}"          # mounted Drive folder; mirrors each whole run dir every stage
DATA_CACHE="${DATA_CACHE:-}"              # shared local PROCESSED-bundle dir; build once, reuse across every run
# Persistent cache dir (mounted Drive folder) layered behind DATA_CACHE; also honoured via $ZTE_CACHE_REMOTE.
CACHE_REMOTE="${CACHE_REMOTE:-${ZTE_CACHE_REMOTE:-}}"
export ZTE_CACHE_REMOTE="${CACHE_REMOTE}"
SMOKE="${SMOKE:-0}"                       # SMOKE=1 -> tiny synthetic run
SPATIAL="${SPATIAL:-exact}"               # build + wire the true ZuCo-105 montage (needs `mne`; degrades gracefully)
MEANING="${MEANING:-keep}"                # leave each config's own meaning target alone
STUDIES="${STUDIES:-audit flagship ablate}"
HOLDOUT="${HOLDOUT:-ZAB}"                 # held-out subject for the single-fold studies

# The configs, by tier (see experiments/README.md).
# The exp12 alignment stack first, then the two raw arms it has to beat.
FLAGSHIP_CONFIGS="${FLAGSHIP_CONFIGS:-\
experiments/flagship/zte_raw_aligned.yaml \
experiments/flagship/clip_e5_meaning_raw.yaml \
experiments/flagship/clip_e5_raw.yaml}"
# One knob of the exp12 stack each, so a win is attributable rather than asserted.
ABLATE_CONFIGS="${ABLATE_CONFIGS:-\
experiments/ablation/exp12_align_off.yaml \
experiments/ablation/exp12_adapter_off.yaml \
experiments/ablation/exp12_orthogonality_off.yaml \
experiments/ablation/exp12_align_fit_train.yaml}"
# The honest floor. A control, not a contender -- it is here to be beaten, not to win.
CONTROL_CONFIGS="${CONTROL_CONFIGS:-experiments/benchmark/baseline_skipgram_loso.yaml}"
CHAMPION="${CHAMPION:-experiments/flagship/zte_raw_aligned.yaml}"

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
# FLAGSHIP -- the CLIP arms, held out on one subject.
#   All three are tied on the held-out board (2026-08-13, ZAB, 700 queries). Rank percentile with a bootstrap CI is
#   what separates arms here; Top-1 does not, since two seeds of one config move it 9 hits to 8.
#   zte_raw_aligned        : exp12 -- rank percentile 0.9672 (0.9635-0.9708). exp10's encoder byte-for-byte, plus
#                            Euclidean alignment, the subject adapter and the identity-orthogonality penalty --
#                            none of which moves a number: see ablation/exp12_align_off.
#   clip_e5_meaning_raw    : exp10 -- 0.9667 (0.9629-0.9705), and the best Top-5 on the board at 37 hits of 700.
#   clip_e5_raw            : exp8  -- 0.9635 (0.9599-0.9673), the only arm whose length-stratified Top-1 clears
#                            p < 0.05 (0.0443, p 0.012). Worse subject probe, no meaning distillation.
# =========================================================================== #
study_flagship() {
  echo "=== FLAGSHIP: exp12 alignment stack vs the two measured raw arms ==="
  run_configs "${FLAGSHIP_CONFIGS}"
}

# =========================================================================== #
# CONTROLS -- what the flagship must beat to earn its place.
#   baseline_skipgram_loso : the previous SOTA recipe (skip-gram + full invariance stack).
# =========================================================================== #
study_controls() {
  echo "=== CONTROLS: skip-gram baseline (the honest floor) ==="
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
# ABLATE -- one lever of the exp12 stack at a time, each config byte-identical to
# the flagship except for its single knob, so a win can be attributed rather than
# asserted. Run this before claiming the alignment stack is what did it.
# =========================================================================== #
study_ablate() {
  echo "=== ABLATE: exp12 one-knob studies (align / adapter / orthogonality / fit) ==="
  run_configs "${ABLATE_CONFIGS}"
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
    *) echo "Unknown study '${study}' (valid: audit flagship ablate controls benchmark loso)"; exit 2 ;;
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
echo "    ${PY} -m zte.cli.visualize --run ${OUT_ROOT}/exp12_zte_raw_aligned_lo${HOLDOUT}_s42 --kind both"
