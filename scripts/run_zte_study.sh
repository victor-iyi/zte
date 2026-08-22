#!/usr/bin/env bash
# =============================================================================
# run_zte_study.sh -- the whole ZTE study, end to end, resumable, in one command.
#
# WHAT THIS EXISTS FOR. Every number this project reports has to survive five questions, and each one is a
# stage below. Is the dataset confounded (audit)? Does the encoder reach a stranger's brain, at more than one
# seed (encoder, loso)? Is the win sentence length rather than meaning (rebaseline, and the within-task pools
# every decode reports)? Does the decoder read the brain or recite the corpus (decoder, against seven
# brain-independent controls)? And which lever actually did it (ablation)? The last stage turns all of that
# into one offline page and a set of CSVs (analysis).
#
# THE BOARD AS OF 2026-08-14, on real ZuCo held out on ZAB, so a new run has something to beat:
#
#   exp12_zte_raw_aligned_loZAB_s42   held-out Top-1 0.0114 (8/700, p=9.9e-06), rank percentile 0.9670
#                                     length-stratified (train-fitted) Top-1 0.0471, rank percentile 0.9229
#                                     ±1-word length oracle                      rank percentile 0.9525
#                                     -> the encoder does NOT yet clear the length floor. Diagnostic, gates nothing.
#                                     word retrieval Top-1 0.0040 vs chance 0.0031 -- no word-level content at all.
#                                     bits: identity needs 9.4512, word count gives 5.1422 free, encoder carries 1.4965.
#
# The two new mechanisms this study exists to test are both aimed at those last two lines: token-level lexical
# alignment (experiments/flagship/zte_lexical_raw.yaml) at the word-content gap, and the rate ladder plus
# word-synchronous evidence (experiments/flagship/decode_zte_v2.yaml) at the bit budget and the confound.
#
# PAUSE / RESUME. Every training command carries --resume, so Ctrl-C or a reclaimed Colab VM costs at most the
# epoch in flight. Re-run the SAME command: finished runs are skipped instantly, an interrupted one continues
# from its last checkpoint, and the processed dataset bundle is reused. With DRIVE_BACKUP set, the whole run
# directory is mirrored to Drive after every stage, so nothing lives only on a VM disk.
#
# USAGE
#   bash scripts/run_zte_study.sh                          # real data at res/data/zuco_extracted
#   bash scripts/run_zte_study.sh /path/to/zuco_extracted  # real data elsewhere
#   SMOKE=1 bash scripts/run_zte_study.sh                  # tiny synthetic wiring check (CPU, minutes)
#   STAGES="audit encoder" bash scripts/run_zte_study.sh   # only some stages
#   SEEDS="42 43 44" bash scripts/run_zte_study.sh         # mean +/- sd over three training seeds
#   SUBJECTS="ZAB ZDM" bash scripts/run_zte_study.sh       # restrict the LOSO sweep
#   STAGES=analysis bash scripts/run_zte_study.sh          # re-draw the analysis from what is already on disk
#
# STAGES (default: "audit encoder decoder ablation rebaseline analysis"; add `loso` for the 12-fold sweep)
#   audit      -- model-free confound audit of the dataset. Run it before believing any result.
#   encoder    -- the flagship encoder, every seed. This is what the decoder arms are built over.
#   loso       -- all 12 held-out subjects on the flagship encoder. MULTI-HOUR, and multiplied by SEEDS.
#   decoder    -- the new decoder and its four one-knob arms, over the encoder checkpoint from `encoder`.
#   ablation   -- the feature-ablation table: raw vs band power, harmonics vs indexing, invariance on vs off.
#   rebaseline -- the length-confound audit against every encoder checkpoint. Trains nothing, gates nothing.
#   analysis   -- zte-analyze over everything: one HTML page, the CSV tables and the Markdown summary.
#
# COLAB (train on local disk, keep a live Drive copy of everything):
#   DRIVE_BACKUP="/content/drive/MyDrive/Sharables/ZTE/$(date +%F)/experiments" \
#   DATA_CACHE="/content/drive/MyDrive/Sharables/ZTE/prepared" \
#   bash scripts/run_zte_study.sh /content/zuco_extracted
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

# Python block-buffers stdout when it is not a terminal, so without this a multi-hour run looks frozen.
export PYTHONUNBUFFERED=1

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT="${1:-res/data/zuco_extracted}"
PY="${PY:-.venv/bin/python}"
[ -x "${PY}" ] || PY="python"                       # Colab / system python fallback
OUT_ROOT="${OUT_ROOT:-res/experiments/study}"       # point straight at Drive to write runs there
DRIVE_BACKUP="${DRIVE_BACKUP:-}"                    # mounted Drive folder; mirrors each run dir every stage
DATA_CACHE="${DATA_CACHE:-}"                        # shared PROCESSED-bundle dir; built once, reused everywhere
CACHE_REMOTE="${CACHE_REMOTE:-${ZTE_CACHE_REMOTE:-}}"
export ZTE_CACHE_REMOTE="${CACHE_REMOTE}"
SMOKE="${SMOKE:-0}"
SPATIAL="${SPATIAL:-exact}"                         # build + wire the true ZuCo-105 montage (needs `mne`)
MEANING="${MEANING:-keep}"                          # leave each config's own meaning target alone
SEEDS="${SEEDS:-42 2 10 95}"                          # >=3 seeds, because run-to-run drift here is effect-sized
HOLDOUT="${HOLDOUT:-ZAB}"                           # held-out subject for the single-fold stages
STAGES="${STAGES:-audit encoder decoder ablation rebaseline analysis}"

ENCODER_CFG="${ENCODER_CFG:-experiments/flagship/zte_encoder_v3.yaml}"
DECODER_CFG="${DECODER_CFG:-experiments/flagship/decode_zte_v2.yaml}"
# One knob each against DECODER_CFG, so a win is attributable rather than asserted.
DECODER_ARMS="${DECODER_ARMS:-\
experiments/decoder/decode_v2_pooled.yaml \
experiments/decoder/decode_v2_ladder_only.yaml \
experiments/decoder/decode_v2_evidence_only.yaml \
experiments/decoder/decode_v2_no_length_stage.yaml}"
# One knob each against ENCODER_CFG: the four exp16 mechanisms, the two lexical directions, and the three rows of
# the feature-ablation table.
ABLATION_CFGS="${ABLATION_CFGS:-\
experiments/ablation/exp16_residual_off.yaml \
experiments/ablation/exp16_consensus_off.yaml \
experiments/ablation/exp16_gallery_off.yaml \
experiments/ablation/exp16_gallery_band_off.yaml \
experiments/ablation/exp16_length_projection_off.yaml \
experiments/ablation/exp14_lexical_off.yaml \
experiments/ablation/exp14_lexical_reader_off.yaml \
experiments/ablation/feature_bandpower_mlp.yaml \
experiments/ablation/feature_spatial_off.yaml \
experiments/ablation/feature_invariance_off.yaml}"

ALL_SUBJECTS="ZAB ZDM ZDN ZGW ZJM ZJN ZJS ZKB ZKH ZKW ZMG ZPH"
SYNTH_SUBJECTS="ZAB ZDM ZJN"

# Built once and reused: the montage, the meaning target and the processed bundle are all run-independent.
PROVISION=()
[ -n "${SPATIAL}" ] && PROVISION+=(--spatial "${SPATIAL}")
[ -n "${MEANING}" ] && PROVISION+=(--meaning "${MEANING}")
[ -n "${DATA_CACHE}" ] && PROVISION+=(--data-cache "${DATA_CACHE}")

BACKUP=()
[ -n "${DRIVE_BACKUP}" ] && BACKUP=(--drive-backup "${DRIVE_BACKUP}")

if [ "${SMOKE}" = "1" ]; then
  SRC=(--synthetic --epochs "${EPOCHS:-2}" --device "${DEVICE:-cpu}")
  SUBJECTS="${SUBJECTS:-$SYNTH_SUBJECTS}"
  SEEDS="${SEEDS_SMOKE:-42}"
  echo ">>> SMOKE mode: synthetic data, ${EPOCHS:-2} epochs, CPU. NOTHING from this run is a result."
else
  SRC=(--root "${ROOT}")
  [ -n "${DEVICE:-}" ] && SRC+=(--device "${DEVICE}")
  [ -n "${EPOCHS:-}" ] && SRC+=(--epochs "${EPOCHS}")
  SUBJECTS="${SUBJECTS:-$ALL_SUBJECTS}"
fi

mkdir -p "${OUT_ROOT}"
FAILED=""

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
banner() { echo; echo "═══ $* ═══"; }

# Echoes a config path to train from. In SMOKE mode it first rewrites the config into an offline-safe copy:
# a wiring check must not depend on a HuggingFace download, and the frozen encoders and the 0.5B LM are the
# only parts of this pipeline that reach the network.
resolve_config() {  # config
  if [ "${SMOKE}" != "1" ]; then
    echo "$1"
    return 0
  fi
  local patched
  patched="$(mktemp -t zte_smoke_XXXXXX.yaml)"
  "${PY}" - "$1" "${patched}" <<'PY'
import sys, pathlib, yaml
src, out = sys.argv[1:3]
cfg = yaml.safe_load(pathlib.Path(src).read_text())
cfg.setdefault('objective', {}).update(
    text_source=None, lexical_source=None, meaning_contextual=None, semantic_hard_negatives=False
)
cfg.setdefault('model', {}).update(spatial_encoding='none', spatial_mix=False, grad_checkpoint=False)
cfg.setdefault('dataset', {}).update(montage_csv=None)
cfg.setdefault('train', {}).update(batch_size=8, num_workers=0)
if cfg.get('decoder'):
    cfg['decoder'].update(
        lm_source='tiny', lm_revision=None, lm_cache_dir=None, stage0_epochs=1,
        max_target_tokens=24, max_new_tokens=16, n_permutations=20, rescore_chunk=16,
        rate_stages=2, rate_codes=16,
    )
cfg['run_name'] = f"smoke_{cfg['run_name']}"
pathlib.Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  echo "${patched}"
}

note_failure() {  # label, exit code
  local label="$1" code="$2"
  if [ "${code}" = "130" ]; then
    echo "⏸  Paused during ${label}. Re-run this exact command to continue from here."
    exit 130
  fi
  echo "✗ ${label} failed (exit ${code}). Re-run to retry; every other run is unaffected."
  FAILED="${FAILED} ${label}"
}

# The run directory `zte-run` will write, given a config and the suffixes it appends itself.
run_dir_for() {  # config, holdout-or-empty, seed-or-empty
  local base
  base="$("${PY}" -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$1")"
  [ "${SMOKE}" = "1" ] && base="smoke_${base}"
  [ -n "$2" ] && base="${base}_lo$2"
  [ -n "$3" ] && base="${base}_s$3"
  echo "${OUT_ROOT}/${base}"
}

train_encoder() {  # config, holdout, seed
  local cfg="$1" holdout="$2" seed="$3" label resolved
  label="$(basename "${cfg}" .yaml)/lo${holdout}/s${seed}"
  resolved="$(resolve_config "${cfg}")"
  banner "encoder ${label}"
  "${PY}" -m zte.cli.run --config "${resolved}" "${SRC[@]}" \
    --loso-holdout "${holdout}" --seed "${seed}" --out-root "${OUT_ROOT}" --resume --skip-explore \
    "${PROVISION[@]+"${PROVISION[@]}"}" "${BACKUP[@]+"${BACKUP[@]}"}" \
    || note_failure "encoder ${label}" "$?"
  [ "${resolved}" != "${cfg}" ] && rm -f "${resolved}"
  return 0
}

# A decoder run must NOT take --loso-holdout: that flag forces split=by_subject_loso, which shares all 700
# stimuli between train and val -- the one configuration in which a decoder recites the corpus and scores well.
# The held-out subject is written into a temporary config instead, so the honest four-cell split survives.
train_decoder() {  # config, encoder-checkpoint, holdout, seed
  local cfg="$1" ckpt="$2" holdout="$3" seed="$4" label patched
  label="$(basename "${cfg}" .yaml)/lo${holdout}/s${seed}"
  if [ ! -f "${ckpt}" ]; then
    echo "⚠  Skipping decoder ${label}: no encoder checkpoint at ${ckpt}. Run the 'encoder' stage first."
    FAILED="${FAILED} decoder-${label}(no-encoder)"
    return 0
  fi
  local resolved
  resolved="$(resolve_config "${cfg}")"
  patched="$(mktemp -t zte_decoder_XXXXXX.yaml)"
  "${PY}" - "${resolved}" "${holdout}" "${seed}" "${patched}" <<'PY' || { note_failure "decoder ${label}" "$?"; return 0; }
import sys, yaml, pathlib
cfg_path, holdout, seed, out = sys.argv[1:5]
cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
cfg['train']['loso_holdout_subject'] = holdout
cfg['train']['seed'] = int(seed)
cfg['run_name'] = f"{cfg['run_name']}_lo{holdout}_s{seed}"
pathlib.Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  [ "${resolved}" != "${cfg}" ] && rm -f "${resolved}"
  banner "decoder ${label}"
  "${PY}" -m zte.cli.run --config "${patched}" "${SRC[@]}" \
    --encoder-ckpt "${ckpt}" --out-root "${OUT_ROOT}" --resume --skip-explore \
    "${PROVISION[@]+"${PROVISION[@]}"}" "${BACKUP[@]+"${BACKUP[@]}"}" \
    || note_failure "decoder ${label}" "$?"
  rm -f "${patched}"
}

has_stage() { case " ${STAGES} " in *" $1 "*) return 0;; *) return 1;; esac; }

echo "ZTE study · out ${OUT_ROOT} · stages: ${STAGES} · seeds: ${SEEDS} · holdout: ${HOLDOUT}"
[ -n "${DRIVE_BACKUP}" ] && echo "Drive mirror: ${DRIVE_BACKUP} (whole run dir, every stage)"
[ -n "${DATA_CACHE}" ]   && echo "Shared bundle cache: ${DATA_CACHE} (built once, reused)"

# --------------------------------------------------------------------------- #
# 1. audit -- is the dataset confounded, before any model is trained
# --------------------------------------------------------------------------- #
if has_stage audit; then
  banner "audit -- model-free confound report"
  AUDIT_SRC=(--root "${ROOT}")
  [ "${SMOKE}" = "1" ] && AUDIT_SRC=(--synthetic)
  "${PY}" -m zte.cli.audit "${AUDIT_SRC[@]}" --out "${OUT_ROOT}/confound_audit.md" \
    || note_failure 'audit' "$?"
fi

# --------------------------------------------------------------------------- #
# 2. encoder -- the flagship, one run per seed on the single held-out subject
# --------------------------------------------------------------------------- #
if has_stage encoder; then
  for seed in ${SEEDS}; do
    train_encoder "${ENCODER_CFG}" "${HOLDOUT}" "${seed}"
  done
fi

# --------------------------------------------------------------------------- #
# 3. loso -- every held-out subject on the flagship. Multi-hour, times SEEDS.
# --------------------------------------------------------------------------- #
if has_stage loso; then
  for subject in ${SUBJECTS}; do
    for seed in ${SEEDS}; do
      train_encoder "${ENCODER_CFG}" "${subject}" "${seed}"
    done
  done
  banner 'loso -- the honest held-out trend'
  "${PY}" -m zte.cli.loso_summary --experiments "${OUT_ROOT}" \
    --out "${OUT_ROOT}/LOSO_SUMMARY.md" || true
fi

# --------------------------------------------------------------------------- #
# 4. decoder -- the new decoder and its one-knob arms over the encoder above
# --------------------------------------------------------------------------- #
if has_stage decoder; then
  for seed in ${SEEDS}; do
    CKPT="$(run_dir_for "${ENCODER_CFG}" "${HOLDOUT}" "${seed}")/checkpoints/best.pt"
    train_decoder "${DECODER_CFG}" "${CKPT}" "${HOLDOUT}" "${seed}"
    # The arms run at the first seed only: they answer "which knob", which one seed settles, while the
    # headline needs the spread. Set DECODER_ARM_SEEDS to widen them.
    for arm in ${DECODER_ARMS}; do
      case " ${DECODER_ARM_SEEDS:-$(echo "${SEEDS}" | awk '{print $1}')} " in
        *" ${seed} "*) train_decoder "${arm}" "${CKPT}" "${HOLDOUT}" "${seed}" ;;
      esac
    done
  done
fi

# --------------------------------------------------------------------------- #
# 5. ablation -- one lever at a time, at the first seed
# --------------------------------------------------------------------------- #
if has_stage ablation; then
  ABLATE_SEED="$(echo "${SEEDS}" | awk '{print $1}')"
  for cfg in ${ABLATION_CFGS}; do
    train_encoder "${cfg}" "${HOLDOUT}" "${ABLATE_SEED}"
  done
  # The band-power decoder arm needs the band-power encoder, so it runs after the ablation encoders exist.
  BP_CKPT="$(run_dir_for experiments/ablation/feature_bandpower_mlp.yaml "${HOLDOUT}" "${ABLATE_SEED}")/checkpoints/best.pt"
  train_decoder experiments/decoder/decode_v2_bandpower.yaml "${BP_CKPT}" "${HOLDOUT}" "${ABLATE_SEED}"
fi

# --------------------------------------------------------------------------- #
# 6. rebaseline -- how much of every number is sentence length. Trains nothing.
# --------------------------------------------------------------------------- #
if has_stage rebaseline; then
  REBASE_SRC=(--root "${ROOT}")
  [ "${SMOKE}" = "1" ] && REBASE_SRC=(--synthetic)
  for dir in "${OUT_ROOT}"/*/; do
    ckpt="${dir}checkpoints/best.pt"
    [ -f "${ckpt}" ] || continue
    [ -f "${dir}rebaseline/rebaseline.json" ] && continue      # already audited; resume skips it
    banner "rebaseline $(basename "${dir}")"
    "${PY}" -m zte.cli.rebaseline --ckpt "${ckpt}" "${REBASE_SRC[@]}" \
      --holdout "${HOLDOUT}" --length-tol 1 --oracle-tol 0,1,2,4 --out "${dir}rebaseline" \
      || note_failure "rebaseline $(basename "${dir}")" "$?"
  done
fi

# --------------------------------------------------------------------------- #
# 7. analysis -- everything collected into one page, its tables and its summary
# --------------------------------------------------------------------------- #
if has_stage analysis; then
  banner 'analysis -- the study dashboard'
  MONTAGE=()
  [ -f res/montage_gsn105.csv ] && MONTAGE=(--montage res/montage_gsn105.csv)
  "${PY}" -m zte.cli.analyze --experiments "${OUT_ROOT}" --out "${OUT_ROOT}/analysis" \
    --title "ZTE — study $(date +%F)" "${MONTAGE[@]+"${MONTAGE[@]}"}" \
    || note_failure 'analysis' "$?"
  "${PY}" -m zte.cli.compare --experiments "${OUT_ROOT}" \
    --out "${OUT_ROOT}/analysis/COMPARE.html" --title 'ZTE — arm comparison' || true
fi

# --------------------------------------------------------------------------- #
# Mirror the study-level artifacts to Drive, so nothing lives only on a VM disk
# --------------------------------------------------------------------------- #
if [ -n "${DRIVE_BACKUP}" ]; then
  mkdir -p "${DRIVE_BACKUP}/analysis"
  for f in LOSO_SUMMARY.md LOSO_SUMMARY.csv confound_audit.md confound_audit.json INDEX.md; do
    [ -f "${OUT_ROOT}/${f}" ] && cp -f "${OUT_ROOT}/${f}" "${DRIVE_BACKUP}/" 2>/dev/null || true
  done
  [ -d "${OUT_ROOT}/analysis" ] && cp -Rf "${OUT_ROOT}/analysis/." "${DRIVE_BACKUP}/analysis/" 2>/dev/null || true
  echo "Mirrored the study-level artifacts to ${DRIVE_BACKUP}"
fi

echo
echo "═══════════════════════════════════════════════════════════════"
if [ -n "${FAILED}" ]; then
  echo "⚠  Completed with failures:${FAILED}"
  echo "   Re-run the same command to retry only those; everything finished is skipped instantly."
else
  echo "✓ Study complete."
fi
[ "${SMOKE}" = "1" ] && echo "⚠  SMOKE mode: every number above is synthetic and is a wiring check, not a result."
echo "Analysis   -> ${OUT_ROOT}/analysis/ANALYSIS.html   (open it; everything is inlined, no network needed)"
echo "Summary    -> ${OUT_ROOT}/analysis/ANALYSIS.md"
echo "Tables     -> ${OUT_ROOT}/analysis/tables/*.csv"
[ -f "${OUT_ROOT}/LOSO_SUMMARY.md" ] && echo "LOSO trend -> ${OUT_ROOT}/LOSO_SUMMARY.md"
exit 0
