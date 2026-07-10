"""Generate the ZTE *study* config YAMLs (bias-controlled experiment suite).

Every study config is built here as a `ZTEConfig` object and serialised with `ZTEConfig.to_yaml`, so each file is guaranteed valid against the *current*
schema (no hand-edited YAML that can drift out of sync). A short comment header is prepended to each file explaining the study, its A/B partner and what to look
at. Re-run this any time the schema changes::

    .venv/bin/python scripts/make_study_configs.py

The generated files are the matched variants for the two studies that need new configs -- Study 2 (subject-invariance A/B under LOSO) and Study 3 (VICReg
anti-collapse ablation). Studies 1, 4 and 5 reuse the shipped `exp*` presets or `zte-benchmark`, so they need no new files (see `docs/EXPERIMENTS.md`).

Design invariants shared by every study config (stated in the doc too):
  * `subjects=None`, `tasks=('SR', 'NR')`  -> maximise the dataset (all 12 subjects, both reading tasks).
  * `normalizer_fit='train'`               -> normaliser/imputer fit on train only (no val/test/held-out leakage).
  * `test_fraction > 0`                    -> every headline number is on HELD-OUT data.
  * `deterministic=True`, fixed `seed`     -> reproducible; the runner sweeps seeds 42/43/44 for bootstrap CIs.
  * `include_eye_tracking=False`           -> EEG-only is the honest headline (no gaze artefact) for the invariance / collapse questions.

Only the levers under test differ between an A/B pair; everything else is held identical so any metric delta is attributable to the lever, not a confound.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from zte.config import DatasetConfig, ObjectiveConfig, TrainConfig, ZTEConfig

# Where the study YAMLs are written (next to the shipped exp*.yaml presets).
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / 'experiments'

# One held-out subject for the LOSO study. ZuCo v1 has 12 subjects; rotate this
# across all of them for a full leave-one-subject-out sweep (see the runner /
# docs). ZAB is a safe default present in `res/data/zuco_extracted`.
LOSO_HOLDOUT = 'ZAB'


def _base_dataset(**overrides: object) -> DatasetConfig:
    """A dataset config that maximises the cohort and fits the normaliser on train only.

    All 12 subjects (`subjects=None`), both reading tasks (SR + NR), band-power
    representation, `normalizer_fit='train'`. `include_eye_tracking` and
    `normalize` are left to the caller because they are levers under test.
    """
    base = DatasetConfig(
        root='res/data/zuco_extracted',
        tasks=('SR', 'NR'),
        subjects=None,
        representation='band_power',
        normalizer_fit='train',
    )
    return replace(base, **overrides)


def _base_train(**overrides: object) -> TrainConfig:
    """Shared optimisation: 40 epochs, batch 128, deterministic, held-out test set."""
    base = TrainConfig(
        epochs=40,
        batch_size=128,
        lr=3e-4,
        weight_decay=0.01,
        test_fraction=0.1,
        seed=42,
        deterministic=True,
        tensorboard=True,
    )
    return replace(base, **overrides)


# --------------------------------------------------------------------------- #
# Study 2 -- subject-invariance A/B under leave-one-subject-out (the north star).
#   Matched: skipgram, EEG-only, by_subject_loso, same held-out subject, 40 ep.
#   Differs: the invariance stack only
#            (VICReg var+cov, per-subject norm, cross-subject positives, adversary).
# --------------------------------------------------------------------------- #

_INVAR_BASELINE = ZTEConfig(
    dataset=_base_dataset(include_eye_tracking=False, normalize='zscore_channel'),
    objective=ObjectiveConfig(
        name='skipgram',
        variance_weight=0.0,
        covariance_weight=0.0,
        cross_subject_positives=False,
        subject_adversary_weight=0.0,
    ),
    train=_base_train(split='by_subject_loso', loso_holdout_subject=LOSO_HOLDOUT),
    run_name='study_invariance_baseline_loso',
)

_INVAR_FULL = ZTEConfig(
    dataset=_base_dataset(include_eye_tracking=False, normalize='zscore_subject'),
    objective=ObjectiveConfig(
        name='skipgram',
        variance_weight=1.0,
        covariance_weight=1.0,
        cross_subject_positives=True,
        subject_adversary_weight=0.3,
    ),
    train=_base_train(split='by_subject_loso', loso_holdout_subject=LOSO_HOLDOUT),
    run_name='study_invariance_full_loso',
)

# --------------------------------------------------------------------------- #
# Study 3 -- VICReg anti-collapse ablation.
#   Matched: skipgram, EEG-only, by_stimulus (text never spans train/test),
#            zscore_channel, no adversary, no cross-subject positives, 40 ep.
#   Differs: VICReg variance+covariance weights ONLY (0 -> 1).
# --------------------------------------------------------------------------- #

_VICREG_OFF = ZTEConfig(
    dataset=_base_dataset(include_eye_tracking=False, normalize='zscore_channel'),
    objective=ObjectiveConfig(
        name='skipgram',
        variance_weight=0.0,
        covariance_weight=0.0,
        cross_subject_positives=False,
        subject_adversary_weight=0.0,
    ),
    train=_base_train(split='by_stimulus'),
    run_name='study_vicreg_off',
)

_VICREG_ON = ZTEConfig(
    dataset=_base_dataset(include_eye_tracking=False, normalize='zscore_channel'),
    objective=ObjectiveConfig(
        name='skipgram',
        variance_weight=1.0,
        covariance_weight=1.0,
        cross_subject_positives=False,
        subject_adversary_weight=0.0,
    ),
    train=_base_train(split='by_stimulus'),
    run_name='study_vicreg_on',
)


# Header comments prepended to each YAML (the schema-driven body follows).
_HEADERS: dict[str, str] = {
    'study_invariance_baseline_loso': (
        '# STUDY 2 (A) -- subject-invariance BASELINE, leave-one-subject-out.\n'
        '# Skip-gram, EEG-only, split=by_subject_loso, holdout={h}, NO invariance levers\n'
        '# (variance/covariance=0, no cross-subject positives, no adversary, zscore_channel).\n'
        '# A/B partner: study_invariance_full_loso.yaml. Compare held-out cross-subject\n'
        '# retrieval and subject-decodability (should stay near baseline / above chance here).\n'
        '# Rotate train.loso_holdout_subject across all 12 subjects for a full LOSO sweep.\n'
    ),
    'study_invariance_full_loso': (
        '# STUDY 2 (B) -- subject-invariance FULL stack, leave-one-subject-out.\n'
        '# Skip-gram, EEG-only, split=by_subject_loso, holdout={h}, WITH the invariance stack:\n'
        '# VICReg variance+covariance, per-subject normalisation (zscore_subject),\n'
        '# cross-subject positives, and a gradient-reversal subject adversary (0.3).\n'
        '# A/B partner: study_invariance_baseline_loso.yaml. Discovery question: does the stack\n'
        '# push subject-decodability toward chance AND lift held-out cross-subject retrieval?\n'
        '# Rotate train.loso_holdout_subject across all 12 subjects for a full LOSO sweep.\n'
    ),
    'study_vicreg_off': (
        '# STUDY 3 (A) -- anti-collapse ablation, VICReg OFF.\n'
        '# Skip-gram, EEG-only, split=by_stimulus (same sentence TEXT never spans train/test),\n'
        '# zscore_channel, no adversary/positives. variance_weight=0, covariance_weight=0.\n'
        '# A/B partner: study_vicreg_on.yaml. Expect dimensional collapse: low effective-rank\n'
        '# ratio, few active neurons (n_active), many dead (n_dead), weak lexical decodability.\n'
    ),
    'study_vicreg_on': (
        '# STUDY 3 (B) -- anti-collapse ablation, VICReg ON.\n'
        '# Identical to study_vicreg_off.yaml EXCEPT variance_weight=1, covariance_weight=1.\n'
        '# Discovery question: how many neurons "come alive" (effective rank, n_active) and does\n'
        '# lexical content (word length / frequency) become decodable once collapse is prevented?\n'
    ),
}


def _write(config: ZTEConfig) -> Path:
    """Serialise one study config to YAML and prepend its comment header."""
    path = EXPERIMENTS_DIR / f'{config.run_name}.yaml'
    config.to_yaml(path)
    header = _HEADERS[config.run_name].format(h=LOSO_HOLDOUT)
    path.write_text(header + path.read_text(encoding='utf-8'), encoding='utf-8')
    return path


def main() -> None:
    """Generate every study config and print the paths written."""
    configs = [_INVAR_BASELINE, _INVAR_FULL, _VICREG_OFF, _VICREG_ON]
    for cfg in configs:
        path = _write(cfg)
        print(f'wrote {path.relative_to(EXPERIMENTS_DIR.parent)}')
    print(f'{len(configs)} study configs written to {EXPERIMENTS_DIR}')


if __name__ == '__main__':
    main()
