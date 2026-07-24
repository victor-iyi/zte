"""Generate the ZTE study config YAMLs from `ZTEConfig` objects, so they cannot drift from the schema."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from zte.config import DatasetConfig, ObjectiveConfig, TrainConfig, ZTEConfig

# Where the study YAMLs are written: the ablation tier, alongside the other single-lever studies.
EXPERIMENTS_DIR: Path = Path(__file__).resolve().parent.parent / 'experiments' / 'ablation'

# Rotate across all 12 ZuCo v1 subjects for a full leave-one-subject-out sweep.
LOSO_HOLDOUT: str = 'ZAB'


def _base_dataset(**overrides: object) -> DatasetConfig:
    """A dataset config that maximises the cohort and fits the normaliser on train only.

    `include_eye_tracking` and `normalize` are left to the caller because they are levers under test.
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
        num_workers=-1,  # auto per backend (a few on GPU/TPU/MPS, 0 on CPU)
    )
    return replace(base, **overrides)


# --------------------------------------------------------------------------- #
# Study 2 -- subject-invariance A/B under LOSO; only the invariance stack differs.
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
# Study 3 -- anti-collapse ablation; only the VICReg variance+covariance weights differ.
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
