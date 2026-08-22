"""`--loso-holdout` refuses to trade a decoder run's honest split, and holdout run names stay distinct."""

import argparse
import logging

import pytest

from zte.cli.run import guard_split_override, resolve_run_name
from zte.config import SplitStrategy, TrainMode, ZTEConfig


def _config(
    mode: TrainMode, applied: SplitStrategy, *, holdout: str | None = 'ZAB', run_name: str = 'arm'
) -> ZTEConfig:
    """Builds the post-override config the guard and the name resolver see."""
    config = ZTEConfig()
    config.run_name = run_name
    config.train.mode = mode
    config.train.split = applied
    config.train.loso_holdout_subject = holdout

    return config


def _args(*, name: str | None = None, loso_holdout: str | None = None, seed: int | None = None) -> argparse.Namespace:
    """The three CLI arguments the run name is derived from."""
    return argparse.Namespace(name=name, loso_holdout=loso_holdout, seed=seed)


def _messages(logger_name: str = 'zte.cli.run') -> tuple[list[str], logging.Handler]:
    """Attaches a collecting handler, since the `zte` logger does not propagate to caplog."""
    collected: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: collected.append(record.getMessage())  # type: ignore[method-assign]
    logging.getLogger(logger_name).addHandler(handler)

    return collected, handler


# --------------------------------------------------------------------------- #
# The refusal
# --------------------------------------------------------------------------- #


def test_the_flag_refuses_a_decoder_run_that_asked_for_the_honest_split() -> None:
    """A decoder run whose split the flag replaced exits instead of training hours it can never headline."""
    config = _config('decoder', 'by_subject_loso')

    with pytest.raises(SystemExit) as excinfo:
        guard_split_override(config, 'by_subject_and_stimulus')

    message = str(excinfo.value)
    assert 'by_subject_and_stimulus' in message  # what the YAML asked for
    assert 'by_subject_loso' in message  # what the flag forced
    assert 'honest_split' in message  # the verdict clause it costs
    assert 'train.loso_holdout_subject: ZAB' in message  # the remedy
    assert 'drop --loso-holdout' in message
    assert '--allow-closed-set' in message  # the opt-out


def test_a_joint_run_is_refused_on_the_same_terms() -> None:
    """`joint` trains the same decoder, so it loses the same headline to the same swap."""
    with pytest.raises(SystemExit):
        guard_split_override(_config('joint', 'by_subject_loso'), 'by_subject_and_stimulus')


def test_the_opt_out_downgrades_the_refusal_to_a_warning() -> None:
    """`--allow-closed-set` keeps the deliberate closed-set control reachable, loudly."""
    collected, handler = _messages()
    try:
        fired = guard_split_override(_config('decoder', 'by_subject_loso'), 'by_sentence', allow_closed_set=True)
    finally:
        logging.getLogger('zte.cli.run').removeHandler(handler)

    assert fired
    assert any('honest_split' in message for message in collected)


def test_an_encoder_run_is_never_refused() -> None:
    """`by_subject_loso` is the encoder's own north-star split, so the swap costs it nothing."""
    assert guard_split_override(_config('encoder', 'by_subject_loso'), 'by_subject_and_stimulus') is False


def test_a_decoder_run_that_asked_for_by_subject_loso_is_not_refused() -> None:
    """The guard refuses a split the flag replaced, never one the config deliberately named."""
    assert guard_split_override(_config('decoder', 'by_subject_loso'), 'by_subject_loso') is False


def test_a_decoder_run_keeping_its_honest_split_is_not_refused() -> None:
    """Without the flag the honest split survives, and the guard has nothing to say."""
    assert guard_split_override(_config('decoder', 'by_subject_and_stimulus'), 'by_subject_and_stimulus') is False


# --------------------------------------------------------------------------- #
# The run-name suffix
# --------------------------------------------------------------------------- #


def test_a_config_named_holdout_gets_the_same_suffix_the_flag_would_have_given() -> None:
    """Dropping the flag must not merge two folds of a decoder sweep into one run directory."""
    assert resolve_run_name(_config('decoder', 'by_subject_and_stimulus', holdout='ZDM'), _args()) == 'arm_loZDM'


def test_the_config_named_suffix_composes_with_the_seed_suffix() -> None:
    """A multi-seed sweep over a held-out subject keeps one directory per (subject, seed)."""
    config = _config('decoder', 'by_subject_and_stimulus', holdout='ZAB')
    assert resolve_run_name(config, _args(seed=43)) == 'arm_loZAB_s43'


def test_a_run_name_already_carrying_its_holdout_is_not_suffixed_twice() -> None:
    """A pre-patched config names its own fold, and doubling the suffix would strand the resumable directory."""
    config = _config('decoder', 'by_subject_and_stimulus', holdout='ZAB', run_name='arm_loZAB_s42')
    assert resolve_run_name(config, _args()) == 'arm_loZAB_s42'


def test_an_encoder_run_keeps_its_config_run_name() -> None:
    """Every catalogued encoder run is named from the flag or `--name`, so the config holdout adds nothing."""
    assert resolve_run_name(_config('encoder', 'by_subject_loso', holdout='ZAB'), _args()) == 'arm'


def test_the_flag_still_names_the_fold_for_an_encoder_run() -> None:
    """The LOSO sweep's per-subject directories come from the flag and must keep coming from it."""
    config = _config('encoder', 'by_subject_loso', holdout='ZAB')
    assert resolve_run_name(config, _args(loso_holdout='ZDM', seed=42)) == 'arm_loZDM_s42'


def test_explicit_name_wins_over_every_suffix() -> None:
    """`--name` is the run directory verbatim, so a caller composing its own name is never second-guessed."""
    config = _config('decoder', 'by_subject_and_stimulus', holdout='ZAB')
    assert resolve_run_name(config, _args(name='chosen', loso_holdout='ZDM', seed=42)) == 'chosen'


def test_no_holdout_anywhere_leaves_the_name_alone() -> None:
    """A split that holds out no subject has no fold to name."""
    assert resolve_run_name(_config('decoder', 'by_stimulus', holdout=None), _args()) == 'arm'
