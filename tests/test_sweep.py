"""Tests for the campaign planner and its `zte-colab sweep` bridge: order, doneness, and what to train next."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from zte.cli.colab import main
from zte.cli.support.sweep import (
    FOLDS,
    LEVELS,
    MECHANISM_HOLDOUT,
    METRICS_ARTIFACT,
    RUN_HOURS,
    SPREAD_SEEDS,
    TIERS,
    PlannedRun,
    Regime,
    arm,
    hours,
    next_run,
    plan,
    progress,
    resolve_tiers,
    resolved_run_name,
    status,
)
from zte.data.schema import SUBJECTS_V1, Task
from zte.logging_utils import configure_logging
from zte.parallax.study import arm_run_name, run_dir_name

# The campaign as commissioned: 12 mechanism runs, 36 folds, 6 reseeds.
TIER_COUNTS: Final[dict[str, int]] = {'mechanism': 12, 'power': 36, 'spread': 6}
"""Planned runs each tier owes."""

# 54 plan rows over 51 directories -- tier 1's ZAB fold is tier 0's combined arm -- at the measured per-run hours.
CAMPAIGN_HOURS: Final[float] = 108.8
"""Wall-clock hours the whole campaign costs."""


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """`main()` reroutes the `zte` logger to stderr, which would otherwise outlive the test that asked for it."""
    yield
    configure_logging()


def _evaluate(root: Path, run_name: str) -> Path:
    """Writes the evaluated metrics that let the campaign skip a run."""
    path = root / run_name / METRICS_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'scoreboard': {'held_out_retrieval': {'top1': 0.004}}}), encoding='utf-8')

    return path


def _of_tier(runs: list[PlannedRun], tier: str) -> list[PlannedRun]:
    """Every planned run one tier asked for."""
    return [run for run in runs if run.tier == tier]


# ---- The plan ---- #


def test_the_plan_is_the_same_list_every_time_it_is_asked_for() -> None:
    """A campaign whose run order moved between calls could never be resumed from where it stopped."""
    first, second = plan(), plan()

    assert [run.as_dict() for run in first] == [run.as_dict() for run in second]


def test_each_tier_plans_the_runs_it_was_commissioned_for() -> None:
    """The counts are the campaign: 3 levels x 4 arms, then 3 levels x 12 folds, then 3 levels x 2 reseeds."""
    runs = plan()

    assert len(runs) == sum(TIER_COUNTS.values()) == 54
    assert {tier: len(_of_tier(runs, tier)) for tier in TIERS} == TIER_COUNTS


def test_a_tier_is_planned_completely_before_the_next_one_starts() -> None:
    """Every stopping point has to be a complete table, and a tier interleaved with the next never is."""
    runs = plan()

    assert [run.tier for run in runs] == [tier for tier in TIERS for _ in range(TIER_COUNTS[tier])]


def test_the_level_varies_fastest_so_any_prefix_of_the_plan_is_a_matched_comparison() -> None:
    """The alignment level is the campaign's one contrast; a level-major order would finish one and strand two."""
    runs = plan()

    assert [run.level for run in runs[: len(LEVELS)]] == list(LEVELS)
    assert {run.holdout for run in runs[: len(LEVELS)]} == {MECHANISM_HOLDOUT}
    assert len({(run.regime, run.task) for run in runs[: len(LEVELS)]}) == 1


def test_the_combined_arm_leads_because_the_later_tiers_train_nothing_else() -> None:
    """Tiers 1 and 2 are combined-only, so its level table is the earliest result the campaign can build on."""
    runs = plan()

    assert [run.regime for run in runs[: len(LEVELS)]] == ['combined'] * len(LEVELS)
    assert all(run.task is None for run in runs[: len(LEVELS)])


def test_tier_zero_holds_out_one_subject_at_one_seed_across_every_arm() -> None:
    """Its arms have to differ in the level and nothing else, or the mechanism question is not being asked."""
    mechanism = _of_tier(plan(), 'mechanism')

    assert {run.holdout for run in mechanism} == {MECHANISM_HOLDOUT}
    assert {run.seed for run in mechanism} == {42}
    assert sorted({(run.regime, run.task) for run in mechanism}) == [
        ('combined', None),
        ('parallax', 'NR'),
        ('parallax', 'SR'),
        ('parallax', 'TSR'),
    ]


def test_tier_one_sweeps_every_zuco_subject_as_a_holdout() -> None:
    """A held-out lift quoted from a partial LOSO is not the honest number `zte-loso-summary` reports."""
    power = _of_tier(plan(), 'power')

    assert set(FOLDS) == set(SUBJECTS_V1)
    assert len(FOLDS) == 12
    assert {run.holdout for run in power} == set(FOLDS)
    assert all(run.regime == 'combined' and run.seed == 42 for run in power)


def test_tier_two_reseeds_around_the_seed_tier_zero_already_ran() -> None:
    """Seed 42 at that holdout IS tier 0's combined arm; replanning it would count one training twice."""
    spread = _of_tier(plan(), 'spread')

    assert {run.seed for run in spread} == set(SPREAD_SEEDS)
    assert 42 not in {run.seed for run in spread}
    assert {run.holdout for run in spread} == {MECHANISM_HOLDOUT}


def test_a_narrowed_plan_keeps_the_campaign_order_whatever_order_it_was_asked_in() -> None:
    """A notebook passing tiers by number, out of order, must not reorder the campaign."""
    runs = plan(resolve_tiers(['2', 'mechanism']), levels=['sentence'], regimes=['combined'])

    assert [run.tier for run in runs] == ['mechanism', 'spread', 'spread']
    assert {run.level for run in runs} == {'sentence'}


def test_an_unknown_tier_is_named_rather_than_silently_dropped() -> None:
    """Dropping it would plan a smaller campaign than was asked for and report it as the whole one."""
    with pytest.raises(ValueError, match='names no tier'):
        resolve_tiers(['mechanism', 'tier3'])


# ---- The run directory a plan row resumes into ---- #


def test_a_planned_run_names_the_directory_zte_run_would_write() -> None:
    """A run directory is named by its `run_name`, so a mismatch here resumes nothing and retrains everything."""
    assert resolved_run_name(arm_run_name('NR'), 'ZAB', 42) == run_dir_name('NR', 'ZAB', 42)
    assert resolved_run_name('align_token_combined', 'ZDM', 43) == 'align_token_combined_loZDM_s43'


def test_every_planned_run_names_a_config_and_the_name_inside_it() -> None:
    """One plan row has to name both the file to train and the directory that run resumes into."""
    config, name = arm('token', 'parallax', 'NR')

    assert config == Path('experiments/alignment/token/nr.yaml')
    assert name == 'align_token_nr'
    assert arm('word', 'combined') == (Path('experiments/alignment/word/combined.yaml'), 'align_word_combined')


def test_every_config_the_plan_names_exists_and_carries_that_run_name() -> None:
    """A plan row pointing at a file that is not there, or at one whose `run_name` differs, resumes nothing."""
    import yaml

    root = Path(__file__).resolve().parents[1]
    for row in plan():
        config = root / row.config
        assert config.is_file(), f'{row.config} is planned but not on disk'
        assert yaml.safe_load(config.read_text(encoding='utf-8'))['run_name'] == arm(row.level, row.regime, row.task)[1]


@pytest.mark.parametrize(
    ('regime', 'task', 'match'),
    [('parallax', None, 'single task'), ('combined', 'NR', 'no task')],
)
def test_an_arm_that_does_not_describe_a_run_is_refused(regime: Regime, task: Task | None, match: str) -> None:
    """A combined arm given a task, or a parallax arm given none, would name a config that trains something else."""
    with pytest.raises(ValueError, match=match):
        arm('token', regime, task)


def test_the_tiers_that_share_a_run_directory_share_one_training() -> None:
    """Tier 1's ZAB fold and tier 0's combined arm are the same run; planning both must not train it twice."""
    runs = plan()
    shared = {run.run_name for run in _of_tier(runs, 'mechanism')} & {run.run_name for run in _of_tier(runs, 'power')}

    assert len(shared) == len(LEVELS)
    assert len({run.run_name for run in runs}) == len(runs) - len(LEVELS) == 51


# ---- Doneness ---- #


def test_a_run_is_done_when_its_metrics_exist_and_not_before(tmp_path: Path) -> None:
    """Keyed on the catalogue instead, a run that died between its metrics and its INDEX row would be retrained."""
    runs = plan(['mechanism'], levels=['sentence'], regimes=['combined'])
    run = runs[0]
    (tmp_path / run.run_name / 'checkpoints').mkdir(parents=True)
    (tmp_path / 'INDEX.md').write_text(f'| {run.run_name} | done |\n', encoding='utf-8')

    assert status(runs, [tmp_path])[0].done is False

    metrics = _evaluate(tmp_path, run.run_name)
    state = status(runs, [tmp_path])[0]

    assert state.done is True
    assert state.metrics == metrics


def test_a_run_evaluated_on_drive_is_read_from_drive_rather_than_the_local_copy(tmp_path: Path) -> None:
    """A reclaimed VM's local tree is a stale mirror of Drive, and Drive is where the campaign's record lives."""
    drive, local = tmp_path / '2026-08-22' / 'experiments', tmp_path / 'res' / 'experiments'
    runs = plan(['mechanism'], levels=['sentence'], regimes=['combined'])
    on_drive = _evaluate(drive, runs[0].run_name)
    _evaluate(local, runs[0].run_name)

    state = status(runs, [drive, local])[0]

    assert state.metrics == on_drive
    assert state.done is True


def test_a_run_missing_from_every_root_is_not_done(tmp_path: Path) -> None:
    """An unreachable Drive must read as work still owed, never as work that landed."""
    runs = plan(['mechanism'], levels=['sentence'], regimes=['combined'])

    assert [state.done for state in status(runs, [tmp_path / 'nowhere', tmp_path])] == [False]


# ---- What to train next ---- #


def test_next_run_skips_what_already_landed(tmp_path: Path) -> None:
    """Re-running the campaign after a lost VM has to pick up where it stopped; that is the whole point."""
    runs = plan()
    for run in runs[:5]:
        _evaluate(tmp_path, run.run_name)

    upcoming = next_run(status(runs, [tmp_path]))

    assert upcoming is not None
    assert upcoming.run_name == runs[5].run_name


def test_next_run_is_none_once_every_planned_run_has_landed(tmp_path: Path) -> None:
    """A finished campaign must say so rather than handing back a run that would be trained a second time."""
    runs = plan(['spread'])
    for run in runs:
        _evaluate(tmp_path, run.run_name)

    assert next_run(status(runs, [tmp_path])) is None


def test_a_shared_run_directory_completes_both_tiers_that_planned_it(tmp_path: Path) -> None:
    """Tier 0's combined arm is tier 1's ZAB fold, so training it once must satisfy the plan row in each."""
    runs = plan(levels=['sentence'])
    combined = next(run for run in runs if run.tier == 'mechanism' and run.regime == 'combined')
    _evaluate(tmp_path, combined.run_name)

    done = [state.run.tier for state in status(runs, [tmp_path]) if state.done]

    assert done == ['mechanism', 'power']


# ---- Progress and the ETA ---- #


def test_the_campaign_costs_the_hours_it_was_budgeted_at() -> None:
    """A run directory two tiers share is one training, so charging it twice would overstate the ETA by 7 hours."""
    runs = plan()

    assert hours(runs) == pytest.approx(CAMPAIGN_HOURS)
    assert progress(status(runs, [])).hours_total == pytest.approx(hours(runs))
    assert hours(runs) < sum(RUN_HOURS[run.level][run.regime] for run in runs)


def test_progress_counts_every_planned_row_and_charges_every_run_once(tmp_path: Path) -> None:
    """The per-tier table shows the rows a tier owes; the ETA shows the hours the campaign actually burns."""
    runs = plan()
    summary = progress(status(runs, [tmp_path]))

    assert {tier.tier: tier.total for tier in summary.tiers} == TIER_COUNTS
    assert summary.done == 0
    assert summary.hours_remaining == pytest.approx(CAMPAIGN_HOURS)
    assert sum(tier.hours_total for tier in summary.tiers) == pytest.approx(summary.hours_total)


def test_a_landed_run_stops_owing_its_hours(tmp_path: Path) -> None:
    """An ETA that does not fall as runs land is not an ETA."""
    runs = plan(['mechanism'], levels=['token'], regimes=['combined'])
    _evaluate(tmp_path, runs[0].run_name)

    summary = progress(status(runs, [tmp_path]))

    assert (summary.done, summary.total) == (1, 1)
    assert summary.hours_remaining == pytest.approx(0.0)
    assert summary.hours_total == pytest.approx(RUN_HOURS['token']['combined'])


# ---- The notebook's bridge ---- #


@pytest.mark.parametrize('verb', ['plan', 'next', 'status'])
def test_every_sweep_verb_prints_one_json_object_on_stdout(
    verb: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`zte-colab` is the only bridge the notebook has, and a kernel `json.loads` the whole stream."""
    argv = [
        'zte-colab',
        'sweep',
        verb,
        '--drive',
        str(tmp_path / 'drive'),
        '--experiments',
        str(tmp_path / 'runs'),
        '--out-root',
        str(tmp_path / 'out'),
        '--log-level',
        'DEBUG',
    ]
    monkeypatch.setattr('sys.argv', argv)

    main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert isinstance(payload, dict)
    assert payload['verb'] == verb
    assert payload['planned'] == 54
    assert payload['distinct'] == 51
    assert payload['hours'] == pytest.approx(CAMPAIGN_HOURS)
    assert 'Campaign' not in captured.out


def test_the_plan_verb_answers_without_reading_a_single_run_root(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """It has to answer on a VM with no Drive mounted, before anything has been trained."""
    monkeypatch.setattr('sys.argv', ['zte-colab', 'sweep', 'plan', '--tiers', '0'])

    main()

    payload = json.loads(capsys.readouterr().out)

    assert payload['tiers'] == ['mechanism']
    assert len(payload['runs']) == TIER_COUNTS['mechanism']
    assert payload['runs'][0]['config'].endswith('.yaml')
    assert 'roots' not in payload


def test_the_next_verb_names_the_run_and_the_directory_it_will_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The notebook trains what this names, so the row has to carry the config, the fold, the seed and the path."""
    out = tmp_path / 'out'
    runs = plan()
    _evaluate(out, runs[0].run_name)
    drive, local = tmp_path / 'drive', tmp_path / 'runs'
    argv = ['zte-colab', 'sweep', 'next', '--drive', str(drive), '--experiments', str(local), '--out-root', str(out)]
    monkeypatch.setattr('sys.argv', argv)

    main()

    captured = capsys.readouterr()
    upcoming = json.loads(captured.out)['next']

    assert upcoming['run_name'] == runs[1].run_name
    assert upcoming['out_dir'] == str(out / runs[1].run_name)
    assert (upcoming['holdout'], upcoming['seed']) == (MECHANISM_HOLDOUT, 42)
    assert 'Campaign' in captured.err


def test_the_status_verb_reports_doneness_per_run_beside_the_eta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A campaign table is only reportable if every row says whether it landed and where it was read from."""
    out = tmp_path / 'out'
    runs = plan(['mechanism'], levels=['sentence'])
    for run in runs[:2]:
        _evaluate(out, run.run_name)
    argv = [
        'zte-colab',
        'sweep',
        'status',
        '--tiers',
        '0',
        '--levels',
        'sentence',
        '--drive',
        str(tmp_path / 'drive'),
        '--experiments',
        str(tmp_path / 'runs'),
        '--out-root',
        str(out),
    ]
    monkeypatch.setattr('sys.argv', argv)

    main()

    payload = json.loads(capsys.readouterr().out)

    assert [row['done'] for row in payload['runs']] == [True, True, False, False]
    assert payload['runs'][0]['metrics'] == str(out / runs[0].run_name / METRICS_ARTIFACT)
    assert payload['runs'][-1]['metrics'] is None
    assert payload['progress']['done'] == 2
    assert payload['progress']['tiers'][0]['total'] == 4
