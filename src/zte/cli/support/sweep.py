"""The campaign planner: the tiered sweep as an ordered list, what already landed, and what is left to burn."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from zte.data.schema import Task
from zte.parallax.study import PARALLAX_TASKS

type Level = Literal['token', 'word', 'sentence']
"""Granularity the encoder's alignment loss is applied at."""

type Regime = Literal['parallax', 'combined']
"""Whether an arm trains one task on its own or SR+NR together."""

type Tier = Literal['mechanism', 'power', 'spread']
"""A stopping point of the campaign, each one a complete table on its own."""

# The level is the campaign's one contrast, so it is the innermost loop everywhere below: every prefix of the
# plan is then a matched comparison across all three rather than one level finished and two untouched.
LEVELS: Final[tuple[Level, ...]] = ('token', 'word', 'sentence')
"""The three alignment levels the campaign contrasts."""

# Combined leads because the later tiers train nothing else: its level table is the earliest complete result,
# and the one the rest of the campaign is built on.
REGIMES: Final[tuple[Regime, ...]] = ('combined', 'parallax')
"""The two training regimes, in the order the plan runs them."""

# Ordered by the question each answers -- does the mechanism move anything, does it survive twelve folds, does
# it survive a reseed -- and a later tier is only worth its hours once the earlier one has answered.
TIERS: Final[tuple[Tier, ...]] = ('mechanism', 'power', 'spread')
"""The campaign's tiers, in the order they must run."""

# The campaign is discussed by tier number and planned by tier name, so both spell the same tier.
TIER_ALIASES: Final[dict[str, Tier]] = {
    '0': 'mechanism',
    'mechanism': 'mechanism',
    '1': 'power',
    'power': 'power',
    '2': 'spread',
    'spread': 'spread',
}
"""Every accepted spelling of a tier, mapped onto the planner's name for it."""

# The campaign's fold order rather than the schema's alphabetical one: ZAB leads so a tier-1 sweep cut short
# still shares its holdout with the other two tiers and the three tables stay comparable.
FOLDS: Final[tuple[str, ...]] = (
    'ZAB',
    'ZDM',
    'ZGW',
    'ZJM',
    'ZJN',
    'ZJS',
    'ZKB',
    'ZKH',
    'ZKW',
    'ZMG',
    'ZPH',
    'ZDN',
)
"""The twelve ZuCo subjects, in the order tier 1 holds them out."""

# One holdout across tiers 0 and 2 is what lets their arms differ in the level and the seed and nothing else.
MECHANISM_HOLDOUT: Final[str] = 'ZAB'
"""Subject tiers 0 and 2 hold out."""

CAMPAIGN_SEED: Final[int] = 42
"""Seed every tier-0 and tier-1 run trains at."""

# Seed 42 at this holdout is tier 0's combined arm, so planning it here would name a run directory that tier
# already owns and charge one training twice.
SPREAD_SEEDS: Final[tuple[int, ...]] = (43, 44)
"""The reseeds tier 2 adds on top of tier 0's combined arm."""

# Measured wall-clock, not a guess: a combined arm reads SR+NR together and so costs about twice its
# single-task counterpart at the same level.
RUN_HOURS: Final[dict[Level, dict[Regime, float]]] = {
    'token': {'parallax': 1.90, 'combined': 3.30},
    'word': {'parallax': 0.90, 'combined': 1.95},
    'sentence': {'parallax': 0.80, 'combined': 1.75},
}
"""Hours one run takes, per level and regime."""

# Doneness is the metrics file and never the session INDEX.md: a run that died between writing its metrics and
# its catalogue row is finished, and keying on the catalogue would spend its hours a second time.
METRICS_ARTIFACT: Final[Path] = Path('evaluation') / 'metrics.json'
"""Artifact whose existence marks a planned run as done."""

# One row of the plan names both the file to train and the run directory it resumes into, so the arm name is the
# `run_name` inside the config rather than the config's own stem.
ARM_NAME: Final[str] = 'align_{level}_{arm}'
"""Arm name of an encoder at one alignment level, `arm` being a task code or `combined`."""

# The three levels live one directory each, so a level's four arms read as a set and the diff between two levels
# is the diff between two directories.
ARM_ROOT: Final[Path] = Path('experiments/alignment')
"""Directory holding one subdirectory per alignment level."""


def arm(level: Level, regime: Regime, task: Task | None = None) -> tuple[Path, str]:
    """Returns the config path and the arm name for one campaign arm.

    Args:
        level (Level): The alignment level the arm trains at.
        regime (Regime): Whether the arm trains one task alone or SR+NR together.
        task (Task | None, optional): The task a parallax arm trains on. Defaults to None, for a combined arm.

    Returns:
        tuple[Path, str]: The config YAML, and the `run_name` inside it.

    Raises:
        ValueError: If `task` is missing for a parallax arm, or supplied for a combined one.
    """
    if regime == 'parallax' and task is None:
        raise ValueError('A parallax arm trains a single task, so `task` is required.')

    if regime == 'combined' and task is not None:
        raise ValueError(f'A combined arm trains SR+NR together and takes no task (got {task!r}).')

    stem = 'combined' if task is None else task.lower()

    return ARM_ROOT / level / f'{stem}.yaml', ARM_NAME.format(level=level, arm=stem)


def resolved_run_name(arm_name: str, holdout: str, seed: int) -> str:
    """Returns the run directory `zte-run` writes for an arm at one holdout and seed.

    Args:
        arm_name (str): The `run_name` inside the arm's config.
        holdout (str): The LOSO holdout subject code.
        seed (int): The training seed.

    Returns:
        str: `<arm>_lo<holdout>_s<seed>` -- the suffixes `zte-run` and the parallax study both apply.
    """
    return f'{arm_name}_lo{holdout}_s{seed}'


def resolve_tiers(values: Sequence[str]) -> tuple[Tier, ...]:
    """Reads tier names or campaign numbers, and returns them in the order the campaign runs them.

    Args:
        values (Sequence[str]): Tiers as named or numbered, in any order.

    Returns:
        tuple[Tier, ...]: The requested tiers, deduplicated and ordered by `TIERS`.

    Raises:
        ValueError: If a value names no tier.
    """
    wanted: set[Tier] = set()
    for value in values:
        if (tier := TIER_ALIASES.get(str(value))) is None:
            raise ValueError(f'{value!r} names no tier; expected one of {sorted(TIER_ALIASES)}.')

        wanted.add(tier)

    return tuple(tier for tier in TIERS if tier in wanted)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannedRun:
    """One planned training run: the arm, the fold and seed it trains at, and the directory it resumes into.

    Attributes:
        tier (Tier): The campaign tier that asked for this run.
        level (Level): The alignment level the arm trains at.
        regime (Regime): Whether the arm trains one task alone or SR+NR together.
        task (Task | None): The parallax arm's task, or None for a combined arm.
        holdout (str): The LOSO holdout subject.
        seed (int): The training seed.
        config (Path): The experiment YAML this run trains.
        run_name (str): The run directory `zte-run` resolves for it.
    """

    tier: Tier
    level: Level
    regime: Regime
    task: Task | None
    holdout: str
    seed: int
    config: Path
    run_name: str

    @property
    def hours(self) -> float:
        """Measured wall-clock hours this run takes."""
        return RUN_HOURS[self.level][self.regime]

    def as_dict(self) -> dict[str, Any]:
        """The planned run as JSON-safe fields, in the order a campaign table reads them.

        Returns:
            dict[str, Any]: Every field of the run, with `config` as a string.
        """
        return {
            'tier': self.tier,
            'level': self.level,
            'regime': self.regime,
            'task': self.task,
            'holdout': self.holdout,
            'seed': self.seed,
            'config': str(self.config),
            'run_name': self.run_name,
            'hours': self.hours,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RunStatus:
    """One planned run, and the evaluated metrics that let the campaign skip it.

    Attributes:
        run (PlannedRun): The planned run this reports on.
        metrics (Path | None): The `evaluation/metrics.json` that was found, or None when there is none.
    """

    run: PlannedRun
    metrics: Path | None

    @property
    def done(self) -> bool:
        """Whether this run left evaluated metrics under one of the search roots."""
        return self.metrics is not None

    def as_dict(self) -> dict[str, Any]:
        """The planned run plus its doneness.

        Returns:
            dict[str, Any]: `PlannedRun.as_dict` extended with `done` and the metrics path.
        """
        return {**self.run.as_dict(), 'done': self.done, 'metrics': str(self.metrics) if self.metrics else None}


@dataclass(frozen=True, slots=True, kw_only=True)
class TierProgress:
    """How much of one tier has landed, and the hours it still owes.

    Attributes:
        tier (Tier): The tier reported on.
        done (int): Planned runs of this tier that already have metrics.
        total (int): Planned runs of this tier.
        hours_remaining (float): Hours of this tier's runs that have not landed.
        hours_total (float): Hours this tier costs, charging a shared run directory to its first tier only.
    """

    tier: Tier
    done: int
    total: int
    hours_remaining: float
    hours_total: float

    def as_dict(self) -> dict[str, Any]:
        """This tier's counts and hours.

        Returns:
            dict[str, Any]: Every field, JSON-safe.
        """
        return {
            'tier': self.tier,
            'done': self.done,
            'total': self.total,
            'hours_remaining': self.hours_remaining,
            'hours_total': self.hours_total,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Progress:
    """The campaign's done/total per tier and the hours still to burn.

    Attributes:
        tiers (tuple[TierProgress, ...]): Per-tier progress, in campaign order.
        done (int): Planned runs that already have metrics.
        total (int): Planned runs.
        hours_remaining (float): The ETA -- hours of training the campaign still owes.
        hours_total (float): Hours the whole plan costs, counting a shared run directory once.
    """

    tiers: tuple[TierProgress, ...]
    done: int
    total: int
    hours_remaining: float
    hours_total: float

    def as_dict(self) -> dict[str, Any]:
        """The progress summary as a JSON-safe payload.

        Returns:
            dict[str, Any]: The campaign totals with a per-tier breakdown under `tiers`.
        """
        return {
            'done': self.done,
            'total': self.total,
            'hours_remaining': self.hours_remaining,
            'hours_total': self.hours_total,
            'tiers': [tier.as_dict() for tier in self.tiers],
        }


def _planned(
    *,
    tier: Tier,
    level: Level,
    regime: Regime,
    task: Task | None,
    holdout: str,
    seed: int,
) -> PlannedRun:
    """One plan row, with the config and the run directory `zte-run` resolves for it."""
    config, name = arm(level, regime, task)

    return PlannedRun(
        tier=tier,
        level=level,
        regime=regime,
        task=task,
        holdout=holdout,
        seed=seed,
        config=config,
        run_name=resolved_run_name(name, holdout, seed),
    )


def _mechanism(levels: Sequence[Level], regimes: Sequence[Regime]) -> list[PlannedRun]:
    """Tier 0: every arm at every level, on one holdout and one seed."""
    runs: list[PlannedRun] = []
    for regime in regimes:
        tasks: tuple[Task | None, ...] = PARALLAX_TASKS if regime == 'parallax' else (None,)
        for task in tasks:
            runs += [
                _planned(
                    tier='mechanism',
                    level=level,
                    regime=regime,
                    task=task,
                    holdout=MECHANISM_HOLDOUT,
                    seed=CAMPAIGN_SEED,
                )
                for level in levels
            ]

    return runs


def _power(levels: Sequence[Level], regimes: Sequence[Regime]) -> list[PlannedRun]:
    """Tier 1: the combined arm across all twelve folds, so the held-out lift has a sweep behind it."""
    if 'combined' not in regimes:
        return []

    return [
        _planned(tier='power', level=level, regime='combined', task=None, holdout=fold, seed=CAMPAIGN_SEED)
        for fold in FOLDS
        for level in levels
    ]


def _spread(levels: Sequence[Level], regimes: Sequence[Regime]) -> list[PlannedRun]:
    """Tier 2: the combined arm reseeded, so run-to-run noise can be read off rather than assumed away."""
    if 'combined' not in regimes:
        return []

    return [
        _planned(tier='spread', level=level, regime='combined', task=None, holdout=MECHANISM_HOLDOUT, seed=seed)
        for seed in SPREAD_SEEDS
        for level in levels
    ]


def plan(
    tiers: Sequence[Tier] = TIERS,
    *,
    levels: Sequence[Level] = LEVELS,
    regimes: Sequence[Regime] = REGIMES,
) -> list[PlannedRun]:
    """Returns the campaign as an ordered list of runs, earlier tiers first.

    The order is the contract: a campaign interrupted anywhere has finished the earlier tiers outright, and
    within a tier the alignment level varies fastest, so what has landed is always a matched comparison.

    Args:
        tiers (Sequence[Tier], optional): Tiers to plan. Defaults to all three, in campaign order.
        levels (Sequence[Level], optional): Alignment levels to plan. Defaults to all three.
        regimes (Sequence[Regime], optional): Regimes to plan. Defaults to both.

    Returns:
        list[PlannedRun]: Every planned run, ordered by tier and then by the tier's own sweep.
    """
    wanted_tiers = [tier for tier in TIERS if tier in set(tiers)]
    wanted_levels = [level for level in LEVELS if level in set(levels)]
    wanted_regimes = [regime for regime in REGIMES if regime in set(regimes)]

    runs: list[PlannedRun] = []
    for tier in wanted_tiers:
        match tier:
            case 'mechanism':
                runs += _mechanism(wanted_levels, wanted_regimes)
            case 'power':
                runs += _power(wanted_levels, wanted_regimes)
            case 'spread':
                runs += _spread(wanted_levels, wanted_regimes)

    return runs


def hours(runs: Sequence[PlannedRun]) -> float:
    """Returns the wall-clock hours a set of planned runs costs.

    Args:
        runs (Sequence[PlannedRun]): The runs to cost.

    Returns:
        float: Their hours, charging a run directory two tiers share only once.
    """
    charged = {run.run_name: run.hours for run in runs}

    return round(sum(charged.values()), 2)


def _find_metrics(run_name: str, roots: Sequence[Path]) -> Path | None:
    """The first root holding this run's evaluated metrics, in the caller's search order."""
    for root in roots:
        if (candidate := Path(root) / run_name / METRICS_ARTIFACT).is_file():
            return candidate

    return None


def status(runs: Sequence[PlannedRun], roots: Sequence[Path]) -> list[RunStatus]:
    """Reports which planned runs already have evaluated metrics, and where.

    Args:
        runs (Sequence[PlannedRun]): The planned runs to look for.
        roots (Sequence[Path]): Run roots in search order -- Drive sessions first, then the local disk.

    Returns:
        list[RunStatus]: One status per planned run, in the plan's order.
    """
    # Tiers share run directories, so each distinct name is resolved once instead of re-stat'd for every row.
    found: dict[str, Path | None] = {}
    for run in runs:
        if run.run_name not in found:
            found[run.run_name] = _find_metrics(run.run_name, roots)

    return [RunStatus(run=run, metrics=found[run.run_name]) for run in runs]


def next_run(statuses: Sequence[RunStatus]) -> PlannedRun | None:
    """Returns the first planned run with no metrics yet.

    Args:
        statuses (Sequence[RunStatus]): Statuses in the plan's order.

    Returns:
        PlannedRun | None: The run to train next, or None once every planned run has landed.
    """
    for state in statuses:
        if not state.done:
            return state.run

    return None


def progress(statuses: Sequence[RunStatus]) -> Progress:
    """Summarises how far the campaign has got, and how many hours it still owes.

    Args:
        statuses (Sequence[RunStatus]): Statuses in the plan's order.

    Returns:
        Progress: done/total per tier, with the remaining hours as the ETA.
    """
    done_rows: Counter[Tier] = Counter()
    total_rows: Counter[Tier] = Counter()
    remaining: dict[Tier, float] = {}
    costed: dict[Tier, float] = {}

    counted: set[str] = set()
    for state in statuses:
        tier = state.run.tier
        total_rows[tier] += 1
        done_rows[tier] += int(state.done)

        # Tier 1's ZAB fold is tier 0's combined arm, so a shared run directory is charged to whichever tier
        # plans it first and the ETA stays the hours the campaign actually burns.
        if state.run.run_name in counted:
            continue

        counted.add(state.run.run_name)
        costed[tier] = costed.get(tier, 0.0) + state.run.hours
        if not state.done:
            remaining[tier] = remaining.get(tier, 0.0) + state.run.hours

    tiers = tuple(
        TierProgress(
            tier=tier,
            done=done_rows[tier],
            total=total_rows[tier],
            hours_remaining=round(remaining.get(tier, 0.0), 2),
            hours_total=round(costed.get(tier, 0.0), 2),
        )
        for tier in TIERS
        if total_rows[tier]
    )

    return Progress(
        tiers=tiers,
        done=sum(done_rows.values()),
        total=sum(total_rows.values()),
        hours_remaining=round(sum(remaining.values()), 2),
        hours_total=round(sum(costed.values()), 2),
    )
