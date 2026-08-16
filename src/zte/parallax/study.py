"""Parallax study geometry: the task triad and the naming/config contracts every cell follows."""

import dataclasses
import re
from typing import Final

from zte.config import ZTEConfig
from zte.data.schema import Task

# NR first: the largest stimulus set anchors the matrix rows and the chamber's reference view.
PARALLAX_TASKS: Final[tuple[Task, ...]] = ('NR', 'SR', 'TSR')
"""The three vantage points, one independently trained encoder each."""

_CELL_RE: Final[re.Pattern[str]] = re.compile(r'^(?P<train>NR|SR|TSR)_to_(?P<eval>NR|SR|TSR)_s(?P<seed>\d+)$')


def arm_run_name(task: Task) -> str:
    """Returns the `run_name` inside a parallax arm's config YAML.

    Args:
        task (Task): The arm's training task.

    Returns:
        str: `parallax_<task-lowercase>`, e.g. `parallax_nr`.
    """
    return f'parallax_{task.lower()}'


def run_dir_name(task: Task, holdout: str, seed: int) -> str:
    """Returns an arm's on-disk run directory name, with the runner's holdout/seed suffix.

    Args:
        task (Task): The arm's training task.
        holdout (str): The LOSO holdout subject code.
        seed (int): The training seed.

    Returns:
        str: `parallax_<task>_lo<holdout>_s<seed>`.
    """
    return f'{arm_run_name(task)}_lo{holdout}_s{seed}'


def cell_name(train_task: str, eval_task: str, seed: int) -> str:
    """Returns a transfer cell's directory name.

    Args:
        train_task (str): The task the scored model was trained on.
        eval_task (str): The task whose readings it was scored on.
        seed (int): The cell's seed.

    Returns:
        str: `<train_task>_to_<eval_task>_s<seed>`.
    """
    return f'{train_task}_to_{eval_task}_s{seed}'


def parse_cell_name(name: str) -> tuple[str, str, int] | None:
    """Parses a transfer cell directory name back into its parts.

    Args:
        name (str): A directory name such as `NR_to_SR_s0`.

    Returns:
        tuple[str, str, int] | None: `(train_task, eval_task, seed)`, or `None` when the name is not a cell.
    """
    match = _CELL_RE.match(name)
    if match is None:
        return None

    return match['train'], match['eval'], int(match['seed'])


def derive_eval_config(config: ZTEConfig, eval_task: Task) -> ZTEConfig:
    """Clones a checkpoint's config with `dataset.tasks` swapped to the single eval task.

    Every other field is kept byte-identical, so the clone's dataset cache key differs from the training
    config's only through `tasks` and an eval-side build reuses every unrelated cached artifact.

    Args:
        config (ZTEConfig): The checkpoint's configuration.
        eval_task (Task): The task whose readings the model is evaluated on.

    Returns:
        ZTEConfig: A deep clone with `dataset.tasks == (eval_task,)`.

    Raises:
        ValueError: If `eval_task` is not one of the parallax tasks.
    """
    if eval_task not in PARALLAX_TASKS:
        raise ValueError(f'eval_task must be one of {PARALLAX_TASKS}, got {eval_task!r}.')

    clone = ZTEConfig.from_dict(config.to_dict())
    clone.dataset = dataclasses.replace(clone.dataset, tasks=(eval_task,))
    return clone


def resolve_transfer_holdout(config: ZTEConfig, requested: str | None) -> str:
    """The only holdout a transfer cell may query: the subject the checkpoint actually held out.

    Any other subject is a brain the model trained on, and a cell scored on it would carry the
    held-out label with nothing downstream able to tell the difference.

    Args:
        config (ZTEConfig): The checkpoint's configuration.
        requested (str | None): The `--holdout` value, or None to accept the training holdout.

    Returns:
        str: The held-out subject code.

    Raises:
        ValueError: If the checkpoint names no LOSO holdout, or `requested` differs from it.
    """
    train_holdout = config.train.loso_holdout_subject
    if config.train.split != 'by_subject_loso' or not train_holdout:
        raise ValueError(
            f'The checkpoint was trained with split {config.train.split!r} and names no LOSO holdout; '
            'a transfer cell has no honest held-out subject to query. Train the arm with by_subject_loso.'
        )
    if requested and str(requested) != str(train_holdout):
        raise ValueError(
            f'--holdout {requested} is not the subject this checkpoint held out of training ({train_holdout}); '
            'scoring a training subject as held-out is the failure mode this guard exists to stop.'
        )

    return str(train_holdout)
