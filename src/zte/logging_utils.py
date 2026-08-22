"""Logging and progress-bar helpers, routed through `rich`/`tqdm` when installed."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

_STATE: dict[str, bool] = {'configured': False}
_LOGGER_NAME: str = 'zte'


def configure_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    *,
    stderr: bool = False,
) -> None:
    """Configures the root `zte` logger exactly once.

    Args:
        level (int | str): Logging level (e.g. `logging.INFO` or `'DEBUG'`).
        log_file (str | Path | None): Optional path; when given, logs are also appended there with a verbose formatter.
        stderr (bool, optional): Route console logs to stderr, so a command whose stdout is a machine-read payload
            stays parsable at any log level. Defaults to False.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    try:
        from rich.console import Console
        from rich.logging import RichHandler

        # `rich` writes to stdout by default; the plain fallback below is already on stderr.
        console = Console(stderr=True) if stderr else None
        console_handler: logging.Handler = RichHandler(
            console=console, rich_tracebacks=True, show_path=False, markup=True
        )
        console_handler.setFormatter(logging.Formatter('%(message)s', datefmt='[%X]'))
    except ImportError:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'))
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s')
        )
        logger.addHandler(file_handler)

    _STATE['configured'] = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Returns a namespaced child of the `zte` logger.

    Args:
        name (str | None): Optional dotted suffix (e.g. `'data.dataset'`); `None` returns the root `zte` logger.

    Returns:
        logging.Logger: A configured logger.
    """
    if not _STATE['configured']:
        configure_logging()
    if name is None:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f'{_LOGGER_NAME}.{name}')


def progress[_T](
    iterable: Iterable[_T],
    description: str = 'working',
    total: int | None = None,
    disable: bool = False,
    unit: str = 'it',
) -> Iterator[_T]:
    """Wraps an iterable in a progress bar, yielding it unchanged when `tqdm` is unavailable.

    Args:
        iterable (Iterable[_T]): The iterable to consume.
        description (str): Text shown to the left of the bar.
        total (int | None): Optional length hint when `iterable` has no `__len__`.
        disable (bool): If `True`, suppress the bar entirely.
        unit (str): Unit label for the rate display.

    Yields:
        _T: Items from `iterable` unchanged.
    """
    if disable:
        yield from iterable
        return
    try:
        from tqdm.auto import tqdm

        yield from tqdm(iterable, desc=description, total=total, unit=unit, leave=False)
    except ImportError:
        yield from iterable
