"""Keeps path-like config fields stored as `str`, whatever the caller assigns."""

from __future__ import annotations

import os
from typing import Any, ClassVar


class PathFields:
    """Coerces declared path-like fields to `str` on every assignment.

    Callers naturally hand these a `Path` -- argparse `type=Path`, `synthetic_root`, `resolve_data_root`
    -- and a dataclass `__post_init__` only fires at construction, so a later assignment slips a `Path`
    into a config that is serialised to YAML (the run config), to JSON (the dataset bundle meta and the
    confound audit) and into every checkpoint payload. None of those can represent a `Path`, and the
    resulting error surfaces far from the assignment that caused it.

    Attributes:
        _PATH_FIELDS (tuple[str, ...]): Field names to coerce; subclasses override.
    """

    _PATH_FIELDS: ClassVar[tuple[str, ...]] = ()

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets an attribute, converting path objects to `str` for the declared path fields."""
        if name in self._PATH_FIELDS and isinstance(value, os.PathLike):
            value = os.fspath(value)
        super().__setattr__(name, value)
