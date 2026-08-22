"""Dict<->dataclass coercion used by `ZTEConfig.from_dict` (YAML round-trips tuples as lists)."""

from __future__ import annotations

import dataclasses
from typing import Any, get_args, get_type_hints

from zte.logging_utils import get_logger

_LOG = get_logger('config.serde')


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Reconstructs a (possibly nested) dataclass, coercing lists back to tuples."""
    if not dataclasses.is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}

    # A misspelled knob is otherwise a run that looks configured and trains with the lever off, with nothing in the
    # log to say so. On a sweep measured in days that is the most expensive kind of silence.
    if unknown := sorted(k for k in data if k not in known):
        _LOG.warning(
            '%s ignores unknown config key(s) %s -- check the spelling against the dataclass.', cls.__name__, unknown
        )
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints.get(f.name)
        if value is None and dataclasses.is_dataclass(_strip_optional(hint)):
            continue  # a YAML section written with no keys parses as None; keep that section's defaults
        if dataclasses.is_dataclass(_strip_optional(hint)) and isinstance(value, dict):
            kwargs[f.name] = _build(_strip_optional(hint), value)
        elif _is_tuple_hint(hint) and isinstance(value, list):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _strip_optional(hint: Any) -> Any:
    """Returns the non-`None` member of an `X | None` hint, else `hint`."""
    args = [a for a in get_args(hint) if a is not type(None)]
    return args[0] if args and len(args) == 1 else hint


def _is_tuple_hint(hint: Any) -> bool:
    """Returns whether a type hint resolves to a `tuple[...]` type."""
    origin = getattr(hint, '__origin__', None)
    if origin is tuple:
        return True
    for arg in get_args(hint):
        if getattr(arg, '__origin__', None) is tuple:
            return True
    return False
