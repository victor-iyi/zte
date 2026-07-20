"""Dict<->dataclass coercion used by `ZTEConfig.from_dict` (YAML round-trips tuples as lists)."""

from __future__ import annotations

import dataclasses
from typing import Any, get_args, get_type_hints


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Reconstructs a (possibly nested) dataclass, coercing lists back to tuples.

    Args:
        cls (type): The dataclass type to instantiate.
        data (dict[str, Any]): Field values, typically parsed from YAML where tuples became lists.

    Returns:
        Any: An instance of `cls` with type-appropriate field values.
    """
    if not dataclasses.is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints.get(f.name)
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
