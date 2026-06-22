"""Data layer: ZuCo parsing, synthetic generation, processing and torch bridges.

Heavy exports are exposed lazily via module `__getattr__` so that importing the lightweight `zte.data.schema`
(which `zte.config` depends on) does not eagerly pull in `zte.data.dataset` and create an import cycle.
"""

# pylint: disable=undefined-all-variable, import-outside-toplevel
from __future__ import annotations

__all__ = ['ZuCoDataset', 'generate_synthetic_zuco']  # type: ignore[reportUndefinedVariable]


def __getattr__(name: str) -> object:
    """Lazily resolves the package's public data classes/functions.

    Args:
        name (str): Attribute being accessed on the package.

    Returns:
        object: The requested object.

    Raises:
        AttributeError: If `name` is not a known lazy export.

    """
    if name == 'ZuCoDataset':
        from zte.data.dataset import ZuCoDataset

        return ZuCoDataset
    if name == 'generate_synthetic_zuco':
        from zte.data.synthetic import generate_synthetic_zuco

        return generate_synthetic_zuco
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
