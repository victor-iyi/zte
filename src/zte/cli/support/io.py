from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Reads and parses a UTF-8 JSON file.

    Args:
        path (str | Path): The JSON file to read.

    Returns:
        Any: The parsed JSON payload.
    """
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(
    path: str | Path,
    obj: Any,
    *,
    indent: int = 2,
    default: Any = None,
) -> Path:
    """Serialises `obj` to a UTF-8 JSON file, creating parent directories.

    Args:
        path (str | Path): Destination file.
        obj (Any): The JSON-serialisable object to write.
        indent (int): Indentation passed to `json.dumps`.
        default (Any): Fallback serialiser passed to `json.dumps` (e.g. `str`).

    Returns:
        Path: The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=indent, default=default), encoding='utf-8')
    return out
