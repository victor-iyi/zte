from __future__ import annotations

from pathlib import Path

DEFAULT_SYNTHETIC_ROOT: Path = Path('res/data/synthetic_zuco')


def synthetic_root(
    tasks: tuple[str, ...] | None = None,
    *,
    out: str | Path = DEFAULT_SYNTHETIC_ROOT,
    show_progress: bool = True,
) -> Path:
    """Fabricates a synthetic ZuCo dataset under `out` and returns that path.

    Args:
        tasks (tuple[str, ...] | None): Tasks to generate; `None` uses the generator's default set.
        out (str | Path): Destination directory for the fabricated `.mat` files.
        show_progress (bool): Whether the generator prints a progress bar.

    Returns:
        Path: The `out` path, ready to hand to a `DatasetConfig(root=...)`.
    """
    from zte.data.synthetic import generate_synthetic_zuco

    out = Path(out)
    if tasks is None:
        generate_synthetic_zuco(out, show_progress=show_progress)
    else:
        generate_synthetic_zuco(out, tasks=tasks, show_progress=show_progress)
    return out
