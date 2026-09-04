"""Publication schematics of the ZTE method: architecture diagrams, scalp maps and the figures a paper sets from.

- `zte.evaluation.schematics._style` -- the palette by role, journal widths and the shared drawing helpers.
- `zte.evaluation.schematics.encoder` -- the pipeline, the stack, one transformer block, the conformer frontend.
- `zte.evaluation.schematics.objective` -- two towers, the multi-positive square, the three levels, LOSO, the gallery.
- `zte.evaluation.schematics.geometry` -- the real montage, the harmonic basis and kernel, transport, the adapter.
- `zte.evaluation.schematics.decoder` -- the frozen-LM prefix decoder.
- `zte.evaluation.schematics.data` -- the presence mask and one word window.
- `zte.evaluation.schematics.artifacts` -- the attention scalp map and curve, and the transfer heatmap.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import numpy as np

from zte.evaluation.schematics import data, decoder, encoder, geometry, objective
from zte.evaluation.schematics._style import (
    DOUBLE_COLUMN_IN,
    RC,
    SINGLE_COLUMN_IN,
    Rendered,
    save_figure,
)
from zte.evaluation.schematics.artifacts import (
    attention_temporal_figure,
    attention_topomap_figure,
    transfer_heatmap_figure,
)
from zte.logging_utils import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_LOG = get_logger('evaluation.schematics')

type Builder = Callable[[], 'Figure']
"""A data-free figure builder."""

__all__ = [
    'DOUBLE_COLUMN_IN',
    'SCHEMATICS',
    'SINGLE_COLUMN_IN',
    'Rendered',
    'attention_temporal_figure',
    'attention_topomap_figure',
    'build_all',
    'contact_sheet',
    'save_figure',
    'transfer_heatmap_figure',
]

SCHEMATICS: Final[dict[str, Builder]] = {
    **encoder.SCHEMATICS,
    **objective.SCHEMATICS,
    **geometry.SCHEMATICS,
    **decoder.SCHEMATICS,
    **data.SCHEMATICS,
}
"""Every data-free schematic, by name."""


def build_all(
    out_dir: Path, names: Sequence[str] | None = None, formats: Sequence[str] = ('png', 'svg')
) -> list[Rendered]:
    """Renders the named data-free schematics (all of them by default) into `out_dir`.

    Args:
        out_dir (Path): Destination directory.
        names (Sequence[str] | None, optional): Schematic names from `SCHEMATICS`; `None` renders every one.
        formats (Sequence[str], optional): Extensions to write. Defaults to PNG and SVG.

    Returns:
        list[Rendered]: What was written, in order.

    Raises:
        KeyError: If a name is not a known schematic.
    """
    chosen = list(names) if names is not None else list(SCHEMATICS)
    unknown = [name for name in chosen if name not in SCHEMATICS]
    if unknown:
        raise KeyError(f'unknown schematic(s) {unknown}; known: {sorted(SCHEMATICS)}')

    rendered: list[Rendered] = []
    with plt.rc_context(RC):
        for name in chosen:
            rendered.append(save_figure(SCHEMATICS[name](), out_dir, name, formats))
            _LOG.info('Wrote %s (%s).', name, ', '.join(formats))

    return rendered


def contact_sheet(rendered: Sequence[Rendered], out_dir: Path, columns: int = 3) -> Path:
    """Tiles every rendered PNG onto one page, named, so the variations can be compared at a glance.

    Args:
        rendered (Sequence[Rendered]): What `build_all` and the artifact-driven builders wrote.
        out_dir (Path): Where `contact_sheet.png` goes.
        columns (int, optional): Thumbnails per row. Defaults to 3.

    Returns:
        Path: The written sheet.
    """
    from matplotlib import image as mimage

    pngs: list[tuple[str, Path]] = []
    for item in rendered:
        png = next((p for p in item.paths if p.suffix == '.png'), None)
        if png is not None:
            pngs.append((item.name, png))
    rows = max(1, math.ceil(len(pngs) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 2.6 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis('off')
    for ax, (name, path) in zip(np.atleast_1d(axes).ravel(), pngs, strict=False):
        ax.imshow(mimage.imread(path))
        ax.set_title(name, fontsize=8)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = out_dir / 'contact_sheet.png'
    fig.savefig(sheet, dpi=110, bbox_inches='tight')
    plt.close(fig)

    return sheet
