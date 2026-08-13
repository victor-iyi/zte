"""Builders for the self-contained HTML explorers, each pairing with a template under `web/`.

- `zte.evaluation.interactive.classic` - the classic word-embedding scatter.
- `zte.evaluation.interactive.explorer` - the Thought-Space Explorer.
- `zte.evaluation.interactive.atlas` - the Neuron Atlas.
- `zte.evaluation.interactive.scoreboard` - the held-out scoreboard dashboard.
- `zte.evaluation.interactive.compare` - the cross-run comparison dashboard.
- `zte.evaluation.interactive.generation` - the reference/hypothesis/controls side-by-side.
"""

from __future__ import annotations

from zte.evaluation.interactive.atlas import neuron_atlas_html
from zte.evaluation.interactive.classic import embedding_explorer_html
from zte.evaluation.interactive.compare import build_comparison, combined_dashboard_html
from zte.evaluation.interactive.explorer import thought_space_explorer_html
from zte.evaluation.interactive.generation import generation_html
from zte.evaluation.interactive.scoreboard import scoreboard_html

__all__ = [
    'build_comparison',
    'combined_dashboard_html',
    'embedding_explorer_html',
    'generation_html',
    'neuron_atlas_html',
    'scoreboard_html',
    'thought_space_explorer_html',
]
