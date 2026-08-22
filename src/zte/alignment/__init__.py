"""The cross-level view: where a token, a word and a sentence land in one ZTE embedding space.

- `zte.alignment.atlas` -- one jointly fitted projection of all three levels, as 2D and 3D plotly figure JSON,
  plus the chart of what the contrastive term bought.
- `zte.alignment.contrastive` -- alignment, uniformity, effective rank and the positive/negative gap, per level.
- `zte.alignment.compare` -- the cross-level table: hit counts, exact tails, rank percentile and the oracle floor.
"""

from typing import TYPE_CHECKING

# The runtime path stays lazy so importing the package costs neither plotly nor the audit's dependencies; a type
# checker follows these instead, which a `__getattr__` returning `object` would otherwise erase.
if TYPE_CHECKING:
    from zte.alignment.atlas import DISCLAIMER, LEVELS, LevelPoints, build_atlas, contrastive_figure
    from zte.alignment.compare import LevelRetrieval, cross_level_table, render_markdown, token_oracle_floor
    from zte.alignment.contrastive import LevelPairs, contrastive_geometry

__all__ = [
    'DISCLAIMER',
    'LEVELS',
    'LevelPairs',
    'LevelPoints',
    'LevelRetrieval',
    'build_atlas',
    'contrastive_figure',
    'contrastive_geometry',
    'cross_level_table',
    'render_markdown',
    'token_oracle_floor',
]

_ATLAS_EXPORTS = frozenset({'DISCLAIMER', 'LEVELS', 'LevelPoints', 'build_atlas', 'contrastive_figure'})
_CONTRASTIVE_EXPORTS = frozenset({'LevelPairs', 'contrastive_geometry'})
_COMPARE_EXPORTS = frozenset({'LevelRetrieval', 'cross_level_table', 'render_markdown', 'token_oracle_floor'})


def __getattr__(name: str) -> object:
    """Lazily resolves the package API, so importing it costs neither plotly nor the audit's dependencies."""
    if name in _ATLAS_EXPORTS:
        from zte.alignment import atlas

        return getattr(atlas, name)
    if name in _CONTRASTIVE_EXPORTS:
        from zte.alignment import contrastive

        return getattr(contrastive, name)
    if name in _COMPARE_EXPORTS:
        from zte.alignment import compare

        return getattr(compare, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
