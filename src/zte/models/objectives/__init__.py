"""Self-supervised training objectives for ZTE -- the EEG analogues of word2vec.

`build_objective` selects the objective named by `config.name`. Each objective is its own module: `skipgram`, `cbow`, `masked`,
`cpc`, `clip`; the shared base and loss terms live in `base` and `losses`.
"""

from __future__ import annotations

from torch import nn

from zte.config import ObjectiveConfig
from zte.models.embedding import ZTEModel
from zte.models.objectives.cbow import CBOWObjective
from zte.models.objectives.clip import SentenceClipObjective, _clip_direction
from zte.models.objectives.cpc import CPCObjective
from zte.models.objectives.losses import alignment_penalty, debiased_infonce, vicreg_terms
from zte.models.objectives.masked import MaskedObjective
from zte.models.objectives.skipgram import SkipGramObjective

__all__ = [
    'CBOWObjective',
    'CPCObjective',
    'MaskedObjective',
    'SentenceClipObjective',
    'SkipGramObjective',
    '_clip_direction',
    'alignment_penalty',
    'build_objective',
    'debiased_infonce',
    'vicreg_terms',
]


def build_objective(
    config: ObjectiveConfig, model: ZTEModel, feature_dim: int | None = None
) -> nn.Module:
    """Constructs the objective module selected by `config.name`.

    Args:
        config (ObjectiveConfig): Objective configuration.
        model (ZTEModel): The ZTE encoder the objective wraps.
        feature_dim (int | None): Band-power feature dimension (used by masked reconstruction).

    Returns:
        An objective `nn.Module` exposing `compute(model, batch)` and the `needs_teacher` flag (and `post_step` when applicable).

    Raises:
        ValueError: If `config.name` is unknown.

    """
    if config.name == 'skipgram':
        return SkipGramObjective(config, model)
    if config.name == 'cbow':
        return CBOWObjective(config, model)
    if config.name == 'masked':
        return MaskedObjective(config, model, feature_dim)
    if config.name == 'cpc':
        return CPCObjective(config, model)
    if config.name == 'clip':
        return SentenceClipObjective(config, model)
    raise ValueError(f'Unknown objective: {config.name!r}')
