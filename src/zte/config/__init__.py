"""Typed, serialisable configuration dataclasses for the whole pipeline.

- `zte.config.dataset` -- `DatasetConfig` and `MissingConfig`.
- `zte.config.model` -- `ModelConfig` (encoder architecture).
- `zte.config.objective` -- `ObjectiveConfig` (losses and regularisers).
- `zte.config.train` -- `TrainConfig` (optimiser and schedule).
- `zte.config.experiment` -- `ZTEConfig`, the aggregate the CLIs read and write.
- `zte.config.types` -- the Literal type aliases.
"""

from __future__ import annotations

from zte.config.dataset import DatasetConfig, MissingConfig
from zte.config.experiment import ZTEConfig
from zte.config.model import ModelConfig
from zte.config.objective import ObjectiveConfig
from zte.config.train import TrainConfig
from zte.config.types import (
    FrontendName,
    Granularity,
    MissingMethod,
    Normalization,
    ObjectiveName,
    PoolName,
    PosEncoding,
    Representation,
    SchedulerName,
    SpatialEncoding,
    SplitStrategy,
)

__all__: list[str] = [
    'DatasetConfig',
    'FrontendName',
    'Granularity',
    'MissingConfig',
    'MissingMethod',
    'ModelConfig',
    'Normalization',
    'ObjectiveConfig',
    'ObjectiveName',
    'PoolName',
    'PosEncoding',
    'Representation',
    'SchedulerName',
    'SpatialEncoding',
    'SplitStrategy',
    'TrainConfig',
    'ZTEConfig',
]
