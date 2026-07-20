"""Typed, serialisable configuration objects for the whole ZTE pipeline.

Every tunable knob lives in a `dataclasses` object so configs are explicit.  The top-level `ZTEConfig` aggregates the dataset,
model, objective and training sub-configs and is what the CLIs read and write.  Each sub-config is its own module
(`dataset`, `model`, `objective`, `train`); the Literal type aliases live in `types`.
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
