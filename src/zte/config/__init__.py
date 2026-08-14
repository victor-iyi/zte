"""Typed, serialisable configuration dataclasses for the whole pipeline.

- `zte.config.dataset` -- `DatasetConfig` and `MissingConfig`.
- `zte.config.decoder` -- `DecoderConfig` (frozen-LM prefix decoder).
- `zte.config.model` -- `ModelConfig` (encoder architecture).
- `zte.config.objective` -- `ObjectiveConfig` (losses and regularisers).
- `zte.config.train` -- `TrainConfig` (optimiser and schedule).
- `zte.config.experiment` -- `ZTEConfig`, the aggregate the CLIs read and write.
- `zte.config.types` -- the Literal type aliases.
"""

from __future__ import annotations

from zte.config.dataset import DatasetConfig, MissingConfig
from zte.config.decoder import DecoderConfig
from zte.config.experiment import ZTEConfig
from zte.config.model import ModelConfig
from zte.config.objective import ObjectiveConfig
from zte.config.train import TrainConfig
from zte.config.types import (
    Conditioning,
    FrontendName,
    GapCorrection,
    Granularity,
    LMDtype,
    MissingMethod,
    Normalization,
    ObjectiveName,
    PoolName,
    PosEncoding,
    Representation,
    SchedulerName,
    SpatialEncoding,
    SplitStrategy,
    TrainMode,
)

__all__: list[str] = [
    'Conditioning',
    'DatasetConfig',
    'DecoderConfig',
    'FrontendName',
    'GapCorrection',
    'Granularity',
    'LMDtype',
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
    'TrainMode',
    'ZTEConfig',
]
