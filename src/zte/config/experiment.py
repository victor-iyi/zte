from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from zte.config._serde import _build
from zte.config.dataset import DatasetConfig
from zte.config.model import ModelConfig
from zte.config.objective import ObjectiveConfig
from zte.config.train import TrainConfig


@dataclass
class ZTEConfig:
    """Top-level configuration aggregating every sub-config."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """Dataset construction options."""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Encoder architecture."""

    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    """Self-supervised objective."""

    train: TrainConfig = field(default_factory=TrainConfig)
    """Optimisation / logging / checkpointing."""

    run_name: str = 'zte-run'
    """Identifier used in log/checkpoint paths."""

    def to_dict(self) -> dict[str, Any]:
        """Returns a plain (YAML-safe) nested dict of the whole config."""
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> Path:
        """Writes the config to `path` as YAML and returns the path.

        Args:
            path (str | Path): Destination `.yaml` file (parent dirs are created).

        Returns:
            Path: The written path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding='utf-8')
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZTEConfig:
        """Builds a `ZTEConfig` from a nested dict, coercing tuples/types.

        Args:
            data (dict[str, Any]): A nested mapping such as one produced by `to_dict` or parsed from YAML.

        Returns:
            ZTEConfig: A fully constructed config with sub-dataclasses rebuilt.
        """
        return cls(
            dataset=_build(DatasetConfig, data.get('dataset', {})),
            model=_build(ModelConfig, data.get('model', {})),
            objective=_build(ObjectiveConfig, data.get('objective', {})),
            train=_build(TrainConfig, data.get('train', {})),
            run_name=data.get('run_name', 'zte-run'),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ZTEConfig:
        """Loads a `ZTEConfig` from a YAML file.

        Args:
            path (str | Path): Path to a YAML config previously written by `to_yaml`.

        Returns:
            ZTEConfig: The parsed config.
        """
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        return cls.from_dict(data)
