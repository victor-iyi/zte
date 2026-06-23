"""Model layer: EEG frontends, the ZTE encoder, and self-supervised objectives."""

from __future__ import annotations

from zte.models.embedding import ZTEModel, build_model
from zte.models.objectives import build_objective

__all__ = ['ZTEModel', 'build_model', 'build_objective']
