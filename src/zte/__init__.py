"""ZuCo Thought Embedding: self-supervised word-level EEG embeddings aligned to language."""

from __future__ import annotations

import os as _os

# An inherited `module://` backend (Colab) only works inside IPython, so force headless before matplotlib imports.
if _os.environ.get('MPLBACKEND', '').startswith('module://'):
    _os.environ['MPLBACKEND'] = 'Agg'

from zte.config import (
    DatasetConfig,
    MissingConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainConfig,
    ZTEConfig,
)
from zte.data.dataset import ZuCoDataset

__all__ = [
    'DatasetConfig',
    'MissingConfig',
    'ModelConfig',
    'ObjectiveConfig',
    'TrainConfig',
    'ZTEConfig',
    'ZuCoDataset',
    'run_training',  # type: ignore[reportUndefinedVariable]
    'ZTEEmbedder',  # type: ignore[reportUndefinedVariable]
    'generate_synthetic_zuco',  # type: ignore[reportUndefinedVariable]
    '__version__',
]

__version__ = '0.1.0'


def __getattr__(name: str) -> object:
    """Lazily exposes heavier entry points to keep `import zte` light.

    Args:
        name (str): The attribute being accessed.

    Returns:
        object: The requested object.

    Raises:
        AttributeError: If `name` is not a known lazy export.
    """
    if name == 'run_training':
        from zte.training.pipeline import run_training

        return run_training
    if name == 'ZTEEmbedder':
        from zte.inference.embed import ZTEEmbedder

        return ZTEEmbedder
    if name == 'generate_synthetic_zuco':
        from zte.data.synthetic import generate_synthetic_zuco

        return generate_synthetic_zuco
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
