"""ZuCo Thought Embedding (ZTE).

A tunable ZuCo dataset toolkit and a state-of-the-art self-supervised EEG embedding pipeline, built for the
*Cross-Modal Transfer Learning: Aligning EEG Signals to Language* project.

ZTE pretrains word-level "thought embeddings" from ZuCo EEG the way word embeddings are pretrained
from text -- via skip-gram/CBOW contrastive, masked (data2vec/MAEEG) and CPC (wav2vec/BENDR) objectives -- and is
the first step toward a device-, subject- and task-agnostic brain representation.

Typical usage::

    from zte import ZuCoDataset, DatasetConfig, ZTEConfig, run_training

    ds = ZuCoDataset(DatasetConfig(root='res/data/zuco_extracted')).build()
    artifacts = run_training(ZTEConfig(), ds)
"""

# pylint: disable=undefined-all-variable,import-outside-toplevel
from __future__ import annotations

import os as _os

# Colab / IPython export MPLBACKEND=module://matplotlib_inline.backend_inline for inline plotting;
# that backend only works *inside* IPython, so `import matplotlib` in a plain subprocess (e.g. a
# `uv run zte-*` command launched from a Colab cell, which inherits the notebook's environment)
# raises a ValueError. Force a headless backend whenever we inherit an interactive/module one, so
# every ZTE process plots safely regardless of who launched it. Must run before matplotlib imports.
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
        name: The attribute being accessed.

    Returns:
        The requested object.

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
