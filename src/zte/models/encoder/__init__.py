"""Encoder-side mechanisms layered on `ZTEModel`, each attacking one measured failure of the v1 encoder.

- `zte.models.encoder.consensus` - cross-reader prototypes, because twelve readings of a sentence beat one.
- `zte.models.encoder.residual` - predictive residual coding, keeping what context did not already explain.
- `zte.models.encoder.gallery` - full-gallery contrastive scoring with length-matched negatives.
- `zte.models.encoder.nuisance` - train-fitted removal of the sentence-length subspace.
"""

from __future__ import annotations

from zte.models.encoder.consensus import ConsensusBank, ConsensusDistiller, build_consensus
from zte.models.encoder.gallery import GalleryContrast, build_gallery_contrast, text_word_counts
from zte.models.encoder.nuisance import LengthProjector, length_leakage
from zte.models.encoder.residual import PredictiveResidual, build_predictive_residual

__all__ = [
    'ConsensusBank',
    'ConsensusDistiller',
    'GalleryContrast',
    'LengthProjector',
    'PredictiveResidual',
    'build_consensus',
    'build_gallery_contrast',
    'build_predictive_residual',
    'length_leakage',
    'text_word_counts',
]
