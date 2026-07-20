from __future__ import annotations

from typing import Literal

type Granularity = Literal['word', 'sentence']
"""Token granularity. Only `'word'` is implemented; `'sentence'` is reserved (use `ZTEEmbedder(level='sentence')`
for pooled sentence embeddings at inference time)."""

type Representation = Literal['band_power', 'raw', 'both']
"""Use compact band-power vectors, raw time-series windows, or both."""

type Normalization = Literal[
    'zscore_channel', 'zscore_global', 'zscore_subject', 'riemannian', 'minmax', 'none'
]
"""Feature normalisation. `zscore_channel`/`zscore_global` fit one mean/std across the whole cohort; `zscore_subject`
fits and applies a **per-subject** mean/std, which removes the constant per-subject offset that otherwise makes subject
identity the cheapest thing to encode (a direct attack on the "learns who, not what" failure mode). `minmax`/`none` as named."""

type MissingMethod = Literal[
    'zero',
    'row_mean',
    'col_mean',
    'global_mean',
    'median',
    'knn',
    'iterative',
    'ffill',
    'interpolate',
    'drop',
    'mask_only',
]
"""How the dataset is split into training/validation/test sets. `random` splits randomly; `by_sentence` splits by sentence; `by_stimulus`
splits by stimulus; `by_subject_loso` splits by subject (leaving out the held-out subject); `by_task` splits by task."""

type SplitStrategy = Literal['random', 'by_sentence', 'by_stimulus', 'by_subject_loso', 'by_task']
"""How the dataset is split into training/validation/test sets. `random` splits randomly; `by_sentence` splits by sentence; `by_stimulus`
splits by stimulus; `by_subject_loso` splits by subject (leaving out the held-out subject); `by_task` splits by task."""

type ObjectiveName = Literal['skipgram', 'cbow', 'masked', 'cpc', 'clip']
"""The objective function to train the model on. `skipgram` trains a skip-gram model; `cbow` trains a continuous bag of words model;
`masked` trains a masked language model; `cpc` trains a context prediction model; `clip` trains a contrastive language-image pre-training model."""

type FrontendName = Literal['band_power_mlp', 'raw_conformer']
"""The frontend to use for the model. `band_power_mlp` uses a multi-layer perceptron on band-power vectors;
`raw_conformer` uses a transformer on raw time-series windows."""

type PoolName = Literal['mean', 'attention', 'cls']
"""The pooling operation to use for the model. `mean` uses a mean pooling operation; `attention` uses a self-attention
pooling operation; `cls` uses a classification pooling operation."""

type SchedulerName = Literal['cosine', 'linear', 'constant']
"""The scheduler to use for the model. `cosine` uses a cosine scheduler; `linear` uses a linear scheduler; `constant` uses a constant scheduler."""

type PosEncoding = Literal['rope', 'sinusoidal', 'learned', 'alibi', 'none']
"""The positional encoding to use for the model. `rope` uses a rotary positional encoding; `sinusoidal` uses a sinusoidal positional encoding;
`learned` uses a learned positional encoding; `alibi` uses an alibi positional encoding; `none` uses no positional encoding."""

type SpatialEncoding = Literal['none', 'spherical_harmonics', 'spatial_attention']
"""The spatial encoding to use for the model. `none` uses no spatial encoding; `spherical_harmonics` uses spherical harmonics;
`spatial_attention` uses a spatial attention operation."""
