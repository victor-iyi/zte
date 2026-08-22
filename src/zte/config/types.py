from __future__ import annotations

from typing import Literal

type Granularity = Literal['word', 'sentence']
"""Token granularity. Only `'word'` is implemented; use `ZTEEmbedder(level='sentence')` for pooled sentences."""

type Representation = Literal['band_power', 'raw', 'both']
"""Use compact band-power vectors, raw time-series windows, or both."""

type Normalization = Literal['zscore_channel', 'zscore_global', 'zscore_subject', 'riemannian', 'minmax', 'none']
"""Feature normalisation. `zscore_channel`/`zscore_global` fit one mean/std across the cohort; `zscore_subject` fits
per subject, removing the constant offset that makes subject identity the cheapest thing to encode."""

type RawAlign = Literal['none', 'euclidean']
"""Per-subject alignment of raw EEG windows. `euclidean` whitens each subject by their own mean channel covariance,
cancelling the linear part of the skull/cap/impedance difference without reading a single label."""

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
"""How missing word-level values are filled; `drop` removes the row and `mask_only` leaves the NaN in place."""

type SplitStrategy = Literal[
    'random',
    'by_sentence',
    'by_stimulus',
    'by_subject_loso',
    'by_task',
    'by_subject_and_stimulus',
]
"""What the train/val/test split groups on; `by_subject_loso` holds out whole subjects and
`by_subject_and_stimulus` crosses that with a stimulus partition, so the test cell is unseen subject x unseen text."""

type ObjectiveName = Literal['skipgram', 'cbow', 'masked', 'cpc', 'clip', 'decode']
"""The self-supervised objective: word2vec-style contrastive, data2vec-style masked, CPC, EEG-text CLIP, or the
frozen-LM prefix decoder."""

type TrainMode = Literal['encoder', 'decoder', 'joint']
"""Which stage the run trains: the encoder alone, a decoder over a frozen encoder, or both jointly."""

type Conditioning = Literal['pooled', 'pooled_plus_words']
"""What the prefix bridge reads: the pooled text-aligned sentence vector, or that plus resampled word tokens."""

type GapCorrection = Literal['none', 'mean_scale', 'whiten']
"""Train-fitted affine correction of the EEG-to-text modality gap before the bridge."""

type RateLadder = Literal['none', 'rvq']
"""Bottleneck between the conditioning vector and the prefix. `rvq` sends it through a residual vector quantiser
whose `stages * log2(codes)` product is a hard ceiling on how many bits of sentence identity can reach the LM."""

type EvidenceSchedule = Literal['none', 'linear', 'fixation']
"""How the word-synchronous pointer walks the EEG word tokens while decoding. `linear` advances at a constant
tokens-per-word rate; `fixation` weights the walk by each word's read duration. `none` disables the evidence path."""

type LMDtype = Literal['auto', 'float32', 'float16', 'bfloat16']
"""Precision the frozen LM's weights are loaded at. `auto` inherits the encoder's, so the two halves always agree;
naming one of the others pins it, and either way the checkpoint's own stored dtype never decides."""

type FrontendName = Literal['band_power_mlp', 'raw_conformer']
"""Per-token encoder: an MLP over band-power vectors or a conformer over raw time-series windows."""

type PoolName = Literal['mean', 'attention', 'cls']
"""How token embeddings are pooled into a sentence embedding."""

type TemporalPool = Literal['mean', 'attention']
"""How the raw-conformer collapses the within-word time axis: a flat average or a learned attentive pool."""

type SchedulerName = Literal['cosine', 'linear', 'constant']
"""Learning-rate schedule shape after warmup."""

type PosEncoding = Literal['rope', 'sinusoidal', 'learned', 'alibi', 'none']
"""Positional encoding over the word axis."""

type SpatialEncoding = Literal['none', 'spherical_harmonics', 'spatial_attention']
"""Positional encoding over the electrode axis, from the montage geometry."""
