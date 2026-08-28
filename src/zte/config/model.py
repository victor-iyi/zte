from __future__ import annotations

from dataclasses import dataclass

from zte.config.types import FrontendName, PoolName, PosEncoding, SpatialEncoding, TemporalPool


@dataclass
class ModelConfig:
    """ZTE encoder architecture."""

    frontend: FrontendName = 'band_power_mlp'
    """`'band_power_mlp'` for band-power vectors, or `'raw_conformer'` / `'eegnet'` / `'deep_conv_net'` for raw
    time-series windows. The last two are the standard convolutional EEG baselines, run through the identical
    pipeline so their numbers are comparable with the conformer's."""

    embed_dim: int = 768
    """Output embedding dimensionality; 768 keeps it plug-compatible with the frozen LLM space used downstream."""

    hidden_dim: int = 256
    """Width of encoder hidden layers / transformer model dim."""

    n_layers: int = 4
    """Number of transformer / MLP blocks."""

    n_heads: int = 8
    """Attention heads (raw_conformer only)."""

    dropout: float = 0.1
    """Dropout probability throughout the encoder."""

    conformer_filters: int = 40
    """Channel count after the temporal convolution."""

    conformer_temporal_kernel: int = 25
    """Temporal conv kernel size (acts as a learnable band-pass filter)."""

    conformer_multiscale_kernels: tuple[int, ...] = ()
    """Optional parallel temporal-conv kernel widths for the raw conformer. Empty keeps the single
    `conformer_temporal_kernel` filter. When set, a bank of band-passes at several widths runs in parallel and a
    pointwise conv fuses them, so one wide token sees fast (gamma) through slow (theta) rhythms at once. At the 500 Hz
    ZuCo rate `(15, 31, 63, 125)` spans roughly 33 Hz down to 4 Hz."""

    conformer_temporal_pool: TemporalPool = 'mean'
    """How the raw conformer collapses the within-word time axis after self-attention. `mean` is the flat average;
    `attention` learns a per-time-step weighting so informative moments (e.g. the N400 window) dominate the token."""

    pos_encoding: PosEncoding = 'rope'
    """Positional encoding over the word axis of the context transformer. `rope` injects relative position inside
    attention and generalises to any sentence length; `none` is the ablation. Recorded in the checkpoint config so
    inference rebuilds the matching encoder."""

    max_positions: int = 512
    """Position table size for `learned`/`sinusoidal`."""

    spatial_encoding: SpatialEncoding = 'none'
    """Electrode positional encoding over the EEG channel axis, orthogonal to `pos_encoding`. `spherical_harmonics`
    injects scalp position via the real spherical-harmonic basis, the generalisation of sinusoidal encoding from a line
    to a sphere. Needs coordinates: exact from `dataset.montage_csv`, else an approximate fallback that is flagged."""

    spatial_harmonic_degree: int = 6
    """Maximum harmonic degree `l_max` for `spherical_harmonics`; `(l_max + 1) ** 2` harmonics resolve finer detail."""

    spatial_mix: bool = True
    """Follow the additive electrode encoding with one self-attention layer over the channel axis (electrodes as
    tokens), so each electrode is contextualised by geometrically-related ones."""

    spatial_encoding_learnable: bool = True
    """Whether the per-degree harmonic gains and projection are trainable (`True`) or the harmonic gains are frozen."""

    spatial_attn_freqs: int = 8
    """Fourier frequencies per axis for `spatial_encoding='spatial_attention'` (learned attention over 2-D electrode
    coordinates); `2 * spatial_attn_freqs ** 2` positional features. Ignored for `spherical_harmonics`/`none`."""

    pool: PoolName = 'attention'
    """How per-word tokens are pooled into a sentence embedding."""

    subject_conditioning: bool = False
    """Add a learned subject embedding (ZTE v1 is not yet subject-invariant; this exposes the knob for ablations)."""

    subject_film: bool = False
    """Condition token hiddens with a per-subject FiLM affine (`(1 + gamma) * h + beta`). The table is zero-initialised,
    so for any subject id unseen in training (the held-out LOSO subject) the transform is the identity rather than an
    untrained random vector."""

    subject_adapter: bool = False
    """Condition the encoder on a subject signature emitted by a hypernetwork instead of an id lookup, so a
    held-out subject is interpolated rather than left unadapted. Requires `dataset.subject_signature`."""

    subject_adapter_width: int = 128
    """Hidden width of the subject hypernetwork's trunk."""

    subject_adapter_spatial: bool = True
    """Also emit a per-electrode gain applied before the frontend, correcting individual cap placement. Raw
    frontend only; ignored for band power."""

    n_subjects: int = 12
    """Vocabulary size for subject conditioning."""

    n_tasks: int = 3
    """Vocabulary size for task conditioning."""

    grad_checkpoint: bool = False
    """Recompute the spatial and temporal attention in the backward pass instead of storing their
    activations. Numerically identical -- same gradients, ~30% slower -- so it never changes a result,
    only whether it fits. A raw batch turns every (sentence, word) pair into its own 105x350 attention
    problem, which is tens of GB of activations on a full batch; this is what makes it fit a 16 GB GPU."""

    projection_hidden: int = 512
    """Width of the projection head's hidden layer."""

    # -- Factored / disentangled embedding ----------------------------------- #
    factored: bool = False
    """Split the embedding into `[content | nuisance]` subspaces, routing retrieval/meaning to content and
    identity/behaviour heads to nuisance, so the meaning target cannot be offloaded onto identity."""

    content_dim: int = 384
    """Width of the content subspace when `factored`; the remaining `embed_dim - content_dim` dims are nuisance."""

    # -- Predictive residual coding ------------------------------------------ #
    residual_coding: bool = False
    """Subtract each token's context-predicted expectation before the token is used, keeping only the part the
    preceding words did not already account for. Reading is predictive and the large language-related EEG deflections
    are surprisal responses, so the reader's tonic state and 1/f background -- all predictable from the last few
    seconds -- cancel in the residual while the word-specific response survives. The expectation head trains on its
    own detached regression, so the encoder cannot cut that loss by making itself predictable."""

    residual_layers: int = 1
    """Depth of the causal expectation head. One layer is usually enough: it is predicting a smooth drift, not
    modelling language."""

    residual_gate: float = 1.0
    """Initial value of the learnable scalar that scales the subtraction. `0.0` makes the coder the identity at step 0
    and lets training decide how much context to remove; `1.0` subtracts the whole expectation from the start."""

    residual_predict_weight: float = 1.0
    """Weight of the expectation head's own regression loss. It reaches the head and nothing else, so this trades how
    fast the de-trender converges against how much of the step budget it takes -- it never competes with the
    objective for the encoder's parameters."""

    # -- Band-family routing ------------------------------------------------- #
    band_routing: bool = False
    """Encode theta/gamma (lexical-semantic) and alpha/beta (attention/state) through separate pathways, so invariance
    pressure can be applied asymmetrically instead of over one flat vector. Band-power frontend only."""

    # -- Convolutional baselines --------------------------------------------- #
    eegnet_f1: int = 8
    """Temporal filters in EEGNet's first block -- the `F1` of the published EEGNet-8,2. Read only by
    `frontend='eegnet'`, so it changes nothing for any other frontend."""

    eegnet_depth: int = 2
    """Spatial projections EEGNet learns per temporal filter -- the depth multiplier `D`, giving `F1 * D` feature
    maps after the depthwise electrode convolution. Read only by `frontend='eegnet'`."""

    eegnet_kernel: int = 64
    """Length of EEGNet's first temporal convolution. 64 is the published value, defined at 128 Hz; at ZuCo's 500 Hz
    it spans 128 ms, so raise it towards half the sampling rate to keep the published half-second band-pass."""

    eegnet_dropout: float = 0.25
    """Dropout inside EEGNet's two blocks. Kept separate from `dropout` because the published value (0.25 across
    subjects, 0.5 within) is far above what the rest of this encoder runs at."""

    deepconv_filters: tuple[int, ...] = (25, 50, 100, 200)
    """Filter width of each DeepConvNet block; its length is the block count. The published four blocks are tuned for
    1000+ sample windows, so a short `dataset.raw_window` may need fewer -- construction raises rather than pooling
    the time axis down to nothing. Read only by `frontend='deep_conv_net'`."""

    deepconv_kernel: int = 5
    """Temporal convolution length in every DeepConvNet block. Each block consumes `deepconv_kernel - 1` time steps,
    which is what bounds how many blocks a given raw window can carry."""

    deepconv_pool: int = 3
    """Max-pool factor after every DeepConvNet block, clamped per block to what the surviving time axis can divide."""

    deepconv_dropout: float = 0.5
    """Dropout before each DeepConvNet block after the first, at the published value."""
