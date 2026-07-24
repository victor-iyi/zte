from __future__ import annotations

from dataclasses import dataclass

from zte.config.types import FrontendName, PoolName, PosEncoding, SpatialEncoding, TemporalPool


@dataclass
class ModelConfig:
    """ZTE encoder architecture."""

    frontend: FrontendName = 'band_power_mlp'
    """`'band_power_mlp'` for band-power vectors or `'raw_conformer'` for raw time-series windows."""

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
    """Maximum harmonic degree `l_max` for `spherical_harmonics`; `(l_max + 1) ** 2` harmonics resolve finer patterns."""

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

    n_subjects: int = 12
    """Vocabulary size for subject conditioning."""

    n_tasks: int = 3
    """Vocabulary size for task conditioning."""

    projection_hidden: int = 512
    """Width of the projection head's hidden layer."""

    # -- Factored / disentangled embedding ----------------------------------- #
    factored: bool = False
    """Split the embedding into `[content | nuisance]` subspaces, routing retrieval/meaning to content and
    identity/behaviour heads to nuisance, so the meaning target cannot be offloaded onto identity."""

    content_dim: int = 384
    """Width of the content subspace when `factored`; the remaining `embed_dim - content_dim` dims are nuisance."""

    # -- Band-family routing ------------------------------------------------- #
    band_routing: bool = False
    """Encode theta/gamma (lexical-semantic) and alpha/beta (attention/state) through separate pathways, so invariance
    pressure can be applied asymmetrically instead of over one flat vector. Band-power frontend only."""
