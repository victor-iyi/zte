from __future__ import annotations

from dataclasses import dataclass

from zte.config.types import FrontendName, PoolName, PosEncoding, SpatialEncoding


@dataclass
class ModelConfig:
    """ZTE encoder architecture."""

    frontend: FrontendName = 'band_power_mlp'
    """`'band_power_mlp'` for band-power vectors or `'raw_conformer'` for raw time-series windows."""

    embed_dim: int = 768
    """Output embedding dimensionality (768 keeps it plug-compatible with the frozen LLM space used downstream in EEG-OT-CLIP)."""

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

    pos_encoding: PosEncoding = 'rope'
    """Sequence positional-encoding scheme for the context transformer. `rope` (rotary, the default) injects relative position inside attention and generalises to any sentence length -- the current SOTA choice; `sinusoidal` adds the fixed Transformer encoding; `learned` adds an absolute learned table; `alibi` adds linear distance attention biases; `none` disables positional information (ablation). Each run records its scheme in the checkpoint config, so inference rebuilds the matching encoder."""

    max_positions: int = 512
    """Position table size for `learned`/`sinusoidal`."""

    spatial_encoding: SpatialEncoding = 'none'
    """**Electrode** positional encoding, applied over the EEG channel axis (orthogonal to the word-sequence `pos_encoding` above). `spherical_harmonics` injects each electrode's scalp position via the real spherical-harmonic basis -- the sphere's Laplace-Beltrami eigenfunctions, i.e. the mathematically correct generalisation of sinusoidal position encoding from a line to the scalp surface (see `zte.models.spatial`). `none` (default) leaves the channel axis position-free, as before. Requires electrode coordinates: exact when `dataset.montage_csv` supplies `channel,x,y,z`, otherwise an approximate coordinate-free fallback is used and flagged."""

    spatial_harmonic_degree: int = 6
    """Maximum harmonic degree `l_max` for `spherical_harmonics`. `(l_max + 1) ** 2` harmonics are used; higher degrees resolve finer scalp patterns (degree = angular frequency), analogous to adding higher-frequency sinusoids."""

    spatial_mix: bool = True
    """If `True`, follow the additive electrode encoding with one self-attention layer over the channel axis (electrodes as tokens), so each electrode is contextualised by geometrically-related electrodes. If `False`, only add the encoding."""

    spatial_encoding_learnable: bool = True
    """Whether the per-degree harmonic gains and projection are trainable (`True`) or the harmonic gains are frozen."""

    spatial_attn_freqs: int = 8
    """Number of Fourier frequencies per axis for `spatial_encoding='spatial_attention'` (the Défossez-style learned attention over
    2-D electrode coordinates); `2 * spatial_attn_freqs ** 2` positional features are used. Ignored for `spherical_harmonics`/`none`."""

    pool: PoolName = 'attention'
    """How per-word tokens are pooled into a sentence embedding."""

    subject_conditioning: bool = False
    """Add a learned subject embedding (ZTE v1 is not yet subject-invariant; this exposes the knob for ablations)."""

    subject_film: bool = False
    """If `True`, condition token hiddens with a per-subject **FiLM** affine (`(1 + gamma) * h + beta`) instead of (or alongside) the
    additive `subject_conditioning` embedding. The FiLM table is zero-initialised, so at start -- and, crucially, for any subject id
    never seen in training (the held-out LOSO subject) -- the transform is the identity (`gamma = 0`, `beta = 0`) rather than an
    untrained random vector. This is the Défossez "condition on identity, don't only adversarially remove it" lever, made safe for
    the held-out-subject north-star. Default is `False`."""

    n_subjects: int = 12
    """Vocabulary size for subject conditioning. Default is 12."""

    n_tasks: int = 3
    """Vocabulary size for task conditioning. Default is 3."""

    projection_hidden: int = 512
    """Width of the projection head's hidden layer."""

    # -- Factored / disentangled embedding ----------------------------------- #
    factored: bool = False
    """If `True`, split the embedding into named subspaces `[content | nuisance]` and route only the content subspace to retrieval/meaning,
    while identity/behaviour heads act on the nuisance subspace. Turns 'delete the shortcut' (adversary) into 'give the shortcut its own room'
    (disentanglement), so the meaning target cannot be offloaded onto identity."""

    content_dim: int = 384
    """Width of the content subspace when `factored` (the remaining `embed_dim - content_dim` dims are the nuisance subspace).
    Retrieval and meaning distillation use only these dims."""

    # -- Band-family routing ------------------------------------------------- #
    band_routing: bool = False
    """If `True` (band-power frontend), encode theta/gamma (lexical-semantic) and alpha/beta (attention/state) through separate pathways,
    so invariance pressure can be applied asymmetrically instead of over a flat 840-vector."""
