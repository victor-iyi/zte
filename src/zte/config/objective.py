from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zte.config.types import ObjectiveName


@dataclass
class ObjectiveConfig:
    """Self-supervised training objective and its hyper-parameters."""

    name: ObjectiveName = 'skipgram'
    """The self-supervised objective: word2vec-style contrastive, data2vec-style masked, CPC, or EEG-text CLIP."""

    temperature: float = 0.07
    """Softmax temperature for contrastive (InfoNCE) losses."""

    context_window: int = 2
    """Number of neighbouring words on each side used as context."""

    mask_ratio: float = 0.5
    """Fraction of tokens masked for the `'masked'` objective."""

    masked_target: Literal['reconstruct', 'latent'] = 'latent'
    """Reconstruct raw features (`reconstruct`) or predict an EMA-teacher latent (`latent`, the data2vec variant)."""

    ema_decay: float = 0.999
    """Starting teacher EMA decay for `masked_target='latent'`."""

    ema_decay_end: float = 0.9999
    """Final teacher EMA decay, ramped linearly from `ema_decay`: fast-moving early, stable late. Equal values = flat."""

    teacher_variance_floor: float = 1e-4
    """Minimum per-dimension std when normalising the data2vec teacher target across tokens. A per-token LayerNorm
    target leaves between-token variance unconstrained and lets teacher and student co-collapse to a constant."""

    cpc_steps: int = 4
    """How many future steps CPC predicts."""

    variance_weight: float = 0.0
    """Weight of the VICReg variance-hinge term (0 disables). Penalises any embedding dimension whose batch std falls
    below `variance_target`, preventing collapse into a handful of the available dimensions."""

    covariance_weight: float = 0.0
    """Weight of the VICReg covariance term (0 disables). Pushes off-diagonal feature covariances toward zero so
    dimensions carry decorrelated information (higher effective rank)."""

    variance_target: float = 1.0
    """Target per-dimension std (`gamma`) for the variance-hinge term."""

    anisotropy_weight: float = 0.0
    """Weight of the anti-cone uniformity penalty (0 disables). Spreads L2-normalised embeddings over the sphere so
    their angular arrangement cannot degenerate; complements `whiten`, which removes the shared-mean cone."""

    whiten: bool = False
    """ZCA-whiten the exported embeddings at evaluation: centring removes the dominant shared direction and whitening
    spreads variance across all dimensions. Label-free, so all downstream metrics are recomputed on the whitened space."""

    cross_subject_positives: bool = False
    """For skip-gram/CBOW, build positives from the same stimulus read by different subjects (via `content_id`) rather
    than same-subject neighbours, making subject identity a nuisance the loss must remove. Needs a stimulus-grouped
    sampler to be effective; falls back to within-sentence neighbours when the batch has no cross-subject positive."""

    meaning_positives: bool = False
    """For skip-gram, also pair the same content word occurring in different sentences, so "same meaning across
    contexts" has room to grow instead of memorising which passage a word came from."""

    subject_adversary_weight: float = 0.0
    """Weight of the gradient-reversal subject-adversary loss (0 disables). A head classifies the subject from the token
    hiddens and the reversed gradient trains the encoder to hide subject identity."""

    identity_orthogonality_weight: float = 0.0
    """Weight of a rank-preserving identity penalty (0 disables): the normalised cross-covariance between content
    and the subject signature. Unlike the adversary above, collapsing does not satisfy it. Requires
    `dataset.subject_signature`."""

    stimulus_adversary_weight: float = 0.0
    """Weight of a second gradient-reversal adversary predicting which stimulus/passage a token came from (0 disables),
    removing the sentence-set shortcut. Requires `content_id` in the batch."""

    # -- Meaning distillation + confound-matched hard negatives -------------- #
    meaning_distill_weight: float = 0.0
    """Weight of a loss pulling each word's embedding toward a frozen language-model vector for the word it read
    (0 disables). Random negatives are separable by identity/task alone, so an explicit meaning target is needed."""

    meaning_source: str | None = None
    """Where the frozen word vectors come from: a `word v1 v2 ...` text file (GloVe/fastText format) or an `.npy`+vocab
    pair, or `None` / `hash` for a deterministic hash embedding (mechanism verification only -- carries no semantics)."""

    meaning_dim: int = 64
    """Meaning-vector dimensionality (must match `meaning_source`; used as-is for `'hash'`)."""

    hard_negatives: bool = False
    """For skip-gram/CBOW, restrict InfoNCE negatives to tokens sharing the confound (same subject and task) so the only
    way to tell anchor from negative is the word itself."""

    hard_negative_keys: tuple[str, ...] = ('subject', 'task')
    """Batch fields that a negative must match the anchor on when `hard_negatives` is set."""

    # -- Eye-tracking privileged supervision -------------------------------- #
    behaviour_weight: float = 0.0
    """Weight of an auxiliary head predicting per-word reading behaviour from the embedding (0 disables). Behaviour is a
    lexical-difficulty proxy the EEG-only space struggles to find, so predicting it injects a meaning-adjacent gradient."""

    behaviour_targets: tuple[str, ...] = ('TRT', 'regression_time', 'is_omitted')
    """Which per-word behaviour signals the auxiliary head regresses/classifies."""

    # -- Fix the retrieval geometry (anti-hubness / anti-anisotropy) ------------------------ #
    all_but_top: int = 0
    """Remove the top-`all_but_top` principal directions from the exported embeddings at evaluation (0 disables). The
    label-free all-but-the-top post-processing strips the shared frequency/hub axis behind below-chance retrieval.
    Applied after `whiten` in `evaluation/report.py`: whiten equalises variance, ABTT strips residual shared axes."""

    csls_neighbors: int = 0
    """Neighbourhood size `k` for CSLS retrieval correction (0 = plain cosine). Each similarity becomes
    `2 * cos - r_query - r_bank` for `r` the mean cosine to a point's `k` nearest neighbours, penalising hub-dense
    regions. A monotone re-ranking, applied to the retrieval index and its permutation null alike."""

    # -- Ramp the subject adversary from zero (domain-adversarial schedule) ----------------- #
    subject_adversary_warmup_ratio: float = 0.0
    """Fraction of total optimiser steps over which the gradient-reversal strength `lambda_` ramps linearly 0 -> 1
    (0 = full strength from step 0). A cold adversary early lets the encoder learn content before invariance pressure
    erases it. Only the reversal strength ramps; the loss weight stays at `subject_adversary_weight`."""

    # -- The missing alignment half of the align+uniformity pair ---------------------------- #
    alignment_weight: float = 0.0
    """Weight of an alignment penalty `E_{(i,j) in pos} ||center_i - context_j||^2` over contrastive positives
    (0 disables). `anisotropy_weight` supplies the uniformity half; this pulls positives together."""

    # -- Debiased contrastive (stop punishing correct answers) ------------------------------ #
    tau_plus: float = 0.0
    """Class-prior for the debiased contrastive estimator (0 = plain InfoNCE). Another EEG trial of the same word is a
    false negative, so the estimator subtracts that positive mass from the negative log-sum-exp. Small (`~0.05-0.1`)."""

    # -- Collapse-proof regression auxiliary (fills idle nuisance dims) --------------------- #
    data2vec_aux_weight: float = 0.0
    """Weight of a frozen-target regression auxiliary on the nuisance subspace (0 disables). Those dims regress toward a
    fixed random projection of the token's own features, which cannot co-collapse, giving them a job."""

    # -- Per-occurrence contextual meaning target ------------------------------------------- #
    meaning_contextual: str | None = None
    """HuggingFace model id for a per-occurrence contextual meaning target, or `None` for the word-type-keyed
    `meaning_source` file. Each word's target is its contextual last-hidden state from a frozen encoder run on the whole
    sentence (sub-words mean-pooled), disambiguating polysemy. Requires `transformers`, else falls back with a warning."""

    meaning_context_layer: int = -1
    """Which hidden layer of the contextual model to read; a middle layer (~7-9) aligns best with brain activity."""

    # -- Evaluation hardening (opt-in, heavier checks) -------------------------------------- #
    eval_phase_shuffle: bool = False
    """Add a phase-scrambled-input control: the trained encoder run on FFT-phase-randomised EEG (power spectrum kept).
    Informative only for raw frontends, since band power is near-phase-invariant."""

    eval_seen_novel: bool = False
    """Split cross-subject word retrieval into seen vs novel word types, so "zero-shot" means unseen types too."""

    eval_freq_matched: bool = False
    """Restrict each query's distractor bank to its own frequency/length bin, so a hit cannot be a lexical shortcut."""

    eval_generation: bool = False
    """Run the free-running generation eval with its brain-independent controls and permutation null."""

    eval_rescoring: bool = True
    """Score the sentence gallery by decoder sequence likelihood. Reported as retrieval, never as generation."""

    eval_length_stratified: bool = True
    """Also report held-out retrieval inside word-count strata, so a hit cannot be a sentence-length shortcut."""

    # -- CLIP sentence-alignment objective (name='clip') ------------------------------------ #
    text_source: str | None = None
    """Frozen text-encoder model id for the CLIP sentence target, e.g. `'intfloat/e5-base-v2'` (sentence-transformers)
    or `'Qwen/Qwen2.5-0.5B'` (a decoder LLM, mean-pooled). Each unique ZuCo sentence is embedded once and the EEG
    encoder aligns to it. `None` falls back to a deterministic hash target (mechanism only, no semantics)."""

    text_backend: Literal['auto', 'sentence-transformers', 'hf'] = 'auto'
    """How to load `text_source`: `sentence-transformers`, `hf` (raw HuggingFace model, mean-pooled over the attention
    mask, for decoder LLMs), or `auto` (sentence-transformers unless the id looks like a decoder LLM)."""

    text_query_prefix: str = ''
    """Instruction prefix prepended before encoding; retrieval encoders such as E5 expect `'query: '` / `'passage: '`."""

    clip_temperature: float = 0.07
    """Initial CLIP temperature; the log-scale is a learnable, clamped parameter."""

    semantic_hard_negatives: bool = False
    """Co-locate every anchor sentence with semantically-hard negatives -- surface/syntactically similar but
    semantically distinct (high token-overlap, low text-embedding cosine) -- forcing meaning over surface form."""

    hard_negative_pool: int = 8
    """Number of semantic-hard negatives mined per sentence (used only when `semantic_hard_negatives`)."""
