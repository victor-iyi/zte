from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zte.config.types import ObjectiveName


@dataclass
class ObjectiveConfig:
    """Self-supervised training objective and its hyper-parameters."""

    name: ObjectiveName = 'skipgram'
    """The objective function to train the model on. `skipgram` trains a skip-gram model; `cbow` trains a continuous bag of words model;
    `masked` trains a masked language model; `cpc` trains a context prediction model; `clip` trains a contrastive language-image pre-training model."""

    temperature: float = 0.07
    """Softmax temperature for contrastive (InfoNCE) losses. Default is 0.07."""

    context_window: int = 2
    """Number of neighbouring words on each side used as context. Default is 2."""

    mask_ratio: float = 0.5
    """Fraction of tokens masked for the `'masked'` objective. Default is 0.5."""

    masked_target: Literal['reconstruct', 'latent'] = 'latent'
    """Reconstruct raw features (`reconstruct`) or predict an EMA-teacher latent (`latent`, the data2vec variant). Default is `'latent'`."""

    ema_decay: float = 0.999
    """Starting teacher EMA decay for `masked_target='latent'`. Default is 0.999."""

    ema_decay_end: float = 0.9999
    """Final teacher EMA decay. The decay is ramped linearly from `ema_decay` to `ema_decay_end` over training (data2vec schedule): a
    fast-moving teacher early (more signal) that stabilises late. Set equal to `ema_decay` for a flat schedule. Default is 0.9999."""

    teacher_variance_floor: float = 1e-4
    """Minimum per-dimension std enforced when normalising the data2vec teacher target **across tokens** (not per-token).
    This is the anti-collapse fix for the masked objective -- a per-token LayerNorm target leaves between-token variance
    unconstrained and lets teacher and student co-collapse to a constant. Default is 1e-4."""

    cpc_steps: int = 4
    """How many future steps CPC predicts. Default is 4."""

    variance_weight: float = 0.0
    """Weight of the VICReg variance-hinge term (0 disables). Penalises any embedding dimension whose batch std falls below
    `variance_target`, which is what prevents the InfoNCE/L1 objectives from collapsing into ~15 of 768 dimensions.
    This is the single biggest metric mover among the anti-collapse levers. Default is 0.0."""

    covariance_weight: float = 0.0
    """Weight of the VICReg covariance term (0 disables). Pushes off-diagonal feature covariances toward zero so dimensions
    carry decorrelated information (higher effective rank). Default is 0.0."""

    variance_target: float = 1.0
    """Target per-dimension std (`gamma`) for the variance-hinge term. Default is 1.0."""

    anisotropy_weight: float = 0.0
    """Weight of an anti-cone *uniformity* penalty (0 disables). A Wang & Isola uniformity term spreads the L2-normalised
    embeddings over the sphere so their angular arrangement cannot degenerate. It complements `whiten` (which removes the shared-mean cone)
    by keeping directions well spread; pair both with VICReg (variance + covariance). Default is 0.0."""

    whiten: bool = False
    """If `True`, the exported embeddings are ZCA-whitened at evaluation (centre + decorrelate + equalise variance).
    This is the direct fix for the "cone" (anisotropy ~0.997) and dimensional collapse that otherwise appears: centring removes
    the dominant shared direction (anisotropy -> ~0) and whitening spreads variance across all dimensions (effective rank -> full).
    Because it is label-free, all downstream metrics are recomputed on the whitened space, so the evaluation report honestly shows whether
    content (retrieval, clustering) survives the fix. Default is `False`."""

    cross_subject_positives: bool = False
    """For skip-gram/CBOW, build contrastive positives from the **same stimulus read by different subjects** (using the batch's `content_id`)
    instead of same-subject neighbours. This turns subject identity from a shortcut into a nuisance the loss must remove.  Requires a
    stimulus-grouped batch sampler to be effective; falls back to within-sentence neighbours when no cross-subject positive is present in the batch."""

    meaning_positives: bool = False
    """If `True` (skip-gram), also draw positive pairs from the *same content word occurring in different sentences*
    (subject-agnostic word identity), not only the same stimulus token. This gives the "same meaning across contexts"
    structure room to grow instead of memorising which passage a word came from. Default is `False`."""

    subject_adversary_weight: float = 0.0
    """Weight of a gradient-reversal subject-adversary loss (0 disables). An auxiliary head tries to classify the subject from the
    token hiddens; the reversed gradient trains the encoder to *hide* subject identity, directly lowering subject decodability toward chance."""

    stimulus_adversary_weight: float = 0.0
    """Weight of a second gradient-reversal adversary that tries to predict *which stimulus/passage* a token came from (0 disables).
    It removes the "which of the sentence-sets" shortcut so the model must encode content rather than passage identity.
    Requires `content_id` in the batch. Default is 0.0."""

    # -- Meaning distillation + confound-matched hard negatives -------------- #
    meaning_distill_weight: float = 0.0
    """Weight of a distillation loss that pulls each word's embedding toward a *frozen language-model vector*
    for the word it read (0 disables). This is the direct attack on `content = 0%`: skip-gram never populated
    content because random negatives are separable by identity/task; an explicit meaning target forces the token in."""

    meaning_source: str | None = None
    """Where the frozen word vectors come from: a path to a `word v1 v2 ...` text file (GloVe/fastText format) or an `.npy`+vocab pair,
    or `None` / `hash` for a deterministic hash embedding (mechanism verification only -- carries no semantics)."""

    meaning_dim: int = 64
    """Meaning-vector dimensionality (must match `meaning_source`; used as-is for `'hash'`)."""

    hard_negatives: bool = False
    """For skip-gram/CBOW, restrict InfoNCE negatives to tokens that *share the confound* (same subject and task) so the only way
    to tell anchor from negative is the word itself.  De-confounds the contrastive objective, since subject and task are
    otherwise near-perfectly predictable from the negatives."""

    hard_negative_keys: tuple[str, ...] = ('subject', 'task')
    """Batch fields that a negative must match the anchor on when `hard_negatives` is set."""

    # -- Eye-tracking privileged supervision -------------------------------- #
    behaviour_weight: float = 0.0
    """Weight of an auxiliary head predicting per-word reading behaviour (fixation difficulty) from the embedding (0 disables).
    Behaviour is a lexical-difficulty proxy the EEG-only space struggles to find, so predicting it injects a meaning-adjacent gradient"""

    behaviour_targets: tuple[str, ...] = ('TRT', 'regression_time', 'is_omitted')
    """Which per-word behaviour signals the auxiliary head regresses/classifies."""

    # -- Fix the retrieval geometry (anti-hubness / anti-anisotropy) ------------------------ #
    all_but_top: int = 0
    """Remove the top-`all_but_top` principal directions from the exported embeddings at evaluation (0 disables).
    The label-free *all-but-the-top* post-processing of Mu & Viswanath (2018): after centring, nulling the few dominant PCA directions
    strips the shared frequency/hub axis that makes a below-chance retrieval space (the textbook symptom of anisotropy + hubness).
    Applied in the same post-processing block as `whiten` in `evaluation/report.py`, so every metric is honestly recomputed on the corrected space.
    Whiten then ABTT is the right order (whiten equalises variance; ABTT strips residual shared axes)."""

    csls_neighbors: int = 0
    """Neighbourhood size `k` for CSLS retrieval correction (0 = plain cosine). Cross-domain Similarity Local Scaling (Conneau et al., 2018):
    each similarity is corrected to `2 * cos - r_query - r_bank`, where `r` is the mean cosine to a point's `k` nearest neighbours,
    penalising hub-dense regions that are everyone's nearest neighbour. A monotone re-ranking (adds no signal), applied consistently to
    the retrieval index and its permutation null so the reported Top-1 and its p-value stay coherent."""

    # -- Ramp the subject adversary from zero (domain-adversarial schedule) ----------------- #
    subject_adversary_warmup_ratio: float = 0.0
    """Fraction of total optimiser steps over which the subject-adversary gradient-reversal strength `lambda_` ramps linearly 0 -> 1
    (0 = full strength from step 0). A cold adversary early lets the encoder learn content before invariance pressure is applied,
    avoiding the over-aggressive early inversion that erases the very content it should preserve (Ganin et al., 2016; Zhao et al., 2019).
    The flat loss weight stays at `subject_adversary_weight`; only the reversal strength ramps."""

    # -- The missing alignment half of the align+uniformity pair ---------------------------- #
    alignment_weight: float = 0.0
    """Weight of an explicit *alignment* penalty `E_{(i,j) in pos} ||center_i - context_j||^2` over the contrastive positive pairs (0 disables).
    `anisotropy_weight` already supplies the Wang & Isola (2020) *uniformity* half; this closes the theory's other half,
    pulling positives together and directly tightening the same-word geometry retrieval depends on."""

    # -- Debiased contrastive (stop punishing correct answers) ------------------------------ #
    tau_plus: float = 0.0
    """Class-prior `tau_plus` for the debiased contrastive estimator of Chuang et al. (2020), 0 = plain InfoNCE. In a word-level batch another
    EEG trial of the same word is a *false negative*; the debiased estimator subtracts an estimate of that positive mass from the negative log-sum-exp,
    so the loss stops shoving semantically-identical items apart. Small (`~0.05-0.1`) in low-SNR EEG batches."""

    # -- Collapse-proof regression auxiliary (fills idle nuisance dims) --------------------- #
    data2vec_aux_weight: float = 0.0
    """Weight of a frozen-target regression auxiliary on the **nuisance** subspace (0 disables). A collapse-resistant complement to InfoNCE
    (data2vec / HuBERT spirit): the nuisance dims of a factored embedding regress toward a *fixed* random projection of the token's own input
    features, which cannot co-collapse (the target is frozen) and gives the otherwise-idle `embed_dim - content_dim` dimensions a job -- so the
    factored design's nuisance room is actually used instead of left ungoverned."""

    # -- Per-occurrence contextual meaning target ------------------------------------------- #
    meaning_contextual: str | None = None
    """HuggingFace model id (e.g. `'bert-base-uncased'`) for a **per-occurrence contextual** meaning target, or `None` to use the word-type-keyed
    `meaning_source` file. When set, each word's distillation target is its contextual last-hidden state from a frozen encoder run on the whole sentence
    (sub-words mean-pooled), so polysemy the static GloVe/fastText target collapses is disambiguated. Requires `transformers`;
    falls back to the static path with a warning if unavailable. See `data.meaning.build_meaning_matrix_hf`."""

    meaning_context_layer: int = -1
    """Which hidden layer of the contextual model to read (a middle layer ~7-9 aligns best with brain activity;
    Toneva & Wehbe 2019, Caucheteux & King 2022). `-1` = the last hidden state."""

    # -- Evaluation hardening (opt-in, heavier checks) -------------------------------------- #
    eval_phase_shuffle: bool = False
    """Add a **phase-scrambled-input** control representation: the same trained encoder run on FFT-phase-randomised EEG (power spectrum preserved,
    temporal/phase structure destroyed). Proves the encoder invents no structure from spectrum alone. Informative only for raw frontends
    (band power is near-phase-invariant); a no-op-by-construction for band-power models, reported honestly."""

    eval_seen_novel: bool = False
    """Split cross-subject word retrieval into *seen* vs *novel* word types (novel = absent from the
    training split), so "zero-shot" means unseen word types, not only unseen subjects."""

    eval_freq_matched: bool = False
    """Restrict each retrieval query's distractor bank to its own frequency/length bin, so a hit reflects
    content rather than a lexical-frequency shortcut (a rare word standing out among common ones)."""

    # -- CLIP sentence-alignment objective (name='clip') ------------------------------------ #
    text_source: str | None = None
    """Frozen text-encoder model id for the CLIP sentence target (`name='clip'`), e.g.  `'intfloat/e5-base-v2'` / `'BAAI/bge-base-en-v1.5'`
    (sentence-transformers) or `'Qwen/Qwen2.5-0.5B'` (a decoder LLM, mean-pooled). Each unique ZuCo sentence is embedded once with this frozen model,
    and the EEG encoder is trained to align to it (Radford et al. 2021, CLIP; Défossez et al. 2023 for EEG/MEG).
    `None` falls back to a deterministic hash target (mechanism only, no semantics). Default is `None`."""

    text_backend: Literal['auto', 'sentence-transformers', 'hf'] = 'auto'
    """How to load `text_source`: `sentence-transformers` (E5/BGE and friends), `hf` (a raw HuggingFace model, mean-pooled over the
    attention mask — for decoder LLMs like Qwen), or `auto` (sentence-transformers if installed and the id is not obviously a decoder LLM, else hf)."""

    text_query_prefix: str = ''
    """Optional instruction prefix prepended to each sentence before encoding (some retrieval encoders such as E5
    expect `'query: '` / `'passage: '`). Empty for BGE/Qwen and most models."""

    clip_temperature: float = 0.07
    """Initial CLIP temperature; the log-scale is a learnable parameter (clamped), as in CLIP. Default is 0.07."""

    semantic_hard_negatives: bool = False
    """Bias each training batch to co-locate every anchor sentence with its *semantically-hard* negatives -- sentences that are
    **surface/syntactically similar but semantically distinct** (high token-overlap, low text-embedding cosine). This forces the encoder to
    represent meaning rather than surface form, and is the novelty lever of the CLIP recipe. Complementary to the confound-matched `hard_negatives`."""

    hard_negative_pool: int = 8
    """Number of semantic-hard negatives mined per sentence (used only when `semantic_hard_negatives`). Default is 8."""
