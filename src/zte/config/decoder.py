from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from zte.config._paths import PathFields
from zte.config.types import Conditioning, EvidenceSchedule, GapCorrection, LMDtype, RateLadder


@dataclass
class DecoderConfig(PathFields):
    """Frozen-LM prefix decoder: bridge geometry, the rate ladder, word-synchronous evidence and the controls."""

    _PATH_FIELDS: ClassVar[tuple[str, ...]] = ('lm_cache_dir',)

    # ---- The frozen language model ---- #

    lm_source: str = 'Qwen/Qwen2.5-0.5B'
    """Frozen causal LM. `'tiny'` builds a 2-layer 32-wide model locally with no network, for tests and smoke runs."""

    lm_revision: str | None = None
    """Pinned commit SHA for `lm_source`. `None` resolves to `main` and is logged as a reproducibility risk."""

    tokenizer_source: str | None = None
    """Tokeniser id for the target text; `None` uses `lm_source`."""

    lm_cache_dir: str | None = None
    """Local snapshot directory for the LM, so a Colab session is offline-reproducible."""

    lm_dtype: LMDtype = 'auto'
    """Precision the frozen LM runs in. `auto` reads it off the encoder the bridge is fed by, so the two halves of the
    pipeline are never at different precisions; naming one of the others pins it instead. Either way it is never taken
    from the HuggingFace checkpoint, whose stored dtype would otherwise make every token log-probability a property of
    the uploader's export choice. The half precisions halve the LM's memory and cost the last digits of every score, so
    a number produced under one is not comparable with a number produced under another."""

    prompt_template: str = '\nSentence: '
    """Fixed scaffold inserted between the prefix and the target, embedded with the frozen token embeddings."""

    # ---- The pooled prefix bridge ---- #

    conditioning: Conditioning = 'pooled'
    """What the bridge reads. `pooled_plus_words` is the ablation arm: it also hands the decoder the word count, which
    carries 5.14 bits of sentence identity on ZuCo, so its controls must be length-matched."""

    prefix_slots: int = 8
    """Soft-prompt slots produced from the pooled vector."""

    word_slots: int = 8
    """Extra slots produced by the word resampler; used only when `conditioning='pooled_plus_words'`."""

    bottleneck: int = 128
    """Rank of the shared low-rank map inside the bridge."""

    bridge_depth: int = 1
    """Residual MLP blocks between the bottleneck and the per-slot FiLM. `1` is the single linear map; deeper buys
    capacity the 700-sentence corpus cannot pay for, so it is an ablation knob rather than a default."""

    gap_correction: GapCorrection = 'mean_scale'
    """How the EEG vector is mapped onto the text manifold. Fitted on the train split only, never transductively."""

    # ---- The semantic rate ladder -- the measured bit budget ---- #

    rate_ladder: RateLadder = 'none'
    """Bottleneck the conditioning vector passes through before the bridge. `rvq` quantises it against text-anchored
    codebooks, so the conditioning channel carries at most `rate_stages * log2(rate_codes)` bits by construction and
    the empirical mutual information of each stage is measurable rather than argued. `none` is the continuous path."""

    rate_stages: int = 4
    """Residual quantiser stages. With `rate_codes=256` this is a 32-bit architectural ceiling against the 9.45 bits
    sentence identity actually needs, so the ladder is not the binding constraint -- it is the instrument."""

    rate_codes: int = 256
    """Codes per stage. The per-stage ceiling is `log2(rate_codes)` bits."""

    rate_commit_weight: float = 0.25
    """Weight of the commitment loss pulling the continuous vector onto its chosen codes."""

    rate_decay: float = 0.99
    """EMA decay for the codebook updates. Codes are updated by EMA rather than by gradient so a stage cannot be
    dragged off the text manifold by the decoding loss."""

    rate_revive_after: int = 200
    """Steps a code may go unused before it is re-seeded onto a high-residual vector. Dead codes silently shrink the
    real bit-rate below the architectural ceiling, which would make the measured budget a lie."""

    rate_length_stage: bool = False
    """Reserve stage 0 for sentence length -- the 5.14 bits ZuCo hands over free through eye-tracking word
    segmentation -- and penalise the remaining stages for carrying it. The reported headline is then the *residual*
    ladder, whose bits are the ones the brain actually supplied."""

    rate_length_weight: float = 1.0
    """Weight of the length head on the reserved stage, and of the length-orthogonality penalty on the others."""

    # ---- Word-synchronous lexical evidence ---- #

    evidence_schedule: EvidenceSchedule = 'none'
    """Whether the per-word EEG tokens steer the decode. A pooled prefix spends its influence in the first few
    generated tokens; the evidence path re-injects the brain at every step, reading the word the eye-tracking
    alignment says was being read at that point in the sentence. `none` is the pooled-only decoder."""

    evidence_rank: int = 64
    """Rank of the map from a word's text-space vector into the frozen LM's embedding space. The vocabulary readout is
    the LM's own frozen output head, so this rank is the entire trainable width of the evidence path."""

    evidence_width: float = 1.5
    """Standard deviation, in words, of the Gaussian pointer window. Wider tolerates a hypothesis that drifts out of
    step with the reference; narrower commits harder to the alignment."""

    evidence_tokens_per_word: float = 0.0
    """Pointer advance in words per generated token. `0` measures it from the tokenised training corpus, which is the
    honest default -- the rate is a property of the tokeniser, not a hyper-parameter."""

    evidence_gate_init: float = 0.0
    """Initial value of the scalar gate multiplying the evidence bias. Zero starts the decoder as the pooled-only
    decoder, so any influence the evidence path ends up with was earned by the loss."""

    evidence_max_bias: float = 4.0
    """Absolute cap, in logits, on the evidence bias. Without it the path can win the loss by saturating the
    distribution on a handful of tokens, which reads as decoding and is not."""

    # ---- The loss ---- #

    null_prefix_prob: float = 0.1
    """Probability of replacing the prefix with the learned null prefix during training, which trains the
    unconditional branch used by the `null_prefix` control and the prefix-influence diagnostic."""

    ground_weight: float = 0.5
    """Weight of the in-batch grounding loss, which forces the prefix to discriminate its own reference."""

    ground_negatives: int = 3
    """In-batch negative references scored against each item's own prefix."""

    ground_hard_length: bool = True
    """Draw the grounding negatives from references of a similar word count. A negative of obviously wrong length is
    separable on length alone, so an easy negative trains the prefix to encode the word count and nothing else."""

    lexical_weight: float = 0.0
    """Weight of the token-level lexical loss inside the decoder: each word's EEG token is scored against its own text
    embedding among the batch's other words. It is what gives the evidence path something true to say, and it is zero
    by default so the pooled decoder is unchanged."""

    clip_aux_weight: float = 1.0
    """Weight of the retained CLIP loss in `joint` mode, anchoring the unfrozen encoder in the text space."""

    # ---- Targets and decoding ---- #

    max_target_tokens: int = 96
    """Tokenised reference length cap (ZuCo p99 is 51 whitespace words)."""

    max_new_tokens: int = 96
    """Free-running decode cap. The reference length is never supplied."""

    beams: int = 1
    """Beam width for free-running decode; 1 is greedy and deterministic."""

    cfg_weight: float = 1.0
    """Classifier-free guidance weight. Asserted `1.0` so the headline decode and every control are the same code
    path; any other value is a labelled ablation and is rejected at generation time."""

    stage0_epochs: int = 20
    """Text-only bridge pretraining epochs on (text embedding -> text) pairs. Train-split stimuli only, asserted."""

    cache_embeddings: bool = True
    """Cache the frozen encoder's sentence vectors by `reading_id` after the first epoch. Errors in `joint` mode."""

    # ---- Evaluation ---- #

    min_prefix_kl: float = 0.05
    """Minimum mean KL (nats) between a reading's own prefix and another reading's for the generation verdict to be
    considered at all; below it the prompt does not depend on the brain."""

    generation_controls: tuple[str, ...] = (
        'mean_prefix',
        'null_prefix',
        'phase',
        'noise',
        'shuffled_z',
        'length_only',
        'mismatch',
    )
    """Brain-independent controls decoded through the identical path; the headline is a paired delta against all.
    `shuffled_z` deranges the conditioning vectors after the encoder, so it isolates the bridge from the encoder;
    `length_only` keeps the word count and destroys everything else, which is the control the 5.14-bit length
    confound demands."""

    n_permutations: int = 1000
    """Permutations for the generation null, which shuffles the hypothesis-to-reference pairing."""

    rescore_gallery: bool = True
    """Also score every gallery sentence by length-normalised sequence likelihood. This is retrieval, not generation,
    and is the statistically powered primary readout."""

    rescore_chunk: int = 64
    """Candidate rows per frozen-LM forward pass during gallery rescoring. The prompt's key/value cache is computed
    once per query and shared across the chunk, so this bounds memory rather than repeated work."""

    rescore_pmi: bool = False
    """PMI rescoring: subtract each candidate's null-prefix per-token log-likelihood from its conditional score,
    cancelling candidate-side familiarity bias -- every trainable part is fitted on train-cell reference texts, so a
    train-cell candidate scores well under *any* prefix and the difference keeps only what the brain added. Off keeps
    the raw conditional score."""

    length_tol: int = 1
    """Word-count tolerance for length-stratified retrieval galleries and the mismatch-control derangement."""

    within_task_pools: tuple[str, ...] = ('SR', 'NR')
    """Tasks whose candidate pool is also reported on its own. Inside one task the passage set is fixed, so a hit
    cannot be a passage or task shortcut -- the pool a sceptical reader asks for."""

    eval_seeds: tuple[int, ...] = field(default_factory=tuple)
    """Extra decode seeds re-run at evaluation time for a mean +/- sd headline. Empty uses the single decode seed;
    training-seed variation is a separate axis and is swept by `scripts/run_zte_study.sh`."""
