from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from zte.config._paths import PathFields
from zte.config.types import Conditioning, GapCorrection


@dataclass
class DecoderConfig(PathFields):
    """Frozen-LM prefix decoder: bridge geometry, targets, decoding and generation controls."""

    _PATH_FIELDS: ClassVar[tuple[str, ...]] = ('lm_cache_dir',)

    lm_source: str = 'Qwen/Qwen2.5-0.5B'
    """Frozen causal LM. `'tiny'` builds a 2-layer 32-wide model locally with no network, for tests and smoke runs."""

    lm_revision: str | None = None
    """Pinned commit SHA for `lm_source`. `None` resolves to `main` and is logged as a reproducibility risk."""

    tokenizer_source: str | None = None
    """Tokeniser id for the target text; `None` uses `lm_source`."""

    lm_cache_dir: str | None = None
    """Local snapshot directory for the LM, so a Colab session is offline-reproducible."""

    conditioning: Conditioning = 'pooled'
    """What the bridge reads. `pooled_plus_words` is the ablation arm: it also hands the decoder the word count, which
    carries 5.14 bits of sentence identity on ZuCo, so its controls must be length-matched."""

    prefix_slots: int = 8
    """Soft-prompt slots produced from the pooled vector."""

    word_slots: int = 8
    """Extra slots produced by the word resampler; used only when `conditioning='pooled_plus_words'`."""

    bottleneck: int = 128
    """Rank of the shared low-rank map inside the bridge."""

    gap_correction: GapCorrection = 'mean_scale'
    """How the EEG vector is mapped onto the text manifold. Fitted on the train split only, never transductively."""

    null_prefix_prob: float = 0.1
    """Probability of replacing the prefix with the learned null prefix during training, which trains the
    unconditional branch used by the `null_prefix` control and the prefix-influence diagnostic."""

    cfg_weight: float = 1.0
    """Classifier-free guidance weight. Asserted `1.0` so the headline decode and every control are the same code
    path; any other value is a labelled ablation and is rejected at generation time."""

    ground_weight: float = 0.5
    """Weight of the in-batch grounding loss, which forces the prefix to discriminate its own reference."""

    ground_negatives: int = 3
    """In-batch negative references scored against each item's own prefix."""

    max_target_tokens: int = 96
    """Tokenised reference length cap (ZuCo p99 is 51 whitespace words)."""

    max_new_tokens: int = 96
    """Free-running decode cap. The reference length is never supplied."""

    beams: int = 1
    """Beam width for free-running decode; 1 is greedy and deterministic."""

    stage0_epochs: int = 20
    """Text-only bridge pretraining epochs on (text embedding -> text) pairs. Train-split stimuli only, asserted."""

    min_prefix_kl: float = 0.05
    """Minimum mean KL (nats) between a reading's own prefix and another reading's for the generation verdict to be
    considered at all; below it the prompt does not depend on the brain."""

    clip_aux_weight: float = 1.0
    """Weight of the retained CLIP loss in `joint` mode, anchoring the unfrozen encoder in the text space."""

    cache_embeddings: bool = True
    """Cache the frozen encoder's sentence vectors by `reading_id` after the first epoch. Errors in `joint` mode."""

    prompt_template: str = '\nSentence: '
    """Fixed scaffold inserted between the prefix and the target, embedded with the frozen token embeddings."""

    generation_controls: tuple[str, ...] = (
        'mean_prefix',
        'null_prefix',
        'phase',
        'noise',
        'mismatch',
    )
    """Brain-independent controls decoded through the identical path; the headline is a paired delta against all."""

    n_permutations: int = 1000
    """Permutations for the generation null, which shuffles the hypothesis-to-reference pairing."""

    rescore_gallery: bool = True
    """Also score every gallery sentence by length-normalised sequence likelihood. This is retrieval, not generation,
    and is the statistically powered primary readout."""

    length_tol: int = 1
    """Word-count tolerance for length-stratified retrieval galleries and the mismatch-control derangement."""
