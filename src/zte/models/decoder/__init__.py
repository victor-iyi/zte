"""The frozen-LM prefix decoder.

- `zte.models.decoder.bridge` -- `PrefixBridge` and `WordResampler`, the trainable soft-prompt surface.
- `zte.models.decoder.evidence` -- `WordEvidence` and `MonotonicPointer`, the word-synchronous path.
- `zte.models.decoder.gap` -- `GapCorrector`, the train-fitted EEG-to-text correction.
- `zte.models.decoder.lm` -- `FrozenLM`: prompt assembly, scoring, rescoring and free-running decode.
- `zte.models.decoder.quantiser` -- `SemanticRateLadder`, the measured bit budget.
"""

from __future__ import annotations

from zte.models.decoder.bridge import PrefixBridge, WordResampler, build_bridge
from zte.models.decoder.evidence import MonotonicPointer, WordEvidence, build_evidence, measure_tokens_per_word
from zte.models.decoder.gap import GapCorrector
from zte.models.decoder.lm import EvidenceFn, FrozenLM, build_lm
from zte.models.decoder.quantiser import LadderOutput, SemanticRateLadder, build_rate_ladder

__all__ = [
    'EvidenceFn',
    'FrozenLM',
    'GapCorrector',
    'LadderOutput',
    'MonotonicPointer',
    'PrefixBridge',
    'SemanticRateLadder',
    'WordEvidence',
    'WordResampler',
    'build_bridge',
    'build_evidence',
    'build_lm',
    'build_rate_ladder',
    'measure_tokens_per_word',
]
