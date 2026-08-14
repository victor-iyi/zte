"""Audits that stress-test the evaluation itself -- is the signal real, or an artefact?

- `zte.evaluation.audit.confound` -- factor-entanglement audit (the decisive task<->stimulus confound check).
- `zte.evaluation.audit.honesty` -- permutation nulls, held-out cross-subject decoding and anchor calibration.
- `zte.evaluation.audit.scoreboard` -- the honest held-out scoreboard (lift over raw features).
- `zte.evaluation.audit.rebaseline` -- the length-confound audit (length oracle, train-fitted post-processing, bit
  budget).
"""

from __future__ import annotations

__all__: list[str] = []
