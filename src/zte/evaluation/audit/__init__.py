"""Audits that stress-test the evaluation itself -- is the signal real, or an artefact?

- `zte.evaluation.audit.confound` -- factor-entanglement audit (the decisive task<->stimulus confound check).
- `zte.evaluation.audit.honesty` -- permutation nulls, held-out cross-subject decoding and anchor calibration.
- `zte.evaluation.audit.scoreboard` -- the honest held-out scoreboard (lift over raw features).
- `zte.evaluation.audit.rebaseline` -- the length-confound audit (length oracle, train-fitted post-processing, bit
  budget).
- `zte.evaluation.audit.menu` -- the menu-capacity audit (K-way closed-set accuracy, certified capacity at a target).
- `zte.evaluation.audit.calibration` -- the anchor-calibration curve (what a few labelled sentences buy a stranger).
- `zte.evaluation.audit.evidence` -- the evidence board (every claim beside the brain-free floor it must clear).
"""

from __future__ import annotations

__all__: list[str] = []
