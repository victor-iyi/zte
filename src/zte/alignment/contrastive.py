"""What the contrastive term bought, per level: alignment, uniformity, effective rank and the positive/negative gap."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from zte.alignment.atlas import LEVELS, Level
from zte.evaluation import metrics as M
from zte.evaluation.audit.scoreboard import _bootstrap_ci

type PositivePolicy = Literal['any', 'cross_subject']
"""Which same-identity pairs count as positives."""

# The training term is `log E[exp(-2 ||x - y||^2)]` over unit vectors, so the analysis has to read at the same t.
UNIFORMITY_T: Final[float] = 2.0
"""RBF temperature of the uniformity term, matched to the anti-cone penalty the objective minimises."""

# One anchor is one query, and queries within a stimulus are not independent, so the anchor is the resampling unit.
MAX_ANCHORS: Final[int] = 512
"""Upper bound on the anchors a level contributes, keeping the similarity matrix bounded."""


@dataclass(slots=True, frozen=True, kw_only=True)
class LevelPairs:
    """One level's vectors plus the identity that decides which pairs the contrastive term pulls together."""

    level: Level
    """Which rung these vectors sit on."""

    vectors: np.ndarray
    """`(n, d)` embeddings for this level."""

    positive_ids: np.ndarray
    """`(n,)` identity per vector -- two rows sharing one id are a positive pair."""

    subjects: np.ndarray | None = None
    """`(n,)` subject code per vector; required by the `cross_subject` policy."""

    def __post_init__(self) -> None:
        """Rejects a ragged level rather than pairing rows against ids that belong to other vectors."""
        if self.level not in LEVELS:
            raise ValueError(f'Unknown level {self.level!r}; expected one of {", ".join(LEVELS)}.')

        vectors = np.asarray(self.vectors)
        if vectors.ndim != 2 or vectors.shape[0] < 2 or vectors.shape[1] == 0:
            raise ValueError(f'Level {self.level!r} needs at least two (n, d) vectors; got shape {vectors.shape}.')

        n = int(vectors.shape[0])
        for name, column in (('positive_ids', self.positive_ids), ('subjects', self.subjects)):
            if column is not None and len(np.asarray(column)) != n:
                raise ValueError(f'Level {self.level!r} has {n} vectors but {len(np.asarray(column))} {name}.')


def contrastive_geometry(
    levels: Sequence[LevelPairs],
    *,
    policy: PositivePolicy = 'any',
    max_anchors: int = MAX_ANCHORS,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Measures the contrastive geometry of each level and returns one payload comparing the three.

    Alignment and uniformity are the two halves the contrastive objective trades off: alignment is the mean
    cosine over positive pairs, and uniformity is `log E[exp(-2 ||x - y||^2)]` over unit vectors, the same
    anti-cone term the training loss minimises. A gap between positive and negative similarity is what the
    term is paid to open, so it carries a bootstrap CI over anchors -- an interval crossing zero bought
    nothing. Effective rank travels beside them because a tight alignment reached by collapsing the space is
    not a result.

    Args:
        levels (Sequence[LevelPairs]): One entry per level.
        policy (PositivePolicy, optional): `'any'` counts every same-id pair; `'cross_subject'` keeps only
            pairs read by different people, which is the honest positive. Defaults to 'any'.
        max_anchors (int, optional): Cap on the anchors scored per level. Defaults to MAX_ANCHORS.
        n_boot (int, optional): Bootstrap resamples for the gap CI. Defaults to 2000.
        seed (int, optional): Seed for the anchor sample and the bootstrap. Defaults to 0.

    Returns:
        dict[str, Any]: `levels` keyed by level name, `positive_policy`, `widest_gap` and a note.

    Raises:
        ValueError: If no levels are given, a level appears twice, or `cross_subject` is asked of a level
            that carries no subject codes.
    """
    if not levels:
        raise ValueError('The contrastive report needs at least one level.')

    named = [pairs.level for pairs in levels]
    if len(set(named)) != len(named):
        raise ValueError(f'Each level may appear at most once; got {named}.')

    blocks = {pairs.level: _level_block(pairs, policy, max_anchors, n_boot, seed) for pairs in levels}
    scored = [(name, block) for name, block in blocks.items() if block['positive_negative_gap'] is not None]
    ranked = sorted(scored, key=lambda item: float(item[1]['positive_negative_gap']), reverse=True)
    widest = ranked[0][0] if ranked else None

    return {
        'schema': 'zte.alignment.contrastive/1',
        'positive_policy': policy,
        'seed': seed,
        'levels': {level: blocks[level] for level in LEVELS if level in blocks},
        'widest_gap': widest,
        'note': (
            'Alignment and uniformity are the two halves of the contrastive objective; the gap is read against '
            'its bootstrap CI, and effective rank says whether alignment was bought by collapsing the space.'
        ),
    }


def _level_block(pairs: LevelPairs, policy: PositivePolicy, max_anchors: int, n_boot: int, seed: int) -> dict[str, Any]:
    """Scores one level's contrastive geometry; an unscorable level reports `None`, never a fabricated zero."""
    vectors = np.asarray(pairs.vectors, dtype=np.float64)
    unit = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    ids = np.asarray(pairs.positive_ids)
    subjects = None if pairs.subjects is None else np.asarray(pairs.subjects)

    if policy == 'cross_subject' and subjects is None:
        raise ValueError(f'Level {pairs.level!r} carries no subjects, so the cross_subject policy cannot be applied.')

    n, dim = int(vectors.shape[0]), int(vectors.shape[1])
    rng = np.random.default_rng(seed)
    anchors = np.arange(n) if n <= max_anchors else np.sort(rng.choice(n, size=max_anchors, replace=False))

    # One (anchors, n) similarity block rather than a loop: the cost is bounded by max_anchors, not by n^2.
    sims = unit[anchors] @ unit.T
    positive = ids[anchors][:, None] == ids[None, :]
    positive[np.arange(len(anchors)), anchors] = False
    if policy == 'cross_subject' and subjects is not None:
        positive &= subjects[anchors][:, None] != subjects[None, :]

    negative = ids[anchors][:, None] != ids[None, :]
    scorable = positive.any(axis=1) & negative.any(axis=1)

    block: dict[str, Any] = {
        'level': pairs.level,
        'n': n,
        'embed_dim': dim,
        'n_groups': int(np.unique(ids).size),
        'n_anchors': int(scorable.sum()),
        'n_positive_pairs': int(positive[scorable].sum()),
        'uniformity': _finite(M.uniformity(vectors, t=UNIFORMITY_T)),
        'effective_rank': _finite(M.effective_rank(vectors)),
    }
    block['effective_rank_ratio'] = None if block['effective_rank'] is None else block['effective_rank'] / dim

    # An anchor with no positive or no negative is dropped and counted, never scored as a zero-similarity pair.
    if not scorable.any():
        return block | {
            'alignment': None,
            'alignment_loss': None,
            'mean_negative_cosine': None,
            'positive_negative_gap': None,
            'positive_negative_gap_ci': None,
            'gap_excludes_zero': None,
        }

    pos_mean = (sims * positive).sum(axis=1)[scorable] / positive.sum(axis=1)[scorable]
    neg_mean = (sims * negative).sum(axis=1)[scorable] / negative.sum(axis=1)[scorable]
    gap, lo, hi = _bootstrap_ci(pos_mean - neg_mean, n_boot=n_boot, seed=seed)

    alignment = float(pos_mean.mean())

    return block | {
        'alignment': alignment,
        # For unit vectors `||a - b||^2 = 2 - 2 a.b`, so this is the alignment penalty the objective adds.
        'alignment_loss': 2.0 - 2.0 * alignment,
        'mean_negative_cosine': float(neg_mean.mean()),
        'positive_negative_gap': float(gap),
        'positive_negative_gap_ci': [float(lo), float(hi)],
        'gap_excludes_zero': bool(lo > 0.0),
    }


def _finite(value: float) -> float | None:
    """Maps a non-finite metric to `None` so the payload stays strict JSON."""
    return float(value) if np.isfinite(value) else None
