"""The cross-level comparison table: hits out of N, an exact binomial tail, rank percentile, and the oracle floor."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from zte.alignment.atlas import LEVELS, Level
from zte.evaluation.audit.rebaseline import piece_profile_report
from zte.evaluation.audit.scoreboard import _binom_tail_p, _bootstrap_ci

# Rank percentile uses every query rather than only the winners, which is the whole reason it is the headline.
HEADLINE_METRIC: Final[str] = 'rank_percentile'
"""The one metric a cross-level claim is allowed to lead with."""

# A token-level number sits on a sub-word signature the model was handed for free, so the floor is not optional.
_TOKEN_FLOOR_REQUIRED: Final[str] = (
    'A token-level row must carry its piece-profile oracle floor -- build it with `token_oracle_floor`. '
    'A token number quoted without that floor is not evidence of decoding.'
)
"""Refusal message for a token row with no floor beside it."""


@dataclass(slots=True, frozen=True, kw_only=True)
class LevelRetrieval:
    """One level's held-out retrieval, as the 1-based rank the correct item was given per query."""

    level: Level
    """Which rung was retrieved."""

    ranks: np.ndarray
    """`(n_queries,)` 1-based rank of the correct item, `1` meaning it came first."""

    gallery_size: int
    """How many candidates each query was ranked against."""

    postprocess_fit: str
    """`none`, `train split` or `transductive` -- the number is unreadable without it."""

    oracle_floor: dict[str, Any] | None = None
    """The brain-free floor this level must clear; mandatory at the token level."""

    def __post_init__(self) -> None:
        """Refuses a row whose ranks cannot have come from its own gallery, or a token row with no floor."""
        if self.level not in LEVELS:
            raise ValueError(f'Unknown level {self.level!r}; expected one of {", ".join(LEVELS)}.')

        if self.level == 'token' and self.oracle_floor is None:
            raise ValueError(_TOKEN_FLOOR_REQUIRED)

        if int(self.gallery_size) < 1:
            raise ValueError(f'Level {self.level!r} needs a gallery of at least one item; got {self.gallery_size}.')

        ranks = np.asarray(self.ranks)
        if ranks.ndim != 1 or ranks.size == 0:
            raise ValueError(f'Level {self.level!r} needs a non-empty 1-D rank vector; got shape {ranks.shape}.')

        if float(ranks.min()) < 1.0 or float(ranks.max()) > float(self.gallery_size):
            raise ValueError(
                f'Level {self.level!r} has ranks in [{ranks.min()}, {ranks.max()}], outside a gallery of '
                f'{self.gallery_size}; ranks are 1-based and bounded by the gallery.'
            )

        if self.oracle_floor is not None and not {'gate_top1', 'worst_case_top1'} & set(self.oracle_floor):
            raise ValueError(f'Level {self.level!r} was handed an oracle floor with no Top-1 to compare against.')


def token_oracle_floor(
    word_pieces: np.ndarray, *, observed_top1: float | None = None, ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, Any]:
    """Scores the brain-free sub-word oracles a token-level number has to clear.

    A token objective that gives a word as many EEG sub-tokens as the reference spells it word-pieces hands
    the model the sentence's piece profile, and on a 700-sentence gallery that profile alone resolves nearly
    every sentence. The floor is therefore the token level's counterpart to the sentence level's word-count
    oracle, and it is measured, not assumed.

    Args:
        word_pieces (np.ndarray): `TokenAlignment.word_pieces`, `(n_text, max_words)`.
        observed_top1 (float | None, optional): The level's measured Top-1, compared against every oracle.
            Defaults to None.
        ks (tuple[int, ...], optional): Top-K cut-offs. Defaults to (1, 5, 10).

    Returns:
        dict[str, Any]: The `piece_profile_report` block, carrying `worst_case_top1`, `worst_case_signature`
            and its verdict.
    """
    return piece_profile_report(word_pieces, observed_top1=observed_top1, ks=ks)


def cross_level_table(
    rows: Sequence[LevelRetrieval], *, ks: tuple[int, ...] = (1, 5, 10), n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Scores every level on one comparable footing: hit counts, exact tails, rank percentile and the floor.

    Top-K is reported as a hit count out of the queries actually scored, because a rate against a gallery of
    hundreds hides that the expected number of chance hits is about one. The exact binomial tail says whether
    the count is surprising; the rank percentile, with a bootstrap CI, is the metric that uses every query.

    Args:
        rows (Sequence[LevelRetrieval]): One entry per level.
        ks (tuple[int, ...], optional): Top-K cut-offs. Defaults to (1, 5, 10).
        n_boot (int, optional): Bootstrap resamples for the rank-percentile CI. Defaults to 2000.
        seed (int, optional): Bootstrap seed. Defaults to 0.

    Returns:
        dict[str, Any]: `levels` (one block per level, in token-word-sentence order), `headline_metric`
            and a note.

    Raises:
        ValueError: If no rows are given, a level appears twice, or a token row carries no oracle floor.
    """
    if not rows:
        raise ValueError('The cross-level table needs at least one level.')

    named = [row.level for row in rows]
    if len(set(named)) != len(named):
        raise ValueError(f'Each level may appear at most once; got {named}.')

    blocks = {row.level: _row_block(row, ks, n_boot, seed) for row in rows}

    return {
        'schema': 'zte.alignment.compare/1',
        'headline_metric': HEADLINE_METRIC,
        'levels': [blocks[level] for level in LEVELS if level in blocks],
        'note': (
            'Top-K is a hit count out of the queries scored, with an exact binomial tail; the headline is the '
            'rank percentile, and every level is read against the brain-free floor printed beside it.'
        ),
    }


def render_markdown(table: dict[str, Any]) -> str:
    """Renders a `cross_level_table` payload as the markdown table a report embeds.

    Args:
        table (dict[str, Any]): A `cross_level_table` payload.

    Returns:
        str: Markdown, one row per level, floors and post-processing included.
    """
    lines = [
        '### Cross-level retrieval',
        '',
        '| Level | Gallery | Queries | Top-1 hits | Top-1 p | Rank pct (95% CI) | Floor Top-1 | Clears floor | Post-proc |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |',
    ]
    for block in table.get('levels', []):
        floor = block.get('oracle_floor') or {}
        ci = block.get('rank_percentile_ci') or [None, None]
        lines.append(
            f'| {block["level"]} | {block["gallery_size"]} | {block["n_queries"]} '
            f'| {block["hits_top1"]} | {_fmt(block.get("top1_p"))} '
            f'| {_fmt(block.get("rank_percentile"))} [{_fmt(ci[0])}, {_fmt(ci[1])}] '
            f'| {_fmt(floor.get("top1"))} | {_verdict(block.get("beats_oracle_floor"))} '
            f'| {block.get("postprocess_fit", "unstated")} |'
        )

    lines += ['', f'Headline metric: `{table.get("headline_metric", HEADLINE_METRIC)}`.', '', table.get('note', '')]

    return '\n'.join(lines).rstrip() + '\n'


def _row_block(row: LevelRetrieval, ks: tuple[int, ...], n_boot: int, seed: int) -> dict[str, Any]:
    """Scores one level's ranks into the comparable block the table prints."""
    ranks = np.asarray(row.ranks, dtype=np.float64)
    n_queries = int(ranks.size)
    n_gallery = int(row.gallery_size)
    chance = 1.0 / n_gallery

    block: dict[str, Any] = {
        'level': row.level,
        'gallery_size': n_gallery,
        'n_queries': n_queries,
        'chance_top1': chance,
        'postprocess_fit': row.postprocess_fit,
        'headline_metric': HEADLINE_METRIC,
    }

    for k in ks:
        hits = int((ranks <= k).sum())
        block[f'hits_top{k}'] = hits
        block[f'top{k}'] = hits / n_queries
        block[f'top{k}_p'] = _binom_tail_p(hits, n_queries, min(chance * k, 1.0))

    # Rank percentile: 1.0 when the correct item came first, 0.0 when it came last.
    percentiles = 1.0 - (ranks - 1.0) / max(n_gallery - 1, 1)
    mean, lo, hi = _bootstrap_ci(percentiles, n_boot=n_boot, seed=seed)
    block['rank_percentile'] = float(mean)
    block['rank_percentile_ci'] = [float(lo), float(hi)]
    block['mean_rank'] = float(ranks.mean())
    block['mrr'] = float((1.0 / ranks).mean())

    if row.oracle_floor is not None:
        # The gate is the signature this encoder can actually reach; the ordered profile is reported beside it as
        # the ceiling a variable-K design would have given away, because gating on it would always read `below`.
        floor = row.oracle_floor
        floor_top1 = float(floor.get('gate_top1', floor['worst_case_top1']))
        block['oracle_floor'] = {
            'top1': floor_top1,
            'signature': floor.get('gate_signature', floor.get('worst_case_signature')),
            'ceiling_top1': floor.get('ceiling_top1'),
            'ceiling_signature': floor.get('ceiling_signature'),
            'clears': floor.get('clears'),
            'verdict': floor.get('verdict'),
        }
        block['beats_oracle_floor'] = bool(block['top1'] > floor_top1)

    return block


def _fmt(value: Any) -> str:
    """Formats a table cell, printing a dash rather than `None`."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '--'

    return f'{float(value):.4f}'


def _verdict(beats: bool | None) -> str:
    """Turns the floor comparison into a cell that reads the same way for every level."""
    if beats is None:
        return 'no floor measured'

    return 'yes' if beats else 'NO -- below a brain-free floor'
