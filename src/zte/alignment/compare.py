"""The cross-level comparison table: hits out of N, an exact binomial tail, rank percentile, and the oracle floor."""

import statistics
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

# Twelve LOSO folds are a sample of the subjects a clinical claim would have to generalise to, so the spread is the
# n-1 sample sd; a population sd under-reports it by about 4% at n=12, which is the size of the effects here.
_SD_MIN_FOLDS: Final[int] = 2
"""Folds a sample standard deviation needs; below it the spread is unmeasured rather than zero."""


@dataclass(slots=True, frozen=True, kw_only=True)
class FoldSeries:
    """One gallery's held-out numbers, one entry per LOSO fold, as an already-evaluated run left them on disk."""

    rank_percentile: np.ndarray
    """`(n_folds,)` mean rank percentile inside that fold's own gallery."""

    top1: np.ndarray
    """`(n_folds,)` Top-1 rate inside that fold's own gallery."""

    n_queries: np.ndarray
    """`(n_folds,)` queries the fold actually scored -- what its hit count is out of."""

    chance_top1: np.ndarray
    """`(n_folds,)` the fold's own chance rate, which stratifying the gallery changes."""

    folds: tuple[str, ...] = ()
    """Which held-out subject each entry belongs to, so the aggregate can be traced back."""

    def __post_init__(self) -> None:
        """Refuses a series whose columns do not describe the same folds, or whose rates are not rates."""
        columns = (self.rank_percentile, self.top1, self.n_queries, self.chance_top1)
        widths = {int(np.asarray(column).size) for column in columns}
        if len(widths) != 1:
            raise ValueError(f'A fold series needs one entry per fold in every column; got sizes {sorted(widths)}.')

        (width,) = widths
        if width == 0:
            raise ValueError('A fold series needs at least one fold.')

        if self.folds and len(self.folds) != width:
            raise ValueError(f'A fold series was given {len(self.folds)} fold name(s) for {width} fold(s).')

        for name in ('rank_percentile', 'top1', 'chance_top1'):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if not bool(np.all(np.isfinite(values))) or float(values.min()) < 0.0 or float(values.max()) > 1.0:
                raise ValueError(f'A fold series needs finite {name} in [0, 1]; got {values.tolist()}.')

        if int(np.asarray(self.n_queries).min()) < 1:
            raise ValueError('Every fold in a series has to have scored at least one query.')


@dataclass(slots=True, frozen=True, kw_only=True)
class LevelRetrieval:
    """One level's held-out retrieval: either the per-query ranks, or the per-fold summary left on disk."""

    level: Level
    """Which rung was retrieved."""

    gallery_size: int
    """How many candidates each query was ranked against."""

    postprocess_fit: str
    """`none`, `train split` or `transductive` -- the number is unreadable without it."""

    ranks: np.ndarray | None = None
    """`(n_queries,)` 1-based rank of the correct item, `1` meaning it came first."""

    folds: FoldSeries | None = None
    """The per-fold numbers instead, for a table assembled from artifacts rather than from live embeddings."""

    stratified_folds: FoldSeries | None = None
    """The same folds inside a length-matched gallery, where a hit cannot be a sentence-length shortcut."""

    oracle_floor: dict[str, Any] | None = None
    """The brain-free Top-1 floor this level must clear; mandatory at the token level."""

    length_floor: dict[str, Any] | None = None
    """The length-only oracle's rank percentile, which the encoder's interval is read against."""

    effective_rank_ratio: float | None = None
    """Mean effective-rank ratio across folds -- invariance bought by destroying capacity shows up here."""

    effective_rank_ratio_sd: float | None = None
    """Sample sd of that ratio across folds."""

    missing: tuple[str, ...] = ()
    """Artifacts or blocks that were not on disk, named so a `--` cell is never mistaken for a pass."""

    def __post_init__(self) -> None:
        """Refuses a row with no measurement, ranks outside its own gallery, or a token row with no floor."""
        if self.level not in LEVELS:
            raise ValueError(f'Unknown level {self.level!r}; expected one of {", ".join(LEVELS)}.')

        if (self.ranks is None) == (self.folds is None):
            raise ValueError(
                f'Level {self.level!r} needs exactly one of `ranks` (per query) or `folds` (per LOSO fold).'
            )

        if self.level == 'token' and self.oracle_floor is None:
            raise ValueError(_TOKEN_FLOOR_REQUIRED)

        if int(self.gallery_size) < 1:
            raise ValueError(f'Level {self.level!r} needs a gallery of at least one item; got {self.gallery_size}.')

        if self.stratified_folds is not None:
            if self.folds is None:
                raise ValueError(f'Level {self.level!r} carries a stratified fold series with no unstratified one.')

            widths = (np.asarray(self.folds.top1).size, np.asarray(self.stratified_folds.top1).size)
            if widths[0] != widths[1]:
                raise ValueError(
                    f'Level {self.level!r} has {widths[0]} unstratified fold(s) against {widths[1]} stratified; '
                    'the two galleries must describe the same folds to be comparable.'
                )

        if self.ranks is not None:
            ranks = np.asarray(self.ranks)
            if ranks.ndim != 1 or ranks.size == 0:
                raise ValueError(f'Level {self.level!r} needs a non-empty 1-D rank vector; got shape {ranks.shape}.')

            if float(ranks.min()) < 1.0 or float(ranks.max()) > float(self.gallery_size):
                raise ValueError(
                    f'Level {self.level!r} has ranks in [{ranks.min()}, {ranks.max()}], outside a gallery of '
                    f'{self.gallery_size}; ranks are 1-based and bounded by the gallery.'
                )

        if self.oracle_floor is not None and all(
            self.oracle_floor.get(key) is None for key in ('gate_top1', 'worst_case_top1')
        ):
            raise ValueError(f'Level {self.level!r} was handed an oracle floor with no Top-1 to compare against.')

        if self.length_floor is not None and self.length_floor.get('rank_percentile') is None:
            raise ValueError(f'Level {self.level!r} was handed a length floor with no rank percentile to clear.')


@dataclass(slots=True, frozen=True, kw_only=True)
class UnmeasuredLevel:
    """A level whose runs were found but whose number cannot honestly be quoted, and what is missing."""

    level: Level
    """Which rung could not be scored."""

    missing: tuple[str, ...]
    """The artifacts or blocks whose absence is the reason, named exactly as they would appear on disk."""

    n_folds: int = 0
    """How many folds were found for it, so the row shows the work exists and only the floor does not."""

    reason: str = ''
    """One sentence a reader can act on."""

    def __post_init__(self) -> None:
        """Refuses an unmeasured row that does not say what is missing."""
        if self.level not in LEVELS:
            raise ValueError(f'Unknown level {self.level!r}; expected one of {", ".join(LEVELS)}.')

        if not self.missing:
            raise ValueError(f'Level {self.level!r} cannot be reported as unmeasured without naming what is missing.')


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
    rows: Sequence[LevelRetrieval],
    *,
    unmeasured: Sequence[UnmeasuredLevel] = (),
    ks: tuple[int, ...] = (1, 5, 10),
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Scores every level on one comparable footing: hit counts, exact tails, rank percentile and the floor.

    Top-K is reported as a hit count out of the queries actually scored, because a rate against a gallery of
    hundreds hides that the expected number of chance hits is about one. The exact binomial tail says whether
    the count is surprising; the rank percentile, with a bootstrap CI, is the metric that uses every query.

    Args:
        rows (Sequence[LevelRetrieval]): One entry per level that could be scored.
        unmeasured (Sequence[UnmeasuredLevel], optional): Levels that were found but cannot be quoted, which
            appear as explicit rows carrying no number. Defaults to ().
        ks (tuple[int, ...], optional): Top-K cut-offs, used by the per-query form. Defaults to (1, 5, 10).
        n_boot (int, optional): Bootstrap resamples for the rank-percentile CI. Defaults to 2000.
        seed (int, optional): Bootstrap seed. Defaults to 0.

    Returns:
        dict[str, Any]: `levels` (one block per level, in token-word-sentence order), `verdict`,
            `headline_metric` and a note.

    Raises:
        ValueError: If no level is given, a level appears twice, or a token row carries no oracle floor.
    """
    if not rows and not unmeasured:
        raise ValueError('The cross-level table needs at least one level.')

    named = [row.level for row in rows] + [entry.level for entry in unmeasured]
    if len(set(named)) != len(named):
        raise ValueError(f'Each level may appear at most once; got {named}.')

    blocks: dict[str, dict[str, Any]] = {row.level: _row_block(row, ks, n_boot, seed) for row in rows}
    blocks |= {entry.level: _unmeasured_block(entry) for entry in unmeasured}
    ordered = [blocks[level] for level in LEVELS if level in blocks]

    return {
        'schema': 'zte.alignment.compare/1',
        'headline_metric': HEADLINE_METRIC,
        'levels': ordered,
        'verdict': _cross_level_verdict(ordered),
        'note': (
            'Top-K is a hit count out of the queries scored, with an exact binomial tail; the headline is the '
            'rank percentile, and every level is read against the brain-free floor printed beside it.'
        ),
    }


def render_markdown(table: dict[str, Any]) -> str:
    """Renders a `cross_level_table` payload as the markdown a report embeds.

    Args:
        table (dict[str, Any]): A `cross_level_table` payload.

    Returns:
        str: Markdown -- the full gallery, its length-stratified twin, the floors, and the reading of them.
    """
    blocks = list(table.get('levels', []))
    lines = ['### Cross-level retrieval', '', *_full_gallery_table(blocks)]

    if any(block.get('length_stratified') or block.get('effective_rank_ratio') is not None for block in blocks):
        lines += ['', '### Length-stratified twin, floors and geometry', '', *_stratified_table(blocks)]

    lines += ['', f'Headline metric: `{table.get("headline_metric", HEADLINE_METRIC)}`.', *_basis_note(blocks)]

    if reading := (table.get('verdict') or {}).get('reading'):
        lines += ['', reading]

    lines += ['', table.get('note', '')]

    return '\n'.join(lines).rstrip() + '\n'


def _basis_note(blocks: Sequence[dict[str, Any]]) -> list[str]:
    """Says what a fold-aggregated interval and tail were actually computed over, which changes how they read."""
    if not any(block.get('n_folds') for block in blocks):
        return []

    return [
        '',
        'Aggregated over LOSO folds: the spread is the n-1 sample sd across folds and the interval is a bootstrap '
        'over fold means, so both describe variation between subjects rather than between queries. The Top-1 tail '
        "pools every fold's queries into one exact binomial, which readings from twelve different subjects are "
        'only approximately independent trials of.',
    ]


def _full_gallery_table(blocks: Sequence[dict[str, Any]]) -> list[str]:
    """The headline table: every level's hit count, tail and rank percentile beside the floor it must clear."""
    lines = [
        '| Level | Folds | Gallery | Queries | Top-1 hits | Top-1 p | Oracle floor Top-1 '
        '| Rank pct ± sd (95% CI) | Length floor rank pct | Clears floor | Post-proc |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |',
    ]
    for block in blocks:
        length = block.get('length_floor') or {}
        lines.append(
            f'| {block["level"]} | {_int(block.get("n_folds"))} | {_int(block.get("gallery_size"))} '
            f'| {_int(block.get("n_queries"))} | {_hits(block)} | {_fmt(block.get("top1_p"))} '
            f'| {_floor_cell(block.get("oracle_floor"))} | {_spread(block)} | {_fmt(length.get("rank_percentile"))} '
            f'| {_verdict(block.get("clears_floor"), block.get("missing") or ())} '
            f'| {block.get("postprocess_fit") or "unstated"} |'
        )

    return lines


def _stratified_table(blocks: Sequence[dict[str, Any]]) -> list[str]:
    """The matched-length twin, where a hit cannot be a word-count shortcut, plus the ceiling and the capacity."""
    lines = [
        '| Level | Gallery | Queries | Top-1 hits | Top-1 p | Rank pct ± sd (95% CI) | Clears length floor '
        '| Oracle ceiling Top-1 | Eff. rank ratio | Not measured |',
        '| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |',
    ]
    for block in blocks:
        cell = block.get('length_stratified') or {}
        floor = block.get('oracle_floor') or {}
        missing = block.get('missing') or []
        lines.append(
            f'| {block["level"]} | {_int(cell.get("gallery_size"))} | {_int(cell.get("n_queries"))} '
            f'| {_hits(cell)} | {_fmt(cell.get("top1_p"))} | {_spread(cell)} '
            f'| {_verdict(block.get("clears_length_floor"), missing)} | {_fmt(floor.get("ceiling_top1"))} '
            f'| {_fmt(block.get("effective_rank_ratio"))} | {", ".join(missing) or "--"} |'
        )

    return lines


def _row_block(row: LevelRetrieval, ks: tuple[int, ...], n_boot: int, seed: int) -> dict[str, Any]:
    """Scores one level into the comparable block the table prints, from per-query ranks or from folds."""
    if row.folds is None:
        block = _ranks_block(row, ks, n_boot, seed)
    else:
        block = _folds_block(row, row.folds, n_boot, seed)

    _attach_floors(block, row)

    return block


def _ranks_block(row: LevelRetrieval, ks: tuple[int, ...], n_boot: int, seed: int) -> dict[str, Any]:
    """Scores one level from the per-query ranks a live evaluation still holds."""
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
    block['expected_hits_top1'] = chance * n_queries
    block['rank_percentile'] = float(mean)
    block['rank_percentile_ci'] = [float(lo), float(hi)]
    block['rank_percentile_ci_basis'] = 'bootstrap over queries'
    block['mean_rank'] = float(ranks.mean())
    block['mrr'] = float((1.0 / ranks).mean())

    return block


def _folds_block(row: LevelRetrieval, series: FoldSeries, n_boot: int, seed: int) -> dict[str, Any]:
    """Aggregates one level across its LOSO folds, which is all a run's `metrics.json` still carries."""
    block: dict[str, Any] = {
        'level': row.level,
        'gallery_size': int(row.gallery_size),
        'postprocess_fit': row.postprocess_fit,
        'headline_metric': HEADLINE_METRIC,
        'n_folds': int(np.asarray(series.top1).size),
        'folds': list(series.folds),
        **_series_block(series, n_boot=n_boot, seed=seed),
    }
    block['length_stratified'] = (
        None
        if row.stratified_folds is None
        else {
            'gallery_size': _effective_gallery(row.stratified_folds),
            **_series_block(row.stratified_folds, n_boot=n_boot, seed=seed),
        }
    )
    block['effective_rank_ratio'] = None if row.effective_rank_ratio is None else float(row.effective_rank_ratio)
    block['effective_rank_ratio_sd'] = (
        None if row.effective_rank_ratio_sd is None else float(row.effective_rank_ratio_sd)
    )

    return block


def _series_block(series: FoldSeries, *, n_boot: int, seed: int) -> dict[str, Any]:
    """Pools one gallery's folds: hits out of the queries scored, an exact tail, and the fold-level spread."""
    percentiles = np.asarray(series.rank_percentile, dtype=np.float64)
    rates = np.asarray(series.top1, dtype=np.float64)
    counts = np.asarray(series.n_queries, dtype=np.float64)
    chances = np.asarray(series.chance_top1, dtype=np.float64)

    # Each fold's hit count is an integer, so it is recovered per fold and then summed.
    hits = int(np.rint(rates * counts).sum())
    n_queries = int(counts.sum())

    # Query-weighted: a fold that scored forty queries says less about the chance rate than one that scored sixty.
    chance = float((chances * counts).sum() / counts.sum())
    _, lo, hi = _bootstrap_ci(percentiles, n_boot=n_boot, seed=seed)

    return {
        'n_queries': n_queries,
        'chance_top1': chance,
        'hits_top1': hits,
        'top1': hits / n_queries,
        'expected_hits_top1': chance * n_queries,
        'top1_p': _binom_tail_p(hits, n_queries, chance),
        'top1_p_basis': 'exact binomial over the pooled fold queries',
        'top1_fold_mean': float(rates.mean()),
        'top1_fold_sd': _sample_sd(rates),
        'rank_percentile': float(percentiles.mean()),
        'rank_percentile_sd': _sample_sd(percentiles),
        'rank_percentile_ci': [float(lo), float(hi)],
        'rank_percentile_ci_basis': 'bootstrap over folds',
    }


def _attach_floors(block: dict[str, Any], row: LevelRetrieval) -> None:
    """Puts every measured floor beside the number and reduces them to one `clears_floor`."""
    if row.oracle_floor is not None:
        # The gate is the signature this encoder can actually reach; the ordered profile is reported beside it as
        # the ceiling a variable-K design would have given away, because gating on it would always read `below`.
        floor = row.oracle_floor
        gate = floor.get('gate_top1')
        floor_top1 = float(floor['worst_case_top1'] if gate is None else gate)
        block['oracle_floor'] = {
            'top1': floor_top1,
            'signature': floor.get('gate_signature', floor.get('worst_case_signature')),
            'ceiling_top1': floor.get('ceiling_top1'),
            'ceiling_signature': floor.get('ceiling_signature'),
            'clears': floor.get('clears'),
            'verdict': floor.get('verdict'),
        }
        block['beats_oracle_floor'] = bool(block['top1'] > floor_top1)

    if row.length_floor is not None:
        block['length_floor'] = {
            'rank_percentile': float(row.length_floor['rank_percentile']),
            'top1': row.length_floor.get('top1'),
            'tol': row.length_floor.get('tol'),
            'source': row.length_floor.get('source'),
            'n_folds': row.length_floor.get('n_folds'),
        }

        # The encoder's interval low against the oracle's point estimate, inside the matched-length gallery where
        # a hit cannot be a word-count shortcut -- the same comparison the length audit makes.
        cell = block.get('length_stratified') or block
        interval = cell.get('rank_percentile_ci') or [float('nan'), float('nan')]
        block['clears_length_floor'] = bool(
            np.isfinite(interval[0]) and float(interval[0]) > block['length_floor']['rank_percentile']
        )
        block['length_floor_gallery'] = 'length_stratified' if block.get('length_stratified') else 'full'

    # A level clears the floor only if it clears every floor that was actually measured; none measured reads as None,
    # which the renderer prints as `floor not measured` rather than as a pass.
    measured = [block[key] for key in ('beats_oracle_floor', 'clears_length_floor') if block.get(key) is not None]
    block['clears_floor'] = all(measured) if measured else None
    block['missing'] = list(row.missing)


def _unmeasured_block(entry: UnmeasuredLevel) -> dict[str, Any]:
    """Turns a level that cannot be quoted into a row carrying no number at all."""
    return {
        'level': entry.level,
        'unmeasured': True,
        'n_folds': int(entry.n_folds),
        'missing': list(entry.missing),
        'reason': entry.reason,
        'clears_floor': None,
        'headline_metric': HEADLINE_METRIC,
    }


def _headline_cell(block: dict[str, Any]) -> dict[str, Any]:
    """The cell a cross-level claim is read on -- the matched-length gallery when there is one, the full one if not."""
    return block.get('length_stratified') or block


def _cross_level_verdict(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """States plainly whether any level cleared its floor, and reads the nominal ordering against that."""
    scored = [block for block in blocks if _headline_cell(block).get('rank_percentile') is not None]
    clearing = [block['level'] for block in blocks if block.get('clears_floor') is True]
    unfloored = [block['level'] for block in blocks if block.get('clears_floor') is None]
    best = max(scored, key=lambda block: float(_headline_cell(block)['rank_percentile'])) if scored else None
    galleries = {'length_stratified' if block.get('length_stratified') else 'full' for block in scored}

    verdict: dict[str, Any] = {
        'levels_scored': [block['level'] for block in scored],
        'levels_clearing_floor': clearing,
        'levels_without_a_floor': unfloored,
        'any_clears_floor': bool(clearing),
        # Read on the same gallery the floor comparison uses, so the ordering and the verdict cannot disagree.
        'headline_gallery': galleries.pop() if len(galleries) == 1 else 'mixed',
        'nominal_best': None if best is None else best['level'],
        'nominal_best_metric': HEADLINE_METRIC,
        'nominal_best_value': None if best is None else float(_headline_cell(best)['rank_percentile']),
        'nominal_best_clears_floor': None if best is None else best.get('clears_floor'),
    }
    verdict['reading'] = _reading(verdict)

    return verdict


def _reading(verdict: dict[str, Any]) -> str:
    """Writes the one paragraph that keeps a nominal ordering from being read as a decoding result."""
    parts: list[str] = []
    if unfloored := verdict['levels_without_a_floor']:
        parts.append(
            f'**Floor not measured for {", ".join(unfloored)}.** Those rows carry no verdict: their numbers are '
            'unread, not passing.'
        )

    if not verdict['any_clears_floor'] and verdict['levels_scored']:
        parts.append(
            '**No level clears the brain-free floor printed beside it**, so the ordering across levels is not a '
            'decoding result.'
        )
    elif verdict['levels_clearing_floor']:
        parts.append(f'Clears its floor: {", ".join(verdict["levels_clearing_floor"])}.')

    if (best := verdict['nominal_best']) is not None:
        clears = verdict['nominal_best_clears_floor']
        state = 'unmeasured against' if clears is None else ('above' if clears else 'below')
        gallery = verdict['headline_gallery']
        parts.append(
            f'`{best}` is nominally highest at {HEADLINE_METRIC} {verdict["nominal_best_value"]:.4f} on the '
            f'{gallery} gallery, and it is {state} its floor.'
        )

        if best == 'token' and verdict['nominal_best_clears_floor'] is not True:
            parts.append(
                'A token-level number is the one most exposed to a brain-free channel, because sizing a word by '
                'how many pieces its reference spells it in hands the model the sentence piece profile. Token '
                'coming out highest while it is not above that floor is the confound signature, not a win.'
            )

    return ' '.join(parts)


def _effective_gallery(series: FoldSeries) -> int:
    """The gallery the measured chance rate implies, which stratifying by length shrinks."""
    counts = np.asarray(series.n_queries, dtype=np.float64)
    chance = float((np.asarray(series.chance_top1, dtype=np.float64) * counts).sum() / counts.sum())

    return int(round(1.0 / chance)) if chance > 0.0 else 0


def _sample_sd(values: np.ndarray) -> float | None:
    """The n-1 sample sd, or `None` for a single fold, where a spread of zero would be a fiction."""
    data = [float(v) for v in np.asarray(values, dtype=np.float64).ravel()]
    if len(data) < _SD_MIN_FOLDS:
        return None

    return float(statistics.stdev(data))


def _spread(cell: dict[str, Any] | None) -> str:
    """Formats a rank percentile as its mean, its sample sd across folds, and its bootstrap interval."""
    if not cell or cell.get('rank_percentile') is None:
        return '--'

    interval = cell.get('rank_percentile_ci') or [None, None]
    sd = cell.get('rank_percentile_sd')
    spread = '' if sd is None else f' ± {float(sd):.4f}'

    return f'{float(cell["rank_percentile"]):.4f}{spread} [{_fmt(interval[0])}, {_fmt(interval[1])}]'


def _hits(cell: dict[str, Any] | None) -> str:
    """The Top-1 hit count with the hits chance alone would have given, because a bare count reads as a result."""
    if not cell or cell.get('hits_top1') is None:
        return '--'

    expected = cell.get('expected_hits_top1')

    return f'{int(cell["hits_top1"])}' + ('' if expected is None else f' (exp {float(expected):.1f})')


def _floor_cell(floor: dict[str, Any] | None) -> str:
    """The Top-1 floor with the brain-free signature behind it, so two levels' different floors are not confused."""
    if not floor or floor.get('top1') is None:
        return '--'

    signature = floor.get('signature')

    return f'{float(floor["top1"]):.4f}' + (f' ({signature})' if signature else '')


def _fmt(value: Any) -> str:
    """Formats a table cell, printing a dash rather than `None`."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '--'

    return f'{float(value):.4f}'


def _int(value: Any) -> str:
    """Formats a count cell, printing a dash rather than `None`."""
    return '--' if value is None else f'{int(value)}'


def _verdict(clears: bool | None, missing: Sequence[str] = ()) -> str:
    """Turns a floor comparison into a cell that reads the same way for every level."""
    if clears is None:
        named = ', '.join(missing) if missing else 'no oracle supplied'

        return f'floor not measured ({named})'

    return 'yes' if clears else 'NO -- below a brain-free floor'
