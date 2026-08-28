"""The evidence board: every claim beside the brain-free floor it has to clear, and the verdict that follows."""

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

# A claim reaches the board as whatever its producing audit wrote to JSON, so the board cannot narrow what an
# audit is allowed to record without forcing every audit to change in step.
type Artifact = dict[str, Any]
"""One audit's parsed JSON payload, keyed exactly as that audit wrote it."""


class Verdict(StrEnum):
    """What the evidence says about one claim."""

    CLEARS = 'clears its floor'
    BELOW_FLOOR = 'below a brain-free floor'
    NULL = 'null -- interval covers chance'
    UNDERPOWERED = 'underpowered -- too few queries to resolve'
    NOT_MEASURED = 'not measured'


# A claim with no floor can never be green. This is the whole point of the board: on ZuCo a single integer --
# the word count -- retrieves more sentences than any encoder this project has trained, so a number quoted
# without the floor it was scored against is not evidence, whatever its p-value.
UNFLOORED_VERDICT: Final[Verdict] = Verdict.NOT_MEASURED
"""The verdict any claim gets when no brain-free floor travels with it."""

# 700 stimuli at chance 1/700 means Top-1 expects exactly one hit, so a handful of hits is noise dressed as a
# headline. Below this the board refuses to call a Top-k difference either way.
MIN_RESOLVABLE_HITS: Final[int] = 10
"""Held-out hits below which a Top-k comparison is reported as underpowered rather than as a result."""


@dataclass(slots=True, frozen=True, kw_only=True)
class Claim:
    """One assertion, its measurement, the floor it is read against, and the verdict that follows."""

    key: str
    """Stable identifier, so a claim can be tracked across sessions."""

    question: str
    """The question in one sentence, as a reader would ask it."""

    metric: str
    """Which number answers it -- the exact metric name, not a description."""

    value: float | None = None
    """The measured value, or `None` when the artifact was absent."""

    ci: tuple[float, float] | None = None
    """Bootstrap interval on `value`, low and high."""

    spread: float | None = None
    """Sample standard deviation across folds, where the claim aggregates a sweep."""

    n_folds: int | None = None
    """Folds behind `value`, so a single-fold number cannot read as a sweep."""

    hits: int | None = None
    """Top-k hit count, never a bare rate."""

    n_queries: int | None = None
    """Queries actually scored -- the denominator `hits` belongs to."""

    p_value: float | None = None
    """Exact binomial tail or permutation p, whichever the producing audit reported."""

    floor: float | None = None
    """The brain-free floor on the same metric and the same gallery."""

    floor_name: str | None = None
    """What the floor is -- `length oracle (tol=1)`, `total piece oracle`, `chance`."""

    verdict: Verdict = Verdict.NOT_MEASURED
    """The board's reading, derived rather than asserted."""

    postprocess_fit: str | None = None
    """`none` / `train split` / `transductive` -- a retrieval number is unreadable without it."""

    caveats: tuple[str, ...] = ()
    """Everything a reader must know before quoting this row."""

    sources: tuple[str, ...] = ()
    """Artifact paths the row was assembled from."""

    def headline_safe(self) -> bool:
        """Whether this row may be quoted on its own, without the floor sentence beside it."""
        return self.verdict is Verdict.CLEARS and self.floor is not None


@dataclass(slots=True, frozen=True, kw_only=True)
class EvidenceBoard:
    """Every claim, plus what could not be assembled and why."""

    claims: tuple[Claim, ...]
    """The board, in the order the report renders it."""

    missing: dict[str, str] = field(default_factory=dict)
    """Claim key -> why no artifact was found, so a gap is named rather than dropped."""

    provenance: dict[str, Any] = field(default_factory=dict)
    """Git commit, run names and checkpoint digests behind the rows."""


def _verdict_for(
    value: float | None,
    floor: float | None,
    *,
    ci: tuple[float, float] | None = None,
    chance: float | None = None,
    hits: int | None = None,
) -> Verdict:
    """Derives a verdict from a value, its floor and its interval."""
    if value is None:
        return Verdict.NOT_MEASURED

    if floor is None:
        return UNFLOORED_VERDICT

    # The hit count is only the evidence when no interval accompanies the headline metric. Rank percentile uses
    # every query rather than only the winners, so a thin Top-1 does not make a rank-percentile claim
    # underpowered -- it is carried as a caveat instead.
    if ci is None and hits is not None and hits < MIN_RESOLVABLE_HITS:
        return Verdict.UNDERPOWERED

    # An interval that brackets chance is a null whatever the point estimate does against the floor.
    if chance is not None and ci is not None and ci[0] <= chance <= ci[1]:
        return Verdict.NULL

    # The floor has to be cleared by the interval, not merely by the point estimate: a point estimate above a
    # floor with an interval straddling it is the shape every retracted result in this project had.
    low = ci[0] if ci is not None else value

    return Verdict.CLEARS if low > floor else Verdict.BELOW_FLOOR


def _interval(block: Any) -> tuple[float, float] | None:
    """Reads a `(mean, lo, hi)` or `(lo, hi)` interval out of whatever an audit recorded."""
    if not isinstance(block, Sequence) or isinstance(block, str | bytes):
        return None

    values = [float(v) for v in block if isinstance(v, int | float)]
    if len(values) >= 3:
        return values[1], values[2]

    return (values[0], values[1]) if len(values) == 2 else None


def _count(value: Any) -> int | None:
    """Rounds a count that arrived as a mean over draws, so a query total never renders fractional."""
    return round(float(value)) if isinstance(value, int | float) else None


def _thin_hits_note(hits: int | None, n_queries: int | None) -> tuple[str, ...]:
    """The caveat a hit count too small to resolve a difference always carries, whatever the verdict says."""
    if hits is None or hits >= MIN_RESOLVABLE_HITS:
        return ()

    denominator = f' of {n_queries}' if n_queries else ''

    return (
        f'Top-k rests on {hits} hits{denominator}, below the {MIN_RESOLVABLE_HITS} this board needs to resolve a '
        'difference; only the interval metric above is powered here.',
    )


def _mean_sd(values: Iterable[float | None]) -> tuple[float | None, float | None, int]:
    """Sample mean and n-1 standard deviation -- the population form under-reports a twelve-fold spread."""
    rows = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not rows:
        return None, None, 0

    return statistics.fmean(rows), (statistics.stdev(rows) if len(rows) > 1 else None), len(rows)


def granularity_claims(levels: Artifact | None, *, source: str = '') -> list[Claim]:
    """Builds the sentence/word/token rows from a `zte-levels` payload.

    Args:
        levels (Artifact | None): A parsed `levels.json`, or `None` when the sweep has not run.
        source (str, optional): Path recorded on each row's provenance. Defaults to `''`.

    Returns:
        list[Claim]: One row per alignment level present in the payload.
    """
    if not levels:
        return []

    claims: list[Claim] = []
    for block in levels.get('levels') or ():
        level = block.get('level', 'unknown')

        # The length-stratified twin is the honest cell: the unstratified gallery lets word count answer the
        # query, and that is the confound the whole floor exists to price.
        cell = block.get('length_stratified') or block
        floor_block = block.get('length_floor') or {}
        floor = floor_block.get('rank_percentile')
        ci = _interval(cell.get('rank_percentile_ci'))
        hits, queries = cell.get('hits_top1'), cell.get('n_queries')
        verdict = _verdict_for(cell.get('rank_percentile'), floor, ci=ci)

        caveats = list(block.get('caveats') or ())
        if ceiling := (block.get('oracle_floor') or {}).get('ceiling_top1'):
            caveats.append(f'The ordered per-word piece profile alone retrieves Top-1 {ceiling:.4f} on this gallery.')
        if (own := block.get('clears_floor')) is not None and own is not (verdict is Verdict.CLEARS):
            caveats.append(
                f'`zte-levels` recorded clears_floor={own}; this board reads the interval rather than the point '
                'estimate and therefore says otherwise.'
            )
        if missing := block.get('missing'):
            caveats.append(f'Not measured for this level: {", ".join(str(m) for m in missing)}.')
        caveats.extend(_thin_hits_note(hits, queries))

        claims.append(
            Claim(
                key=f'granularity.{level}',
                question=f"Does the {level}-level encoder retrieve a held-out reader's sentence?",
                metric='held-out rank percentile, length-stratified gallery',
                value=cell.get('rank_percentile'),
                ci=ci,
                spread=cell.get('rank_percentile_sd'),
                n_folds=block.get('n_folds'),
                hits=int(hits) if hits is not None else None,
                n_queries=int(queries) if queries is not None else None,
                p_value=cell.get('top1_p'),
                floor=floor,
                floor_name=(
                    f'length oracle (tol={floor_block.get("tol")})' if floor is not None else 'floor not measured'
                ),
                verdict=verdict,
                postprocess_fit=block.get('postprocess_fit'),
                caveats=tuple(caveats),
                sources=(source,) if source else (),
            )
        )

    return claims


def piece_oracle_claim(audit: Artifact | None, *, source: str = '') -> Claim | None:
    """Builds the resolution-limit row: how much of sentence identity spelling alone gives away.

    Args:
        audit (Artifact | None): A parsed `confound_audit.json` or `rebaseline.json` carrying `piece_oracle`.
        source (str, optional): Path recorded on the row. Defaults to `''`.

    Returns:
        Claim | None: The row, or `None` when no piece oracle was scored.
    """
    piece = (audit or {}).get('piece_oracle')
    if not piece:
        return None

    observed, gate = piece.get('observed_top1'), piece.get('gate_top1')
    ceiling = piece.get('ceiling_top1')
    coverage = piece.get('alignment_coverage')

    caveats = [
        f'The ordered per-word piece profile alone retrieves Top-1 {ceiling:.4f} on this gallery.'
        if ceiling is not None
        else 'No ceiling signature was scored.',
        f'Tokeniser: {piece.get("tokenizer", "unknown")}.',
    ]
    if coverage is not None and coverage <= 0.99:
        caveats.append(f'Alignment coverage {coverage:.4f} is below 0.99, so the piece counts are partly wrong.')

    return Claim(
        key='resolution_limit.piece_oracle',
        question='Can a sub-word readout mean anything, or does spelling alone resolve the gallery?',
        metric='held-out Top-1 against the fixed sub-token count oracle',
        value=observed,
        floor=gate,
        floor_name=f'{piece.get("gate_signature", "piece")} piece oracle',
        verdict=_verdict_for(observed, gate),
        caveats=tuple(caveats),
        sources=(source,) if source else (),
    )


def calibration_claim(calibration: Artifact | None, *, source: str = '') -> Claim | None:
    """Builds the deployment row: what a handful of labelled sentences buys a reader nobody trained on.

    Args:
        calibration (Artifact | None): A parsed `calibration.json`, or `None` when the sweep has not run.
        source (str, optional): Path recorded on the row. Defaults to `''`.

    Returns:
        Claim | None: The row, or `None` when no calibration curve was scored.
    """
    curve = (calibration or {}).get('curve')
    if not curve:
        return None

    points = sorted(curve, key=lambda p: p.get('n_anchors', 0))
    best = max(points, key=lambda p: p.get('rank_percentile') or -math.inf)
    baseline = next((p for p in points if p.get('n_anchors') == 0), points[0])
    shuffled = (calibration or {}).get('shuffled_control') or {}

    lift = None
    if best.get('rank_percentile') is not None and baseline.get('rank_percentile') is not None:
        lift = best['rank_percentile'] - baseline['rank_percentile']

    # The shuffled-anchor arm fits the same map on deliberately wrong pairings, so whatever it lifts is the
    # transform's capacity rather than calibration; the true arm has to beat it, not merely beat zero.
    control = shuffled.get('best_rank_percentile')

    # Query counts are means over anchor draws, so they arrive fractional; a reader wants the count, not the mean.
    before, after = _count(baseline.get('n_queries')), _count(best.get('n_queries'))
    caveats = [
        f'Anchor stimuli are excluded from both the query set and the gallery ({before} -> {after} queries).',
        f'Best at n={best.get("n_anchors")} anchors, family {best.get("family", "unknown")}.',
    ]
    if control is not None:
        caveats.append(f'Shuffled-anchor control reaches {control:.4f} on the same reduced gallery.')
    else:
        caveats.append('No shuffled-anchor control was scored, so the lift is not separable from transform capacity.')

    return Claim(
        key='deployment.anchor_calibration',
        question='How much does a new reader gain from a few labelled sentences, with no retraining?',
        metric='held-out rank percentile after anchor calibration',
        value=best.get('rank_percentile'),
        ci=_interval(best.get('rank_percentile_ci')),
        n_queries=after,
        floor=control if control is not None else baseline.get('rank_percentile'),
        floor_name='shuffled-anchor control' if control is not None else 'uncalibrated, same gallery',
        verdict=_verdict_for(best.get('rank_percentile'), control, ci=_interval(best.get('rank_percentile_ci'))),
        postprocess_fit=(calibration or {}).get('postprocess_fit'),
        caveats=(*caveats, f'Lift over uncalibrated: {lift:+.4f}.' if lift is not None else 'Lift not computable.'),
        sources=(source,) if source else (),
    )


def _transfer_cells(transfer: Artifact | None) -> list[dict[str, Any]]:
    """Flattens either parallax artifact into one list of cells: the aggregated matrix, or a single cell's report."""
    if not transfer:
        return []

    # `PARALLAX.json` nests as {train_task: {eval_task: [one record per seed]}}; the task pair is the nesting key
    # rather than a field, so it has to be put back before the rows can be compared.
    nested = transfer.get('cells')
    if isinstance(nested, dict):
        return [
            {**record, 'train_task': train, 'eval_task': evaluate}
            for train, columns in nested.items()
            for evaluate, records in (columns or {}).items()
            for record in (records or [])
        ]

    if isinstance(nested, list):
        return [c for c in nested if isinstance(c, dict)]

    # A lone `transfer.json` is one cell, with its metrics under `held_out` rather than at the top level.
    if 'train_task' not in transfer:
        return []

    held = transfer.get('held_out') or {}
    top1, queries = held.get('top1'), held.get('n_queries')

    return [
        {
            **transfer,
            'rank_percentile': held.get('rank_percentile'),
            'rank_percentile_ci': held.get('rank_percentile_ci'),
            'top1_p': held.get('top1_p'),
            'top1_hits': round(float(top1) * float(queries)) if top1 is not None and queries else None,
            'n_queries': queries or transfer.get('n_queries'),
        }
    ]


def transfer_claim(transfer: Artifact | None, *, source: str = '') -> Claim | None:
    """Builds the passage-confound row: does a model trained on one task read another task's sentences?

    Args:
        transfer (Artifact | None): A parsed `PARALLAX.json` (the aggregated matrix) or a single cell's
            `transfer.json`, or `None` when neither has run.
        source (str, optional): Path recorded on the row. Defaults to `''`.

    Returns:
        Claim | None: The row, or `None` when no cross-task cell was scored.
    """
    cells = _transfer_cells(transfer)
    novel = [c for c in cells if c.get('train_task') != c.get('eval_task') and c.get('novel_stimuli')]
    if not novel:
        return None

    best = max(novel, key=lambda c: c.get('rank_percentile') or -math.inf)
    diagonal = [c for c in cells if c.get('train_task') == c.get('eval_task')]
    within, _, _ = _mean_sd(c.get('rank_percentile') for c in diagonal)

    return Claim(
        key='confound.passage_memorisation',
        question='Is cross-subject retrieval semantics, or memorised passages?',
        metric='held-out rank percentile, never-seen subject reading never-seen stimuli',
        value=best.get('rank_percentile'),
        ci=_interval(best.get('rank_percentile_ci')),
        hits=best.get('top1_hits'),
        n_queries=best.get('n_queries'),
        p_value=best.get('top1_p'),
        floor=0.5,
        floor_name='chance rank percentile',
        verdict=_verdict_for(best.get('rank_percentile'), 0.5, ci=_interval(best.get('rank_percentile_ci'))),
        postprocess_fit=best.get('postprocess_fit'),
        caveats=(
            f'Best novel cell: train {best.get("train_task")} -> eval {best.get("eval_task")}.',
            f'Within-task diagonal averages {within:.4f}.' if within is not None else 'No diagonal cell to compare.',
            "The tasks' sentence sets are disjoint, so this cell shares no stimulus with training.",
            'Rank percentile resists the length confound; the Top-k on this cell does not.',
            *_thin_hits_note(best.get('top1_hits'), best.get('n_queries')),
        ),
        sources=(source,) if source else (),
    )


def decoder_control_claim(generation: Artifact | None, *, source: str = '') -> Claim | None:
    """Builds the decoder reality-check row: does the readout beat a prefix that carries no brain at all?

    Args:
        generation (Artifact | None): A parsed `generation.json`, or `None` when no decode ran.
        source (str, optional): Path recorded on the row. Defaults to `''`.

    Returns:
        Claim | None: The row, or `None` when the decode carried no controls.
    """
    deltas = (generation or {}).get('deltas') or {}
    if not deltas:
        return None

    verdict_block = (generation or {}).get('verdict') or {}
    scored = {
        name: block for name, block in deltas.items() if isinstance(block, dict) and block.get('delta') is not None
    }
    if not scored:
        return None

    worst = min(scored.items(), key=lambda kv: kv[1].get('delta', math.inf))
    name, block = worst
    ci = _interval(block.get('ci'))

    missing = tuple((generation or {}).get('controls_absent') or ())
    caveats = [
        f'Worst control is `{name}` at delta {block.get("delta"):+.4f}.',
        'A delta whose interval includes zero means the control matched the EEG readout.',
    ]
    if missing:
        caveats.append(f'Controls absent and therefore failing the gate: {", ".join(missing)}.')

    return Claim(
        key='decoder.beats_controls',
        question="Does the generated text carry the EEG, or the language model's prior?",
        metric='paired content-metric delta against the worst pre-registered control',
        value=block.get('delta'),
        ci=ci,
        p_value=verdict_block.get('permutation_p'),
        floor=0.0,
        floor_name='zero -- the control matched',
        verdict=_verdict_for(block.get('delta'), 0.0, ci=ci),
        caveats=tuple(caveats),
        sources=(source,) if source else (),
    )


def evidence_report(
    *,
    levels: Artifact | None = None,
    piece_audit: Artifact | None = None,
    calibration: Artifact | None = None,
    transfer: Artifact | None = None,
    generation: Artifact | None = None,
    sources: dict[str, str] | None = None,
    provenance: dict[str, Any] | None = None,
) -> EvidenceBoard:
    """Assembles the evidence board from whatever audits have actually run.

    Note:
        Nothing here recomputes a number. Every row is read from an artifact its own audit wrote, so the board
        cannot disagree with the run it describes. A claim whose artifact is missing is named in `missing`
        rather than dropped, because a silently absent row reads as a claim nobody made.

    Args:
        levels (Artifact | None, optional): Parsed `levels.json`. Defaults to `None`.
        piece_audit (Artifact | None, optional): Parsed payload carrying `piece_oracle`. Defaults to `None`.
        calibration (Artifact | None, optional): Parsed `calibration.json`. Defaults to `None`.
        transfer (Artifact | None, optional): Parsed parallax `transfer.json`. Defaults to `None`.
        generation (Artifact | None, optional): Parsed `generation.json`. Defaults to `None`.
        sources (dict[str, str] | None, optional): Claim family -> artifact path, recorded on each row.
        provenance (dict[str, Any] | None, optional): Git commit, run names, checkpoint digests.

    Returns:
        EvidenceBoard: The claims that could be assembled, and the reason for every one that could not.
    """
    where = sources or {}
    claims: list[Claim] = []
    missing: dict[str, str] = {}

    claims.extend(granularity_claims(levels, source=where.get('levels', '')))
    if not levels:
        missing['granularity'] = 'no levels.json -- run zte-levels over the three alignment arms'

    for builder, payload, key, hint in (
        (
            piece_oracle_claim,
            piece_audit,
            'resolution_limit',
            'run zte-audit --piece-oracle or zte-rebaseline --piece-oracle',
        ),
        (calibration_claim, calibration, 'deployment', 'run zte-calibrate against a held-out checkpoint'),
        (transfer_claim, transfer, 'confound', 'run zte-parallax transfer for the cross-task cells'),
        (decoder_control_claim, generation, 'decoder', 'run zte-decode with the pre-registered controls'),
    ):
        claim = builder(payload, source=where.get(key, ''))
        if claim is None:
            missing[key] = f'no artifact -- {hint}'
            continue

        claims.append(claim)

    return EvidenceBoard(claims=tuple(claims), missing=missing, provenance=provenance or {})


def _cell(value: float | None, digits: int = 4) -> str:
    """Formats a number for the board, or a dash when it was never measured."""
    return f'{value:.{digits}f}' if isinstance(value, int | float) else '--'


def _hits_cell(claim: Claim) -> str:
    """Renders Top-k as a hit count over the queries actually scored, never as a bare rate."""
    if claim.hits is None or claim.n_queries is None:
        return '--'

    tail = f' (p={claim.p_value:.3g})' if claim.p_value is not None else ''

    return f'{claim.hits}/{claim.n_queries}{tail}'


def render_markdown(board: EvidenceBoard, title: str = 'ZTE Evidence Board') -> str:
    """Renders an `EvidenceBoard` as a self-contained Markdown document.

    Args:
        board (EvidenceBoard): The assembled board.
        title (str, optional): Document heading. Defaults to `'ZTE Evidence Board'`.

    Returns:
        str: The board as Markdown.
    """
    lines: list[str] = [f'# {title}', '']
    lines += [
        'Every row carries the brain-free floor it was scored against. A number with no floor is reported as',
        '`not measured`, never as a result: on this corpus a single integer -- the word count -- retrieves more',
        'sentences than any encoder this project has trained, so an unfloored number is not evidence whatever',
        'its p-value.',
        '',
    ]

    header = '| claim | metric | value | 95% CI | floor | verdict |\n| --- | --- | --- | --- | --- | --- |\n'
    rows = ''.join(
        f'| {c.question} | `{c.metric}` | {_cell(c.value)}'
        f'{f" ± {c.spread:.4f}" if c.spread is not None else ""} '
        f'| {f"[{c.ci[0]:.4f}, {c.ci[1]:.4f}]" if c.ci else "--"} '
        f'| {_cell(c.floor)} ({c.floor_name or "none"}) | **{c.verdict.value}** |\n'
        for c in board.claims
    )
    lines += ['## The board', '', header + rows]

    lines += ['## Hit counts and post-processing', '']
    lines += [
        '| claim | Top-k hits | folds | postprocess_fit |\n| --- | --- | --- | --- |',
        *(
            f'| {c.key} | {_hits_cell(c)} | {c.n_folds if c.n_folds else "--"} | {c.postprocess_fit or "unrecorded"} |'
            for c in board.claims
        ),
        '',
    ]

    lines += ['## What each row does not say', '']
    for claim in board.claims:
        if not claim.caveats:
            continue

        lines += [f'**{claim.key}**', '', *(f'- {c}' for c in claim.caveats), '']

    if board.missing:
        lines += ['## Not measured', '']
        lines += [f'- `{key}` -- {why}' for key, why in sorted(board.missing.items())]
        lines += ['']

    headline = [c.key for c in board.claims if c.headline_safe()]
    lines += ['## What may be quoted on its own', '']
    lines += (
        [f'- `{key}`' for key in headline]
        if headline
        else ['Nothing on this board clears its floor, so no row may be quoted without the floor sentence beside it.']
    )
    lines += ['']

    if board.provenance:
        lines += ['## Provenance', '', *(f'- `{k}`: {v}' for k, v in sorted(board.provenance.items())), '']

    return '\n'.join(lines)


def board_to_dict(board: EvidenceBoard) -> dict[str, Any]:
    """Serialises an `EvidenceBoard` to the JSON payload the notebook renders.

    Args:
        board (EvidenceBoard): The assembled board.

    Returns:
        dict[str, Any]: A flat, JSON-safe payload with one entry per claim.
    """
    return {
        'claims': [
            {
                'key': c.key,
                'question': c.question,
                'metric': c.metric,
                'value': c.value,
                'ci': list(c.ci) if c.ci else None,
                'spread': c.spread,
                'n_folds': c.n_folds,
                'hits': c.hits,
                'n_queries': c.n_queries,
                'p_value': c.p_value,
                'floor': c.floor,
                'floor_name': c.floor_name,
                'verdict': c.verdict.value,
                'postprocess_fit': c.postprocess_fit,
                'caveats': list(c.caveats),
                'sources': list(c.sources),
                'headline_safe': c.headline_safe(),
            }
            for c in board.claims
        ],
        'missing': dict(board.missing),
        'provenance': dict(board.provenance),
        'n_headline_safe': sum(1 for c in board.claims if c.headline_safe()),
    }
