"""`zte-levels` -- the granularity ablation: sentence vs word vs token, aggregated over folds, against their floors."""

import argparse
import hashlib
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from zte.alignment.atlas import LEVELS, Level
from zte.alignment.compare import FoldSeries, LevelRetrieval, UnmeasuredLevel, cross_level_table, render_markdown
from zte.cli.support.done import add_force_argument, is_done, mark_done, signature
from zte.cli.support.io import read_json, write_json
from zte.evaluation.analysis.collect import dig
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.levels')

# The artifacts a level row is assembled from. Nothing here is recomputed: no model is loaded, no query re-scored.
_METRICS: Final[str] = 'evaluation/metrics.json'
"""Where a run records its scoreboard, relative to the run directory."""

_AUDIT: Final[str] = 'rebaseline/rebaseline.json'
"""Where `zte-rebaseline` records the brain-free floors, relative to the run directory."""

_CONFIG: Final[str] = 'config.yaml'
"""The resolved config mirrored beside a run, which is authoritative for the level."""

# Which lever a run turned on says which rung it aligned at, and the config outranks the directory name because a
# run directory can be renamed while `run_name` cannot.
_LEVEL_LEVERS: Final[tuple[tuple[tuple[str, ...], Level], ...]] = (
    (('objective', 'token_weight'), 'token'),
    (('objective', 'lexical_weight'), 'word'),
)
"""Config paths whose non-zero weight names the alignment level; neither on means the sentence level."""

# `align_token_combined_loZAB_s42` -- the fallback for a run whose config was not mirrored beside it.
_LEVEL_IN_NAME: Final[re.Pattern[str]] = re.compile(r'(?:^|[_-])(token|word|sentence)(?:[_-]|$)')
"""Recovers the level from a run directory name when the config is missing."""


@dataclass(slots=True, frozen=True, kw_only=True)
class RetrievalCell:
    """One gallery's held-out numbers for one fold, exactly as `metrics.json` records them."""

    rank_percentile: float
    """Mean rank percentile over that fold's queries."""

    top1: float
    """Top-1 rate over that fold's queries."""

    n_queries: int
    """Queries the fold actually scored -- what its hit count is out of."""

    chance_top1: float
    """The fold's own chance rate, which stratifying the gallery changes."""


@dataclass(slots=True, frozen=True, kw_only=True)
class FoldRecord:
    """One evaluated run read off disk: which rung it aligned at, whose brain was held out, and its numbers."""

    run: str
    """The run directory's name."""

    path: Path
    """Where it was read from."""

    level: Level
    """The rung this run aligned at."""

    fold: str
    """The held-out subject whose readings were the queries."""

    full: RetrievalCell | None
    """The unstratified gallery, or `None` when the run carries no held-out block."""

    stratified: RetrievalCell | None
    """The matched-length gallery, where a hit cannot be a sentence-length shortcut."""

    postprocess_fit: str | None
    """`none`, `train split` or `transductive`, as the scoreboard stamped it."""

    effective_rank_ratio: float | None
    """How much of the embedding space the fold actually spans."""

    length_floor: dict[str, Any] | None
    """The word-count oracle this fold has to clear, from its `rebaseline.json`."""

    piece_floor: dict[str, Any] | None
    """The sub-word piece oracle, from `zte-rebaseline --piece-oracle`."""

    missing: tuple[str, ...]
    """Artifacts or blocks this fold did not carry, named as they would appear on disk."""


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Defines and parses the `zte-levels` command-line arguments.

    Args:
        argv (list[str] | None, optional): Arguments to parse instead of `sys.argv`. Defaults to None.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='The granularity ablation: sentence, word and token alignment on one comparable footing. '
        'Reads already-evaluated runs -- no model is loaded and no query re-scored -- aggregates them across LOSO '
        'folds, and prints every level against the brain-free floor it has to clear.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--runs',
        type=Path,
        nargs='+',
        default=None,
        help='Run directories, each carrying evaluation/metrics.json and optionally rebaseline/rebaseline.json.',
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=None,
        help='A tree to search instead of naming every run; combined with --pattern.',
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='**',
        help='Glob applied under --root to find run directories; `*` for a flat tree of runs.',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=None,
        help='Output directory for levels.json and levels.md. Default: a `levels/` folder beside the runs.',
    )
    parser.add_argument(
        '--n-boot',
        type=int,
        default=2000,
        dest='n_boot',
        help='Bootstrap resamples behind each rank-percentile interval, resampled over folds.',
    )
    parser.add_argument('--seed', type=int, default=0, help='Bootstrap seed.')
    add_force_argument(parser)
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    return parser.parse_args(argv)


def discover_runs(runs: Sequence[Path] | None, root: Path | None, pattern: str) -> list[Path]:
    """Resolves the run directories to read, from an explicit list, a tree, or both.

    Args:
        runs (Sequence[Path] | None): Directories named outright.
        root (Path | None): A tree to search under.
        pattern (str): Glob applied under `root`.

    Returns:
        list[Path]: Every directory carrying `evaluation/metrics.json`, deduplicated and sorted.

    Raises:
        SystemExit: If neither `runs` nor `root` was given.
    """
    if not runs and root is None:
        raise SystemExit('Nothing to read: pass --runs with run directories, or --root with a tree to search.')

    found: set[Path] = set()
    for candidate in runs or ():
        path = Path(candidate)
        if (path / _METRICS).is_file():
            found.add(path.resolve())
        else:
            _LOG.warning('Skipping %s: it carries no %s.', path, _METRICS)

    if root is not None:
        for candidate in sorted(Path(root).glob(pattern)):
            if candidate.is_dir() and (candidate / _METRICS).is_file():
                found.add(candidate.resolve())

    return sorted(found)


def default_out_dir(run_dirs: Sequence[Path], root: Path | None) -> Path:
    """Where the table lands when `--out` is not given: a `levels/` folder beside the tree it was read from."""
    if root is not None:
        return Path(root) / 'levels'

    parents = sorted({run_dir.resolve().parent for run_dir in run_dirs})

    return (parents[0] if parents else Path('res/experiments')) / 'levels'


def level_of(config: dict[str, Any], run_name: str) -> Level | None:
    """Which rung a run aligned at: the config's own levers, falling back to the run directory's name.

    Args:
        config (dict[str, Any]): The run's resolved `config.yaml`, or an empty dict when it is not mirrored.
        run_name (str): The run directory's name.

    Returns:
        Level | None: The level, or `None` when neither the config nor the name says.
    """
    if config:
        for path, level in _LEVEL_LEVERS:
            if (weight := _float(dig(config, *path))) is not None and weight > 0.0:
                return level

        return 'sentence'

    if match := _LEVEL_IN_NAME.search(run_name):
        named = match.group(1)

        return next((level for level in LEVELS if level == named), None)

    return None


def read_fold(run_dir: Path) -> FoldRecord | None:
    """Reads one already-evaluated run into the row material a level needs, recomputing nothing.

    Args:
        run_dir (Path): A run directory carrying `evaluation/metrics.json`.

    Returns:
        FoldRecord | None: The record, or `None` when the run is unreadable or names no level.
    """
    metrics = _read_json(run_dir / _METRICS)
    if not isinstance(metrics, dict):
        _LOG.warning('Skipping %s: its %s is not an object.', run_dir, _METRICS)

        return None

    config = _read_yaml(run_dir / _CONFIG)
    level = level_of(config, run_dir.name)
    if level is None:
        _LOG.warning('Skipping %s: neither its config nor its name says which alignment level it is.', run_dir)

        return None

    missing: list[str] = []
    if not config:
        missing.append(_CONFIG)

    board = dig(metrics, 'scoreboard', 'held_out_retrieval') or {}
    full = _cell(board)
    if full is None:
        missing.append('scoreboard.held_out_retrieval')

    stratified = _cell(board.get('length_stratified') or dig(metrics, 'scoreboard', 'held_out_retrieval_stratified'))
    if stratified is None:
        missing.append('scoreboard.held_out_retrieval.length_stratified')

    audit_path = run_dir / _AUDIT
    audit = _read_json(audit_path)
    if not isinstance(audit, dict):
        audit = None
        missing.append(_AUDIT)

    length_floor = _length_floor(audit)
    if audit is not None and length_floor is None:
        missing.append(f'{_AUDIT}:length_oracle')

    # The sub-word oracle is only a floor where the design hands the model a piece profile, so its absence is
    # noted at the token level alone rather than reported against every sentence-level run.
    piece_floor = audit.get('piece_oracle') if audit is not None else None
    if not isinstance(piece_floor, dict):
        piece_floor = None
        if audit is not None and level == 'token':
            missing.append(f'{_AUDIT}:piece_oracle')

    fold = (
        dig(metrics, 'honesty', 'loso_holdout')
        or dig(metrics, 'scoreboard', 'holdout_subject')
        or dig(config, 'train', 'loso_holdout_subject')
        or run_dir.name
    )

    return FoldRecord(
        run=run_dir.name,
        path=run_dir,
        level=level,
        fold=str(fold),
        full=full,
        stratified=stratified,
        postprocess_fit=_text(board.get('postprocess_fit')),
        effective_rank_ratio=_float(dig(metrics, 'embedding_health', 'effective_rank_ratio')),
        length_floor=length_floor,
        piece_floor=piece_floor,
        missing=tuple(missing),
    )


def build_rows(records: Sequence[FoldRecord]) -> tuple[list[LevelRetrieval], list[UnmeasuredLevel]]:
    """Groups records by level and aggregates each level across its folds.

    Note:
        A level that cannot honestly be quoted is returned as an `UnmeasuredLevel` rather than dropped -- silently
        omitting a level is how a granularity table comes to read as if every rung had been measured.

    Args:
        records (Sequence[FoldRecord]): Every fold read off disk, across every level.

    Returns:
        tuple[list[LevelRetrieval], list[UnmeasuredLevel]]: The scored rows, and the levels that carry no number.
    """
    rows: list[LevelRetrieval] = []
    unmeasured: list[UnmeasuredLevel] = []

    for level in LEVELS:
        found = [record for record in records if record.level == level]
        if not found:
            continue

        folds = _one_per_fold(found)
        usable = [record for record in folds if record.full is not None]
        if not usable:
            unmeasured.append(
                UnmeasuredLevel(
                    level=level,
                    missing=('scoreboard.held_out_retrieval',),
                    n_folds=len(folds),
                    reason=f'{len(folds)} {level}-level run(s) were found, none of which carries a held-out block.',
                )
            )
            continue

        row = _level_row(level, usable)
        if isinstance(row, UnmeasuredLevel):
            unmeasured.append(row)
        else:
            rows.append(row)

    return rows, unmeasured


def build_table(run_dirs: Sequence[Path], *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Reads every run, aggregates it by level and returns the cross-level payload.

    Args:
        run_dirs (Sequence[Path]): Run directories, each carrying `evaluation/metrics.json`.
        n_boot (int, optional): Bootstrap resamples behind each interval. Defaults to 2000.
        seed (int, optional): Bootstrap seed. Defaults to 0.

    Returns:
        dict[str, Any]: The `cross_level_table` payload with a `runs` list and `provenance` beside it.

    Raises:
        SystemExit: If no run named a level the table can group by.
    """
    from zte.utils.provenance import git_info

    records = [record for run_dir in run_dirs if (record := read_fold(run_dir)) is not None]
    if not records:
        raise SystemExit('No run named an alignment level; nothing to compare.')

    rows, unmeasured = build_rows(records)
    table = cross_level_table(rows, unmeasured=unmeasured, n_boot=n_boot, seed=seed)

    git = git_info()
    table['runs'] = [
        {
            'run': record.run,
            'path': str(record.path),
            'level': record.level,
            'fold': record.fold,
            'missing': list(record.missing),
        }
        for record in records
    ]
    table['provenance'] = {
        'tool': 'zte-levels',
        'n_runs': len(records),
        'n_boot': int(n_boot),
        'seed': int(seed),
        'git_commit': git['commit'],
        'git_dirty': git['dirty'],
        'recomputed': 'nothing -- every number is read from an artifact already on disk',
    }

    return table


def write_table(table: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Writes `levels.json` and `levels.md` into `out_dir` and returns both paths.

    Args:
        table (dict[str, Any]): A `build_table` payload.
        out_dir (Path): Destination directory, created if absent.

    Returns:
        tuple[Path, Path]: The written JSON and Markdown paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(out_dir / 'levels.json', table, default=str)
    markdown = ['# Alignment levels — the granularity ablation', '', render_markdown(table), *_runs_section(table)]
    md_path = out_dir / 'levels.md'
    md_path.write_text('\n'.join(markdown).rstrip() + '\n', encoding='utf-8')

    return json_path, md_path


def _level_row(level: Level, usable: Sequence[FoldRecord]) -> LevelRetrieval | UnmeasuredLevel:
    """Turns one level's folds into a single row, or into the reason it cannot be quoted."""
    missing: list[str] = []
    full = _series(usable, stratified=False)

    # Both galleries must describe the same folds to be comparable, so a partial stratified twin is dropped whole.
    with_stratified = [record for record in usable if record.stratified is not None]
    stratified = _series(with_stratified, stratified=True) if len(with_stratified) == len(usable) else None
    if stratified is None:
        absent = sorted({record.fold for record in usable} - {record.fold for record in with_stratified})
        missing.append(f'scoreboard.held_out_retrieval.length_stratified ({", ".join(absent) or "every fold"})')

    length_floor = _mean_length_floor(usable)
    if absent := _absent_folds(usable, 'length_floor'):
        missing.append(f'{_AUDIT}:length_oracle ({", ".join(absent)})')

    piece_floor = _mean_piece_floor(usable)
    if level == 'token' and (absent := _absent_folds(usable, 'piece_floor')):
        missing.append(f'{_AUDIT}:piece_oracle ({", ".join(absent)})')

    # The refusal `LevelRetrieval` already enforces, reported as a row instead of as an exception: a token number
    # quoted without its sub-word floor is not evidence of decoding, so the number is withheld, not the row.
    if level == 'token' and piece_floor is None:
        return UnmeasuredLevel(
            level=level,
            missing=(f'{_AUDIT}:piece_oracle',),
            n_folds=len(usable),
            reason=f'{len(usable)} token-level fold(s) were scored, but none carries the sub-word piece oracle a '
            'token number has to clear. Re-run `zte-rebaseline --piece-oracle` against those checkpoints.',
        )

    if piece_floor is None:
        piece_floor = _length_as_top1_floor(length_floor)

    ratios = [record.effective_rank_ratio for record in usable if record.effective_rank_ratio is not None]
    if not ratios:
        missing.append('embedding_health.effective_rank_ratio')

    return LevelRetrieval(
        level=level,
        gallery_size=_effective_gallery(usable, stratified=False),
        postprocess_fit=_consensus(record.postprocess_fit for record in usable),
        folds=full,
        stratified_folds=stratified,
        oracle_floor=piece_floor,
        length_floor=length_floor,
        effective_rank_ratio=statistics.mean(ratios) if ratios else None,
        effective_rank_ratio_sd=_sd(ratios),
        missing=tuple(missing),
    )


def _absent_folds(records: Sequence[FoldRecord], attribute: str) -> list[str]:
    """The folds that did not carry a floor, so a level-wide mean never hides a partial measurement."""
    return sorted(record.fold for record in records if getattr(record, attribute) is None)


def _series(records: Sequence[FoldRecord], *, stratified: bool) -> FoldSeries:
    """Stacks one gallery's per-fold cells into the series `cross_level_table` aggregates."""
    cells = [record.stratified if stratified else record.full for record in records]

    return FoldSeries(
        rank_percentile=np.array([cell.rank_percentile for cell in cells if cell is not None], dtype=np.float64),
        top1=np.array([cell.top1 for cell in cells if cell is not None], dtype=np.float64),
        n_queries=np.array([cell.n_queries for cell in cells if cell is not None], dtype=np.int64),
        chance_top1=np.array([cell.chance_top1 for cell in cells if cell is not None], dtype=np.float64),
        folds=tuple(record.fold for record in records),
    )


def _one_per_fold(records: Sequence[FoldRecord]) -> list[FoldRecord]:
    """Keeps one run per held-out subject, so a re-run does not enter the mean twice."""
    kept: dict[str, FoldRecord] = {}
    for record in sorted(records, key=lambda r: r.run):
        if (existing := kept.get(record.fold)) is not None:
            _LOG.warning(
                'Fold %s appears twice at the %s level (%s and %s); keeping %s.',
                record.fold,
                record.level,
                existing.run,
                record.run,
                existing.run,
            )
            continue

        kept[record.fold] = record

    return [kept[fold] for fold in sorted(kept)]


def _cell(block: Any) -> RetrievalCell | None:
    """One gallery's numbers out of a scoreboard block, or `None` when any of the four is absent."""
    if not isinstance(block, dict):
        return None

    percentile, top1, chance, queries = (
        _float(block.get(key)) for key in ('rank_percentile', 'top1', 'chance_top1', 'n_queries')
    )
    if percentile is None or top1 is None or chance is None or queries is None or queries < 1.0:
        return None

    # The three rates come out of a mean over queries, so only float error can put them a hair outside [0, 1].
    return RetrievalCell(
        rank_percentile=_clamp(percentile),
        top1=_clamp(top1),
        n_queries=int(queries),
        chance_top1=_clamp(chance),
    )


def _clamp(value: float) -> float:
    """Pins a rate into [0, 1], where float error from a mean can leave it a hair outside."""
    return min(max(value, 0.0), 1.0)


def _length_floor(audit: dict[str, Any] | None) -> dict[str, Any] | None:
    """The word-count oracle at the tolerance the audit's own floor comparison used."""
    if audit is None:
        return None

    tol = (audit.get('floor_comparison') or {}).get('oracle_tol')
    oracle = (audit.get('length_oracle') or {}).get(str(tol))
    if not isinstance(oracle, dict) or (rank := _float(oracle.get('rank_percentile'))) is None:
        return None

    return {
        'rank_percentile': rank,
        'top1': _float(oracle.get('top1')),
        'tol': None if tol is None else int(tol),
        'source': f'{_AUDIT}:length_oracle',
    }


def _mean_length_floor(records: Sequence[FoldRecord]) -> dict[str, Any] | None:
    """The length oracle averaged over the folds that measured it -- one gallery, so the folds should agree."""
    floors = [record.length_floor for record in records if record.length_floor is not None]
    if not floors:
        return None

    tolerances = {floor['tol'] for floor in floors}
    top1 = [floor['top1'] for floor in floors if floor['top1'] is not None]

    return {
        'rank_percentile': statistics.mean(floor['rank_percentile'] for floor in floors),
        'top1': statistics.mean(top1) if top1 else None,
        'tol': tolerances.pop() if len(tolerances) == 1 else None,
        'source': f'{_AUDIT}:length_oracle over {len(floors)} fold(s)',
        'n_folds': len(floors),
    }


def _mean_piece_floor(records: Sequence[FoldRecord]) -> dict[str, Any] | None:
    """The sub-word piece oracle averaged over the folds that measured it, gate and ceiling kept apart."""
    floors = [record.piece_floor for record in records if record.piece_floor is not None]
    gates = [gate for floor in floors if (gate := _float(floor.get('gate_top1'))) is not None]
    if not gates:
        return None

    ceilings = [top1 for floor in floors if (top1 := _float(floor.get('ceiling_top1'))) is not None]
    signatures = {str(floor.get('gate_signature')) for floor in floors}

    return {
        'gate_top1': statistics.mean(gates),
        'gate_signature': signatures.pop() if len(signatures) == 1 else 'mixed',
        'ceiling_top1': statistics.mean(ceilings) if ceilings else None,
        'ceiling_signature': 'profile',
        'source': f'{_AUDIT}:piece_oracle over {len(gates)} fold(s)',
    }


def _length_as_top1_floor(length_floor: dict[str, Any] | None) -> dict[str, Any] | None:
    """The word-count oracle read as a Top-1 floor, which is what a sentence- or word-level row is measured on."""
    if length_floor is None or length_floor.get('top1') is None:
        return None

    return {
        'gate_top1': float(length_floor['top1']),
        'gate_signature': _length_signature(length_floor.get('tol')),
        'ceiling_top1': None,
        'ceiling_signature': None,
        'source': length_floor.get('source'),
    }


def _length_signature(tol: Any) -> str:
    """Names the brain-free channel a word-count floor comes from, tolerance included when the folds agree on one."""
    return 'word count' if tol is None else f'word count ±{int(tol)}'


def _effective_gallery(records: Sequence[FoldRecord], *, stratified: bool) -> int:
    """The gallery the measured chance rate implies, query-weighted across folds."""
    cells = [record.stratified if stratified else record.full for record in records]
    weights = np.array([cell.n_queries for cell in cells if cell is not None], dtype=np.float64)
    chances = np.array([cell.chance_top1 for cell in cells if cell is not None], dtype=np.float64)
    chance = float((chances * weights).sum() / weights.sum())

    return max(int(round(1.0 / chance)), 1) if chance > 0.0 else 1


def _consensus(values: Any) -> str:
    """One post-processing label for a level, saying so plainly when its folds do not agree."""
    named = sorted({str(value) for value in values if value})
    if not named:
        return 'unstated'

    return named[0] if len(named) == 1 else f'mixed ({", ".join(named)})'


def _sd(values: Sequence[float]) -> float | None:
    """The n-1 sample sd, or `None` below two folds, where a spread of zero would be a fiction."""
    if len(values) < 2:
        return None

    return float(statistics.stdev(values))


def _runs_section(table: dict[str, Any]) -> list[str]:
    """The provenance footer: which run supplied which fold, and what each one did not carry."""
    lines = ['', '### Runs read', '', '| Level | Fold | Run | Not carried |', '| --- | --- | --- | --- |']
    for entry in table.get('runs') or []:
        absent = ', '.join(entry.get('missing') or []) or '--'
        lines.append(f'| {entry["level"]} | {entry["fold"]} | `{entry["run"]}` | {absent} |')

    provenance = table.get('provenance') or {}
    lines += ['', f'Recomputed: {provenance.get("recomputed", "nothing")}. Commit `{provenance.get("git_commit")}`.']

    return lines


def _inputs_digest(run_dirs: Sequence[Path]) -> str:
    """A fingerprint of what the table is built from -- names and sizes, never mtimes, which mirroring resets."""
    parts: list[str] = []
    for run_dir in sorted(run_dirs):
        for name in (_METRICS, _CONFIG, _AUDIT):
            path = run_dir / name
            parts.append(f'{run_dir.name}/{name}:{path.stat().st_size if path.is_file() else -1}')

    return hashlib.sha256('\n'.join(parts).encode('utf-8')).hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    """Reads a JSON artifact, returning `None` when it is missing or unreadable rather than failing the whole table."""
    if not path.is_file():
        return None

    try:
        loaded = read_json(path)
    except (OSError, ValueError) as exc:
        _LOG.warning('Could not read %s (%r).', path, exc)

        return None

    return loaded if isinstance(loaded, dict) else None


def _read_yaml(path: Path) -> dict[str, Any]:
    """Reads a YAML file, returning an empty dict when it is missing or unreadable."""
    if not path.is_file():
        return {}

    import yaml

    try:
        loaded = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as exc:
        _LOG.warning('Could not read %s (%r).', path, exc)

        return {}

    return loaded if isinstance(loaded, dict) else {}


def _float(value: Any) -> float | None:
    """Coerces a metric to a finite float, tolerating `None`, strings and non-finite values."""
    if value is None or isinstance(value, bool):
        return None

    try:
        out = float(value)
    except TypeError, ValueError:
        return None

    return out if np.isfinite(out) else None


def _text(value: Any) -> str | None:
    """A metadata string, or `None` when the field is absent."""
    return None if value is None else str(value)


def main() -> None:
    """Builds the cross-level comparison table from runs already on disk, retraining and re-scoring nothing."""
    args = parse_arguments()
    configure_logging(args.log_level)

    run_dirs = discover_runs(args.runs, args.root, args.pattern)
    if not run_dirs:
        raise SystemExit('No run directory carrying evaluation/metrics.json was found.')

    out_dir = Path(args.out) if args.out else default_out_dir(run_dirs, args.root)
    artifacts = (out_dir / 'levels.json', out_dir / 'levels.md')
    sig = signature(
        args,
        tool='levels',
        extra={'inputs_sha256': _inputs_digest(run_dirs), 'runs': [str(run_dir) for run_dir in run_dirs]},
        ignore=('runs', 'root', 'pattern'),
    )

    if is_done(artifacts, sig, force=args.force):
        table = read_json(artifacts[0])
    else:
        table = build_table(run_dirs, n_boot=args.n_boot, seed=args.seed)
        write_table(table, out_dir)
        mark_done(artifacts, sig)

    _log_table(table, out_dir)


def _log_table(table: dict[str, Any], out_dir: Path) -> None:
    """Logs one line per level and the reading that keeps a nominal ordering from being read as a result."""
    for block in table.get('levels') or []:
        if block.get('unmeasured'):
            _LOG.warning(
                '%s: not measured -- missing %s. %s',
                block['level'],
                ', '.join(block.get('missing') or []),
                block.get('reason', ''),
            )
            continue

        _LOG.info(
            '%s: %s fold(s), rank percentile %.4f ± %s, Top-1 %s/%s (p=%.4g), clears floor: %s [%s].',
            block['level'],
            block.get('n_folds'),
            block.get('rank_percentile', float('nan')),
            'unmeasured' if block.get('rank_percentile_sd') is None else f'{block["rank_percentile_sd"]:.4f}',
            block.get('hits_top1'),
            block.get('n_queries'),
            block.get('top1_p', float('nan')),
            block.get('clears_floor'),
            block.get('postprocess_fit'),
        )

    if reading := (table.get('verdict') or {}).get('reading'):
        _LOG.info('%s', reading)

    _LOG.info('Cross-level table written to %s (levels.json + levels.md).', out_dir)


if __name__ == '__main__':
    main()
