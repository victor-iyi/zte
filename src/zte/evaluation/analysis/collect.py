"""Walks a study tree and turns every run's artifacts into tidy frames the analysis reads."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from zte.logging_utils import get_logger

_LOG = get_logger('evaluation.analysis.collect')

# `exp12_zte_raw_aligned_loZAB_s42` -> arm `exp12_zte_raw_aligned`, holdout `ZAB`, seed 42. The config is
# authoritative for both; this only recovers them for a run whose config was not mirrored alongside it.
_RUN_NAME = re.compile(r'^(?P<arm>.+?)(?:_lo(?P<holdout>[A-Z]{2,4}))?(?:_s(?P<seed>\d+))?$')

# What every headline table reads, and where it lives inside `evaluation/metrics.json`. The pooled and word-level
# keys travel beside the held-out ones because a reader chasing a suspicious number wants both, but only
# `held_out_*` is the result: `pooled_top1` inverted the champion once already.
HEADLINES: Final[dict[str, tuple[str, ...]]] = {
    'held_out_top1': ('scoreboard', 'held_out_retrieval', 'top1'),
    'held_out_top5': ('scoreboard', 'held_out_retrieval', 'top5'),
    'held_out_rank_percentile': ('scoreboard', 'held_out_retrieval', 'rank_percentile'),
    'held_out_chance': ('scoreboard', 'held_out_retrieval', 'chance_top1'),
    'held_out_n_queries': ('scoreboard', 'held_out_retrieval', 'n_queries'),
    'stratified_top1': ('scoreboard', 'held_out_retrieval_length_stratified', 'top1'),
    'stratified_rank_percentile': ('scoreboard', 'held_out_retrieval_length_stratified', 'rank_percentile'),
    'pooled_top1': ('sentence_retrieval', 'top1'),
    'word_top1': ('word_retrieval', 'top1'),
    'word_chance': ('word_retrieval', 'chance_top1'),
    'effective_rank_ratio': ('embedding_health', 'effective_rank_ratio'),
    'anisotropy': ('embedding_health', 'anisotropy'),
    'uniformity': ('embedding_health', 'uniformity'),
    'who_vs_what': ('neurons', 'who_vs_what_ratio'),
    'content_probe_r2': ('scoreboard', 'lift_over_raw', 'content_probe', 'raw_content_r2_best'),
    'content_probe_passes': ('scoreboard', 'lift_over_raw', 'content_probe', 'passes'),
    'probe_machinery_r2': ('scoreboard', 'lift_over_raw', 'content_probe', 'machinery', 'word_len_r2'),
    'probe_machinery_passes': ('scoreboard', 'lift_over_raw', 'content_probe', 'machinery', 'passes'),
    'rescoring_top1': ('rescoring', 'top1'),
    'rescoring_rank_percentile': ('rescoring', 'rank_percentile'),
    'rescoring_stratified_top1': ('rescoring', 'length_stratified', 'top1'),
    # A decode-only run writes the capacity block to `evaluation/capacity.json` alone, which `_load_run` falls back to.
    'capacity_certified': ('decoder_capacity', 'verdict', 'capacity_certified'),
    'capacity_k': ('decoder_capacity', 'certified_k'),
    'capacity_bits': ('decoder_capacity', 'bits', 'bits_certified'),
    'capacity_bits_unrecovered': ('decoder_capacity', 'bits', 'bits_unrecovered'),
    'capacity_fraction_of_residual': ('decoder_capacity', 'bits', 'fraction_of_residual'),
    'capacity_readout': ('decoder_capacity', 'readout'),
    'capacity_flavor': ('decoder_capacity', 'headline', 'flavor'),
    'capacity_n_queries': ('decoder_capacity', 'n_queries'),
    'capacity_reason': ('decoder_capacity', 'verdict', 'reason'),
    'generation_verdict': ('scoreboard', 'verdict', 'generation_above_controls'),
    'generation_delta': ('scoreboard', 'held_out_generation', 'worst_control_ci', 'point'),
    'generation_delta_lo': ('scoreboard', 'held_out_generation', 'worst_control_ci', 'lo'),
    'generation_p': ('scoreboard', 'held_out_generation', 'permutation_p'),
    'prefix_influence_kl': ('scoreboard', 'held_out_generation', 'prefix_influence_kl'),
    'length_leakage_before': ('length_projection', 'length_leakage_before'),
    'length_leakage_after': ('length_projection', 'length_leakage_after'),
    'subject_probe': ('scoreboard', 'lift_over_raw', 'subject', 'zte_linear'),
    'subject_probe_raw': ('scoreboard', 'lift_over_raw', 'subject', 'raw_linear'),
    'word_len_probe': ('scoreboard', 'lift_over_raw', 'word_len', 'zte_linear'),
    'same_word_gap': ('emergence', 'cross_subject', 'same_word', 'gap'),
    'same_meaning_gap': ('emergence', 'cross_subject', 'same_meaning', 'gap'),
    'same_word_purity': ('emergence', 'neighbourhood', 'same_word_purity'),
    'cross_subject_neighbours': ('emergence', 'neighbourhood', 'cross_subject_neighbour_fraction'),
}
"""Every headline metric, keyed by name, as a path into a run's `evaluation/metrics.json`."""

# Config levers the ablation tables pivot on. Each is a dotted path into the resolved `config.yaml`.
_LEVERS: dict[str, str] = {
    'frontend': 'model.frontend',
    'spatial_encoding': 'model.spatial_encoding',
    'pos_encoding': 'model.pos_encoding',
    'objective': 'objective.name',
    'text_source': 'objective.text_source',
    'lexical_weight': 'objective.lexical_weight',
    'lexical_reader_weight': 'objective.lexical_reader_weight',
    'raw_align': 'dataset.raw_align',
    'subject_adapter': 'model.subject_adapter',
    'identity_orthogonality': 'objective.identity_orthogonality_weight',
    'subject_adversary': 'objective.subject_adversary_weight',
    'variance_weight': 'objective.variance_weight',
    'mode': 'train.mode',
    'split': 'train.split',
    'seed': 'train.seed',
    'holdout': 'train.loso_holdout_subject',
    'epochs': 'train.epochs',
    'residual_coding': 'model.residual_coding',
    'consensus_weight': 'objective.consensus_weight',
    'consensus_gallery_weight': 'objective.consensus_gallery_weight',
    'consensus_word_weight': 'objective.consensus_word_weight',
    'gallery_weight': 'objective.gallery_weight',
    'gallery_length_band': 'objective.gallery_length_band',
    'length_projection': 'objective.length_projection',
    'rate_ladder': 'decoder.rate_ladder',
    'rate_stages': 'decoder.rate_stages',
    'evidence_schedule': 'decoder.evidence_schedule',
    'conditioning': 'decoder.conditioning',
    'gap_correction': 'decoder.gap_correction',
    'stage0_epochs': 'decoder.stage0_epochs',
}


@dataclass(slots=True)
class Study:
    """Every frame the analysis draws from, collected once from a tree of run directories.

    Attributes:
        runs (pd.DataFrame): One row per run: its levers, its headline metrics and where its artifacts live.
        folds (pd.DataFrame): One row per (arm, seed, held-out subject) -- the LOSO trend.
        probes (pd.DataFrame): Long-form probe scores: run x target x representation.
        subjects (pd.DataFrame): Per-subject retrieval and probe rows, for the who-is-hard question.
        history (pd.DataFrame): Per-epoch training history, for the learning curves.
        generations (pd.DataFrame): Per-sentence decodes with their controls -- the text-level analysis.
        rebaseline (pd.DataFrame): The length-oracle audit rows, per run and tolerance.
        capacity (pd.DataFrame): The decoder menu-capacity sweep, one row per run x score family x pool flavor x
            subset x menu size x arm, with the paired control comparison carried on the control rows.
        roots (list[Path]): The directories that were walked.
    """

    runs: pd.DataFrame = field(default_factory=pd.DataFrame)
    folds: pd.DataFrame = field(default_factory=pd.DataFrame)
    probes: pd.DataFrame = field(default_factory=pd.DataFrame)
    subjects: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    generations: pd.DataFrame = field(default_factory=pd.DataFrame)
    rebaseline: pd.DataFrame = field(default_factory=pd.DataFrame)
    capacity: pd.DataFrame = field(default_factory=pd.DataFrame)
    roots: list[Path] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether no evaluated run was found."""
        return self.runs.empty


def dig(obj: Any, *path: str, default: Any = None) -> Any:
    """Walks nested dict keys, returning `default` on any miss.

    Args:
        obj (Any): The nested structure.
        *path (str): Keys to follow in order.
        default (Any, optional): Value when a key is missing. Defaults to None.

    Returns:
        Any: The value at `path`, or `default`.
    """
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def collect_study(roots: str | Path | list[str | Path], *, max_generation_rows: int = 40000) -> Study:
    """Reads every evaluated run under one or more experiment trees into tidy frames.

    Args:
        roots (str | Path | list[str | Path]): Directories holding per-run folders. A Drive mirror and a local tree
            can be passed together; a run present in both is kept once, from whichever was named first.
        max_generation_rows (int, optional): Cap on per-sentence decode rows loaded, so a twelve-fold sweep does not
            pull a hundred thousand strings into memory. Defaults to 40000.

    Returns:
        Study: The collected frames.
    """
    directories = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else list(roots))]
    seen: set[str] = set()
    runs: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    subjects: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    rebaseline: list[dict[str, Any]] = []
    capacity: list[dict[str, Any]] = []

    for root in directories:
        if not root.is_dir():
            _LOG.warning('Study root %s does not exist; skipping.', root)
            continue
        for run_dir in sorted(p for p in root.rglob('*') if (p / 'evaluation' / 'metrics.json').is_file()):
            if run_dir.name in seen:
                continue
            seen.add(run_dir.name)
            record = _load_run(run_dir)
            if record is None:
                continue
            runs.append(record)
            metrics = _read_json(run_dir / 'evaluation' / 'metrics.json')
            probes.extend(_probe_rows(run_dir.name, metrics))
            subjects.extend(_subject_rows(run_dir.name, metrics))
            history.extend(_history_rows(run_dir.name, metrics, run_dir))
            rebaseline.extend(_rebaseline_rows(run_dir.name, run_dir))
            capacity.extend(_capacity_rows(run_dir.name, _capacity_payload(run_dir, metrics)))
            if len(generations) < max_generation_rows:
                generations.extend(_generation_rows(run_dir.name, run_dir, max_generation_rows - len(generations)))

    study = Study(
        runs=pd.DataFrame(runs),
        probes=pd.DataFrame(probes),
        subjects=pd.DataFrame(subjects),
        history=pd.DataFrame(history),
        generations=pd.DataFrame(generations),
        rebaseline=pd.DataFrame(rebaseline),
        capacity=pd.DataFrame(capacity),
        roots=directories,
    )
    study.folds = _fold_frame(study.runs)
    _LOG.info(
        'Collected %d run(s): %d probe rows, %d subject rows, %d generation rows, %d rebaseline rows, '
        '%d capacity rows.',
        len(study.runs),
        len(study.probes),
        len(study.subjects),
        len(study.generations),
        len(study.rebaseline),
        len(study.capacity),
    )
    return study


def _load_run(run_dir: Path) -> dict[str, Any] | None:
    """Reads one run folder into a flat record of levers, headlines and artifact paths."""
    metrics = _read_json(run_dir / 'evaluation' / 'metrics.json')
    if not metrics:
        return None
    config = _read_yaml(run_dir / 'config.yaml')
    manifest = _read_json(run_dir / 'manifest.json')
    generation = _read_json(run_dir / 'evaluation' / 'generation.json')

    parsed = _RUN_NAME.match(run_dir.name)
    fallback = parsed.groupdict() if parsed else {}
    record: dict[str, Any] = {
        'run': run_dir.name,
        'arm': fallback.get('arm') or run_dir.name,
        'path': str(run_dir),
        # `zte-run --synthetic` records the flag outright; the data-root suffix is the fallback for a run whose
        # manifest predates it. Reading only the path would let a smoke run with a relocated cache read as real.
        'real_data': not (
            bool(dig(manifest, 'synthetic', default=False))
            or str(dig(manifest, 'data_root', default='')).endswith('synthetic_zuco')
        ),
        'n_words': dig(manifest, 'dataset', 'n_words'),
        'n_subjects': dig(manifest, 'dataset', 'n_subjects'),
        'wall_seconds': dig(manifest, 'wall_seconds'),
        'device': dig(manifest, 'device'),
        'git_commit': dig(manifest, 'git_commit'),
    }
    for name, dotted in _LEVERS.items():
        record[name] = dig(config, *dotted.split('.'))
    record['holdout'] = record.get('holdout') or fallback.get('holdout')
    if record.get('seed') is None and fallback.get('seed'):
        record['seed'] = int(str(fallback['seed']))

    capacity = _capacity_payload(run_dir, metrics)
    for name, path in HEADLINES.items():
        value = dig(metrics, *path)
        if value is None and path[0] in {'rescoring'}:
            value = dig(generation, *path)
        if value is None and path[0] == 'decoder_capacity':
            value = dig(capacity, *path[1:])
        record[name] = value
    record['held_out_lift'] = _sub(record.get('held_out_top1'), record.get('held_out_chance'))
    record['word_lift'] = _sub(record.get('word_top1'), record.get('word_chance'))
    record['has_generation'] = bool(dig(generation, 'generation', 'applicable'))
    record['bit_capacity'] = dig(generation, 'bit_budget', 'capacity_bits')
    record['bit_mutual_information'] = dig(generation, 'bit_budget', 'mutual_information_bits')
    record['bit_residual_mi'] = dig(generation, 'bit_budget', 'residual_mutual_information_bits')

    within = dig(generation, 'rescoring', 'within_task', default={}) or {}
    for task, cell in within.items():
        record[f'within_{task}_top1'] = cell.get('top1')
        record[f'within_{task}_rank_percentile'] = cell.get('rank_percentile')
        record[f'within_{task}_chance'] = cell.get('chance_top1')
        record[f'within_{task}_n_candidates'] = cell.get('n_candidates')
    return record


def _probe_rows(run: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Flattens `probe_comparison` into long form so probes pivot across runs."""
    return [
        {
            'run': run,
            'target': row.get('target'),
            'representation': row.get('representation'),
            'metric': row.get('metric'),
            'linear': _float(row.get('linear_score')),
            'knn': _float(row.get('knn_score')),
            'baseline': _float(row.get('baseline')),
        }
        for row in metrics.get('probe_comparison') or []
    ]


def _subject_rows(run: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Extracts the per-subject breakdown, which is where "who is hard" is visible."""
    rows: list[dict[str, Any]] = []
    for entry in metrics.get('per_subject') or []:
        if not isinstance(entry, dict):
            continue
        rows.append({'run': run, **{k: _float(v) if k != 'subject' else v for k, v in entry.items()}})
    return rows


def _history_rows(run: str, metrics: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Reads the per-epoch history, from `metrics.json` when it is there and `history.json` when it is not."""
    hist = metrics.get('history')
    if not isinstance(hist, dict):
        hist = _read_json(run_dir / 'history.json') or _read_json(run_dir / 'evaluation' / 'history.json')
    if not isinstance(hist, dict) or not hist:
        return []
    length = max((len(v) for v in hist.values() if isinstance(v, list)), default=0)
    rows: list[dict[str, Any]] = []
    for epoch in range(length):
        row: dict[str, Any] = {'run': run, 'epoch': epoch + 1}
        for key, series in hist.items():
            if isinstance(series, list) and epoch < len(series):
                row[key] = _float(series[epoch])
        rows.append(row)
    return rows


def _generation_rows(run: str, run_dir: Path, budget: int) -> list[dict[str, Any]]:
    """Reads `generation.jsonl` into one row per (sentence, condition), which is the text-level analysis."""
    path = run_dir / 'evaluation' / 'generation.jsonl'
    if not path.is_file() or budget <= 0:
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if len(rows) >= budget:
                    break
                record = json.loads(line)
                base = {
                    'run': run,
                    'index': record.get('index'),
                    'subject': record.get('subject'),
                    'task': record.get('task'),
                    'n_words': record.get('n_words'),
                    'reference': record.get('reference'),
                    'prefix_influence_kl': record.get('prefix_influence_kl'),
                }
                rows.append({**base, 'condition': 'hypothesis', 'text': record.get('hypothesis'), **_scores(record)})
                for name, cell in (record.get('controls') or {}).items():
                    rows.append({**base, 'condition': name, 'text': cell.get('text'), **_scores(cell)})
                if record.get('oracle'):
                    rows.append(
                        {
                            **base,
                            'condition': 'oracle',
                            'text': record['oracle'].get('text'),
                            **_scores(record['oracle']),
                        }
                    )
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning('Could not read %s (%r); skipping its generations.', path, exc)
    return rows


def _rebaseline_rows(run: str, run_dir: Path) -> list[dict[str, Any]]:
    """Reads the length-confound audit: what the encoder scores against a length-only oracle.

    Note:
        The audit's `grid` is nested `postprocess -> gallery -> metrics`, and only the `train_fitted` row is the
        one a decoder could reproduce -- `transductive` fits its whitening on the held-out subject too. Both are
        flattened here with `postprocess` carried, so the distinction survives into every chart.
    """
    payload = _read_json(run_dir / 'rebaseline' / 'rebaseline.json') or _read_json(run_dir / 'rebaseline.json')
    if not payload:
        return []

    rows: list[dict[str, Any]] = []
    for tolerance, cell in (payload.get('length_oracle') or {}).items():
        if isinstance(cell, dict):
            rows.append(
                {
                    'run': run,
                    'kind': 'oracle',
                    'tolerance': _float(tolerance),
                    **{k: _float(v) for k, v in cell.items()},
                }
            )
    for postprocess, galleries in (payload.get('grid') or {}).items():
        if not isinstance(galleries, dict):
            continue
        for gallery, cell in galleries.items():
            if not isinstance(cell, dict):
                continue
            rows.append(
                {
                    'run': run,
                    'kind': 'model',
                    'postprocess': postprocess,
                    'gallery': gallery,
                    **{k: _float(v) for k, v in cell.items() if not isinstance(v, (list, dict))},
                }
            )
    budget = payload.get('bit_budget') or {}
    if isinstance(budget, dict) and budget:
        rows.append({'run': run, 'kind': 'bit_budget', **{k: _float(v) for k, v in budget.items()}})
    floor = payload.get('floor_comparison') or {}
    if isinstance(floor, dict) and floor:
        rows.append(
            {
                'run': run,
                'kind': 'floor',
                'postprocess': floor.get('condition'),
                'gallery': floor.get('gallery'),
                'clears_floor': floor.get('clears_floor'),
                **{
                    k: _float(v) for k, v in floor.items() if k in {'encoder', 'encoder_ci_low', 'oracle', 'oracle_tol'}
                },
            }
        )
    return rows


def _capacity_payload(run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    """The decoder capacity block, from `metrics.json` or from the `capacity.json` a decode-only run writes."""
    block = metrics.get('decoder_capacity')
    if isinstance(block, dict) and block:
        return block

    payload = _read_json(run_dir / 'evaluation' / 'capacity.json') or _read_json(run_dir / 'capacity.json')
    block = payload.get('capacity')

    return block if isinstance(block, dict) else {}


def _capacity_rows(run: str, capacity: dict[str, Any]) -> list[dict[str, Any]]:
    """Flattens a capacity report into one row per score family, pool flavor, subset, menu size and arm.

    Note:
        `feasible` marks the sizes a candidate pool could actually fill. Exact word-count pools hold a median of
        eight candidates, so the largest swept sizes are routinely unreachable -- which is not the same fact as
        the model failing there, and a chart that conflates the two reads as a collapse.
    """
    headline = capacity.get('headline') or {}
    rows: list[dict[str, Any]] = []

    for family, flavors in (capacity.get('scores') or {}).items():
        if not isinstance(flavors, dict):
            continue
        for flavor, block in flavors.items():
            if not isinstance(block, dict):
                continue
            unreachable = {int(k) for k in (block.get('ks_unreachable') or [])}
            shared = {
                'run': run,
                'holdout': capacity.get('holdout'),
                'seed': dig(capacity, 'provenance', 'seed'),
                'score': str(family),
                'flavor': str(flavor),
                'headline': family == headline.get('score') and flavor == headline.get('flavor'),
                'alpha': _float(headline.get('alpha')),
                'certifiable': bool(block.get('certifiable')),
                'gamed': bool(block.get('gamed')),
                'certified_k': block.get('certified_k'),
            }
            for subset in ('per_k', 'common_subset'):
                for key, cell in (block.get(subset) or {}).items():
                    size = int(key)
                    base = shared | {'subset': subset, 'k': size, 'feasible': size not in unreachable}
                    rows.extend(_capacity_cell_rows(base, cell if isinstance(cell, dict) else {}))

    return rows


def _capacity_cell_rows(base: dict[str, Any], cell: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per arm inside a scored menu size, with the paired comparison carried on the control rows."""
    common = base | {
        'chance': _float(cell.get('chance')),
        'n_queries': cell.get('n_queries'),
        'perm_p': _float(cell.get('perm_p')),
        'perm_p_floor': _float(cell.get('perm_p_floor')),
        'certified': bool(cell.get('certified')),
        'failed_clauses': ';'.join(cell.get('failed_clauses') or []),
    }
    arms = cell.get('arms') or {}

    # A size no pool could fill still gets a row, so the sweep's shape survives into the frame with an empty
    # accuracy rather than a missing size a chart would silently close over.
    if not arms:
        return [common | {'arm': 'model', 'accuracy': None}]

    paired = cell.get('paired') or {}
    rows: list[dict[str, Any]] = []
    for arm, block in arms.items():
        interval = block.get('ci') or []
        pair = paired.get(arm) or {}
        pair_ci = pair.get('ci') or []
        rows.append(
            common
            | {
                'arm': str(arm),
                'accuracy': _float(block.get('accuracy')),
                'ci_lo': _float(interval[1]) if len(interval) >= 3 else None,
                'ci_hi': _float(interval[2]) if len(interval) >= 3 else None,
                'delta': _float(pair.get('delta')),
                'delta_lo': _float(pair_ci[1]) if len(pair_ci) >= 3 else None,
                'delta_hi': _float(pair_ci[2]) if len(pair_ci) >= 3 else None,
                'sign_test_p': _float(pair.get('sign_test_p')),
                'model_wins': pair.get('model_wins'),
                'control_wins': pair.get('control_wins'),
                'ties': pair.get('ties'),
                'n_pairs': pair.get('n_pairs'),
            }
        )

    return rows


def _fold_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """Returns the (arm, seed, holdout) view of the run frame -- the LOSO trend, one row per fold."""
    if runs.empty or 'holdout' not in runs.columns:
        return pd.DataFrame()
    folds = runs[runs['holdout'].notna()].copy()
    if folds.empty:
        return pd.DataFrame()
    keep = [
        c
        for c in (
            'run',
            'arm',
            'holdout',
            'seed',
            'frontend',
            'spatial_encoding',
            'held_out_top1',
            'held_out_top5',
            'held_out_rank_percentile',
            'held_out_chance',
            'held_out_lift',
            'held_out_n_queries',
            'stratified_top1',
            'stratified_rank_percentile',
            'effective_rank_ratio',
            'word_lift',
            'real_data',
        )
        if c in folds.columns
    ]
    return folds[keep].reset_index(drop=True)


def _scores(record: dict[str, Any]) -> dict[str, Any]:
    """Flattens a generation record's per-sentence score dict onto the row."""
    return {f'score_{k}': _float(v) for k, v in (record.get('scores') or {}).items()}


def _read_json(path: Path) -> dict[str, Any]:
    """Reads a JSON file, returning an empty dict when it is missing or unreadable."""
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning('Could not read %s (%r).', path, exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


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
    """Coerces a metric to float, tolerating `None`, strings and non-finite values."""
    if value is None or isinstance(value, bool):
        return None if value is None else float(value)
    try:
        out = float(value)
    except TypeError, ValueError:
        return None
    return out if np.isfinite(out) else None


def _sub(a: Any, b: Any) -> float | None:
    """`a - b` as a float, tolerating missing operands."""
    left, right = _float(a), _float(b)
    return None if left is None or right is None else left - right
