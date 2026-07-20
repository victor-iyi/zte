"""Collects, scores and renders the cross-run experiment comparison dashboard."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zte.evaluation.interactive._assets import load_page
from zte.logging_utils import get_logger

_LOG = get_logger('evaluation.compare')


# The four headline pass/fail checks, as `(verdict key, label, meaning)`.
CHECKS: tuple[tuple[str, str, str], ...] = (
    ('beats_noise', 'Beats noise', 'A lexical probe beats a noise control (CI-backed).'),
    ('no_collapse', 'No collapse', 'The space uses real dimensions, not ~10 of 768.'),
    (
        'retrieval_above_chance',
        'Cross-subject retrieval > chance',
        "Read a sentence from another person's EEG, above chance (CI > 0).",
    ),
    (
        'subject_arithmetic_above_chance',
        'Thought arithmetic > chance',
        'emb(t,A) - A + B lands on emb(t,B), above chance (CI > 0).',
    ),
)


# Scored quality metrics: the ranking rubric, whose weights the dashboard displays.
@dataclass(slots=True, frozen=True)
class MetricSpec:
    """One scored quality metric in the ranking rubric.

    Attributes:
        key (str): Field name inside a run record's `metrics` block.
        label (str): Human label shown in the dashboard.
        higher_better (bool): Whether a larger raw value is better.
        weight (float): Relative weight in the composite score (weights are renormalised to sum 1).
        fmt (str): A printf-style format for display.
        note (str): Short "why it matters" shown under the rubric.
    """

    key: str
    label: str
    higher_better: bool
    weight: float
    fmt: str
    note: str


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        'retrieval_margin',
        'Retrieval lift over chance',
        True,
        0.25,
        '%+.3f',
        'The headline capability: reading a sentence from another brain.',
    ),
    MetricSpec(
        'who_vs_what',
        'Subject-variance share (who)',
        False,
        0.20,
        '%.3f',
        'Share of the space spent encoding WHO read it — lower is more invariant.',
    ),
    MetricSpec(
        'same_meaning_gap',
        'Same-meaning clustering',
        True,
        0.15,
        '%+.3f',
        'The north star: same meaning across people sits closer than random.',
    ),
    MetricSpec(
        'same_word_gap',
        'Same-word clustering',
        True,
        0.15,
        '%+.3f',
        'Same word across people clusters (can be a stimulus shortcut — see docs).',
    ),
    MetricSpec(
        'effective_rank_ratio',
        'Effective-rank ratio',
        True,
        0.10,
        '%.3f',
        'How many of the 768 dimensions are actually used.',
    ),
    MetricSpec(
        'anisotropy',
        'Anisotropy (cone)',
        False,
        0.10,
        '%.3f',
        'Near 1.0 means a degenerate cone — rank can look high yet carry no structure.',
    ),
    MetricSpec(
        'subject_knn',
        'Subject readability (kNN)',
        False,
        0.05,
        '%.3f',
        'Can a classifier still tell WHO from the code — lower is more invariant.',
    ),
)


def _dig(obj: Any, *path: str, default: Any = None) -> Any:
    """Safely walks nested dict keys, returning `default` on any miss."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _subject_knn(metrics: dict) -> float | None:
    """Extracts the ZTE subject kNN probe accuracy (lower = more subject-invariant)."""
    for row in metrics.get('probe_comparison', []) or []:
        if row.get('target') == 'subject' and row.get('representation') == 'ZTE':
            val = row.get('knn_score')
            return float(val) if val is not None else None
    return None


def _config_summary(config: dict) -> dict[str, Any]:
    """Pulls the small set of levers that distinguish one experiment from another."""
    obj = config.get('objective', {}) or {}
    ds = config.get('dataset', {}) or {}
    tr = config.get('train', {}) or {}
    mdl = config.get('model', {}) or {}
    return {
        'objective': obj.get('name'),
        'pos_encoding': mdl.get('pos_encoding'),
        'eye_tracking': bool(ds.get('include_eye_tracking', False)),
        'normalize': ds.get('normalize'),
        'split': tr.get('split'),
        'loso_holdout': tr.get('loso_holdout_subject'),
        'cross_subject_positives': bool(obj.get('cross_subject_positives', False)),
        'subject_adversary_weight': float(obj.get('subject_adversary_weight', 0.0) or 0.0),
        'variance_weight': float(obj.get('variance_weight', 0.0) or 0.0),
        'covariance_weight': float(obj.get('covariance_weight', 0.0) or 0.0),
    }


def load_run_record(run_dir: Path) -> dict[str, Any] | None:
    """Reads one run folder into a normalised, comparable record.

    Args:
        run_dir (Path): A `res/experiments/<name>/` directory.

    Returns:
        dict[str, Any] | None: A record dict, or `None` if the run has no evaluation yet.
    """
    metrics_path = run_dir / 'evaluation' / 'metrics.json'
    if not metrics_path.is_file():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    except OSError, json.JSONDecodeError:
        _LOG.warning('Could not read %s; skipping.', metrics_path)
        return None

    manifest = _read_json(run_dir / 'manifest.json')
    config = _read_yaml(run_dir / 'config.yaml')

    # Resolve each headline check; `beats_noise` accepts either verdict key spelling.
    verdict = metrics.get('verdict', {}) or {}
    checks: dict[str, bool] = {}
    for key, _label, _desc in CHECKS:
        if key == 'beats_noise':
            checks[key] = bool(verdict.get('beats_noise_all_targets')) or bool(
                verdict.get('beats_noise_on')
            )
        else:
            checks[key] = bool(verdict.get(key))
    checks_passed = sum(1 for v in checks.values() if v)

    # Retrieval is scored as a lift over its own chance level, so runs of different sizes compare.
    sr = metrics.get('sentence_retrieval', {}) or {}
    retr_top1 = _as_float(sr.get('top1'))
    retr_chance = _as_float(sr.get('chance_top1'))
    retrieval_margin = (
        retr_top1 - retr_chance if retr_top1 is not None and retr_chance is not None else None
    )

    record_metrics = {
        'retrieval_margin': retrieval_margin,
        'who_vs_what': _as_float(_dig(metrics, 'neurons', 'who_vs_what_ratio')),
        'same_meaning_gap': _as_float(
            _dig(metrics, 'emergence', 'cross_subject', 'same_meaning', 'gap')
        ),
        'same_word_gap': _as_float(_dig(metrics, 'emergence', 'cross_subject', 'same_word', 'gap')),
        'effective_rank_ratio': _as_float(
            _dig(metrics, 'embedding_health', 'effective_rank_ratio')
        ),
        'anisotropy': _as_float(_dig(metrics, 'embedding_health', 'anisotropy')),
        'subject_knn': _subject_knn(metrics),
    }

    name = run_dir.name
    return {
        'name': name,
        'real': not str(_dig(manifest, 'data_root', default='')).endswith('synthetic_zuco'),
        'n_words': _dig(manifest, 'dataset', 'n_words'),
        'n_subjects': _dig(manifest, 'dataset', 'n_subjects'),
        'subjects': _dig(manifest, 'dataset', 'subjects', default=[]),
        'tasks': _dig(manifest, 'dataset', 'tasks', default=[]),
        'config': _config_summary(config),
        'checks': checks,
        'checks_passed': checks_passed,
        'ci': {
            'retrieval': verdict.get('retrieval_ci'),
            'arithmetic': verdict.get('subject_arithmetic_ci'),
        },
        'metrics': record_metrics,
        'emergence_headline': _dig(metrics, 'emergence', 'headline'),
        'paths': _relative_paths(run_dir),
    }


def collect_runs(experiments_root: str | Path) -> list[dict[str, Any]]:
    """Loads every evaluated run under `experiments_root` into comparable records.

    Args:
        experiments_root (str | Path): The `res/experiments` directory.

    Returns:
        list[dict[str, Any]]: One record per run that has an `evaluation/metrics.json`.
    """
    root = Path(experiments_root)
    records: list[dict[str, Any]] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        rec = load_run_record(child)
        if rec is not None:
            records.append(rec)
    _LOG.info('Loaded %d evaluated run(s) from %s.', len(records), root)
    return records


def score_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adds a normalised per-metric score and a composite `score` to each record.

    Normalisation is min-max across the supplied records, so scores are only meaningful relative to this cohort.

    Args:
        records (list[dict[str, Any]]): Run records from `collect_runs` (mutated in place and returned).

    Returns:
        list[dict[str, Any]]: The records with `norm`, `score` and a 1-based `rank` added, sorted best first.
    """
    # Min-max each metric across the cohort, flipping the sense of lower-is-better ones.
    total_weight = sum(spec.weight for spec in METRIC_SPECS) or 1.0
    for spec in METRIC_SPECS:
        values = [
            r['metrics'].get(spec.key)
            for r in records
            if _as_float(r['metrics'].get(spec.key)) is not None
        ]
        values = [v for v in values if v is not None]
        lo = min(values) if values else 0.0
        hi = max(values) if values else 1.0
        span = (hi - lo) or 1.0
        for r in records:
            raw = _as_float(r['metrics'].get(spec.key))
            r.setdefault('norm', {})
            if raw is None:
                r['norm'][spec.key] = None
                continue
            unit = (raw - lo) / span
            r['norm'][spec.key] = unit if spec.higher_better else 1.0 - unit

    # Weighted mean of whatever normalised metrics a run has, then rank checks-first.
    for r in records:
        acc = 0.0
        for spec in METRIC_SPECS:
            unit = r['norm'].get(spec.key)
            if unit is not None:
                acc += spec.weight * unit
        r['score'] = acc / total_weight

    records.sort(key=lambda r: (r['checks_passed'], r['score']), reverse=True)
    for i, r in enumerate(records, start=1):
        r['rank'] = i
    return records


def _rubric_payload() -> dict[str, Any]:
    """Serialises the checks and metric rubric for display in the dashboard."""
    return {
        'checks': [{'key': k, 'label': lab, 'desc': d} for k, lab, d in CHECKS],
        'metrics': [
            {
                'key': s.key,
                'label': s.label,
                'higher_better': s.higher_better,
                'weight': s.weight,
                'fmt': s.fmt,
                'note': s.note,
            }
            for s in METRIC_SPECS
        ],
    }


def combined_dashboard_html(
    records: list[dict[str, Any]],
    out_path: str | Path,
    *,
    title: str = 'ZTE — Experiment Comparison',
) -> Path:
    """Writes the single-file interactive comparison dashboard.

    Args:
        records (list[dict[str, Any]]): Scored run records (call `score_runs` first).
        out_path (str | Path): Destination `.html` path.
        title (str): Page title.

    Returns:
        Path: The written path.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'title': title,
        'runs': records,
        'rubric': _rubric_payload(),
        'best': records[0]['name'] if records else None,
    }
    html = _COMPARE_TEMPLATE.replace(
        '"__PAYLOAD__"', json.dumps(payload, separators=(',', ':'))
    ).replace('__TITLE__', _escape(title))
    out.write_text(html, encoding='utf-8')
    _LOG.info('Wrote comparison dashboard (%d runs) to %s', len(records), out)
    return out


def build_comparison(
    experiments_root: str | Path,
    out_path: str | Path | None = None,
    *,
    title: str = 'ZTE — Experiment Comparison',
) -> Path:
    """End-to-end: collect, score, and render the comparison dashboard.

    Args:
        experiments_root (str | Path): The `res/experiments` directory to scan.
        out_path (str | Path | None): Output HTML path (defaults to `<experiments_root>/COMPARE.html`).
        title (str): Page title.

    Returns:
        Path: The written HTML path.

    Raises:
        ValueError: If no evaluated runs were found.
    """
    root = Path(experiments_root)
    records = collect_runs(root)
    if not records:
        raise ValueError(f'No evaluated runs found under {root} (need evaluation/metrics.json).')
    score_runs(records)
    out = Path(out_path) if out_path is not None else root / 'COMPARE.html'
    return combined_dashboard_html(records, out, title=title)


# Small IO / coercion helpers.
def _read_json(path: Path) -> dict[str, Any]:
    """Reads a JSON file, returning `{}` on any error."""
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except OSError, json.JSONDecodeError:
        return {}


def _read_yaml(path: Path) -> dict[str, Any]:
    """Reads a YAML config file, returning `{}` on any error."""
    if not path.is_file():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except Exception:  # noqa: BLE001 -- config summary is best-effort, never fatal.
        return {}


def _as_float(value: Any) -> float | None:
    """Coerces to float, mapping non-finite / missing to `None`."""
    if value is None:
        return None
    try:
        f = float(value)
    except TypeError, ValueError:
        return None
    if not math.isfinite(f):  # NaN / inf
        return None
    return f


def _data_uri(path: Path) -> str | None:
    """Reads a PNG and returns a base64 `data:` URI, or `None` if unreadable.

    Thumbnails are embedded rather than linked because Colab's sandboxed HTML display and Drive copies cannot
    resolve a relative `<img src>` path.

    Args:
        path (Path): Path to a PNG file.

    Returns:
        str | None: A `data:image/png;base64,...` string, or `None` if the file is missing or unreadable.
    """
    import base64

    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')


def _relative_paths(run_dir: Path) -> dict[str, str | None]:
    """Builds links (relative to the run's parent) to a run's rich artifacts."""
    base = run_dir.name
    candidates = {
        'explorer': 'evaluation/interactive/thought_space_explorer.html',
        'atlas': 'evaluation/interactive/neuron_atlas.html',
        'report': 'evaluation/report.md',
        'pca': 'evaluation/figures/pca_by_subject.png',
    }
    out: dict[str, str | None] = {}
    for key, rel in candidates.items():
        out[key] = f'{base}/{rel}' if (run_dir / rel).is_file() else None
    # The card image is embedded, not linked, so it survives Colab and Drive copies.
    out['pca_thumb'] = _data_uri(run_dir / candidates['pca'])
    return out


def _escape(text: str) -> str:
    """Minimal HTML escaping for text substituted into the template."""
    return (
        text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
    )


_COMPARE_TEMPLATE: str = load_page('compare')
