"""Combine every catalogued ZTE run into one interactive comparison dashboard.

Each `zte-run` writes a self-contained folder under `res/experiments/<name>/` with an `evaluation/metrics.json`, a `manifest.json`, and a resolved `config.yaml`.
This module reads all of them, normalises the load-bearing numbers into one comparable record per run, scores the runs against a **transparent, in-page rubric**,
and emits a single offline HTML dashboard: a red/green pass-fail **scorecard matrix** (runs x the four CI-backed checks), a sortable metric table with confidence-interval
bars, per-run cards that link out to each run's own Thought-Space Explorer / Neuron Atlas / report, and a "best run" verdict that shows *why* it won.

The dashboard is dependency-free (no Plotly): it renders from an embedded JSON payload with a little vanilla JS, so it stays small and fast and links to — rather than re-inlines — the
heavy per-run explorers.  It is safe to regenerate at any time; it simply reflects whatever runs are currently on disk.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zte.logging_utils import get_logger

_LOG = get_logger('evaluation.compare')


# --- The four headline checks (mirrors the evaluation verdict) --------------------------
# key -> (label, one-line meaning). `verdict[key]` is the boolean pass/fail.
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


# --- Scored quality metrics (the ranking rubric; weights are shown in the dashboard) -----
@dataclass(frozen=True)
class MetricSpec:
    """One scored quality metric.

    Attributes:
        key: Field name inside a run record's `metrics` block.
        label: Human label shown in the dashboard.
        higher_better: Whether a larger raw value is better.
        weight: Relative weight in the composite score (weights are renormalised to sum 1).
        fmt: A printf-style format for display.
        note: Short "why it matters" shown under the rubric.
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
        run_dir: A `res/experiments/<name>/` directory.

    Returns:
        A record dict (see `collect_runs`) or `None` if the run has no evaluation yet.
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
        experiments_root: The `res/experiments` directory.

    Returns:
        A list of run records, one per run that has an `evaluation/metrics.json`.
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

    Each metric is min-max normalised across the runs (direction-aware, so higher is always better), then combined with the rubric weights.  Runs are ranked primarily by the number
    of hard checks passed, then by the composite score — the same "pass the checks first, then optimise quality" ordering the report uses.

    Args:
        records (list[dict[str, Any]]): Run records from `collect_runs` (mutated in place and returned).

    Returns:
        The records, each with `norm` (per-metric 0..1) and `score` (0..1) added, sorted best first.  A `rank` field (1-based) is also set.

    """
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
        The written path.

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


# --- small IO / coercion helpers --------------------------------------------------------
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
    except Exception:  # noqa: BLE001 — config summary is best-effort, never fatal.
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

    The dashboard embeds the PCA thumbnail as a data URI so it renders **inline everywhere** —
    including Colab's sandboxed HTML display and a copy shared on Drive — where a relative
    `<img src>` path cannot be resolved. Opening the file locally still works either way.

    Args:
        path (Path): Path to a PNG file.

    Returns:
        A `data:image/png;base64,...` string, or `None` if the file is missing/unreadable.
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
    # Embed the PCA thumbnail itself (not just a relative link) so the card image shows
    # inline in Colab / a Drive copy; the relative links above still open in a new tab locally.
    out['pca_thumb'] = _data_uri(run_dir / candidates['pca'])
    return out


def _escape(text: str) -> str:
    """Minimal HTML escaping for text substituted into the template."""
    return (
        text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
    )


_COMPARE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
:root{
  --bg:#f7f8fb; --panel:#ffffff; --ink:#141922; --muted:#5c6773; --line:#e4e8ef;
  --pass:#0f9d6b; --passbg:#e3f6ee; --fail:#e5484d; --failbg:#fce9ea;
  --accent:#6d4aff; --accent2:#ff4d8d; --warn:#e8a13a;
  --heat0:#eef1f6; --shadow:0 1px 3px rgba(20,25,34,.08),0 8px 24px rgba(20,25,34,.06);
}
:root[data-theme="dark"]{
  --bg:#0e1117; --panel:#161b25; --ink:#e8edf5; --muted:#96a2b4; --line:#242c3a;
  --pass:#3ddc97; --passbg:#123227; --fail:#ff6b6f; --failbg:#331a1d;
  --accent:#9b7bff; --accent2:#ff77a9; --warn:#f0b95e;
  --heat0:#1b2230; --shadow:0 1px 3px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0e1117; --panel:#161b25; --ink:#e8edf5; --muted:#96a2b4; --line:#242c3a;
    --pass:#3ddc97; --passbg:#123227; --fail:#ff6b6f; --failbg:#331a1d;
    --accent:#9b7bff; --accent2:#ff77a9; --warn:#f0b95e;
    --heat0:#1b2230; --shadow:0 1px 3px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
h1{font-size:22px;margin:0;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px;margin-top:2px}
.themebtn{border:1px solid var(--line);background:var(--panel);color:var(--ink);
  border-radius:999px;padding:7px 13px;cursor:pointer;font-size:13px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  box-shadow:var(--shadow);padding:18px 18px}
.section{margin-top:22px}
.section h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  margin:0 0 10px 2px;font-weight:700}
/* Best banner */
.best{display:flex;align-items:center;gap:18px;flex-wrap:wrap;
  background:linear-gradient(120deg,color-mix(in srgb,var(--accent) 14%,var(--panel)),
  color-mix(in srgb,var(--accent2) 12%,var(--panel)));}
.best .medal{font-size:34px}
.best .name{font-size:20px;font-weight:750;letter-spacing:-.01em}
.pill{display:inline-block;padding:3px 9px;border-radius:999px;font-size:12px;font-weight:650}
.pill.pass{background:var(--passbg);color:var(--pass)}
.pill.real{background:color-mix(in srgb,var(--warn) 22%,transparent);color:var(--warn)}
.scorebar{height:8px;border-radius:6px;background:var(--heat0);overflow:hidden;min-width:120px;flex:1}
.scorebar>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2))}
/* Scorecard matrix */
.scroll{overflow-x:auto}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:13px}
th,td{padding:8px 10px;text-align:center;white-space:nowrap}
th.rowhead,td.rowhead{text-align:left;position:sticky;left:0;background:var(--panel);z-index:2;
  border-right:1px solid var(--line);min-width:190px}
thead th{color:var(--muted);font-weight:650;border-bottom:1px solid var(--line);cursor:pointer}
thead th.run .rn{display:block;font-weight:700;color:var(--ink)}
thead th.run .cfg{display:block;font-size:11px;color:var(--muted);font-weight:500}
tbody tr:hover td{background:color-mix(in srgb,var(--accent) 6%,transparent)}
.cell{border-radius:8px}
.mark{display:inline-flex;width:24px;height:24px;border-radius:7px;align-items:center;
  justify-content:center;font-weight:800}
.mark.y{background:var(--passbg);color:var(--pass)}
.mark.n{background:var(--failbg);color:var(--fail)}
.count{font-weight:800}
.heat{border-radius:7px;font-variant-numeric:tabular-nums}
.desc{color:var(--muted);font-size:11px;font-weight:500;display:block}
/* CI bars */
.ci{position:relative;height:16px;width:120px;margin:0 auto;background:var(--heat0);border-radius:6px}
.ci .zero{position:absolute;top:-2px;bottom:-2px;width:1px;background:var(--muted);opacity:.7}
.ci .bar{position:absolute;top:4px;height:8px;border-radius:4px;background:var(--muted)}
.ci .bar.good{background:var(--pass)}
.ci .bar.bad{background:var(--fail)}
.ci .pt{position:absolute;top:2px;width:2px;height:12px;background:var(--ink)}
/* Cards */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}
.runcard{padding:0;overflow:hidden}
.runcard .thumb{height:120px;background:var(--heat0);display:block;object-fit:cover;width:100%}
.runcard .body{padding:12px 13px}
.runcard .rname{font-weight:700;display:flex;align-items:center;gap:6px}
.runcard .links{display:flex;gap:8px;flex-wrap:wrap;margin-top:9px}
.runcard .links a{font-size:12px;border:1px solid var(--line);border-radius:8px;padding:4px 8px}
.rank{font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:999px;
  padding:1px 7px}
.rubric{color:var(--muted);font-size:12.5px}
.rubric b{color:var(--ink)}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px}
.empty{color:var(--muted);opacity:.6}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>__TITLE__</h1>
      <div class="sub" id="sub"></div>
    </div>
    <button class="themebtn" id="themebtn">◐ Theme</button>
  </header>

  <div class="section"><div class="card best" id="best"></div></div>

  <div class="section">
    <h2>Scorecard — pass/fail checks &amp; key geometry across runs</h2>
    <div class="card scroll"><table id="scorecard"></table></div>
    <div class="legend" id="legend"></div>
  </div>

  <div class="section">
    <h2>Ranked detail — sortable (click a column)</h2>
    <div class="card scroll"><table id="detail"></table></div>
  </div>

  <div class="section">
    <h2>Open a run — its own explorer, neuron atlas &amp; report</h2>
    <div class="grid" id="cards"></div>
  </div>

  <div class="section">
    <h2>How "best" is scored (transparent rubric)</h2>
    <div class="card rubric" id="rubric"></div>
  </div>
</div>

<script>
const P = "__PAYLOAD__";
const runs = P.runs, checks = P.rubric.checks, metrics = P.rubric.metrics;

/* ---- theme ---- */
const root=document.documentElement;
function applyTheme(t){ if(t) root.setAttribute('data-theme',t); }
document.getElementById('themebtn').onclick=()=>{
  const cur=root.getAttribute('data-theme');
  const next = cur==='dark' ? 'light' : cur==='light' ? 'dark'
    : (matchMedia('(prefers-color-scheme: dark)').matches ? 'light' : 'dark');
  applyTheme(next);
};

function fmt(v,f){ if(v===null||v===undefined) return '—';
  if(f==='%+.3f') return (v>=0?'+':'')+v.toFixed(3);
  if(f==='%.3f') return v.toFixed(3);
  return String(v); }
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function cfgLine(r){
  const c=r.config||{}; const bits=[];
  bits.push(c.objective||'?');
  bits.push(c.eye_tracking?'ET':'EEG-only');
  if(c.split) bits.push(c.split.replace('by_',''));
  if(c.loso_holdout) bits.push('−'+c.loso_holdout);
  if(c.cross_subject_positives) bits.push('xsubj+');
  if(c.subject_adversary_weight>0) bits.push('adv'+c.subject_adversary_weight);
  if(c.variance_weight>0||c.covariance_weight>0) bits.push('vicreg');
  return bits.join(' · ');
}

document.getElementById('sub').textContent =
  runs.length+' run'+(runs.length===1?'':'s')+' · '+
  runs.filter(r=>r.real).length+' on real data, '+
  runs.filter(r=>!r.real).length+' synthetic · green = pass, red = fail';

/* ---- best banner ---- */
(function(){
  const b=runs[0]; if(!b){document.getElementById('best').innerHTML='<span class="empty">No runs.</span>';return;}
  const why = metrics.map(m=>({m,u:(b.norm||{})[m.key]})).filter(x=>x.u!=null)
    .sort((a,z)=>z.u*z.m.weight-a.u*a.m.weight).slice(0,3)
    .map(x=>esc(x.m.label)).join(', ');
  document.getElementById('best').innerHTML =
    '<div class="medal">🥇</div>'+
    '<div style="min-width:200px"><div class="name">'+esc(b.name)+'</div>'+
      '<div class="sub">'+esc(cfgLine(b))+(b.real?' · <span class="pill real">real data</span>':'')+'</div></div>'+
    '<div class="pill pass">'+b.checks_passed+' / '+checks.length+' checks</div>'+
    '<div class="scorebar"><i style="width:'+(b.score*100).toFixed(0)+'%"></i></div>'+
    '<div class="sub" style="min-width:210px">Leads on: '+(why||'—')+'</div>';
})();

/* ---- scorecard matrix ---- */
function heatColor(u){ // u in 0..1, higher better -> greener
  if(u==null) return 'var(--heat0)';
  const g='var(--pass)', r='var(--fail)';
  return 'color-mix(in srgb, '+g+' '+Math.round(u*55+8)+'%, '+
         'color-mix(in srgb, '+r+' '+Math.round((1-u)*40)+'%, var(--panel)))';
}
(function(){
  const t=document.getElementById('scorecard');
  let h='<thead><tr><th class="rowhead">metric</th>';
  runs.forEach(r=>{ h+='<th class="run"><span class="rn">'+esc(r.name)+'</span>'+
    '<span class="cfg">'+esc(cfgLine(r))+'</span></th>'; });
  h+='</tr></thead><tbody>';
  // checks
  checks.forEach(c=>{
    h+='<tr><td class="rowhead">'+esc(c.label)+'<span class="desc">'+esc(c.desc)+'</span></td>';
    runs.forEach(r=>{ const ok=r.checks[c.key];
      h+='<td><span class="mark '+(ok?'y':'n')+'">'+(ok?'✓':'✕')+'</span></td>'; });
    h+='</tr>';
  });
  // checks passed
  h+='<tr><td class="rowhead">Checks passed</td>';
  runs.forEach(r=>{ h+='<td><span class="count">'+r.checks_passed+' / '+checks.length+'</span></td>'; });
  h+='</tr>';
  // scored metrics as heat
  metrics.forEach(m=>{
    h+='<tr><td class="rowhead">'+esc(m.label)+
       '<span class="desc">'+(m.higher_better?'▲ higher':'▼ lower')+' is better</span></td>';
    runs.forEach(r=>{ const v=r.metrics[m.key], u=(r.norm||{})[m.key];
      h+='<td><span class="cell heat" style="display:inline-block;padding:5px 9px;background:'+
        heatColor(u)+'">'+fmt(v,m.fmt)+'</span></td>'; });
    h+='</tr>';
  });
  h+='</tbody>';
  t.innerHTML=h;
})();
document.getElementById('legend').innerHTML =
  '<span>✓ pass&nbsp;·&nbsp;✕ fail</span><span>Heat cells: greener = better among these runs</span>';

/* ---- CI bar ---- */
function ciBar(ci){ // ci = [point, lo, hi]; good if lo>0
  if(!ci || ci.length<3 || ci[0]==null) return '<span class="empty">—</span>';
  const [pt,lo,hi]=ci; const span=Math.max(Math.abs(lo),Math.abs(hi),0.02)*1.15;
  const x=v=>(0.5+0.5*(v/span))*100;
  const good=lo>0;
  return '<div class="ci"><span class="zero" style="left:50%"></span>'+
    '<span class="bar '+(good?'good':(hi<0?'bad':''))+'" style="left:'+x(lo).toFixed(1)+
    '%;width:'+Math.max(1,(x(hi)-x(lo))).toFixed(1)+'%"></span>'+
    '<span class="pt" style="left:'+x(pt).toFixed(1)+'%"></span></div>';
}

/* ---- detail table (sortable) ---- */
const detailCols = [
  {k:'rank',label:'#',get:r=>r.rank,num:true},
  {k:'name',label:'run',get:r=>r.name,num:false},
  {k:'checks_passed',label:'checks',get:r=>r.checks_passed,num:true},
  {k:'score',label:'score',get:r=>r.score,num:true,fmt:v=>(v*100).toFixed(0)},
];
metrics.forEach(m=>detailCols.push({k:m.key,label:m.label,get:r=>r.metrics[m.key],num:true,fmt:v=>fmt(v,m.fmt),spec:m}));
detailCols.push({k:'retr_ci',label:'retrieval CI',get:r=>r.ci.retrieval,num:false,ci:true});
detailCols.push({k:'arith_ci',label:'arithmetic CI',get:r=>r.ci.arithmetic,num:false,ci:true});

let sortKey='rank', sortAsc=true;
function renderDetail(){
  const rows=[...runs].sort((a,z)=>{
    const col=detailCols.find(c=>c.k===sortKey);
    let av=col.get(a), zv=col.get(z);
    if(col.ci){av=(av&&av[1])||0; zv=(zv&&zv[1])||0;}
    if(av==null)av=-1e9; if(zv==null)zv=-1e9;
    if(typeof av==='string') return sortAsc?av.localeCompare(zv):zv.localeCompare(av);
    return sortAsc?av-zv:zv-av;
  });
  let h='<thead><tr>';
  detailCols.forEach(c=>{ h+='<th class="'+(c.k==='name'?'rowhead':'')+'">'+esc(c.label)+
    (sortKey===c.k?(sortAsc?' ▲':' ▼'):'')+'</th>'; });
  h+='</tr></thead><tbody>';
  rows.forEach(r=>{
    h+='<tr>';
    detailCols.forEach(c=>{
      let v=c.get(r), disp;
      if(c.ci) disp=ciBar(v);
      else if(v==null) disp='<span class="empty">—</span>';
      else if(c.fmt) disp=c.fmt(v);
      else disp=esc(v);
      const cls=c.k==='name'?'rowhead':(c.spec?'heat':'');
      let style='';
      if(c.spec){const u=(r.norm||{})[c.spec.key]; style=' style="background:'+heatColor(u)+'"';}
      h+='<td class="'+cls+'"'+style+'>'+disp+'</td>';
    });
    h+='</tr>';
  });
  h+='</tbody>';
  const t=document.getElementById('detail'); t.innerHTML=h;
  t.querySelectorAll('thead th').forEach((th,i)=>{ th.onclick=()=>{
    const c=detailCols[i]; if(sortKey===c.k) sortAsc=!sortAsc; else {sortKey=c.k; sortAsc=c.num?false:true;}
    renderDetail(); }; });
}
renderDetail();

/* ---- run cards ---- */
(function(){
  let h='';
  runs.forEach(r=>{
    const p=r.paths||{};
    const thumb=p.pca_thumb? '<img class="thumb" src="'+p.pca_thumb+'" alt="PCA by subject"/>'
      : '<div class="thumb"></div>';
    const links=[];
    if(p.explorer) links.push('<a href="'+esc(p.explorer)+'" target="_blank">🌌 Explorer</a>');
    if(p.atlas) links.push('<a href="'+esc(p.atlas)+'" target="_blank">🧠 Neuron Atlas</a>');
    if(p.report) links.push('<a href="'+esc(p.report)+'" target="_blank">📄 Report</a>');
    h+='<div class="card runcard">'+thumb+'<div class="body">'+
      '<div class="rname"><span class="rank">#'+r.rank+'</span>'+esc(r.name)+
        (r.real?' <span class="pill real">real</span>':'')+'</div>'+
      '<div class="sub">'+esc(cfgLine(r))+'</div>'+
      '<div class="sub" style="margin-top:5px">'+r.checks_passed+'/'+checks.length+
        ' checks · score '+(r.score*100).toFixed(0)+'</div>'+
      '<div class="links">'+(links.join('')||'<span class="empty">no artifacts</span>')+'</div>'+
      '</div></div>';
  });
  document.getElementById('cards').innerHTML=h;
})();

/* ---- rubric ---- */
(function(){
  let h='<p>Runs are ranked <b>first by the number of hard checks passed</b>, then by a weighted '+
    'quality score. Every metric is min–max normalised across the runs shown (direction-aware, so '+
    'higher is always better), then combined with these weights:</p><ul>';
  const tot=metrics.reduce((s,m)=>s+m.weight,0);
  metrics.forEach(m=>{ h+='<li><b>'+esc(m.label)+'</b> — '+(m.higher_better?'▲ higher':'▼ lower')+
    ' is better · weight '+(m.weight/tot*100).toFixed(0)+'% — <span>'+esc(m.note)+'</span></li>'; });
  h+='</ul><p class="empty">Normalisation is relative to the runs on this page, so the score '+
    'compares these experiments against each other — it is not an absolute quality scale.</p>';
  document.getElementById('rubric').innerHTML=h;
})();
</script>
</body>
</html>
"""
