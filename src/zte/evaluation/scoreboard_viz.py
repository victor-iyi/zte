"""A self-contained interactive dashboard for the *held-out* honest scoreboard.

`scoreboard.build_scoreboard` returns the single most load-bearing block in the whole
evaluation: what the space looks like **on a stranger's brain** (the LOSO held-out
subject) and whether cross-subject retrieval clears its own chance line. Everything in
the improvement plan hinges on that one retrieval number, so it deserves its own view
rather than a row buried in a report.

This module renders that block as one dependency-free, theme-aware `.html` file — the
same offline single-file idiom as `zte.evaluation.compare` (an inline `<style>` with CSS
variables for light/dark, an embedded JSON island, and vanilla JS that computes every
colour/verdict in the browser). It matches the ZTE HTML family palette
(`interactive.py`: accents ``#5a4bff`` / ``#ff4d8d``, good ``#0f9d6b``, warn ``#e29008``,
bad ``#e5484d``).

Honesty rules it keeps: every metric is judged against an *explicitly named* reference
line (never a bare threshold), and every probe/retrieval number is framed as a lift over
the raw band-power control — because raw band power currently beats the encoder on a
stranger, so only ``ZTE - raw`` is progress.
"""

from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
from typing import Any

# Blocks that can appear as the held-out ("new brain") view and the optional in-sample
# view. `build_scoreboard` stores retrieval under `held_out_retrieval`; the function that
# computes it is `cross_subject_holdout_retrieval`, so we accept either key defensively.
_HELD_GEOMETRY_KEYS: tuple[str, ...] = ('held_out_geometry',)
_HELD_RETRIEVAL_KEYS: tuple[str, ...] = ('held_out_retrieval', 'cross_subject_holdout_retrieval')
_INSAMPLE_GEOMETRY_KEYS: tuple[str, ...] = ('in_sample_geometry', 'geometry')
_INSAMPLE_RETRIEVAL_KEYS: tuple[str, ...] = (
    'in_sample_retrieval',
    'in_sample',
    'cross_subject_retrieval',
)


def scoreboard_html(scoreboard: dict, out_path: str | Path, run_name: str = 'ZTE run') -> Path:
    """Writes the interactive held-out scoreboard dashboard as a single offline HTML file.

    The output is fully self-contained (no external hosts, CDNs or fonts): a JSON island
    carrying the rounded scoreboard numbers plus vanilla JS that derives the verdict,
    colour-codes each stat card against its own named reference line, and draws the
    ``ZTE - raw`` lift comparison. It degrades to a "no scoreboard data" card when the
    dict is empty and hides any block whose numbers are absent.

    Args:
        scoreboard (dict): The dict from `zte.evaluation.scoreboard.build_scoreboard`.
            Read defensively — `held_out_geometry`, `held_out_retrieval` /
            `cross_subject_holdout_retrieval`, `lift_over_raw` (with its `content_probe`
            sub-block) and any in-sample blocks are all optional.
        out_path (str | Path): Destination `.html` path (parents are created; a non-html
            suffix is rewritten to `.html`).
        run_name (str): Human label for the run, shown in the header and the page title.

    Returns:
        Path: The written HTML file path.
    """
    out = Path(out_path)
    if out.suffix != '.html':
        out = out.with_suffix('.html')
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = _build_payload(scoreboard or {}, run_name)
    data_json = json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c')
    html = _TEMPLATE.replace('__TITLE__', _esc(run_name)).replace('__DATA__', data_json)
    out.write_text(html, encoding='utf-8')
    return out


def _build_payload(board: dict, run_name: str) -> dict[str, Any]:
    """Normalises the raw scoreboard dict into the small JSON island the page consumes."""
    lift_block = board.get('lift_over_raw') or {}
    content_probe = lift_block.get('content_probe') if isinstance(lift_block, dict) else None

    lift_list: list[dict[str, Any]] = []
    if isinstance(lift_block, dict):
        for target, v in lift_block.items():
            if target == 'content_probe' or not isinstance(v, dict):
                continue
            lift_list.append(
                {
                    'target': target,
                    'metric': v.get('metric'),
                    'zte': v.get('zte_linear'),
                    'raw': v.get('raw_linear'),
                    'noise': v.get('noise_linear'),
                    'lift_linear': v.get('lift_linear'),
                    'lift_knn': v.get('lift_knn'),
                    'is_content': bool(v.get('is_content')),
                    'is_identity': bool(v.get('is_identity')),
                }
            )

    views: dict[str, Any] = {}
    held = _view(
        'Held-out (new brain)',
        _first(board, _HELD_GEOMETRY_KEYS),
        _first(board, _HELD_RETRIEVAL_KEYS),
    )
    if held is not None:
        views['held_out'] = held
    insample = _view(
        'In-sample', _first(board, _INSAMPLE_GEOMETRY_KEYS), _first(board, _INSAMPLE_RETRIEVAL_KEYS)
    )
    if insample is not None:
        views['in_sample'] = insample

    payload = {
        'run_name': run_name,
        'is_loso': bool(board.get('is_loso')),
        'holdout_subject': board.get('holdout_subject'),
        'factored': bool(board.get('factored')),
        'view_order': [k for k in ('held_out', 'in_sample') if k in views],
        'views': views,
        'lift': lift_list,
        'content_probe': content_probe,
    }
    return _clean(payload)


def _view(label: str, geometry: Any, retrieval: Any) -> dict[str, Any] | None:
    """Wraps a geometry/retrieval pair into a view, or `None` if both are absent."""
    geometry = geometry if isinstance(geometry, dict) else None
    retrieval = retrieval if isinstance(retrieval, dict) else None
    if geometry is None and retrieval is None:
        return None
    return {'label': label, 'geometry': geometry, 'retrieval': retrieval}


def _first(board: dict, keys: tuple[str, ...]) -> Any:
    """Returns the first present, non-`None` value among `keys`."""
    for k in keys:
        v = board.get(k)
        if v is not None:
            return v
    return None


def _clean(obj: Any) -> Any:
    """Recursively coerces to JSON-safe types, rounding floats and dropping non-finite values."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        f = float(obj)
        return round(f, 5) if math.isfinite(f) else None
    return str(obj)


def _esc(text: str) -> str:
    """Minimal HTML escaping for text substituted into the template."""
    return (
        str(text)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
    )


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__ — held-out scoreboard</title>
<style>
:root{
  --bg:#f7f8fb; --panel:#ffffff; --ink:#131720; --muted:#5c6773; --line:#e3e7ef;
  --good:#0f9d6b; --goodbg:#e3f6ee; --warn:#e29008; --warnbg:#fbf0dc; --bad:#e5484d; --badbg:#fce9ea;
  --accent:#5a4bff; --accent2:#ff4d8d; --track:#eef1f6;
  --shadow:0 1px 3px rgba(20,25,34,.08),0 8px 24px rgba(20,25,34,.06);
}
:root[data-theme="dark"]{
  --bg:#0e1117; --panel:#161b26; --ink:#eef2f9; --muted:#aab4c6; --line:#232a37;
  --good:#3ddc97; --goodbg:#123227; --warn:#f0b95e; --warnbg:#33280f; --bad:#ff6b6f; --badbg:#331a1d;
  --accent:#9a86ff; --accent2:#ff77a9; --track:#1b2230;
  --shadow:0 1px 3px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0e1117; --panel:#161b26; --ink:#eef2f9; --muted:#aab4c6; --line:#232a37;
    --good:#3ddc97; --goodbg:#123227; --warn:#f0b95e; --warnbg:#33280f; --bad:#ff6b6f; --badbg:#331a1d;
    --accent:#9a86ff; --accent2:#ff77a9; --track:#1b2230;
    --shadow:0 1px 3px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1040px;margin:0 auto;padding:26px 20px 72px}
header{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap}
h1{font-size:22px;margin:0;letter-spacing:-.01em}
.verdict{color:var(--muted);font-size:13.5px;margin-top:5px;max-width:660px}
.themebtn{border:1px solid var(--line);background:var(--panel);color:var(--ink);
  border-radius:999px;padding:7px 13px;cursor:pointer;font-size:13px}
.themebtn:hover{border-color:color-mix(in srgb,var(--accent) 45%,var(--line))}
.section{margin-top:22px}
.section h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  margin:0 0 10px 2px;font-weight:700}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  box-shadow:var(--shadow);padding:16px}
.empty{color:var(--muted);text-align:center;padding:40px}

/* content-probe positive control banner */
.probe{display:flex;align-items:flex-start;gap:10px;border-radius:12px;padding:11px 14px;
  margin-top:14px;font-size:12.5px;border:1px solid var(--line);background:var(--panel);box-shadow:var(--shadow)}
.probe-pill{font-weight:800;border-radius:999px;padding:2px 9px;font-size:11px;white-space:nowrap}
.probe.ok .probe-pill{background:var(--goodbg);color:var(--good)}
.probe.no .probe-pill{background:var(--badbg);color:var(--bad)}

/* segmented view control */
.seg{display:inline-flex;gap:4px;background:var(--panel);border:1px solid var(--line);
  border-radius:999px;padding:4px;margin-top:16px}
.segbtn{border:none;background:transparent;color:var(--muted);border-radius:999px;
  padding:6px 14px;cursor:pointer;font-size:12.5px;font-weight:600}
.segbtn.on{background:linear-gradient(90deg,var(--accent),var(--accent2));color:#fff}

/* stat cards / gauges */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.statcard{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);
  padding:15px 16px;outline:none;transition:border-color .18s,transform .18s}
.statcard:hover,.statcard:focus{border-color:color-mix(in srgb,var(--accent) 45%,var(--line));transform:translateY(-2px)}
.stat-top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.stat-label{font-weight:650;font-size:13px}
.stat-tag{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  border:1px solid var(--line);border-radius:999px;padding:1px 7px;white-space:nowrap}
.stat-value{font-size:26px;font-weight:750;letter-spacing:-.02em;margin:8px 0 2px;font-variant-numeric:tabular-nums}
.c-good .stat-value{color:var(--good)} .c-warn .stat-value{color:var(--warn)}
.c-bad .stat-value{color:var(--bad)} .c-na .stat-value{color:var(--muted)}
.meter{position:relative;height:12px;border-radius:6px;background:var(--track);overflow:hidden;margin:6px 0 8px}
.fill{position:absolute;top:0;bottom:0;border-radius:6px;transition:width .9s cubic-bezier(.22,.61,.36,1)}
.fill.good{background:var(--good)} .fill.warn{background:var(--warn)} .fill.bad{background:var(--bad)}
.meter .tick{position:absolute;top:0;bottom:0;width:2px;background:var(--ink);opacity:.55}
.stat-ref{font-size:11px;color:var(--muted)}
.stat-why{font-size:12px;line-height:1.45;color:var(--muted);max-height:0;opacity:0;overflow:hidden;
  transition:max-height .25s ease,opacity .25s ease,margin .25s ease}
.statcard:hover .stat-why,.statcard:focus .stat-why{max-height:180px;opacity:1;margin-top:8px}

/* lift-over-raw lollipop comparison */
.lift{display:flex;flex-direction:column;gap:9px}
.liftrow{display:grid;grid-template-columns:minmax(104px,150px) 1fr 62px;align-items:center;gap:10px}
.lift-label{font-size:12.5px;font-weight:600;display:flex;flex-direction:column;min-width:0}
.lift-label>span.nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lift-kind{font-size:10px;color:var(--muted);font-weight:500}
.lift-track{position:relative;height:14px;background:var(--track);border-radius:7px}
.lift-zero{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--muted);opacity:.7}
.lift-bar{position:absolute;top:3px;height:8px;border-radius:4px;transition:width .9s cubic-bezier(.22,.61,.36,1)}
.lift-bar.good{background:var(--good)} .lift-bar.bad{background:var(--bad)} .lift-bar.warn{background:var(--warn)}
.lift-dot{position:absolute;top:1px;width:12px;height:12px;border-radius:50%;transform:translateX(-50%);border:2px solid var(--panel)}
.lift-dot.good{background:var(--good)} .lift-dot.bad{background:var(--bad)} .lift-dot.warn{background:var(--warn)}
.lift-val{text-align:right;font-variant-numeric:tabular-nums;font-weight:650;font-size:12.5px}
.lift-val.good{color:var(--good)} .lift-val.bad{color:var(--bad)} .lift-val.warn{color:var(--warn)}
.lift-legend{margin-top:14px;color:var(--muted);font-size:11.5px;display:flex;gap:18px;flex-wrap:wrap}
@media (max-width:560px){ .liftrow{grid-template-columns:1fr} .lift-val{text-align:left} }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1 id="run"></h1>
      <div class="verdict" id="verdict"></div>
    </div>
    <button class="themebtn" id="themebtn" aria-label="Toggle light / dark theme">◐ Theme</button>
  </header>

  <div class="probe" id="probe" style="display:none"></div>
  <div class="seg" id="seg" style="display:none"></div>

  <div class="section" id="cardsSection">
    <h2 id="cardsHead">Headline metrics</h2>
    <div class="grid" id="grid"></div>
  </div>

  <div class="section" id="liftSection">
    <h2>Lift over the raw band-power control (ZTE − raw)</h2>
    <div class="card">
      <div class="lift" id="liftBody"></div>
      <div class="lift-legend" id="liftLegend"></div>
    </div>
  </div>

  <div class="section" id="emptySection" style="display:none">
    <div class="card empty">No scoreboard data was available for this run.</div>
  </div>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const P = JSON.parse(document.getElementById('data').textContent);
const root = document.documentElement;

/* ---------- theme (persisted) ---------- */
const TKEY='zte-scoreboard-theme';
try{ const s=localStorage.getItem(TKEY); if(s==='dark'||s==='light') root.setAttribute('data-theme',s); }catch(e){}
document.getElementById('themebtn').onclick=()=>{
  const cur = root.getAttribute('data-theme') ||
    (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const next = cur==='dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  try{ localStorage.setItem(TKEY, next); }catch(e){}
};

/* ---------- helpers ---------- */
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function nv(x){ return (x==null || (typeof x==='number' && !isFinite(x))) ? null : x; }
function clamp(x,a,b){ return Math.max(a,Math.min(b,x)); }
function num(v,f){
  if(v==null || (typeof v==='number' && !isFinite(v))) return '—';
  if(f==='ratio')  return (+v).toFixed(3);
  if(f==='signed') return (v>=0?'+':'')+(+v).toFixed(3);
  if(f==='pct')    return ((+v)*100).toFixed(1)+'%';
  if(f==='count')  return String(Math.round(v));
  return String(v);
}

/* ---------- lift lookups (view-independent: probes run over all embeddings) ---------- */
function bestContentLift(){
  let best=null;
  (P.lift||[]).forEach(l=>{ if(l.is_content && nv(l.lift_linear)!=null) best = best==null ? l.lift_linear : Math.max(best,l.lift_linear); });
  return best;
}
function identityLift(){
  let val=null;
  (P.lift||[]).forEach(l=>{ if(l.is_identity && nv(l.lift_linear)!=null && val==null) val=l.lift_linear; });
  return val;
}

/* ---------- card specs: each names its own reference line ---------- */
const CARDS=[
  {key:'erank', label:'Effective-rank ratio', tag:'view', src:'geometry', field:'effective_rank_ratio', fmt:'ratio',
   refLabel:'healthy > 0.10',
   scale:v=>({min:0, max:Math.max((v||0)*1.3, 0.30), ref:0.10}),
   ev:v=> v==null?'na': v>=0.10?'good': v>=0.05?'warn':'bad',
   why:'Share of the embedding dimensions the space actually spends on the held-out brain. Above the 0.10 healthy line means it has not collapsed onto a handful of directions.'},
  {key:'aniso', label:'Anisotropy', tag:'view', src:'geometry', field:'anisotropy', fmt:'ratio',
   refLabel:'degenerate cone at 1.0 — lower is better',
   scale:v=>({min:0, max:1.0, ref:1.0}),
   ev:v=> v==null?'na': v<=0.5?'good': v<=0.85?'warn':'bad',
   why:'How cone-shaped the cloud is. Near 1.0 is a degenerate cone that can look high-rank yet carry almost no usable structure.'},
  {key:'clift', label:'Content lift over raw', tag:'all', src:'lift-content', fmt:'signed',
   refLabel:'raw band-power line at 0 — sign is the story',
   scale:v=>{const m=Math.max(0.05,Math.abs(v||0)*1.3); return {min:-m, max:m, ref:0};},
   ev:v=> v==null?'na': v>0?'good': v>=-0.005?'warn':'bad',
   why:'Best margin by which the encoder reads word content better than the raw band-power control (linear R²). Negative means the raw control wins and the encoder has not earned its place.'},
  {key:'ident', label:'Identity leak vs raw', tag:'all', src:'lift-identity', fmt:'signed',
   refLabel:'raw band-power line at 0 — want below',
   scale:v=>{const m=Math.max(0.05,Math.abs(v||0)*1.3); return {min:-m, max:m, ref:0};},
   ev:v=> v==null?'na': v<0?'good': v<=0.005?'warn':'bad',
   why:'How much more (positive) or less (negative) the code exposes WHO read the sentence than the raw control. Below the 0 line means it leaks less identity than raw — what we want.'},
  {key:'top1', label:'Cross-subject retrieval Top-1', tag:'view', src:'retrieval', field:'top1', fmt:'pct',
   refField:'chance_top1', refLabel:'chance line',
   scale:(v,ref)=>({min:0, max:Math.max((v||0)*1.6,(ref||0)*1.6,0.10), ref:ref||0}),
   ev:(v,ref)=> v==null?'na': (ref==null ? (v>0?'warn':'bad') : (v-ref>=0.02?'good': v-ref>=-0.005?'warn':'bad')),
   why:'Given a held-out person’s sentence, how often the closest match among other people shares the same stimulus — judged against its own chance line. This is the north-star capability.'},
  {key:'rankpct', label:'Rank percentile', tag:'view', src:'retrieval', field:'rank_percentile', fmt:'pct',
   refLabel:'1.0 = correct match ranked first',
   scale:v=>({min:0, max:1.0, ref:1.0}),
   ev:v=> v==null?'na': v>=0.7?'good': v>=0.5?'warn':'bad',
   why:'Average position of the correct cross-subject match in the ranked gallery of other people. 1.0 means it is always first; 0.5 is no better than random ordering.'},
];

function cardValue(c, view){
  if(c.src==='geometry')  return view && view.geometry  ? nv(view.geometry[c.field])  : null;
  if(c.src==='retrieval') return view && view.retrieval ? nv(view.retrieval[c.field]) : null;
  if(c.src==='lift-content')  return bestContentLift();
  if(c.src==='lift-identity') return identityLift();
  return null;
}
function cardRef(c, view){
  if(c.refField && view && view.retrieval) return nv(view.retrieval[c.refField]);
  return null;
}

/* ---------- meter (animated fill + named reference tick) ---------- */
function meterHTML(value, sc, cls){
  const span=(sc.max-sc.min)||1;
  const pos=x=>clamp((x-sc.min)/span*100,0,100);
  let h='<div class="meter">';
  if(value!=null){
    const base = sc.min<0 ? 0 : sc.min;
    const a=pos(Math.min(base,value)), b=pos(Math.max(base,value));
    h+='<div class="fill '+cls+'" style="left:'+a.toFixed(1)+'%;width:0" data-w="'+Math.max(0,b-a).toFixed(1)+'"></div>';
  }
  if(sc.ref!=null) h+='<div class="tick" style="left:'+pos(sc.ref).toFixed(1)+'%"></div>';
  h+='</div>';
  return h;
}

/* ---------- render: stat cards ---------- */
const state={ view: (P.view_order && P.view_order[0]) || null };
function renderCards(){
  const view = state.view ? P.views[state.view] : null;
  let h='';
  CARDS.forEach(c=>{
    const val=cardValue(c,view), ref=cardRef(c,view);
    const cls=c.ev(val,ref);
    const sc=c.scale(val,ref);
    const tag = c.tag==='all' ? 'all subjects' : (view ? view.label : '—');
    const refExtra = (c.refField && ref!=null) ? ' ('+num(ref,c.fmt)+')' : '';
    h+='<div class="statcard c-'+cls+'" tabindex="0">'
      +'<div class="stat-top"><span class="stat-label">'+esc(c.label)+'</span>'
        +'<span class="stat-tag">'+esc(tag)+'</span></div>'
      +'<div class="stat-value">'+num(val,c.fmt)+'</div>'
      + meterHTML(val,sc,cls)
      +'<div class="stat-ref">'+esc(c.refLabel)+refExtra+'</div>'
      +'<div class="stat-why">'+esc(c.why)+'</div>'
      +'</div>';
  });
  document.getElementById('grid').innerHTML=h;
  animate();
}

/* ---------- render: lift lollipops ---------- */
function renderLift(){
  const lifts=(P.lift||[]).filter(l=>nv(l.lift_linear)!=null);
  const sec=document.getElementById('liftSection');
  if(!lifts.length){ sec.style.display='none'; return; }
  const m=Math.max(0.02, ...lifts.map(l=>Math.abs(l.lift_linear)));
  const pos=x=>clamp((x/m+1)/2*100,0,100);
  let h='';
  lifts.forEach(l=>{
    const good = l.is_identity ? (l.lift_linear<0) : (l.lift_linear>0);
    const cls  = Math.abs(l.lift_linear)<1e-9 ? 'warn' : (good?'good':'bad');
    const kind = l.is_content ? 'content ▲ (want +)' : l.is_identity ? 'identity ▼ (want −)' : '—';
    const zero=50, p=pos(l.lift_linear), a=Math.min(zero,p), b=Math.max(zero,p);
    h+='<div class="liftrow">'
      +'<div class="lift-label"><span class="nm">'+esc(l.target)+'</span>'
        +'<span class="lift-kind">'+esc(kind)+'</span></div>'
      +'<div class="lift-track"><div class="lift-zero"></div>'
        +'<div class="lift-bar '+cls+'" style="left:'+a.toFixed(1)+'%;width:0" data-w="'+(b-a).toFixed(1)+'"></div>'
        +'<div class="lift-dot '+cls+'" style="left:'+p.toFixed(1)+'%"></div></div>'
      +'<div class="lift-val '+cls+'">'+num(l.lift_linear,'signed')+'</div>'
      +'</div>';
  });
  document.getElementById('liftBody').innerHTML=h;
  document.getElementById('liftLegend').innerHTML=
    '<span>content ▲ wants a positive lift</span>'
    +'<span>identity ▼ wants a negative lift</span>'
    +'<span>bar and dot mark ZTE − raw against the 0 control line</span>';
  animate();
}

/* ---------- animate every meter/bar fill on paint ---------- */
function animate(){
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    document.querySelectorAll('[data-w]').forEach(f=>{ f.style.width=f.getAttribute('data-w')+'%'; });
  }));
}

/* ---------- render: segmented view control ---------- */
function renderSeg(){
  const seg=document.getElementById('seg');
  if(!P.view_order || P.view_order.length<2){ seg.style.display='none'; return; }
  seg.style.display='inline-flex';
  let h='';
  P.view_order.forEach(k=>{
    h+='<button class="segbtn'+(k===state.view?' on':'')+'" data-v="'+esc(k)+'">'+esc(P.views[k].label)+'</button>';
  });
  seg.innerHTML=h;
  seg.querySelectorAll('.segbtn').forEach(b=>b.onclick=()=>{
    state.view=b.getAttribute('data-v'); renderSeg(); renderCards();
  });
}

/* ---------- render: content-probe positive control ---------- */
function renderProbe(){
  const cp=P.content_probe, el=document.getElementById('probe');
  if(!cp){ el.style.display='none'; return; }
  const pass=!!cp.passes;
  el.style.display='flex';
  el.className='probe '+(pass?'ok':'no');
  el.innerHTML='<span class="probe-pill">'+(pass?'PASS':'FAIL')+'</span>'
    +'<span>Content-probe positive control — raw band-power reads lexical content at R² '
    + num(nv(cp.raw_content_r2_best),'ratio') +' (floor '+ num(nv(cp.floor),'ratio') +'). '
    + (pass ? 'The probe can detect content, so a 0% content budget is a real absence.'
            : 'The probe cannot read content even from raw features — treat any “content 0%” as untrustworthy until this is fixed.')
    +'</span>';
}

/* ---------- honest one-line verdict, derived from the numbers ---------- */
function verdict(){
  const parts=[];
  const v = state.view ? P.views[state.view] : null;
  const g = v && v.geometry, r = v && v.retrieval;
  if(g && (nv(g.effective_rank_ratio)!=null || nv(g.anisotropy)!=null)){
    const err=nv(g.effective_rank_ratio), an=nv(g.anisotropy);
    const healthy = (err==null || err>=0.10) && (an==null || an<=0.85);
    parts.push(healthy ? 'healthy held-out space' : 'held-out space near collapse');
  }
  const cl=bestContentLift();
  if(cl!=null) parts.push(cl>0.005 ? 'content above raw' : (cl<-0.005 ? 'content below raw' : 'content at raw'));
  if(r){
    let lift=nv(r.lift_top1);
    if(lift==null && nv(r.top1)!=null && nv(r.chance_top1)!=null) lift=r.top1-r.chance_top1;
    if(lift!=null) parts.push(lift>0.005 ? 'cross-subject retrieval above chance'
      : (lift<-0.005 ? 'cross-subject retrieval below chance' : 'cross-subject retrieval at chance'));
  }
  if(!parts.length) return 'Scoreboard summary.';
  const s=parts.join(', ');
  let out=s.charAt(0).toUpperCase()+s.slice(1)+'.';
  if(P.is_loso && P.holdout_subject) out+=' Away game on held-out subject '+P.holdout_subject+'.';
  return out;
}

/* ---------- boot ---------- */
(function(){
  document.getElementById('run').textContent = (P.run_name||'ZTE run') + ' — held-out scoreboard';
  const hasViews = P.view_order && P.view_order.length;
  const hasLift  = P.lift && P.lift.length;
  const hasAny   = hasViews || hasLift || P.content_probe;
  if(!hasAny){
    document.getElementById('verdict').textContent = 'No scoreboard numbers were available for this run.';
    document.getElementById('cardsSection').style.display='none';
    document.getElementById('liftSection').style.display='none';
    document.getElementById('emptySection').style.display='';
    return;
  }
  document.getElementById('cardsHead').textContent =
    hasViews ? (P.views[state.view].label + ' headline metrics') : 'Headline metrics';
  document.getElementById('verdict').textContent = verdict();
  renderProbe();
  renderSeg();
  renderCards();
  renderLift();
})();
</script>
</body>
</html>
"""
