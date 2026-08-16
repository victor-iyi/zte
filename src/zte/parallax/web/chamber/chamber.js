"use strict";

const RAW = JSON.parse(document.getElementById("chamber-data").textContent || "{}");
const D = RAW.data || {};
const PROV = RAW.provenance || null;

const TASKS = Array.isArray(D.tasks) ? D.tasks : [];
const POINTS = D.points || {};
const TRANSFER = D.transfer || {};
const CAPACITY = D.capacity || {};
const CKA = D.cka || {};

/* First three categorical slots (all-pairs CVD-safe): one fixed hue per task, never re-assigned. */
const TASK_COLOR = { NR: "#3987e5", SR: "#d95926", TSR: "#199e70" };
/* Cluster hues cycle past eight; identity precision there is carried by hover text, not hue alone. */
const CLUSTER = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const INK = "#eaf0fb";
const MUTED = "#93a0ba";
const GRID = "rgba(147,160,186,0.14)";
const FONT = 'ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif';
const CONFIG = { displayModeBar: false, responsive: true };
const BASE = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: FONT, color: MUTED, size: 12 },
};
const REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const fin = (x) => typeof x === "number" && isFinite(x);
const fmt = (x, d) => (fin(x) ? Number(x).toFixed(d == null ? 3 : d) : "n/a");
const taskColor = (t) => TASK_COLOR[t] || "#9085e9";
const esc = (s) =>
  String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function emptyPanel(hostId, msg) {
  const host = document.getElementById(hostId);
  if (host) host.innerHTML = `<div class="emptycell">${esc(msg)}</div>`;
}

/* ---- header ---- */

document.getElementById("sub").textContent =
  "Three vantage points on the same minds — one encoder per task, closed-set retrieval readout.";
document.getElementById("holdoutpill").textContent = `held-out subject ${D.holdout || "?"}`;

/* ---- panel 1: the parallax view ---- */

function buildParallax() {
  const evalTasks = TASKS.filter((t) => Array.isArray(POINTS[t]) && POINTS[t].length);
  if (!evalTasks.length) {
    emptyPanel("parallax3d", "No sentence points in this report yet.");
    return;
  }

  const host = document.getElementById("parallax3d");
  let evalTask = evalTasks[0];
  let vantage = null;
  let trails = false;
  let vantages = [];
  let pts = [];
  let cur = [];
  let animId = 0;

  function prep() {
    const raw = POINTS[evalTask] || [];
    vantages = TASKS.filter((m) => raw.some((p) => p.views && Array.isArray(p.views[m])));
    /* Only points seen by every available vantage stay, so a vantage switch morphs a consistent set. */
    pts = raw.filter((p) => vantages.every((m) => p.views && Array.isArray(p.views[m])));
    if (!vantages.includes(vantage)) vantage = vantages[0];
    cur = pts.map((p) => p.views[vantage].slice(0, 3));
  }

  function hoverText() {
    return pts.map((p) => {
      const rp = p.rank_percentile || {};
      const per = vantages.map((m) => `${m} ${fmt(rp[m], 2)}`).join(" · ");
      return `${esc(p.text)}<br>${fin(p.n_words) ? p.n_words : "?"} words · rank pctl ${per}`;
    });
  }

  function trailCoords() {
    const xs = [], ys = [], zs = [];
    for (const p of pts) {
      for (const m of vantages) {
        const v = p.views[m];
        xs.push(v[0]); ys.push(v[1]); zs.push(v[2]);
      }
      xs.push(null); ys.push(null); zs.push(null);
    }
    return { xs, ys, zs };
  }

  function draw() {
    const t = trailCoords();
    Plotly.react(
      host,
      [
        {
          type: "scatter3d",
          mode: "lines",
          x: t.xs, y: t.ys, z: t.zs,
          line: { color: "rgba(157,139,255,0.28)", width: 1.6 },
          hoverinfo: "skip",
          visible: trails,
          name: "trails",
        },
        {
          type: "scatter3d",
          mode: "markers",
          x: cur.map((v) => v[0]),
          y: cur.map((v) => v[1]),
          z: cur.map((v) => v[2]),
          marker: {
            size: pts.map((p) => 3.5 + Math.min(18, fin(p.n_words) ? p.n_words : 4) * 0.5),
            color: pts.map((p) => CLUSTER[(fin(p.cluster) ? p.cluster : 0) % CLUSTER.length]),
            opacity: 0.88,
            line: { width: 0 },
          },
          text: hoverText(),
          hovertemplate: "%{text}<extra></extra>",
          name: "sentences",
        },
      ],
      {
        ...BASE,
        margin: { l: 0, r: 0, t: 0, b: 0 },
        showlegend: false,
        hoverlabel: { bgcolor: "#101a30", bordercolor: "rgba(255,255,255,0.15)", font: { color: INK, size: 12 } },
        scene: {
          bgcolor: "rgba(0,0,0,0)",
          xaxis: { visible: false },
          yaxis: { visible: false },
          zaxis: { visible: false },
          aspectmode: "data",
          camera: { eye: { x: 1.55, y: 1.15, z: 0.85 } },
        },
      },
      CONFIG
    );
  }

  function setVantage(next) {
    if (next === vantage || !vantages.includes(next)) return;
    vantage = next;
    paintButtons();
    const from = cur.map((v) => v.slice());
    const to = pts.map((p) => p.views[vantage]);
    if (REDUCED) {
      cur = to.map((v) => v.slice(0, 3));
      Plotly.restyle(host, { x: [cur.map((v) => v[0])], y: [cur.map((v) => v[1])], z: [cur.map((v) => v[2])] }, [1]);
      return;
    }
    const t0 = performance.now(), dur = 750, id = ++animId;
    const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
    function step(now) {
      if (id !== animId) return;
      const u = Math.min(1, (now - t0) / dur);
      const e = ease(u);
      cur = from.map((f, i) => [0, 1, 2].map((k) => f[k] + (to[i][k] - f[k]) * e));
      Plotly.restyle(host, { x: [cur.map((v) => v[0])], y: [cur.map((v) => v[1])], z: [cur.map((v) => v[2])] }, [1]);
      if (u < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function paintButtons() {
    const van = document.getElementById("vantage");
    van.innerHTML = "";
    for (const m of vantages) {
      const b = document.createElement("button");
      b.textContent = m;
      b.className = m === vantage ? "on" : "";
      b.style.color = m === vantage ? taskColor(m) : "";
      b.addEventListener("click", () => setVantage(m));
      van.appendChild(b);
    }
    const set = document.getElementById("sentset");
    set.innerHTML = "";
    for (const e of evalTasks) {
      const b = document.createElement("button");
      b.textContent = e;
      b.className = e === evalTask ? "on" : "";
      b.addEventListener("click", () => {
        if (e === evalTask) return;
        evalTask = e;
        animId += 1;
        prep();
        paintButtons();
        draw();
      });
      set.appendChild(b);
    }
  }

  const chip = document.getElementById("trails");
  chip.addEventListener("click", () => {
    trails = !trails;
    chip.classList.toggle("on", trails);
    Plotly.restyle(host, { visible: trails }, [0]);
  });

  prep();
  paintButtons();
  draw();
}

/* ---- panel 2: transfer flow ---- */

function buildFlow() {
  const trains = TASKS.filter((t) => TRANSFER[t] && Object.keys(TRANSFER[t]).length);
  if (!trains.length) {
    emptyPanel("sankey", "No transfer cells in this report yet.");
    return;
  }
  const evals = TASKS.filter((e) => trains.some((t) => TRANSFER[t][e]));

  const labels = [], colors = [], idx = {};
  for (const t of trains) {
    idx["m" + t] = labels.length;
    labels.push(`${t} model`);
    colors.push(taskColor(t));
  }
  for (const e of evals) {
    idx["s" + e] = labels.length;
    labels.push(`${e} sentences`);
    colors.push(hexA(taskColor(e), 0.45));
  }

  const src = [], dst = [], val = [], lcol = [], custom = [];
  for (const t of trains) {
    for (const e of evals) {
      const c = TRANSFER[t][e];
      if (!c) continue;
      const rp = fin(c.rank_percentile) ? c.rank_percentile : null;
      const lift = Math.max(0, (rp == null ? 0.5 : rp) - 0.5);
      const diag = t === e;
      src.push(idx["m" + t]);
      dst.push(idx["s" + e]);
      /* A tiny floor keeps a null lift drawable as a hairline instead of vanishing. */
      val.push(Math.max(lift, 0.004));
      lcol.push(diag ? hexA(taskColor(t), 0.16 + Math.min(0.7, lift * 3)) : hexA("#9d8bff", 0.14 + Math.min(0.7, lift * 3)));
      const ci = Array.isArray(c.ci) ? `[${fmt(c.ci[0])}, ${fmt(c.ci[1])}]` : "n/a";
      const kind = diag ? "in-task · new subject, familiar stimuli" : "new subject × new sentences";
      custom.push(
        `${t} → ${e} · ${kind}<br>rank pctl ${fmt(rp)} · 95% CI ${ci}<br>` +
          `Top-1 ${fmt(c.top1, 4)} vs chance ${fmt(c.chance, 4)} · ${fin(c.n_seeds) ? c.n_seeds : "?"} seed(s)`
      );
    }
  }

  Plotly.newPlot(
    "sankey",
    [
      {
        type: "sankey",
        arrangement: "snap",
        node: {
          label: labels,
          color: colors,
          pad: 24,
          thickness: 14,
          line: { width: 0 },
          hoverinfo: "none",
        },
        link: { source: src, target: dst, value: val, color: lcol, customdata: custom, hovertemplate: "%{customdata}<extra></extra>" },
      },
    ],
    {
      ...BASE,
      height: 320,
      margin: { l: 8, r: 8, t: 12, b: 12 },
      hoverlabel: { bgcolor: "#101a30", bordercolor: "rgba(255,255,255,0.15)", font: { color: INK, size: 12 } },
    },
    CONFIG
  );
}

/* ---- panel 3: capacity dials ---- */

function buildDials() {
  const host = document.getElementById("dials");
  const trains = TASKS.filter((t) => CAPACITY[t] && typeof CAPACITY[t] === "object");
  if (!trains.length) {
    emptyPanel("dials", "No menu-capacity audit in this report yet.");
    return;
  }

  const top = Math.max(
    5,
    Math.ceil(Math.log2(Math.max(2, ...trains.map((t) => (fin(CAPACITY[t].k_at_target) ? CAPACITY[t].k_at_target : 2)))))
  );
  const ticks = [];
  for (let i = 1; i <= top; i++) ticks.push(i);

  /* All cells enter the grid before any plot is drawn: the auto-fit tracks must settle first, because a
     gauge drawn while its cell still spans the whole row keeps that width and bleeds into its neighbours. */
  const cells = trains.map((t) => {
    const cap = CAPACITY[t];
    const certified = fin(cap.k_at_target) && cap.k_at_target >= 2;
    const cell = document.createElement("div");
    cell.className = "dial";
    cell.innerHTML =
      `<div class="dialname">${esc(t)} model</div><div class="dialplot"></div>` +
      `<div class="dialvalue ${certified ? "" : "none"}">${certified ? "K = " + cap.k_at_target : "none certified"}</div>` +
      `<div class="dialsub">K=2 accuracy ${fmt(cap.k2_accuracy, 2)}</div>`;
    host.appendChild(cell);
    return cell;
  });

  trains.forEach((t, i) => {
    const cell = cells[i];
    const k = CAPACITY[t].k_at_target;
    const certified = fin(k) && k >= 2;

    Plotly.newPlot(
      cell.querySelector(".dialplot"),
      [
        {
          type: "indicator",
          mode: "gauge",
          value: certified ? Math.log2(k) : 0,
          gauge: {
            axis: {
              range: [0, top],
              tickvals: ticks,
              ticktext: ticks.map((i) => String(Math.pow(2, i))),
              tickcolor: MUTED,
              tickfont: { size: 10, color: MUTED },
            },
            bar: { color: certified ? taskColor(t) : "rgba(0,0,0,0)", thickness: 0.55 },
            bgcolor: "rgba(255,255,255,0.05)",
            borderwidth: 0,
          },
        },
      ],
      { ...BASE, margin: { l: 18, r: 18, t: 12, b: 4 }, height: 130 },
      CONFIG
    );
  });
}

/* ---- panel 4: percentile rain ---- */

function buildRain() {
  const traces = [];
  for (const t of TASKS) {
    for (const e of TASKS) {
      const ps = POINTS[e];
      if (!Array.isArray(ps) || !ps.length) continue;
      const ys = [], texts = [];
      for (const p of ps) {
        const rp = p.rank_percentile && p.rank_percentile[t];
        if (fin(rp)) {
          ys.push(rp);
          texts.push(esc(p.text));
        }
      }
      if (!ys.length) continue;
      const diag = t === e;
      const col = diag ? taskColor(t) : "#9085e9";
      traces.push({
        type: "violin",
        y: ys,
        name: `${t}→${e}`,
        text: texts,
        hovertemplate: `%{text}<br>rank pctl %{y:.3f}<extra>${t}→${e}</extra>`,
        points: "all",
        pointpos: 0,
        jitter: 0.55,
        marker: { size: 3.4, color: hexA(col, 0.85) },
        line: { color: hexA(col, diag ? 0.95 : 0.7), width: 1.4 },
        fillcolor: hexA(col, 0.12),
        spanmode: "hard",
        width: 0.85,
        showlegend: false,
      });
    }
  }
  if (!traces.length) {
    emptyPanel("rain", "No per-sentence percentiles in this report yet.");
    return;
  }

  Plotly.newPlot("rain", traces, {
    ...BASE,
    height: 360,
    margin: { l: 52, r: 18, t: 10, b: 40 },
    hoverlabel: { bgcolor: "#101a30", bordercolor: "rgba(255,255,255,0.15)", font: { color: INK, size: 12 } },
    yaxis: {
      title: { text: "rank percentile (1 = retrieved first)", font: { size: 11 } },
      range: [-0.03, 1.03],
      gridcolor: GRID,
      zeroline: false,
    },
    xaxis: { tickfont: { size: 11 } },
    shapes: [
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        y0: 0.5,
        y1: 0.5,
        line: { color: "rgba(255,107,166,0.65)", dash: "dash", width: 1 },
      },
    ],
    annotations: [
      {
        xref: "paper",
        x: 0.998,
        xanchor: "right",
        y: 0.5,
        yanchor: "bottom",
        text: "chance",
        showarrow: false,
        font: { color: "#ff6ba6", size: 11 },
      },
    ],
  }, CONFIG);
}

/* ---- panel 5: CKA triad ---- */

function buildTriad() {
  const measured = (t) =>
    (TRANSFER[t] && Object.keys(TRANSFER[t]).length) ||
    (Array.isArray(POINTS[t]) && POINTS[t].length) ||
    Object.entries(CKA).some(([k, v]) => fin(v) && k.split("|").includes(t));
  const present = TASKS.filter(measured);
  const svg = document.getElementById("triad");
  if (present.length < 2) {
    svg.outerHTML = '<div class="emptycell">Fewer than two models measured — no pairwise geometry yet.</div>';
    return;
  }

  const cx = 190, cy = 172, R = 118;
  const pos = {};
  present.forEach((t, i) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / present.length;
    pos[t] = [cx + R * Math.cos(a), cy + R * Math.sin(a)];
  });

  let inner = "";
  for (let i = 0; i < present.length; i++) {
    for (let j = i + 1; j < present.length; j++) {
      const a = present[i], b = present[j];
      const v = fin(CKA[`${a}|${b}`]) ? CKA[`${a}|${b}`] : CKA[`${b}|${a}`];
      const [x1, y1] = pos[a], [x2, y2] = pos[b];
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      /* Push the label off the edge along its perpendicular so thick edges never swallow it. */
      const len = Math.hypot(x2 - x1, y2 - y1) || 1;
      const nx = (-(y2 - y1) / len) * 16, ny = ((x2 - x1) / len) * 16;
      if (fin(v)) {
        const w = 1.5 + Math.max(0, Math.min(1, v)) * 14;
        inner +=
          `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="rgba(157,139,255,${0.25 + 0.55 * v})" ` +
          `stroke-width="${w.toFixed(1)}" stroke-linecap="round"><title>CKA(${a}, ${b}) = ${fmt(v)}</title></line>` +
          `<text class="tlabel" x="${mx + nx}" y="${my + ny}">${fmt(v, 2)}</text>`;
      } else {
        inner +=
          `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="rgba(147,160,186,0.35)" stroke-width="1" ` +
          `stroke-dasharray="5 5"></line><text class="tlabel" x="${mx + nx}" y="${my + ny}">n/a</text>`;
      }
    }
  }
  for (const t of present) {
    const [x, y] = pos[t];
    inner +=
      `<circle cx="${x}" cy="${y}" r="27" fill="${hexA(taskColor(t), 0.2)}" stroke="${taskColor(t)}" stroke-width="2"></circle>` +
      `<text class="tnode" x="${x}" y="${y + 5}">${esc(t)}</text>`;
  }
  svg.innerHTML = inner;
}

/* ---- provenance ---- */

function buildProv() {
  const el = document.getElementById("prov");
  const bits = [];
  if (D.holdout) bits.push(`holdout ${D.holdout}`);
  if (PROV && Array.isArray(PROV.seeds) && PROV.seeds.length) bits.push(`seeds ${PROV.seeds.join(", ")}`);
  const p = PROV && PROV.provenance && typeof PROV.provenance === "object" ? PROV.provenance : null;
  if (p) {
    for (const [k, v] of Object.entries(p)) {
      if (v == null || typeof v === "object") continue;
      const s = String(v);
      bits.push(`${k} ${k === "git_commit" ? s.slice(0, 12) : s}`);
    }
  }
  el.textContent = bits.length
    ? "Provenance · " + bits.join(" · ")
    : "Provenance · PARALLAX.json was not found beside CHAMBER_DATA.json.";
}

buildParallax();
buildFlow();
buildDials();
buildRain();
buildTriad();
buildProv();
