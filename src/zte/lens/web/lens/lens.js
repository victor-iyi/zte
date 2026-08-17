"use strict";

const D = JSON.parse(document.getElementById("lens-data").textContent || "{}");

const READING = D.reading && typeof D.reading === "object" ? D.reading : {};
const WORDS = Array.isArray(READING.words) ? READING.words : [];
const WSAL = D.word_saliency && typeof D.word_saliency === "object" ? D.word_saliency : {};
const SCALP = D.channel_saliency && typeof D.channel_saliency === "object" ? D.channel_saliency : null;
const NEIGHBORS = Array.isArray(D.neighbors) ? D.neighbors : [];
const DECODE = D.decode && typeof D.decode === "object" ? D.decode : null;
const EMB = D.embedding && typeof D.embedding === "object" ? D.embedding : {};
const PROV = D.provenance && typeof D.provenance === "object" ? D.provenance : {};

const INK = "#eaf0fb";
const MUTED = "#93a0ba";
const ACCENT = "#9d8bff";
const WARM = "#ff6ba6";
const COOL = "#3fe0cd";
const FONT = 'ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif';
const CONFIG = { displayModeBar: false, responsive: true };
const BASE = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: FONT, color: MUTED, size: 12 },
  hoverlabel: { bgcolor: "#101a30", bordercolor: "rgba(255,255,255,0.15)", font: { color: INK, size: 12 } },
};
/* Diverging for word tints (raw drops can go negative); sequential warm for the clamped channel scores. */
const DIVERGING = [
  [0, COOL],
  [0.5, "rgba(147,160,186,0.22)"],
  [1, WARM],
];
const WARMSEQ = [
  [0, "rgba(147,160,186,0.15)"],
  [1, WARM],
];

const fin = (x) => typeof x === "number" && isFinite(x);
const fmt = (x, d) => (fin(x) ? Number(x).toFixed(d == null ? 3 : d) : "n/a");
const esc = (s) =>
  String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/* ---- header, badge, disclaimer ---- */

function buildHeader() {
  const text = String(READING.text || "");
  const short = text.length > 110 ? text.slice(0, 107) + "…" : text;
  document.getElementById("sub").textContent =
    `${READING.subject || "?"} · ${READING.task || "?"} · “${short}” · ${WORDS.length} words · ${D.mode} mode`;

  /* Honesty guard: whether this brain was in the training set is the first thing the page must say. */
  const badge = document.getElementById("subjectbadge");
  if (READING.is_holdout) {
    badge.className = "badge ok";
    badge.textContent = `held-out brain · ${READING.subject || "?"}`;
  } else {
    badge.className = "badge warn";
    badge.textContent = `TRAINING brain — the model has seen ${READING.subject || "?"}`;
  }

  /* The disclaimer comes from the capture itself; the builder refused to render a file without one. */
  document.getElementById("disclaimer").textContent = D.disclaimer || "";

  const pill = document.getElementById("embedpill");
  pill.textContent = `embedding · dim ${fin(EMB.dim) ? EMB.dim : "?"} · ‖z‖ ${fmt(EMB.norm, 3)}`;
}

/* ---- panel 1: the reading ---- */

function buildReading() {
  const host = document.getElementById("sentence");
  const scores = Array.isArray(WSAL.raw) ? WSAL.raw : Array.isArray(WSAL.scores) ? WSAL.scores : [];
  const maxAbs = Math.max(1e-9, ...scores.filter(fin).map(Math.abs));

  WORDS.forEach((w, i) => {
    const span = document.createElement("span");
    span.className = "word";
    span.textContent = String(w);
    const s = scores[i];
    if (fin(s)) {
      const u = Math.max(-1, Math.min(1, s / maxAbs));
      span.style.background = u >= 0 ? hexA(WARM, 0.07 + 0.5 * u) : hexA(COOL, 0.07 + 0.5 * -u);
      span.title =
        `masking “${w}” drops cosine by ${fmt(s, 5)} (${WSAL.method || "occlusion"}); ` +
        "positive = the embedding leaned on this word";
    } else {
      span.title = "no saliency recorded for this word";
    }
    host.appendChild(span);
  });

  if (WSAL.method) {
    const legend = document.getElementById("readinglegend");
    legend.innerHTML += ` Method: <b>${esc(WSAL.method)}</b>.`;
  }
}

/* ---- panel 2: the scalp ---- */

function buildScalp() {
  /* When the capture has no montage, the builder already unhid the honest note; nothing to draw here. */
  if (!SCALP) return;

  const labels = Array.isArray(SCALP.labels) ? SCALP.labels : [];
  const regions = Array.isArray(SCALP.regions) ? SCALP.regions : [];
  const xy = Array.isArray(SCALP.xy) ? SCALP.xy : [];
  const xyz = Array.isArray(SCALP.xyz) ? SCALP.xyz : [];
  const scores = Array.isArray(SCALP.scores) ? SCALP.scores : [];
  const host = document.getElementById("scalp");

  const maxAbs = Math.max(1e-9, ...scores.filter(fin).map(Math.abs));
  const sizes = scores.map((s) => 9 + (fin(s) ? (Math.abs(s) / maxAbs) * 15 : 0));
  const hover = labels.map(
    (l, i) => `${esc(l)} · ${esc(regions[i] || "?")}<br>Δcos ${fmt(scores[i], 5)} · ${esc(SCALP.method || "occlusion")}`
  );
  const marker = {
    size: sizes,
    color: scores,
    colorscale: WARMSEQ,
    cmin: 0,
    cmax: maxAbs,
    opacity: 0.92,
    line: { width: 1, color: "rgba(255,255,255,0.25)" },
  };

  function draw2d() {
    const r = Math.max(1e-6, ...xy.map((p) => Math.hypot(p[0], p[1]))) * 1.14;
    const outline = { line: { color: "rgba(147,160,186,0.5)", width: 1.6 } };
    Plotly.react(
      host,
      [
        {
          type: "scatter",
          mode: "markers",
          x: xy.map((p) => p[0]),
          y: xy.map((p) => p[1]),
          marker,
          text: hover,
          hovertemplate: "%{text}<extra></extra>",
        },
      ],
      {
        ...BASE,
        height: 400,
        margin: { l: 10, r: 10, t: 10, b: 10 },
        showlegend: false,
        xaxis: { visible: false, range: [-1.45 * r, 1.45 * r] },
        yaxis: { visible: false, scaleanchor: "x", range: [-1.35 * r, 1.5 * r] },
        shapes: [
          { type: "circle", x0: -r, x1: r, y0: -r, y1: r, ...outline },
          /* nose */
          { type: "path", path: `M ${-0.16 * r} ${0.99 * r} L 0 ${1.22 * r} L ${0.16 * r} ${0.99 * r}`, ...outline },
          /* ears */
          { type: "path", path: `M ${-r} ${0.18 * r} Q ${-1.18 * r} 0 ${-r} ${-0.18 * r}`, ...outline },
          { type: "path", path: `M ${r} ${0.18 * r} Q ${1.18 * r} 0 ${r} ${-0.18 * r}`, ...outline },
        ],
      },
      CONFIG
    );
  }

  function draw3d() {
    Plotly.react(
      host,
      [
        {
          type: "scatter3d",
          mode: "markers",
          x: xyz.map((p) => p[0]),
          y: xyz.map((p) => p[1]),
          z: xyz.map((p) => p[2]),
          marker,
          text: hover,
          hovertemplate: "%{text}<extra></extra>",
        },
      ],
      {
        ...BASE,
        height: 400,
        margin: { l: 0, r: 0, t: 0, b: 0 },
        showlegend: false,
        scene: {
          bgcolor: "rgba(0,0,0,0)",
          xaxis: { visible: false },
          yaxis: { visible: false },
          zaxis: { visible: false },
          aspectmode: "data",
          camera: { eye: { x: 0.0, y: -0.9, z: 1.6 } },
        },
      },
      CONFIG
    );
  }

  const has3d = xyz.length === xy.length && xyz.length > 0;
  const views = has3d ? ["2D", "3D"] : ["2D"];
  let view = "2D";
  const ctl = document.getElementById("scalpview");
  function paint() {
    ctl.innerHTML = "";
    for (const v of views) {
      const b = document.createElement("button");
      b.textContent = v;
      b.className = v === view ? "on" : "";
      b.addEventListener("click", () => {
        if (v === view) return;
        view = v;
        paint();
        (view === "3D" ? draw3d : draw2d)();
      });
      ctl.appendChild(b);
    }
  }

  paint();
  draw2d();
}

/* ---- panel 3: the neighbourhood ---- */

function buildNeighbors() {
  const host = document.getElementById("strips");
  if (!NEIGHBORS.length) {
    host.innerHTML = '<div class="emptycell">No gallery neighbours travelled with this capture.</div>';
    return;
  }

  const cs = NEIGHBORS.map((n) => (fin(n.cosine) ? n.cosine : 0));
  const lo = Math.min(...cs);
  const hi = Math.max(...cs);

  NEIGHBORS.forEach((n, i) => {
    const c = cs[i];
    /* Widths are rank strips scaled within this list, so ordering stays visible even when cosines bunch up. */
    const w = hi > lo ? 0.1 + 0.9 * ((c - lo) / (hi - lo)) : 1;
    const same = READING.subject != null && n.subject === READING.subject;
    const row = document.createElement("div");
    row.className = "strip" + (n.is_true_sentence ? " true" : "") + (same ? " same" : "");
    const tag =
      esc(n.subject) +
      (n.is_true_sentence ? " · this sentence" : "") +
      (same ? " · same subject" : "");
    row.innerHTML =
      `<span class="rank">${i + 1}</span>` +
      `<div class="bar"><div class="fill" style="width:${(w * 100).toFixed(1)}%"></div></div>` +
      `<span class="ncos">${fmt(c, 3)}</span>` +
      `<span class="ntext" title="${esc(n.text)}">${esc(n.text)}</span>` +
      `<span class="ntag">${tag}</span>`;
    host.appendChild(row);
  });
}

/* ---- panel 4: the decode trace ---- */

function buildDecode() {
  if (!DECODE) return;

  const tokens = Array.isArray(DECODE.tokens) ? DECODE.tokens : [];
  const tokHost = document.getElementById("tokens");
  tokens.forEach((t, i) => {
    const chip = document.createElement("span");
    chip.className = "tokchip";
    chip.textContent = String(t);
    chip.title = `token ${i}`;
    tokHost.appendChild(chip);
  });

  document.getElementById("gentext").textContent = String(DECODE.generated || "");
  document.getElementById("nulltext").textContent = String(DECODE.null_prefix_generated || "");

  buildRibbon(tokens);
  buildSlots();
}

function buildRibbon(tokens) {
  const host = document.getElementById("ribbon");
  const we = Array.isArray(DECODE.word_evidence) ? DECODE.word_evidence : null;
  if (!we || !we.length || !tokens.length || !WORDS.length) {
    host.innerHTML =
      '<div class="emptycell">This checkpoint carries no word-synchronous evidence head — there is no honest token → word ribbon to draw.</div>';
    return;
  }

  const z = tokens.map(() => WORDS.map(() => 0));
  for (const trip of we) {
    if (!Array.isArray(trip) || trip.length < 3) continue;
    const [t, w, v] = trip;
    if (Number.isInteger(t) && Number.isInteger(w) && t >= 0 && t < tokens.length && w >= 0 && w < WORDS.length) {
      z[t][w] = fin(v) ? v : 0;
    }
  }

  /* Index-prefixed category labels keep repeated words and repeated tokens as distinct axis positions. */
  Plotly.newPlot(
    host,
    [
      {
        type: "heatmap",
        z,
        x: WORDS.map((w, i) => `${i}·${w}`),
        y: tokens.map((t, i) => `${i}·${t}`),
        colorscale: [
          [0, "rgba(157,139,255,0.02)"],
          [1, ACCENT],
        ],
        showscale: false,
        hovertemplate: "token %{y} ← word %{x}<br>evidence weight %{z:.4f}<extra></extra>",
      },
    ],
    {
      ...BASE,
      height: Math.max(180, Math.min(420, 40 + tokens.length * 16)),
      margin: { l: 90, r: 10, t: 8, b: 70 },
      xaxis: { tickfont: { size: 10 }, tickangle: -45 },
      yaxis: { tickfont: { size: 10 }, autorange: "reversed" },
    },
    CONFIG
  );
}

function buildSlots() {
  const host = document.getElementById("slots");
  const sl = Array.isArray(DECODE.slot_influence) ? DECODE.slot_influence : [];
  if (!sl.length) {
    host.innerHTML = '<div class="emptycell">No slot-occlusion trace in this capture.</div>';
    return;
  }

  const step = 360 / sl.length;
  Plotly.newPlot(
    host,
    [
      {
        type: "barpolar",
        r: sl.map((v) => (fin(v) ? Math.max(0, v) : 0)),
        theta: sl.map((_, i) => i * step),
        width: sl.map(() => step * 0.82),
        marker: {
          color: sl,
          colorscale: [
            [0, "rgba(63,224,205,0.3)"],
            [1, ACCENT],
          ],
          line: { width: 0 },
        },
        hovertemplate: "slot %{pointNumber}<br>occlusion divergence %{r:.4f}<extra></extra>",
      },
    ],
    {
      ...BASE,
      height: 200,
      margin: { l: 12, r: 12, t: 12, b: 12 },
      polar: {
        bgcolor: "rgba(0,0,0,0)",
        radialaxis: { visible: false },
        angularaxis: { visible: false, rotation: 90, direction: "clockwise" },
      },
      showlegend: false,
    },
    CONFIG
  );
}

/* ---- provenance ---- */

function buildProv() {
  const bits = [];
  if (PROV.run_name) bits.push(`run ${PROV.run_name}`);
  if (PROV.ckpt) bits.push(`ckpt ${String(PROV.ckpt).split("/").pop()}`);
  if (PROV.ckpt_sha256) bits.push(`sha256 ${String(PROV.ckpt_sha256).slice(0, 12)}`);
  if (PROV.git_commit) bits.push(`commit ${String(PROV.git_commit).slice(0, 12)}`);
  if (PROV.train_holdout) bits.push(`train holdout ${PROV.train_holdout}`);
  document.getElementById("prov").textContent = bits.length
    ? "Provenance · " + bits.join(" · ")
    : "Provenance · none travelled with this capture.";
}

buildHeader();
buildReading();
buildScalp();
buildNeighbors();
buildDecode();
buildProv();
