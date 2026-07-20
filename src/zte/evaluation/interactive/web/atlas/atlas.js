const RAW = "__ATLAS_PAYLOAD__";
const N = RAW.data || {};
const MAXB = RAW.max_bars || null;
const FONT = 'system-ui,-apple-system,"Segoe UI",sans-serif';

const INK = {
  light: {
    paper: "#fcfcfb",
    grid: "#e1e0d9",
    axis: "#c3c2b7",
    text: "#0b0b0b",
    muted: "#898781",
  },
  dark: {
    paper: "#1a1a19",
    grid: "#2c2c2a",
    axis: "#383835",
    text: "#ffffff",
    muted: "#898781",
  },
};
const SEQ = {
  light: [
    [0, "#cde2fb"],
    [0.5, "#3987e5"],
    [1, "#0d366b"],
  ],
  dark: [
    [0, "#173a63"],
    [0.5, "#3987e5"],
    [1, "#cde2fb"],
  ],
};
const ATTR_META = {
  subject: { c: { light: "#eda100", dark: "#c98500" }, label: "subject · who" },
  word_len: {
    c: { light: "#2a78d6", dark: "#3987e5" },
    label: "word_len · what",
  },
  log_freq: {
    c: { light: "#1baf7a", dark: "#199e70" },
    label: "log_freq · what",
  },
  category: {
    c: { light: "#4a3aa7", dark: "#9085e9" },
    label: "category · what",
  },
  task: { c: { light: "#008300", dark: "#008300" }, label: "task" },
  none: {
    c: { light: "#b8b6ad", dark: "#5c5b57" },
    label: "none · negligible",
  },
};
const ATTR_ORDER = [
  "subject",
  "word_len",
  "log_freq",
  "category",
  "task",
  "none",
];
const FALLBACK = {
  light: ["#e34948", "#e87ba4", "#eb6834"],
  dark: ["#e66767", "#d55181", "#d95926"],
};
const _fbi = {};

// ---- normalise the report (robust to missing keys) -----------------------
const imp = N.importance || {};
imp.std = imp.std || [];
imp.var_share = imp.var_share || imp.std.map(() => 0);
imp.active = imp.active || imp.std.map(() => true);
imp.rank = imp.rank || imp.std.map((_, i) => i);
imp.order =
  imp.order ||
  imp.std
    .map((_, i) => i)
    .sort((a, b) => (imp.std[b] || 0) - (imp.std[a] || 0));
if (imp.active_threshold == null) imp.active_threshold = 0;
const sel = N.selectivity || {
  targets: [],
  scores: {},
  dominant: [],
  dominant_score: [],
};
const targets = sel.targets || [];
const dom = sel.dominant || imp.std.map(() => "none");
const domScore = sel.dominant_score || imp.std.map(() => 0);
const topByDim = {};
(N.top_neurons || []).forEach((t) => {
  topByDim[t.dim] = t;
});
const D = (N.meta && N.meta.embed_dim) || imp.std.length;

const state = {
  sort: "importance",
  color: "dominant",
  metric: "var_share",
  hideDead: false,
  sel: imp.order.length ? imp.order[0] : 0,
  theme:
    window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light",
};

// ---- helpers -------------------------------------------------------------
function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
function pct(x) {
  return x == null ? "–" : (x * 100).toFixed(1) + "%";
}
function attrColor(a) {
  if (ATTR_META[a]) return ATTR_META[a].c[state.theme];
  if (!(a in _fbi)) _fbi[a] = Object.keys(_fbi).length;
  const arr = FALLBACK[state.theme];
  return arr[_fbi[a] % arr.length];
}
function attrLabel(a) {
  return ATTR_META[a] ? ATTR_META[a].label : a;
}
function value(d) {
  return state.metric === "std" ? imp.std[d] : imp.var_share[d];
}
function threshold() {
  if (state.metric === "std") return imp.active_threshold;
  const tot = imp.std.reduce((a, s) => a + s * s, 0) || 1;
  return (imp.active_threshold * imp.active_threshold) / tot;
}
function orderedDims() {
  let dims;
  if (state.sort === "importance") {
    dims = imp.order.slice();
  } else {
    const sc = sel.scores[state.sort] || [];
    dims = Array.from({ length: D }, (_, i) => i).sort(
      (a, b) => (sc[b] || 0) - (sc[a] || 0),
    );
  }
  if (state.hideDead) dims = dims.filter((d) => imp.active[d]);
  if (MAXB && dims.length > MAXB) dims = dims.slice(0, MAXB);
  return dims;
}

// ---- view 1: summary tiles + legend --------------------------------------
function renderTiles() {
  const S = N.summary || {};
  const r = S.who_vs_what_ratio;
  const ratio = r == null ? "∞" : (+r).toFixed(2) + "×";
  const tiles = [
    {
      k: "neurons active",
      v: `${S.n_active == null ? "–" : S.n_active} / ${D}`,
      cls: "",
    },
    { k: "who variance · subject", v: pct(S.who_variance), cls: "who" },
    { k: "what variance · content", v: pct(S.what_variance), cls: "what" },
    {
      k: "who / what ratio",
      v: ratio,
      cls: r != null && r > 1 ? "who" : "what",
    },
    { k: "variance in active", v: pct(S.active_variance_share), cls: "" },
  ];
  document.getElementById("tiles").innerHTML = tiles
    .map(
      (o) =>
        `<div class="tile"><div class="v ${o.cls}">${o.v}</div><div class="k">${o.k}</div></div>`,
    )
    .join("");
}
function renderLegend() {
  const present = new Set();
  (dom || []).forEach((x) => present.add(x));
  targets.forEach((t) => present.add(t));
  const ordered = ATTR_ORDER.filter((a) => present.has(a)).concat(
    [...present].filter((a) => !ATTR_ORDER.includes(a)),
  );
  document.getElementById("legend").innerHTML =
    ordered
      .map(
        (a) =>
          `<span class="lg"><span class="sw" style="background:${attrColor(a)}"></span>${attrLabel(a)}</span>`,
      )
      .join("") || '<span class="lg">no attributes probed</span>';
}

// ---- view 2: ranked importance of every neuron ---------------------------
function renderRanked() {
  const dims = orderedDims(),
    ink = INK[state.theme],
    n = dims.length;
  const cd = (d) => [
    d,
    imp.std[d],
    imp.var_share[d],
    dom[d] || "none",
    domScore[d] || 0,
  ];
  const HT =
    "neuron #%{customdata[0]}<br>std %{customdata[1]:.3f} · var %{customdata[2]:.3f}" +
    "<br>dominant %{customdata[3]} (%{customdata[4]:.2f})<extra></extra>";
  let traces = [];
  if (state.color === "dominant") {
    const groups = {};
    dims.forEach((d, pos) => {
      const a = dom[d] || "none";
      (groups[a] = groups[a] || []).push([pos, d]);
    });
    const keys = ATTR_ORDER.filter((a) => groups[a]).concat(
      Object.keys(groups).filter((a) => !ATTR_ORDER.includes(a)),
    );
    keys.forEach((a) => {
      const g = groups[a];
      traces.push({
        type: "bar",
        name: attrLabel(a),
        x: g.map((o) => o[0]),
        y: g.map((o) => value(o[1])),
        customdata: g.map((o) => cd(o[1])),
        marker: { color: attrColor(a), line: { width: 0 } },
        hovertemplate: HT,
      });
    });
  } else {
    const sc = sel.scores[state.color] || [];
    const HTS =
      "neuron #%{customdata[0]}<br>std %{customdata[1]:.3f} · var %{customdata[2]:.3f}" +
      "<br>sel(" +
      state.color +
      ") %{customdata[5]:.2f}<extra></extra>";
    traces.push({
      type: "bar",
      showlegend: false,
      x: dims.map((d, i) => i),
      y: dims.map(value),
      customdata: dims.map((d) => cd(d).concat([sc[d] || 0])),
      marker: {
        color: dims.map((d) => sc[d] || 0),
        colorscale: SEQ[state.theme],
        cmin: 0,
        cmax: 1,
        showscale: true,
        colorbar: {
          title: { text: "|sel| " + state.color, side: "right" },
          thickness: 12,
          len: 0.7,
          outlinewidth: 0,
          tickfont: { color: ink.muted, size: 10 },
        },
      },
      hovertemplate: HTS,
    });
  }
  const thr = threshold();
  Plotly.react(
    "ranked",
    traces,
    {
      barmode: "overlay",
      bargap: 0,
      paper_bgcolor: ink.paper,
      plot_bgcolor: ink.paper,
      font: { color: ink.text, size: 12, family: FONT },
      margin: { l: 58, r: 16, t: 8, b: 36 },
      uirevision: "ranked",
      legend: {
        orientation: "h",
        y: 1.1,
        x: 0,
        font: { color: ink.text, size: 11 },
        bgcolor: "rgba(0,0,0,0)",
      },
      hoverlabel: {
        bgcolor: ink.paper,
        bordercolor: ink.axis,
        font: { color: ink.text, size: 12 },
      },
      xaxis: {
        title: {
          text:
            state.sort === "importance"
              ? "importance rank (0 = most important) ->"
              : "rank by |sel| " + state.sort + " ->",
          font: { size: 11, color: ink.muted },
        },
        gridcolor: ink.grid,
        zeroline: false,
        color: ink.muted,
        range: [-0.5, Math.max(0.5, n - 0.5)],
      },
      yaxis: {
        title: {
          text: state.metric === "std" ? "std (spread)" : "variance share",
          font: { size: 11, color: ink.muted },
        },
        gridcolor: ink.grid,
        zeroline: false,
        color: ink.muted,
        rangemode: "tozero",
      },
      shapes: [
        {
          type: "line",
          x0: -0.5,
          x1: Math.max(0.5, n - 0.5),
          y0: thr,
          y1: thr,
          line: {
            color: state.theme === "dark" ? "#e66767" : "#e34948",
            width: 1.5,
            dash: "dash",
          },
        },
      ],
      annotations: [
        {
          x: Math.max(0.5, n - 0.5),
          y: thr,
          xanchor: "right",
          yanchor: "bottom",
          text: "active threshold — tail below is negligible",
          showarrow: false,
          font: {
            color: state.theme === "dark" ? "#e66767" : "#e34948",
            size: 10.5,
          },
        },
      ],
    },
    {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d"],
    },
  );
}

// ---- view 3: per-neuron detail panel -------------------------------------
function selectNeuron(dim) {
  if (dim == null || isNaN(dim) || dim < 0 || dim >= D) return;
  state.sel = dim;
  renderDetail();
}
const HEAD_REGIONS = [
  "frontopolar",
  "frontal",
  "frontocentral",
  "central",
  "centroparietal",
  "parietal",
  "parieto_occipital",
  "occipital",
];
const HEAD_LABEL = {
  frontopolar: "Frontopolar",
  frontal: "Frontal",
  frontocentral: "Frontocentral",
  central: "Central",
  centroparietal: "Centroparietal",
  parietal: "Parietal",
  parieto_occipital: "Parieto-occipital",
  occipital: "Occipital",
};
function renderHeadMap(at) {
  const scores = {},
    topBand = {};
  HEAD_REGIONS.forEach((r) => {
    scores[r] = 0;
  });
  (at || []).forEach((o) => {
    const parts = String(o.feature || "").split("·");
    const region = (parts[parts.length - 1] || "").trim();
    const band = (parts.length > 1 ? parts[0] : "").trim();
    if (region in scores) {
      const m = Math.abs(+o.corr || 0);
      scores[region] += m;
      if (!topBand[region] || m > topBand[region].v)
        topBand[region] = { band: band, v: m };
    }
  });
  const max = Math.max.apply(
    null,
    HEAD_REGIONS.map((r) => scores[r]).concat([0]),
  );
  HEAD_REGIONS.forEach((r) => {
    const rect = document.getElementById("hz-" + r);
    if (!rect) return;
    const s = scores[r],
      norm = max > 0 ? s / max : 0;
    if (s > 1e-9) {
      rect.style.fill = "var(--accent)";
      rect.style.fillOpacity = (0.12 + 0.82 * norm).toFixed(3);
    } else {
      rect.style.fill = "var(--axis)";
      rect.style.fillOpacity = "0.14";
    }
    const tb = topBand[r] ? ", top band " + topBand[r].band : "";
    const t = rect.querySelector("title");
    if (t) t.textContent = HEAD_LABEL[r] + " — score " + s.toFixed(2) + tb;
  });
}
function fillWords(id, rows) {
  const t = document.getElementById(id);
  t.innerHTML =
    "<tr><th>word</th><th>subject</th><th>act</th></tr>" +
    (rows || [])
      .map(
        (r) =>
          `<tr><td>${esc(r.word)}</td><td>${esc(r.subject)}</td>` +
          `<td class="num">${(+r.activation).toFixed(2)}</td></tr>`,
      )
      .join("");
}
function renderDetail() {
  const dim = state.sel,
    ink = INK[state.theme];
  const a = dom[dim] || "none",
    sc = domScore[dim] || 0;
  document.getElementById("detail-title").innerHTML =
    `Neuron <b>#${dim}</b> · rank ${imp.rank[dim]} · dominant: ` +
    `<span style="color:${attrColor(a)};font-weight:640">${attrLabel(a)}</span> (${sc.toFixed(2)}) ` +
    `· std ${(+imp.std[dim]).toFixed(3)} · var ${((imp.var_share[dim] || 0) * 100).toFixed(2)}%`;

  if (targets.length) {
    const vals = targets.map((t) => (sel.scores[t] || [])[dim] || 0);
    Plotly.react(
      "selbar",
      [
        {
          type: "bar",
          orientation: "h",
          x: vals,
          y: targets,
          marker: {
            color: targets.map((t) => attrColor(t)),
            line: { width: 0 },
          },
          text: vals.map((v) => v.toFixed(2)),
          textposition: "auto",
          textfont: { color: "#fff", size: 11 },
          hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
        },
      ],
      {
        paper_bgcolor: ink.paper,
        plot_bgcolor: ink.paper,
        font: { color: ink.text, size: 11, family: FONT },
        margin: { l: 74, r: 12, t: 6, b: 24 },
        uirevision: "sel",
        xaxis: {
          range: [0, 1],
          gridcolor: ink.grid,
          zeroline: false,
          color: ink.muted,
        },
        yaxis: { color: ink.text, automargin: true },
      },
      { displayModeBar: false, responsive: true },
    );
  } else {
    Plotly.purge("selbar");
    document.getElementById("selbar").innerHTML =
      '<div class="empty">No probe targets were available.</div>';
  }

  const entry = topByDim[dim];
  const exWrap = document.getElementById("exemplars");
  const attrWrap = document.getElementById("attrwrap");
  if (entry) {
    document.getElementById("detail-note").textContent = "";
    exWrap.classList.remove("hide");
    const h = entry.activation_hist || {},
      edges = h.edges || [],
      counts = h.counts || [];
    const centers = [],
      width = [];
    for (let i = 0; i < counts.length; i++) {
      centers.push((edges[i] + edges[i + 1]) / 2);
      width.push(edges[i + 1] - edges[i]);
    }
    Plotly.react(
      "acthist",
      [
        {
          type: "bar",
          x: centers,
          y: counts,
          width: width,
          marker: { color: attrColor(a), line: { width: 0 } },
          hovertemplate: "act %{x:.3f}: %{y} words<extra></extra>",
        },
      ],
      {
        paper_bgcolor: ink.paper,
        plot_bgcolor: ink.paper,
        font: { color: ink.text, size: 11, family: FONT },
        margin: { l: 46, r: 10, t: 6, b: 30 },
        uirevision: "hist",
        xaxis: {
          title: { text: "activation", font: { size: 10, color: ink.muted } },
          gridcolor: ink.grid,
          zeroline: false,
          color: ink.muted,
        },
        yaxis: {
          title: { text: "words", font: { size: 10, color: ink.muted } },
          gridcolor: ink.grid,
          zeroline: false,
          color: ink.muted,
        },
      },
      { displayModeBar: false, responsive: true },
    );
    fillWords("topwords", entry.top_words);
    fillWords("botwords", entry.bottom_words);
    const at = entry.attribution;
    if (at && at.length) {
      attrWrap.classList.remove("hide");
      renderHeadMap(at);
      const rev = at.slice().reverse();
      Plotly.react(
        "attrbar",
        [
          {
            type: "bar",
            orientation: "h",
            x: rev.map((o) => o.corr),
            y: rev.map((o) => o.feature),
            marker: {
              color: state.theme === "dark" ? "#199e70" : "#1baf7a",
              line: { width: 0 },
            },
            text: rev.map((o) => o.corr.toFixed(2)),
            textposition: "auto",
            textfont: { color: "#fff", size: 10 },
            hovertemplate: "%{y}: |r|=%{x:.3f}<extra></extra>",
          },
        ],
        {
          paper_bgcolor: ink.paper,
          plot_bgcolor: ink.paper,
          font: { color: ink.text, size: 10, family: FONT },
          margin: { l: 132, r: 12, t: 6, b: 22 },
          uirevision: "attr",
          xaxis: {
            range: [0, 1],
            gridcolor: ink.grid,
            zeroline: false,
            color: ink.muted,
          },
          yaxis: { color: ink.text, automargin: true },
        },
        { displayModeBar: false, responsive: true },
      );
    } else {
      attrWrap.classList.add("hide");
    }
  } else {
    exWrap.classList.add("hide");
    attrWrap.classList.add("hide");
    document.getElementById("detail-note").textContent =
      "Exemplars, the activation histogram and scalp/band attribution are computed for the " +
      "top neurons only. This neuron’s importance and selectivity (above) are available for " +
      "every dimension.";
  }
}

// ---- view 4: controls ----------------------------------------------------
function fill(id, opts, val) {
  const s = document.getElementById(id);
  s.innerHTML = "";
  opts.forEach(([v, label]) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = label;
    s.appendChild(o);
  });
  s.value = val;
}
function buildControls() {
  fill(
    "sortby",
    [["importance", "importance"]].concat(
      targets.map((t) => [t, "selectivity: " + t]),
    ),
    state.sort,
  );
  document.getElementById("sortby").onchange = (e) => {
    state.sort = e.target.value;
    renderRanked();
  };
  fill(
    "colorby",
    [["dominant", "dominant attribute"]].concat(
      targets.map((t) => [t, "selectivity: " + t]),
    ),
    state.color,
  );
  document.getElementById("colorby").onchange = (e) => {
    state.color = e.target.value;
    renderRanked();
  };
  document.getElementById("metric").value = state.metric;
  document.getElementById("metric").onchange = (e) => {
    state.metric = e.target.value;
    renderRanked();
  };
  document.getElementById("hidedead").onchange = (e) => {
    state.hideDead = e.target.checked;
    renderRanked();
  };
  const go = () => {
    selectNeuron(parseInt(document.getElementById("search").value, 10));
  };
  document.getElementById("searchbtn").onclick = go;
  document.getElementById("search").onkeydown = (e) => {
    if (e.key === "Enter") go();
  };
  document.getElementById("themebtn").onclick = () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    applyTheme();
    renderTiles();
    renderLegend();
    renderRanked();
    renderDetail();
  };
}
function applyTheme() {
  document.documentElement.setAttribute("data-theme", state.theme);
}

// ---- init ----------------------------------------------------------------
applyTheme();
buildControls();
renderTiles();
renderLegend();
if (D > 0) {
  renderRanked();
  document.getElementById("ranked").on("plotly_click", (ev) => {
    const p = ev.points && ev.points[0];
    if (p && p.customdata) selectNeuron(p.customdata[0]);
  });
  selectNeuron(state.sel);
} else {
  document.getElementById("ranked").innerHTML =
    '<div class="empty">Empty report — no neurons to display.</div>';
}
window.addEventListener("resize", () => {
  ["ranked", "selbar", "acthist", "attrbar"].forEach((id) => {
    const el = document.getElementById(id);
    if (el && el.data) Plotly.Plots.resize(id);
  });
});
