const P = "__PAYLOAD__";
const runs = P.runs,
  checks = P.rubric.checks,
  metrics = P.rubric.metrics;

/* ---- theme ---- */
const root = document.documentElement;
function applyTheme(t) {
  if (t) root.setAttribute("data-theme", t);
}
document.getElementById("themebtn").onclick = () => {
  const cur = root.getAttribute("data-theme");
  const next =
    cur === "dark"
      ? "light"
      : cur === "light"
        ? "dark"
        : matchMedia("(prefers-color-scheme: dark)").matches
          ? "light"
          : "dark";
  applyTheme(next);
};

function fmt(v, f) {
  if (v === null || v === undefined) return "—";
  if (f === "%+.3f") return (v >= 0 ? "+" : "") + v.toFixed(3);
  if (f === "%.3f") return v.toFixed(3);
  return String(v);
}
function esc(s) {
  return String(s).replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
  );
}
function cfgLine(r) {
  const c = r.config || {};
  const bits = [];
  bits.push(c.objective || "?");
  bits.push(c.eye_tracking ? "ET" : "EEG-only");
  if (c.split) bits.push(c.split.replace("by_", ""));
  if (c.loso_holdout) bits.push("−" + c.loso_holdout);
  if (c.cross_subject_positives) bits.push("xsubj+");
  if (c.subject_adversary_weight > 0)
    bits.push("adv" + c.subject_adversary_weight);
  if (c.variance_weight > 0 || c.covariance_weight > 0) bits.push("vicreg");
  return bits.join(" · ");
}

document.getElementById("sub").textContent =
  runs.length +
  " run" +
  (runs.length === 1 ? "" : "s") +
  " · " +
  runs.filter((r) => r.real).length +
  " on real data, " +
  runs.filter((r) => !r.real).length +
  " synthetic · green = pass, red = fail";

/* ---- best banner ---- */
(function () {
  const b = runs[0];
  if (!b) {
    document.getElementById("best").innerHTML =
      '<span class="empty">No runs.</span>';
    return;
  }
  const why = metrics
    .map((m) => ({ m, u: (b.norm || {})[m.key] }))
    .filter((x) => x.u != null)
    .sort((a, z) => z.u * z.m.weight - a.u * a.m.weight)
    .slice(0, 3)
    .map((x) => esc(x.m.label))
    .join(", ");
  document.getElementById("best").innerHTML =
    '<div class="medal">🥇</div>' +
    '<div style="min-width:200px"><div class="name">' +
    esc(b.name) +
    "</div>" +
    '<div class="sub">' +
    esc(cfgLine(b)) +
    (b.real ? ' · <span class="pill real">real data</span>' : "") +
    "</div></div>" +
    '<div class="pill pass">' +
    b.checks_passed +
    " / " +
    checks.length +
    " checks</div>" +
    '<div class="scorebar"><i style="width:' +
    (b.score * 100).toFixed(0) +
    '%"></i></div>' +
    '<div class="sub" style="min-width:210px">Leads on: ' +
    (why || "—") +
    "</div>";
})();

/* ---- scorecard matrix ---- */
function heatColor(u) {
  // u in 0..1, higher better -> greener
  if (u == null) return "var(--heat0)";
  const g = "var(--pass)",
    r = "var(--fail)";
  return (
    "color-mix(in srgb, " +
    g +
    " " +
    Math.round(u * 55 + 8) +
    "%, " +
    "color-mix(in srgb, " +
    r +
    " " +
    Math.round((1 - u) * 40) +
    "%, var(--panel)))"
  );
}
(function () {
  const t = document.getElementById("scorecard");
  let h = '<thead><tr><th class="rowhead">metric</th>';
  runs.forEach((r) => {
    h +=
      '<th class="run"><span class="rn">' +
      esc(r.name) +
      "</span>" +
      '<span class="cfg">' +
      esc(cfgLine(r)) +
      "</span></th>";
  });
  h += "</tr></thead><tbody>";
  // checks
  checks.forEach((c) => {
    h +=
      '<tr><td class="rowhead">' +
      esc(c.label) +
      '<span class="desc">' +
      esc(c.desc) +
      "</span></td>";
    runs.forEach((r) => {
      const ok = r.checks[c.key];
      h +=
        '<td><span class="mark ' +
        (ok ? "y" : "n") +
        '">' +
        (ok ? "✓" : "✕") +
        "</span></td>";
    });
    h += "</tr>";
  });
  // checks passed
  h += '<tr><td class="rowhead">Checks passed</td>';
  runs.forEach((r) => {
    h +=
      '<td><span class="count">' +
      r.checks_passed +
      " / " +
      checks.length +
      "</span></td>";
  });
  h += "</tr>";
  // scored metrics as heat
  metrics.forEach((m) => {
    h +=
      '<tr><td class="rowhead">' +
      esc(m.label) +
      '<span class="desc">' +
      (m.higher_better ? "▲ higher" : "▼ lower") +
      " is better</span></td>";
    runs.forEach((r) => {
      const v = r.metrics[m.key],
        u = (r.norm || {})[m.key];
      h +=
        '<td><span class="cell heat" style="display:inline-block;padding:5px 9px;background:' +
        heatColor(u) +
        '">' +
        fmt(v, m.fmt) +
        "</span></td>";
    });
    h += "</tr>";
  });
  h += "</tbody>";
  t.innerHTML = h;
})();
document.getElementById("legend").innerHTML =
  "<span>✓ pass&nbsp;·&nbsp;✕ fail</span><span>Heat cells: greener = better among these runs</span>";

/* ---- CI bar ---- */
function ciBar(ci) {
  // ci = [point, lo, hi]; good if lo>0
  if (!ci || ci.length < 3 || ci[0] == null)
    return '<span class="empty">—</span>';
  const [pt, lo, hi] = ci;
  const span = Math.max(Math.abs(lo), Math.abs(hi), 0.02) * 1.15;
  const x = (v) => (0.5 + 0.5 * (v / span)) * 100;
  const good = lo > 0;
  return (
    '<div class="ci"><span class="zero" style="left:50%"></span>' +
    '<span class="bar ' +
    (good ? "good" : hi < 0 ? "bad" : "") +
    '" style="left:' +
    x(lo).toFixed(1) +
    "%;width:" +
    Math.max(1, x(hi) - x(lo)).toFixed(1) +
    '%"></span>' +
    '<span class="pt" style="left:' +
    x(pt).toFixed(1) +
    '%"></span></div>'
  );
}

/* ---- detail table (sortable) ---- */
const detailCols = [
  { k: "rank", label: "#", get: (r) => r.rank, num: true },
  { k: "name", label: "run", get: (r) => r.name, num: false },
  {
    k: "checks_passed",
    label: "checks",
    get: (r) => r.checks_passed,
    num: true,
  },
  {
    k: "score",
    label: "score",
    get: (r) => r.score,
    num: true,
    fmt: (v) => (v * 100).toFixed(0),
  },
];
metrics.forEach((m) =>
  detailCols.push({
    k: m.key,
    label: m.label,
    get: (r) => r.metrics[m.key],
    num: true,
    fmt: (v) => fmt(v, m.fmt),
    spec: m,
  }),
);
detailCols.push({
  k: "retr_ci",
  label: "retrieval CI",
  get: (r) => r.ci.retrieval,
  num: false,
  ci: true,
});
detailCols.push({
  k: "arith_ci",
  label: "arithmetic CI",
  get: (r) => r.ci.arithmetic,
  num: false,
  ci: true,
});

let sortKey = "rank",
  sortAsc = true;
function renderDetail() {
  const rows = [...runs].sort((a, z) => {
    const col = detailCols.find((c) => c.k === sortKey);
    let av = col.get(a),
      zv = col.get(z);
    if (col.ci) {
      av = (av && av[1]) || 0;
      zv = (zv && zv[1]) || 0;
    }
    if (av == null) av = -1e9;
    if (zv == null) zv = -1e9;
    if (typeof av === "string")
      return sortAsc ? av.localeCompare(zv) : zv.localeCompare(av);
    return sortAsc ? av - zv : zv - av;
  });
  let h = "<thead><tr>";
  detailCols.forEach((c) => {
    h +=
      '<th class="' +
      (c.k === "name" ? "rowhead" : "") +
      '">' +
      esc(c.label) +
      (sortKey === c.k ? (sortAsc ? " ▲" : " ▼") : "") +
      "</th>";
  });
  h += "</tr></thead><tbody>";
  rows.forEach((r) => {
    h += "<tr>";
    detailCols.forEach((c) => {
      let v = c.get(r),
        disp;
      if (c.ci) disp = ciBar(v);
      else if (v == null) disp = '<span class="empty">—</span>';
      else if (c.fmt) disp = c.fmt(v);
      else disp = esc(v);
      const cls = c.k === "name" ? "rowhead" : c.spec ? "heat" : "";
      let style = "";
      if (c.spec) {
        const u = (r.norm || {})[c.spec.key];
        style = ' style="background:' + heatColor(u) + '"';
      }
      h += '<td class="' + cls + '"' + style + ">" + disp + "</td>";
    });
    h += "</tr>";
  });
  h += "</tbody>";
  const t = document.getElementById("detail");
  t.innerHTML = h;
  t.querySelectorAll("thead th").forEach((th, i) => {
    th.onclick = () => {
      const c = detailCols[i];
      if (sortKey === c.k) sortAsc = !sortAsc;
      else {
        sortKey = c.k;
        sortAsc = c.num ? false : true;
      }
      renderDetail();
    };
  });
}
renderDetail();

/* ---- run cards ---- */
(function () {
  let h = "";
  runs.forEach((r) => {
    const p = r.paths || {};
    const thumb = p.pca_thumb
      ? '<img class="thumb" src="' + p.pca_thumb + '" alt="PCA by subject"/>'
      : '<div class="thumb"></div>';
    const links = [];
    if (p.explorer)
      links.push(
        '<a href="' + esc(p.explorer) + '" target="_blank">🌌 Explorer</a>',
      );
    if (p.atlas)
      links.push(
        '<a href="' + esc(p.atlas) + '" target="_blank">🧠 Neuron Atlas</a>',
      );
    if (p.report)
      links.push(
        '<a href="' + esc(p.report) + '" target="_blank">📄 Report</a>',
      );
    h +=
      '<div class="card runcard">' +
      thumb +
      '<div class="body">' +
      '<div class="rname"><span class="rank">#' +
      r.rank +
      "</span>" +
      esc(r.name) +
      (r.real ? ' <span class="pill real">real</span>' : "") +
      "</div>" +
      '<div class="sub">' +
      esc(cfgLine(r)) +
      "</div>" +
      '<div class="sub" style="margin-top:5px">' +
      r.checks_passed +
      "/" +
      checks.length +
      " checks · score " +
      (r.score * 100).toFixed(0) +
      "</div>" +
      '<div class="links">' +
      (links.join("") || '<span class="empty">no artifacts</span>') +
      "</div>" +
      "</div></div>";
  });
  document.getElementById("cards").innerHTML = h;
})();

/* ---- rubric ---- */
(function () {
  let h =
    "<p>Runs are ranked <b>first by the number of hard checks passed</b>, then by a weighted " +
    "quality score. Every metric is min–max normalised across the runs shown (direction-aware, so " +
    "higher is always better), then combined with these weights:</p><ul>";
  const tot = metrics.reduce((s, m) => s + m.weight, 0);
  metrics.forEach((m) => {
    h +=
      "<li><b>" +
      esc(m.label) +
      "</b> — " +
      (m.higher_better ? "▲ higher" : "▼ lower") +
      " is better · weight " +
      ((m.weight / tot) * 100).toFixed(0) +
      "% — <span>" +
      esc(m.note) +
      "</span></li>";
  });
  h +=
    '</ul><p class="empty">Normalisation is relative to the runs on this page, so the score ' +
    "compares these experiments against each other — it is not an absolute quality scale.</p>";
  document.getElementById("rubric").innerHTML = h;
})();
