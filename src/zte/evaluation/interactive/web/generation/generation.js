const P = JSON.parse(document.getElementById("data").textContent);
const root = document.documentElement;

/* ---------- theme (persisted) ---------- */
const TKEY = "zte-generation-theme";
try {
  const s = localStorage.getItem(TKEY);
  if (s === "dark" || s === "light") root.setAttribute("data-theme", s);
} catch (e) {}
document.getElementById("themebtn").onclick = () => {
  const cur =
    root.getAttribute("data-theme") ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  try {
    localStorage.setItem(TKEY, next);
  } catch (e) {}
};

/* ---------- helpers ---------- */
function esc(s) {
  return String(s == null ? "" : s).replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
  );
}
function nv(x) {
  return x == null || (typeof x === "number" && !isFinite(x)) ? null : x;
}
function num(v, d) {
  return nv(v) == null ? "—" : (+v).toFixed(d == null ? 4 : d);
}
function signed(v, d) {
  return nv(v) == null
    ? "—"
    : (v >= 0 ? "+" : "") + (+v).toFixed(d == null ? 4 : d);
}

const METRICS = P.metrics && P.metrics.length ? P.metrics : ["content_f1"];
const PRIMARY = P.primary_metric || METRICS[0];
const state = { query: "", sort: "index", controls: true };

/* ---------- verdict banner ---------- */
function renderVerdict() {
  const v = P.verdict || {};
  const bits = [];
  bits.push(
    (v.beats_all_controls ? "Beats" : "Does not beat") +
      " every brain-independent control on " +
      PRIMARY,
  );
  if (v.worst_control)
    bits.push(
      "worst control " +
        v.worst_control +
        " at " +
        signed((v.worst_ci || {}).point) +
        " [" +
        signed((v.worst_ci || {}).lo) +
        ", " +
        signed((v.worst_ci || {}).hi) +
        "]",
    );
  if (nv(v.permutation_p) != null)
    bits.push("permutation p = " + num(v.permutation_p, 4));
  if (nv(v.prefix_kl) != null)
    bits.push("prefix-influence KL = " + num(v.prefix_kl, 4) + " nats");
  bits.push(
    P.n +
      " held-out readings on the " +
      (P.split || "unknown") +
      " cell of " +
      (P.split_strategy || "unknown") +
      (P.free ? ", free decode" : ", CONSTRAINED decode"),
  );
  document.getElementById("verdict").textContent = bits.join(" · ") + ".";

  const warn = document.getElementById("warn");
  const problems = [];
  if (!P.free)
    problems.push(
      "This decode chose from a candidate set of " +
        P.n_candidate_sentences +
        " sentences. That is retrieval, not generation.",
    );
  if (!P.honest_split)
    problems.push(
      "Decoded the " +
        (P.split || "unknown") +
        " cell of " +
        (P.split_strategy || "unknown") +
        ", which does not hold both the subject and the stimulus out. A headline needs the " +
        (P.honest_split_required || "") +
        ".",
    );
  const absent = (v.controls_absent || []).join(", ");
  if (absent)
    problems.push(
      "Pre-registered controls that never ran: " +
        absent +
        ". Every control must run and be beaten before any delta is readable.",
    );
  if (problems.length) {
    warn.className = "warn on";
    warn.innerHTML = problems.map((t) => "<div>" + esc(t) + "</div>").join("");
  }
}

/* ---------- paired deltas ---------- */
function renderDeltas() {
  const rows = Object.keys(P.deltas || {});
  const host = document.getElementById("deltas");
  if (!rows.length) {
    document.getElementById("deltaSection").style.display = "none";
    return;
  }
  let lim = 0;
  rows.forEach((k) => {
    const d = P.deltas[k];
    ["point", "lo", "hi"].forEach((f) => {
      if (nv(d[f]) != null) lim = Math.max(lim, Math.abs(d[f]));
    });
  });
  lim = lim > 0 ? lim * 1.15 : 1;
  const pos = (x) => ((x + lim) / (2 * lim)) * 100;

  host.innerHTML = rows
    .map((k) => {
      const d = P.deltas[k] || {};
      const lo = nv(d.lo),
        hi = nv(d.hi),
        pt = nv(d.point);
      const left = lo == null ? 50 : pos(lo);
      const width = lo == null || hi == null ? 0 : Math.max(pos(hi) - left, 0.6);
      return (
        '<div class="delta">' +
        '<div class="name">' +
        esc(k) +
        "</div>" +
        '<div class="bar">' +
        '<div class="zero" style="left:50%"></div>' +
        '<div class="span' +
        (d.beats ? " pass" : "") +
        '" style="left:' +
        left +
        "%;width:" +
        width +
        '%"></div>' +
        (pt == null
          ? ""
          : '<div class="pt" style="left:calc(' + pos(pt) + '% - 1px)"></div>') +
        "</div>" +
        '<div class="num">' +
        signed(pt) +
        " [" +
        signed(lo) +
        ", " +
        signed(hi) +
        "]</div>" +
        '<div class="mark ' +
        (d.beats ? "pass" : "fail") +
        '">' +
        (d.beats ? "✓" : "·") +
        "</div>" +
        "</div>"
      );
    })
    .join("");
  document.getElementById("deltaLegend").textContent =
    "Per-sentence " +
    PRIMARY +
    " delta, hypothesis minus control, with a 95% paired bootstrap interval. The bar must sit wholly " +
    "right of the zero line. A decoder that ignores its conditioning vector and recites the corpus lands exactly on zero.";
}

/* ---------- absolute score table ---------- */
function renderAbsolute() {
  const conds = P.condition_order || [];
  if (!conds.length) {
    document.getElementById("absSection").style.display = "none";
    return;
  }
  let html =
    "<thead><tr><th>condition</th>" +
    METRICS.map((m) => "<th>" + esc(m) + "</th>").join("") +
    "</tr></thead><tbody>";
  conds.forEach((c) => {
    const scores = (P.absolute || {})[c] || {};
    const cls =
      c === "hypothesis" ? " class='self'" : c === "oracle" ? " class='oracle'" : "";
    html +=
      "<tr" +
      cls +
      "><td>" +
      esc(c) +
      "</td>" +
      METRICS.map((m) => "<td>" + num(scores[m]) + "</td>").join("") +
      "</tr>";
  });
  document.getElementById("absTable").innerHTML = html + "</tbody>";
}

/* ---------- per-sentence rows ---------- */
function rowMatches(r) {
  if (!state.query) return true;
  const q = state.query.toLowerCase();
  if (String(r.reference || "").toLowerCase().includes(q)) return true;
  if (String(r.hypothesis || "").toLowerCase().includes(q)) return true;
  return Object.keys(r.controls || {}).some((k) =>
    String(r.controls[k].text || "").toLowerCase().includes(q),
  );
}

function renderRows() {
  const all = P.rows || [];
  let rows = all.filter(rowMatches);
  if (state.sort !== "index") {
    const key = state.sort;
    rows = rows.slice().sort((a, b) => {
      const av = (a.scores || {})[key],
        bv = (b.scores || {})[key];
      return (nv(bv) == null ? -1 : bv) - (nv(av) == null ? -1 : av);
    });
  }
  document.getElementById("count").textContent =
    rows.length + " of " + all.length + " shown" + (P.truncated ? " (page caps the list)" : "");

  document.getElementById("rows").innerHTML = rows
    .map((r) => {
      const parts = [
        '<div class="idx">reading ' + r.index + "</div>",
        line("ref", "reference", r.reference, null),
        line("hyp", "hypothesis", r.hypothesis, (r.scores || {})[PRIMARY]),
      ];
      if (r.oracle != null) parts.push(line("orc", "text oracle", r.oracle, null));
      if (state.controls)
        Object.keys(r.controls || {}).forEach((k) =>
          parts.push(
            line("ctl", k, r.controls[k].text, (r.controls[k].scores || {})[PRIMARY]),
          ),
        );
      return '<div class="row">' + parts.join("") + "</div>";
    })
    .join("");
}

function line(cls, tag, text, score) {
  return (
    '<div class="line ' +
    cls +
    '"><div class="tag">' +
    esc(tag) +
    '</div><div class="txt">' +
    (text ? esc(text) : "<em>(empty)</em>") +
    '</div><div class="sc">' +
    (nv(score) == null ? "" : num(score, 3)) +
    "</div></div>"
  );
}

/* ---------- boot ---------- */
(function () {
  document.getElementById("run").textContent =
    (P.run_name || "ZTE run") + " — generation side-by-side";
  if (!P.applicable) {
    document.getElementById("verdict").textContent =
      P.reason || "No generation numbers were available for this run.";
    ["deltaSection", "absSection", "rowSection"].forEach(
      (id) => (document.getElementById(id).style.display = "none"),
    );
    document.getElementById("emptySection").style.display = "";
    return;
  }
  const sel = document.getElementById("sortby");
  sel.innerHTML =
    '<option value="index">reading order</option>' +
    METRICS.map(
      (m) =>
        '<option value="' + esc(m) + '">best ' + esc(m) + " first</option>",
    ).join("");
  sel.onchange = () => {
    state.sort = sel.value;
    renderRows();
  };
  const search = document.getElementById("search");
  search.oninput = () => {
    state.query = search.value.trim();
    renderRows();
  };
  const box = document.getElementById("showControls");
  box.onchange = () => {
    state.controls = box.checked;
    renderRows();
  };
  renderVerdict();
  renderDeltas();
  renderAbsolute();
  renderRows();
})();
