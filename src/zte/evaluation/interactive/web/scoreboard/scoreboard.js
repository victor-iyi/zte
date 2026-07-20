const P = JSON.parse(document.getElementById("data").textContent);
const root = document.documentElement;

/* ---------- theme (persisted) ---------- */
const TKEY = "zte-scoreboard-theme";
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
function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}
function num(v, f) {
  if (v == null || (typeof v === "number" && !isFinite(v))) return "—";
  if (f === "ratio") return (+v).toFixed(3);
  if (f === "signed") return (v >= 0 ? "+" : "") + (+v).toFixed(3);
  if (f === "pct") return (+v * 100).toFixed(1) + "%";
  if (f === "count") return String(Math.round(v));
  return String(v);
}

/* ---------- lift lookups (view-independent: probes run over all embeddings) ---------- */
function bestContentLift() {
  let best = null;
  (P.lift || []).forEach((l) => {
    if (l.is_content && nv(l.lift_linear) != null)
      best = best == null ? l.lift_linear : Math.max(best, l.lift_linear);
  });
  return best;
}
function identityLift() {
  let val = null;
  (P.lift || []).forEach((l) => {
    if (l.is_identity && nv(l.lift_linear) != null && val == null)
      val = l.lift_linear;
  });
  return val;
}

/* ---------- card specs: each names its own reference line ---------- */
const CARDS = [
  {
    key: "erank",
    label: "Effective-rank ratio",
    tag: "view",
    src: "geometry",
    field: "effective_rank_ratio",
    fmt: "ratio",
    refLabel: "healthy > 0.10",
    scale: (v) => ({ min: 0, max: Math.max((v || 0) * 1.3, 0.3), ref: 0.1 }),
    ev: (v) =>
      v == null ? "na" : v >= 0.1 ? "good" : v >= 0.05 ? "warn" : "bad",
    why: "Share of the embedding dimensions the space actually spends on the held-out brain. Above the 0.10 healthy line means it has not collapsed onto a handful of directions.",
  },
  {
    key: "aniso",
    label: "Anisotropy",
    tag: "view",
    src: "geometry",
    field: "anisotropy",
    fmt: "ratio",
    refLabel: "degenerate cone at 1.0 — lower is better",
    scale: (v) => ({ min: 0, max: 1.0, ref: 1.0 }),
    ev: (v) =>
      v == null ? "na" : v <= 0.5 ? "good" : v <= 0.85 ? "warn" : "bad",
    why: "How cone-shaped the cloud is. Near 1.0 is a degenerate cone that can look high-rank yet carry almost no usable structure.",
  },
  {
    key: "clift",
    label: "Content lift over raw",
    tag: "all",
    src: "lift-content",
    fmt: "signed",
    refLabel: "raw band-power line at 0 — sign is the story",
    scale: (v) => {
      const m = Math.max(0.05, Math.abs(v || 0) * 1.3);
      return { min: -m, max: m, ref: 0 };
    },
    ev: (v) =>
      v == null ? "na" : v > 0 ? "good" : v >= -0.005 ? "warn" : "bad",
    why: "Best margin by which the encoder reads word content better than the raw band-power control (linear R²). Negative means the raw control wins and the encoder has not earned its place.",
  },
  {
    key: "ident",
    label: "Identity leak vs raw",
    tag: "all",
    src: "lift-identity",
    fmt: "signed",
    refLabel: "raw band-power line at 0 — want below",
    scale: (v) => {
      const m = Math.max(0.05, Math.abs(v || 0) * 1.3);
      return { min: -m, max: m, ref: 0 };
    },
    ev: (v) =>
      v == null ? "na" : v < 0 ? "good" : v <= 0.005 ? "warn" : "bad",
    why: "How much more (positive) or less (negative) the code exposes WHO read the sentence than the raw control. Below the 0 line means it leaks less identity than raw — what we want.",
  },
  {
    key: "top1",
    label: "Cross-subject retrieval Top-1",
    tag: "view",
    src: "retrieval",
    field: "top1",
    fmt: "pct",
    refField: "chance_top1",
    refLabel: "chance line",
    scale: (v, ref) => ({
      min: 0,
      max: Math.max((v || 0) * 1.6, (ref || 0) * 1.6, 0.1),
      ref: ref || 0,
    }),
    ev: (v, ref) =>
      v == null
        ? "na"
        : ref == null
          ? v > 0
            ? "warn"
            : "bad"
          : v - ref >= 0.02
            ? "good"
            : v - ref >= -0.005
              ? "warn"
              : "bad",
    why: "Given a held-out person’s sentence, how often the closest match among other people shares the same stimulus — judged against its own chance line. This is the north-star capability.",
  },
  {
    key: "rankpct",
    label: "Rank percentile",
    tag: "view",
    src: "retrieval",
    field: "rank_percentile",
    fmt: "pct",
    refLabel: "1.0 = correct match ranked first",
    scale: (v) => ({ min: 0, max: 1.0, ref: 1.0 }),
    ev: (v) =>
      v == null ? "na" : v >= 0.7 ? "good" : v >= 0.5 ? "warn" : "bad",
    why: "Average position of the correct cross-subject match in the ranked gallery of other people. 1.0 means it is always first; 0.5 is no better than random ordering.",
  },
];

function cardValue(c, view) {
  if (c.src === "geometry")
    return view && view.geometry ? nv(view.geometry[c.field]) : null;
  if (c.src === "retrieval")
    return view && view.retrieval ? nv(view.retrieval[c.field]) : null;
  if (c.src === "lift-content") return bestContentLift();
  if (c.src === "lift-identity") return identityLift();
  return null;
}
function cardRef(c, view) {
  if (c.refField && view && view.retrieval)
    return nv(view.retrieval[c.refField]);
  return null;
}

/* ---------- meter (animated fill + named reference tick) ---------- */
function meterHTML(value, sc, cls) {
  const span = sc.max - sc.min || 1;
  const pos = (x) => clamp(((x - sc.min) / span) * 100, 0, 100);
  let h = '<div class="meter">';
  if (value != null) {
    const base = sc.min < 0 ? 0 : sc.min;
    const a = pos(Math.min(base, value)),
      b = pos(Math.max(base, value));
    h +=
      '<div class="fill ' +
      cls +
      '" style="left:' +
      a.toFixed(1) +
      '%;width:0" data-w="' +
      Math.max(0, b - a).toFixed(1) +
      '"></div>';
  }
  if (sc.ref != null)
    h +=
      '<div class="tick" style="left:' + pos(sc.ref).toFixed(1) + '%"></div>';
  h += "</div>";
  return h;
}

/* ---------- render: stat cards ---------- */
const state = { view: (P.view_order && P.view_order[0]) || null };
function renderCards() {
  const view = state.view ? P.views[state.view] : null;
  let h = "";
  CARDS.forEach((c) => {
    const val = cardValue(c, view),
      ref = cardRef(c, view);
    const cls = c.ev(val, ref);
    const sc = c.scale(val, ref);
    const tag = c.tag === "all" ? "all subjects" : view ? view.label : "—";
    const refExtra =
      c.refField && ref != null ? " (" + num(ref, c.fmt) + ")" : "";
    h +=
      '<div class="statcard c-' +
      cls +
      '" tabindex="0">' +
      '<div class="stat-top"><span class="stat-label">' +
      esc(c.label) +
      "</span>" +
      '<span class="stat-tag">' +
      esc(tag) +
      "</span></div>" +
      '<div class="stat-value">' +
      num(val, c.fmt) +
      "</div>" +
      meterHTML(val, sc, cls) +
      '<div class="stat-ref">' +
      esc(c.refLabel) +
      refExtra +
      "</div>" +
      '<div class="stat-why">' +
      esc(c.why) +
      "</div>" +
      "</div>";
  });
  document.getElementById("grid").innerHTML = h;
  animate();
}

/* ---------- render: lift lollipops ---------- */
function renderLift() {
  const lifts = (P.lift || []).filter((l) => nv(l.lift_linear) != null);
  const sec = document.getElementById("liftSection");
  if (!lifts.length) {
    sec.style.display = "none";
    return;
  }
  const m = Math.max(0.02, ...lifts.map((l) => Math.abs(l.lift_linear)));
  const pos = (x) => clamp(((x / m + 1) / 2) * 100, 0, 100);
  let h = "";
  lifts.forEach((l) => {
    const good = l.is_identity ? l.lift_linear < 0 : l.lift_linear > 0;
    const cls = Math.abs(l.lift_linear) < 1e-9 ? "warn" : good ? "good" : "bad";
    const kind = l.is_content
      ? "content ▲ (want +)"
      : l.is_identity
        ? "identity ▼ (want −)"
        : "—";
    const zero = 50,
      p = pos(l.lift_linear),
      a = Math.min(zero, p),
      b = Math.max(zero, p);
    h +=
      '<div class="liftrow">' +
      '<div class="lift-label"><span class="nm">' +
      esc(l.target) +
      "</span>" +
      '<span class="lift-kind">' +
      esc(kind) +
      "</span></div>" +
      '<div class="lift-track"><div class="lift-zero"></div>' +
      '<div class="lift-bar ' +
      cls +
      '" style="left:' +
      a.toFixed(1) +
      '%;width:0" data-w="' +
      (b - a).toFixed(1) +
      '"></div>' +
      '<div class="lift-dot ' +
      cls +
      '" style="left:' +
      p.toFixed(1) +
      '%"></div></div>' +
      '<div class="lift-val ' +
      cls +
      '">' +
      num(l.lift_linear, "signed") +
      "</div>" +
      "</div>";
  });
  document.getElementById("liftBody").innerHTML = h;
  document.getElementById("liftLegend").innerHTML =
    "<span>content ▲ wants a positive lift</span>" +
    "<span>identity ▼ wants a negative lift</span>" +
    "<span>bar and dot mark ZTE − raw against the 0 control line</span>";
  animate();
}

/* ---------- animate every meter/bar fill on paint ---------- */
function animate() {
  requestAnimationFrame(() =>
    requestAnimationFrame(() => {
      document.querySelectorAll("[data-w]").forEach((f) => {
        f.style.width = f.getAttribute("data-w") + "%";
      });
    }),
  );
}

/* ---------- render: segmented view control ---------- */
function renderSeg() {
  const seg = document.getElementById("seg");
  if (!P.view_order || P.view_order.length < 2) {
    seg.style.display = "none";
    return;
  }
  seg.style.display = "inline-flex";
  let h = "";
  P.view_order.forEach((k) => {
    h +=
      '<button class="segbtn' +
      (k === state.view ? " on" : "") +
      '" data-v="' +
      esc(k) +
      '">' +
      esc(P.views[k].label) +
      "</button>";
  });
  seg.innerHTML = h;
  seg.querySelectorAll(".segbtn").forEach(
    (b) =>
      (b.onclick = () => {
        state.view = b.getAttribute("data-v");
        renderSeg();
        renderCards();
      }),
  );
}

/* ---------- render: content-probe positive control ---------- */
function renderProbe() {
  const cp = P.content_probe,
    el = document.getElementById("probe");
  if (!cp) {
    el.style.display = "none";
    return;
  }
  const pass = !!cp.passes;
  el.style.display = "flex";
  el.className = "probe " + (pass ? "ok" : "no");
  el.innerHTML =
    '<span class="probe-pill">' +
    (pass ? "PASS" : "FAIL") +
    "</span>" +
    "<span>Content-probe positive control — raw band-power reads lexical content at R² " +
    num(nv(cp.raw_content_r2_best), "ratio") +
    " (floor " +
    num(nv(cp.floor), "ratio") +
    "). " +
    (pass
      ? "The probe can detect content, so a 0% content budget is a real absence."
      : "The probe cannot read content even from raw features — treat any “content 0%” as untrustworthy until this is fixed.") +
    "</span>";
}

/* ---------- honest one-line verdict, derived from the numbers ---------- */
function verdict() {
  const parts = [];
  const v = state.view ? P.views[state.view] : null;
  const g = v && v.geometry,
    r = v && v.retrieval;
  if (g && (nv(g.effective_rank_ratio) != null || nv(g.anisotropy) != null)) {
    const err = nv(g.effective_rank_ratio),
      an = nv(g.anisotropy);
    const healthy = (err == null || err >= 0.1) && (an == null || an <= 0.85);
    parts.push(
      healthy ? "healthy held-out space" : "held-out space near collapse",
    );
  }
  const cl = bestContentLift();
  if (cl != null)
    parts.push(
      cl > 0.005
        ? "content above raw"
        : cl < -0.005
          ? "content below raw"
          : "content at raw",
    );
  if (r) {
    let lift = nv(r.lift_top1);
    if (lift == null && nv(r.top1) != null && nv(r.chance_top1) != null)
      lift = r.top1 - r.chance_top1;
    if (lift != null)
      parts.push(
        lift > 0.005
          ? "cross-subject retrieval above chance"
          : lift < -0.005
            ? "cross-subject retrieval below chance"
            : "cross-subject retrieval at chance",
      );
  }
  if (!parts.length) return "Scoreboard summary.";
  const s = parts.join(", ");
  let out = s.charAt(0).toUpperCase() + s.slice(1) + ".";
  if (P.is_loso && P.holdout_subject)
    out += " Away game on held-out subject " + P.holdout_subject + ".";
  return out;
}

/* ---------- boot ---------- */
(function () {
  document.getElementById("run").textContent =
    (P.run_name || "ZTE run") + " — held-out scoreboard";
  const hasViews = P.view_order && P.view_order.length;
  const hasLift = P.lift && P.lift.length;
  const hasAny = hasViews || hasLift || P.content_probe;
  if (!hasAny) {
    document.getElementById("verdict").textContent =
      "No scoreboard numbers were available for this run.";
    document.getElementById("cardsSection").style.display = "none";
    document.getElementById("liftSection").style.display = "none";
    document.getElementById("emptySection").style.display = "";
    return;
  }
  document.getElementById("cardsHead").textContent = hasViews
    ? P.views[state.view].label + " headline metrics"
    : "Headline metrics";
  document.getElementById("verdict").textContent = verdict();
  renderProbe();
  renderSeg();
  renderCards();
  renderLift();
})();
