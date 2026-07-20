const P = "__PAYLOAD__";
const M = P.meta,
  N = P.n;

const PAL = {
  light: [
    "#2a78d6",
    "#1baf7a",
    "#eda100",
    "#008300",
    "#4a3aa7",
    "#e34948",
    "#e87ba4",
    "#eb6834",
  ],
  dark: [
    "#3987e5",
    "#199e70",
    "#c98500",
    "#008300",
    "#9085e9",
    "#e66767",
    "#d55181",
    "#d95926",
  ],
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
const INK = {
  light: {
    paper: "#fcfcfb",
    grid: "#e1e0d9",
    axis: "#c3c2b7",
    text: "#0b0b0b",
    muted: "#898781",
    faint: "#d7d7d2",
  },
  dark: {
    paper: "#1a1a19",
    grid: "#2c2c2a",
    axis: "#383835",
    text: "#ffffff",
    muted: "#898781",
    faint: "#3a3a37",
  },
};

const DOMAINS = {};
P.fields.forEach((f) => {
  DOMAINS[f] = [...new Set(M[f])].filter((v) => v !== "").sort();
});
if (!DOMAINS["subject"]) DOMAINS["subject"] = P.subjects.slice();
const HAS_CAT =
  P.fields.includes("category") &&
  DOMAINS["category"] &&
  DOMAINS["category"].length > 1;

// ---- helpers -------------------------------------------------------------
const pick = (arr, idx) => idx.map((i) => arr[i]);
const catColor = (field, val) => {
  const d = DOMAINS[field] || [];
  const i = d.indexOf(val);
  return PAL[state.theme][(i < 0 ? 0 : i) % 8];
};
function hover(i) {
  return (
    `<b>${M.word[i] || "."}</b><br>subject ${M.subject[i]} - ${M.task[i]}` +
    `<br>${M.category[i]} - len ${M.word_len[i]}`
  );
}
function idxVisible() {
  const o = [];
  for (let i = 0; i < N; i++) if (state.visible.has(M.subject[i])) o.push(i);
  return o;
}
function idxWord(w) {
  const o = [];
  for (let i = 0; i < N; i++)
    if (M.word[i] === w && state.visible.has(M.subject[i])) o.push(i);
  return o;
}
function findIdx(w, s) {
  for (let i = 0; i < N; i++)
    if (M.word[i] === w && M.subject[i] === s) return i;
  return -1;
}
function findFirst(w) {
  for (let i = 0; i < N; i++) if (M.word[i] === w) return i;
  return -1;
}
function norm(v) {
  let s = 0;
  for (const x of v) s += x * x;
  s = Math.sqrt(s) || 1;
  return v.map((x) => x / s);
}
function dot(a, b) {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += a[k] * b[k];
  return s;
}
function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// reduced (64-d) normalised vectors + per-subject index lists (primary "et" set)
const RN = (P.reduced.et || []).map(norm);
const SUBJIDX = {};
for (let i = 0; i < N; i++) {
  (SUBJIDX[M.subject[i]] = SUBJIDX[M.subject[i]] || []).push(i);
}

function bestWord() {
  let best = P.words[0] || "",
    bn = -1;
  for (const w in P.word_stats) {
    if (P.word_stats[w].n_subj > bn) {
      bn = P.word_stats[w].n_subj;
      best = w;
    }
  }
  return best;
}

const state = {
  view: 1,
  colorBy: P.fields.includes("subject")
    ? "subject"
    : P.fields[0] || P.numeric_fields[0],
  dims: P.dims_default === 2 ? 2 : 3,
  visible: new Set(P.subjects),
  subj1: P.subjects[0],
  metric1: P.numeric_fields[0] || "word_len",
  word2: bestWord(),
  wordT: bestWord(),
  subjA: P.subjects[0],
  subjB: P.subjects[1] || P.subjects[0],
  wordN: bestWord(),
  subjN: "any",
  kN: 12,
  embSet: "et",
  sent6: 0,
  cat7: (DOMAINS["category"] && DOMAINS["category"][0]) || "",
  deident: 0,
  subjH: P.subjects[0],
  kAnchor: 8,
  calT: 0,
  guide: false,
  theme:
    window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light",
  lbSort: "default",
  lbDir: -1,
};

// ---- in-browser analytics: verdict banners -------------------------------
function verdict(delta) {
  if (delta == null) return { label: "n/a", cls: "na" };
  if (delta < 0.03) return { label: "not clustered", cls: "bad" };
  if (delta < 0.1) return { label: "weakly clustered", cls: "warn" };
  return { label: "clustered", cls: "good" };
}
// mean cosine of random point pairs in the reduced space (in-browser baseline)
let RAND_BL = 0;
function randBaseline() {
  let s = 0,
    c = 0;
  if (N < 2) return 0;
  for (let t = 0; t < 4000; t++) {
    const a = (Math.random() * N) | 0,
      b = (Math.random() * N) | 0;
    if (a !== b) {
      s += dot(RN[a], RN[b]);
      c++;
    }
  }
  return c ? s / c : 0;
}
// same-word across subjects: mean over words of the pairwise cosine of per-subject centroids
function sameWordAcross() {
  let s = 0,
    nw = 0,
    nsub = 0;
  const byWord = {};
  for (let i = 0; i < N; i++) {
    const w = M.word[i];
    if (!w) continue;
    (byWord[w] = byWord[w] || {})[M.subject[i]] = byWord[w][M.subject[i]] || [];
    byWord[w][M.subject[i]].push(i);
  }
  for (const w in byWord) {
    const subs = Object.keys(byWord[w]);
    if (subs.length < 2) continue;
    const cents = subs.map((su) => {
      const ids = byWord[w][su];
      const c = new Array(RN[0].length).fill(0);
      ids.forEach((i) => {
        const r = RN[i];
        for (let k = 0; k < c.length; k++) c[k] += r[k];
      });
      return norm(c);
    });
    let sm = 0,
      np = 0;
    for (let a = 0; a < cents.length; a++)
      for (let b = a + 1; b < cents.length; b++) {
        sm += dot(cents[a], cents[b]);
        np++;
      }
    s += sm / np;
    nw++;
    nsub += subs.length;
  }
  return nw ? { cos: s / nw, words: nw } : null;
}
// same-category across subjects: sampled cosine of cross-subject same-category pairs
function sameCategoryAcross() {
  if (!HAS_CAT) return null;
  const byCat = {};
  for (let i = 0; i < N; i++) {
    const c = M.category[i];
    if (!c) continue;
    (byCat[c] = byCat[c] || []).push(i);
  }
  const cats = Object.keys(byCat).filter((c) => {
    const subs = new Set(byCat[c].map((i) => M.subject[i]));
    return subs.size >= 2;
  });
  if (!cats.length) return null;
  let s = 0,
    c = 0,
    tries = 0;
  while (c < 3000 && tries < 160000) {
    tries++;
    const cat = cats[(Math.random() * cats.length) | 0],
      pool = byCat[cat];
    const i = pool[(Math.random() * pool.length) | 0],
      j = pool[(Math.random() * pool.length) | 0];
    if (i !== j && M.subject[i] !== M.subject[j]) {
      s += dot(RN[i], RN[j]);
      c++;
    }
  }
  return c ? { cos: s / c, pairs: c } : null;
}

function vcls(s) {
  return s === "clustered"
    ? "good"
    : s === "weakly clustered"
      ? "warn"
      : s === "not clustered"
        ? "bad"
        : "na";
}
function fmt(x, d) {
  return x == null || !isFinite(x) ? "--" : (+x).toFixed(d == null ? 3 : d);
}
function pctOr(x) {
  return x == null || !isFinite(x) ? "--" : (x * 100).toFixed(0) + "%";
}
function transferVerdict(gap) {
  return gap == null || !isFinite(gap)
    ? { label: "n/a", cls: "na" }
    : gap < 0.02
      ? { label: "rarely transfers", cls: "bad" }
      : gap < 0.1
        ? { label: "sometimes transfers", cls: "warn" }
        : { label: "often transfers", cls: "good" };
}
function bannerCard(title, num, sub, vlabel, cls, cap) {
  return `<div class="banner">
     <div class="bt">${title}</div>
     <div class="brow"><span class="bnum">${num}</span>
       ${sub ? `<span class="bsub">${sub}</span>` : ""}
       <span class="verdict ${cls}">${vlabel}</span></div>
     <div class="bcap">${cap}</div>
   </div>`;
}

function renderBanners() {
  RAND_BL = randBaseline();
  const box = document.getElementById("banners");
  box.innerHTML = "";
  const EM = P.emergence;
  const CS =
    EM && EM.cross_subject && EM.cross_subject.applicable
      ? EM.cross_subject
      : null;

  // ---- Banner 1: same word across subjects --------------------------------
  const w = sameWordAcross();
  const liveW = w ? w.cos : null,
    liveWd = w ? verdict(w.cos - RAND_BL) : verdict(null);
  const sw = CS ? CS.same_word : null;
  if (sw) {
    box.insertAdjacentHTML(
      "beforeend",
      bannerCard(
        "Same word across subjects",
        fmt(sw.mean_cosine),
        `vs ${fmt(sw.random_baseline)} random`,
        sw.verdict,
        vcls(sw.verdict),
        `Full-embedding-space mean cosine of the same word read by different subjects, minus the random
       cross-subject baseline (&Delta;=${fmt(sw.gap)}).
       <b>Live estimate (PCA space):</b> ${fmt(liveW)} vs ${fmt(RAND_BL)} random &rarr; ${liveWd.label}.`,
      ),
    );
  } else {
    box.insertAdjacentHTML(
      "beforeend",
      bannerCard(
        "Same word across subjects",
        fmt(liveW),
        `vs ${fmt(RAND_BL)} random`,
        liveWd.label,
        liveWd.cls,
        `Mean cosine of the same word's per-subject centroids (${w ? w.words : 0} words read by &ge;2 people),
       minus the random-pair baseline. &Delta;=${w ? fmt(w.cos - RAND_BL) : "--"} <i>(in-browser PCA-space estimate)</i>.`,
      ),
    );
  }

  // ---- Banner 2: do meanings cluster (same category) ----------------------
  const c = sameCategoryAcross();
  const liveC = c ? c.cos : null,
    liveCd = c ? verdict(c.cos - RAND_BL) : null;
  const sm = CS ? CS.same_meaning : null;
  if (sm) {
    box.insertAdjacentHTML(
      "beforeend",
      bannerCard(
        "Do meanings cluster across people?",
        fmt(sm.mean_cosine),
        `vs ${fmt(sm.random_baseline)} random`,
        sm.verdict,
        vcls(sm.verdict),
        `Full-embedding-space mean cosine of <b>same-category, different-subject</b> pairs (meaning proxy =
       <code>category</code>), minus random (&Delta;=${fmt(sm.gap)}).` +
          (c
            ? ` <b>Live estimate (PCA space):</b> ${fmt(liveC)} &rarr; ${liveCd.label}.`
            : ""),
      ),
    );
  } else if (c) {
    box.insertAdjacentHTML(
      "beforeend",
      bannerCard(
        "Do meanings cluster across people?",
        fmt(liveC),
        `vs ${fmt(RAND_BL)} random`,
        liveCd.label,
        liveCd.cls,
        `Mean cosine of <b>same-category, different-subject</b> pairs (meaning proxy = <code>category</code>;
       ${c.pairs} sampled), minus random. &Delta;=${fmt(c.cos - RAND_BL)} <i>(in-browser PCA-space estimate)</i>.`,
      ),
    );
  } else {
    box.insertAdjacentHTML(
      "beforeend",
      bannerCard(
        "Do meanings cluster across people?",
        "--",
        "",
        "no category column",
        "na",
        `Needs a <code>category</code> label read by &ge;2 subjects to use it as a meaning proxy.
       Falling back to the same-word measure on the left.`,
      ),
    );
  }

  // ---- Banner 3: can we translate a thought (analogy) ---------------------
  const hr = LB.length ? LB.filter((x) => x.hit).length / LB.length : null;
  const liveHd = transferVerdict(hr);
  const an = EM && EM.analogy ? EM.analogy : null;
  const top1 = an ? an.subject_transfer_top1 : null,
    chance = an ? an.chance_top1 : null;
  if (an && top1 != null && isFinite(top1)) {
    const gap = isFinite(chance) ? top1 - chance : top1;
    const ad = transferVerdict(gap);
    box.insertAdjacentHTML(
      "beforeend",
      bannerCard(
        "Can we translate a thought between people?",
        pctOr(top1),
        `vs ${pctOr(chance)} chance`,
        ad.label,
        ad.cls,
        `Full-embedding-space subject-transfer Top-1 hit rate: v = emb(t,A) &minus; centroid(A) + centroid(B)
       retrieves <i>t</i> read by B.` +
          (hr == null
            ? ""
            : ` <b>Live estimate (PCA space):</b> ${pctOr(hr)} over ${LB.length} analogies &rarr; ${liveHd.label}.`),
      ),
    );
  } else {
    box.insertAdjacentHTML(
      "beforeend",
      bannerCard(
        "Can we translate a thought between people?",
        pctOr(hr),
        `${LB.length} analogies`,
        liveHd.label,
        liveHd.cls,
        `Share of A&rarr;B transfers whose nearest neighbour is the right word for B
       <i>(in-browser PCA-space estimate)</i>. Open <b>View 3</b> for the ranked leaderboard.`,
      ),
    );
  }

  // ---- provenance caption -------------------------------------------------
  box.insertAdjacentHTML(
    "beforeend",
    EM
      ? `<div class="bsrc">Headline figures are the canonical <b>full-embedding-space</b> values from
        <code>metrics.json</code> (emergence report); "live estimate" numbers are computed in-browser from
        the PCA-reduced vectors and may differ slightly.</div>`
      : `<div class="bsrc">Figures are in-browser estimates over the PCA-reduced vectors. When an emergence
        report is present, these banners instead headline the canonical full-embedding-space values from
        <code>metrics.json</code>.</div>`,
  );
}

function renderGuide() {
  const g = document.getElementById("guide");
  if (!state.guide) {
    g.classList.add("hide");
    return;
  }
  g.classList.remove("hide");
  const subjLg = P.subjects
    .slice(0, 8)
    .map(
      (s, i) =>
        `<span class="lg"><span class="sw" style="background:${PAL[state.theme][i % 8]}"></span>${esc(s)}</span>`,
    )
    .join("");
  g.innerHTML = `<b>What am I looking at?</b> Each point is <b>one word read by one person</b>, placed by
     their EEG (brain) response. ZTE's goal is that the <b>same meaning read by different people</b>
     lands in the same place &mdash; the way word embeddings put "cat" and "dog" near each other.
     The banners below measure whether that is happening yet; the five views and colours let you
     test it yourself. <span style="color:var(--muted)">Honest status: ZTE v1 largely encodes
     <i>who</i> is reading, not <i>what</i> &mdash; expect weak cross-subject clustering.</span>
     <div class="lgd"><b style="color:var(--ink2)">subjects:</b>${subjLg}</div>`;
}

// ---- analogy leaderboard (computed once, in-browser, on the primary set) --
let LB = [];
function computeLeaderboard() {
  const R = P.reduced.et,
    CENT = P.centroids.et,
    cands = P.analogy_candidates || [];
  LB = cands
    .map((c) => {
      const A = CENT[c.A],
        B = CENT[c.B],
        base = R[c.ai];
      if (!A || !B || !base) return null;
      const v = norm(base.map((x, k) => x - A[k] + B[k]));
      const bl = SUBJIDX[c.B] || [];
      let best = -1,
        bestSim = -2,
        trueSim = -2;
      for (const i of bl) {
        const s = dot(RN[i], v);
        if (i === c.bi) trueSim = s;
        if (s > bestSim) {
          bestSim = s;
          best = i;
        }
      }
      let rank = 1;
      for (const i of bl) {
        if (i !== c.bi && dot(RN[i], v) > trueSim) rank++;
      }
      return {
        t: c.t,
        A: c.A,
        B: c.B,
        ai: c.ai,
        bi: c.bi,
        hit: best >= 0 && M.word[best] === c.t,
        nn: best >= 0 ? M.word[best] : "-",
        sim: bestSim,
        rank,
      };
    })
    .filter(Boolean);
  sortLB();
}
function sortLB() {
  const k = state.lbSort,
    d = state.lbDir;
  const key = {
    default: (x) => (x.hit ? 1e6 + x.sim : x.sim),
    word: (x) => x.t,
    ab: (x) => x.A + x.B,
    hit: (x) => (x.hit ? 1 : 0),
    nn: (x) => x.nn,
    rank: (x) => -x.rank,
    sim: (x) => x.sim,
  };
  const f = key[k] || key.default;
  LB.sort((a, b) => {
    const x = f(a),
      y = f(b);
    if (x < y) return -d;
    if (x > y) return d;
    return 0;
  });
}
function renderLeaderboard() {
  const hr = LB.length ? LB.filter((x) => x.hit).length / LB.length : 0;
  const hd = document.getElementById("hitrate");
  const cls = hr < 0.15 ? "bad" : hr < 0.4 ? "warn" : "good";
  hd.className = "big " + cls;
  hd.innerHTML = `hit-rate ${(hr * 100).toFixed(0)}% <span style="color:var(--muted);font-weight:500">(${LB.filter((x) => x.hit).length}/${LB.length})</span>`;
  const cols = [
    ["word", "word t"],
    ["ab", "A -> B"],
    ["hit", "hit"],
    ["nn", "v’s NN"],
    ["rank", "true rank"],
    ["sim", "cos(v,NN)"],
  ];
  let h =
    "<tr>" +
    cols
      .map(([k, l]) => {
        const arrow =
          state.lbSort === k ? (state.lbDir < 0 ? "up" : "down") : ""; // note: default not on a col
        return `<th data-k="${k}" class="${arrow}">${l}</th>`;
      })
      .join("") +
    "</tr>";
  const rows = LB.slice(0, 120)
    .map(
      (x, i) =>
        `<tr class="clk" data-i="${LB.indexOf(x)}">
       <td>${esc(x.t)}</td><td>${esc(x.A)} -> ${esc(x.B)}</td>
       <td><span class="pill ${x.hit ? "hit" : "miss"}">${x.hit ? "✓" : "✗"}</span></td>
       <td>${esc(x.nn)}</td><td class="num">${x.rank}</td><td class="num">${x.sim.toFixed(3)}</td>
     </tr>`,
    )
    .join("");
  const t = document.getElementById("leaderboard");
  t.innerHTML = h + rows;
  t.querySelectorAll("th").forEach(
    (th) =>
      (th.onclick = () => {
        const k = th.dataset.k;
        if (state.lbSort === k) state.lbDir = -state.lbDir;
        else {
          state.lbSort = k;
          state.lbDir = -1;
        }
        sortLB();
        renderLeaderboard();
      }),
  );
  t.querySelectorAll("tr.clk").forEach(
    (tr) => (tr.onclick = () => openAnalogy(LB[+tr.dataset.i])),
  );
}
function openAnalogy(x) {
  if (!x) return;
  state.wordT = x.t;
  state.subjA = x.A;
  state.subjB = x.B;
  const wt = document.getElementById("wordT");
  if (wt) wt.value = x.t;
  const sa = document.getElementById("subjA");
  if (sa) sa.value = x.A;
  const sb = document.getElementById("subjB");
  if (sb) sb.value = x.B;
  setView(3);
}

// ---- semantic neighbourhood ----------------------------------------------
function neighbourQuery() {
  return state.subjN === "any"
    ? findFirst(state.wordN)
    : findIdx(state.wordN, state.subjN);
}
function neighbours(qi, k) {
  const q = RN[qi],
    out = [];
  for (let i = 0; i < N; i++) {
    if (i === qi) continue;
    out.push([i, dot(RN[i], q)]);
  }
  out.sort((a, b) => b[1] - a[1]);
  return out.slice(0, k);
}
function renderNeighTable(qi, nn) {
  const t = document.getElementById("neightable");
  if (qi < 0) {
    t.innerHTML =
      '<tr><td class="num" style="color:var(--muted)">token not found</td></tr>';
    document.getElementById("coherence").textContent = "";
    return;
  }
  const qw = M.word[qi],
    qc = M.category[qi];
  let sw = 0,
    sc = 0;
  const rows = nn
    .map(([i, s]) => {
      const isw = M.word[i] === qw,
        isc = HAS_CAT && M.category[i] === qc;
      if (isw) sw++;
      if (isc) sc++;
      const tag = isw
        ? '<span class="pill hit">same word</span>'
        : isc
          ? `<span class="pill" style="background:color-mix(in srgb,var(--accent) 16%,transparent);color:var(--accent)">same cat</span>`
          : "";
      return `<tr><td>${esc(M.word[i])}</td><td>${esc(M.subject[i])}</td><td>${esc(M.category[i])}</td>
      <td class="num">${s.toFixed(3)}</td><td>${tag}</td></tr>`;
    })
    .join("");
  t.innerHTML =
    "<tr><th>neighbour word</th><th>subject</th><th>category</th><th>cosine</th><th></th></tr>" +
    rows;
  const k = nn.length || 1;
  const wc = M.word.filter((w) => w === qw).length,
    cc = HAS_CAT ? M.category.filter((c) => c === qc).length : 0;
  const chW = (wc - 1) / Math.max(1, N - 1),
    chC = (cc - 1) / Math.max(1, N - 1);
  const co = document.getElementById("coherence");
  const lift = sw / k / Math.max(1e-6, chW);
  const cls = lift > 3 ? "good" : lift > 1.3 ? "warn" : "bad";
  co.className = "big " + cls;
  co.innerHTML =
    `same-word ${((100 * sw) / k).toFixed(0)}% <span style="color:var(--muted);font-weight:500">(chance ${(100 * chW).toFixed(1)}%)</span>` +
    (HAS_CAT
      ? ` &middot; same-cat ${((100 * sc) / k).toFixed(0)}% <span style="color:var(--muted);font-weight:500">(chance ${(100 * chC).toFixed(1)}%)</span>`
      : "");
}

// ---- plotly trace builders ----------------------------------------------
let CUR = null;
function mk(idx, o) {
  o = o || {};
  const is3 = state.dims === 3;
  const t = {
    type: is3 ? "scatter3d" : "scattergl",
    mode: "markers",
    x: pick(CUR.x, idx),
    y: pick(CUR.y, idx),
    text: idx.map(hover),
    hoverinfo: "text",
    name: o.name || "",
    showlegend: o.showlegend !== false && !!o.name,
    marker: {
      size: o.size || (is3 ? 3.4 : 7.5),
      opacity: o.opacity == null ? 0.9 : o.opacity,
      line: { width: 0 },
    },
  };
  if (is3) t.z = pick(CUR.z, idx);
  if (o.color) t.marker.color = o.color;
  if (o.cvals) {
    t.marker.color = o.cvals;
    t.marker.colorscale = o.colorscale;
    t.marker.showscale = !!o.showscale;
    t.marker.colorbar = o.colorbar;
    t.marker.cmin = o.cmin;
    t.marker.cmax = o.cmax;
  }
  if (o.symbol) t.marker.symbol = o.symbol;
  return t;
}
function markerAt(pt, o) {
  const is3 = state.dims === 3,
    ink = INK[state.theme];
  const t = {
    type: is3 ? "scatter3d" : "scatter",
    mode: o.text ? "markers+text" : "markers",
    x: [pt[0]],
    y: [pt[1]],
    text: o.text ? [o.text] : undefined,
    textposition: "top center",
    textfont: { color: ink.text, size: 11 },
    marker: {
      size: o.size || (is3 ? 9 : 15),
      color: o.color,
      symbol: o.symbol || "circle",
      line: { width: 2, color: ink.paper },
      opacity: 1,
    },
    name: o.name || "",
    showlegend: o.showlegend !== false,
    hovertext: [o.hover || o.text || o.name],
    hoverinfo: "text",
  };
  if (is3) t.z = [pt[2]];
  return t;
}
function segment(a, b, o) {
  const is3 = state.dims === 3;
  const t = {
    type: is3 ? "scatter3d" : "scatter",
    mode: "lines",
    x: [a[0], b[0]],
    y: [a[1], b[1]],
    line: { width: o.width || 5, color: o.color, dash: o.dash || "solid" },
    hoverinfo: "skip",
    showlegend: !!o.name,
    name: o.name || "",
  };
  if (is3) t.z = [a[2], b[2]];
  return t;
}
function avg(a) {
  let s = 0;
  for (const x of a) s += x;
  return a.length ? s / a.length : 0;
}
function pathTrace(seq, o) {
  const is3 = state.dims === 3;
  const t = {
    type: is3 ? "scatter3d" : "scatter",
    mode: "lines+markers",
    x: pick(CUR.x, seq),
    y: pick(CUR.y, seq),
    text: seq.map(hover),
    hoverinfo: "text",
    line: { width: is3 ? 4 : 3, color: o.color, shape: "linear" },
    marker: {
      size: is3 ? 4.6 : 9,
      color: o.color,
      line: { width: 1.4, color: INK[state.theme].paper },
    },
    name: o.name,
    showlegend: o.showlegend !== false,
  };
  if (is3) t.z = pick(CUR.z, seq);
  return t;
}
function polyline(pts, o) {
  const is3 = state.dims === 3;
  const t = {
    type: is3 ? "scatter3d" : "scatter",
    mode: "lines",
    x: pts.map((p) => p[0]),
    y: pts.map((p) => p[1]),
    hoverinfo: "skip",
    showlegend: false,
    line: { width: o.width || 2, color: o.color, dash: o.dash || "dot" },
  };
  if (is3) t.z = pts.map((p) => p[2]);
  return t;
}
function categoryCohesion(cat) {
  const pool = [];
  for (let i = 0; i < N; i++)
    if (M.category[i] === cat && state.visible.has(M.subject[i])) pool.push(i);
  const subs = new Set(pool.map((i) => M.subject[i]));
  let s = 0,
    c = 0;
  for (let t = 0; t < 3000 && pool.length > 1; t++) {
    const i = pool[(Math.random() * pool.length) | 0],
      j = pool[(Math.random() * pool.length) | 0];
    if (i !== j && M.subject[i] !== M.subject[j]) {
      s += dot(RN[i], RN[j]);
      c++;
    }
  }
  return { cos: c ? s / c : null, nsubj: subs.size, n: pool.length };
}
// ---- anchor-calibration (3-D Procrustes / Kabsch) for the "new brain" demo ----
function _t3(M) {
  return [
    [M[0][0], M[1][0], M[2][0]],
    [M[0][1], M[1][1], M[2][1]],
    [M[0][2], M[1][2], M[2][2]],
  ];
}
function _m3(A, B) {
  const R = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += A[i][k] * B[k][j];
      R[i][j] = s;
    }
  return R;
}
function _det3(M) {
  return (
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
  );
}
function _jac3(A0) {
  const A = A0.map((r) => r.slice());
  let V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let it = 0; it < 60; it++) {
    let p = 0,
      q = 1,
      mx = Math.abs(A[0][1]);
    if (Math.abs(A[0][2]) > mx) {
      mx = Math.abs(A[0][2]);
      p = 0;
      q = 2;
    }
    if (Math.abs(A[1][2]) > mx) {
      mx = Math.abs(A[1][2]);
      p = 1;
      q = 2;
    }
    if (mx < 1e-14) break;
    const phi = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]),
      c = Math.cos(phi),
      s = Math.sin(phi);
    for (let i = 0; i < 3; i++) {
      const a = A[i][p],
        b = A[i][q];
      A[i][p] = c * a - s * b;
      A[i][q] = s * a + c * b;
    }
    for (let i = 0; i < 3; i++) {
      const a = A[p][i],
        b = A[q][i];
      A[p][i] = c * a - s * b;
      A[q][i] = s * a + c * b;
    }
    for (let i = 0; i < 3; i++) {
      const a = V[i][p],
        b = V[i][q];
      V[i][p] = c * a - s * b;
      V[i][q] = s * a + c * b;
    }
  }
  return { vals: [A[0][0], A[1][1], A[2][2]], V };
}
function _svd3(H) {
  const { vals, V } = _jac3(_m3(_t3(H), H));
  const idx = [0, 1, 2].sort((a, b) => vals[b] - vals[a]);
  const S = idx.map((i) => Math.sqrt(Math.max(0, vals[i])));
  const Vs = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) Vs[r][c] = V[r][idx[c]];
  const U = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let c = 0; c < 3; c++) {
    const sv = S[c] || 1e-9;
    for (let r = 0; r < 3; r++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += H[r][k] * Vs[k][c];
      U[r][c] = s / sv;
    }
  }
  return { U, S, V: Vs };
}
function kabsch(src, tgt) {
  const n = src.length;
  if (n < 3) return null;
  const cen = (a) => {
    const m = [0, 0, 0];
    a.forEach((p) => {
      m[0] += p[0];
      m[1] += p[1];
      m[2] += p[2];
    });
    return [m[0] / a.length, m[1] / a.length, m[2] / a.length];
  };
  const ms = cen(src),
    mt = cen(tgt);
  const P = src.map((p) => [p[0] - ms[0], p[1] - ms[1], p[2] - ms[2]]),
    Q = tgt.map((p) => [p[0] - mt[0], p[1] - mt[1], p[2] - mt[2]]);
  const H = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++)
      for (let b = 0; b < 3; b++) H[a][b] += P[i][a] * Q[i][b];
  const { U, S, V } = _svd3(H);
  let Rm = _m3(V, _t3(U));
  if (_det3(Rm) < 0) {
    const V2 = V.map((r) => r.slice());
    for (let a = 0; a < 3; a++) V2[a][2] = -V2[a][2];
    Rm = _m3(V2, _t3(U));
  }
  let vp = 0;
  P.forEach((p) => (vp += p[0] * p[0] + p[1] * p[1] + p[2] * p[2]));
  return { R: Rm, s: (S[0] + S[1] + S[2]) / (vp || 1), ms, mt };
}
function applyKabsch(k, p) {
  const d = [p[0] - k.ms[0], p[1] - k.ms[1], p[2] - k.ms[2]];
  const r = [
    k.R[0][0] * d[0] + k.R[0][1] * d[1] + k.R[0][2] * d[2],
    k.R[1][0] * d[0] + k.R[1][1] * d[1] + k.R[1][2] * d[2],
    k.R[2][0] * d[0] + k.R[2][1] * d[1] + k.R[2][2] * d[2],
  ];
  return [k.s * r[0] + k.mt[0], k.s * r[1] + k.mt[1], k.s * r[2] + k.mt[2]];
}
function centroidOf(ids) {
  let x = 0,
    y = 0,
    z = 0;
  ids.forEach((i) => {
    x += CUR.x[i];
    y += CUR.y[i];
    z += CUR.z ? CUR.z[i] : 0;
  });
  const n = ids.length || 1;
  return [x / n, y / n, z / n];
}
function calibrationTransform(H, K) {
  const wH = {},
    wO = {};
  for (let i = 0; i < N; i++) {
    const w = M.word[i];
    if (!w) continue;
    if (M.subject[i] === H) {
      (wH[w] = wH[w] || []).push(i);
    } else {
      (wO[w] = wO[w] || []).push(i);
    }
  }
  const shared = Object.keys(wH).filter((w) => wO[w] && wO[w].length);
  if (shared.length < 3) return null;
  shared.sort(
    (a, b) =>
      Math.min(wH[b].length, wO[b].length) -
      Math.min(wH[a].length, wO[a].length),
  );
  const K2 = Math.max(3, Math.min(K | 0 || 8, shared.length - 1));
  const anchors = shared.slice(0, K2),
    testW = shared.slice(K2);
  const cal = kabsch(
    anchors.map((w) => centroidOf(wH[w])),
    anchors.map((w) => centroidOf(wO[w])),
  );
  if (!cal) return null;
  const dist = (a, b) =>
    Math.hypot(a[0] - b[0], a[1] - b[1], (a[2] || 0) - (b[2] || 0));
  const tw = testW.length ? testW : anchors;
  let gb = 0,
    ga = 0;
  tw.forEach((w) => {
    const hc = centroidOf(wH[w]),
      oc = centroidOf(wO[w]);
    gb += dist(hc, oc);
    ga += dist(applyKabsch(cal, hc), oc);
  });
  cal.k = anchors.length;
  cal.gapBefore = gb / tw.length;
  cal.gapAfter = ga / tw.length;
  cal.anchors = anchors;
  cal.wH = wH;
  cal.wO = wO;
  return cal;
}
function rawScatter(xs, ys, zs, ids, o) {
  const is3 = state.dims === 3;
  const t = {
    type: is3 ? "scatter3d" : "scattergl",
    mode: "markers",
    x: xs,
    y: ys,
    text: ids.map(hover),
    hoverinfo: "text",
    name: o.name,
    showlegend: o.showlegend !== false,
    marker: {
      size: o.size || (is3 ? 4 : 8),
      color: o.color,
      opacity: o.opacity == null ? 0.92 : o.opacity,
      line: { width: 0 },
    },
  };
  if (is3) t.z = zs;
  return t;
}
let animRAF = null;
function animate(setFn, from, to, ms) {
  if (animRAF) cancelAnimationFrame(animRAF);
  const start = performance.now();
  function step(now) {
    let p = Math.min(1, (now - start) / ms);
    const e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
    setFn(from + (to - from) * e);
    render();
    if (p < 1) animRAF = requestAnimationFrame(step);
    else animRAF = null;
  }
  animRAF = requestAnimationFrame(step);
}
function colouredTraces(idx, field, opt) {
  opt = opt || {};
  const traces = [];
  if (P.numeric_fields.includes(field)) {
    const vals = idx.map((i) => M[field][i]);
    let mn = Math.min(...vals),
      mx = Math.max(...vals);
    if (mn === mx) {
      mx = mn + 1;
    }
    traces.push(
      mk(idx, {
        cvals: vals,
        colorscale: SEQ[state.theme],
        showscale: true,
        cmin: mn,
        cmax: mx,
        size: opt.size,
        opacity: opt.opacity,
        colorbar: {
          title: { text: field, side: "right" },
          thickness: 12,
          len: 0.6,
          x: 1.0,
          outlinewidth: 0,
          tickfont: { color: INK[state.theme].muted, size: 10 },
        },
      }),
    );
  } else {
    const dom =
      DOMAINS[field] || [...new Set(idx.map((i) => M[field][i]))].sort();
    dom.forEach((v) => {
      const sub = idx.filter((i) => M[field][i] === v);
      if (!sub.length) return;
      traces.push(
        mk(sub, {
          color: catColor(field, v),
          name: String(v),
          showlegend: opt.legend !== false,
          size: opt.size,
          opacity: opt.opacity,
        }),
      );
    });
  }
  return traces;
}
function baseLayout() {
  const ink = INK[state.theme],
    is3 = state.dims === 3;
  const L = {
    paper_bgcolor: ink.paper,
    plot_bgcolor: ink.paper,
    font: {
      color: ink.text,
      family: 'system-ui,-apple-system,"Segoe UI",sans-serif',
      size: 12,
    },
    margin: { l: 0, r: 0, t: 6, b: 0 },
    showlegend: true,
    uirevision: "keep",
    legend: {
      bgcolor: "rgba(0,0,0,0)",
      font: { color: ink.text, size: 11 },
      itemsizing: "constant",
      orientation: "v",
      x: 0,
      y: 1,
      xanchor: "left",
      yanchor: "top",
    },
    hoverlabel: {
      bgcolor: ink.paper,
      bordercolor: ink.axis,
      font: { color: ink.text, size: 12 },
    },
  };
  if (is3) {
    const ax = () => ({
      showbackground: false,
      gridcolor: ink.grid,
      zerolinecolor: ink.axis,
      color: ink.muted,
      showspikes: false,
      title: { text: "" },
    });
    L.scene = {
      xaxis: ax(),
      yaxis: ax(),
      zaxis: ax(),
      bgcolor: ink.paper,
      camera: { eye: { x: 1.5, y: 1.4, z: 1.1 } },
    };
  } else {
    const ax = (t) => ({
      gridcolor: ink.grid,
      zeroline: false,
      showline: false,
      color: ink.muted,
      title: { text: t, font: { size: 11, color: ink.muted } },
    });
    L.xaxis = ax("PC1");
    L.yaxis = ax("PC2");
  }
  return L;
}

// ---- the views -----------------------------------------------------------
function render() {
  const set = state.view === 5 && P.has_eeg_only ? state.embSet : "et";
  CUR = P.coords[set];
  const R = P.reduced[set],
    CENT = P.centroids[set],
    CENTP = P.centroids_proj[set];
  const ink = INK[state.theme];
  let traces = [],
    story = "",
    stat = [];

  // "remove reader identity" morph: slide each point toward its subject-centred position
  if (
    state.deident > 0 &&
    (state.view === 1 || state.view === 2 || state.view === 7)
  ) {
    const t = state.deident,
      nx = new Array(N),
      ny = new Array(N),
      nz = new Array(N);
    for (let i = 0; i < N; i++) {
      const c = CENTP[M.subject[i]] || [0, 0, 0];
      nx[i] = CUR.x[i] - t * c[0];
      ny[i] = CUR.y[i] - t * c[1];
      nz[i] = (CUR.z ? CUR.z[i] : 0) - t * (c[2] || 0);
    }
    CUR = { x: nx, y: ny, z: nz };
  }

  if (state.view === 1) {
    const sel = state.subj1;
    const bg = idxVisible().filter((i) => M.subject[i] !== sel);
    const fg = idxVisible().filter((i) => M.subject[i] === sel);
    if (bg.length)
      traces.push(
        mk(bg, {
          color: ink.faint,
          opacity: 0.45,
          showlegend: false,
          size: state.dims === 3 ? 2.6 : 5.5,
        }),
      );
    traces = traces.concat(
      colouredTraces(fg, state.metric1, {
        legend: true,
        size: state.dims === 3 ? 4.2 : 9,
      }),
    );
    story =
      `Subject <b>${sel}</b> in colour (${fg.length} words), shaded by <b>${state.metric1}</b>; ` +
      `everyone else greyed. One reader is a coherent sub-cloud - identity is legible in the space.`;
    stat = [
      ["words", fg.length],
      ["subjects hidden", P.subjects.length - state.visible.size],
    ];
  } else if (state.view === 2) {
    const w = state.word2;
    const base = idxVisible();
    traces.push(
      mk(base, {
        color: ink.faint,
        opacity: 0.28,
        showlegend: false,
        size: state.dims === 3 ? 2.4 : 5,
      }),
    );
    const hit = idxWord(w);
    traces = traces.concat(
      colouredTraces(hit, "subject", {
        legend: true,
        size: state.dims === 3 ? 5.5 : 12,
      }),
    );
    // dotted loop through each reader's mean position for this word -> the cross-subject spread
    const wcps = [];
    P.subjects.forEach((s) => {
      if (!state.visible.has(s)) return;
      const ids = hit.filter((i) => M.subject[i] === s);
      if (!ids.length) return;
      wcps.push([
        avg(pick(CUR.x, ids)),
        avg(pick(CUR.y, ids)),
        state.dims === 3 ? avg(pick(CUR.z, ids)) : 0,
      ]);
    });
    if (wcps.length > 1)
      traces.push(
        polyline(wcps.concat([wcps[0]]), {
          color: ink.muted,
          width: 1.6,
          dash: "dot",
        }),
      );
    const st = P.word_stats[w];
    const bl = P.random_baseline;
    story =
      `The word <b>"${w}"</b> read across brains (${hit.length} occurrences). If a thought code were ` +
      `subject-invariant these would coincide - instead they scatter. ` +
      `Cross-subject similarity barely beats unrelated thoughts: <b>the core finding.</b>`;
    if (st) {
      stat = [
        ["mean cos - across brains", st.mean_cos.toFixed(3)],
        ["random baseline", bl.toFixed(3)],
        ["subjects", st.n_subj],
      ];
      stat._flag = st.mean_cos < bl + 0.15 ? "warn" : "good";
    } else {
      stat = [["note", "read by <2 subjects"]];
    }
  } else if (state.view === 3) {
    const t = state.wordT,
      A = state.subjA,
      B = state.subjB;
    traces.push(
      mk(idxVisible(), {
        color: ink.faint,
        opacity: 0.22,
        showlegend: false,
        size: state.dims === 3 ? 2.2 : 4.5,
      }),
    );
    const ai = findIdx(t, A),
      bi = findIdx(t, B);
    if (ai < 0 || A === B) {
      story =
        `Pick a word <i>t</i> read by <b>${A}</b>, and a different target brain B, ` +
        `or just click a row in the leaderboard below. ` +
        (A === B
          ? `Source and target must differ.`
          : `"${t}" was not found for ${A}.`);
      stat = [["status", "-"]];
    } else {
      const aP = [CUR.x[ai], CUR.y[ai], CUR.z[ai]];
      const cAp = CENTP[A],
        cBp = CENTP[B];
      const vP = [
        aP[0] - cAp[0] + cBp[0],
        aP[1] - cAp[1] + cBp[1],
        aP[2] - cAp[2] + cBp[2],
      ];
      const Rv = norm(R[ai].map((x, k) => x - CENT[A][k] + CENT[B][k]));
      let best = -1,
        bestSim = -2;
      for (let i = 0; i < N; i++) {
        if (M.subject[i] !== B) continue;
        const s = dot(norm(R[i]), Rv);
        if (s > bestSim) {
          bestSim = s;
          best = i;
        }
      }
      const hit = best >= 0 && M.word[best] === t;
      traces.push(
        segment(aP, vP, {
          color: state.theme === "dark" ? "#9085e9" : "#4a3aa7",
          width: 5,
        }),
      );
      traces.push(
        markerAt(aP, {
          color: catColor("subject", A),
          size: state.dims === 3 ? 9 : 15,
          name: `emb(t, ${A})`,
          text: "t@" + A,
          hover: `emb("${t}", ${A})`,
        }),
      );
      traces.push(
        markerAt(vP, {
          color: state.theme === "dark" ? "#9085e9" : "#4a3aa7",
          symbol: "diamond",
          name: "v = t - A + B",
          text: "v",
          hover: "v = emb(t,A) - centroid(A) + centroid(B)",
        }),
      );
      if (bi >= 0) {
        const bP = [CUR.x[bi], CUR.y[bi], CUR.z[bi]];
        traces.push(
          markerAt(bP, {
            color: catColor("subject", B),
            size: state.dims === 3 ? 9 : 15,
            name: `true emb(t, ${B})`,
            text: "t@" + B,
            hover: `emb("${t}", ${B})`,
          }),
        );
      }
      if (best >= 0) {
        const nP = [CUR.x[best], CUR.y[best], CUR.z[best]];
        traces.push(
          markerAt(nP, {
            color: hit ? "#0ca30c" : "#e34948",
            symbol: "x",
            name: "nearest neighbour of v",
            text: M.word[best],
            hover: `NN of v: "${M.word[best]}" [${M.subject[best]}]`,
          }),
        );
      }
      story =
        `<b>"${t}"</b> re-aimed from <b>${A}</b> to <b>${B}</b>. The arrow adds the centroid offset; ` +
        `v's nearest neighbour under B is <b>"${best >= 0 ? M.word[best] : "-"}"</b>. ` +
        (hit
          ? `It lands on the same word - the offset cancelled identity.`
          : `It misses the target word - identity is not a clean translation.`);
      stat = [
        ["cos(v, NN)", bestSim.toFixed(3)],
        ["analogy", hit ? "HIT" : "miss"],
      ];
      stat._pill = hit ? "hit" : "miss";
    }
  } else if (state.view === 4) {
    const qi = neighbourQuery(),
      k = Math.max(3, Math.min(50, state.kN | 0 || 12));
    traces.push(
      mk(idxVisible(), {
        color: ink.faint,
        opacity: 0.2,
        showlegend: false,
        size: state.dims === 3 ? 2.2 : 4.5,
      }),
    );
    let nn = [];
    if (qi >= 0) {
      nn = neighbours(qi, k);
      const qw = M.word[qi],
        qc = M.category[qi];
      const same = [],
        cat = [],
        other = [];
      nn.forEach(([i]) => {
        if (M.word[i] === qw) same.push(i);
        else if (HAS_CAT && M.category[i] === qc) cat.push(i);
        else other.push(i);
      });
      if (other.length)
        traces.push(
          mk(other, {
            color: ink.muted,
            opacity: 0.9,
            name: "other",
            size: state.dims === 3 ? 4.5 : 9,
          }),
        );
      if (cat.length)
        traces.push(
          mk(cat, {
            color: PAL[state.theme][0],
            name: "same category",
            size: state.dims === 3 ? 5.5 : 11,
          }),
        );
      if (same.length)
        traces.push(
          mk(same, {
            color: PAL[state.theme][1],
            name: "same word",
            size: state.dims === 3 ? 6 : 12,
          }),
        );
      const qP = [CUR.x[qi], CUR.y[qi], CUR.z[qi]];
      traces.push(
        markerAt(qP, {
          color: PAL[state.theme][2],
          symbol: "star",
          size: state.dims === 3 ? 11 : 18,
          name: "query",
          text: qw,
          hover: `query: "${qw}" [${M.subject[qi]}]`,
        }),
      );
      story =
        `The <b>${k}</b> nearest thoughts to <b>"${qw}"</b>` +
        (state.subjN === "any" ? "" : ` (read by ${state.subjN})`) +
        `. Green = the same word elsewhere, blue = same category. Coherence far above chance would ` +
        `mean similar thoughts really do sit together.`;
      stat = [["neighbours", k]];
    } else {
      story = `Type a word read in the dataset to see its nearest neighbours.`;
      stat = [["status", "not found"]];
    }
    renderNeighTable(qi, nn);
  } else if (state.view === 5) {
    const field = state.colorBy;
    traces = colouredTraces(idxVisible(), field, { legend: true });
    const label = set === "et" ? "EEG + eye-tracking" : "EEG-only";
    story =
      `Whole space recomputed from <b>${label}</b> signals` +
      (P.has_eeg_only
        ? `. Toggle the set - gaze behaviour reshapes the reading-evoked geometry, ` +
          `but the imagined-thought (EEG-only) space must stand on neural signal alone.`
        : `. No EEG-only set was supplied, so the toggle is disabled.`);
    stat = [
      ["signal set", label],
      ["coloured by", field],
    ];
  } else if (state.view === 6) {
    const S = (P.sentences || [])[state.sent6];
    traces.push(
      mk(idxVisible(), {
        color: ink.faint,
        opacity: 0.13,
        showlegend: false,
        size: state.dims === 3 ? 2 : 4,
      }),
    );
    if (!S) {
      story = `No multi-word sentences are available in this run.`;
      stat = [["status", "—"]];
    } else {
      let drawn = 0,
        wc = 0;
      P.subjects.forEach((s) => {
        if (!state.visible.has(s)) return;
        const seq = (S.by_subj[s] || []).filter((i) => i < N);
        if (seq.length < 2) return;
        drawn++;
        wc = Math.max(wc, seq.length);
        traces.push(pathTrace(seq, { color: catColor("subject", s), name: s }));
      });
      story =
        `Sentence: <b>&ldquo;${esc(S.label)}&rdquo;</b> &mdash; each coloured path is one ` +
        `reader&rsquo;s route through the space, word by word. Overlapping paths mean readers place ` +
        `the sentence alike; divergent paths mean the code still depends on <b>who</b> read it.`;
      stat = [
        ["readers shown", drawn],
        ["read by", S.n_subj],
        ["words", wc],
      ];
    }
  } else if (state.view === 7) {
    const cat = state.cat7;
    traces.push(
      mk(idxVisible(), {
        color: ink.faint,
        opacity: 0.16,
        showlegend: false,
        size: state.dims === 3 ? 2.2 : 4.5,
      }),
    );
    const hits = idxVisible().filter((i) => M.category[i] === cat);
    traces = traces.concat(
      colouredTraces(hits, "subject", {
        legend: true,
        size: state.dims === 3 ? 5 : 11,
      }),
    );
    const cps = [];
    P.subjects.forEach((s) => {
      if (!state.visible.has(s)) return;
      const ids = hits.filter((i) => M.subject[i] === s);
      if (!ids.length) return;
      cps.push([
        avg(pick(CUR.x, ids)),
        avg(pick(CUR.y, ids)),
        state.dims === 3 ? avg(pick(CUR.z, ids)) : 0,
      ]);
    });
    if (cps.length > 1)
      traces.push(
        polyline(cps.concat([cps[0]]), {
          color: ink.muted,
          width: 2,
          dash: "dot",
        }),
      );
    const coh = categoryCohesion(cat);
    story =
      `Meaning <b>&ldquo;${esc(cat) || "(none)"}&rdquo;</b> read across people (${hits.length} words), ` +
      `coloured by reader. If the code were subject-invariant, the same meaning would cluster tightly ` +
      `<b>no matter who read it</b> &mdash; the project&rsquo;s north star.`;
    stat = [
      ["cos across people", fmt(coh.cos, 3)],
      ["random", fmt(RAND_BL, 3)],
      ["readers", coh.nsubj],
    ];
    stat._flag =
      coh.cos != null && RAND_BL != null && coh.cos > RAND_BL + 0.1
        ? "good"
        : "warn";
  } else if (state.view === 8) {
    const H = state.subjH;
    const others = idxVisible().filter((i) => M.subject[i] !== H);
    traces.push(
      mk(others, {
        color: ink.faint,
        opacity: 0.28,
        showlegend: false,
        size: state.dims === 3 ? 2.6 : 5.5,
      }),
    );
    const hIdx = [];
    for (let i = 0; i < N; i++) if (M.subject[i] === H) hIdx.push(i);
    const cal = calibrationTransform(H, state.kAnchor);
    const t = state.calT,
      hx = [],
      hy = [],
      hz = [];
    hIdx.forEach((i) => {
      const raw = [CUR.x[i], CUR.y[i], state.dims === 3 ? CUR.z[i] : 0];
      let pos = raw;
      if (cal) {
        const al = applyKabsch(cal, raw);
        pos = [
          raw[0] + t * (al[0] - raw[0]),
          raw[1] + t * (al[1] - raw[1]),
          raw[2] + t * (al[2] - raw[2]),
        ];
      }
      hx.push(pos[0]);
      hy.push(pos[1]);
      hz.push(pos[2]);
    });
    traces.push(
      rawScatter(hx, hy, hz, hIdx, {
        color: catColor("subject", H),
        name: H + " (new brain)",
        size: state.dims === 3 ? 4.6 : 10,
      }),
    );
    if (cal) {
      cal.anchors.forEach((w) => {
        const tc = centroidOf(cal.wO[w]);
        traces.push(
          markerAt([tc[0], tc[1], state.dims === 3 ? tc[2] : 0], {
            color: ink.muted,
            symbol: "diamond-open",
            size: state.dims === 3 ? 7 : 12,
            showlegend: false,
            hover: `anchor "${w}" — shared target`,
          }),
        );
      });
    }
    if (!cal) {
      story = `<b>${H}</b> shares too few words with the others to calibrate.`;
      stat = [["status", "—"]];
    } else {
      const drop = cal.gapBefore > 0 ? 1 - cal.gapAfter / cal.gapBefore : 0;
      story =
        `Treat <b>${H}</b> as a brand-new brain. From <b>${cal.k}</b> shared <b>anchor</b> words we fit ` +
        `an alignment (Procrustes) and slide <b>${H}</b>&rsquo;s whole space into the shared frame &mdash; ` +
        `no retraining. Gap to the others&rsquo; words falls <b>${(drop * 100).toFixed(0)}%</b> on held-out words. ` +
        `<span style="color:var(--muted)">Illustration in the projected space, not the trained model.</span>`;
      stat = [
        ["anchors", cal.k],
        ["gap before", fmt(cal.gapBefore, 3)],
        ["gap after", fmt(cal.gapAfter, 3)],
      ];
      stat._flag = cal.gapAfter < cal.gapBefore ? "good" : "warn";
    }
  }

  Plotly.react("plot", traces, baseLayout(), {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d"],
  });
  paintStory(story, stat);
  updateBottom();
}

function updateBottom() {
  const bottom = document.getElementById("bottom");
  const lp = document.getElementById("leaderpanel"),
    np = document.getElementById("neighpanel"),
    bw = document.getElementById("barwrap");
  lp.classList.add("hide");
  np.classList.add("hide");
  bw.classList.add("hide");
  let show = false;
  if (state.view === 3) {
    lp.classList.remove("hide");
    renderLeaderboard();
    show = true;
  } else if (state.view === 4) {
    np.classList.remove("hide");
    show = true;
  } else if (state.view === 5 && P.probe) {
    bw.classList.remove("hide");
    drawBar();
    show = true;
  }
  bottom.classList.toggle("hide", !show);
}

function paintStory(story, stat) {
  document.getElementById("story").innerHTML = story;
  const box = document.getElementById("statbox");
  box.innerHTML = "";
  (stat || []).forEach(([k, v]) => {
    const flag = stat._flag && k.indexOf("across") >= 0 ? " " + stat._flag : "";
    let vhtml = String(v);
    if (stat._pill && k === "analogy")
      vhtml = `<span class="pill ${stat._pill}">${v}</span>`;
    box.insertAdjacentHTML(
      "beforeend",
      `<div class="cell"><div class="k">${k}</div><div class="v${flag}">${vhtml}</div></div>`,
    );
  });
}

let barMade = false;
function drawBar() {
  if (!P.probe) return;
  const reps = Object.keys(P.probe);
  const vals = reps.map((r) => {
    const v = P.probe[r];
    return typeof v === "number"
      ? v
      : v.word_len != null
        ? v.word_len
        : Object.values(v)[0];
  });
  const ink = INK[state.theme];
  const colors = reps.map((r) =>
    /eeg-only/i.test(r) ? PAL[state.theme][2] : PAL[state.theme][0],
  );
  Plotly.react(
    "bar",
    [
      {
        type: "bar",
        x: vals,
        y: reps,
        orientation: "h",
        marker: { color: colors, line: { width: 0 } },
        text: vals.map((v) => Number(v).toFixed(3)),
        textposition: "auto",
        textfont: { color: "#fff", size: 12 },
        hoverinfo: "x+y",
      },
    ],
    {
      paper_bgcolor: ink.paper,
      plot_bgcolor: ink.paper,
      margin: { l: 130, r: 16, t: 4, b: 22 },
      font: { color: ink.text, size: 11 },
      uirevision: "bar",
      xaxis: {
        range: [0, Math.max(1, Math.max.apply(null, vals) * 1.15)],
        gridcolor: ink.grid,
        zeroline: false,
        color: ink.muted,
      },
      yaxis: { color: ink.text, automargin: true },
    },
    { displayModeBar: false, responsive: true },
  );
  barMade = true;
}

// ---- controls ------------------------------------------------------------
function buildControls() {
  const tabsMeta = [
    ["1", "◉", "By reader", "Spotlight one person"],
    ["2", "⁂", "One word", "The same word, many brains"],
    ["3", "±", "Arithmetic", "Translate a thought A->B"],
    ["4", "◈", "Neighbours", "The k closest thoughts"],
    ["5", "◐", "EEG vs gaze", "Does gaze drive the space?"],
    ["6", "↝", "Sentence", "A path per reader, word by word"],
    ["7", "◇", "Meaning", "Same meaning, across people"],
    ["8", "⌖", "Calibrate", "Snap a new brain into the frame"],
  ];
  const tabs = document.getElementById("tabs");
  tabsMeta.forEach(([n, ico, title, sub]) => {
    const b = document.createElement("button");
    b.className = "tab" + (n === "1" ? " on" : "");
    b.dataset.v = n;
    b.innerHTML = `<span class="ico">${ico}</span><span><b>${title}</b><span class="d">${sub}</span></span>`;
    b.onclick = () => setView(+n);
    tabs.appendChild(b);
  });

  const cb = document.getElementById("colorby");
  P.fields.concat(P.numeric_fields).forEach((f) => {
    const o = document.createElement("option");
    o.value = f;
    o.textContent = f + (P.numeric_fields.includes(f) ? " (scale)" : "");
    cb.appendChild(o);
  });
  cb.value = state.colorBy;
  cb.onchange = (e) => {
    state.colorBy = e.target.value;
    render();
  };

  document.querySelectorAll("#dims button").forEach((b) => {
    if (+b.dataset.d === state.dims) b.classList.add("on");
    b.onclick = () => {
      state.dims = +b.dataset.d;
      document
        .querySelectorAll("#dims button")
        .forEach((x) => x.classList.toggle("on", x === b));
      render();
    };
  });

  const sc = document.getElementById("subjchecks");
  P.subjects.forEach((s, i) => {
    const l = document.createElement("label");
    l.className = "chk";
    l.innerHTML = `<input type="checkbox" checked><span class="sw" style="background:${PAL[state.theme][i % 8]}"></span>${s}`;
    l.querySelector("input").onchange = (e) => {
      e.target.checked ? state.visible.add(s) : state.visible.delete(s);
      render();
    };
    sc.appendChild(l);
  });

  fillSel("subj1", P.subjects, state.subj1, (v) => {
    state.subj1 = v;
  });
  fillSel("metric1", P.numeric_fields, state.metric1, (v) => {
    state.metric1 = v;
  });
  fillSel("subjA", P.subjects, state.subjA, (v) => {
    state.subjA = v;
  });
  fillSel("subjB", P.subjects, state.subjB, (v) => {
    state.subjB = v;
  });
  fillSel("subjN", ["any"].concat(P.subjects), state.subjN, (v) => {
    state.subjN = v;
  });

  const dl = document.getElementById("wordlist");
  P.words.forEach((w) => {
    const o = document.createElement("option");
    o.value = w;
    dl.appendChild(o);
  });
  const w2 = document.getElementById("word2");
  w2.value = state.word2;
  w2.onchange = (e) => {
    if (e.target.value) state.word2 = e.target.value;
    render();
  };
  const wt = document.getElementById("wordT");
  wt.value = state.wordT;
  wt.onchange = (e) => {
    if (e.target.value) state.wordT = e.target.value;
    render();
  };
  const wn = document.getElementById("wordN");
  wn.value = state.wordN;
  wn.onchange = (e) => {
    if (e.target.value) state.wordN = e.target.value;
    render();
  };
  const kn = document.getElementById("kN");
  kn.value = state.kN;
  kn.onchange = (e) => {
    state.kN = +e.target.value || 12;
    render();
  };

  const sent6 = document.getElementById("sent6");
  if (sent6) {
    const ss = P.sentences || [];
    ss.forEach((s) => {
      const o = document.createElement("option");
      o.value = s.id;
      o.textContent = (s.n_subj > 1 ? "[" + s.n_subj + "×] " : "") + s.label;
      sent6.appendChild(o);
    });
    if (!ss.length) {
      const o = document.createElement("option");
      o.textContent = "(none available)";
      sent6.appendChild(o);
    }
    sent6.value = state.sent6;
    sent6.onchange = (e) => {
      state.sent6 = +e.target.value;
      render();
    };
  }
  const rand6 = document.getElementById("rand6");
  if (rand6)
    rand6.onclick = () => {
      const ss = P.sentences || [];
      if (!ss.length) return;
      state.sent6 = (Math.random() * ss.length) | 0;
      const el = document.getElementById("sent6");
      if (el) el.value = state.sent6;
      render();
    };
  const cat7 = document.getElementById("cat7");
  if (cat7) {
    const cs = DOMAINS["category"] || [];
    cs.forEach((c) => {
      const o = document.createElement("option");
      o.value = c;
      o.textContent = c;
      cat7.appendChild(o);
    });
    if (!cs.length) {
      const o = document.createElement("option");
      o.textContent = "(no category labels)";
      cat7.appendChild(o);
    }
    cat7.value = state.cat7;
    cat7.onchange = (e) => {
      state.cat7 = e.target.value;
      render();
    };
  }
  fillSel("subjH", P.subjects, state.subjH, (v) => {
    state.subjH = v;
    state.calT = 0;
    const b = document.getElementById("calbtn");
    if (b) b.textContent = "Calibrate ▶";
  });
  const ka = document.getElementById("kAnchor");
  if (ka) {
    ka.value = state.kAnchor;
    ka.onchange = (e) => {
      state.kAnchor = +e.target.value || 8;
      state.calT = 0;
      render();
    };
  }
  const calbtn = document.getElementById("calbtn");
  if (calbtn)
    calbtn.onclick = () => {
      const to = state.calT < 0.5 ? 1 : 0;
      calbtn.textContent = to > 0.5 ? "↺ Reset" : "Calibrate ▶";
      animate((v) => (state.calT = v), state.calT, to, 1100);
    };
  const debtn = document.getElementById("deidentbtn");
  if (debtn)
    debtn.onclick = () => {
      const to = state.deident < 0.5 ? 1 : 0;
      debtn.textContent =
        to > 0.5 ? "↺ Restore identity" : "▶ Remove reader identity";
      animate((v) => (state.deident = v), state.deident, to, 950);
    };

  document.querySelectorAll("#embset button").forEach((b) => {
    if (b.dataset.s === state.embSet) b.classList.add("on");
    b.disabled = !P.has_eeg_only;
    b.onclick = () => {
      if (!P.has_eeg_only) return;
      state.embSet = b.dataset.s;
      document
        .querySelectorAll("#embset button")
        .forEach((x) => x.classList.toggle("on", x === b));
      render();
    };
  });
  document.getElementById("v5note").innerHTML = P.has_eeg_only
    ? "Both spaces are trained independently and PCA-projected. The bar below shows a word-length linear probe for each."
    : "Supply <code>eeg_only_emb</code> (and <code>probe_scores</code>) to enable the toggle and the probe bar.";

  const randWord = () => {
    const list = P.words.length ? P.words : [""];
    return list[(Math.random() * list.length) | 0];
  };
  const r2 = document.getElementById("rand2");
  if (r2)
    r2.onclick = () => {
      const w = randWord();
      state.word2 = w;
      const el = document.getElementById("word2");
      if (el) el.value = w;
      render();
    };
  const r4 = document.getElementById("rand4");
  if (r4)
    r4.onclick = () => {
      const w = randWord();
      state.wordN = w;
      const el = document.getElementById("wordN");
      if (el) el.value = w;
      render();
    };

  document.getElementById("surprise").onclick = () => {
    if (!LB.length) return;
    const hits = LB.filter((x) => x.hit);
    const pool = hits.length ? hits : LB;
    openAnalogy(pool[(Math.random() * pool.length) | 0]);
  };
  document.getElementById("guidebtn").onclick = () => {
    state.guide = !state.guide;
    document.getElementById("guidebtn").textContent = state.guide
      ? "Hide guide"
      : "What am I looking at?";
    renderGuide();
    setTimeout(() => Plotly.Plots.resize("plot"), 0);
  };
  document.getElementById("themebtn").onclick = () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    applyTheme();
    refreshSwatches();
    renderGuide();
    renderBanners();
    render();
  };
}
function fillSel(id, opts, val, cb) {
  const s = document.getElementById(id);
  s.innerHTML = "";
  opts.forEach((o) => {
    const e = document.createElement("option");
    e.value = o;
    e.textContent = o;
    s.appendChild(e);
  });
  s.value = val;
  s.onchange = (e) => {
    cb(e.target.value);
    render();
  };
}
function refreshSwatches() {
  document.querySelectorAll("#subjchecks .chk").forEach((l, i) => {
    l.querySelector(".sw").style.background = PAL[state.theme][i % 8];
  });
}
function setView(v) {
  state.view = v;
  document
    .querySelectorAll(".tab")
    .forEach((t) => t.classList.toggle("on", +t.dataset.v === v));
  document.querySelectorAll(".v-ctrl").forEach((g) => g.classList.add("hide"));
  document.querySelector(".v-ctrl.v" + v).classList.remove("hide");
  render();
  setTimeout(() => Plotly.Plots.resize("plot"), 0);
}
function applyTheme() {
  document.documentElement.setAttribute("data-theme", state.theme);
}

applyTheme();
computeLeaderboard();
buildControls();
renderGuide();
renderBanners();
render();
window.addEventListener("resize", () => {
  Plotly.Plots.resize("plot");
  if (barMade) Plotly.Plots.resize("bar");
});
