(function () {
  var CFG = JSON.parse(document.getElementById("cfg").textContent);
  var DIMS = CFG.dims,
    PAL = CFG.palette,
    COLORBY = CFG.colorby,
    TOK = CFG.tokens,
    COLS = CFG.columns;
  var theme = "light";
  var col = COLS.length ? COLS[0] : null;

  function colorsFor(c, th) {
    var cb = COLORBY[c];
    if (!cb) return null;
    var p = PAL[th];
    return cb.codes.map(function (k) {
      return p[k % p.length];
    });
  }
  function plotLayout(th) {
    var t = TOK[th];
    var L = {
      paper_bgcolor: t.panel,
      plot_bgcolor: t.panel,
      "font.color": t.ink,
    };
    if (DIMS === 3) {
      ["xaxis", "yaxis", "zaxis"].forEach(function (a) {
        L["scene." + a + ".gridcolor"] = t.grid;
        L["scene." + a + ".zerolinecolor"] = t.border;
        L["scene." + a + ".backgroundcolor"] = t.plane;
        L["scene." + a + ".showbackground"] = true;
        L["scene." + a + ".color"] = t.ink2;
      });
    } else {
      ["xaxis", "yaxis"].forEach(function (a) {
        L[a + ".gridcolor"] = t.grid;
        L[a + ".zerolinecolor"] = t.border;
        L[a + ".linecolor"] = t.border;
        L[a + ".color"] = t.ink2;
      });
    }
    return L;
  }
  function renderLegend() {
    var el = document.getElementById("legend");
    if (!col || !COLORBY[col]) {
      el.innerHTML = "";
      return;
    }
    var cb = COLORBY[col],
      p = PAL[theme];
    el.innerHTML = cb.cats
      .map(function (c, i) {
        return (
          '<span class="lg"><span class="sw" style="background:' +
          p[i % p.length] +
          '"></span>' +
          String(c).replace(/&/g, "&amp;").replace(/</g, "&lt;") +
          "</span>"
        );
      })
      .join("");
  }
  function applyTheme() {
    document.documentElement.setAttribute("data-theme", theme);
    var btns = document.querySelectorAll("#themeseg button");
    for (var i = 0; i < btns.length; i++) {
      btns[i].classList.toggle("on", btns[i].dataset.t === theme);
    }
    if (window.Plotly) {
      Plotly.relayout("plot", plotLayout(theme));
      var cs = colorsFor(col, theme);
      if (cs) Plotly.restyle("plot", { "marker.color": [cs] });
    }
    renderLegend();
  }
  function setCol(c) {
    col = c;
    if (window.Plotly) {
      var cs = colorsFor(col, theme);
      if (cs) Plotly.restyle("plot", { "marker.color": [cs] });
    }
    renderLegend();
  }

  var sel = document.getElementById("colorby");
  if (COLS.length) {
    COLS.forEach(function (c) {
      var o = document.createElement("option");
      o.value = c;
      o.textContent = c;
      sel.appendChild(o);
    });
    sel.value = col;
    sel.addEventListener("change", function (e) {
      setCol(e.target.value);
    });
  } else {
    document.getElementById("colorbyctl").style.display = "none";
  }

  document.getElementById("themeseg").addEventListener("click", function (e) {
    var b = e.target.closest("button");
    if (!b) return;
    theme = b.dataset.t;
    applyTheme();
  });

  applyTheme();
})();
