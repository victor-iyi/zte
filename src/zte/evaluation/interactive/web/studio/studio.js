/* The decode studio: a reading, its scalp field, and the frozen LM's own step-by-step behaviour. */
(() => {
  'use strict';

  const DATA = JSON.parse(document.getElementById('data').textContent);
  const $ = (id) => document.getElementById(id);
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  /* ---------- theme ---------- */
  const root = document.documentElement;
  const stored = (() => {
    try {
      return localStorage.getItem('zte-studio-theme');
    } catch {
      return null;
    }
  })();
  if (stored) root.setAttribute('data-theme', stored);
  $('themebtn').addEventListener('click', () => {
    const dark = getComputedStyle(document.body).backgroundColor.match(/\d+/g)[0] < 128;
    const next = dark ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try {
      localStorage.setItem('zte-studio-theme', next);
    } catch {
      /* private mode: the toggle still works for this session */
    }
    draw();
  });

  if (!DATA.applicable || !(DATA.readings || []).length) {
    $('main').hidden = true;
    document.querySelector('.transport').hidden = true;
    $('empty').hidden = false;
    $('emptyreason').textContent = DATA.reason || 'This run produced no decodes to inspect.';
    $('run').textContent = DATA.run_name || 'ZTE';
    return;
  }

  /* ---------- colour ---------- */
  const STOPS = [
    [16, 12, 46],
    [64, 44, 160],
    [122, 82, 255],
    [255, 63, 136],
    [255, 122, 54],
    [255, 214, 102],
  ];
  function ramp(t) {
    const x = clamp(t, 0, 1) * (STOPS.length - 1);
    const i = Math.min(Math.floor(x), STOPS.length - 2);
    const f = x - i;
    const a = STOPS[i];
    const b = STOPS[i + 1];
    return [
      Math.round(a[0] + (b[0] - a[0]) * f),
      Math.round(a[1] + (b[1] - a[1]) * f),
      Math.round(a[2] + (b[2] - a[2]) * f),
    ];
  }
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const ink = () => getComputedStyle(document.body).getPropertyValue('color') || '#111';
  const muted = () => getComputedStyle(root).getPropertyValue('--muted').trim() || '#888';
  const accent = () => getComputedStyle(root).getPropertyValue('--accent').trim() || '#5a4bff';
  const hot = () => getComputedStyle(root).getPropertyValue('--hot').trim() || '#ff6b35';

  /* ---------- state ---------- */
  const state = { reading: 0, step: 0, band: 0, view: '2d', playing: false, yaw: 0.5, pitch: 0.35, timer: null };
  const current = () => DATA.readings[state.reading];
  const nSteps = () => (current().steps || []).length;

  /* ---------- header ---------- */
  $('run').textContent = `${DATA.run_name} — decode studio`;
  $('sub').textContent =
    `${DATA.readings.length} reading${DATA.readings.length === 1 ? '' : 's'} · ` +
    `${DATA.uses_evidence ? 'word-synchronous evidence on' : 'pooled prefix only'}` +
    (DATA.montage ? ` · ${DATA.montage.labels.length} electrodes${DATA.montage.approximate ? ' (approximate)' : ''}` : '');
  $('caveat').innerHTML =
    '<b>This is an inspection tool, not a result.</b> The decode below is free-running — no reference, no reference ' +
    'length, no candidate set — and it comes from the same call the evaluation makes. But a handful of readings ' +
    'chosen to look at is not an audit: the pre-registered controls, the permutation null and the verdict live in ' +
    '<code>zte-decode</code> and the evaluation report. Absolute scores here mean nothing on their own; read them ' +
    'against the controls at the bottom.';

  const picker = $('reading');
  DATA.readings.forEach((r, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    const words = (r.target || '').split(/\s+/).slice(0, 9).join(' ');
    opt.textContent = `${r.subject} · ${r.task} · ${r.n_words}w — ${words}${(r.target || '').split(/\s+/).length > 9 ? '…' : ''}`;
    picker.appendChild(opt);
  });
  picker.addEventListener('change', () => {
    state.reading = Number(picker.value);
    state.step = 0;
    buildBands();
    buildField();
    syncScrub();
    draw();
  });

  /* ---------- bands ---------- */
  function buildBands() {
    const host = $('bands');
    host.innerHTML = '';
    const bands = DATA.bands || [];
    const has = (current().power || []).length > 0;
    if (!has || !bands.length) {
      host.innerHTML = '<span class="scale">no band power for this reading</span>';
      return;
    }
    bands.forEach((name, i) => {
      const b = document.createElement('button');
      b.textContent = name;
      b.className = i === state.band ? 'on' : '';
      b.addEventListener('click', () => {
        state.band = i;
        buildBands();
        draw();
      });
      host.appendChild(b);
    });
  }

  /* ---------- scalp field (inverse-distance interpolation, precomputed once) ---------- */
  const FIELD = { size: 190, idx: null, wgt: null, inside: null };
  function buildField() {
    if (!DATA.montage) return;
    const xy = DATA.montage.xy;
    const n = FIELD.size;
    const K = Math.min(6, xy.length);
    const idx = new Int16Array(n * n * K);
    const wgt = new Float32Array(n * n * K);
    const inside = new Uint8Array(n * n);
    for (let py = 0; py < n; py++) {
      for (let px = 0; px < n; px++) {
        const cell = py * n + px;
        // The montage arrives normalised to a unit disc, so the grid spans [-1.06, 1.06] and clips at 1.
        const gx = (px / (n - 1) - 0.5) * 2.12;
        const gy = (py / (n - 1) - 0.5) * 2.12;
        if (Math.hypot(gx, gy) > 1) continue;
        inside[cell] = 1;
        const d = xy.map((p, i) => [Math.hypot(p[0] - gx, p[1] - gy), i]);
        d.sort((a, b) => a[0] - b[0]);
        let total = 0;
        for (let k = 0; k < K; k++) {
          const w = 1 / Math.max(d[k][0], 1e-3) ** 2;
          idx[cell * K + k] = d[k][1];
          wgt[cell * K + k] = w;
          total += w;
        }
        for (let k = 0; k < K; k++) wgt[cell * K + k] /= total;
      }
    }
    FIELD.idx = idx;
    FIELD.wgt = wgt;
    FIELD.inside = inside;
    FIELD.k = K;
  }

  /* ---------- which word is the pointer on ---------- */
  function activeWord() {
    const r = current();
    const ptr = r.pointer;
    if (!ptr || !ptr.length) return null;
    const row = ptr[Math.min(state.step, ptr.length - 1)] || [];
    let best = 0;
    for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
    return row[best] > 0 ? best : null;
  }

  function channelValues() {
    const r = current();
    const power = r.power || [];
    if (!power.length) return null;
    const word = activeWord();
    const band = clamp(state.band, 0, (power[0] || []).length - 1);
    if (word !== null && word < power.length) return power[word][band];

    // No pointer: the reading has no word-synchronous path, so the map shows its mean rather than a fake cursor.
    const out = new Float64Array((power[0][band] || []).length);
    power.forEach((w) => (w[band] || []).forEach((v, c) => (out[c] += v)));
    return Array.from(out, (v) => v / power.length);
  }

  /* ---------- canvases ---------- */
  function fit(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth || canvas.width;
    const h = Number(canvas.getAttribute('height'));
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(canvas.width / w, 0, 0, canvas.width / w, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function drawBrain() {
    const canvas = $('brain');
    const { ctx, w, h } = fit(canvas);
    const values = channelValues();
    if (!DATA.montage || !values) {
      ctx.fillStyle = muted();
      ctx.font = '13px ui-sans-serif, system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('No electrode geometry or band power for this reading.', w / 2, h / 2);
      return;
    }
    const max = Math.max(1, ...values);
    state.view === '3d' ? drawHead3d(ctx, w, h, values, max) : drawCap2d(ctx, w, h, values, max);
  }

  function drawCap2d(ctx, w, h, values, max) {
    const cx = w / 2;
    const cy = h / 2 + 6;
    const R = Math.min(w, h) * 0.42;
    const n = FIELD.size;
    const K = FIELD.k;

    const img = ctx.createImageData(n, n);
    for (let cell = 0; cell < n * n; cell++) {
      if (!FIELD.inside[cell]) continue;
      let v = 0;
      for (let k = 0; k < K; k++) v += FIELD.wgt[cell * K + k] * values[FIELD.idx[cell * K + k]];
      const c = ramp(v / max);
      const o = cell * 4;
      img.data[o] = c[0];
      img.data[o + 1] = c[1];
      img.data[o + 2] = c[2];
      img.data[o + 3] = 255;
    }
    const buffer = document.createElement('canvas');
    buffer.width = n;
    buffer.height = n;
    buffer.getContext('2d').putImageData(img, 0, 0);

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.clip();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(buffer, cx - R * 1.06, cy - R * 1.06, R * 2.12, R * 2.12);
    ctx.restore();

    // Head outline, nose and ears, so the map reads as a scalp rather than a disc.
    ctx.strokeStyle = muted();
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - R * 0.13, cy - R * 0.99);
    ctx.lineTo(cx, cy - R * 1.16);
    ctx.lineTo(cx + R * 0.13, cy - R * 0.99);
    ctx.stroke();
    [-1, 1].forEach((s) => {
      ctx.beginPath();
      ctx.ellipse(cx + s * R, cy, R * 0.06, R * 0.15, 0, 0, Math.PI * 2);
      ctx.stroke();
    });

    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    DATA.montage.xy.forEach((p) => {
      ctx.beginPath();
      ctx.arc(cx + p[0] * R, cy + p[1] * R, 1.5, 0, Math.PI * 2);
      ctx.fill();
    });

    const word = activeWord();
    ctx.fillStyle = muted();
    ctx.font = '12px ui-sans-serif, system-ui';
    ctx.textAlign = 'center';
    // The pointer indexes EEG word slots and `target_words` is a text tokenisation; they usually agree and are not
    // guaranteed to, so the text is only named when the two counts line up.
    const named = (current().target_words || []).length === (current().power || []).length;
    const label =
      word === null ? 'reading mean (no pointer)' : `word ${word + 1}${named ? `: ${wordAt(word)}` : ''}`;
    ctx.fillText(label, cx, cy + R + 26);
  }

  function drawHead3d(ctx, w, h, values, max) {
    const cx = w / 2;
    const cy = h / 2 + 6;
    const R = Math.min(w, h) * 0.40;
    const cosY = Math.cos(state.yaw);
    const sinY = Math.sin(state.yaw);
    const cosP = Math.cos(state.pitch);
    const sinP = Math.sin(state.pitch);

    const pts = DATA.montage.xyz.map((p, i) => {
      const x1 = p[0] * cosY - p[1] * sinY;
      const y1 = p[0] * sinY + p[1] * cosY;
      const y2 = y1 * cosP - p[2] * sinP;
      const z2 = y1 * sinP + p[2] * cosP;
      return { x: x1, y: -z2, depth: y2, value: values[i], label: DATA.montage.labels[i] };
    });
    pts.sort((a, b) => a.depth - b.depth);

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, R * 1.02, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(120,120,160,0.07)';
    ctx.fill();
    ctx.restore();

    pts.forEach((p) => {
      const scale = 1 / (1.9 - p.depth * 0.55);
      const px = cx + p.x * R * 1.55 * scale;
      const py = cy + p.y * R * 1.55 * scale;
      const t = p.value / max;
      const c = ramp(t);
      const radius = (3.4 + 5.2 * t) * scale;
      const glow = ctx.createRadialGradient(px, py, 0, px, py, radius * 3);
      glow.addColorStop(0, rgba(c, 0.55 * (0.45 + 0.55 * t)));
      glow.addColorStop(1, rgba(c, 0));
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(px, py, radius * 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = rgba(c, 0.35 + 0.65 * ((p.depth + 1) / 2));
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = muted();
    ctx.font = '12px ui-sans-serif, system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('drag to rotate · nose is +y', cx, h - 8);
  }

  let dragging = null;
  $('brain').addEventListener('pointerdown', (e) => {
    dragging = { x: e.clientX, y: e.clientY };
    $('brain').setPointerCapture(e.pointerId);
  });
  $('brain').addEventListener('pointermove', (e) => {
    if (!dragging || state.view !== '3d') return;
    state.yaw += (e.clientX - dragging.x) * 0.01;
    state.pitch = clamp(state.pitch + (e.clientY - dragging.y) * 0.01, -1.2, 1.2);
    dragging = { x: e.clientX, y: e.clientY };
    drawBrain();
  });
  ['pointerup', 'pointercancel', 'pointerleave'].forEach((ev) =>
    $('brain').addEventListener(ev, () => {
      dragging = null;
    }),
  );
  $('viewtoggle').addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    state.view = btn.dataset.view;
    [...$('viewtoggle').children].forEach((b) => b.classList.toggle('on', b === btn));
    drawBrain();
  });

  /* ---------- text ---------- */
  const wordAt = (i) => (current().target_words || [])[i] || '';

  function drawText() {
    const r = current();
    const ptr = r.pointer;
    const row = ptr && ptr.length ? ptr[Math.min(state.step, ptr.length - 1)] : null;
    const peak = row ? Math.max(...row, 1e-9) : 1;

    const target = $('target');
    target.innerHTML = '';
    (r.target_words || []).forEach((word, i) => {
      const span = document.createElement('span');
      span.className = 'w';
      span.textContent = word + ' ';
      const weight = row && i < row.length ? row[i] / peak : 0;
      if (weight > 0.02) {
        const c = ramp(0.35 + 0.6 * weight);
        span.style.backgroundColor = rgba(c, 0.16 + 0.7 * weight);
        if (weight > 0.55) span.style.color = '#fff';
      }
      target.appendChild(span);
    });

    const steps = r.steps || [];
    const hypo = $('hypo');
    hypo.innerHTML = '';
    if (!steps.length) {
      hypo.innerHTML = '<span class="pending">no steps were traced for this reading</span>';
    }
    steps.forEach((s, i) => {
      const span = document.createElement('span');
      span.className = 'tk' + (i === state.step ? ' now' : '') + (i > state.step ? ' pending' : '');
      span.textContent = s.piece;
      if (i <= state.step) {
        const c = ramp(0.25 + 0.7 * (s.probability || 0));
        span.style.backgroundColor = rgba(c, 0.1 + 0.35 * (s.probability || 0));
      }
      span.title = `step ${s.step} · p=${(s.probability || 0).toFixed(3)}`;
      span.addEventListener('click', () => {
        state.step = i;
        syncScrub();
        draw();
      });
      hypo.appendChild(span);
    });

    const s = steps[Math.min(state.step, steps.length - 1)];
    $('tokenmeta').textContent = s
      ? `emitted “${s.piece}” · p ${(s.probability || 0).toFixed(3)} · entropy ${(s.entropy || 0).toFixed(2)} nats`
      : '';

    const alts = $('alts');
    alts.innerHTML = '';
    ((s && s.alternatives) || []).forEach((a, i) => {
      const rowEl = document.createElement('div');
      rowEl.className = 'alt' + (i === 0 ? ' pick' : '');
      rowEl.innerHTML =
        `<code>${escapeHtml(a.piece === ' ' ? '␣' : a.piece)}</code>` +
        `<div class="bar"><i style="width:${(a.probability * 100).toFixed(1)}%"></i></div>` +
        `<span>${a.probability.toFixed(3)}</span>`;
      alts.appendChild(rowEl);
    });

    const word = activeWord();
    $('wordpill').textContent = word === null ? `${r.n_words} words` : `word ${word + 1} / ${r.n_words}`;
    $('steppill').textContent = `step ${state.step + 1} / ${Math.max(steps.length, 1)}`;
  }

  /* ---------- firing ---------- */
  function drawGauges() {
    const steps = current().steps || [];
    const s = steps[Math.min(state.step, steps.length - 1)] || {};
    const cells = [
      ['probability', (s.probability || 0).toFixed(3), 'token probability'],
      ['entropy', (s.entropy || 0).toFixed(2), 'entropy (nats)'],
      ['kl', (s.evidence_kl || 0).toFixed(4), 'evidence KL'],
      ['norm', (s.evidence_norm || 0).toFixed(3), 'nudge ‖Δh‖'],
    ];
    $('gauges').innerHTML = cells
      .map(([, value, label]) => `<div class="gauge"><b>${value}</b><span>${label}</span></div>`)
      .join('');
  }

  function drawTrace() {
    const canvas = $('trace');
    const { ctx, w, h } = fit(canvas);
    const steps = current().steps || [];
    if (steps.length < 2) return;

    const pad = { l: 34, r: 10, t: 16, b: 20 };
    const iw = w - pad.l - pad.r;
    const ih = h - pad.t - pad.b;
    const series = [
      { key: 'evidence_kl', colour: hot(), label: 'evidence KL — how hard the brain pushed on this token' },
      { key: 'entropy', colour: accent(), label: 'entropy' },
    ];

    ctx.strokeStyle = muted();
    ctx.globalAlpha = 0.2;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t + ih);
    ctx.lineTo(pad.l + iw, pad.t + ih);
    ctx.stroke();
    ctx.globalAlpha = 1;

    series.forEach((s, si) => {
      const values = steps.map((d) => d[s.key] || 0);
      const max = Math.max(...values, 1e-6);
      ctx.strokeStyle = s.colour;
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = pad.l + (i / (values.length - 1)) * iw;
        const y = pad.t + ih - (v / max) * ih;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = s.colour;
      ctx.font = '11px ui-sans-serif, system-ui';
      ctx.textAlign = 'left';
      ctx.fillText(s.label, pad.l + 2, pad.t - 4 + si * 0);
      if (si === 0) ctx.fillText(`max ${max.toFixed(3)}`, pad.l + iw - 66, pad.t + 9);
    });

    const x = pad.l + (state.step / Math.max(steps.length - 1, 1)) * iw;
    ctx.strokeStyle = ink();
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + ih);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawPointer() {
    const canvas = $('pointer');
    const { ctx, w, h } = fit(canvas);
    const ptr = current().pointer;
    if (!ptr || !ptr.length) {
      ctx.fillStyle = muted();
      ctx.font = '12px ui-sans-serif, system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('This run has no word-synchronous evidence path.', w / 2, h / 2);
      return;
    }
    const pad = { l: 34, r: 10, t: 18, b: 18 };
    const rows = ptr.length;
    const cols = ptr[0].length;
    const cw = (w - pad.l - pad.r) / cols;
    const ch = (h - pad.t - pad.b) / rows;
    let peak = 1e-9;
    ptr.forEach((r) => r.forEach((v) => (peak = Math.max(peak, v))));

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = ptr[i][j] / peak;
        if (v < 0.01) continue;
        ctx.fillStyle = rgba(ramp(0.2 + 0.8 * v), 0.15 + 0.85 * v);
        ctx.fillRect(pad.l + j * cw, pad.t + i * ch, Math.max(cw, 1), Math.max(ch, 1));
      }
    }
    ctx.strokeStyle = ink();
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1.5;
    const y = pad.t + clamp(state.step, 0, rows - 1) * ch + ch / 2;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = muted();
    ctx.font = '11px ui-sans-serif, system-ui';
    ctx.textAlign = 'left';
    ctx.fillText('pointer walk — decoding step (down) against word (across)', pad.l, 12);
  }

  function drawCodes() {
    const codes = current().codes;
    $('codes').innerHTML = !codes
      ? '<span class="code">no rate ladder in this run</span>'
      : codes.map((c, i) => `<span class="code">stage ${i} <b>#${c}</b></span>`).join('');
  }

  /* ---------- controls ---------- */
  function drawControls() {
    const r = current();
    const rows = [
      ['<b>target</b>', r.target, '—', '—'],
      ['<b>decoded (EEG)</b>', r.hypothesis, r.scores.content_f1.toFixed(4), r.scores.wer.toFixed(3)],
      ...(r.controls || []).map((c) => [c.name, c.text, c.content_f1.toFixed(4), c.wer.toFixed(3)]),
    ];
    $('controls').innerHTML =
      '<thead><tr><th>condition</th><th>text</th><th>content F1</th><th>WER</th></tr></thead><tbody>' +
      rows
        .map(
          (row, i) =>
            `<tr class="${i === 1 ? 'head' : ''}"><td>${row[0]}</td><td>${escapeHtml(row[1] || '')}</td>` +
            `<td class="num">${row[2]}</td><td class="num">${row[3]}</td></tr>`,
        )
        .join('') +
      '</tbody>';
    $('controlnote').textContent =
      'One reading is one observation. A hypothesis above its controls here is an anecdote; the claim needs the ' +
      'paired delta over every held-out reading, its bootstrap interval and the permutation null.';
  }

  const escapeHtml = (s) =>
    String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

  /* ---------- transport ---------- */
  function syncScrub() {
    const max = Math.max(nSteps() - 1, 0);
    const scrub = $('scrub');
    scrub.max = String(max);
    scrub.value = String(clamp(state.step, 0, max));
    $('clock').textContent = `${Math.min(state.step + 1, nSteps())} / ${nSteps()}`;
  }
  function seek(step) {
    state.step = clamp(step, 0, Math.max(nSteps() - 1, 0));
    syncScrub();
    draw();
  }
  function stop() {
    state.playing = false;
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
    $('play').textContent = '▶';
  }
  $('play').addEventListener('click', () => {
    if (state.playing) return stop();
    if (state.step >= nSteps() - 1) state.step = 0;
    state.playing = true;
    $('play').textContent = '⏸';
    state.timer = setInterval(() => {
      if (state.step >= nSteps() - 1) return stop();
      seek(state.step + 1);
    }, Number($('speed').value));
  });
  $('speed').addEventListener('change', () => {
    if (state.playing) {
      stop();
      $('play').click();
    }
  });
  $('rewind').addEventListener('click', () => seek(0));
  $('prev').addEventListener('click', () => seek(state.step - 1));
  $('next').addEventListener('click', () => seek(state.step + 1));
  $('scrub').addEventListener('input', (e) => {
    stop();
    seek(Number(e.target.value));
  });
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if (e.key === ' ') {
      e.preventDefault();
      $('play').click();
    } else if (e.key === 'ArrowRight') seek(state.step + 1);
    else if (e.key === 'ArrowLeft') seek(state.step - 1);
  });

  /* ---------- go ---------- */
  function draw() {
    drawBrain();
    drawText();
    drawGauges();
    drawTrace();
    drawPointer();
    drawCodes();
    drawControls();
  }
  window.addEventListener('resize', draw);
  $('scalenote').textContent = 'relative within this reading';
  buildBands();
  buildField();
  syncScrub();
  draw();
})();
