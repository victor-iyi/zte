"""The honest scoreboard: the held-out subject's numbers, each stated as a lift over the raw band-power control."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zte.evaluation import metrics as M
from zte.evaluation.neurons import neuron_report

if TYPE_CHECKING:
    import pandas as pd

# Probe targets a meaning code should carry, and those a thought code should not; mirrors zte.evaluation.neurons.
_CONTENT_TARGETS: tuple[str, ...] = ('word_len', 'log_freq', 'category')
_IDENTITY_TARGETS: tuple[str, ...] = ('subject',)
# Positive-control floor: below this linear R2 on raw band power the content probe is not trustworthy.
_CONTENT_PROBE_FLOOR: float = 0.02


def holdout_subject(config: Any | None) -> str | None:
    """Returns the LOSO held-out subject, or `None` when the run is not LOSO."""
    train = getattr(config, 'train', None)
    if train is None or getattr(train, 'split', None) != 'by_subject_loso':
        return None
    return getattr(train, 'loso_holdout_subject', None)


def lift_over_raw(comparison: list[dict[str, Any]]) -> dict[str, Any]:
    """Turns the probe-comparison rows into per-target `ZTE - raw` lifts.

    Args:
        comparison (list[dict]): Rows from `metrics.representation_comparison`, each with
            `target`, `representation`, `metric`, `linear_score`, `knn_score`.

    Returns:
        dict: `{target: {metric, zte, raw, noise, lift_linear, lift_knn}}` plus a
        `content_probe` positive-control block.
    """
    by_target: dict[str, dict[str, Any]] = {}
    for row in comparison:
        t = row['target']
        by_target.setdefault(t, {'metric': row['metric']})
        by_target[t][row['representation']] = {
            'linear': row['linear_score'],
            'knn': row['knn_score'],
        }

    lifts: dict[str, Any] = {}
    for t, reps in by_target.items():
        zte = reps.get('ZTE', {})
        raw = reps.get('raw band-power', {})
        noise = reps.get('noise (matched)', {})
        lifts[t] = {
            'metric': reps['metric'],
            'zte_linear': zte.get('linear'),
            'raw_linear': raw.get('linear'),
            'noise_linear': noise.get('linear'),
            'lift_linear': _sub(zte.get('linear'), raw.get('linear')),
            'lift_knn': _sub(zte.get('knn'), raw.get('knn')),
            'is_content': t in _CONTENT_TARGETS,
            'is_identity': t in _IDENTITY_TARGETS,
        }

    # Positive control: raw features must expose lexical content above the floor. This is filled in by
    # `build_scoreboard` from the GENUINELY raw band power -- see `raw_content_positive_control`. The
    # value derived here from the probe-comparison's "raw band-power" row is a fallback only, and is
    # unreliable under whitening normalisers (riemannian/zscore_subject) that strip amplitude, which is
    # exactly the signal word_len/log_freq ride on. Do not gate "content 0%" on this fallback.
    raw_content = [
        by_target[t]['raw band-power']['linear']
        for t in ('word_len', 'log_freq')
        if t in by_target and 'raw band-power' in by_target[t]
    ]
    best = max(raw_content) if raw_content else float('nan')
    lifts['content_probe'] = {
        'raw_content_r2_best': best,
        'floor': _CONTENT_PROBE_FLOOR,
        'passes': bool(np.isfinite(best) and best >= _CONTENT_PROBE_FLOOR),
        'source': 'normalised-features (fallback)',
    }
    return lifts


def raw_content_positive_control(
    word_band_power: np.ndarray | None, word_meta: 'pd.DataFrame'
) -> dict[str, Any] | None:
    """Probes genuinely-raw band power for lexical content -- the honest positive control (docs/EVALUATION.md).

    Args:
        word_band_power (np.ndarray | None): Raw band power `(n, bands, channels)`, or `None` for a
            raw-signal frontend (the control is then not applicable).
        word_meta (pd.DataFrame): Per-word metadata carrying `word_len` and `log_freq`.

    Returns:
        dict | None: The positive-control block (best R2, per-target R2, floor, verdict), or `None`.
    """
    if word_band_power is None:
        return None
    flat = np.asarray(word_band_power, dtype=np.float32).reshape(len(word_band_power), -1)
    # Omitted words carry NaN band power; impute to the column mean so the probe sees every row.
    col_mean = np.nanmean(np.where(np.isfinite(flat), flat, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    flat = np.where(np.isfinite(flat), flat, col_mean)

    scores: dict[str, float] = {}
    for target in ('word_len', 'log_freq'):
        if target not in word_meta.columns:
            continue
        y = np.asarray(word_meta[target].to_numpy(), dtype=np.float64)
        keep = np.isfinite(y)
        if keep.sum() < 32:
            continue
        result = M.linear_probe(flat[keep], y[keep], task='regression')
        scores[target] = round(float(result.get('score', float('nan'))), 4)

    if not scores:
        return None
    best = max(scores.values())
    return {
        'raw_content_r2_best': best,
        'per_target_r2': scores,
        'floor': _CONTENT_PROBE_FLOOR,
        'passes': bool(np.isfinite(best) and best >= _CONTENT_PROBE_FLOOR),
        'source': 'raw band-power',
    }


def _sub(a: float | None, b: float | None) -> float | None:
    """`a - b`, tolerating missing operands."""
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 4)


def held_out_geometry(
    word_emb: np.ndarray, word_meta: 'pd.DataFrame', holdout: str
) -> dict[str, Any] | None:
    """Geometry + variance budget over the held-out subject's rows only, undiluted by the training subjects."""
    if 'subject' not in word_meta.columns:
        return None
    mask = (word_meta['subject'] == holdout).to_numpy()
    if mask.sum() < 8:  # too few to characterise geometry
        return None
    sub_emb = np.asarray(word_emb, dtype=np.float64)[mask]
    sub_meta = word_meta.loc[mask].reset_index(drop=True)
    d = sub_emb.shape[1]
    budget = neuron_report(sub_emb, sub_meta).get('summary', {})
    return {
        'subject': holdout,
        'n_words': int(mask.sum()),
        'effective_rank': float(M.effective_rank(sub_emb)),
        'effective_rank_ratio': float(M.effective_rank(sub_emb) / d),
        'anisotropy': float(M.anisotropy(sub_emb)),
        'subject_variance': None,  # single subject -> identity budget is not meaningful here
        'content_variance': budget.get('what_variance'),
        'task_variance': budget.get('variance_budget', {}).get('task'),
    }


def cross_subject_holdout_retrieval(
    sent_emb: np.ndarray,
    content_ids: np.ndarray,
    subjects: np.ndarray,
    holdout: str,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any] | None:
    """Held-out queries vs a cross-subject gallery: find a stranger's thought in others.

    Args:
        sent_emb (np.ndarray): Sentence embeddings `(n, d)`.
        content_ids (np.ndarray): Stimulus id per sentence `(n,)`.
        subjects (np.ndarray): Subject code per sentence `(n,)`.
        holdout (str): The held-out subject code.
        ks (tuple[int, ...]): Top-K cut-offs.

    Returns:
        dict | None: `top{k}`, `mrr`, `chance_top1`, `n_queries`, `lift_top1` (top1 - chance),
            or `None` when the run has too few subjects/queries to be meaningful.
    """
    subjects = np.asarray(subjects)
    content_ids = np.asarray(content_ids)
    if len(np.unique(subjects)) < 2:
        return None
    q_mask = subjects == holdout
    if q_mask.sum() < 2:
        return None

    # Cosine similarity over the whole sentence set; the gallery is masked per query.
    emb = np.asarray(sent_emb, dtype=np.float64)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = emb @ emb.T  # cosine, (n, n)

    q_idx = np.where(q_mask)[0]
    out = {f'top{k}': 0.0 for k in ks}
    rr = 0.0
    chances = []
    percentiles: list[float] = []  # rank-percentile per query (1.0 = correct match ranked first)
    n_scored = 0

    for i in q_idx:
        cross = subjects != subjects[i]  # gallery: other people only
        if not cross.any():
            continue

        cand = np.where(cross)[0]
        order = cand[np.argsort(-sims[i, cand])]
        same = content_ids[order] == content_ids[i]
        if not same.any():
            # Still counts as a query (a miss); chance uses this query's gallery.
            chances.append(float((content_ids[cand] == content_ids[i]).mean()))
            percentiles.append(0.0)
            n_scored += 1
            continue

        for k in ks:
            out[f'top{k}'] += float(same[:k].any())

        rr += 1.0 / (np.argmax(same) + 1)
        rank = int(np.argmax(same)) + 1
        percentiles.append(1.0 - (rank - 1) / max(len(order) - 1, 1))
        chances.append(float((content_ids[cand] == content_ids[i]).mean()))
        n_scored += 1

    if n_scored == 0:
        return None

    # Pool the per-query counters into rates.
    for k in ks:
        out[f'top{k}'] /= n_scored
    out['mrr'] = rr / n_scored
    out['chance_top1'] = float(np.mean(chances)) if chances else float('nan')
    out['rank_percentile'] = float(np.mean(percentiles)) if percentiles else float('nan')
    out['n_queries'] = int(n_scored)
    out['lift_top1'] = _sub(out['top1'], out['chance_top1'])

    return out


def build_scoreboard(
    word_emb: np.ndarray,
    word_meta: 'pd.DataFrame',
    comparison: list[dict[str, Any]],
    sent_emb: np.ndarray,
    sent_content_ids: np.ndarray,
    sent_meta: 'pd.DataFrame | None',
    config: Any | None,
    word_band_power: np.ndarray | None = None,
) -> dict[str, Any]:
    """Assembles the honest scoreboard from already-computed evaluation artefacts."""
    holdout = holdout_subject(config)

    # A factored model's thought code is the content subspace, so the headline is judged on those dims.
    model_cfg = getattr(config, 'model', None)
    if model_cfg is not None and getattr(model_cfg, 'factored', False):
        cdim = int(getattr(model_cfg, 'content_dim', word_emb.shape[1]))
        word_emb = np.asarray(word_emb)[:, :cdim]
        sent_emb = np.asarray(sent_emb)[:, :cdim]

    board: dict[str, Any] = {
        'is_loso': holdout is not None,
        'holdout_subject': holdout,
        'factored': bool(model_cfg is not None and getattr(model_cfg, 'factored', False)),
        'lift_over_raw': lift_over_raw(comparison),
    }

    # Replace the whitening-corrupted fallback control with the genuinely-raw one when band power exists.
    control = raw_content_positive_control(word_band_power, word_meta)
    if control is not None:
        board['lift_over_raw']['content_probe'] = control

    # The LOSO block: geometry and cross-subject retrieval for the stranger alone.
    if holdout is not None:
        board['held_out_geometry'] = held_out_geometry(word_emb, word_meta, holdout)
        subjects = (
            sent_meta['subject'].to_numpy()
            if sent_meta is not None and 'subject' in sent_meta.columns
            else None
        )
        board['held_out_retrieval'] = (
            cross_subject_holdout_retrieval(sent_emb, sent_content_ids, subjects, holdout)
            if subjects is not None
            else None
        )
    return board


def render_markdown(board: dict[str, Any]) -> str:
    """Renders the scoreboard as a compact Markdown block for the top of the report."""
    lines = ['## Scoreboard — the honest headline', '']
    cp = board['lift_over_raw'].get('content_probe', {})
    ok = '✅ PASS' if cp.get('passes') else '❌ FAIL'
    lines.append(
        f'**Content-probe positive control:** {ok} '
        f'(raw band-power reads lexical content at R²={cp.get("raw_content_r2_best", float("nan")):.3f}, '
        f'floor {cp.get("floor")}). '
        + (
            'The probe can detect content, so a 0% content budget is a real absence.'
            if cp.get('passes')
            else 'The probe cannot read content even from raw features — fix the probe before trusting "content 0%".'
        )
    )
    lines.append('')

    if board['is_loso']:
        g = board.get('held_out_geometry') or {}
        r = board.get('held_out_retrieval') or {}
        lines.append(f'**Held-out subject:** `{board["holdout_subject"]}` (the away game).')
        lines.append('')
        if g:
            lines += [
                f'- Effective-rank ratio (held-out only): **{g.get("effective_rank_ratio", float("nan")):.3f}** '
                '(want high — not collapsed)',
                f'- Anisotropy (held-out only): **{g.get("anisotropy", float("nan")):.3f}** '
                '(want low — not a cone)',
                f'- Content variance budget (held-out only): **{_pct(g.get("content_variance"))}**',
            ]
        if r:
            lines.append(
                f'- Cross-subject held-out retrieval Top-1: **{_pct(r.get("top1"))}** '
                f'vs chance {_pct(r.get("chance_top1"))} — lift **{_signed_pct(r.get("lift_top1"))}** '
                f'({r.get("n_queries")} queries)'
            )
        lines.append('')

    lines += [
        '**Lift over raw band-power** (ZTE - raw; positive = the encoder earns its place):',
        '',
    ]
    lines.append('| target | kind | ZTE | raw | lift | ')
    lines.append('| --- | --- | --- | --- | --- |')
    for t, v in board['lift_over_raw'].items():
        if t == 'content_probe':
            continue
        kind = (
            'content ▲' if v.get('is_content') else ('identity ▼' if v.get('is_identity') else '—')
        )
        lines.append(
            f'| {t} | {kind} | {_num(v.get("zte_linear"))} | {_num(v.get("raw_linear"))} '
            f'| {_signed(v.get("lift_linear"))} |'
        )
    lines.append('')
    return '\n'.join(lines)


def _num(v: float | None) -> str:
    """Formats a score, or a dash when missing."""
    return '—' if v is None else f'{v:.3f}'


def _signed(v: float | None) -> str:
    """Formats a delta with an explicit sign, or a dash when missing."""
    return '—' if v is None else f'{v:+.3f}'


def _pct(v: float | None) -> str:
    """Formats a fraction as a percentage, or a dash when missing."""
    return '—' if v is None else f'{100 * v:.1f}%'


def _signed_pct(v: float | None) -> str:
    """Formats a fraction as signed percentage points, or a dash when missing."""
    return '—' if v is None else f'{100 * v:+.2f}pp'
