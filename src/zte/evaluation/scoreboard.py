"""The honest scoreboard -- *the held-out number, stated as a lift over raw*.

The full evaluation computes geometry, probes and retrieval over **every** embedding, which for a LOSO run
mixes the 11 training subjects with the 1 held-out stranger and dilutes the only number that matters.

1. **On the held-out subject alone**, does the space stay healthy (effective rank, anisotropy) and
    does it stop spending itself on identity (variance budget)?
2. **Is every probe/retrieval number a lift over the raw band-power control?** Raw band power currently
    beats the learned embedding on a stranger, so an absolute score is meaningless -- only `ZTE - raw` is progress.
3. **Can the content probe even detect content in principle?** A positive control: the raw features must expose
    word length / frequency, or "content 0%" is a dead probe, not a meaningful absence.

Everything here is derived from artefacts the evaluation already computed (the probe comparison rows, the sentence embeddings,
the word metadata), so it adds no model work and cannot disagree with the rest of the report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zte.evaluation import metrics as M
from zte.evaluation.neurons import neuron_report

if TYPE_CHECKING:
    import pandas as pd

# Probe targets that count as content (a meaning/lexical code should carry these) and
# identity (a thought code should NOT). Mirrors zte.evaluation.neurons.
_CONTENT_TARGETS: tuple[str, ...] = ('word_len', 'log_freq', 'category')
_IDENTITY_TARGETS: tuple[str, ...] = ('subject',)
# The positive-control floor: raw band power must expose lexical content at least this
# well (linear R2), or the content probe is not trustworthy. The bake-off saw ~0.09-0.14.
_CONTENT_PROBE_FLOOR: float = 0.02


def holdout_subject(config: Any | None) -> str | None:
    """Returns the LOSO held-out subject, or `None` when the run is not LOSO."""
    train = getattr(config, 'train', None)
    if train is None or getattr(train, 'split', None) != 'by_subject_loso':
        return None
    return getattr(train, 'loso_holdout_subject', None)


def lift_over_raw(comparison: list[dict[str, Any]]) -> dict[str, Any]:
    """Turns the probe-comparison rows into per-target `ZTE − raw` lifts.

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

    # Positive control: raw features must expose lexical content above the floor.
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
    }
    return lifts


def _sub(a: float | None, b: float | None) -> float | None:
    """`a - b`, tolerating missing operands."""
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 4)


def held_out_geometry(
    word_emb: np.ndarray, word_meta: 'pd.DataFrame', holdout: str
) -> dict[str, Any] | None:
    """Geometry + variance budget computed on the held-out subject's rows *only*.

    This is the honest "does the space work for a stranger" number, undiluted by the training subjects that share the same run.
    """
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

    For each sentence read by the held-out subject, search every *other* subject's
    readings and score a hit when a neighbour shares the stimulus. This is the true
    north-star: not "does content cluster" in the mixed space, but "can we retrieve the
    held-out person's specific reading from people the model trained on".

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
            # still counts as a query (a miss); chance uses this query's gallery
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
) -> dict[str, Any]:
    """Assembles the honest scoreboard from already-computed evaluation artefacts."""
    holdout = holdout_subject(config)
    # Unit C: for a factored model the *thought code* is the content subspace, so the
    # geometry/retrieval headline is judged on those dims, not the full embedding.
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
    return '—' if v is None else f'{v:.3f}'


def _signed(v: float | None) -> str:
    return '—' if v is None else f'{v:+.3f}'


def _pct(v: float | None) -> str:
    return '—' if v is None else f'{100 * v:.1f}%'


def _signed_pct(v: float | None) -> str:
    return '—' if v is None else f'{100 * v:+.2f}pp'
