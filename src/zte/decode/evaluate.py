"""Full EEG→language decoding evaluation: metrics, report, predictions, figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from zte.decode.alignment import OTCLIPAligner
from zte.decode.decoders import PrefixLanguageDecoder, RetrievalDecoder
from zte.decode.metrics import (
    bleu_score,
    cross_modal_retrieval,
    decode_breakdown,
    exact_match,
    noise_anchored_retrieval,
    token_f1,
    wer_score,
)
from zte.logging_utils import get_logger

_LOG = get_logger('decode.evaluate')


def evaluate_decoding(
    eeg_emb: np.ndarray,
    text_emb: np.ndarray,
    texts: list[str],
    *,
    meta: pd.DataFrame | dict[str, Any] | None = None,
    aligner: OTCLIPAligner | None = None,
    decoder: RetrievalDecoder | PrefixLanguageDecoder | None = None,
    out_dir: str | Path | None = None,
    run_name: str = 'decode-eval',
    predictions: list[str] | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any]:
    """Evaluates cross-modal retrieval and optional generative decoding.

    When ``out_dir`` is set, writes ``metrics.json``, ``report.md``,
    ``predictions.csv`` and figures. Returns a metrics dict including verdict
    flags ``retrieval_above_chance`` and ``beats_noise_anchor``.

    Args:
        eeg_emb: Query EEG embeddings ``(N, D_eeg)``.
        text_emb: Gallery text embeddings ``(N, D_text)`` aligned with ``texts``.
        texts: Ground-truth strings of length ``N``.
        meta: Optional metadata (``subject``, ``task``, …).
        aligner: Optional trained OT-CLIP aligner.
        decoder: Optional retrieval or prefix-LM decoder for hypotheses.
        out_dir: Destination directory for artifacts (``None`` skips disk writes).
        run_name: Label used in the Markdown report.
        predictions: Optional precomputed hypotheses (overrides ``decoder``).
        ks: Top-K cutoffs.

    Returns:
        Metrics dict (also written to disk when ``out_dir`` is set).
    """
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    eeg = np.asarray(eeg_emb, dtype=np.float32)
    text = np.asarray(text_emb, dtype=np.float32)
    texts = list(texts)
    if len(eeg) != len(texts) or len(text) != len(texts):
        raise ValueError('eeg_emb, text_emb and texts must share the same length.')

    if aligner is not None:
        aligner.eval()
        with torch.no_grad():
            eeg_z = aligner.encode_eeg(torch.from_numpy(eeg)).cpu().numpy().astype(np.float32)
            text_z = aligner.encode_text(torch.from_numpy(text)).cpu().numpy().astype(np.float32)
    else:
        eeg_z, text_z = eeg, text

    n = len(texts)
    chance = 1.0 / max(n, 1)
    retrieval = cross_modal_retrieval(eeg_z, text_z, ks=ks)
    anchored = noise_anchored_retrieval(eeg_z, text_z, ks=ks)

    hyps = predictions
    if hyps is None and decoder is not None:
        hyps = _decode_hypotheses(decoder, eeg, eeg_z, aligner)
    if hyps is None:
        # Default: retrieval top-1 from the aligned gallery.
        bank = RetrievalDecoder(text_z, texts, aligner=None, bank_already_aligned=True)
        hyps = bank.decode(eeg_z, k=1)

    gen_metrics = {
        'exact_match': exact_match(hyps, texts),
        'bleu': bleu_score(hyps, texts),
        'token_f1': token_f1(hyps, texts),
        'wer': wer_score(hyps, texts),
    }
    breakdown = decode_breakdown(hyps, texts, meta)

    # Flat aliases (top1 / mrr) keep the lightweight API used by smoke tests.
    metrics: dict[str, Any] = {
        'run_name': run_name,
        'n': n,
        'chance_top1': chance,
        **retrieval,
        **{f'retrieval_{k}': v for k, v in retrieval.items()},
        **anchored,
        **gen_metrics,
        'breakdown': breakdown,
        'retrieval_above_chance': bool(retrieval.get('top1', 0.0) > chance),
        'beats_noise_anchor': bool(anchored.get('beats_noise_anchor', 0.0) >= 0.5),
        'verdict': {
            'retrieval_above_chance': bool(retrieval.get('top1', 0.0) > chance),
            'beats_noise_anchor': bool(anchored.get('beats_noise_anchor', 0.0) >= 0.5),
        },
    }

    if out is not None:
        fig_dir = out / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        frame = _predictions_frame(hyps, texts, meta)
        frame.to_csv(out / 'predictions.csv', index=False)
        (out / 'metrics.json').write_text(
            json.dumps(metrics, indent=2, default=str), encoding='utf-8'
        )
        (out / 'report.md').write_text(_render_report(metrics, run_name), encoding='utf-8')
        _save_figures(fig_dir, retrieval, breakdown, chance, ks)

    _LOG.info(
        'decode-eval %s | n=%d top1=%.3f (chance=%.3f) lift_top1=%.3f em=%.3f',
        run_name,
        n,
        retrieval.get('top1', float('nan')),
        chance,
        anchored.get('lift_top1', float('nan')),
        gen_metrics['exact_match'],
    )
    return metrics


def _decode_hypotheses(
    decoder: RetrievalDecoder | PrefixLanguageDecoder,
    eeg_raw: np.ndarray,
    eeg_aligned: np.ndarray,
    aligner: OTCLIPAligner | None,
) -> list[str]:
    """Produces hypothesis strings from a decoder."""
    if isinstance(decoder, RetrievalDecoder):
        # Prefer raw EEG when the decoder owns an aligner; else use aligned.
        query = eeg_raw if decoder.aligner is not None else eeg_aligned
        return decoder.decode(query, k=1)
    # Prefix LM expects the same space it was trained on (usually aligned).
    return decoder.generate(eeg_aligned)


def _predictions_frame(
    hyps: list[str],
    refs: list[str],
    meta: pd.DataFrame | dict[str, Any] | None,
) -> pd.DataFrame:
    """Builds the predictions CSV frame."""
    data: dict[str, Any] = {
        'hyp': hyps,
        'ref': refs,
        'correct': [_norm(h) == _norm(r) for h, r in zip(hyps, refs, strict=True)],
    }
    if meta is not None:
        frame = meta if isinstance(meta, pd.DataFrame) else pd.DataFrame(meta)
        for col in ('subject', 'task', 'sentence_idx', 'word'):
            if col in frame.columns and len(frame) == len(hyps):
                data[col] = frame[col].tolist()
    return pd.DataFrame(data)


def _save_figures(
    fig_dir: Path,
    retrieval: dict[str, float],
    breakdown: dict[str, Any],
    chance: float,
    ks: tuple[int, ...],
) -> None:
    """Writes retrieval curve + per-subject bar charts."""
    try:
        from zte.evaluation.plots import retrieval_curve
    except ImportError:  # pragma: no cover
        retrieval_curve = None  # type: ignore[assignment]

    topk = {k: retrieval.get(f'top{k}', float('nan')) for k in ks}
    if retrieval_curve is not None:
        fig = retrieval_curve(topk, chance=chance, title='EEG→text retrieval')
        fig.savefig(fig_dir / 'retrieval_curve.png', dpi=120, bbox_inches='tight')
        _close(fig)
    else:  # pragma: no cover
        _simple_retrieval_curve(fig_dir / 'retrieval_curve.png', topk, chance)

    by_subject = breakdown.get('by_subject') or {}
    if by_subject:
        _bar_metric(
            fig_dir / 'per_subject_exact_match.png',
            {k: v.get('exact_match', 0.0) for k, v in by_subject.items()},
            title='Exact match by subject',
            ylabel='exact_match',
        )


def _bar_metric(path: Path, values: dict[str, float], title: str, ylabel: str) -> None:
    """Simple bar chart helper."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = list(values.keys())
    scores = [values[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(labels) + 2), 4))
    ax.bar(labels, scores, color='#3b82f6', alpha=0.85)
    ax.set(title=title, ylabel=ylabel, ylim=(0, 1.05))
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    _close(fig)


def _simple_retrieval_curve(path: Path, topk: dict[int, float], chance: float) -> None:
    """Fallback retrieval curve without evaluation.plots."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ks = sorted(topk)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ks, [topk[k] for k in ks], marker='o', label='retrieval')
    ax.axhline(chance, color='crimson', linestyle='dashed', linewidth=1, label='chance')
    ax.set(xlabel='K', ylabel='Top-K accuracy', title='EEG→text retrieval', ylim=(0, 1.02))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    _close(fig)


def _render_report(metrics: dict[str, Any], run_name: str) -> str:
    """Markdown summary of decode evaluation."""
    v = metrics.get('verdict', {})
    lines = [
        f'# Decode evaluation — {run_name}',
        '',
        f'- **N**: {metrics.get("n")}',
        f'- **Top-1**: {metrics.get("retrieval_top1", metrics.get("top1", float("nan"))):.4f} '
        f'(chance {metrics.get("chance_top1", float("nan")):.4f})',
        f'- **MRR**: {metrics.get("retrieval_mrr", metrics.get("mrr", float("nan"))):.4f}',
        f'- **Noise-anchor lift (top1)**: {metrics.get("lift_top1", float("nan")):.4f}',
        f'- **Exact match**: {metrics.get("exact_match", float("nan")):.4f}',
        f'- **BLEU**: {metrics.get("bleu", float("nan")):.4f}',
        f'- **Token F1**: {metrics.get("token_f1", float("nan")):.4f}',
        f'- **WER**: {metrics.get("wer", float("nan")):.4f}',
        '',
        '## Verdict',
        '',
        f'- retrieval_above_chance: `{v.get("retrieval_above_chance")}`',
        f'- beats_noise_anchor: `{v.get("beats_noise_anchor")}`',
        '',
    ]
    return '\n'.join(lines)


def _norm(text: str) -> str:
    return ' '.join(str(text).strip().lower().split())


def _close(fig: Any) -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass
