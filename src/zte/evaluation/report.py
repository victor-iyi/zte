"""Orchestrates the ZTE evaluation: metrics + figures + a written report.

`evaluate_representation` takes aligned word/sentence embeddings and their metadata and produces, in one call: a `metrics.json`,
a `comparison.csv` table, a set of figures, and a human-readable `report.md` with a pass/fail-style verdict. It is decoupled from training/inference
so it can be unit-tested with arrays alone; `zte.cli.evaluate` wires it to a checkpoint + dataset.
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zte.evaluation import metrics as M
from zte.evaluation import plots as P
from zte.evaluation.analogy import analogy_report
from zte.evaluation.breakdown import stratified_report, stratified_retrieval
from zte.inference.retrieval import NearestNeighborIndex
from zte.logging_utils import get_logger
from zte.training.metrics import noise_matched

_LOG = get_logger('evaluation.report')


def _encode(values: np.ndarray) -> np.ndarray:
    """Encodes categorical values to integer codes.

    Args:
        values (np.ndarray): Categorical labels.

    Returns:
        np.ndarray: Integer codes `(n_samples,)`.
    """
    return pd.factorize(pd.Series(values))[0]


def _adjacency_pairs(word_meta: pd.DataFrame) -> np.ndarray:
    """Builds positive pairs of adjacent words within each sentence (for alignment).

    Args:
        word_meta (pd.DataFrame): Word metadata with subject/task/sentence_idx/word_idx.

    Returns:
        np.ndarray: Integer pairs `(n_pairs, 2)` of row indices into the embeddings.
    """
    wm = word_meta.reset_index(drop=True)
    pairs: list[tuple[int, int]] = []
    for _, grp in wm.groupby(['subject', 'task', 'sentence_idx']):
        rows = grp.sort_values('word_idx').index.to_numpy()
        pairs.extend(zip(rows[:-1].tolist(), rows[1:].tolist(), strict=True))
    return np.asarray(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)


def _word_targets(word_meta: pd.DataFrame) -> dict[str, tuple[np.ndarray, str]]:
    """Builds the supervised probe targets available from word metadata.

    Args:
        word_meta (pd.DataFrame): Word metadata.

    Returns:
        dict[str, tuple[np.ndarray, str]]: target name -> (values, task).
    """
    targets: dict[str, tuple[np.ndarray, str]] = {}
    if 'word_len' in word_meta:
        targets['word_len'] = (word_meta['word_len'].to_numpy(), 'regression')
    if 'log_freq' in word_meta:
        targets['log_freq'] = (word_meta['log_freq'].to_numpy(), 'regression')
    if 'subject' in word_meta and word_meta['subject'].nunique() > 1:
        targets['subject'] = (_encode(word_meta['subject'].to_numpy()), 'classification')
    if 'task' in word_meta and word_meta['task'].nunique() > 1:
        targets['task'] = (_encode(word_meta['task'].to_numpy()), 'classification')
    return targets


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    """Renders comparison rows as a Markdown table.

    Args:
        rows (list[dict[str, Any]]): Comparison rows.

    Returns:
        str: A Markdown table.
    """
    head = '| target | representation | metric | linear | kNN | baseline |\n'
    head += '| --- | --- | --- | --- | --- | --- |\n'
    body = ''.join(
        f'| {r["target"]} | {r["representation"]} | {r["metric"]} | '
        f'{r["linear_score"]} | {r["knn_score"]} | {r["baseline"]} |\n'
        for r in rows
    )
    return head + body


def evaluate_representation(
    word_emb: np.ndarray,
    word_meta: pd.DataFrame,
    raw_word_feats: np.ndarray,
    sent_emb: np.ndarray,
    sent_content_ids: np.ndarray,
    out_dir: str | Path,
    run_name: str = 'zte-eval',
    sent_meta: pd.DataFrame | None = None,
    word_band_power: np.ndarray | None = None,
    config: Any | None = None,
    tensorboard: bool | str = False,
    interactive: bool = True,
) -> dict[str, Any]:
    """Runs the full evaluation and writes metrics, tables, figures and a report.

    Beyond the global probes/retrieval/health, this computes **per-subject / per-task / per-category** breakdowns, **vector-arithmetic transfer**
    analogies, and (when band power is supplied) **scalp-region importance**, then emits an interactive HTML explorer and a rich TensorBoard log.

    Args:
        word_emb (np.ndarray): Word-level ZTE embeddings `(n_words, embed_dim)`.
        word_meta (pd.DataFrame): Aligned word metadata (word/word_len/log_freq/ subject/task/category/sentence_idx/word_idx), length `n_words`.
        raw_word_feats (np.ndarray): Aligned raw band-power features `(n_words, n_features)` for the baseline comparison.
        sent_emb (np.ndarray): Sentence-level embeddings `(n_sentences, embed_dim)`.
        sent_content_ids (np.ndarray): Content/group id per sentence `(n_sentences,)` (same stimulus across subjects shares an id).
        out_dir (str | Path): Output directory for artifacts.
        run_name (str): Identifier used in the report header.
        sent_meta (pd.DataFrame | None): Aligned sentence metadata (with `category`) enabling per-category retrieval breakdowns and projector colouring.
        word_band_power (np.ndarray | None): Aligned per-word band power
            `(n_words, n_bands, n_channels)` for scalp-region importance (skipped when `None`).
        config (Any | None): The run :class:`~zte.config.ZTEConfig` for HParams logging.
        tensorboard (bool | str): `True` (write under `out/tb/run_name`), a path string, or `False` to disable TensorBoard logging.
        interactive (bool): Whether to write the interactive HTML explorer.

    Returns:
        dict[str, Any]: The full metrics dictionary (also written to `metrics.json`).
    """
    out = Path(out_dir)
    fig_dir = out / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Transfer probes: ZTE vs raw band-power vs noise-matched control.
    representations = {
        'ZTE': np.asarray(word_emb, dtype=np.float32),
        'raw band-power': np.asarray(raw_word_feats, dtype=np.float32),
        'noise (matched)': noise_matched(np.asarray(raw_word_feats, dtype=np.float32)),
    }
    targets = _word_targets(word_meta)
    comparison = M.representation_comparison(representations, targets)

    # 2) Geometry / health (with adjacency positives for alignment).
    pairs = _adjacency_pairs(word_meta)
    health = M.embedding_health(word_emb, pairs=pairs)

    # 3) Content retrieval (sentence-level across subjects, and word-level by token).
    sent_ret = M.content_retrieval(sent_emb, np.asarray(sent_content_ids))
    word_ret = M.content_retrieval(word_emb, _encode(word_meta['word'].to_numpy()))

    # 4) Stratified breakdowns, vector arithmetic, and scalp-region importance.
    breakdown_words = stratified_report(word_emb, word_meta)
    breakdown_categories = (
        stratified_retrieval(sent_emb, sent_meta, np.asarray(sent_content_ids), 'category')
        if sent_meta is not None
        else []
    )
    analogy = analogy_report(word_emb, word_meta, raw_word_feats)
    region_rows = _region_importance(word_band_power, word_meta)

    metrics: dict[str, Any] = {
        'run_name': run_name,
        'n_word_embeddings': int(len(word_emb)),
        'n_sentence_embeddings': int(len(sent_emb)),
        'embedding_health': health,
        'sentence_retrieval': sent_ret,
        'word_retrieval': word_ret,
        'probe_comparison': comparison,
        'breakdown_words': breakdown_words,
        'retrieval_by_category': breakdown_categories,
        'analogy': analogy,
        'region_importance': region_rows,
        'verdict': _verdict(comparison, health, sent_ret, analogy),
    }

    # 5) Figures (each guarded so tiny inputs never abort the run).
    figures = _render_figures(
        word_emb, word_meta, sent_emb, sent_content_ids, comparison, sent_ret, fig_dir
    )
    figures += _render_extended_figures(analogy, breakdown_words, region_rows, fig_dir)
    metrics['figures'] = [str(p.relative_to(out)) for p in figures]

    # 6) Interactive HTML explorer (self-contained; static PNG fallback).
    if interactive:
        metrics['interactive'] = _write_interactive(word_emb, word_meta, out)

    # 7) TensorBoard (projector + hparams + scalars + histograms + figures + text).
    if tensorboard:
        tb_dir = tensorboard if isinstance(tensorboard, str) else str(out / 'tb' / run_name)
        _write_tensorboard(
            tb_dir, word_emb, word_meta, sent_emb, sent_meta, metrics, figures, config
        )

    # 8) Persist artifacts.
    (out / 'metrics.json').write_text(
        json.dumps(metrics, indent=2, default=float), encoding='utf-8'
    )
    pd.DataFrame(comparison).to_csv(out / 'comparison.csv', index=False)
    if breakdown_words:
        pd.DataFrame(breakdown_words).to_csv(out / 'breakdown.csv', index=False)
    if region_rows:
        pd.DataFrame(region_rows).to_csv(out / 'region_importance.csv', index=False)
    (out / 'report.md').write_text(
        _render_report(metrics, comparison, figures, out), encoding='utf-8'
    )
    _LOG.info('Evaluation written to %s (%d figures)', out, len(figures))
    return metrics


def _region_importance(
    word_band_power: np.ndarray | None, word_meta: pd.DataFrame
) -> list[dict[str, Any]]:
    """Scalp-region importance for reading vs cognitive targets (empty if no band power)."""
    if word_band_power is None:
        return []
    from zte.data.regions import region_importance

    targets: dict[str, tuple[np.ndarray, str]] = {}
    if 'word_len' in word_meta:
        targets['word_len (reading)'] = (word_meta['word_len'].to_numpy(), 'regression')
    freq_col = 'corpus_log_freq' if 'corpus_log_freq' in word_meta else 'log_freq'
    if freq_col in word_meta:
        targets['frequency (lexical)'] = (word_meta[freq_col].to_numpy(), 'regression')
    if 'task' in word_meta and word_meta['task'].nunique() > 1:
        targets['task (cognitive)'] = (_encode(word_meta['task'].to_numpy()), 'classification')
    if 'subject' in word_meta and word_meta['subject'].nunique() > 1:
        targets['subject (identity)'] = (_encode(word_meta['subject'].to_numpy()), 'classification')
    if not targets:
        return []
    return region_importance(word_band_power, targets)  # type: ignore[arg-type]


def _write_interactive(word_emb: np.ndarray, word_meta: pd.DataFrame, out: Path) -> str | None:
    """Writes the interactive explorer, returning its path relative to `out`."""
    from zte.evaluation.interactive import embedding_explorer_html

    try:
        path = embedding_explorer_html(
            word_emb, word_meta, out / 'interactive' / 'word_explorer.html'
        )
        return str(path.relative_to(out))
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:  # pragma: no cover
        _LOG.warning('Interactive explorer failed: %r', exc)
        return None


def _render_extended_figures(
    analogy: dict[str, Any],
    breakdown_words: list[dict[str, Any]],
    region_rows: list[dict[str, Any]],
    fig_dir: Path,
) -> list[Path]:
    """Renders analogy / breakdown / region figures, skipping any that fail."""
    import matplotlib.pyplot as plt

    written: list[Path] = []

    def _save(fig: Any, name: str) -> None:
        path = fig_dir / name
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        written.append(path)

    builders: list[tuple[str, Any]] = [('analogy_transfer.png', lambda: P.analogy_bars(analogy))]
    if region_rows:
        builders.append(('region_importance.png', lambda: P.region_importance_heatmap(region_rows)))
    for group, metric in (('subject', 'retrieval_top1'), ('task', 'retrieval_top1')):
        if any(r.get('group') == group for r in breakdown_words):
            builders.append(
                (
                    f'breakdown_{group}.png',
                    lambda g=group, m=metric: P.breakdown_bars(breakdown_words, m, g),
                )
            )
    for name, builder in builders:
        try:
            _save(builder(), name)
        except (ValueError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
            _LOG.warning('Skipped figure %s: %r', name, exc)
    return written


def _write_tensorboard(
    tb_dir: str,
    word_emb: np.ndarray,
    word_meta: pd.DataFrame,
    sent_emb: np.ndarray,
    sent_meta: pd.DataFrame | None,
    metrics: dict[str, Any],
    figures: list[Path],
    config: Any | None,
) -> None:
    """Writes the full TensorBoard log (projector, hparams, scalars, images, text)."""
    from zte.evaluation.tensorboard import TensorBoardReporter

    with TensorBoardReporter(tb_dir) as tb:
        if not tb.enabled:
            return
        tb.log_embeddings(word_emb, word_meta, tag='word_embeddings')
        if sent_meta is not None:
            tb.log_embeddings(
                sent_emb,
                sent_meta,
                tag='sentence_embeddings',
                columns=('subject', 'task', 'category'),
            )
        tb.log_embedding_stats(word_emb)
        tb.log_scalars('health', metrics['embedding_health'])
        tb.log_scalars('retrieval/sentence', metrics['sentence_retrieval'])
        tb.log_scalars('retrieval/word', metrics['word_retrieval'])
        tb.log_scalars('analogy/subject', metrics['analogy'].get('subject_transfer', {}))
        tb.log_scalars('analogy/task', metrics['analogy'].get('task_transfer', {}))
        for row in metrics['breakdown_words']:
            tb.log_scalars(f'breakdown/{row["group"]}/{row["value"]}', row)
        tb.log_table('tables/probe_comparison', metrics['probe_comparison'])
        tb.log_table('tables/region_importance', metrics['region_importance'])
        for fig_path in figures:
            tb.log_image_file(f'figures/{fig_path.stem}', fig_path)
        headline = {
            'health/effective_rank_ratio': metrics['embedding_health'].get('effective_rank_ratio'),
            'retrieval/sentence_top1': metrics['sentence_retrieval'].get('top1'),
            'analogy/subject_top1': metrics['analogy'].get('subject_transfer', {}).get('top1'),
        }
        if config is not None:
            tb.log_hparams(config, {k: v for k, v in headline.items() if v is not None})


def _render_figures(
    word_emb: np.ndarray,
    word_meta: pd.DataFrame,
    sent_emb: np.ndarray,
    sent_content_ids: np.ndarray,
    comparison: list[dict[str, Any]],
    sent_ret: dict[str, float],
    fig_dir: Path,
) -> list[Path]:
    """Renders and saves all evaluation figures, skipping any that fail.

    Args:
        word_emb (np.ndarray): Word embeddings.
        word_meta (pd.DataFrame): Word metadata.
        sent_emb (np.ndarray): Sentence embeddings.
        sent_content_ids (np.ndarray): Sentence content ids.
        comparison (list[dict[str, Any]]): Probe comparison rows.
        sent_ret (dict[str, float]): Sentence retrieval metrics.
        fig_dir (Path): Output directory.

    Returns:
        list[Path]: Written figure paths.
    """
    import matplotlib.pyplot as plt

    written: list[Path] = []

    def _save(fig: Any, name: str) -> None:
        path = fig_dir / name
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        written.append(path)

    builders: list[tuple[str, Any]] = [
        (
            'pca_by_word_length.png',
            lambda: P.scatter_2d(
                word_emb,
                word_meta['word_len'].to_numpy(),
                'ZTE space by word length',
                categorical=False,
                label_name='word length',
            ),
        ),
        (
            'pca_by_subject.png',
            lambda: P.scatter_2d(
                word_emb,
                word_meta['subject'].to_numpy(),
                'ZTE space by subject',
                categorical=True,
                label_name='subject',
            ),
        ),
        ('probe_linear.png', lambda: P.bar_probe_comparison(comparison, 'linear_score')),
        ('probe_knn.png', lambda: P.bar_probe_comparison(comparison, 'knn_score')),
        ('embedding_health.png', lambda: P.embedding_health_plot(word_emb)),
        (
            'similarity_by_content.png',
            lambda: P.similarity_distribution(
                sent_emb,
                np.asarray(sent_content_ids),
                'Sentence similarity: same vs different content',
            ),
        ),
    ]
    ks = {int(key[3:]): val for key, val in sent_ret.items() if key.startswith('top')}
    if ks:
        builders.append(
            (
                'retrieval_sentence.png',
                lambda: P.retrieval_curve(
                    ks, sent_ret.get('chance_top1', 0.0), 'Cross-subject sentence retrieval'
                ),
            )
        )
    if 'log_freq' in word_meta:
        builders.append(
            ('probe_logfreq_scatter.png', lambda: _logfreq_scatter(word_emb, word_meta))
        )

    for name, builder in builders:
        try:
            _save(builder(), name)
        except (ValueError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
            _LOG.warning('Skipped figure %s: %r', name, exc)
    return written


def _logfreq_scatter(word_emb: np.ndarray, word_meta: pd.DataFrame) -> Any:
    """Builds a kNN leave-one-out predicted-vs-true scatter for log frequency.

    Args:
        word_emb (np.ndarray): Word embeddings.
        word_meta (pd.DataFrame): Word metadata containing `log_freq`.

    Returns:
        Figure: The predicted-vs-true scatter.
    """
    index = NearestNeighborIndex(word_emb, word_meta[['log_freq']].reset_index(drop=True))
    pred = index.predict(
        word_emb, 'log_freq', k=10, task='regression', self_indices=np.arange(len(word_emb))
    )
    return P.probe_scatter(word_meta['log_freq'].to_numpy(), pred, 'kNN probe: log frequency')


def _verdict(
    comparison: list[dict[str, Any]],
    health: dict[str, float],
    sent_ret: dict[str, float],
    analogy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derives simple pass/fail-style headline checks from the metrics.

    Args:
        comparison (list[dict[str, Any]]): Probe comparison rows.
        health (dict[str, float]): Geometry/health metrics.
        sent_ret (dict[str, float]): Sentence retrieval metrics.
        analogy (dict[str, Any] | None): Vector-arithmetic transfer report.

    Returns:
        dict[str, Any]: Named boolean checks plus the figures-of-merit behind them.
    """
    zte = {r['target']: r for r in comparison if r['representation'] == 'ZTE'}
    noise = {r['target']: r for r in comparison if r['representation'] == 'noise (matched)'}
    beats_noise = [
        t for t in zte if zte[t]['linear_score'] > noise.get(t, {}).get('linear_score', -1) + 1e-3
    ]
    verdict = {
        'beats_noise_on': beats_noise,
        'beats_noise_all_targets': len(beats_noise) == len(zte) and len(zte) > 0,
        'no_collapse': bool(
            health.get('effective_rank_ratio', 0) > 0.1 and health.get('dead_dim_fraction', 1) < 0.5
        ),
        'retrieval_above_chance': bool(
            sent_ret.get('top1', 0) > sent_ret.get('chance_top1', 1) + 1e-6
        ),
    }
    if analogy:
        st = analogy.get('subject_transfer', {})
        verdict['subject_arithmetic_above_chance'] = bool(
            st.get('top1', 0) > st.get('chance_top1', 1) + 1e-6
        )
    return verdict


def _render_report(
    metrics: dict[str, Any],
    comparison: list[dict[str, Any]],
    figures: list[Path],
    out: Path,
) -> str:
    """Renders the Markdown evaluation report.

    Args:
        metrics (dict[str, Any]): Full metrics dictionary.
        comparison (list[dict[str, Any]]): Probe comparison rows.
        figures (list[Path]): Written figure paths.
        out (Path): Report root (for relative figure links).

    Returns:
        str: The Markdown document.
    """
    health = metrics['embedding_health']
    sent = metrics['sentence_retrieval']
    verdict = metrics['verdict']
    lines = [
        f'# ZTE evaluation report -- {metrics["run_name"]}',
        '',
        f'Word embeddings: **{metrics["n_word_embeddings"]}** | '
        f'sentence embeddings: **{metrics["n_sentence_embeddings"]}**',
        '',
        '## Verdict',
        '',
        f'- Beats noise control on: **{", ".join(verdict["beats_noise_on"]) or "none"}** '
        f'(all targets: {verdict["beats_noise_all_targets"]})',
        f'- No representation collapse: **{verdict["no_collapse"]}** '
        f'(effective-rank ratio {health["effective_rank_ratio"]:.2f}, '
        f'dead dims {health["dead_dim_fraction"]:.0%})',
        f'- Cross-subject retrieval above chance: **{verdict["retrieval_above_chance"]}** '
        f'(Top-1 {sent.get("top1", float("nan")):.3f} vs chance '
        f'{sent.get("chance_top1", float("nan")):.3f})',
        '',
        '## Transfer probes (frozen embeddings)',
        '',
        'Higher is better. R^2 for regression, accuracy for classification; dashed '
        'baseline = predict-the-mean / majority. ZTE should beat the noise control '
        'and rival raw band-power in far fewer dimensions.',
        '',
        _markdown_table(comparison),
        '## Embedding geometry / health',
        '',
        f'- Effective rank: **{health["effective_rank"]:.1f}** / {health["embed_dim"]} '
        f'(ratio {health["effective_rank_ratio"]:.2f})',
        f'- Uniformity: {health["uniformity"]:.3f} (lower = more spread)',
        f'- Anisotropy: {health["anisotropy"]:.3f} (lower = better)',
        f'- Alignment (adjacent words): {health.get("alignment", float("nan")):.3f}',
        f'- Dead dimensions: {health["dead_dim_fraction"]:.1%} | mean norm '
        f'{health["mean_norm"]:.2f}',
        '',
        '## Retrieval',
        '',
        f'- Sentence (cross-subject): Top-1 {sent.get("top1", float("nan")):.3f}, '
        f'Top-5 {sent.get("top5", float("nan")):.3f}, MRR {sent.get("mrr", float("nan")):.3f}, '
        f'chance {sent.get("chance_top1", float("nan")):.3f}',
        f'- Word (same token): Top-1 {metrics["word_retrieval"].get("top1", float("nan")):.3f}, '
        f'chance {metrics["word_retrieval"].get("chance_top1", float("nan")):.3f}',
        '',
    ]
    lines += _extended_report_sections(metrics)
    lines += ['## Figures', '']
    lines += [f'![{p.stem}]({p.relative_to(out).as_posix()})' for p in figures]
    lines.append('')
    return '\n'.join(lines)


def _extended_report_sections(metrics: dict[str, Any]) -> list[str]:
    """Markdown for the arithmetic, breakdown, per-category and region analyses."""
    lines: list[str] = []
    analogy = metrics.get('analogy', {})
    st = analogy.get('subject_transfer', {})
    if st:
        lines += [
            '## Vector arithmetic (thought-code analogies)',
            '',
            'Can we cancel *who* produced a thought? For a stimulus token `t`, '
            '`emb(t, A) - centroid(A) + centroid(B)` should retrieve `emb(t, B)`.',
            '',
            f'- Subject transfer: Top-1 **{st.get("top1", float("nan")):.3f}**, '
            f'Top-5 {st.get("top5", float("nan")):.3f}, MRR {st.get("mrr", float("nan")):.3f} '
            f'(chance {st.get("chance_top1", float("nan")):.3f}, '
            f'{int(st.get("n_queries", 0))} analogies)',
        ]
        raw = analogy.get('subject_transfer_raw', {})
        if raw:
            lines.append(
                f'- Raw-feature control: Top-1 {raw.get("top1", float("nan")):.3f} '
                '(ZTE should beat it -- arithmetic is a property of the learned space)'
            )
        tt = analogy.get('task_transfer', {})
        if tt:
            lines.append(
                f'- Task transfer: Top-1 {tt.get("top1", float("nan")):.3f} '
                f'(chance {tt.get("chance_top1", float("nan")):.3f})'
            )
        examples = analogy.get('examples', [])
        if examples:
            lines += ['', '| expression | retrieved | hit | cos |', '| --- | --- | --- | --- |']
            lines += [
                f'| {e["expression"]} | {e["retrieved"]} | {"✓" if e["hit"] else "·"} | '
                f'{e["similarity"]} |'
                for e in examples[:6]
            ]
        lines.append('')

    region = metrics.get('region_importance', [])
    if region:
        frame = pd.DataFrame(region).pivot(index='region', columns='target', values='importance')
        lines += ['## Scalp-region importance (which areas encode what)', '']
        lines.append('| region | ' + ' | '.join(str(c) for c in frame.columns) + ' |')
        lines.append('| --- |' + ' --- |' * len(frame.columns))
        for region_name, row in frame.iterrows():
            lines.append(
                f'| {region_name} | ' + ' | '.join(f'{v:.2f}' for v in row.to_numpy()) + ' |'
            )
        lines.append('')

    by_cat = metrics.get('retrieval_by_category', [])
    if by_cat:
        lines += [
            '## Cross-subject retrieval by sentence category',
            '',
            '| category | n | Top-1 | Top-5 | MRR | chance |',
            '| --- | --- | --- | --- | --- | --- |',
        ]
        lines += [
            f'| {r["value"]} | {r["n"]} | {r["top1"]} | {r["top5"]} | {r["mrr"]} | {r["chance_top1"]} |'
            for r in by_cat
        ]
        lines.append('')

    breakdown = metrics.get('breakdown_words', [])
    subject_rows = [r for r in breakdown if r.get('group') == 'subject']
    if subject_rows:
        lines += [
            '## Per-subject probe / retrieval',
            '',
            '| subject | n | r2(word_len) | retrieval Top-1 | eff-rank ratio |',
            '| --- | --- | --- | --- | --- |',
        ]
        lines += [
            f'| {r["value"]} | {r.get("n", "")} | {r.get("r2_word_len", "-")} | '
            f'{r.get("retrieval_top1", "-")} | {r.get("eff_rank_ratio", "-")} |'
            for r in subject_rows
        ]
        lines.append('')
    return lines
