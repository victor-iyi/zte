"""Orchestrates the ZTE evaluation: embeddings in, `metrics.json` + tables + figures + `report.md` out."""

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
    """Encodes categorical values to integer codes `(n_samples,)`."""
    return pd.factorize(pd.Series(values))[0]


def _adjacency_pairs(word_meta: pd.DataFrame) -> np.ndarray:
    """Builds `(n_pairs, 2)` row-index pairs of adjacent words within each sentence, for alignment."""
    wm = word_meta.reset_index(drop=True)
    pairs: list[tuple[int, int]] = []
    for _, grp in wm.groupby(['subject', 'task', 'sentence_idx']):
        rows = grp.sort_values('word_idx').index.to_numpy()
        pairs.extend(zip(rows[:-1].tolist(), rows[1:].tolist(), strict=True))
    return np.asarray(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)


def _word_targets(word_meta: pd.DataFrame) -> dict[str, tuple[np.ndarray, str]]:
    """Builds the `name -> (values, task)` probe targets available from the word metadata."""
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
    """Renders the probe-comparison rows as a Markdown table."""
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
    phase_word_emb: np.ndarray | None = None,
    train_vocab: set[str] | None = None,
) -> dict[str, Any]:
    """Runs the full evaluation and writes metrics, tables, figures and a report.

    Beyond the global probes/retrieval/health this adds per-subject / per-task / per-category
    breakdowns, vector-arithmetic transfer analogies and (given band power) scalp-region importance,
    then emits the interactive HTML explorers and a TensorBoard log.

    Args:
        word_emb (np.ndarray): Word-level ZTE embeddings `(n_words, embed_dim)`.
        word_meta (pd.DataFrame): Aligned word metadata (word/word_len/log_freq/subject/task/category/
            sentence_idx/word_idx), length `n_words`.
        raw_word_feats (np.ndarray): Aligned raw band-power features `(n_words, n_features)` for the baseline comparison.
        sent_emb (np.ndarray): Sentence-level embeddings `(n_sentences, embed_dim)`.
        sent_content_ids (np.ndarray): Content/group id per sentence `(n_sentences,)` (same stimulus across subjects shares an id).
        out_dir (str | Path): Output directory for artifacts.
        run_name (str): Identifier used in the report header.
        sent_meta (pd.DataFrame | None): Aligned sentence metadata (with `category`) enabling per-category retrieval breakdowns and projector colouring.
        word_band_power (np.ndarray | None): Aligned per-word band power
            `(n_words, n_bands, n_channels)` for scalp-region importance (skipped when `None`).
        config (Any | None): The run `ZTEConfig` for HParams logging.
        tensorboard (bool | str): `True` (write under `out/tb/run_name`), a path string, or `False` to disable TensorBoard logging.
        interactive (bool): Whether to write the interactive HTML explorer.
        phase_word_emb (np.ndarray | None): Embeddings of phase-scrambled EEG, added as a control representation.
        train_vocab (set[str] | None): Word types seen in training, enabling the seen-vs-novel retrieval split.

    Returns:
        dict[str, Any]: The full metrics dictionary (also written to `metrics.json`).
    """
    out = Path(out_dir)
    fig_dir = out / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 0) Label-free geometry post-processing; order matters: whiten, THEN all-but-the-top.
    obj_cfg = getattr(config, 'objective', None)
    # A raw snapshot, so report.md can show the geometry before vs after.
    word_emb_raw = np.asarray(word_emb, dtype=np.float32).copy()
    if obj_cfg is not None and getattr(obj_cfg, 'whiten', False):
        word_emb = M.whiten_features(word_emb)
        sent_emb = M.whiten_features(sent_emb)
        _LOG.info(
            'Applied ZCA whitening to the exported embeddings (config.objective.whiten=True).'
        )
    n_top = int(getattr(obj_cfg, 'all_but_top', 0) or 0) if obj_cfg is not None else 0
    if n_top > 0:
        word_emb = M.all_but_the_top(word_emb, n_top)
        sent_emb = M.all_but_the_top(sent_emb, n_top)
        _LOG.info('Removed the top-%d principal directions (config.objective.all_but_top).', n_top)
    csls_k = int(getattr(obj_cfg, 'csls_neighbors', 0) or 0) if obj_cfg is not None else 0
    use_csls = csls_k > 0
    if use_csls:
        _LOG.info(
            'Retrieval uses CSLS hubness correction (k=%d, config.objective.csls_neighbors).',
            csls_k,
        )

    # 1) Transfer probes: ZTE vs raw band-power vs noise-matched control vs phase-shuffled ZTE.
    representations = {
        'ZTE': np.asarray(word_emb, dtype=np.float32),
        'raw band-power': np.asarray(raw_word_feats, dtype=np.float32),
        'noise (matched)': noise_matched(np.asarray(raw_word_feats, dtype=np.float32)),
    }
    if phase_word_emb is not None:
        # The control must get the same post-processing as ZTE or the comparison is rigged.
        phase_word_emb = np.asarray(phase_word_emb, dtype=np.float32)
        if obj_cfg is not None and getattr(obj_cfg, 'whiten', False):
            phase_word_emb = M.whiten_features(phase_word_emb)
        if n_top > 0:
            phase_word_emb = M.all_but_the_top(phase_word_emb, n_top)
        representations['phase-shuffled ZTE'] = phase_word_emb
    targets = _word_targets(word_meta)
    comparison = M.representation_comparison(representations, targets)

    # 2) Geometry / health (with adjacency positives for alignment).
    pairs = _adjacency_pairs(word_meta)
    health = M.embedding_health(word_emb, pairs=pairs)

    # 3) Content retrieval (sentence-level across subjects, and word-level by token).
    sent_ret = M.content_retrieval(
        sent_emb,
        np.asarray(sent_content_ids),
        return_hits=True,
        return_ranks=True,
        csls=use_csls,
        csls_k=csls_k,
    )
    # Popped, not kept: the per-query vectors feed the CI verdict but would bloat metrics.json.
    sent_top1_hits = sent_ret.pop('top1_hits', [])  # type: ignore[arg-type]
    sent_ranks = sent_ret.pop('ranks', [])  # per-query ranks for the rank-distribution figure
    word_ret = M.content_retrieval(
        word_emb, _encode(word_meta['word'].to_numpy()), csls=use_csls, csls_k=csls_k
    )
    eval_seen_novel = bool(getattr(obj_cfg, 'eval_seen_novel', False)) if obj_cfg else False
    eval_freq_matched = bool(getattr(obj_cfg, 'eval_freq_matched', False)) if obj_cfg else False
    # 3.2c) Seen vs novel word types: does retrieval hold for types absent from the training split?
    word_ret_by_novelty: dict[str, Any] = {}
    if eval_seen_novel and train_vocab is not None and 'word' in word_meta.columns:
        words_arr = word_meta['word'].astype(str).to_numpy()
        seen_mask = np.array([w in train_vocab for w in words_arr])
        word_codes = _encode(word_meta['word'].to_numpy())
        for label, mask in (('seen', seen_mask), ('novel', ~seen_mask)):
            if int(mask.sum()) >= 4:
                word_ret_by_novelty[label] = M.content_retrieval(
                    word_emb[mask], word_codes[mask], csls=use_csls, csls_k=csls_k
                )
    # 3.2d) Frequency-matched distractors, so a hit cannot be a lexical-frequency shortcut.
    word_ret_freq_matched: dict[str, float] | None = None
    if eval_freq_matched:
        freq_col = 'corpus_log_freq' if 'corpus_log_freq' in word_meta else 'log_freq'
        if freq_col in word_meta.columns:
            lf = pd.to_numeric(word_meta[freq_col], errors='coerce').to_numpy()
            nbin = min(5, max(1, int(np.unique(lf[np.isfinite(lf)]).size)))
            if nbin >= 2:
                fbin = pd.qcut(pd.Series(lf), q=nbin, labels=False, duplicates='drop').to_numpy()
                word_ret_freq_matched = M.matched_content_retrieval(
                    word_emb,
                    _encode(word_meta['word'].to_numpy()),
                    fbin,
                    csls=use_csls,
                    csls_k=csls_k,
                )

    # 4) Stratified breakdowns, vector arithmetic, and scalp-region importance.
    breakdown_words = stratified_report(word_emb, word_meta)
    breakdown_categories = (
        stratified_retrieval(sent_emb, sent_meta, np.asarray(sent_content_ids), 'category')
        if sent_meta is not None
        else []
    )
    analogy = analogy_report(word_emb, word_meta, raw_word_feats, return_hits=True)
    st = analogy.get('subject_transfer', {})
    subj_top1_hits = st.pop('top1_hits', []) if isinstance(st, dict) else []
    subj_chances = st.pop('chances', []) if isinstance(st, dict) else []
    region_map = _load_region_map(config)
    region_rows = _region_importance(word_band_power, word_meta, region_map)
    region_approximate = region_map is None or region_map.approximate

    # 4b) Neuron-level interpretability: which dimensions fire, what they encode, who-vs-what budget.
    neurons = _neuron_report(word_emb, word_meta, word_band_power, region_map, config)

    # 4c) Emergent properties: do the same / related thoughts cluster ACROSS subjects (the north star)?
    emergence = _emergence_report(word_emb, word_meta, analogy)

    metrics: dict[str, Any] = {
        'run_name': run_name,
        'n_word_embeddings': int(len(word_emb)),
        'n_sentence_embeddings': int(len(sent_emb)),
        'embedding_health': health,
        'sentence_retrieval': sent_ret,
        'word_retrieval': word_ret,
        'word_retrieval_by_novelty': word_ret_by_novelty,
        'word_retrieval_freq_matched': word_ret_freq_matched,
        'probe_comparison': comparison,
        'breakdown_words': breakdown_words,
        'retrieval_by_category': breakdown_categories,
        'analogy': analogy,
        'region_importance': region_rows,
        'region_map_approximate': region_approximate,
        'neurons': _neuron_summary(neurons),
        'emergence': emergence,
        'verdict': _verdict(
            comparison,
            health,
            sent_ret,
            analogy,
            sent_top1_hits=sent_top1_hits,
            subj_top1_hits=subj_top1_hits,
            subj_chances=subj_chances,
        ),
    }

    # 4b) Honesty add-ons: permutation null, held-out cross-subject decode, anchor calibration.
    honesty = _honesty_block(word_emb, word_meta, sent_emb, sent_content_ids, config)
    metrics['honesty'] = honesty

    # 4d) The honest scoreboard: every headline metric stated as a lift over the raw control.
    from zte.evaluation.audit.scoreboard import build_scoreboard

    metrics['scoreboard'] = build_scoreboard(
        word_emb, word_meta, comparison, sent_emb, sent_content_ids, sent_meta, config
    )
    perm = honesty.get('retrieval_permutation') or {}
    if perm.get('applicable'):
        metrics['verdict']['retrieval_permutation_p'] = perm['p_value']
        metrics['verdict']['retrieval_above_chance_perm'] = perm['above_chance']
        # The headline needs both the bootstrap-CI lift and the permutation null; this only demotes.
        metrics['verdict']['retrieval_above_chance'] = bool(
            metrics['verdict']['retrieval_above_chance'] and perm['above_chance']
        )

    # 5) Figures (each guarded so tiny inputs never abort the run).
    figures = _render_figures(
        word_emb, word_meta, sent_emb, sent_content_ids, comparison, sent_ret, fig_dir
    )
    figures += _render_extended_figures(analogy, breakdown_words, region_rows, fig_dir)
    montage_csv = getattr(getattr(config, 'dataset', None), 'montage_csv', None)
    figures += _render_sota_figures(
        word_emb_raw,
        np.asarray(word_emb, dtype=np.float32),
        word_meta,
        sent_ranks,
        sent_ret,
        len(sent_emb),
        neurons,
        word_band_power,
        montage_csv,
        fig_dir,
    )
    metrics['figures'] = [str(p.relative_to(out)) for p in figures]

    # 6) Interactive HTML explorers (self-contained; static PNG fallback).
    if interactive:
        metrics['interactive'] = _write_interactive(word_emb, word_meta, out, emergence)
        metrics['neuron_atlas'] = _write_neuron_atlas(neurons, out)
        metrics['scoreboard_html'] = _write_scoreboard_html(
            metrics.get('scoreboard'), out, run_name
        )

    # 6b) The per-dimension arrays are large, so only the compact summary goes in metrics.json.
    (out / 'neurons.json').write_text(
        json.dumps(neurons, indent=2, default=float), encoding='utf-8'
    )

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
    # `linear_scores` is dropped from the flat CSV so the table keeps one scalar per cell.
    pd.DataFrame([{k: v for k, v in r.items() if k != 'linear_scores'} for r in comparison]).to_csv(
        out / 'comparison.csv', index=False
    )
    if breakdown_words:
        pd.DataFrame(breakdown_words).to_csv(out / 'breakdown.csv', index=False)
    if region_rows:
        pd.DataFrame(region_rows).to_csv(out / 'region_importance.csv', index=False)
    (out / 'report.md').write_text(
        _render_report(metrics, comparison, figures, out), encoding='utf-8'
    )
    _LOG.info('Evaluation written to %s (%d figures)', out, len(figures))
    return metrics


def _honesty_block(
    word_emb: np.ndarray,
    word_meta: pd.DataFrame,
    sent_emb: np.ndarray,
    sent_content_ids: np.ndarray,
    config: Any | None,
) -> dict[str, Any]:
    """Computes the permutation / held-out-decode / calibration add-ons; each degrades rather than aborting."""
    from zte.evaluation.audit.honesty import (
        anchor_calibration_lift,
        cross_subject_decode,
        retrieval_permutation_test,
    )

    if 'subject' not in word_meta.columns or word_meta['subject'].nunique() < 2:
        return {
            'applicable': False,
            'reason': 'need >= 2 subjects for cross-subject honesty checks',
        }
    holdout = None
    if config is not None:
        holdout = getattr(getattr(config, 'train', None), 'loso_holdout_subject', None)
    block: dict[str, Any] = {'applicable': True, 'loso_holdout': holdout}
    try:
        block['retrieval_permutation'] = retrieval_permutation_test(sent_emb, sent_content_ids)
    except Exception as exc:  # noqa: BLE001 — honesty add-ons never abort the run.
        block['retrieval_permutation'] = {'applicable': False, 'reason': f'{type(exc).__name__}'}
    try:
        block['cross_subject_decode'] = cross_subject_decode(word_emb, word_meta)
    except Exception as exc:  # noqa: BLE001
        block['cross_subject_decode'] = {'applicable': False, 'reason': f'{type(exc).__name__}'}
    try:
        block['calibration'] = anchor_calibration_lift(word_emb, word_meta, holdout=holdout)
    except Exception as exc:  # noqa: BLE001
        block['calibration'] = {'applicable': False, 'reason': f'{type(exc).__name__}'}
    return block


def _load_region_map(config: Any | None) -> Any | None:
    """Loads an exact `RegionMap` from `config.dataset.montage_csv` when available.

    Args:
        config (Any | None): The run config; only `dataset.montage_csv` is consulted.

    Returns:
        Any | None: An exact `RegionMap`, or `None` to signal the approximate coordinate-free
            fallback (which also softens the region-importance wording in the report).
    """
    montage = getattr(getattr(config, 'dataset', None), 'montage_csv', None)
    if not montage or not Path(montage).is_file():
        return None
    from zte.data.montage.regions import RegionMap

    try:
        region_map = RegionMap.from_csv(montage)
    except (OSError, ValueError, KeyError) as exc:  # pragma: no cover - defensive
        _LOG.warning('Could not load montage %s: %r; using approximate regions.', montage, exc)
        return None
    _LOG.info('Loaded exact scalp montage from %s (%d regions).', montage, region_map.n_regions)
    return region_map


def _region_importance(
    word_band_power: np.ndarray | None,
    word_meta: pd.DataFrame,
    region_map: Any | None = None,
) -> list[dict[str, Any]]:
    """Scalp-region importance for reading vs cognitive targets (empty if no band power).

    Args:
        word_band_power (np.ndarray | None): Per-word band power `(n_words, n_bands, n_channels)`.
        word_meta (pd.DataFrame): Aligned word metadata.
        region_map (Any | None): Exact montage-derived `RegionMap`; when `None` the approximate coordinate-free default is used inside `region_importance`.

    Returns:
        list[dict[str, Any]]: Tidy region-importance rows (empty when no band power)
    """
    if word_band_power is None:
        return []
    from zte.data.montage.regions import region_importance

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
    return region_importance(word_band_power, targets, region_map=region_map)  # type: ignore[arg-type]


def _write_interactive(
    word_emb: np.ndarray,
    word_meta: pd.DataFrame,
    out: Path,
    emergence: dict[str, Any] | None = None,
) -> str | None:
    """Writes the interactive explorers, returning the flagship explorer path relative to `out`.

    Emits the classic PCA `word_explorer.html` and the richer `thought_space_explorer.html`; passing
    `emergence` lets the latter quote the authoritative full-space clustering numbers.
    """
    from zte.evaluation.interactive import embedding_explorer_html, thought_space_explorer_html

    flagship: str | None = None
    try:
        path = thought_space_explorer_html(
            word_emb,
            word_meta,
            out / 'interactive' / 'thought_space_explorer.html',
            emergence=emergence,
        )
        flagship = str(path.relative_to(out))
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:  # pragma: no cover
        _LOG.warning('Thought-space explorer failed: %r', exc)
    try:
        path = embedding_explorer_html(
            word_emb, word_meta, out / 'interactive' / 'word_explorer.html'
        )
        classic = str(path.relative_to(out))
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:  # pragma: no cover
        _LOG.warning('Interactive explorer failed: %r', exc)
        classic = None
    return flagship or classic


def _neuron_report(
    word_emb: np.ndarray,
    word_meta: pd.DataFrame,
    word_band_power: np.ndarray | None,
    region_map: Any | None,
    config: Any | None,
) -> dict[str, Any]:
    """Computes the neuron-level interpretability report, degrading gracefully on failure."""
    from zte.evaluation.neurons import neuron_report

    band_names = None
    if config is not None and word_band_power is not None:
        bands = tuple(config.dataset.bands)
        if word_band_power.ndim == 3 and word_band_power.shape[1] == len(bands):
            band_names = bands
    try:
        return neuron_report(
            word_emb,
            word_meta,
            band_power=word_band_power,
            band_names=band_names,
            region_map=region_map,
        )
    except (ValueError, KeyError, np.linalg.LinAlgError) as exc:  # pragma: no cover
        _LOG.warning('Neuron report failed: %r', exc)
        return {'summary': {}, 'top_neurons': []}


def _emergence_section(emergence: dict[str, Any]) -> list[str]:
    """Markdown for the emergent-property (cross-subject clustering) metrics."""
    if not emergence:
        return []
    lines = [
        '## Emergent properties -- do similar thoughts cluster across people?',
        '',
        'The north-star property: the *same or related meaning read by different subjects* should sit '
        'together (as in word embeddings). Each number below is a same-pair mean cosine vs a random '
        'baseline; the **gap** is the honest signal (a collapsed/anisotropic space makes all raw '
        'cosines high, so only the gap matters).',
        '',
        f'**{emergence.get("headline", "")}**',
        '',
    ]
    cross = emergence.get('cross_subject', {})
    if cross.get('applicable'):
        lines += [
            '| test | same-pair cosine | random | gap | verdict |',
            '| --- | --- | --- | --- | --- |',
        ]
        for key, label in [
            ('same_word', 'same word, diff subject'),
            ('same_meaning', 'same category, diff subject'),
        ]:
            blk = cross.get(key)
            if blk:
                lines.append(
                    f'| {label} | {blk.get("mean_cosine", float("nan")):.3f} | '
                    f'{blk.get("random_baseline", float("nan")):.3f} | '
                    f'{blk.get("gap", float("nan")):+.3f} | {blk.get("verdict", "n/a")} |'
                )
        lines.append('')
    neigh = emergence.get('neighbourhood', {})
    if neigh.get('applicable'):
        lines += [
            f'- Nearest-neighbour coherence (k={neigh.get("k")}): '
            f'{neigh.get("same_word_purity", float("nan")):.1%} of neighbours are the same word; '
            f'category coherence {neigh.get("category_coherence", float("nan")):+.1%} '
            f'(vs {neigh.get("category_chance", float("nan")):.1%} chance); '
            f'{neigh.get("cross_subject_neighbour_fraction", float("nan")):.1%} of neighbours come from a '
            'different subject.',
            '',
            '_A working thought code wants a positive category coherence **and** a high cross-subject '
            'neighbour fraction: related meanings near each other, regardless of who read them. See the '
            'interactive `thought_space_explorer.html` (auto-analogy leaderboard + neighbourhood view)._',
            '',
        ]
    return lines


def _emergence_report(
    word_emb: np.ndarray, word_meta: pd.DataFrame, analogy: dict[str, Any]
) -> dict[str, Any]:
    """Computes the cross-subject clustering / semantic-coherence metrics, degrading gracefully."""
    from zte.evaluation.emergence import emergence_report

    try:
        return emergence_report(word_emb, word_meta, analogy=analogy)
    except (ValueError, KeyError, np.linalg.LinAlgError) as exc:  # pragma: no cover
        _LOG.warning('Emergence report failed: %r', exc)
        return {}


def _neuron_summary(neurons: dict[str, Any]) -> dict[str, Any]:
    """Builds the compact neuron block embedded in `metrics.json` (full report is in `neurons.json`)."""
    summary = dict(neurons.get('summary', {}))
    summary['top'] = [
        {
            'dim': t['dim'],
            'rank': t['rank'],
            'dominant': t['dominant'],
            'dominant_score': round(float(t['dominant_score']), 3),
            'var_share': round(float(t['var_share']), 4),
            'top_words': [w['word'] for w in t.get('top_words', [])[:3]],
        }
        for t in neurons.get('top_neurons', [])[:10]
    ]
    return summary


def _write_scoreboard_html(board: dict[str, Any] | None, out: Path, run_name: str) -> str | None:
    """Writes the held-out ("new brain") scoreboard dashboard, returning its path relative to `out`.

    Each held-out number is plotted against its named reference line. Degrades to `None` on failure so
    a run never aborts here.
    """
    if not board:
        return None
    try:
        from zte.evaluation.interactive import scoreboard_html
    except ImportError:  # pragma: no cover
        return None
    try:
        path = scoreboard_html(board, out / 'interactive' / 'held_out_scoreboard.html', run_name)
        return str(path.relative_to(out))
    except (ValueError, OSError, KeyError, TypeError) as exc:  # pragma: no cover
        _LOG.warning('Scoreboard dashboard failed: %r', exc)
        return None


def _write_neuron_atlas(neurons: dict[str, Any], out: Path) -> str | None:
    """Writes the interactive Neuron Atlas HTML, returning its path relative to `out`."""
    try:
        from zte.evaluation.interactive import neuron_atlas_html
    except ImportError:  # pragma: no cover - atlas viz optional
        return None
    try:
        path = neuron_atlas_html(neurons, out / 'interactive' / 'neuron_atlas.html')
        return str(path.relative_to(out))
    except (ValueError, OSError, KeyError) as exc:  # pragma: no cover
        _LOG.warning('Neuron atlas failed: %r', exc)
        return None


def _render_sota_figures(
    word_emb_raw: np.ndarray,
    word_emb: np.ndarray,
    word_meta: pd.DataFrame,
    sent_ranks: list[float],
    sent_ret: dict[str, float],
    n_sent: int,
    neurons: dict[str, Any],
    word_band_power: np.ndarray | None,
    montage_csv: str | None,
    fig_dir: Path,
) -> list[Path]:
    """Renders the geometry, rank-distribution, variance-budget, neuron and scalp figures; skips any that fail."""
    import matplotlib.pyplot as plt

    written: list[Path] = []

    def _save(fig: Any, name: str) -> None:
        path = fig_dir / name
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        written.append(path)

    word_ids = _encode(word_meta['word'].to_numpy()) if 'word' in word_meta.columns else None
    builders: list[tuple[str, Any]] = []
    if word_ids is not None:
        builders.append(
            (
                'geometry_before_after.png',
                lambda: P.geometry_before_after(word_emb_raw, word_emb, word_ids),
            )
        )
    if sent_ranks:
        builders.append(
            (
                'retrieval_rank_distribution.png',
                lambda: P.retrieval_rank_distribution(
                    np.asarray(sent_ranks), n_sent, float(sent_ret.get('chance_top1', 0.0) or 0.0)
                ),
            )
        )
    summary = neurons.get('summary', {})
    if summary.get('variance_budget'):
        builders.append(('variance_budget_pie.png', lambda: P.variance_budget_pie(summary)))
    top_neurons = neurons.get('top_neurons', [])
    if top_neurons:
        builders.append(
            ('neuron_selectivity.png', lambda: P.neuron_selectivity_heatmap(top_neurons[:12]))
        )
    if 'subject' in word_meta.columns and word_meta['subject'].nunique() > 1:
        builders.append(
            (
                'subject_similarity.png',
                lambda: P.subject_similarity_heatmap(word_emb, word_meta['subject'].to_numpy()),
            )
        )
    scalp = _scalp_topomap_data(word_band_power, word_meta, montage_csv)
    if scalp is not None:
        vals, coords = scalp
        builders.append(
            (
                'scalp_topomap.png',
                lambda: P.scalp_topomap(vals, coords, 'Scalp map: lexical-frequency importance'),
            )
        )
    for name, builder in builders:
        try:
            _save(builder(), name)
        except (ValueError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
            _LOG.warning('Skipped SOTA figure %s: %r', name, exc)
    return written


def _scalp_topomap_data(
    word_band_power: np.ndarray | None, word_meta: pd.DataFrame, montage_csv: str | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-channel lexical-frequency importance + 2-D electrode coordinates for the scalp topomap.

    Importance is the per-channel absolute correlation between mean band power and log word frequency;
    coordinates come from the montage geometry, exact only when a montage CSV was supplied.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: `(per_channel_importance (C,), coords_2d (C, 2))`, or
            `None` when band power / a frequency column is unavailable.
    """
    freq_col = 'corpus_log_freq' if 'corpus_log_freq' in word_meta else 'log_freq'
    if word_band_power is None or word_band_power.ndim != 3 or freq_col not in word_meta.columns:
        return None
    freq = pd.to_numeric(word_meta[freq_col], errors='coerce').to_numpy(dtype=float)
    finite = np.isfinite(freq)
    if int(finite.sum()) < 8:
        return None
    from zte.models.spatial import resolve_geometry

    bp = np.asarray(word_band_power, dtype=np.float64)  # (n, n_bands, n_channels)
    n_channels = bp.shape[2]
    # mean band power per (word, channel) over bands, then |corr with log_freq| per channel.
    chan_power = np.nanmean(bp, axis=1)[finite]  # (n_finite, n_channels)
    f = freq[finite]
    imp = np.zeros(n_channels, dtype=np.float64)
    for c in range(n_channels):
        col = chan_power[:, c]
        ok = np.isfinite(col)
        if int(ok.sum()) >= 8 and np.std(col[ok]) > 1e-9:
            imp[c] = abs(float(np.corrcoef(col[ok], f[ok])[0, 1]))
    geo = resolve_geometry(n_channels, montage_csv)
    return np.nan_to_num(imp), geo.coords_2d


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
    """Builds a kNN leave-one-out predicted-vs-true scatter for log frequency."""
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
    *,
    sent_top1_hits: list[float] | None = None,
    subj_top1_hits: list[float] | None = None,
    subj_chances: list[float] | None = None,
    effect_floor: float = 0.01,
    seed: int = 0,
) -> dict[str, Any]:
    """Derives headline checks backed by bootstrap effect-size confidence intervals.

    Every check must clear a bootstrap 95% CI lower bound: `effect_floor` for the per-fold
    ZTE-minus-noise probe gap, and 0 for a Top-1 hit rate minus its random-chance rate.

    Args:
        comparison (list[dict[str, Any]]): Probe comparison rows (with `linear_scores`).
        health (dict[str, float]): Geometry/health metrics.
        sent_ret (dict[str, float]): Sentence retrieval metrics.
        analogy (dict[str, Any] | None): Vector-arithmetic transfer report.
        sent_top1_hits (list[float] | None): Per-query sentence-retrieval Top-1 hits.
        subj_top1_hits (list[float] | None): Per-query subject-arithmetic Top-1 hits.
        subj_chances (list[float] | None): Per-query subject-arithmetic chance rates.
        effect_floor (float): Minimum ZTE-minus-noise effect the CI must clear.
        seed (int): Bootstrap seed.

    Returns:
        dict[str, Any]: Named boolean checks, the CIs behind them (`beats_noise_ci`,
            `retrieval_ci`, `subject_arithmetic_ci`) and the `effect_size_floor` used.
    """
    zte = {r['target']: r for r in comparison if r['representation'] == 'ZTE'}
    noise = {r['target']: r for r in comparison if r['representation'] == 'noise (matched)'}

    # Per-target probe gap over the noise control, paired fold by fold.
    beats_noise: list[str] = []
    beats_noise_ci: dict[str, list[float]] = {}
    for t, z_row in zte.items():
        z_scores = np.asarray(z_row.get('linear_scores', []), dtype=np.float64)
        n_scores = np.asarray(noise.get(t, {}).get('linear_scores', []), dtype=np.float64)
        if z_scores.size and z_scores.size == n_scores.size:
            point, lo, hi = M.bootstrap_ci(z_scores - n_scores, seed=seed)
        else:  # No per-fold scores (tiny target) -> fall back to the point difference.
            point = float(z_row['linear_score'] - noise.get(t, {}).get('linear_score', 0.0))
            lo = hi = point
        beats_noise_ci[t] = [round(point, 4), round(lo, 4), round(hi, 4)]
        if lo > effect_floor:
            beats_noise.append(t)

    # Retrieval lift over its query-weighted chance rate.
    ret_chance = float(sent_ret.get('chance_top1', float('nan')))
    ret_point, ret_lo, ret_hi = _diff_ci(sent_top1_hits, ret_chance, seed=seed)
    retrieval_pass = bool(np.isfinite(ret_lo) and ret_lo > 0.0)

    verdict: dict[str, Any] = {
        'beats_noise_on': beats_noise,
        'beats_noise_all_targets': len(beats_noise) == len(zte) and len(zte) > 0,
        'beats_noise_ci': beats_noise_ci,
        'effect_size_floor': effect_floor,
        'no_collapse': bool(
            health.get('effective_rank_ratio', 0) > 0.1 and health.get('dead_dim_fraction', 1) < 0.5
        ),
        'retrieval_above_chance': retrieval_pass,
        'retrieval_ci': [round(ret_point, 4), round(ret_lo, 4), round(ret_hi, 4)],
    }
    if analogy:
        st = analogy.get('subject_transfer', {})
        subj_chance = (
            float(np.mean(subj_chances))
            if subj_chances
            else float(st.get('chance_top1', float('nan')))
        )
        subj_point, subj_lo, subj_hi = _diff_ci(subj_top1_hits, subj_chance, seed=seed)
        verdict['subject_arithmetic_above_chance'] = bool(np.isfinite(subj_lo) and subj_lo > 0.0)
        verdict['subject_arithmetic_ci'] = [
            round(subj_point, 4),
            round(subj_lo, 4),
            round(subj_hi, 4),
        ]
    return verdict


def _diff_ci(hits: list[float] | None, chance: float, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI of `mean(hits) - chance` (all `nan` when hits/chance are absent)."""
    if not hits or not np.isfinite(chance):
        return (float('nan'), float('nan'), float('nan'))
    point, lo, hi = M.bootstrap_ci(np.asarray(hits, dtype=np.float64), seed=seed)
    return (point - chance, lo - chance, hi - chance)


def _fmt_ci(ci: list[float] | None) -> str:
    """Formats a `[point, lo, hi]` CI triple as `point [lo, hi]` (or `n/a`)."""
    if not ci or not np.isfinite(ci[0]):
        return 'n/a'
    return f'{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]'


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
    ]
    if metrics.get('scoreboard'):
        from zte.evaluation.audit.scoreboard import render_markdown as _render_scoreboard

        lines.append(_render_scoreboard(metrics['scoreboard']))
        lines.append('')
    lines += [
        '## Verdict',
        '',
        'Checks are backed by bootstrap 95% confidence intervals (CI), not sign-only '
        f'comparisons: *beats noise* needs the ZTE-minus-noise probe gap CI lower bound '
        f'above the effect floor ({verdict.get("effect_size_floor", 0.01):.2g}); '
        '*above chance* needs the (Top-1 - chance) CI lower bound above 0.',
        '',
        f'- Beats noise control on: **{", ".join(verdict["beats_noise_on"]) or "none"}** '
        f'(all targets: {verdict["beats_noise_all_targets"]})',
        f'- No representation collapse: **{verdict["no_collapse"]}** '
        f'(effective-rank ratio {health["effective_rank_ratio"]:.2f}, '
        f'dead dims {health["dead_dim_fraction"]:.0%})',
        f'- Cross-subject retrieval above chance: **{verdict["retrieval_above_chance"]}** '
        f'(Top-1 {sent.get("top1", float("nan")):.3f} vs query-weighted chance '
        f'{sent.get("chance_top1", float("nan")):.3f}; '
        f'lift CI {_fmt_ci(verdict.get("retrieval_ci"))}'
        + (
            f'; permutation p={verdict["retrieval_permutation_p"]:.3f}'
            if 'retrieval_permutation_p' in verdict
            else ''
        )
        + (
            f'; rank-percentile {sent["rank_percentile"]:.3f}, median rank {sent["median_rank"]:.0f}'
            if 'rank_percentile' in sent
            else ''
        )
        + '). The headline requires BOTH the CI lift and the permutation null; '
        'rank-percentile (1.0 = correct match ranked first) shows the whole distribution, not just the tail.',
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
        f'query-weighted chance {sent.get("chance_top1", float("nan")):.3f}',
        f'- Word (same token): Top-1 {metrics["word_retrieval"].get("top1", float("nan")):.3f}, '
        f'query-weighted chance {metrics["word_retrieval"].get("chance_top1", float("nan")):.3f}',
    ]
    fm = metrics.get('word_retrieval_freq_matched')
    if fm:
        lines.append(
            f'- Word, frequency-matched distractors: Top-1 {fm.get("top1", float("nan")):.3f} '
            f'vs matched chance {fm.get("chance_top1", float("nan")):.3f} over {int(fm.get("n_bins", 0))} bins '
            '-- a hit here cannot be a lexical-frequency shortcut.'
        )
    nov = metrics.get('word_retrieval_by_novelty') or {}
    if nov:
        for label in ('seen', 'novel'):
            blk = nov.get(label)
            if blk:
                lines.append(
                    f'- Word, {label} types: Top-1 {blk.get("top1", float("nan")):.3f} '
                    f'vs chance {blk.get("chance_top1", float("nan")):.3f} '
                    f'({int(blk.get("n_queries", 0))} queries)'
                )
        lines.append(
            '  _"novel" = a word type absent from the training split. In a LOSO run the held-out '
            'subject reads the same stimuli, so most types are seen and the novel bucket is small._'
        )
    lines += [
        '',
        '_Chance is query-weighted (matches the per-occurrence hit rate); the legacy '
        'type-weighted value is kept as `chance_top1_typeweighted` in `metrics.json`._',
        '',
    ]
    lines += _extended_report_sections(metrics)
    lines += ['## Figures', '']
    lines += [f'![{p.stem}]({p.relative_to(out).as_posix()})' for p in figures]
    lines.append('')
    return '\n'.join(lines)


def _honesty_section(honesty: dict[str, Any]) -> list[str]:
    """Markdown for the permutation null, held-out cross-subject decode and calibration lift."""
    if not honesty or not honesty.get('applicable'):
        return []
    lines = ['## Honesty checks (permutation null · held-out decode · calibration)', '']

    perm = honesty.get('retrieval_permutation', {})
    if perm.get('applicable'):
        verdict = 'above chance' if perm['above_chance'] else 'not above chance'
        lines += [
            f'- **Permutation null (cross-subject retrieval):** observed Top-1 '
            f'{perm["observed_top1"]:.3f} vs a label-shuffled null of '
            f'{perm["null_mean"]:.3f}±{perm["null_std"]:.3f} over {perm["n_perm"]} permutations '
            f'-> p = **{perm["p_value"]:.3f}** ({verdict}). An empirical null, not an analytic one.',
        ]

    xd = honesty.get('cross_subject_decode', {})
    if xd.get('applicable') and xd.get('targets'):
        lines += [
            '',
            f'- **Held-out cross-subject decoding** (train on {xd["n_subjects"] - 1} subjects, '
            'test on the held-out one, one fold per subject) — the honest generalization test:',
            '',
            '| target | task | held-out score | chance | beats chance |',
            '| --- | --- | --- | --- | --- |',
        ]
        for name, t in xd['targets'].items():
            lo = t['ci'][1]
            lines.append(
                f'| {name} | {t["task"]} | {t["mean"]:.3f} (CI lo {lo:.3f}) | '
                f'{t["chance"]:.3f} | {"✓" if t["above_chance"] else "·"} |'
            )

    cal = honesty.get('calibration', {})
    if cal.get('applicable'):
        who = cal.get('holdout') or 'each subject in turn'
        lines += [
            '',
            f'- **Anchor calibration** (align {who} into the shared frame from '
            f'{cal["n_anchors"]} shared anchor words, then measure same-word cross-subject '
            f'cohesion on held-out words): before **{cal["mean_cohesion_before"]:.3f}** -> after '
            f'**{cal["mean_cohesion_after"]:.3f}** (lift {cal["mean_lift"]:+.3f}, '
            f'{"helps" if cal["helps"] else "no help"}). A metrics-side preview of whether a new '
            'brain can be snapped into the space without retraining.',
        ]
    lines.append('')
    return lines


def _extended_report_sections(metrics: dict[str, Any]) -> list[str]:
    """Markdown for the arithmetic, breakdown, per-category and region analyses."""
    lines: list[str] = []
    lines += _emergence_section(metrics.get('emergence', {}))
    lines += _honesty_section(metrics.get('honesty', {}))
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
            f'{int(st.get("n_queries", 0))} analogies; '
            f'lift CI {_fmt_ci(metrics.get("verdict", {}).get("subject_arithmetic_ci"))})',
        ]
        raw = analogy.get('subject_transfer_raw', {})
        if raw:
            lines.append(
                f'- Raw-feature control: Top-1 {raw.get("top1", float("nan")):.3f} '
                '(ZTE should beat it -- arithmetic is a property of the learned space)'
            )
        tt = analogy.get('task_transfer', {})
        if tt and tt.get('reason') == 'disjoint_stimuli':
            lines.append(
                '- Task transfer: **not applicable** -- the tasks read disjoint stimuli '
                '(no stimulus token is shared across tasks), so the cross-task '
                'arithmetic is undefined rather than failed.'
            )
        elif tt:
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
        approximate = metrics.get('region_map_approximate', True)
        if approximate:
            lines += [
                '## Scalp-region importance (approximate region proxy, no montage)',
                '',
                '_No electrode montage was supplied, so channels are grouped by an '
                '**approximate** coordinate-free anterior->posterior proxy. Region '
                'labels are indicative only; supply `dataset.montage_csv` for exact '
                'per-channel regions._',
                '',
            ]
        else:
            lines += ['## Scalp-region importance (exact montage)', '']
        lines.append('| region | ' + ' | '.join(str(c) for c in frame.columns) + ' |')
        lines.append('| --- |' + ' --- |' * len(frame.columns))
        for region_name, row in frame.iterrows():
            lines.append(
                f'| {region_name} | ' + ' | '.join(f'{v:.2f}' for v in row.to_numpy()) + ' |'
            )
        lines.append('')

    neurons = metrics.get('neurons', {})
    if neurons:
        budget = neurons.get('variance_budget', {})
        who = neurons.get('who_variance', 0.0)
        what = neurons.get('what_variance', 0.0)
        ratio = neurons.get('who_vs_what_ratio', float('nan'))
        lines += [
            '## Neurons -- what the dimensions encode',
            '',
            f'Of {neurons.get("embed_dim", 0)} dimensions, **{neurons.get("n_active", 0)} are active** '
            f'and {neurons.get("n_dead", 0)} are dead (near-constant). Each active neuron is scored for '
            'what it tracks (variance explained: r^2 for word length / log-frequency, eta^2 for '
            'subject / task / category); '
            'its *dominant* attribute is the argmax. The **variance budget** below is the share of the '
            "space's total variance whose dominant attribute is each target -- i.e. how much of the "
            'representation is spent encoding *who* vs *what*.',
            '',
            '| dominant attribute | variance share |',
            '| --- | --- |',
        ]
        for attr, share in sorted(budget.items(), key=lambda kv: -kv[1]):
            lines.append(f'| {attr} | {share:.1%} |')
        lines += [
            '',
            f'**Who (subject) vs what (content): {who:.1%} vs {what:.1%}** '
            f'(ratio {ratio:.2f}; > 1 means the space encodes identity more than content -- the ZTE v1 '
            'failure mode). See `neurons.json` and the interactive `neuron_atlas.html` for per-neuron detail.',
            '',
        ]
        top = neurons.get('top', [])
        if top:
            lines += [
                '| neuron | dominant | score | var share | top-firing words |',
                '| --- | --- | --- | --- | --- |',
            ]
            lines += [
                f'| #{t["dim"]} (rank {t["rank"]}) | {t["dominant"]} | {t["dominant_score"]} | '
                f'{t["var_share"]:.3f} | {", ".join(t["top_words"])} |'
                for t in top
            ]
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
