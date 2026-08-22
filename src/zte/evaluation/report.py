"""Orchestrates the ZTE evaluation: embeddings in, `metrics.json` + tables + figures + `report.md` out."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from zte.evaluation import metrics as M
from zte.evaluation import plots as P
from zte.evaluation.analogy import analogy_report
from zte.evaluation.breakdown import stratified_report, stratified_retrieval
from zte.inference.retrieval import NearestNeighborIndex
from zte.logging_utils import get_logger
from zte.models.encoder.nuisance import LengthProjector, length_leakage
from zte.training.metrics import noise_matched

_LOG = get_logger('evaluation.report')

# The one (strategy, cell) pair whose held-out readings generalise over the subject AND the stimulus.
HONEST_SPLIT: tuple[str, str] = ('by_subject_and_stimulus', 'test')

# Stands in for the control stack of a block that never recorded which controls it pre-registered.
UNRECORDED_CONTROLS: str = '<no controls_requested ledger>'

# Evaluation costs roughly twice what training does, in one call with a single write at the very end, so a VM
# reclaimed at minute 60 of it repeats all 60 minutes unless each block records itself as it lands.
PARTIAL_FILE: Final[str] = '_partial.json'
"""Per-block progress file written beside `metrics.json` and deleted once that file lands."""

# Named once, because a sweep-profile `metrics.json` whose absent blocks are not declared reads as a full one.
SWEEP_SKIPPED: Final[tuple[str, ...]] = (
    'analogy',
    'neurons',
    'emergence',
    'word_retrieval_by_novelty',
    'word_retrieval_freq_matched',
    'figures',
    'interactive',
)
"""Metrics blocks the `sweep` profile does not compute."""


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


def _eval_profile(config: Any | None) -> str:
    """Reads `train.eval_profile`, falling back to the full suite for a config written before the knob existed."""
    profile = getattr(getattr(config, 'train', None), 'eval_profile', None) or 'full'
    if profile not in {'full', 'sweep'}:
        _LOG.warning('Unknown train.eval_profile %r -- running the full evaluation suite.', profile)
        return 'full'

    return str(profile)


def _jsonable(value: Any) -> Any:
    """Coerces numpy scalars and arrays into JSON types, raising on anything a block cannot round-trip."""
    match value:
        case np.generic():
            return value.item()
        case np.ndarray():
            return value.tolist()
        case None | bool() | int() | float() | str():
            return value
        case dict():
            # A non-string key would come back as a string and quietly change the block's shape.
            if any(not isinstance(k, str) for k in value):
                raise TypeError('block keys must be strings')
            return {k: _jsonable(v) for k, v in value.items()}
        case list() | tuple():
            return [_jsonable(v) for v in value]
        case Path():
            return str(value)
        case _:
            raise TypeError(f'{type(value).__name__} is not JSON-serialisable')


def _eval_fingerprint(
    word_emb: np.ndarray,
    raw_word_feats: np.ndarray,
    sent_emb: np.ndarray,
    sent_content_ids: np.ndarray,
    decoder_blocks: tuple[dict[str, Any] | None, ...],
) -> str:
    """Digest of everything a recorded block was computed from, so a re-entry can never reuse a stale number."""
    from zte.evaluation.audit.scoreboard import embedding_checksum

    digest = hashlib.sha256()
    for array in (word_emb, raw_word_feats, sent_emb, np.asarray(sent_content_ids, dtype=np.float64)):
        digest.update(embedding_checksum(array).encode())
    digest.update(json.dumps(decoder_blocks, sort_keys=True, default=str).encode())

    return digest.hexdigest()[:16]


class _EvalStages:
    """Block-level progress for one evaluation directory, so a reclaim costs the block in flight and not the hour."""

    def __init__(self, path: Path, fingerprint: str, profile: str, mirror: Path | None = None) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.profile = profile
        # Without a mirror the file dies with the VM it protects, which on Colab is the only failure it was
        # built for -- the run directory is mirrored at stage boundaries, and evaluation is one whole stage.
        self.mirror = mirror
        self.blocks: dict[str, Any] = self._read()

    def _read(self) -> dict[str, Any]:
        """Recorded blocks from a previous entry; empty when the file is absent, torn or from other embeddings."""
        source = self.path if self.path.is_file() else self.mirror
        if source is None or not source.is_file():
            return {}

        try:
            stored = json.loads(source.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            _LOG.warning('Partial evaluation %s unreadable (%r); every block is recomputed.', self.path.name, exc)
            return {}

        if not isinstance(stored, dict) or stored.get('fingerprint') != self.fingerprint:
            _LOG.info('Partial evaluation %s measures other embeddings; every block is recomputed.', self.path.name)
            return {}

        blocks = dict(stored.get('blocks') or {})
        _LOG.info('Partial evaluation %s carries %d completed block(s).', self.path.name, len(blocks))

        return blocks

    def run[T](self, name: str, compute: Callable[[], T]) -> T:
        """Returns the block recorded by an earlier entry, else computes it and records it before returning."""
        if name in self.blocks:
            _LOG.info('Evaluation block %r read from %s.', name, self.path.name)
            # A copy: callers pop the per-query vectors out of a block, and must not empty what was recorded.
            return _jsonable(self.blocks[name])

        value = compute()
        self._record(name, value)

        return value

    def _record(self, name: str, value: Any) -> None:
        """Writes the block out atomically; a value JSON cannot carry is returned but never checkpointed."""
        try:
            self.blocks[name] = _jsonable(value)
        except TypeError as exc:
            _LOG.debug('Evaluation block %r is not checkpointable: %r', name, exc)
            return

        payload = {'fingerprint': self.fingerprint, 'eval_profile': self.profile, 'blocks': self.blocks}
        tmp = self.path.with_name(f'{self.path.name}.tmp')
        try:
            tmp.write_text(json.dumps(payload), encoding='utf-8')
            tmp.replace(self.path)
        except (OSError, ValueError) as exc:  # pragma: no cover - defensive
            _LOG.warning('Could not record evaluation block %r: %r', name, exc)
            return

        self._to_drive()

    def _to_drive(self) -> None:
        """Copies the partial file to its mirror, so a reclaimed VM resumes rather than repeating the hour."""
        if self.mirror is None:
            return

        try:
            self.mirror.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.path, self.mirror)
        except OSError as exc:  # pragma: no cover -- an unmounted Drive is a normal off-Colab state
            _LOG.debug('Could not mirror %s: %r', self.path.name, exc)

    def clear(self) -> None:
        """Deletes the partial file, which has nothing left to protect once `metrics.json` is on disk."""
        self.blocks.clear()
        self.path.unlink(missing_ok=True)
        if self.mirror is not None:
            self.mirror.unlink(missing_ok=True)


def _relative(paths: list[Path], out: Path) -> list[str]:
    """Figure paths relative to the evaluation directory, which is how `metrics.json` names them."""
    return [str(p.relative_to(out)) for p in paths]


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
    *,
    phase_sent_emb: np.ndarray | None = None,
    train_sent_emb: np.ndarray | None = None,
    train_sent_n_words: np.ndarray | None = None,
    sent_n_words: np.ndarray | None = None,
    generation: dict[str, Any] | None = None,
    rescoring: dict[str, Any] | None = None,
    decoder_capacity: dict[str, Any] | None = None,
    min_prefix_kl: float = 0.05,
    partial_mirror: str | Path | None = None,
) -> dict[str, Any]:
    """Runs the full evaluation and writes metrics, tables, figures and a report.

    Beyond the global probes/retrieval/health this adds per-subject / per-task / per-category
    breakdowns, vector-arithmetic transfer analogies and (given band power) scalp-region importance,
    then emits the interactive HTML explorers and a TensorBoard log.

    Note:
        `config.train.eval_profile` decides how much of that runs: `sweep` keeps only the blocks a headline may
        be read from and stamps `eval_profile` plus `eval_skipped` into the metrics. `partial_mirror` names a
        durable directory the block-progress file is copied to as it grows, because evaluation is two thirds of a
        run and the run directory is only mirrored once evaluation has already returned. Every block records itself
        into `_partial.json` as it lands, so re-entering after a reclaimed machine resumes at the interrupted
        block instead of repeating the hour before it; the file is deleted once `metrics.json` is written, and
        deleting it by hand forces a full recompute.

    Args:
        word_emb (np.ndarray): Word-level ZTE embeddings `(n_words, embed_dim)`.
        word_meta (pd.DataFrame): Aligned word metadata (word/word_len/log_freq/subject/task/category/
            sentence_idx/word_idx), length `n_words`.
        raw_word_feats (np.ndarray): Aligned raw band-power features `(n_words, n_features)` for the baseline
            comparison.
        sent_emb (np.ndarray): Sentence-level embeddings `(n_sentences, embed_dim)`.
        sent_content_ids (np.ndarray): Content/group id per sentence `(n_sentences,)` (same stimulus across subjects
            shares an id).
        out_dir (str | Path): Output directory for artifacts.
        run_name (str): Identifier used in the report header.
        sent_meta (pd.DataFrame | None): Aligned sentence metadata (with `category`) enabling per-category retrieval
            breakdowns and projector colouring.
        word_band_power (np.ndarray | None): Aligned per-word band power
            `(n_words, n_bands, n_channels)` for scalp-region importance (skipped when `None`).
        config (Any | None): The run `ZTEConfig` for HParams logging.
        tensorboard (bool | str): `True` (write under `out/tb/run_name`), a path string, or `False` to disable
            TensorBoard logging.
        interactive (bool): Whether to write the interactive HTML explorer.
        phase_word_emb (np.ndarray | None): Embeddings of phase-scrambled EEG, added as a control representation.
        train_vocab (set[str] | None): Word types seen in training, enabling the seen-vs-novel retrieval split.

    Keyword Args:
        phase_sent_emb (np.ndarray | None): Sentence embeddings of phase-scrambled EEG through the identical
            encoder, routed into retrieval, the scoreboard and the verdict rather than the probe table alone.
        train_sent_emb (np.ndarray | None): Training-split sentence embeddings. When given, whitening and
            all-but-the-top are fitted on these rows and both the train-fitted and the transductive
            retrieval numbers are reported, since only the former is reproducible one sentence at a time.
        train_sent_n_words (np.ndarray | None): Word count of each training-split sentence, which is what the
            `objective.length_projection` regression is fitted against.
        sent_n_words (np.ndarray | None): Word count per sentence, enabling the length-stratified gallery.
        generation (dict[str, Any] | None): A `generation.generation_report` block from a decoder run.
        rescoring (dict[str, Any] | None): A `scoreboard.decoder_rescoring_retrieval` block.
        decoder_capacity (dict[str, Any] | None): An `audit.capacity.capacity_report` block from a decoder run.
            It is kept under `decoder_capacity`, apart from the encoder-side cosine `menu` audit and from
            generation, because it certifies a menu-selection readout and nothing else.
        min_prefix_kl (float): Minimum prefix-influence KL (nats) the generation verdict requires.

    Returns:
        dict[str, Any]: The full metrics dictionary (also written to `metrics.json`).
    """
    out = Path(out_dir)
    fig_dir = out / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    profile = _eval_profile(config)
    sweep = profile == 'sweep'

    # 0) Label-free geometry post-processing; order matters: whiten, THEN all-but-the-top.
    obj_cfg = getattr(config, 'objective', None)
    # A raw snapshot, so report.md can show the geometry before vs after.
    word_emb_raw = np.asarray(word_emb, dtype=np.float32).copy()
    do_whiten = obj_cfg is not None and bool(getattr(obj_cfg, 'whiten', False))
    n_top = int(getattr(obj_cfg, 'all_but_top', 0) or 0) if obj_cfg is not None else 0
    sent_emb_transductive: np.ndarray | None = None
    # A basis fitted in one coordinate frame and subtracted in another adds an arbitrary vector field instead of
    # removing the nuisance, so each scored array carries the training rows through its own transform.
    train_sent_emb_fitted: np.ndarray | None = None
    train_sent_emb_transductive: np.ndarray | None = None
    if do_whiten or n_top > 0:
        # Fitting on the scored rows is transductive, and a decoder scoring one sentence cannot reproduce it.
        if train_sent_emb is None:
            sent_emb_transductive = _postprocess(sent_emb, None, do_whiten, n_top)
            sent_emb = sent_emb_transductive
        else:
            sent_emb_transductive, train_sent_emb_transductive = _postprocess_transductive(
                sent_emb, train_sent_emb, do_whiten, n_top
            )
            train_sent_emb_fitted = _postprocess(train_sent_emb, train_sent_emb, do_whiten, n_top)
            sent_emb = _postprocess(sent_emb, train_sent_emb, do_whiten, n_top)
        word_emb = _postprocess(word_emb, None, do_whiten, n_top)
        _LOG.info(
            'Post-processed embeddings (whiten=%s, all_but_top=%d, fit=%s).',
            do_whiten,
            n_top,
            'transductive' if train_sent_emb is None else 'train split',
        )
    # Length is 5.14 of the 9.45 bits needed to name a ZuCo sentence, so removing it says what is left underneath.
    length_projection: dict[str, Any] | None = None
    if obj_cfg is not None and bool(getattr(obj_cfg, 'length_projection', False)):
        fit_rows = train_sent_emb if train_sent_emb_fitted is None else train_sent_emb_fitted
        projector, length_projection = _fit_length_projector(sent_n_words, fit_rows, train_sent_n_words)
        if projector is not None and sent_n_words is not None:
            n_words = np.asarray(sent_n_words)
            length_projection['length_leakage_before'] = length_leakage(sent_emb, n_words)
            sent_emb = projector.transform(sent_emb, n_words)
            length_projection['length_leakage_after'] = length_leakage(sent_emb, n_words)
            transductive_projector, _ = _fit_length_projector(
                sent_n_words, train_sent_emb_transductive, train_sent_n_words
            )
            if sent_emb_transductive is not None and transductive_projector is not None:
                sent_emb_transductive = transductive_projector.transform(sent_emb_transductive, n_words)
            _LOG.info(
                'Length projection: word count explained %.4f of sentence-embedding variance before, %.4f after.',
                length_projection['length_leakage_before'],
                length_projection['length_leakage_after'],
            )

    csls_k = int(getattr(obj_cfg, 'csls_neighbors', 0) or 0) if obj_cfg is not None else 0
    use_csls = csls_k > 0
    if use_csls:
        _LOG.info(
            'Retrieval uses CSLS hubness correction (k=%d, config.objective.csls_neighbors).',
            csls_k,
        )

    # Every block below records itself as it lands, keyed by what it was computed from, so re-entering this
    # function after a reclaim resumes at the block that was interrupted.
    stages = _EvalStages(
        out / PARTIAL_FILE,
        _eval_fingerprint(
            word_emb, raw_word_feats, sent_emb, sent_content_ids, (generation, rescoring, decoder_capacity)
        ),
        profile,
        Path(partial_mirror) / PARTIAL_FILE if partial_mirror else None,
    )

    # 1) Transfer probes: ZTE vs raw band-power vs noise-matched control vs phase-shuffled ZTE.
    representations = {
        'ZTE': np.asarray(word_emb, dtype=np.float32),
        'raw band-power': np.asarray(raw_word_feats, dtype=np.float32),
        'noise (matched)': noise_matched(np.asarray(raw_word_feats, dtype=np.float32)),
    }
    if phase_word_emb is not None:
        # The control must get the same post-processing as ZTE or the comparison is rigged.
        phase_word_emb = _postprocess(phase_word_emb, None, do_whiten, n_top)
        representations['phase-shuffled ZTE'] = phase_word_emb
    targets = _word_targets(word_meta)
    comparison = stages.run('probe_comparison', lambda: M.representation_comparison(representations, targets))

    # 2) Geometry / health (with adjacency positives for alignment).
    health = stages.run('embedding_health', lambda: M.embedding_health(word_emb, pairs=_adjacency_pairs(word_meta)))

    # 3) Content retrieval (sentence-level across subjects, and word-level by token).
    sent_ret = stages.run(
        'sentence_retrieval',
        lambda: M.content_retrieval(
            sent_emb,
            np.asarray(sent_content_ids),
            return_hits=True,
            return_ranks=True,
            csls=use_csls,
            csls_k=csls_k,
        ),
    )
    # Popped, not kept: the per-query vectors feed the CI verdict but would bloat metrics.json.
    sent_top1_hits = sent_ret.pop('top1_hits', [])  # type: ignore[arg-type]
    sent_ranks = sent_ret.pop('ranks', [])  # per-query ranks for the rank-distribution figure

    # 3.1) The phase-scrambled control and the transductive counterpart, both through the identical path.
    phase_ret: dict[str, Any] | None = None
    phase_top1_hits: list[float] = []
    phase_sent_post: np.ndarray | None = None
    if phase_sent_emb is not None:
        phase_sent_post = _postprocess(phase_sent_emb, None, do_whiten, n_top)
        phase_ret = stages.run(
            'sentence_retrieval_phase_control',
            lambda: M.content_retrieval(
                phase_sent_post,
                np.asarray(sent_content_ids),
                return_hits=True,
                csls=use_csls,
                csls_k=csls_k,
            ),
        )
        raw_hits: Any = phase_ret.pop('top1_hits', [])
        phase_top1_hits = [float(h) for h in raw_hits] if isinstance(raw_hits, list) else []
    sent_ret_transductive: dict[str, float] | None = None
    if train_sent_emb is not None and sent_emb_transductive is not None:
        sent_ret_transductive = stages.run(
            'sentence_retrieval_transductive',
            lambda: M.content_retrieval(
                sent_emb_transductive, np.asarray(sent_content_ids), csls=use_csls, csls_k=csls_k
            ),
        )
    word_ret = stages.run(
        'word_retrieval',
        lambda: M.content_retrieval(word_emb, _encode(word_meta['word'].to_numpy()), csls=use_csls, csls_k=csls_k),
    )
    eval_seen_novel = not sweep and bool(getattr(obj_cfg, 'eval_seen_novel', False))
    eval_freq_matched = not sweep and bool(getattr(obj_cfg, 'eval_freq_matched', False))
    # 3.2c) Seen vs novel word types: does retrieval hold for types absent from the training split?
    word_ret_by_novelty: dict[str, Any] = {}
    if eval_seen_novel and train_vocab is not None and 'word' in word_meta.columns:
        words_arr = word_meta['word'].astype(str).to_numpy()
        seen_mask = np.array([w in train_vocab for w in words_arr])
        word_codes = _encode(word_meta['word'].to_numpy())
        word_ret_by_novelty = stages.run(
            'word_retrieval_by_novelty',
            lambda: {
                label: M.content_retrieval(word_emb[mask], word_codes[mask], csls=use_csls, csls_k=csls_k)
                for label, mask in (('seen', seen_mask), ('novel', ~seen_mask))
                if int(mask.sum()) >= 4
            },
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
                word_ret_freq_matched = stages.run(
                    'word_retrieval_freq_matched',
                    lambda: M.matched_content_retrieval(
                        word_emb,
                        _encode(word_meta['word'].to_numpy()),
                        fbin,
                        csls=use_csls,
                        csls_k=csls_k,
                    ),
                )

    # 4) Stratified breakdowns, vector arithmetic, and scalp-region importance.
    breakdown_words = stages.run('breakdown_words', lambda: stratified_report(word_emb, word_meta))
    breakdown_categories: list[dict[str, Any]] = []
    if sent_meta is not None:
        breakdown_categories = stages.run(
            'retrieval_by_category',
            lambda: stratified_retrieval(sent_emb, sent_meta, np.asarray(sent_content_ids), 'category'),
        )
    analogy: dict[str, Any] = {}
    if not sweep:
        analogy = stages.run('analogy', lambda: analogy_report(word_emb, word_meta, raw_word_feats, return_hits=True))
    st = analogy.get('subject_transfer', {})
    subj_top1_hits = st.pop('top1_hits', []) if isinstance(st, dict) else []
    subj_chances = st.pop('chances', []) if isinstance(st, dict) else []
    region_map = _load_region_map(config)
    region_rows = stages.run('region_importance', lambda: _region_importance(word_band_power, word_meta, region_map))
    region_approximate = region_map is None or region_map.approximate

    # 4b/4c) What the dimensions encode, and whether related thoughts cluster ACROSS subjects (the north star).
    neurons: dict[str, Any] = {}
    emergence: dict[str, Any] = {}
    if not sweep:
        neurons = stages.run(
            'neurons', lambda: _neuron_report(word_emb, word_meta, word_band_power, region_map, config)
        )
        emergence = stages.run('emergence', lambda: _emergence_report(word_emb, word_meta, analogy))

    fit_label = _postprocess_fit(do_whiten or n_top > 0, train_sent_emb is not None)
    metrics: dict[str, Any] = {
        'run_name': run_name,
        'eval_profile': profile,
        'n_word_embeddings': int(len(word_emb)),
        'n_sentence_embeddings': int(len(sent_emb)),
        'embedding_health': health,
        'sentence_retrieval': sent_ret,
        'sentence_retrieval_phase_control': phase_ret,
        'sentence_retrieval_transductive': sent_ret_transductive,
        'postprocess_fit': fit_label,
        'length_projection': length_projection,
        'gallery_exposure': _gallery_exposure(config),
        'word_retrieval': word_ret,
        'word_retrieval_by_novelty': word_ret_by_novelty,
        'word_retrieval_freq_matched': word_ret_freq_matched,
        'probe_comparison': comparison,
        'breakdown_words': breakdown_words,
        'retrieval_by_category': breakdown_categories,
        'analogy': analogy,
        'region_importance': region_rows,
        'region_map_approximate': region_approximate,
        'neurons': {} if sweep else _neuron_summary(neurons),
        'emergence': emergence,
        'verdict': _verdict(
            comparison,
            health,
            sent_ret,
            analogy,
            sent_top1_hits=sent_top1_hits,
            subj_top1_hits=subj_top1_hits,
            subj_chances=subj_chances,
            phase_top1_hits=phase_top1_hits,
            generation=generation,
            min_prefix_kl=min_prefix_kl,
        ),
    }
    # An empty block and an uncomputed one are indistinguishable, so a sweep names what it did not measure.
    if sweep:
        metrics['eval_skipped'] = list(SWEEP_SKIPPED)
    if generation is not None:
        metrics['generation'] = generation
    if rescoring is not None:
        metrics['rescoring'] = rescoring

    # Merged additively under its own keys: a certified menu is a forced choice among K candidates, so it
    # must never reach a generation clause, and its key is not `menu`, which is the encoder cosine audit.
    if decoder_capacity is not None:
        metrics['decoder_capacity'] = decoder_capacity
        metrics['verdict'].update(capacity_verdict(decoder_capacity))

    # 4b) Honesty add-ons: permutation null, held-out cross-subject decode, anchor calibration.
    honesty = stages.run('honesty', lambda: _honesty_block(word_emb, word_meta, sent_emb, sent_content_ids, config))
    metrics['honesty'] = honesty

    # 4d) The honest scoreboard: every headline metric stated as a lift over the raw control.
    from zte.evaluation.audit.scoreboard import build_scoreboard

    # Guarded like every other block here: the scoreboard materialises a large similarity matrix, and
    # losing it must not discard an evaluation whose numbers are already computed.
    try:
        metrics['scoreboard'] = stages.run(
            'scoreboard',
            lambda: build_scoreboard(
                word_emb,
                word_meta,
                comparison,
                sent_emb,
                sent_content_ids,
                sent_meta,
                config,
                word_band_power=word_band_power,
                sent_n_words=sent_n_words,
                phase_sent_emb=phase_sent_post,
                generation=generation,
                rescoring=rescoring,
                postprocess_fit=fit_label,
            ),
        )
    except (ValueError, KeyError, IndexError, MemoryError) as exc:  # pragma: no cover - defensive
        _LOG.warning('Scoreboard skipped: %r', exc)
        metrics['scoreboard'] = None

    # The retrieval clause reads the held-out block when one exists; pooled `sentence_retrieval` is never a headline.
    board = metrics.get('scoreboard') or {}
    phase_block = board.get('phase_control_retrieval')
    if isinstance(phase_block, dict):
        phase_block.pop('top1_hits', None)
    held_block = board.get('held_out_retrieval')
    if isinstance(held_block, dict):
        # Popped, not kept: the per-query vector feeds the CI here and would bloat metrics.json.
        held_hits = [float(h) for h in held_block.pop('top1_hits', [])]
        held_point, held_lo, held_hi = _diff_ci(held_hits, float(held_block.get('chance_top1', float('nan'))))
        held_pass = bool(np.isfinite(held_lo) and held_lo > 0.0)
        if 'retrieval_above_phase' in metrics['verdict']:
            held_pass = held_pass and bool(metrics['verdict']['retrieval_above_phase'])
        metrics['verdict']['retrieval_above_chance'] = held_pass
        metrics['verdict']['retrieval_ci'] = [round(held_point, 4), round(held_lo, 4), round(held_hi, 4)]
        metrics['verdict']['retrieval_basis'] = 'held_out_retrieval'
    else:
        metrics['verdict']['retrieval_basis'] = 'sentence_retrieval (pooled; the split holds no subject out)'

    perm = honesty.get('retrieval_permutation') or {}
    if perm.get('applicable'):
        metrics['verdict']['retrieval_permutation_p'] = perm['p_value']
        metrics['verdict']['retrieval_above_chance_perm'] = perm['above_chance']
        # The headline needs both the bootstrap-CI lift and the permutation null; this only demotes.
        metrics['verdict']['retrieval_above_chance'] = bool(
            metrics['verdict']['retrieval_above_chance'] and perm['above_chance']
        )

    # 5) Figures (each guarded so tiny inputs never abort the run).
    figure_names: list[str] = []
    if not sweep:
        montage_csv = getattr(getattr(config, 'dataset', None), 'montage_csv', None)
        figure_names += stages.run(
            'figures_core',
            lambda: _relative(
                _render_figures(word_emb, word_meta, sent_emb, sent_content_ids, comparison, sent_ret, fig_dir), out
            ),
        )
        figure_names += stages.run(
            'figures_extended',
            lambda: _relative(_render_extended_figures(analogy, breakdown_words, region_rows, fig_dir), out),
        )
        figure_names += stages.run(
            'figures_sota',
            lambda: _relative(
                _render_sota_figures(
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
                ),
                out,
            ),
        )
    figures = [out / name for name in figure_names]
    metrics['figures'] = figure_names

    # 6) Interactive HTML explorers (self-contained; static PNG fallback).
    if interactive and not sweep:
        metrics.update(
            stages.run(
                'interactive',
                lambda: {
                    'interactive': _write_interactive(word_emb, word_meta, out, emergence),
                    'neuron_atlas': _write_neuron_atlas(neurons, out),
                    'scoreboard_html': _write_scoreboard_html(metrics.get('scoreboard'), out, run_name),
                    'generation_html': _write_generation_html(generation, out, run_name, min_prefix_kl),
                },
            )
        )

    # 6b) Per-dimension arrays are large, so metrics.json keeps only the compact summary. `default=str`
    # never raises, where `default=float` would crash on the last write of a multi-hour run.
    if not sweep:
        (out / 'neurons.json').write_text(json.dumps(neurons, indent=2, default=str), encoding='utf-8')

    # 7) Persist the results BEFORE the optional extras below, so nothing optional can cost the run
    # an evaluation that is already computed.
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, default=str), encoding='utf-8')
    # The partial file only ever guarded an evaluation in flight, and this write supersedes it.
    stages.clear()

    # 8) TensorBoard (projector + hparams + scalars + histograms + figures + text). Best-effort: a full
    # Drive mount or an odd figure must not discard the evaluation.
    if tensorboard:
        tb_dir = tensorboard if isinstance(tensorboard, str) else str(out / 'tb' / run_name)
        try:
            _write_tensorboard(tb_dir, word_emb, word_meta, sent_emb, sent_meta, metrics, figures, config)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:  # pragma: no cover - defensive
            _LOG.warning('TensorBoard export skipped: %r', exc)
    # `linear_scores` is dropped from the flat CSV so the table keeps one scalar per cell.
    pd.DataFrame([{k: v for k, v in r.items() if k != 'linear_scores'} for r in comparison]).to_csv(
        out / 'comparison.csv', index=False
    )
    if breakdown_words:
        pd.DataFrame(breakdown_words).to_csv(out / 'breakdown.csv', index=False)
    if region_rows:
        pd.DataFrame(region_rows).to_csv(out / 'region_importance.csv', index=False)
    (out / 'report.md').write_text(_render_report(metrics, comparison, figures, out), encoding='utf-8')
    _LOG.info('Evaluation written to %s (%d figures)', out, len(figures))
    return metrics


def _gallery_exposure(config: Any | None) -> dict[str, Any] | None:
    """Records whether the training loss discriminated the very stimuli the retrieval gallery is made of.

    A subject-only split holds out people, not sentences, so under it every gallery item was a training item. That
    was already true of the sentence-level CLIP target; a full-gallery or consensus-gallery term sharpens it from
    "the text was seen" into "separating these exact items *was* the training objective", which turns the headline
    from open-set retrieval into closed-set identification. The number is not wrong -- it answers a narrower
    question -- so it is labelled rather than adjusted.
    """
    objective = getattr(config, 'objective', None)
    train = getattr(config, 'train', None)
    if objective is None or train is None:
        return None

    terms = {
        'gallery_weight': float(getattr(objective, 'gallery_weight', 0.0)),
        'consensus_gallery_weight': float(getattr(objective, 'consensus_gallery_weight', 0.0)),
        'consensus_word_weight': float(getattr(objective, 'consensus_word_weight', 0.0)),
    }
    split = str(getattr(train, 'split', ''))
    active = sorted(name for name, weight in terms.items() if weight > 0.0)

    return {
        'split': split,
        'stimuli_held_out': split in {'by_stimulus', 'by_subject_and_stimulus'},
        'gallery_terms_active': active,
        'closed_set': bool(active) and split not in {'by_stimulus', 'by_subject_and_stimulus'},
    }


def _closed_set_lines(block: dict[str, Any] | None) -> list[str]:
    """Markdown naming the retrieval task the split and the loss together define."""
    if not block or not block.get('gallery_terms_active'):
        return []
    if not block.get('closed_set'):
        return [
            f'- Gallery exposure: `{block["split"]}` holds out stimuli, so the '
            f'{", ".join(f"`{t}`" for t in block["gallery_terms_active"])} denominator was restricted to training '
            'sentences and the scored stimuli were never negatives. This is open-set retrieval.'
        ]

    return [
        f'- **Closed-set caveat:** `{block["split"]}` holds out subjects, not sentences, so every sentence in the '
        f'gallery was in training, and {", ".join(f"`{t}`" for t in block["gallery_terms_active"])} trained the '
        'model to separate these exact items. The number below is therefore **identification over a known sentence '
        'set for an unseen reader**, not retrieval of an unseen sentence. It is comparable only with other arms '
        'carrying this same caveat -- for the open-set claim, run `by_subject_and_stimulus` and read its `test` cell.'
    ]


def _length_projection_lines(block: dict[str, Any] | None) -> list[str]:
    """Markdown for the length projection, saying plainly when it was skipped and why."""
    if not block:
        return []
    if block.get('status') != 'applied':
        return [f'- Length projection: **not applied** ({block.get("status")})']

    before, after = block.get('length_leakage_before', float('nan')), block.get('length_leakage_after', float('nan'))
    return [
        f'- Length projection (fitted on {block.get("n_fit")} train sentences, in the same post-processed frame the '
        f'retrieval below is scored in): word count explained **{before:.4f}** of sentence-embedding variance before '
        f'and **{after:.4f}** after, so every retrieval number below is measured on the projected space. The residual '
        'is the length a five-term basis fitted on other readers does not reach on these rows, and retrieval is free '
        'to use it -- read it beside the length-stratified gallery, which bounds the confound a different way.'
    ]


def _postprocess_fit(applied: bool, on_train: bool) -> str:
    """Names what the retrieval geometry was fitted on: `none`, `train split` or `transductive`."""
    if not applied:
        return 'none'
    return 'train split' if on_train else 'transductive'


def _postprocess(emb: np.ndarray, fit_on: np.ndarray | None, whiten: bool, n_top: int) -> np.ndarray:
    """Whitening then all-but-the-top, fitted on `fit_on` when given and on `emb` itself otherwise.

    Args:
        emb (np.ndarray): Embeddings to transform `(n, d)`.
        fit_on (np.ndarray | None): Rows the transform may be fitted on; `None` fits transductively.
        whiten (bool): Apply ZCA whitening.
        n_top (int): Leading principal directions to remove.
    """
    if fit_on is not None:
        from zte.evaluation.audit.rebaseline import fit_postprocess

        fitted = fit_postprocess(np.asarray(fit_on, dtype=np.float32), whiten=whiten, n_top=n_top)
        return fitted(emb)
    out = np.asarray(emb, dtype=np.float32)
    if whiten:
        out = M.whiten_features(out)
    if n_top > 0:
        out = M.all_but_the_top(out, n_top)
    return out


def _postprocess_transductive(
    emb: np.ndarray, carry: np.ndarray, whiten: bool, n_top: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fits whitening then all-but-the-top on `emb` itself and returns `emb` and `carry` in that one frame.

    Note:
        Whitening equalises the variance of every direction, so the leading principal direction of the whitened
        rows is numerically arbitrary and an independent refit picks a different one. `carry` therefore has to
        ride the same eigendecomposition and the same SVD rather than being transformed by a second call.

    Args:
        emb (np.ndarray): Rows the transform is fitted on and applied to `(n, d)`.
        carry (np.ndarray): Further rows to place in the same frame `(m, d)`.
        whiten (bool): Apply ZCA whitening.
        n_top (int): Leading principal directions to remove.

    Returns:
        tuple[np.ndarray, np.ndarray]: The transformed `emb` and `carry`.
    """
    x = np.asarray(emb, dtype=np.float32)
    c = np.asarray(carry, dtype=np.float32)
    if whiten:
        c = M.whiten_features(c, fit_on=x)
        x = M.whiten_features(x)

    if n_top > 0 and len(x) >= 2:
        centred = np.asarray(x, dtype=np.float64)
        mean = centred.mean(axis=0, keepdims=True)
        centred = centred - mean
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        u = vt[: min(n_top, vt.shape[0])]
        x = (centred - (centred @ u.T) @ u).astype(np.float32)
        carried = np.asarray(c, dtype=np.float64) - mean
        c = (carried - (carried @ u.T) @ u).astype(np.float32)

    return x, c


def _fit_length_projector(
    sent_n_words: np.ndarray | None,
    train_sent_emb: np.ndarray | None,
    train_sent_n_words: np.ndarray | None,
) -> tuple[LengthProjector | None, dict[str, Any]]:
    """Fits the length projection on the training split, saying plainly when it cannot be fitted and why.

    A silently skipped de-confounding is worse than none at all: the report would show retrieval numbers that look
    length-free and are not, so every refusal returns a `status` string that reaches `report.md`.
    """
    if sent_n_words is None:
        return None, {'status': 'skipped: no word counts for the scored sentences'}
    if train_sent_emb is None or train_sent_n_words is None:
        return None, {'status': 'skipped: no training split to fit on (fitting here would be transductive)'}

    projector = LengthProjector(int(np.asarray(train_sent_emb).shape[1]))
    try:
        projector.fit(np.asarray(train_sent_emb), np.asarray(train_sent_n_words))
    except ValueError as exc:
        return None, {'status': f'skipped: {exc}'}

    return projector, {
        'status': 'applied',
        'fit': 'train split',
        'n_fit': projector.n_fit,
        'basis': projector.state['basis'],
    }


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
        region_map (Any | None): Exact montage-derived `RegionMap`; when `None` the approximate coordinate-free default
            is used inside `region_importance`.

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
    # The package builds its HTML templates at import time, so a packaging problem surfaces here rather
    # than at call time; either way the run keeps its metrics and just loses the interactive views.
    try:
        from zte.evaluation.interactive import embedding_explorer_html, thought_space_explorer_html
    except (ImportError, OSError) as exc:  # pragma: no cover - packaging dependent
        _LOG.warning('Interactive explorers unavailable: %r', exc)
        return None

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
        path = embedding_explorer_html(word_emb, word_meta, out / 'interactive' / 'word_explorer.html')
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


def _emergence_report(word_emb: np.ndarray, word_meta: pd.DataFrame, analogy: dict[str, Any]) -> dict[str, Any]:
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


def _write_generation_html(block: dict[str, Any] | None, out: Path, run_name: str, min_prefix_kl: float) -> str | None:
    """Writes the reference/hypothesis/controls side-by-side page, returning its path relative to `out`.

    The page is the artifact that makes an absolute BLEU unreadable in isolation: every hypothesis sits
    beside the same row for each brain-independent control. Degrades to `None` rather than aborting.
    """
    if not block or not block.get('applicable'):
        return None
    try:
        from zte.evaluation.interactive import generation_html
    except ImportError:  # pragma: no cover
        return None
    try:
        path = generation_html(block, out / 'interactive' / 'generation.html', run_name, min_prefix_kl)
        return str(path.relative_to(out))
    except (ValueError, OSError, KeyError, TypeError) as exc:  # pragma: no cover
        _LOG.warning('Generation dashboard failed: %r', exc)
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
        builders.append(('neuron_selectivity.png', lambda: P.neuron_selectivity_heatmap(top_neurons[:12])))
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
    """Renders and saves all evaluation figures, skipping any that fail."""
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
                lambda: P.retrieval_curve(ks, sent_ret.get('chance_top1', 0.0), 'Cross-subject sentence retrieval'),
            )
        )
    if 'log_freq' in word_meta:
        builders.append(('probe_logfreq_scatter.png', lambda: _logfreq_scatter(word_emb, word_meta)))

    for name, builder in builders:
        try:
            _save(builder(), name)
        except (ValueError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
            _LOG.warning('Skipped figure %s: %r', name, exc)
    return written


def _logfreq_scatter(word_emb: np.ndarray, word_meta: pd.DataFrame) -> Any:
    """Builds a kNN leave-one-out predicted-vs-true scatter for log frequency."""
    index = NearestNeighborIndex(word_emb, word_meta[['log_freq']].reset_index(drop=True))
    pred = index.predict(word_emb, 'log_freq', k=10, task='regression', self_indices=np.arange(len(word_emb)))
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
    phase_top1_hits: list[float] | None = None,
    generation: dict[str, Any] | None = None,
    min_prefix_kl: float = 0.05,
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
        phase_top1_hits (list[float] | None): Per-query Top-1 hits of the phase-scrambled control through
            the identical retrieval path; supplying them demotes `retrieval_above_chance`.
        generation (dict[str, Any] | None): A `generation.generation_report` block.
        min_prefix_kl (float): Minimum prefix-influence KL (nats) the generation clause requires.
        effect_floor (float): Minimum ZTE-minus-noise effect the CI must clear.
        seed (int): Bootstrap seed.

    Returns:
        dict[str, Any]: Named boolean checks, the CIs behind them (`beats_noise_ci`,
            `retrieval_ci`, `subject_arithmetic_ci`) and the `effect_size_floor` used.
    """
    from zte.evaluation.generation import strip_quarantined

    # No `*_DIAGNOSTIC` or `*_RETRIEVAL` key reaches a clause below, whatever the caller passed.
    generation = strip_quarantined(generation) if generation else None
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

    # Pooled retrieval lift; the held-out scoreboard block supersedes this clause whenever the split provides one.
    ret_chance = float(sent_ret.get('chance_top1', float('nan')))
    ret_point, ret_lo, ret_hi = _diff_ci(sent_top1_hits, ret_chance, seed=seed)
    retrieval_pass = bool(np.isfinite(ret_lo) and ret_lo > 0.0)

    verdict: dict[str, Any] = {
        'beats_noise_on': beats_noise,
        'beats_noise_all_targets': len(beats_noise) == len(zte) and len(zte) > 0,
        'beats_noise_ci': beats_noise_ci,
        'effect_size_floor': effect_floor,
        'no_collapse': bool(health.get('effective_rank_ratio', 0) > 0.1 and health.get('dead_dim_fraction', 1) < 0.5),
        'retrieval_above_chance': retrieval_pass,
        'retrieval_ci': [round(ret_point, 4), round(ret_lo, 4), round(ret_hi, 4)],
    }

    # The phase-scrambled control travels the identical retrieval path, so the comparison is paired.
    if phase_top1_hits and sent_top1_hits and len(phase_top1_hits) == len(sent_top1_hits):
        real = np.asarray(sent_top1_hits, dtype=np.float64)
        phase = np.asarray(phase_top1_hits, dtype=np.float64)
        ph_point, ph_lo, ph_hi = M.bootstrap_ci(real - phase, seed=seed)
        verdict['retrieval_above_phase'] = bool(np.isfinite(ph_lo) and ph_lo > 0.0)
        verdict['retrieval_phase_ci'] = [round(ph_point, 4), round(ph_lo, 4), round(ph_hi, 4)]
        verdict['retrieval_above_chance'] = bool(verdict['retrieval_above_chance'] and verdict['retrieval_above_phase'])
    if generation:
        verdict.update(generation_verdict(generation, min_prefix_kl))
    if analogy:
        st = analogy.get('subject_transfer', {})
        subj_chance = float(np.mean(subj_chances)) if subj_chances else float(st.get('chance_top1', float('nan')))
        subj_point, subj_lo, subj_hi = _diff_ci(subj_top1_hits, subj_chance, seed=seed)
        verdict['subject_arithmetic_above_chance'] = bool(np.isfinite(subj_lo) and subj_lo > 0.0)
        verdict['subject_arithmetic_ci'] = [
            round(subj_point, 4),
            round(subj_lo, 4),
            round(subj_hi, 4),
        ]
    return verdict


def missing_controls(generation: dict[str, Any], metric: str) -> list[str]:
    """Names every pre-registered control that did not run, or ran and was not beaten.

    A control the decode recorded in `controls_unavailable` or `controls_skipped` produced no delta, so an AND over
    the deltas alone would silently drop it; here it counts as missing. A block that names no `controls_requested` at
    all cannot show which controls it pre-registered, and a control that vanished from such a block leaves no trace
    anywhere, so the absent ledger is itself reported as missing.

    Returns:
        list[str]: Sorted control names, or `UNRECORDED_CONTROLS` for an absent ledger; empty only when a
            block that named its controls ran and beat every one of them.
    """
    deltas = generation.get('deltas') or {}
    unavailable = generation.get('controls_unavailable') or {}
    skipped = generation.get('controls_skipped') or {}
    requested = generation.get('controls_requested')
    beaten = {name for name, d in deltas.items() if (d.get(metric) or {}).get('beats')}
    missing = (set(requested or deltas) | set(unavailable) | set(skipped)) - beaten
    if not requested:
        missing.add(UNRECORDED_CONTROLS)
    return sorted(missing)


def generation_verdict(generation: dict[str, Any], min_prefix_kl: float) -> dict[str, Any]:
    """The pre-registered generation gate: an AND over split, candidate set, controls, null and prefix KL.

    Every clause is reported with its number even when it fails, and any failing clause demotes the whole verdict --
    the same demotion pattern the retrieval permutation null uses above.

    Returns:
        dict[str, Any]: `generation_above_controls`, `generation_ci`, `generation_clauses` and the
            numbers behind each clause.
    """
    if not generation.get('applicable'):
        return {
            'generation_above_controls': False,
            'generation_clauses': {'applicable': False},
            'generation_reason': generation.get('reason', 'not applicable'),
        }

    metric = generation.get('primary_metric', 'content_f1')
    deltas = generation.get('deltas') or {}
    perm = generation.get('permutation') or {}
    p_value = perm.get('p_value') if perm.get('applicable') else None
    kl = generation.get('prefix_influence_kl')
    worst = generation.get('worst_control_ci') or {}
    strategy = generation.get('split_strategy')
    cell = generation.get('split')
    missing = missing_controls(generation, metric)
    clauses = {
        'honest_split': (strategy, cell) == HONEST_SPLIT,
        'no_candidate_set': generation.get('n_candidate_sentences') is None,
        'beats_every_control': bool(deltas) and not missing,
        'permutation_significant': bool(p_value is not None and p_value < 0.05),
        'prefix_influences_output': bool(kl is not None and kl >= min_prefix_kl),
    }
    return {
        'generation_above_controls': all(clauses.values()),
        'generation_clauses': clauses,
        'generation_metric': metric,
        'generation_split_strategy': strategy,
        'generation_split_cell': cell,
        'generation_controls_missing': missing,
        'generation_ci': [
            round(float(worst.get('point', float('nan'))), 4),
            round(float(worst.get('lo', float('nan'))), 4),
            round(float(worst.get('hi', float('nan'))), 4),
        ],
        'generation_worst_control': generation.get('worst_control'),
        'generation_permutation_p': p_value,
        'generation_prefix_kl': kl,
        'generation_min_prefix_kl': float(min_prefix_kl),
        'generation_n': generation.get('n'),
    }


def capacity_verdict(capacity: dict[str, Any]) -> dict[str, Any]:
    """The decoder menu-capacity outcome, restated with `capacity_` keys so it merges into the run verdict.

    The seven-clause certification rule has exactly one implementation, in `zte.evaluation.audit.capacity`;
    this reads the outcome it already reached rather than re-deriving it. Every key is namespaced, because
    a menu is a forced choice among K candidates and can never license a free-generation headline.

    Args:
        capacity (dict[str, Any]): A `capacity_report` block.

    Returns:
        dict[str, Any]: `capacity_certified`, `capacity_k`, `capacity_bits`, `capacity_clauses`,
            `capacity_readout` and `capacity_reason` -- and no key any other clause reads.
    """
    verdict = capacity.get('verdict') or {}

    return {
        'capacity_certified': bool(verdict.get('capacity_certified', False)),
        'capacity_k': verdict.get('capacity_k'),
        'capacity_bits': verdict.get('capacity_bits'),
        'capacity_clauses': verdict.get('capacity_clauses') or {},
        'capacity_readout': capacity.get('readout', 'menu selection'),
        'capacity_reason': verdict.get('reason'),
    }


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
    """Renders the Markdown evaluation report."""
    health = metrics['embedding_health']
    sent = metrics['sentence_retrieval']
    verdict = metrics['verdict']
    # The retrieval clause is judged on the held-out block when one exists, so its numbers come from there too.
    held = (metrics.get('scoreboard') or {}).get('held_out_retrieval') or {}
    on_held_out = verdict.get('retrieval_basis') == 'held_out_retrieval'
    ret_src = held if on_held_out else sent
    lines = [
        f'# ZTE evaluation report -- {metrics["run_name"]}',
        '',
        f'Word embeddings: **{metrics["n_word_embeddings"]}** | '
        f'sentence embeddings: **{metrics["n_sentence_embeddings"]}**',
        '',
    ]
    if metrics.get('eval_profile') == 'sweep':
        lines += [
            '_Sweep evaluation profile: '
            + ', '.join(f'`{block}`' for block in metrics.get('eval_skipped', SWEEP_SKIPPED))
            + ' were not computed, so their absence below is by configuration and not a measurement. Every '
            'number that is here is the one the full profile would have produced._',
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
        f'(judged on `{verdict.get("retrieval_basis", "sentence_retrieval")}`; '
        f'Top-1 {ret_src.get("top1", float("nan")):.3f} vs query-weighted chance '
        f'{ret_src.get("chance_top1", float("nan")):.3f}; '
        f'lift CI {_fmt_ci(verdict.get("retrieval_ci"))}'
        + (f'; permutation p={verdict["retrieval_permutation_p"]:.3f}' if 'retrieval_permutation_p' in verdict else '')
        + (f'; rank-percentile {ret_src["rank_percentile"]:.3f}' if 'rank_percentile' in ret_src else '')
        + (f', median rank {ret_src["median_rank"]:.0f}' if 'median_rank' in ret_src else '')
        + '). The headline requires BOTH the CI lift and the permutation null; '
        'rank-percentile (1.0 = correct match ranked first) shows the whole distribution, not just the tail.',
    ]
    if 'retrieval_above_phase' in verdict:
        lines.append(
            f'- Above the phase-scrambled control: **{verdict["retrieval_above_phase"]}** '
            f'(paired Top-1 delta CI {_fmt_ci(verdict.get("retrieval_phase_ci"))}). The control runs '
            'through the identical encoder and retrieval path, so this is the same-path floor, '
            'not an analytic one.'
        )
    if 'generation_above_controls' in verdict:
        lines.append(
            f'- Free-running generation above every control: '
            f'**{verdict["generation_above_controls"]}** '
            f'(worst control `{verdict.get("generation_worst_control")}`, '
            f'CI {_fmt_ci(verdict.get("generation_ci"))})'
        )
    if 'capacity_certified' in verdict:
        bits = verdict.get('capacity_bits')
        lines.append(
            f'- Decoder menu capacity certified: **{verdict["capacity_certified"]}** '
            f'(K = {"—" if verdict.get("capacity_k") is None else verdict["capacity_k"]}, '
            f'{"—" if bits is None else format(bits, ".4f")} bits). The readout is '
            f'{verdict.get("capacity_readout", "menu selection")} -- a forced choice among K candidates, '
            'judged on its own clauses and never part of the generation gate.'
        )
    lines += [
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
        f'- Dead dimensions: {health["dead_dim_fraction"]:.1%} | mean norm {health["mean_norm"]:.2f}',
        *_length_projection_lines(metrics.get('length_projection')),
        *_closed_set_lines(metrics.get('gallery_exposure')),
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


def _generation_section(metrics: dict[str, Any]) -> list[str]:
    """Markdown for free-running generation and decoder-rescoring retrieval, each labelled for what it is."""
    generation = metrics.get('generation') or {}
    rescoring = metrics.get('rescoring') or {}
    if not generation.get('applicable') and not rescoring:
        return []
    verdict = metrics.get('verdict', {})
    lines = ['## Decoding (free-running generation · decoder rescoring)', '']

    if rescoring:
        ci = rescoring.get('rank_percentile_ci') or (float('nan'),) * 3
        lines += [
            f'- **Decoder-rescoring retrieval** — {int(rescoring.get("n_queries", 0))} queries over a '
            f'{int(rescoring.get("n_gallery", 0))}-sentence gallery: Top-1 '
            f'{rescoring.get("top1", float("nan")):.4f} vs chance '
            f'{rescoring.get("chance_top1", float("nan")):.4f} '
            f'(p={rescoring.get("top1_p", float("nan")):.1e}), rank percentile {ci[0]:.4f} '
            f'[{ci[1]:.4f}, {ci[2]:.4f}]. Forced choice over a known candidate set — this is '
            'retrieval, and it is the statistically powered readout, not a generation claim.',
            '',
        ]

    if generation.get('applicable'):
        metric = generation.get('primary_metric', 'content_f1')
        absolute = (generation.get('absolute') or {}).get('hypothesis') or {}
        oracle = (generation.get('absolute') or {}).get('oracle') or {}
        clauses = verdict.get('generation_clauses') or {}
        lines += [
            f'Free-running decode of {int(generation.get("n", 0))} held-out readings on the '
            f'`{generation.get("split")}` cell of `{generation.get("split_strategy")}`, no reference '
            f'length and '
            f'{"no candidate set" if generation.get("n_candidate_sentences") is None else "A CANDIDATE SET"}. '
            'Absolute scores are not results: a frozen LM reaches ROUGE-1 in the 0.10-0.18 range against '
            'any English reference from function words alone.',
            '',
            f'- Absolute: BLEU-4 {absolute.get("bleu4", float("nan")):.4f}, '
            f'ROUGE-1 {absolute.get("rouge1", float("nan")):.4f}, '
            f'ROUGE-L {absolute.get("rougeL", float("nan")):.4f}, '
            f'WER {absolute.get("wer", float("nan")):.4f}, '
            f'content-word F1 {absolute.get("content_f1", float("nan")):.4f}',
        ]
        if oracle:
            lines.append(
                f'- Text oracle (true sentence embedding through the identical bridge and LM): '
                f'BLEU-4 {oracle.get("bleu4", float("nan")):.4f}, '
                f'content-word F1 {oracle.get("content_f1", float("nan")):.4f} — a positive control on '
                'the head, saying nothing about EEG.'
            )
        lines += [
            '',
            f'| control | {metric} delta | 95% CI | clears zero |',
            '| --- | --- | --- | --- |',
        ]
        for name, delta in (generation.get('deltas') or {}).items():
            d = delta.get(metric, {})
            lines.append(
                f'| {name} | {d.get("point", float("nan")):+.4f} '
                f'| [{d.get("lo", float("nan")):+.4f}, {d.get("hi", float("nan")):+.4f}] '
                f'| {"✓" if d.get("beats") else "·"} |'
            )
        absent = (generation.get('controls_unavailable') or {}) | (generation.get('controls_skipped') or {})
        for name, reason in sorted(absent.items()):
            lines.append(f'| {name} | NEVER RAN ({reason}) | -- | · |')
        lines += [
            '',
            f'**Verdict — generation above controls: {verdict.get("generation_above_controls")}** '
            f'(worst control `{verdict.get("generation_worst_control")}`, '
            f'CI {_fmt_ci(verdict.get("generation_ci"))}). An AND over: '
            + ', '.join(f'{k}={v}' for k, v in clauses.items())
            + '.',
        ]
        if not clauses.get('honest_split', True):
            lines.append(
                f'- The headline is reserved for the `{HONEST_SPLIT[1]}` cell of `{HONEST_SPLIT[0]}`, '
                'the only readings held out from the subject and the stimulus at once.'
            )
        missing = verdict.get('generation_controls_missing') or []
        if missing:
            registered = generation.get('controls_requested') or missing
            lines.append(
                f'- Pre-registered controls not beaten: {len(missing)} of {len(registered)} -- '
                + ', '.join(f'`{c}`' for c in missing)
                + '. A control that never ran fails its clause rather than being dropped from the AND.'
            )
        from zte.evaluation.generation import quarantined_keys

        quarantined = quarantined_keys(generation)
        if quarantined:
            lines.append(f'- Quarantined and never read by the verdict: {", ".join(f"`{k}`" for k in quarantined)}.')
    lines.append('')
    return lines


def _capacity_section(metrics: dict[str, Any]) -> list[str]:
    """Markdown for the certified menu capacity, with the sizes no pool could fill named as unreachable."""
    capacity = metrics.get('decoder_capacity') or {}
    if not capacity:
        return []

    from zte.evaluation.audit.capacity import capacity_markdown_lines

    headline = capacity.get('headline') or {}
    block = ((capacity.get('scores') or {}).get(headline.get('score')) or {}).get(headline.get('flavor')) or {}
    feasible = [str(k) for k in (block.get('ks_feasible') or [])]
    unreachable = [str(k) for k in (block.get('ks_unreachable') or [])]

    # Exact word-count pools hold a median of ~8 candidates on a 300-sentence gallery, so the larger sizes
    # have no queries at all; read as failures they would understate a decoder that was never asked.
    reach = (
        f'Unreachable on this gallery -- no query had K-1 distractors at its own word count and task, so '
        f'these sizes were never put to the decoder and their rows are absence of evidence rather than a '
        f'failed certification: {", ".join(unreachable)}.'
        if unreachable
        else 'Every swept size had queries, so no row here is unreachable rather than failed.'
    )

    return [
        *capacity_markdown_lines(capacity),
        f'- Menu sizes with queries: {", ".join(feasible) or "none"}. {reach}',
        f'- The readout is {capacity.get("readout", "menu selection")}: the decoder picks the read sentence out '
        'of K candidates. It is retrieval-shaped, so it is gated on its own clauses and can never stand in for '
        'the free-generation verdict above.',
        '',
    ]


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
    lines += _generation_section(metrics)
    lines += _capacity_section(metrics)
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
                f'| {e["expression"]} | {e["retrieved"]} | {"✓" if e["hit"] else "·"} | {e["similarity"]} |'
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
            lines.append(f'| {region_name} | ' + ' | '.join(f'{v:.2f}' for v in row.to_numpy()) + ' |')
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
            f'| {r["value"]} | {r["n"]} | {r["top1"]} | {r["top5"]} | {r["mrr"]} | {r["chance_top1"]} |' for r in by_cat
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
