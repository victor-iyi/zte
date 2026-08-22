"""High-level training orchestration: leakage-aware splits, model, objective, loaders, then `Trainer`."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from zte.config import ObjectiveConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.torch_dataset import ZuCoTorchDataset, build_subject_vocab, make_dataloader
from zte.device import DeviceSpec, auto_num_workers, resolve_device, seed_everything
from zte.logging_utils import get_logger
from zte.models.embedding import ZTEModel, build_model
from zte.models.encoder.gallery import text_word_counts
from zte.models.objectives import PrefixDecodeObjective, build_objective
from zte.models.objectives.base import _ObjectiveBase  # noqa: PLC2701
from zte.training.init import EncoderSource, load_encoder
from zte.training.trainer import Trainer

_LOG = get_logger('training.pipeline')

# Objectives whose loss compares items within a batch, so a final short batch distorts it.
_DROP_LAST = frozenset({'skipgram', 'cbow', 'cpc', 'clip', 'decode'})

# Frontend shapes that must be carried in the checkpoint for the encoder to be rebuilt exactly.
_SHAPE_KEYS = (
    'in_dim',
    'raw_shape',
    'n_channels',
    'bp_features_per_channel',
    'montage_csv',
    'signature_dim',
)


@dataclass(slots=True)
class TrainingArtifacts:
    """Everything `run_training` produces.

    Attributes:
        trainer (Trainer): The (already-run) trainer, exposing the model and checkpoints.
        history (dict[str, list[float]]): The per-epoch metric history.
        device (DeviceSpec): The resolved device spec used for training.
        test_indices (np.ndarray | None): Held-out test row indices (from `train.test_fraction`), or
            `None` when no test split was carved. Never seen during training.
    """

    trainer: Trainer
    history: dict[str, list[float]]
    device: DeviceSpec
    test_indices: np.ndarray | None = None


def run_training(
    config: ZTEConfig,
    dataset: ZuCoDataset,
    device: DeviceSpec | None = None,
    resume: bool = False,
) -> TrainingArtifacts:
    """Builds and runs a full ZTE job over `dataset`, in whichever mode `config.train.mode` names.

    `'encoder'` pretrains the encoder from scratch. `'decoder'` and `'joint'` start from another run's checkpoint:
    the encoder, its fitted normaliser and its fitted aligner are all restored rather than refitted, because a frozen
    encoder handed differently scaled inputs does not fail -- it quietly underperforms.

    Args:
        config (ZTEConfig): The complete run configuration.
        dataset (ZuCoDataset): A built `ZuCoDataset`.
        device (DeviceSpec | None): Optional pre-resolved device spec.
        resume (bool): Continue an interrupted run from its `last.pt` checkpoint (see `Trainer`).

    Returns:
        TrainingArtifacts: A `TrainingArtifacts` with the trainer, history and device.

    Raises:
        ValueError: If the configured representation has no matching tensors in the dataset (e.g. raw frontend but
            band-power-only dataset).
    """
    device = device or resolve_device(config.train.device, config.train.precision)

    # `Trainer` seeds too, but weight initialisation happens here, so without this `train.seed` never reaches it.
    seed_everything(config.train.seed, deterministic=config.train.deterministic)
    splits = dataset.split(
        config.train.split,
        val_fraction=config.train.val_fraction,
        test_fraction=config.train.test_fraction,
        holdout_subject=config.train.loso_holdout_subject,
        seed=config.train.seed,
    )

    source = _load_source(config, device)
    if source is None:
        # Fit the normaliser (and imputer) on the TRAIN split only, so val/test statistics never leak in.
        dataset.refit_normalizer(splits['train'])

        # Label-free, so unlike the normaliser this may see the held-out subject -- calibration, not a peek.
        dataset.align_raw(splits['train'])
    else:
        dataset.set_normalizer_state(source.normalizer_state)
        dataset.set_aligner_state(source.aligner_state)

    # Size the encoder to the data, then build the objective on top of it.
    in_dim, raw_shape, feature_dim = _shapes(dataset, config)
    shapes = _frontend_shapes(dataset, config, in_dim, raw_shape)
    if source is None:
        model = build_model(config.model, **shapes)
    else:
        # The inherited encoder *is* the model, so the run's own `model` section must describe it rather than
        # whatever the decoder config happened to say -- that section is what rebuilds the encoder at inference.
        model, shapes = source.model, _source_shapes(source, shapes)
        config.model = source.config.model
    objective = build_objective(config.objective, model, feature_dim=feature_dim, decoder_config=config.decoder)
    if source is not None:
        _attach_source_head(objective, source)

    subject_vocab = build_subject_vocab(dataset)
    # Auto-pick DataLoader workers per backend when config.train.num_workers < 0 (else honour it).
    workers = auto_num_workers(device, config.train.num_workers)
    # Only emit per-word behaviour targets when the behaviour head is active.
    beh_targets = config.objective.behaviour_targets if config.objective.behaviour_weight > 0.0 else ()
    # `None` keeps the word-type-keyed static meaning path instead of a per-occurrence target.
    mctx = config.objective.meaning_contextual if config.objective.meaning_distill_weight > 0.0 else None

    # Build the torch datasets up front so static-shape padding can be sized from actual lengths.
    train_td = dataset.to_torch(
        split=splits['train'],
        subject_vocab=subject_vocab,
        behaviour_targets=beh_targets,
        meaning_contextual=mctx,
        meaning_context_layer=config.objective.meaning_context_layer,
    )
    val_td = (
        dataset.to_torch(split=splits['val'], subject_vocab=subject_vocab, behaviour_targets=beh_targets)
        if len(splits['val']) > 0
        else None
    )
    clip_hard_negs = None  # (n_text, k) semantic-hard negatives for the CLIP loader (set below)

    # Attach the auxiliary targets, which need the word vocabulary and behaviour spec to exist first.
    obj = config.objective
    if obj.meaning_distill_weight > 0.0 or obj.behaviour_weight > 0.0 or obj.data2vec_aux_weight > 0.0:
        from zte.data.targets.meaning import build_meaning_matrix

        meaning_mat = None
        # A contextual target rides in the batch, so the head is sized from its width instead.
        if obj.meaning_distill_weight > 0.0 and not obj.meaning_contextual:
            mat = build_meaning_matrix(train_td.word_vocab, obj.meaning_source, obj.meaning_dim)
            meaning_mat = torch.from_numpy(mat)
        elif obj.meaning_distill_weight > 0.0 and obj.meaning_contextual:
            objective._meaning_contextual_dim = int(getattr(train_td, 'meaning_dim', 0))  # noqa: SLF001
        beh_binary = torch.from_numpy(train_td.behaviour_binary) if beh_targets else None
        objective.attach_auxiliary(meaning_matrix=meaning_mat, behaviour_binary=beh_binary, feature_dim=in_dim)

    # One word type -> one frozen embedding, so the token-level loss has something true to pull each word toward.
    if obj.lexical_weight > 0.0 or obj.lexical_reader_weight > 0.0:
        _attach_lexical(config, objective, train_td, device)

    # One word-piece -> one frozen embedding, so the sub-word loss has something true to pull each slice toward.
    if obj.token_weight > 0.0 or obj.token_reader_weight > 0.0:
        _attach_tokens(config, objective, train_td, device)

    # Built only when the knob is on, so a run without it never trips the one-text-one-task assertion.
    text_tasks = train_td.text_task_ids() if obj.within_task_negatives else None

    # Twelve subjects read the same stimuli, so every stimulus has a cross-reader consensus worth distilling toward.
    if _consensus_requested(obj):
        objective.attach_consensus(len(train_td.text_vocab), train_td.n_content, text_tasks=text_tasks)

    # Embed every unique sentence once with the frozen text encoder, then attach it as the alignment target.
    ordered_texts = train_td.ordered_texts()
    attach_text = getattr(objective, 'attach_text', None)
    if obj.name in {'clip', 'decode'} and callable(attach_text):
        head = getattr(objective, 'clip_head', None)
        text_matrix = _text_matrix(config, ordered_texts, device, fallback_dim=_head_width(head))
        _attach_text_target(attach_text, text_matrix, ordered_texts, train_td.split_text_ids, text_tasks)
        if obj.semantic_hard_negatives:
            from zte.data.targets.text import mine_hard_negatives

            clip_hard_negs = mine_hard_negatives(ordered_texts, text_matrix, k=obj.hard_negative_pool)
            _LOG.info(
                'Mined %d semantic-hard negatives per sentence (surface-similar, meaning-distinct).',
                obj.hard_negative_pool,
            )

    # Wire the loaders: static shapes pad every batch alike so XLA compiles a single graph.
    pad_to = _static_pad_length(config.train.static_shapes, device, train_td, val_td)
    # Cross-subject positives need one stimulus read by several subjects in the same batch.
    group_by_stimulus = bool(config.objective.cross_subject_positives)
    train_loader = make_dataloader(
        train_td,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.supports_pin_memory,
        drop_last=config.objective.name in _DROP_LAST,
        group_by_stimulus=group_by_stimulus,
        seed=config.train.seed,
        pad_to=pad_to,
        hard_negatives=clip_hard_negs,
    )
    val_loader = None
    if val_td is not None:
        val_loader = make_dataloader(
            val_td,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.supports_pin_memory,
            pad_to=pad_to,
        )

    if isinstance(objective, PrefixDecodeObjective):
        _wire_decoder(config, objective, model, train_td, device, workers, pad_to)

    # Everything inference needs to rebuild the model is embedded in the checkpoint.
    extra: dict[str, Any] = {
        'subject_vocab': subject_vocab,
        'text_vocab': train_td.text_vocab,
        'normalizer': None if dataset.normalizer is None else dataset.normalizer.state,
        'feature_names': dataset.feature_names,
        'aligner': None if dataset.aligner is None else dataset.aligner.state,
        **shapes,
    }
    if source is not None:
        extra['encoder_source'] = source.provenance()
    trainer = Trainer(
        model=model,
        objective=objective,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        extra_state=extra,
        resume=resume,
    )
    history = trainer.train()
    _LOG.info(
        'Training complete: %d epochs, final train_loss=%.4f',
        config.train.epochs,
        history['train_loss'][-1] if history['train_loss'] else float('nan'),
    )
    return TrainingArtifacts(trainer=trainer, history=history, device=device, test_indices=splits.get('test'))


def _consensus_requested(obj: ObjectiveConfig) -> bool:
    """Returns whether any cross-reader consensus term carries weight in this objective configuration."""
    return max(obj.consensus_weight, obj.consensus_gallery_weight, obj.consensus_word_weight) > 0.0


def _attach_text_target(
    attach: Any,
    text_matrix: np.ndarray,
    texts: list[str],
    split_text_ids: list[int],
    text_tasks: torch.Tensor | None = None,
) -> None:
    """Attaches the frozen gallery, handing the extras to the objectives whose signature accepts them.

    Only the CLIP objective scores against a length-matched, split-restricted, optionally task-matched denominator;
    the decoder's `attach_text` takes the matrix alone, so the extras are offered rather than forced.
    """
    matrix = torch.from_numpy(text_matrix)
    params = inspect.signature(attach).parameters
    if 'text_lengths' in params:
        kwargs = {'text_tasks': text_tasks} if 'text_tasks' in params else {}
        attach(matrix, text_word_counts(texts), split_text_ids, **kwargs)
        return

    attach(matrix)


def _load_source(config: ZTEConfig, device: DeviceSpec) -> EncoderSource | None:
    """Loads the encoder a decoder or joint run starts from, or `None` for an encoder run.

    Raises:
        ValueError: If a decoder/joint run names no `train.encoder_ckpt`, or a joint run also asks for
            `train.freeze_encoder`, which would hold the encoder frozen through the stage B it exists to run.
    """
    if config.train.mode == 'encoder':
        return None
    if not config.train.encoder_ckpt:
        raise ValueError(
            f'train.mode={config.train.mode!r} needs train.encoder_ckpt: the decoder is built over an encoder that '
            'was trained separately.'
        )
    if config.train.mode == 'joint' and config.train.freeze_encoder:
        raise ValueError(
            "train.mode='joint' cannot be combined with train.freeze_encoder=true: the encoder would stay frozen for "
            'every epoch and train.stage_a_epochs would decide nothing. Set train.freeze_encoder=false for a joint '
            "run, or train.mode='decoder' to keep the encoder frozen throughout."
        )
    return load_encoder(config.train.encoder_ckpt, config, device)


def _frontend_shapes(
    dataset: ZuCoDataset,
    config: ZTEConfig,
    in_dim: int | None,
    raw_shape: tuple[int, int] | None,
) -> dict[str, Any]:
    """Resolves the frontend geometry this dataset implies, in `build_model` keyword form."""
    n_channels, bp_features_per_channel = _channel_shape(dataset, config, raw_shape)
    signature_dim = (
        dataset.aligner.signature_dim if (dataset.aligner is not None and config.dataset.subject_signature) else 0
    )
    return {
        'in_dim': in_dim,
        'raw_shape': raw_shape,
        'n_channels': n_channels,
        'bp_features_per_channel': bp_features_per_channel,
        'montage_csv': config.dataset.montage_csv,
        'signature_dim': signature_dim,
    }


def _source_shapes(source: EncoderSource, current: dict[str, Any]) -> dict[str, Any]:
    """Returns the shapes the loaded encoder was built with, warning when this dataset implies others."""
    shapes = {key: source.shapes.get(key) for key in _SHAPE_KEYS}
    shapes['signature_dim'] = int(shapes['signature_dim'] or 0)
    drifted = [key for key in ('in_dim', 'raw_shape') if current.get(key) != shapes.get(key)]
    if drifted:
        _LOG.warning(
            'This dataset implies %s but the loaded encoder was built with %s; the checkpoint wins and a forward '
            'pass will fail if the tensors really differ.',
            {k: current.get(k) for k in drifted},
            {k: shapes.get(k) for k in drifted},
        )
    return shapes


def _attach_source_head(objective: _ObjectiveBase, source: EncoderSource) -> None:
    """Restores the source run's projections into the frozen text space, which is what the bridge reads.

    They are attached frozen in every mode: the gap correction is fitted once against the vectors these projections
    produce, and stage A is bridge-only, so a projection that moved would invalidate both.
    """
    _restore_lexical_head(objective, source)
    attach = getattr(objective, 'attach_clip_head', None)
    if not callable(attach):
        return
    weight = source.objective_state.get('clip_head.weight')
    if weight is None:
        _LOG.warning(
            'Encoder checkpoint %s carries no clip_head, so the projection into the text space is learned here '
            'instead of inherited; the embedding cache is unavailable while it moves.',
            source.path,
        )
        return
    attach(weight, source.objective_state.get('clip_head.bias'))


def _restore_lexical_head(objective: _ObjectiveBase, source: EncoderSource) -> None:
    """Reinstates the encoder run's per-word text projection, which the evidence path reads word by word.

    A decoder run cannot learn this from scratch and stay honest: the projection is what makes a word's EEG mean that
    word, and it was trained contrastively across readers on the encoder's own split. Rebuilt here it would be fitted
    on the decoder's split instead, so the run is told loudly when the source carries none.
    """
    weight = source.objective_state.get('lexical.head.weight')
    if weight is None:
        return
    from zte.models.objectives.lexical import LexicalAligner

    aligner = LexicalAligner(int(weight.shape[1]), int(weight.shape[0]))
    inherited = {k.removeprefix('lexical.'): v for k, v in source.objective_state.items() if k.startswith('lexical.')}
    aligner.load_state_dict(inherited, strict=False)
    aligner.requires_grad_(False)
    objective.lexical = aligner
    _LOG.info('Restored the source run lexical projection %d -> %d (frozen).', weight.shape[1], weight.shape[0])


def _attach_lexical(
    config: ZTEConfig, objective: _ObjectiveBase, train_td: ZuCoTorchDataset, device: DeviceSpec
) -> None:
    """Builds and attaches the frozen per-word-type embedding target for token-level lexical alignment.

    Note:
        With no frozen encoder available the target falls back to a deterministic hash, exactly as the sentence-level
        CLIP target does, so the mechanism stays testable offline. A hash carries no semantics, so the alignment then
        trains the encoder to predict an arbitrary code per word type and nothing about the run's lexical numbers is
        meaningful -- which is why the fallback is a warning and not a note.
    """
    from zte.data.targets.lexical import build_lexical_matrix

    inherited = objective.lexical
    if inherited is not None and not any(p.requires_grad for p in inherited.parameters()):
        # A decoder run reads the encoder's frozen projection to place each word in the text space; it never
        # trains it, so it needs no target -- and building one here could only disagree with the space the
        # projection was fitted to.
        _LOG.info('Lexical projection inherited frozen from the source encoder; no target is built for it here.')
        return

    obj = config.objective
    matrix, dim = build_lexical_matrix(
        train_td.word_vocab,
        obj.lexical_source or obj.text_source,
        backend=obj.text_backend,
        prefix=obj.text_query_prefix,
        device=str(device.device),
    )
    if matrix is None:
        # Match the head when one already exists, so the fallback cannot introduce a width mismatch of its own.
        dim = int(inherited.head.out_features) if inherited is not None else (obj.meaning_dim or 384)
        rng = np.random.default_rng(config.train.seed)
        matrix = rng.standard_normal((max(len(train_td.word_vocab), 1), dim)).astype(np.float32)
        matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8, None)
        _LOG.warning(
            'Lexical target unavailable; using a hash target (dim %d, NO semantics). Every lexical number from '
            'this run is a wiring check, not a result.',
            dim,
        )
    objective.attach_lexical(torch.from_numpy(matrix))
    _LOG.info('Lexical alignment target attached: %d word types x %d dims.', len(matrix), dim)


def _attach_tokens(
    config: ZTEConfig, objective: _ObjectiveBase, train_td: ZuCoTorchDataset, device: DeviceSpec
) -> None:
    """Builds and attaches the frozen sub-word target for the token level, keyed by `content_id`.

    Note:
        The target is indexed by `content_id`, which is `(stimulus_key, word_idx)` -- not by a word's position in
        the batch, which a per-word split or a dropped omitted row can shift. That is also why the level needs no
        collate change and cannot invalidate the prepared bundle cache.
    """
    from zte.data.targets.tokens import build_subword_matrix, build_target_tokens, build_token_alignment

    obj = config.objective
    source = obj.token_source or config.decoder.lm_source
    texts, words = train_td.ordered_texts(), train_td.ordered_words()
    if not texts:
        _LOG.warning('No gallery texts: the sub-word level has nothing to align against and stays off.')
        return

    alignment = build_token_alignment(
        texts,
        words,
        source,
        max_length=obj.token_max_length,
        cache_dir=str(Path(config.dataset.cache_dir) / 'tokens'),
    )
    cache = str(Path(config.dataset.cache_dir) / 'tokens')
    targets = build_target_tokens(texts, source, max_length=obj.token_max_length, cache_dir=cache)
    subword = build_subword_matrix(targets.ids, source)

    # (n_content, n_sub): the row of the frozen matrix for slot k of each word, -1 where the word is shorter.
    text_ids, word_idx = train_td.content_rows()
    n_sub = obj.token_sub_tokens
    piece_target = np.full((len(text_ids), n_sub), -1, dtype=np.int64)
    ok = (text_ids >= 0) & (word_idx >= 0) & (word_idx < alignment.max_words)
    rows = np.nonzero(ok)[0]
    for content_id in rows:
        tid, widx = int(text_ids[content_id]), int(word_idx[content_id])
        slots = np.nonzero((alignment.token_word[tid] == widx) & (alignment.piece_index[tid] >= 0))[0]
        for slot in slots[:n_sub]:
            piece = int(alignment.piece_index[tid, slot])
            if piece < n_sub:
                piece_target[content_id, piece] = subword.rows.get(int(targets.ids[tid, slot]), -1)

    covered = float((piece_target >= 0).any(axis=1).mean())
    matrix = torch.from_numpy(subword.matrix).to(device.device)
    objective.attach_subwords(matrix, torch.from_numpy(piece_target).to(device.device))
    if subword.source.endswith('#hash'):
        _LOG.warning(
            'Sub-word target unavailable for %r; using a hash target (dim %d, NO semantics). The level trains the '
            'encoder to predict an arbitrary code per word-piece, so every sub-word number from this run is a '
            'wiring check, not a result.',
            source,
            subword.matrix.shape[1],
        )
    _LOG.info(
        'Sub-word target attached: %d types x %d dims from %s; %.1f%% of word slots carry at least one piece.',
        subword.matrix.shape[0],
        subword.matrix.shape[1],
        subword.source,
        100.0 * covered,
    )


def _head_width(head: Any) -> int | None:
    """Returns an attached projection's output width, or `None` when there is none to match."""
    return int(head.out_features) if head is not None and hasattr(head, 'out_features') else None


def _text_matrix(
    config: ZTEConfig, texts: list[str], device: DeviceSpec, fallback_dim: int | None = None
) -> np.ndarray:
    """Builds the frozen `(n_texts, dim)` sentence-embedding target, falling back to a semantics-free hash target.

    Note:
        `fallback_dim` is the width of an already-attached projection. A decoder run inherits its `clip_head` from
        the encoder, so a hash fallback sized from `meaning_dim` instead would be rejected outright -- which is the
        right failure, but only after the run has already loaded the dataset and the LM.
    """
    from zte.data.targets.text import build_sentence_text_matrix

    mat, dim = build_sentence_text_matrix(
        texts,
        config.objective.text_source,
        backend=config.objective.text_backend,
        prefix=config.objective.text_query_prefix,
        device=str(device.device),
    )
    if mat is None:  # dependency / model unavailable -> hash target (mechanism only)
        dim = fallback_dim or config.objective.meaning_dim or 384
        rng = np.random.default_rng(config.train.seed)
        mat = rng.standard_normal((max(len(texts), 1), dim)).astype(np.float32)
        mat /= np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8, None)
        _LOG.warning('CLIP text target unavailable; using a hash target (dim %d, no semantics).', dim)
    _LOG.info(
        'Attached frozen text target: %d sentences x %d dims (%s).',
        len(mat),
        dim,
        config.objective.text_source or 'hash',
    )
    return mat


def _wire_decoder(
    config: ZTEConfig,
    objective: PrefixDecodeObjective,
    model: ZTEModel,
    train_td: ZuCoTorchDataset,
    device: DeviceSpec,
    workers: int,
    pad_to: int | None,
) -> None:
    """Attaches the decoder's targets and cache, fits the gap correction, then pretrains the bridge on text alone.

    The order is forced: the gap correction and the embedding cache both come from one pass of the frozen encoder over
    the training split, and text-only pretraining must see training stimuli only, which is asserted rather than
    filtered.

    Args:
        config (ZTEConfig): The run configuration.
        objective (PrefixDecodeObjective): The decode objective owning the bridge, cache and targets.
        model (ZTEModel): The encoder.
        train_td (ZuCoTorchDataset): The training split.
        device (DeviceSpec): The resolved device.
        workers (int): DataLoader worker count.
        pad_to (int | None): Fixed padding length, or `None`.
    """
    from zte.data.targets.tokens import build_target_tokens

    decoder = config.decoder
    texts = train_td.ordered_texts()
    targets = build_target_tokens(
        texts,
        decoder.tokenizer_source or decoder.lm_source,
        keys=_ordered_keys(train_td.text_vocab),
        revision=decoder.lm_revision,
        max_length=decoder.max_target_tokens,
        model_cache_dir=decoder.lm_cache_dir,
    )
    # Word counts per gallery sentence: they set the evidence pointer's walking rate and length-match the grounding
    # negatives, and they come from the reference text rather than from the reading, so no split sees the other's.
    n_words = torch.tensor([max(len(text.split()), 1) for text in texts], dtype=torch.long)

    # The gallery is whole-dataset by construction, so the rows this split actually reads have to be named or the
    # pointer's walking rate is averaged over the held-out stimuli too.
    train_ids = sorted({train_td.text_vocab[k] for k in train_td.stimulus_keys if k in train_td.text_vocab})
    objective.attach_tokens(
        torch.from_numpy(targets.ids), torch.from_numpy(targets.mask), n_words, rate_text_ids=train_ids
    )
    objective.attach_cache(train_td.n_readings, mode=config.train.mode)
    model.to(device.device)
    objective.to(device.device)

    warm_loader = make_dataloader(
        train_td,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.supports_pin_memory,
        pad_to=pad_to,
    )
    n_fit = objective.fit_gap(model, warm_loader, device.device)

    holdout_ids = sorted(set(train_td.text_vocab.values()) - set(train_ids))
    metrics = objective.pretrain_text(
        train_ids,
        holdout_text_ids=holdout_ids,
        epochs=decoder.stage0_epochs,
        lr=config.train.bridge_lr,
        batch_size=config.train.batch_size,
        seed=config.train.seed,
    )
    if config.model.grad_checkpoint and config.train.freeze_encoder:
        _LOG.info('model.grad_checkpoint is inert in this run: a frozen encoder produces no activations to recompute.')
    _LOG.info(
        'Decoder wired: %d target sentences (truncation %.1f%%, tokeniser %s), gap fitted on %d readings, '
        'stage 0 over %d train stimuli to CE %.4f.',
        len(targets),
        100.0 * targets.truncation_rate,
        targets.fingerprint,
        n_fit,
        len(train_ids),
        metrics['stage0_loss'],
    )


def _ordered_keys(vocab: dict[str, int]) -> list[str]:
    """Returns the stimulus keys in `text_vocab` id order, so row `i` names the text of `sentence_text_id == i`."""
    keys = [''] * len(vocab)
    for key, text_id in vocab.items():
        keys[text_id] = key
    return keys


def _shapes(dataset: ZuCoDataset, config: ZTEConfig) -> tuple[int | None, tuple[int, int] | None, int | None]:
    """Resolves frontend input shapes from the dataset and config.

    Raises:
        ValueError: If required tensors are missing for the chosen frontend.
    """
    in_dim = None if dataset.features is None else int(dataset.features.shape[1])
    raw_shape = None if dataset.raw_eeg is None else (int(dataset.raw_eeg.shape[1]), int(dataset.raw_eeg.shape[2]))
    if config.model.frontend == 'band_power_mlp' and in_dim is None:
        raise ValueError('band_power_mlp frontend needs band-power features in the dataset.')
    if config.model.frontend == 'raw_conformer' and raw_shape is None:
        raise ValueError('raw_conformer frontend needs raw EEG in the dataset.')
    # The masked-reconstruct head predicts the per-token input, so its width follows the frontend.
    recon_dim = in_dim
    if config.model.frontend == 'raw_conformer' and raw_shape is not None:
        recon_dim = raw_shape[0] * raw_shape[1]
    return in_dim, raw_shape, recon_dim


def _static_pad_length(
    setting: str,
    device: DeviceSpec,
    train_td: ZuCoTorchDataset,
    val_td: ZuCoTorchDataset | None,
) -> int | None:
    """Resolves the fixed padding length for static shapes, or `None` for per-batch padding.

    `auto` enables static shapes only on XLA/TPU, where dynamic shapes force recompilation. When active the
    length is the dataset-wide maximum sentence length, so no sentence is ever truncated.

    Args:
        setting (str): `'auto'`, `'on'` or `'off'`.
        device (DeviceSpec): The resolved device (its `kind` gates the `auto` decision).
        train_td (Any): The training torch dataset (exposes `.sequences`).
        val_td (Any): The validation torch dataset or `None`.

    Returns:
        int | None: The fixed pad length, or `None` to pad each batch to its own maximum.
    """
    active = setting == 'on' or (setting == 'auto' and device.kind == 'xla')
    if not active:
        return None
    lengths = [len(s) for s in train_td.sequences]
    if val_td is not None:
        lengths += [len(s) for s in val_td.sequences]
    return max(lengths) if lengths else None


def _channel_shape(
    dataset: ZuCoDataset, config: ZTEConfig, raw_shape: tuple[int, int] | None
) -> tuple[int | None, int | None]:
    """Resolves the EEG channel geometry needed for electrode spatial encoding.

    Args:
        dataset (ZuCoDataset): A built dataset.
        config (ZTEConfig): The run configuration.
        raw_shape (tuple[int, int] | None): `(n_channels, time_steps)` for the raw frontend.

    Returns:
        `(n_channels, bp_features_per_channel)`. Either value is `None` when it cannot be determined,
        which disables spatial encoding; the raw frontend always has `None` band-power width.
    """
    if config.model.frontend == 'raw_conformer':
        return (raw_shape[0] if raw_shape is not None else None), None
    bp = dataset.band_power_raw
    if bp is None:
        return None, None
    return int(bp.shape[2]), int(bp.shape[1])
