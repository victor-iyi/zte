"""`zte-evaluate` -- produce evidence that a trained ZTE encodes EEG well.

Loads a checkpoint + dataset, embeds words and sentences, and runs the evaluation suite (transfer probes vs raw features
and a noise control, geometry/health, and cross-subject content retrieval), writing figures, tables and a Markdown report.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from zte.cli.extract import load_dataset
from zte.cli.sources import add_data_source_args, add_extract_dir
from zte.data.dataset import ZuCoDataset
from zte.device import resolve_device
from zte.evaluation.report import evaluate_representation
from zte.inference.embed import ZTEEmbedder
from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.evaluate')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-evaluate` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.

    """
    parser = argparse.ArgumentParser(
        description='Evaluate a trained ZTE representation (probes, retrieval, geometry).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint (best.pt/last.pt).')
    add_data_source_args(parser, include_bundle=True)
    add_extract_dir(parser)

    parser.add_argument('--out', type=str, default='res/evaluation')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--run-name', type=str, default='zte-eval')
    parser.add_argument(
        '--montage-csv',
        type=str,
        default=None,
        help='Electrode-montage CSV (channel,region) for exact scalp-region importance; '
        'overrides the checkpoint config. Without it, an approximate region proxy is used.',
    )
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Write the full TensorBoard log (projector, hparams, scalars, figures).',
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Skip the self-contained interactive HTML embedding explorer.',
    )
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def collect_embeddings(
    embedder: ZTEEmbedder, dataset: ZuCoDataset, indices: np.ndarray | None = None
) -> tuple[
    np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray | None
]:
    """Produces aligned word/sentence embeddings + a raw-feature baseline.

    Word-level arrays are built in dataset row order over the present tokens so the ZTE embeddings, the raw-feature
    baseline and the metadata line up exactly. Sentence content ids group sentences by their stimulus text (so the same
    sentence read by different subjects shares an id).

    Args:
        embedder (ZTEEmbedder): The restored embedder.
        dataset (ZuCoDataset): A built dataset.
        indices (np.ndarray | None): Optional word-row indices restricting embedding
            to a split (e.g. the held-out test set); `None` embeds every present word.

    Returns:
        tuple: `(word_emb, word_meta, raw_word_feats, sent_emb, sent_content_ids, sent_meta, word_band_power)`,
            where `sent_meta` is the merged sentence metadata and `word_band_power` holds per-word band power
            (or `None` when the model consumes raw signals or no band power is available).

    """
    present = (
        np.ones(len(dataset.words), dtype=bool)
        if dataset.presence is None
        else dataset.presence.copy()
    )
    if indices is not None:
        in_split = np.zeros(len(dataset.words), dtype=bool)
        in_split[np.asarray(indices, dtype=int)] = True
        present = present & in_split
    if embedder.model.uses_raw:
        raw_windows = dataset.raw_eeg[present]  # type: ignore[index]
        word_emb = embedder.embed_signals(raw=raw_windows)
        raw_word_feats = raw_windows.reshape(len(raw_windows), -1)
    else:
        # dataset.features are already normalised and carry the exact model input
        # width (band power + any appended eye-tracking dims), so feed them straight
        # in with the normaliser disabled -- this stays aligned to `present` order
        # and is agnostic to the eye-tracking toggle.
        feats = dataset.features[present]  # type: ignore[index]
        word_emb = embedder.embed_signals(band_power=feats, apply_normalizer=False)
        raw_word_feats = feats
    word_meta = dataset.words.loc[present].reset_index(drop=True)

    sent_emb, sent_meta = embedder.embed(dataset, level='sentence', indices=indices)
    sent_cols = ['subject', 'task', 'sentence_idx', 'text']
    for extra in ('category', 'length_band'):
        if extra in dataset.sentences:
            sent_cols.append(extra)
    merged = sent_meta.merge(
        dataset.sentences[sent_cols], on=['subject', 'task', 'sentence_idx'], how='left'
    )
    sent_content_ids = pd.factorize(merged['text'])[0]
    # Per-word band power (present rows), aligned to word_emb, for region analysis.
    word_band_power = (
        None
        if embedder.model.uses_raw or dataset.band_power_raw is None
        else dataset.band_power_raw[present]
    )
    return word_emb, word_meta, raw_word_feats, sent_emb, sent_content_ids, merged, word_band_power


def phase_shuffled_word_emb(
    embedder: ZTEEmbedder, dataset: ZuCoDataset, indices: np.ndarray | None = None
) -> np.ndarray | None:
    """Embeds phase-scrambled EEG through the trained encoder (a signal-destroyed control).

    Passing FFT-phase-randomised raw EEG through the *same* trained encoder tests whether the encoder
    invents structure from destroyed signal. Only meaningful for a raw frontend: band-power features are
    (near-)phase-invariant, so scrambling raw EEG barely moves them -- for band-power models this returns
    `None` rather than presenting an identical-looking baseline as if it destroyed signal.

    Args:
        embedder (ZTEEmbedder): The restored embedder.
        dataset (ZuCoDataset): A built dataset.
        indices (np.ndarray | None): Optional word-row restriction (kept aligned with `present`).

    Returns:
        np.ndarray | None: Phase-scrambled ZTE word embeddings `(n, d)`, or `None` for band-power models.
    """
    if not embedder.model.uses_raw or dataset.raw_eeg is None:
        return None
    from zte.data.transforms import phase_scramble

    present = (
        np.ones(len(dataset.words), dtype=bool)
        if dataset.presence is None
        else dataset.presence.copy()
    )
    if indices is not None:
        in_split = np.zeros(len(dataset.words), dtype=bool)
        in_split[np.asarray(indices, dtype=int)] = True
        present = present & in_split
    scrambled = phase_scramble(dataset.raw_eeg[present], axis=-1)  # (n, channels, time)
    return embedder.embed_signals(raw=scrambled)


def training_vocab(dataset: ZuCoDataset, config: Any) -> set[str] | None:
    """Returns the set of word types in the training split (for the seen-vs-novel retrieval split).

    Recomputes the deterministic, seeded train/val/test split the run used and reads its word types, so
    a word can be labelled *novel* (never seen in training) at evaluation time.

    Args:
        dataset (ZuCoDataset): A built dataset.
        config (ZTEConfig): The run config (its `train` split settings are reused verbatim).

    Returns:
        set[str] | None: Training word types, or `None` if unavailable.
    """
    if 'word' not in dataset.words.columns:
        return None
    try:
        splits = dataset.split(
            config.train.split,
            val_fraction=config.train.val_fraction,
            test_fraction=config.train.test_fraction,
            holdout_subject=config.train.loso_holdout_subject,
            seed=config.train.seed,
        )
    except ValueError, KeyError:  # pragma: no cover - defensive
        return None
    train_idx = splits.get('train')
    if train_idx is None or len(train_idx) == 0:
        return None
    return set(dataset.words.iloc[np.asarray(train_idx, dtype=int)]['word'].astype(str))


def main() -> None:
    """Runs the evaluation end-to-end from the command line."""
    args = parse_arguments()
    configure_logging(args.log_level)

    dataset = load_dataset(args)
    embedder = ZTEEmbedder.from_checkpoint(args.ckpt, dataset, device=resolve_device(args.device))
    # A CLI montage overrides whatever the checkpoint carried, so exact scalp-region
    # importance can be requested at evaluation time (report.py loads it from config).
    if args.montage_csv is not None and getattr(embedder.config, 'dataset', None) is not None:
        embedder.config.dataset.montage_csv = args.montage_csv
    word_emb, word_meta, raw_feats, sent_emb, sent_ids, sent_meta, word_bp = collect_embeddings(
        embedder, dataset
    )
    obj_cfg = getattr(embedder.config, 'objective', None)
    phase_emb = (
        phase_shuffled_word_emb(embedder, dataset)
        if getattr(obj_cfg, 'eval_phase_shuffle', False)
        else None
    )
    train_vocab = (
        training_vocab(dataset, embedder.config)
        if getattr(obj_cfg, 'eval_seen_novel', False)
        else None
    )

    metrics = evaluate_representation(
        word_emb,
        word_meta,
        raw_feats,
        sent_emb,
        sent_ids,
        out_dir=args.out,
        run_name=args.run_name,
        sent_meta=sent_meta,
        word_band_power=word_bp,
        config=embedder.config,
        tensorboard=bool(args.tensorboard),
        interactive=not args.no_interactive,
        phase_word_emb=phase_emb,
        train_vocab=train_vocab,
    )
    _LOG.info('Verdict: %s', json.dumps(metrics['verdict']))
    _LOG.info('Report + figures written to %s', args.out)


if __name__ == '__main__':
    main()
