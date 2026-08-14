"""`zte-decode` -- free-running generation with its brain-independent controls, plus decoder-rescoring retrieval."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from zte.cli.support.io import write_json
from zte.cli.support.sources import add_data_source_args, add_extract_dir, dataset_for_config
from zte.config import ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.torch_dataset import ZuCoTorchDataset, build_subject_vocab
from zte.device import DeviceKind, resolve_device
from zte.evaluation.generation import generation_report, per_sentence_scores, tokenise
from zte.evaluation.report import missing_controls
from zte.inference.decode import ZTEDecoder
from zte.logging_utils import configure_logging, get_logger
from zte.training.checkpoint import CheckpointManager

_LOG = get_logger('cli.decode')

# Every control decodes through the identical path; only the conditioning vector or the prefix changes.
CONTROLS: tuple[str, ...] = ('mean_prefix', 'null_prefix', 'phase', 'noise', 'mismatch')
SPLITS: tuple[str, ...] = ('test', 'test_seen_stim', 'val', 'train')


@dataclass(slots=True)
class DecodeOptions:
    """Everything the generation evaluation needs beyond the checkpoint and the split.

    Attributes:
        controls (tuple[str, ...]): Brain-independent controls to decode.
        oracle (bool): Decode the true text embedding through the identical bridge as a positive control.
        beams (int | None): Beam width; `None` uses the checkpoint's.
        max_new_tokens (int | None): Free-running decode cap; `None` uses the checkpoint's.
        batch_size (int): Readings per encoder and decode call.
        n_perm (int): Permutations behind the generation null.
        n_boot (int): Resamples behind every paired-delta interval.
        rescore (bool): Also rank the sentence gallery by decoder likelihood, reported as retrieval.
        length_tol (int): Word-count tolerance for the stratified gallery and the mismatch derangement.
        mean_prefix_readings (int): Training readings averaged into the `mean_prefix` control.
        seed (int): Seed for the surrogates, the derangement, the bootstrap and the permutation null.
    """

    controls: tuple[str, ...] = CONTROLS
    oracle: bool = True
    beams: int | None = None
    max_new_tokens: int | None = None
    batch_size: int = 8
    n_perm: int = 1000
    n_boot: int = 2000
    rescore: bool = True
    length_tol: int = 1
    mean_prefix_readings: int = 512
    seed: int = 0


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-decode` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Decode text from a trained prefix bridge on a held-out split, against five '
        'brain-independent controls and a text oracle, and rank the sentence gallery by decoder likelihood.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ckpt', type=str, required=True, help='Decoder checkpoint (best.pt/last.pt).')
    add_data_source_args(parser, include_bundle=True, include_synthetic=True)
    add_extract_dir(parser)

    parser.add_argument(
        '--split',
        choices=list(SPLITS),
        default='test',
        help='Which cell to decode. `test` is unseen subject x unseen stimulus, the honest headline.',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=None,
        help="Output directory. Default: the run's own `evaluation/` beside `checkpoints/`.",
    )
    parser.add_argument(
        '--controls',
        type=str,
        default=None,
        help="Comma-separated controls. Default: the checkpoint's decoder.generation_controls.",
    )
    parser.add_argument(
        '--oracle',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Decode the true sentence embedding through the identical bridge (a positive control).',
    )
    parser.add_argument('--beams', type=int, default=None, help='Override decoder.beams.')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=None,
        dest='max_new_tokens',
        help='Override decoder.max_new_tokens. The reference length is never supplied.',
    )
    parser.add_argument('--batch-size', type=int, default=8, dest='batch_size')
    parser.add_argument(
        '--n-perm',
        type=int,
        default=None,
        dest='n_perm',
        help='Permutations for the generation null. Default: decoder.n_permutations.',
    )
    parser.add_argument('--n-boot', type=int, default=2000, dest='n_boot')
    parser.add_argument(
        '--rescore',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Rank the sentence gallery by decoder likelihood. Default: decoder.rescore_gallery.',
    )
    parser.add_argument('--length-tol', type=int, default=None, dest='length_tol')
    parser.add_argument(
        '--mean-prefix-readings',
        type=int,
        default=512,
        dest='mean_prefix_readings',
        help='Training readings averaged into the mean_prefix control (0 = the whole training split).',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--run-name', type=str, default=None, dest='run_name')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    return parser.parse_args()


def options_from_args(args: argparse.Namespace, config: ZTEConfig) -> DecodeOptions:
    """Merges the CLI flags over the checkpoint's own decoder configuration.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        config (ZTEConfig): The checkpoint's configuration.

    Returns:
        DecodeOptions: The resolved options.
    """
    decoder = config.decoder
    controls = (
        tuple(c.strip() for c in args.controls.split(',') if c.strip())
        if args.controls
        else tuple(decoder.generation_controls)
    )
    unknown = [c for c in controls if c not in CONTROLS]
    if unknown:
        raise ValueError(f'unknown control(s) {unknown}; expected a subset of {list(CONTROLS)}')
    return DecodeOptions(
        controls=controls,
        oracle=bool(args.oracle),
        beams=args.beams,
        max_new_tokens=args.max_new_tokens,
        batch_size=int(args.batch_size),
        n_perm=int(args.n_perm if args.n_perm is not None else decoder.n_permutations),
        n_boot=int(args.n_boot),
        rescore=bool(decoder.rescore_gallery if args.rescore is None else args.rescore),
        length_tol=int(args.length_tol if args.length_tol is not None else decoder.length_tol),
        mean_prefix_readings=int(args.mean_prefix_readings),
        seed=int(args.seed),
    )


def split_indices(dataset: ZuCoDataset, config: ZTEConfig, name: str) -> np.ndarray | None:
    """Recomputes the run's seeded split and returns one cell's word-row indices.

    Args:
        dataset (ZuCoDataset): A built dataset.
        config (ZTEConfig): The run configuration whose split settings are reused verbatim.
        name (str): The cell to return.

    Returns:
        np.ndarray | None: Word-row indices, or `None` when the strategy produces no such cell.
    """
    splits = dataset.split(
        config.train.split,
        val_fraction=config.train.val_fraction,
        test_fraction=config.train.test_fraction,
        holdout_subject=config.train.loso_holdout_subject,
        seed=config.train.seed,
    )
    idx = splits.get(name)
    if idx is None or len(idx) == 0:
        _LOG.warning('Split %r has no %r cell (it produces %s).', config.train.split, name, sorted(splits))
        return None
    return np.asarray(idx, dtype=int)


# ---- Controls ---- #


def phase_transform(seed: int = 0) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Returns a batch transform that phase-scrambles the raw windows, preserving their power spectrum."""
    from zte.data.features.transforms import phase_scramble

    def apply(batch: dict[str, Any]) -> dict[str, Any]:
        raw = batch.get('raw')
        if raw is None:
            return batch
        scrambled = phase_scramble(raw.detach().cpu().numpy(), axis=-1, seed=seed)
        return {**batch, 'raw': torch.from_numpy(scrambled).to(raw.device, raw.dtype)}

    return apply


def noise_transform(seed: int = 0) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Returns a batch transform replacing the signal with Gaussian noise of matched per-feature moments."""
    from zte.training.metrics import noise_matched

    def apply(batch: dict[str, Any]) -> dict[str, Any]:
        out = dict(batch)
        for key in ('raw', 'features'):
            tensor = batch.get(key)
            if tensor is None:
                continue
            arr = tensor.detach().cpu().numpy()
            flat = arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
            noise = noise_matched(flat, seed=seed).reshape(arr.shape)
            out[key] = torch.from_numpy(noise).to(tensor.device, tensor.dtype)
        return out

    return apply


def mismatch_partners(
    n_words: np.ndarray, content_ids: np.ndarray, *, length_tol: int = 1, seed: int = 0
) -> np.ndarray:
    """Maps each reading to a different reading of matched word count and different content.

    The pairing is length-stratified because word count carries several bits of sentence identity on ZuCo, so an
    unstratified mismatch control would be easier than the real decode for reasons that have nothing to do with the
    brain. Each row's partner is a genuinely different stimulus, so the control answers "which brain", not "any brain".

    Args:
        n_words (np.ndarray): Word count per reading `(n,)`.
        content_ids (np.ndarray): Stimulus id per reading `(n,)`.
        length_tol (int, optional): Half the tolerated word-count span inside a group. Defaults to 1.
        seed (int, optional): Shuffle seed. Defaults to 0.

    Returns:
        np.ndarray: Partner index per reading `(n,)`; the identity only when fewer than two readings exist.
    """
    lengths = np.asarray(n_words, dtype=np.float64).ravel()
    ids = np.asarray(content_ids).ravel()
    n = lengths.size
    partner = np.arange(n)
    if n < 2:
        return partner

    rng = np.random.default_rng(seed)
    span = 2.0 * max(int(length_tol), 0)
    groups: list[list[int]] = []
    current: list[int] = []
    for idx in np.argsort(lengths, kind='stable'):
        if current and lengths[idx] - lengths[current[0]] > span:
            groups.append(current)
            current = []
        current.append(int(idx))
    if current:
        groups.append(current)

    # A group of one cannot be deranged, so it joins its length neighbour rather than pairing with itself.
    merged: list[list[int]] = []
    for group in groups:
        if len(group) < 2 and merged:
            merged[-1].extend(group)
        else:
            merged.append(group)
    if len(merged) > 1 and len(merged[0]) < 2:
        merged[1] = merged[0] + merged[1]
        merged.pop(0)

    for group in merged:
        members = np.asarray(group)
        rng.shuffle(members)
        size = members.size
        if size < 2:
            continue
        for pos in range(size):
            row = int(members[pos])
            partner[row] = int(members[(pos + 1) % size])
            for offset in range(1, size):
                cand = int(members[(pos + offset) % size])
                if ids[cand] != ids[row]:
                    partner[row] = cand
                    break

    # A stratum holding one stimulus has no valid partner inside it, so those rows widen to the nearest length.
    for stranded in np.flatnonzero(ids[partner] == ids):
        row = int(stranded)
        elsewhere = np.flatnonzero(ids != ids[row])
        if elsewhere.size:
            partner[row] = int(elsewhere[np.argmin(np.abs(lengths[elsewhere] - lengths[row]))])
    return partner


# ---- The evaluation ---- #


def decode_evaluation(
    decoder: ZTEDecoder,
    dataset: ZuCoDataset,
    indices: np.ndarray | None,
    *,
    split: str,
    config: ZTEConfig,
    options: DecodeOptions | None = None,
    out_dir: Path | None = None,
    run_name: str = 'zte-decode',
) -> dict[str, Any]:
    """Decodes a split, runs every control through the identical path, and scores the whole thing honestly.

    No absolute score produced here is a result. The readable numbers are the paired deltas against each control, the
    permutation p and the prefix-influence KL; the gallery rescoring is reported separately and is labelled retrieval.

    Args:
        decoder (ZTEDecoder): A restored decoder checkpoint.
        dataset (ZuCoDataset): The built dataset the checkpoint was trained on.
        indices (np.ndarray | None): Word-row indices of the split to decode; `None` decodes everything.
        split (str): Cell being decoded; it and `config.train.split` are carried into the verdict, which
            headlines only the cell that generalises over the subject and the stimulus at once.
        config (ZTEConfig): The run configuration, for the split strategy and the text-encoder settings behind the
            oracle.
        options (DecodeOptions | None, optional): Decode options. Defaults to None, which uses the defaults.
        out_dir (Path | None, optional): Where `generation.jsonl` / `generation.json` / the interactive page are
            written. Defaults to None, which writes nothing.
        run_name (str, optional): Label for the interactive page. Defaults to 'zte-decode'.

    Returns:
        dict[str, Any]: `{'generation', 'rescoring', 'split', 'n', 'controls_unavailable', 'provenance'}`.
    """
    opts = options or DecodeOptions()
    z, meta = decoder.conditioning(dataset, indices, opts.batch_size)
    n = int(len(meta))
    if n == 0:
        return {'generation': {'applicable': False, 'reason': f'split {split!r} is empty'}, 'n': 0}

    references = [str(t) for t in meta['text']]
    _LOG.info('Decoding %d held-out readings from split %r ...', n, split)
    hypotheses = decoder.generate(z, max_new_tokens=opts.max_new_tokens, beams=opts.beams, batch_size=opts.batch_size)

    controls, unavailable = _controls(decoder, dataset, indices, z, meta, opts, config)
    gallery = _gallery_dataset(decoder, dataset)
    oracle = _oracle(decoder, gallery, meta, config, opts) if opts.oracle else None
    prefix_kl = decoder.prefix_influence_kl(z, batch_size=opts.batch_size)

    block = generation_report(
        hypotheses,
        references,
        controls,
        oracle=oracle,
        prefix_kl=float(np.mean(prefix_kl)) if prefix_kl.size else None,
        n_candidate_sentences=candidate_set_size(hypotheses, gallery.ordered_texts()),
        split=split,
        n_boot=opts.n_boot,
        n_perm=opts.n_perm,
        seed=opts.seed,
    )
    # The strategy names what the cell is held out from; the cell alone cannot say whether it shares stimuli.
    block['split_strategy'] = config.train.split
    block['controls_requested'] = list(opts.controls)
    block['controls_unavailable'] = unavailable
    block['prefix_influence_kl_median'] = float(np.median(prefix_kl)) if prefix_kl.size else float('nan')
    block['teacher_forced_ppl_DIAGNOSTIC'] = _teacher_forced_ppl(decoder, z, references, opts)

    rescoring = _rescoring(decoder, gallery, z, meta, opts) if opts.rescore and n >= 2 else None
    provenance = _provenance(decoder, config, split, opts, n)
    if out_dir is not None:
        _write_artifacts(
            out_dir,
            run_name,
            block,
            rescoring,
            provenance,
            meta,
            references,
            hypotheses,
            controls,
            oracle,
            prefix_kl,
            config.decoder.min_prefix_kl,
        )
    return {
        'generation': block,
        'rescoring': rescoring,
        'split': split,
        'n': n,
        'controls_unavailable': unavailable,
        'provenance': provenance,
    }


def _controls(
    decoder: ZTEDecoder,
    dataset: ZuCoDataset,
    indices: np.ndarray | None,
    z: np.ndarray,
    meta: pd.DataFrame,
    opts: DecodeOptions,
    config: ZTEConfig,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Decodes each brain-independent control through the same `generate` path the headline uses."""
    n = len(z)
    out: dict[str, list[str]] = {}
    unavailable: dict[str, str] = {}

    def free(prefix: torch.Tensor) -> list[str]:
        return decoder.generate_from_prefix(
            prefix,
            max_new_tokens=opts.max_new_tokens,
            beams=opts.beams,
            batch_size=opts.batch_size,
        )

    for name in opts.controls:
        _LOG.info('Control %r ...', name)
        if name == 'null_prefix':
            out[name] = free(decoder.null_prefix(n))
        elif name == 'mean_prefix':
            train = _train_conditioning(decoder, dataset, config, opts)
            if train is None:
                unavailable[name] = 'no training split to average'
                _LOG.warning('Control %r could not run: %s. It fails its verdict clause.', name, unavailable[name])
                continue
            out[name] = free(decoder.mean_prefix(train, n))
        elif name == 'mismatch':
            partner = mismatch_partners(
                meta['n_words'].to_numpy(),
                meta['text_id'].to_numpy(),
                length_tol=opts.length_tol,
                seed=opts.seed,
            )
            out[name] = decoder.generate(
                z[partner],
                max_new_tokens=opts.max_new_tokens,
                beams=opts.beams,
                batch_size=opts.batch_size,
            )
        else:
            surrogate = _surrogate_conditioning(decoder, dataset, indices, name, opts)
            if surrogate is None:
                unavailable[name] = 'the encoder consumes no raw signal to destroy'
                _LOG.warning('Control %r could not run: %s. It fails its verdict clause.', name, unavailable[name])
                continue
            out[name] = decoder.generate(
                surrogate,
                max_new_tokens=opts.max_new_tokens,
                beams=opts.beams,
                batch_size=opts.batch_size,
            )
    return out, unavailable


def _surrogate_conditioning(
    decoder: ZTEDecoder,
    dataset: ZuCoDataset,
    indices: np.ndarray | None,
    name: str,
    opts: DecodeOptions,
) -> np.ndarray | None:
    """Runs the phase / noise surrogate signals through the identical frozen encoder."""
    if name == 'phase' and not decoder.model.uses_raw:
        return None
    transform = phase_transform(opts.seed) if name == 'phase' else noise_transform(opts.seed)
    surrogate, _ = decoder.conditioning(dataset, indices, opts.batch_size, transform=transform)
    return surrogate


def _train_conditioning(
    decoder: ZTEDecoder, dataset: ZuCoDataset, config: ZTEConfig, opts: DecodeOptions
) -> np.ndarray | None:
    """Embeds a capped sample of the training split, whose mean vector is the `mean_prefix` control."""
    train = split_indices(dataset, config, 'train')
    if train is None:
        return None
    z, _ = decoder.conditioning(dataset, train, opts.batch_size)
    if len(z) == 0:
        return None
    cap = opts.mean_prefix_readings
    if 0 < cap < len(z):
        rows = np.random.default_rng(opts.seed).choice(len(z), size=cap, replace=False)
        z = z[np.sort(rows)]
    return z


def _oracle(
    decoder: ZTEDecoder,
    gallery: ZuCoTorchDataset,
    meta: pd.DataFrame,
    config: ZTEConfig,
    opts: DecodeOptions,
) -> list[str] | None:
    """Decodes the true sentence embedding through the identical bridge and LM: what the head can do at all."""
    from zte.data.targets.text import build_sentence_text_matrix

    matrix, dim = build_sentence_text_matrix(
        gallery.ordered_texts(),
        config.objective.text_source,
        backend=config.objective.text_backend,
        prefix=config.objective.text_query_prefix,
        device=str(decoder.device.device),
    )
    if matrix is None or dim != decoder.z_dim:
        _LOG.warning(
            'Text oracle skipped: objective.text_source=%r yields dim %s against a %d-wide bridge input.',
            config.objective.text_source,
            dim,
            decoder.z_dim,
        )
        return None
    rows = meta['text_id'].to_numpy()
    if int((rows < 0).sum()):
        _LOG.warning('Text oracle skipped: %d readings carry no text id.', int((rows < 0).sum()))
        return None
    prefix = decoder.prefix_from_z(matrix[rows])
    return decoder.generate_from_prefix(
        prefix, max_new_tokens=opts.max_new_tokens, beams=opts.beams, batch_size=opts.batch_size
    )


def _teacher_forced_ppl(decoder: ZTEDecoder, z: np.ndarray, references: list[str], opts: DecodeOptions) -> float:
    """Mean teacher-forced perplexity of the references -- quarantined, and provably unread by any verdict."""
    nll = decoder.teacher_forced_nll(z, references, batch_size=opts.batch_size)
    return float(np.exp(np.mean(nll))) if nll.size else float('nan')


def _gallery_dataset(decoder: ZTEDecoder, dataset: ZuCoDataset) -> ZuCoTorchDataset:
    """Builds the whole-dataset view whose `text_vocab` ids every batch already carries."""
    vocab = decoder.subject_vocab or build_subject_vocab(dataset)
    return ZuCoTorchDataset(dataset, subject_vocab=vocab)


def candidate_set_size(hypotheses: list[str], gallery: Sequence[str]) -> int | None:
    """Sizes the candidate set a decode chose from, reading it back off the decodes themselves.

    Free generation writes tokens and a constrained decode picks a row, so a run whose every hypothesis is
    a gallery sentence is a forced choice and is scored as retrieval -- with no declaration to remember and
    nothing for a later constrained decode to slip past.

    Args:
        hypotheses (list[str]): The free-running decodes.
        gallery (Sequence[str]): Every sentence a constrained decode could have picked.

    Returns:
        int | None: Distinct gallery sentences when all decodes are gallery sentences, else `None`.
    """
    known = {' '.join(tokenise(text)) for text in gallery}
    known.discard('')
    if not hypotheses or not known:
        return None
    if any(' '.join(tokenise(h)) not in known for h in hypotheses):
        return None
    return len(known)


def _rescoring(
    decoder: ZTEDecoder,
    gallery: ZuCoTorchDataset,
    z: np.ndarray,
    meta: pd.DataFrame,
    opts: DecodeOptions,
) -> dict[str, Any] | None:
    """Ranks the sentence gallery by decoder likelihood. This is RETRIEVAL and is named so everywhere."""
    from zte.evaluation.audit.scoreboard import decoder_rescoring_retrieval

    texts = gallery.ordered_texts()
    if len(texts) < 2:
        return None
    scores = decoder.rescore(z, texts, batch_size=opts.batch_size)
    block = decoder_rescoring_retrieval(
        scores,
        meta['text_id'].to_numpy(),
        np.arange(len(texts)),
        query_n_words=meta['n_words'].to_numpy(),
        gallery_n_words=_gallery_lengths(gallery, len(texts)),
        length_tol=opts.length_tol,
        seed=opts.seed,
    )
    if block is not None:
        block['n_candidate_sentences'] = len(texts)
    return block


def _gallery_lengths(torch_ds: ZuCoTorchDataset, n_text: int) -> np.ndarray:
    """Median read word count per gallery sentence, in the same units as the queries' `n_words`."""
    per_text: dict[int, list[int]] = {}
    vocab = torch_ds.text_vocab
    for i, key in enumerate(torch_ds.stimulus_keys):
        text_id = vocab.get(key, -1)
        if text_id >= 0:
            per_text.setdefault(text_id, []).append(len(torch_ds.sequences[i]))
    lengths = np.zeros(n_text, dtype=np.float64)
    for text_id, values in per_text.items():
        lengths[text_id] = float(np.median(values))
    return lengths


def _provenance(decoder: ZTEDecoder, config: ZTEConfig, split: str, opts: DecodeOptions, n: int) -> dict[str, Any]:
    """Records the LM, tokeniser, device, seeds and code state a decode cannot be reproduced without."""
    from zte.utils.provenance import git_info, package_versions

    git = git_info()
    lm = getattr(decoder.lm, 'provenance', None)
    record: dict[str, Any] = {
        'run_name': config.run_name,
        'split': split,
        'split_strategy': config.train.split,
        'controls': list(opts.controls),
        'n_readings': int(n),
        'device': decoder.device.name,
        'device_kind': decoder.device.kind,
        'beams': opts.beams or decoder.decoder_config.beams,
        'max_new_tokens': opts.max_new_tokens or decoder.decoder_config.max_new_tokens,
        'cfg_weight': decoder.decoder_config.cfg_weight,
        'conditioning': decoder.decoder_config.conditioning,
        'gap_correction': decoder.decoder_config.gap_correction,
        'gap_fitted': bool(decoder.gap.fitted),
        'gap_n_fit': int(decoder.gap.n_fit),
        # The decoder whitens nothing, but it inherits three train-fitted transforms, and the field exists so a
        # number is never read without knowing what was fitted on what.
        'postprocess_fit': 'none',
        'normalizer_fit': decoder.normalizer is not None,
        'aligner_fit': decoder.aligner is not None,
        'text_source': config.objective.text_source,
        'seed': opts.seed,
        'n_perm': opts.n_perm,
        'n_boot': opts.n_boot,
        'length_tol': opts.length_tol,
        'git_commit': git['commit'],
        'git_dirty': git['dirty'],
        'torch': torch.__version__,
        'lm': lm() if callable(lm) else lm,
    }
    # `torch.__version__` keeps the build suffix (`+cu121`) that the distribution version drops.
    record.update({k: v for k, v in package_versions().items() if v is not None and k != 'torch'})
    return record


def _write_artifacts(
    out_dir: Path,
    run_name: str,
    block: dict[str, Any],
    rescoring: dict[str, Any] | None,
    provenance: dict[str, Any],
    meta: pd.DataFrame,
    references: list[str],
    hypotheses: list[str],
    controls: dict[str, list[str]],
    oracle: list[str] | None,
    prefix_kl: np.ndarray,
    min_prefix_kl: float,
) -> None:
    """Writes the per-sentence side-by-side, the scored block and the offline HTML page."""
    from zte.evaluation.interactive import generation_html

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / 'generation.json',
        {'generation': block, 'rescoring': rescoring, 'provenance': provenance},
        default=str,
    )
    lines = _jsonl_rows(meta, references, hypotheses, controls, oracle, prefix_kl)
    (out_dir / 'generation.jsonl').write_text(
        ''.join(json.dumps(row, default=str) + '\n' for row in lines), encoding='utf-8'
    )
    try:
        generation_html(
            block,
            out_dir / 'interactive' / 'generation.html',
            run_name=run_name,
            min_prefix_kl=min_prefix_kl,
        )
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        _LOG.warning('Interactive generation page skipped: %r', exc)
    _LOG.info('Wrote generation.jsonl, generation.json and interactive/generation.html to %s', out_dir)


def _jsonl_rows(
    meta: pd.DataFrame,
    references: list[str],
    hypotheses: list[str],
    controls: dict[str, list[str]],
    oracle: list[str] | None,
    prefix_kl: np.ndarray,
) -> list[dict[str, Any]]:
    """Builds one record per held-out reading: reference, hypothesis, every control and the oracle, all scored."""
    hyp_scores = per_sentence_scores(hypotheses, references)
    control_scores = {n: per_sentence_scores(t, references) for n, t in controls.items()}
    oracle_scores = per_sentence_scores(oracle, references) if oracle is not None else None
    columns = [c for c in ('subject', 'task', 'sentence_idx', 'n_words', 'stimulus_key') if c in meta]

    rows: list[dict[str, Any]] = []
    for i in range(len(references)):
        row: dict[str, Any] = {c: meta.iloc[i][c] for c in columns}
        row['index'] = i
        row['reference'] = references[i]
        row['hypothesis'] = hypotheses[i]
        row['scores'] = {m: float(v[i]) for m, v in hyp_scores.items()}
        row['prefix_influence_kl'] = float(prefix_kl[i]) if i < prefix_kl.size else None
        row['controls'] = {
            name: {
                'text': texts[i],
                'scores': {m: float(v[i]) for m, v in control_scores[name].items()},
            }
            for name, texts in controls.items()
        }
        if oracle is not None and oracle_scores is not None:
            row['oracle'] = {
                'text': oracle[i],
                'scores': {m: float(v[i]) for m, v in oracle_scores.items()},
            }
        rows.append(row)
    return rows


# ---- Reused by zte-evaluate and zte-run ---- #


def decoder_blocks(
    ckpt: str | Path,
    dataset: ZuCoDataset,
    config: ZTEConfig,
    *,
    out_dir: Path | None = None,
    device: DeviceKind | Literal['auto'] = 'auto',
    split: str | None = None,
    options: DecodeOptions | None = None,
    run_name: str = 'zte-eval',
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Returns `(generation, rescoring)` for a decoder checkpoint, or `(None, None)` for an encoder run.

    This is the hook `zte-evaluate` and `zte-run` call behind `objective.eval_generation` /
    `objective.eval_rescoring`, so the report, the scoreboard and the verdict see exactly what `zte-decode` writes.

    Args:
        ckpt (str | Path): The checkpoint to decode from.
        dataset (ZuCoDataset): The built dataset.
        config (ZTEConfig): The run configuration.
        out_dir (Path | None, optional): Where the generation artifacts go. Defaults to None.
        device (DeviceKind | Literal['auto'], optional): Device selector. Defaults to 'auto'.
        split (str | None, optional): Cell to decode. Defaults to None, which picks `test` then `val`.
        options (DecodeOptions | None, optional): Decode options. Defaults to None.
        run_name (str, optional): Label for the interactive page. Defaults to 'zte-eval'.

    Returns:
        tuple[dict | None, dict | None]: The generation block and the rescoring block.
    """
    obj = config.objective
    want_generation = bool(getattr(obj, 'eval_generation', False))
    want_rescoring = bool(getattr(obj, 'eval_rescoring', False))
    if config.train.mode == 'encoder' or not (want_generation or want_rescoring):
        return None, None
    try:
        decoder = ZTEDecoder.from_checkpoint(ckpt, dataset, device=resolve_device(device))
    except ValueError as exc:
        _LOG.info('Generation eval skipped: %s', exc)
        return None, None

    opts = options or DecodeOptions(
        n_perm=config.decoder.n_permutations,
        rescore=want_rescoring,
        length_tol=config.decoder.length_tol,
    )
    for name in (split, 'test', 'val') if split else ('test', 'val'):
        indices = split_indices(dataset, config, name)
        if indices is not None:
            result = decode_evaluation(
                decoder,
                dataset,
                indices,
                split=name,
                config=config,
                options=opts,
                out_dir=out_dir,
                run_name=run_name,
            )
            return (
                result.get('generation') if want_generation else None,
                result.get('rescoring'),
            )
    _LOG.warning('Generation eval skipped: the run produces no held-out cell to decode.')
    return None, None


def main() -> None:
    """Decodes a held-out split with all its controls and writes the generation artifacts."""
    args = parse_arguments()
    configure_logging(args.log_level)

    payload = CheckpointManager.load(args.ckpt, map_location='cpu')
    config = ZTEConfig.from_dict(payload['config'])
    options = options_from_args(args, config)
    dataset = dataset_for_config(args, config.dataset)

    decoder = ZTEDecoder.from_checkpoint(args.ckpt, dataset, device=resolve_device(args.device))
    out_dir = Path(args.out) if args.out else Path(args.ckpt).resolve().parent.parent / 'evaluation'
    result = decode_evaluation(
        decoder,
        dataset,
        split_indices(dataset, config, args.split),
        split=args.split,
        config=config,
        options=options,
        out_dir=out_dir,
        run_name=args.run_name or config.run_name,
    )

    block = result.get('generation') or {}
    if not block.get('applicable'):
        _LOG.warning('Generation not scoreable: %s', block.get('reason'))
        return
    # `beats_all_controls` counts only controls that produced a delta, so a control that never ran is not a control
    # it beat. The summary reports the same composition the verdict gates on.
    missing = missing_controls(block, str(block.get('primary_metric')))
    worst = block.get('worst_control_ci') or {}
    _LOG.info(
        'n=%d | %s delta vs the worst surviving control (%s): %.4f [%.4f, %.4f] | beats all controls: %s',
        block.get('n', 0),
        block.get('primary_metric'),
        block.get('worst_control'),
        float(worst.get('point', float('nan'))),
        float(worst.get('lo', float('nan'))),
        float(worst.get('hi', float('nan'))),
        not missing,
    )
    if missing:
        _LOG.warning('Controls not beaten (a control that did not run fails its clause): %s', ', '.join(missing))
    _LOG.info(
        'permutation p=%s | prefix-influence KL=%s nats (floor %s)',
        (block.get('permutation') or {}).get('p_value'),
        block.get('prefix_influence_kl'),
        config.decoder.min_prefix_kl,
    )
    rescoring = result.get('rescoring')
    if rescoring:
        _LOG.info(
            'Decoder-rescoring RETRIEVAL over %s candidates, UNSTRATIFIED: Top-1 %.4f (chance %.4f), '
            'rank percentile %.4f.',
            rescoring.get('n_candidate_sentences'),
            rescoring.get('top1', float('nan')),
            rescoring.get('chance_top1', float('nan')),
            rescoring.get('rank_percentile', float('nan')),
        )
        # Sentence length alone carries 5.14 bits of identity on ZuCo and beats the encoder on every top-k, so an
        # unstratified top-k quoted on its own is not evidence of decoding.
        stratified = rescoring.get('length_stratified') or {}
        if stratified:
            _LOG.info(
                'Length-stratified (+/-%s words): Top-1 %.4f (chance %.4f), rank percentile %.4f over %s queries.',
                options.length_tol,
                stratified.get('top1', float('nan')),
                stratified.get('chance_top1', float('nan')),
                stratified.get('rank_percentile', float('nan')),
                stratified.get('n_queries'),
            )
        else:
            _LOG.warning('No length-stratified cell was computed; the top-k above is not evidence of decoding.')


if __name__ == '__main__':
    main()
