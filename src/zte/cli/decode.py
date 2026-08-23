"""`zte-decode` -- free-running generation with its brain-independent controls, plus decoder-rescoring retrieval."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
import torch

from zte.cli.support.done import add_force_argument, checkpoint_digest, is_done, mark_done, signature
from zte.cli.support.io import read_json, write_json
from zte.cli.support.sources import add_data_source_args, add_extract_dir, dataset_for_config, dataset_key
from zte.config import ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.torch_dataset import ZuCoTorchDataset, build_subject_vocab
from zte.device import DeviceKind, resolve_device
from zte.evaluation.audit.capacity import CAPACITY_KS, DEFAULT_N_PERM, HEADLINE_FLAVOR, HEADLINE_SCORE
from zte.evaluation.generation import generation_report, per_sentence_scores, tokenise
from zte.evaluation.report import missing_controls
from zte.inference.capacity import GalleryScores, capacity_arms, gallery_scores
from zte.inference.decode import ReadingBatch, ZTEDecoder, paired_shuffle
from zte.logging_utils import configure_logging, get_logger
from zte.training.checkpoint import CheckpointManager

_LOG = get_logger('cli.decode')

# Every control decodes through the identical path; only the conditioning vector or the prefix changes.
CONTROLS: tuple[str, ...] = ('mean_prefix', 'null_prefix', 'phase', 'noise', 'shuffled_z', 'length_only', 'mismatch')
SPLITS: tuple[str, ...] = ('test', 'test_seen_stim', 'val', 'train')

# The capacity gallery rows are stimulus prototypes rather than readings, so they need a subject label that no
# real subject code can collide with; it is what marks them as the reference side of the audit.
GALLERY_ROW: Final[str] = '<gallery>'
"""Subject label of the prototype rows that define the capacity audit's gallery."""

# A capacity that did not certify is a real, reportable outcome; rendered as a blank or a zero it reads as a
# number, so the one thing it must never look like is a measurement.
EM_DASH: Final[str] = '—'
"""What an uncertified menu size and its bits are printed as."""


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
        within_task_pools (tuple[str, ...]): Tasks whose candidate pool is also reported on its own.
        capacity (bool): Certify the largest K-way menu the decoder serves, sliced out of the gallery pass.
        capacity_ks (tuple[int, ...]): Menu sizes the certification sweeps.
        capacity_alpha (float): Significance level of every certification clause.
        capacity_n_perm (int): Label permutations behind each per-K p-value.
        capacity_score (str): Score families to certify -- `pmi`, `raw` or `both`.
        seeds (tuple[int, ...]): Extra decode seeds re-run for a mean +/- sd headline.
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
    within_task_pools: tuple[str, ...] = ('SR', 'NR')
    capacity: bool = False
    capacity_ks: tuple[int, ...] = CAPACITY_KS
    capacity_alpha: float = 0.05
    capacity_n_perm: int = DEFAULT_N_PERM
    capacity_score: str = HEADLINE_SCORE
    seeds: tuple[int, ...] = field(default_factory=tuple)
    seed: int = 0


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-decode` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Decode text from a trained prefix bridge on a held-out split, against every '
        'brain-independent control and a text oracle, and rank the sentence gallery by decoder likelihood.',
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
        '--within-task',
        type=str,
        default=None,
        dest='within_task',
        help="Comma-separated tasks whose candidate pool is also reported alone (e.g. 'SR,NR'), or '' to skip.",
    )
    parser.add_argument(
        '--capacity',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Certify the largest K-way menu the decoder serves. Default: objective.eval_capacity.',
    )
    parser.add_argument(
        '--capacity-ks',
        type=str,
        default=None,
        dest='capacity_ks',
        help="Comma-separated menu sizes to sweep. Default: decoder.capacity_ks. A size the gallery's "
        'word-count pools cannot fill is reported as unreachable, never silently dropped.',
    )
    parser.add_argument(
        '--capacity-alpha',
        type=float,
        default=None,
        dest='capacity_alpha',
        help='Significance level of every certification clause. Default: decoder.capacity_alpha.',
    )
    parser.add_argument(
        '--capacity-n-perm',
        type=int,
        default=None,
        dest='capacity_n_perm',
        help='Permutations behind each per-K p-value; the attainable floor is 1/(n+1). '
        'Default: decoder.capacity_n_perm.',
    )
    parser.add_argument(
        '--seeds',
        type=str,
        default=None,
        help='Comma-separated extra decode seeds; every headline is then reported as mean +/- sd across them.',
    )
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
    add_force_argument(parser)
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    return parser.parse_args()


def options_from_args(args: argparse.Namespace, config: ZTEConfig) -> DecodeOptions:
    """Merges the CLI flags over the checkpoint's own decoder configuration.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        config (ZTEConfig): The checkpoint's configuration.

    Returns:
        DecodeOptions: The resolved options.

    Raises:
        ValueError: If a named control is not one this evaluation knows how to decode.
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
    within = getattr(args, 'within_task', None)
    pools = (
        tuple(t.strip() for t in within.split(',') if t.strip())
        if within is not None
        else tuple(decoder.within_task_pools)
    )
    seeds = getattr(args, 'seeds', None)
    capacity = getattr(args, 'capacity', None)
    capacity_ks = getattr(args, 'capacity_ks', None)
    capacity_alpha = getattr(args, 'capacity_alpha', None)
    capacity_n_perm = getattr(args, 'capacity_n_perm', None)
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
        within_task_pools=pools,
        capacity=bool(config.objective.eval_capacity if capacity is None else capacity),
        capacity_ks=(
            tuple(int(k) for k in capacity_ks.split(',') if k.strip()) if capacity_ks else tuple(decoder.capacity_ks)
        ),
        capacity_alpha=float(capacity_alpha if capacity_alpha is not None else decoder.capacity_alpha),
        capacity_n_perm=int(capacity_n_perm if capacity_n_perm is not None else decoder.capacity_n_perm),
        capacity_score=str(decoder.capacity_score),
        seeds=tuple(int(s) for s in seeds.split(',') if s.strip()) if seeds else tuple(decoder.eval_seeds),
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
        dict[str, Any]: `{'generation', 'rescoring', 'capacity', 'bit_budget', 'split', 'n',
            'controls_unavailable', 'provenance'}`.
    """
    opts = options or DecodeOptions()
    readings = decoder.conditioning(dataset, indices, opts.batch_size)
    n = len(readings)
    if n == 0:
        return {'generation': {'applicable': False, 'reason': f'split {split!r} is empty'}, 'n': 0}

    references = [str(t) for t in readings.meta['text']]
    _LOG.info('Decoding %d held-out readings from split %r (strictly autoregressive, no reference) ...', n, split)
    hypotheses = decoder.generate(
        readings, max_new_tokens=opts.max_new_tokens, beams=opts.beams, batch_size=opts.batch_size
    )

    # The capacity audit's `length_only` arm is built from the training split, so it is embedded up front and the
    # control layer reuses it rather than paying for a second pass over the same readings.
    train = _train_conditioning(decoder, dataset, config, opts) if opts.capacity else None
    controls, unavailable = _controls(decoder, dataset, indices, readings, opts, config, train=train)
    gallery = _gallery_dataset(decoder, dataset)
    oracle = _oracle(decoder, gallery, readings, config, opts) if opts.oracle else None
    prefix_kl = decoder.prefix_influence_kl(readings.z, batch_size=opts.batch_size)

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
    block['teacher_forced_ppl_DIAGNOSTIC'] = _teacher_forced_ppl(decoder, readings, references, opts)
    # Stated in the artifact rather than in a reader's assumption: no reference token entered the decode loop.
    block['teacher_forced'] = False
    block['decode_strategy'] = 'greedy'
    if opts.seeds:
        block['seed_spread'] = _seed_spread(decoder, dataset, indices, readings, references, hypotheses, opts, config)

    # One gallery pass feeds both readouts: scoring is per-(query, candidate), so every K-way menu the capacity
    # audit certifies is a column slice of the same matrix retrieval ranks.
    rescore_bundle = (
        _gallery_bundle(decoder, dataset, gallery, readings, opts, evidence_content=True)
        if opts.rescore and n >= 2
        else None
    )
    rescoring = _rescoring(decoder, readings, rescore_bundle, opts) if rescore_bundle is not None else None
    capacity = (
        _capacity(decoder, dataset, gallery, readings, train, opts, config, split, bundle=rescore_bundle)
        if opts.capacity and n >= 2
        else None
    )

    provenance = _provenance(decoder, config, split, opts, n)
    result = {
        'generation': block,
        'rescoring': rescoring,
        'capacity': capacity,
        'bit_budget': decoder.bit_report(readings),
        'split': split,
        'n': n,
        'controls_unavailable': unavailable,
        'provenance': provenance,
    }
    if out_dir is not None:
        _write_artifacts(
            out_dir,
            run_name,
            result,
            readings.meta,
            references,
            hypotheses,
            controls,
            oracle,
            prefix_kl,
            config.decoder.min_prefix_kl,
        )
    return result


def _controls(
    decoder: ZTEDecoder,
    dataset: ZuCoDataset,
    indices: np.ndarray | None,
    readings: ReadingBatch,
    opts: DecodeOptions,
    config: ZTEConfig,
    *,
    train: ReadingBatch | None = None,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Decodes each brain-independent control through the same `generate` path the headline uses."""
    n = len(readings)
    out: dict[str, list[str]] = {}
    unavailable: dict[str, str] = {}

    def free_prefix(prefix: torch.Tensor, *, content: bool = True) -> list[str]:
        return decoder.generate_from_prefix(
            prefix,
            readings=readings,
            max_new_tokens=opts.max_new_tokens,
            beams=opts.beams,
            batch_size=opts.batch_size,
            evidence_content=content,
        )

    def free(batch: ReadingBatch) -> list[str]:
        return decoder.generate(batch, max_new_tokens=opts.max_new_tokens, beams=opts.beams, batch_size=opts.batch_size)

    for name in opts.controls:
        _LOG.info('Control %r ...', name)
        if name == 'null_prefix':
            out[name] = free_prefix(decoder.null_prefix(n), content=False)
        elif name in {'mean_prefix', 'length_only'}:
            train = train if train is not None else _train_conditioning(decoder, dataset, config, opts)
            if train is None:
                unavailable[name] = 'no training split to average'
                _LOG.warning('Control %r could not run: %s. It fails its verdict clause.', name, unavailable[name])
                continue
            if name == 'mean_prefix':
                out[name] = free_prefix(decoder.mean_prefix(train.z, n), content=False)
            else:
                matched = decoder.length_matched_z(
                    train.z,
                    train.meta['n_words'].to_numpy(),
                    readings.meta['n_words'].to_numpy(),
                    tol=opts.length_tol,
                )
                out[name] = free_prefix(decoder.prefix_from_z(matched), content=False)
        elif name == 'shuffled_z':
            out[name] = free(readings.take(paired_shuffle(n, opts.seed)))
        elif name == 'mismatch':
            partner = mismatch_partners(
                readings.meta['n_words'].to_numpy(),
                readings.meta['text_id'].to_numpy(),
                length_tol=opts.length_tol,
                seed=opts.seed,
            )
            out[name] = free(readings.take(partner))
        else:
            surrogate = _surrogate_conditioning(decoder, dataset, indices, name, opts)
            if surrogate is None:
                unavailable[name] = 'the encoder consumes no raw signal to destroy'
                _LOG.warning('Control %r could not run: %s. It fails its verdict clause.', name, unavailable[name])
                continue
            out[name] = free(surrogate)
    return out, unavailable


def _surrogate_conditioning(
    decoder: ZTEDecoder,
    dataset: ZuCoDataset,
    indices: np.ndarray | None,
    name: str,
    opts: DecodeOptions,
) -> ReadingBatch | None:
    """Runs the phase / noise surrogate signals through the identical frozen encoder."""
    if name == 'phase' and not decoder.model.uses_raw:
        return None
    transform = phase_transform(opts.seed) if name == 'phase' else noise_transform(opts.seed)
    return decoder.conditioning(dataset, indices, opts.batch_size, transform=transform)


def _train_conditioning(
    decoder: ZTEDecoder, dataset: ZuCoDataset, config: ZTEConfig, opts: DecodeOptions
) -> ReadingBatch | None:
    """Embeds a capped sample of the training split, whose mean vector is the `mean_prefix` control."""
    train = split_indices(dataset, config, 'train')
    if train is None:
        return None
    batch = decoder.conditioning(dataset, train, opts.batch_size)
    if len(batch) == 0:
        return None
    cap = opts.mean_prefix_readings
    if 0 < cap < len(batch):
        rows = np.sort(np.random.default_rng(opts.seed).choice(len(batch), size=cap, replace=False))
        return ReadingBatch(z=batch.z[rows], meta=batch.meta.iloc[rows].reset_index(drop=True))
    return batch


def _seed_spread(
    decoder: ZTEDecoder,
    dataset: ZuCoDataset,
    indices: np.ndarray | None,
    readings: ReadingBatch,
    references: list[str],
    hypotheses: list[str],
    opts: DecodeOptions,
    config: ZTEConfig,
) -> dict[str, Any]:
    """Re-runs the control layer at each extra seed and reports the headline delta as mean +/- sd.

    Note:
        The headline decode is greedy and therefore identical at every seed; what moves is the *controls* -- the
        surrogate signals, the derangements, the bootstrap and the permutation null. So this is an error bar on the
        comparison, which is the number the verdict reads, and not on the hypothesis text.
    """
    points: list[float] = []
    seeds = tuple(dict.fromkeys((opts.seed, *opts.seeds)))
    for seed in seeds:
        per_seed = replace(opts, seed=seed, seeds=())
        controls, _ = _controls(decoder, dataset, indices, readings, per_seed, config)
        block = generation_report(
            hypotheses,
            references,
            controls,
            n_boot=opts.n_boot,
            n_perm=opts.n_perm,
            seed=seed,
        )
        worst = block.get('worst_control_ci') or {}
        points.append(float(worst.get('point', float('nan'))))

    values = np.asarray(points, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        'seeds': list(seeds),
        'metric': 'worst_control_delta',
        'values': [float(v) for v in values],
        'mean': float(finite.mean()) if finite.size else float('nan'),
        'sd': float(finite.std(ddof=1)) if finite.size > 1 else float('nan'),
        'n_seeds': int(finite.size),
    }


def _oracle(
    decoder: ZTEDecoder,
    gallery: ZuCoTorchDataset,
    readings: ReadingBatch,
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
    rows = readings.meta['text_id'].to_numpy()
    if int((rows < 0).sum()):
        _LOG.warning('Text oracle skipped: %d readings carry no text id.', int((rows < 0).sum()))
        return None

    # The pooled path only: the oracle bounds what the bridge can write from a perfect sentence vector, and its
    # word slots are zeroed so it never borrows lexical evidence the EEG path would have had to earn.
    prefix = decoder.prefix_from_z(matrix[rows])
    return decoder.generate_from_prefix(
        prefix,
        readings=readings,
        max_new_tokens=opts.max_new_tokens,
        beams=opts.beams,
        batch_size=opts.batch_size,
        evidence_content=False,
    )


def _teacher_forced_ppl(
    decoder: ZTEDecoder, readings: ReadingBatch, references: list[str], opts: DecodeOptions
) -> float:
    """Mean teacher-forced perplexity of the references -- quarantined, and provably unread by any verdict."""
    nll = decoder.teacher_forced_nll(readings, references, batch_size=opts.batch_size)
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


def _gallery_bundle(
    decoder: ZTEDecoder,
    dataset: ZuCoDataset,
    gallery: ZuCoTorchDataset,
    readings: ReadingBatch,
    opts: DecodeOptions,
    *,
    evidence_content: bool,
) -> GalleryScores | None:
    """Scores every gallery sentence under every reading once, with the stimulus-level length and task labels."""
    texts = gallery.ordered_texts()
    if len(texts) < 2:
        return None

    return gallery_scores(
        decoder,
        readings,
        texts,
        gallery_n_words=_gallery_lengths(gallery, len(texts)),
        gallery_tasks=_gallery_tasks(dataset, gallery, len(texts)),
        batch_size=opts.batch_size,
        evidence_content=evidence_content,
    )


def _rescoring(
    decoder: ZTEDecoder,
    readings: ReadingBatch,
    bundle: GalleryScores,
    opts: DecodeOptions,
) -> dict[str, Any] | None:
    """Ranks the sentence gallery by decoder likelihood. This is RETRIEVAL and is named so everywhere."""
    from zte.evaluation.audit.scoreboard import decoder_rescoring_retrieval, within_task_retrieval

    texts = bundle.texts
    raw_scores = bundle.raw
    # The null scores are query-independent, so the unconditional gallery pass ran once, not per query.
    scores = bundle.pmi if decoder.decoder_config.rescore_pmi else raw_scores
    gallery_words = bundle.gallery_n_words
    block = decoder_rescoring_retrieval(
        scores,
        readings.meta['text_id'].to_numpy(),
        np.arange(len(texts)),
        query_n_words=readings.meta['n_words'].to_numpy(),
        gallery_n_words=gallery_words,
        length_tol=opts.length_tol,
        seed=opts.seed,
    )
    if block is None:
        return None
    block['n_candidate_sentences'] = len(texts)
    if decoder.decoder_config.rescore_pmi:
        block['score'] = 'pmi'
        block['pmi_vs_raw'] = _pmi_vs_raw(
            scores,
            raw_scores,
            readings.meta['text_id'].to_numpy(),
            np.arange(len(texts)),
            n_boot=opts.n_boot,
            seed=opts.seed,
        )
    if opts.within_task_pools and bundle.gallery_tasks is not None:
        block['within_task'] = within_task_retrieval(
            scores,
            readings.meta,
            gallery_tasks=bundle.gallery_tasks,
            gallery_n_words=gallery_words,
            pools=opts.within_task_pools,
            length_tol=opts.length_tol,
            seed=opts.seed,
        )
    return block


def _capacity(
    decoder: ZTEDecoder,
    dataset: ZuCoDataset,
    gallery: ZuCoTorchDataset,
    readings: ReadingBatch,
    train: ReadingBatch | None,
    opts: DecodeOptions,
    config: ZTEConfig,
    split: str,
    *,
    bundle: GalleryScores | None,
) -> dict[str, Any] | None:
    """Certifies the largest K-way menu the decoder serves. This is SELECTION and is named so everywhere.

    Note:
        No arm may keep the word-synchronous evidence path, because a control built from a bare conditioning
        vector has no words to run it on and would lose for that reason alone. A checkpoint that decodes with
        one therefore cannot share the retrieval pass, and buys its own evidence-free gallery pass here.
    """
    from zte.evaluation.audit.capacity import capacity_report

    evidence_content = not decoder.uses_evidence
    scored = bundle if bundle is not None and bundle.evidence_content == evidence_content else None
    if scored is None:
        scored = _gallery_bundle(decoder, dataset, gallery, readings, opts, evidence_content=evidence_content)
    if scored is None:
        return None

    if train is None:
        _LOG.warning(
            'The capacity audit has no training split, so the length_only arm is omitted and the '
            'beats_length_only_paired clause fails; nothing can certify.'
        )

    query_ids = readings.meta['text_id'].to_numpy()
    query_words = readings.meta['n_words'].to_numpy()
    arms = {
        family: capacity_arms(
            scored,
            decoder,
            readings,
            query_n_words=query_words,
            query_content_ids=query_ids,
            score=family,
            train=train,
            seed=opts.seed,
            batch_size=opts.batch_size,
        )
        for family in _capacity_families(opts.capacity_score)
    }

    rows = _capacity_rows(readings, scored, split)

    return capacity_report(
        arms,
        rows['content_ids'],
        rows['subjects'],
        rows['holdout'],
        rows['n_words'],
        tasks=rows['tasks'],
        train_mask=rows['train_mask'],
        ks=tuple(opts.capacity_ks),
        alpha=opts.capacity_alpha,
        n_perm=opts.capacity_n_perm,
        n_boot=opts.n_boot,
        seed=opts.seed,
        # `test` under a subject-and-stimulus split is the only cell whose queries share neither a brain nor a
        # sentence with anything the bridge was fitted on; every other cell fails the clause on purpose.
        honest_split=split == 'test',
        split_strategy=str(config.train.split),
        split_cell=split,
        evidence_content=scored.evidence_content,
    )


def _capacity_families(score: str) -> tuple[str, ...]:
    """Score families to certify; `both` costs a second length-matched gallery pass for the raw family."""
    return ('raw', 'pmi') if score == 'both' else (str(score),)


def _capacity_rows(readings: ReadingBatch, bundle: GalleryScores, split: str) -> dict[str, Any]:
    """Lays the queries and one prototype row per gallery sentence out in the order the score columns follow.

    Note:
        `capacity_report` derives its gallery from the reference rows, taking each stimulus-level word count as
        the median over them. One prototype row per gallery sentence carrying the count `gallery_scores` already
        computed reproduces that exactly, and keeps the column order the arms were scored in.
    """
    n_query = len(readings)
    n_gallery = len(bundle.texts)
    subjects = sorted({str(s) for s in readings.meta.get('subject', [])})
    holdout = subjects[0] if len(subjects) == 1 else f'{split} cell ({len(subjects)} subjects)'

    query_tasks = readings.meta['task'].astype(str).to_numpy() if 'task' in readings.meta else np.full(n_query, '')
    gallery_tasks = None if bundle.gallery_tasks is None else bundle.gallery_tasks.astype(str)
    # A gallery whose sentences all carry one task label makes a task-matched pool identical to a length-matched
    # one, so the labels are withheld and the report headlines `length_matched` under its own name.
    if gallery_tasks is not None and len({t for t in gallery_tasks.tolist() if t}) < 2:
        gallery_tasks = None
    tasks = None if gallery_tasks is None else np.concatenate([query_tasks, gallery_tasks])

    return {
        'content_ids': np.concatenate(
            [readings.meta['text_id'].to_numpy(dtype=np.int64), np.arange(n_gallery, dtype=np.int64)]
        ),
        'subjects': np.array([holdout] * n_query + [GALLERY_ROW] * n_gallery, dtype=object),
        'n_words': np.concatenate(
            [readings.meta['n_words'].to_numpy(dtype=np.float64), bundle.gallery_n_words.astype(np.float64)]
        ),
        'tasks': tasks,
        'train_mask': np.arange(n_query + n_gallery) >= n_query,
        'holdout': holdout,
    }


def _rank_percentiles(scores: np.ndarray, query_ids: np.ndarray, gallery_ids: np.ndarray) -> np.ndarray:
    """Per-query rank percentile of the true sentence (1.0 = ranked first, 0.0 = absent), scoreboard's convention."""
    out = np.zeros(len(scores), dtype=np.float64)
    n_cand = scores.shape[1]
    for i in range(len(scores)):
        same = gallery_ids[np.argsort(-scores[i])] == query_ids[i]
        if same.any():
            out[i] = 1.0 - int(np.argmax(same)) / max(n_cand - 1, 1)

    return out


def _pmi_vs_raw(
    pmi_scores: np.ndarray,
    raw_scores: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Paired per-query comparison of the PMI and raw rankings, so the correction's effect is itself measured.

    Args:
        pmi_scores (np.ndarray): PMI score matrix `(n_queries, n_gallery)`.
        raw_scores (np.ndarray): Raw conditional score matrix of the same shape.
        query_ids (np.ndarray): Stimulus id of each query `(n_queries,)`.
        gallery_ids (np.ndarray): Stimulus id of each gallery sentence `(n_gallery,)`.
        n_boot (int): Bootstrap resamples behind the delta interval.
        seed (int): Bootstrap seed.

    Returns:
        dict[str, Any]: Per-query rank percentiles under both scores and the paired delta (pmi minus raw) with a
            percentile-bootstrap CI over queries.
    """
    from zte.evaluation.metrics import bootstrap_ci

    per_pmi = _rank_percentiles(pmi_scores, query_ids, gallery_ids)
    per_raw = _rank_percentiles(raw_scores, query_ids, gallery_ids)
    point, lo, hi = bootstrap_ci(per_pmi - per_raw, n_boot=n_boot, seed=seed)

    return {
        'metric': 'rank_percentile',
        'raw_rank_percentile': float(per_raw.mean()) if per_raw.size else float('nan'),
        'pmi_rank_percentile': float(per_pmi.mean()) if per_pmi.size else float('nan'),
        'per_query_rank_percentile_raw': [float(v) for v in per_raw],
        'per_query_rank_percentile_pmi': [float(v) for v in per_pmi],
        'rank_percentile_delta': {'point': point, 'lo': lo, 'hi': hi},
        'n_queries': int(per_raw.size),
    }


def _gallery_lengths(torch_ds: ZuCoTorchDataset, n_text: int) -> np.ndarray:
    """Median read word count per gallery sentence -- the stimulus-level unit the scoreboard stratifies both sides in."""
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


def _gallery_tasks(dataset: ZuCoDataset, torch_ds: ZuCoTorchDataset, n_text: int) -> np.ndarray:
    """The reading task each gallery sentence belongs to, for the within-task candidate pools.

    Note:
        On ZuCo no stimulus appears under more than one task -- the confound audit measures Cramer's V(task, stimulus)
        at 0.998 -- so a task label is a property of the sentence, and a within-task pool is a well-defined partition
        of the gallery rather than a filter that could drop a query's own reference.
    """
    tasks = np.array([''] * n_text, dtype=object)
    words = dataset.words
    if 'stimulus_key' not in words.columns or 'task' not in words.columns:
        return tasks
    per_key = words.groupby('stimulus_key', observed=True)['task'].first()
    for key, text_id in torch_ds.text_vocab.items():
        if 0 <= text_id < n_text and key in per_key.index:
            tasks[text_id] = str(per_key.loc[key])
    return tasks


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
        'rate_ladder': decoder.decoder_config.rate_ladder,
        'evidence_schedule': decoder.decoder_config.evidence_schedule,
        'evidence_active': decoder.uses_evidence,
        'gap_correction': decoder.decoder_config.gap_correction,
        'gap_fitted': bool(decoder.gap.fitted),
        'gap_n_fit': int(decoder.gap.n_fit),
        # Anything that reopens these artifacts re-reads the verdict against this floor, and it is recoverable
        # from nowhere else in them: a dropped clause reads exactly like a passing one.
        'min_prefix_kl': config.decoder.min_prefix_kl,
        # The decoder whitens nothing, but it inherits three train-fitted transforms, and the field exists so a
        # number is never read without knowing what was fitted on what.
        'postprocess_fit': 'none',
        'normalizer_fit': decoder.normalizer is not None,
        'aligner_fit': decoder.aligner is not None,
        'text_source': config.objective.text_source,
        'teacher_forced': False,
        'seed': opts.seed,
        'seeds': list(opts.seeds),
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
    result: dict[str, Any],
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
        {
            'generation': result['generation'],
            'rescoring': result['rescoring'],
            'bit_budget': result.get('bit_budget'),
            'provenance': result['provenance'],
        },
        default=str,
    )
    if result.get('capacity') is not None:
        write_json(
            out_dir / 'capacity.json',
            {'capacity': result['capacity'], 'provenance': result['provenance']},
            default=str,
        )
    lines = _jsonl_rows(meta, references, hypotheses, controls, oracle, prefix_kl)
    (out_dir / 'generation.jsonl').write_text(
        ''.join(json.dumps(row, default=str) + '\n' for row in lines), encoding='utf-8'
    )
    try:
        generation_html(
            result['generation'],
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
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Returns `(generation, rescoring, capacity)` for a decoder checkpoint, or all `None` for an encoder run.

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
        tuple[dict | None, dict | None, dict | None]: The generation, rescoring and menu-capacity blocks.
    """
    obj = config.objective
    want_generation = bool(getattr(obj, 'eval_generation', False))
    want_rescoring = bool(getattr(obj, 'eval_rescoring', False))
    want_capacity = bool(getattr(obj, 'eval_capacity', False))
    if config.train.mode == 'encoder' or not (want_generation or want_rescoring or want_capacity):
        return None, None, None
    try:
        decoder = ZTEDecoder.from_checkpoint(ckpt, dataset, device=resolve_device(device))
    except ValueError as exc:
        _LOG.info('Generation eval skipped: %s', exc)
        return None, None, None

    opts = options or DecodeOptions(
        controls=tuple(config.decoder.generation_controls),
        n_perm=config.decoder.n_permutations,
        rescore=want_rescoring,
        length_tol=config.decoder.length_tol,
        within_task_pools=tuple(config.decoder.within_task_pools),
        capacity=want_capacity,
        capacity_ks=tuple(config.decoder.capacity_ks),
        capacity_alpha=config.decoder.capacity_alpha,
        capacity_n_perm=config.decoder.capacity_n_perm,
        capacity_score=config.decoder.capacity_score,
        seeds=tuple(config.decoder.eval_seeds),
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
                result.get('capacity'),
            )
    _LOG.warning('Generation eval skipped: the run produces no held-out cell to decode.')
    return None, None, None


def main() -> None:
    """Decodes a held-out split with all its controls and writes the generation artifacts."""
    args = parse_arguments()
    configure_logging(args.log_level)

    payload = CheckpointManager.load(args.ckpt, map_location='cpu')
    config = ZTEConfig.from_dict(payload['config'])
    options = options_from_args(args, config)

    out_dir = Path(args.out) if args.out else Path(args.ckpt).resolve().parent.parent / 'evaluation'
    artifacts = [out_dir / 'generation.json', out_dir / 'generation.jsonl']
    if options.capacity:
        artifacts.append(out_dir / 'capacity.json')
    sig = signature(
        args,
        tool='decode',
        extra={'ckpt_sha256': checkpoint_digest(args.ckpt), 'dataset': dataset_key(config.dataset)},
        ignore=('ckpt', 'run_name'),
    )

    # Decided before the dataset is built and the LM is loaded, which is where the hours go: this decodes a
    # checkpoint rather than training one, so identical weights and identical options give the decode on disk.
    if is_done(artifacts, sig, force=args.force):
        result = _existing_decode(artifacts)
    else:
        dataset = dataset_for_config(args, config.dataset)
        decoder = ZTEDecoder.from_checkpoint(args.ckpt, dataset, device=resolve_device(args.device))
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
        mark_done(artifacts, sig)

    _report(result, config, options)


def _existing_decode(artifacts: Sequence[Path]) -> dict[str, Any]:
    """Reassembles the scored blocks of a finished decode from the files it wrote."""
    result: dict[str, Any] = dict(read_json(artifacts[0]))
    capacity = next((a for a in artifacts if a.name == 'capacity.json'), None)
    if capacity is not None:
        result['capacity'] = read_json(capacity).get('capacity')

    return result


def _report(result: dict[str, Any], config: ZTEConfig, options: DecodeOptions) -> None:
    """Logs the readable numbers: the paired deltas, the permutation p and the retrieval readout."""
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
    spread = block.get('seed_spread')
    if spread:
        _LOG.info(
            'Across %d decode seeds: worst-control delta %.4f +/- %.4f.',
            spread.get('n_seeds', 0),
            spread.get('mean', float('nan')),
            spread.get('sd', float('nan')),
        )
    _LOG.info(
        'permutation p=%s | prefix-influence KL=%s nats (floor %s)',
        (block.get('permutation') or {}).get('p_value'),
        block.get('prefix_influence_kl'),
        config.decoder.min_prefix_kl,
    )

    budget = result.get('bit_budget')
    if budget:
        _LOG.info(
            'Rate ladder: %.1f bit ceiling, %.2f bits of code entropy, %.2f bits of mutual information with '
            'sentence identity (upper bound).',
            budget.get('capacity_bits', float('nan')),
            budget.get('code_entropy_bits', float('nan')),
            budget.get('mutual_information_bits', float('nan')),
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
        for task, pool in (rescoring.get('within_task') or {}).items():
            _LOG.info(
                'Within %s only (%s candidates, chance %.4f): Top-1 %.4f, rank percentile %.4f over %s queries.',
                task,
                pool.get('n_candidates'),
                pool.get('chance_top1', float('nan')),
                pool.get('top1', float('nan')),
                pool.get('rank_percentile', float('nan')),
                pool.get('n_queries'),
            )

    capacity = result.get('capacity')
    if capacity:
        _log_capacity(capacity)


def _log_capacity(capacity: dict[str, Any]) -> None:
    """Logs the certified menu size, or a dash and the clauses that failed -- never a blank or a zero."""
    headline = capacity.get('headline') or {}
    score = str(headline.get('score', HEADLINE_SCORE))
    flavor = str(headline.get('flavor', HEADLINE_FLAVOR))
    block = ((capacity.get('scores') or {}).get(score) or {}).get(flavor) or {}
    bits = capacity.get('bits') or {}
    certified = capacity.get('certified_k')
    certified_bits = bits.get('bits_certified')

    _LOG.info(
        'Decoder menu SELECTION capacity (%s / %s, %s queries over %s candidates): certified K = %s, worth %s of '
        'the %s bits of stimulus identity that survive knowing word count.',
        score,
        flavor,
        capacity.get('n_queries'),
        capacity.get('n_gallery'),
        EM_DASH if certified is None else certified,
        EM_DASH if certified_bits is None else f'{certified_bits:.4f} bits',
        bits.get('entropy_identity_given_length'),
    )
    if unreachable := (block.get('ks_unreachable') or []):
        _LOG.warning(
            'Menu sizes %s are unreachable on this gallery: their word-count pools hold too few candidates. '
            'Feasible sizes: %s.',
            ', '.join(str(k) for k in unreachable),
            ', '.join(str(k) for k in (block.get('ks_feasible') or [])) or 'none',
        )
    if certified is None:
        _LOG.warning('Nothing certified. %s', (capacity.get('verdict') or {}).get('reason'))


if __name__ == '__main__':
    main()
