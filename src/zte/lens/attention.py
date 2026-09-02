"""The attention read-out: what the trained encoder's own attention weights attend to, in time and over the scalp."""

import io
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from zte.data.dataset import ZuCoDataset
from zte.data.schema import SAMPLING_RATE_HZ
from zte.data.torch_dataset import ZuCoTorchDataset, build_subject_vocab, collate_sentences
from zte.evaluation.audit.scoreboard import _bootstrap_ci
from zte.inference.embed import ZTEEmbedder
from zte.lens.saliency import DISCLAIMER, _azimuthal_xy, _load_montage
from zte.lens.temporal import CAVEAT as SEGMENTATION_CAVEAT
from zte.lens.temporal import N400_WINDOW_MS
from zte.logging_utils import get_logger
from zte.models.spatial import SpatialAttention, SpatialChannelMixer
from zte.utils.provenance import git_info

_LOG = get_logger('lens.attention')

__all__ = [
    'ATTENTION_CAVEAT',
    'AttentionRecorder',
    'DISCLAIMER',
    'attention_modules',
    'attention_profile',
    'held_out_ranks',
    'record_attention',
    'render_markdown',
    'write_figures',
]

ATTENTION_CAVEAT: Final[str] = (
    'Attention weights are what the model computed, not why its output moved: a heavily attended sample or '
    'electrode may carry nothing the readout uses, and a weight is not a counterfactual. Read this beside the '
    'occlusion profile, never instead of it. '
) + SEGMENTATION_CAVEAT
"""The caveat every attention artifact carries: a weight is a description of the model, not an explanation of it."""

# The bootstrap over readings is vectorised as a weight matrix times the per-reading curves, so its cost is
# n_boot x n_readings x window; 1000 draws keeps 700 readings x 350 samples at a quarter of a gigabyte-second.
N_BOOT_CURVES: Final[int] = 1000
"""Bootstrap draws behind the per-sample and per-channel intervals."""

# Attention received per electrode has one entry per channel; listing every one would be the topomap in prose.
N_TOP_CHANNELS: Final[int] = 10
"""Electrodes named in the summary, most attended first."""

# The plotted radius of the scalp in metres, the size mne's head outline is drawn for.
_HEAD_RADIUS_M: Final[float] = 0.095
"""Scalp radius the unit-sphere electrode coordinates are scaled to for mne's topomap."""

# The mne-free scalp map scatters at least four points; a topomap of fewer electrodes is a decoration.
_MIN_TOPOMAP_CHANNELS: Final[int] = 4
"""Fewest electrodes a scalp map is drawn for."""

# The correct/incorrect groups the profile is reported for, in the order the figures show them.
GROUPS: Final[tuple[str, ...]] = ('correct', 'incorrect', 'all')
"""Reading groups every curve and every scalp map is reported for."""


@dataclass(slots=True)
class AttentionRecorder:
    """Collects, per hooked attention module, the attention each key received: `(n_tokens, n_keys)` per call.

    Attributes:
        temporal (dict[int, list[np.ndarray]]): Per intra-word transformer layer, one `(n_tokens, time_steps)`
            block per forward call, each row the mean over heads and queries of that token's attention matrix.
        spatial (list[np.ndarray]): One `(n_tokens, n_channels)` block per forward call of the channel mixer.
        n_heads (dict[str, int]): Head counts observed, keyed `'temporal'` / `'spatial'`.
    """

    temporal: dict[int, list[np.ndarray]] = field(default_factory=dict)
    spatial: list[np.ndarray] = field(default_factory=list)
    n_heads: dict[str, int] = field(default_factory=dict)

    def clear(self) -> None:
        """Drops everything recorded so far, keeping the head counts."""
        self.temporal = {}
        self.spatial = []

    def sink(self, kind: str, layer: int) -> Callable[[nn.Module, tuple[Any, ...], dict[str, Any], Any], None]:
        """A forward hook that reduces the module's `(n_tokens, heads, queries, keys)` weights and stores them."""

        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            weights = output[1]
            if weights is None:
                raise RuntimeError(f'{type(module).__name__} returned no attention weights; the pre-hook did not run.')

            if weights.ndim != 4:
                raise RuntimeError(f'Expected per-head weights (n, heads, q, k), got shape {tuple(weights.shape)}.')

            self.n_heads[kind] = int(weights.shape[1])
            received = weights.float().mean(dim=(1, 2)).cpu().numpy()  # (n_tokens, n_keys)
            if kind == 'spatial':
                self.spatial.append(received)
            else:
                self.temporal.setdefault(layer, []).append(received)

        return hook


def _force_weights(
    module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """A forward pre-hook that makes `nn.MultiheadAttention` return its per-head weights whatever the caller asked."""
    forced = dict(kwargs)
    forced['need_weights'] = True
    forced['average_attn_weights'] = False

    return args, forced


def attention_modules(model: nn.Module) -> tuple[nn.Module | None, list[nn.Module], dict[str, str | None]]:
    """Finds the channel mixer's and the intra-word transformer's attention modules inside a built encoder.

    Args:
        model (nn.Module): A `ZTEModel`, whose `frontend` is inspected.

    Returns:
        tuple[nn.Module | None, list[nn.Module], dict[str, str | None]]: The spatial `nn.MultiheadAttention` (or
            `None`), the per-layer temporal `nn.MultiheadAttention` modules (empty when there is no intra-word
            transformer), and a reason per missing kind under `'spatial'` / `'temporal'` (`None` when present).
    """
    frontend = getattr(model, 'frontend', None)
    reasons: dict[str, str | None] = {'spatial': None, 'temporal': None}

    spatial: nn.Module | None = None
    mixer = getattr(frontend, 'spatial_mixer', None)
    match mixer:
        case SpatialChannelMixer() if mixer.mix:
            spatial = mixer.attn
        case SpatialChannelMixer():
            reasons['spatial'] = 'spatial_mix is off: the mixer adds a positional encoding and attends to nothing.'
        case SpatialAttention():
            reasons['spatial'] = (
                'spatial_attention is geometry-only: its weights are a constant of the montage, not of the input.'
            )
        case _:
            reasons['spatial'] = 'the checkpoint has no electrode mixer (spatial_encoding is off).'

    transformer = getattr(frontend, 'transformer', None)
    temporal: list[nn.Module] = []
    if isinstance(transformer, nn.TransformerEncoder):
        temporal = [layer.self_attn for layer in transformer.layers]  # type: ignore[arg-type]
    else:
        frontend_name = type(frontend).__name__ if frontend is not None else 'none'
        reasons['temporal'] = f'the {frontend_name} frontend has no intra-word transformer over the raw window.'

    return spatial, temporal, reasons


@contextmanager
def record_attention(model: nn.Module) -> Iterator[AttentionRecorder]:
    """Hooks every attention module of `model` for the duration of the block and yields their recorder.

    Note:
        In eval mode under `no_grad`, `nn.TransformerEncoderLayer` prefers a fused path that never calls its
        `self_attn` submodule; torch steps aside from it when hooks are attached, but that check is torch's to
        keep, so the fast path is switched off for the block and restored afterwards. The numbers are identical
        either way, only the route through the code differs.

    Args:
        model (nn.Module): The encoder to hook, left in whatever mode it was in.

    Yields:
        AttentionRecorder: Filled by every forward pass made inside the block.
    """
    spatial, temporal, _ = attention_modules(model)
    recorder = AttentionRecorder()
    handles: list[RemovableHandle] = []

    for kind, layer, module in [('spatial', 0, spatial), *(('temporal', i, m) for i, m in enumerate(temporal))]:
        if module is None:
            continue

        handles.append(module.register_forward_pre_hook(_force_weights, with_kwargs=True))
        handles.append(module.register_forward_hook(recorder.sink(kind, layer), with_kwargs=True))

    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield recorder
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        for handle in handles:
            handle.remove()


# ---- Which readings count as retrieved ---- #


def held_out_ranks(
    gallery: np.ndarray, subjects: np.ndarray, stimulus_keys: np.ndarray, holdout: str
) -> tuple[np.ndarray, np.ndarray]:
    """Ranks each held-out reading's true sentence among other subjects' readings only.

    The same question `held_out_retrieval` scores: can a stranger's reading be found in the brains the model
    trained on. The gallery excludes every reading by the query's own subject, so a sentence the holdout read twice
    cannot retrieve itself.

    Args:
        gallery (np.ndarray): `(n_readings, embed_dim)` sentence embeddings over the whole dataset order.
        subjects (np.ndarray): `(n_readings,)` subject code per reading.
        stimulus_keys (np.ndarray): `(n_readings,)` normalised sentence key per reading.
        holdout (str): The subject whose readings are the queries.

    Returns:
        tuple[np.ndarray, np.ndarray]: The query positions, and the 1-based rank of the first other-subject reading
            of the same sentence for each; `-1` where no other subject read it, so it cannot be scored.
    """
    subjects = np.asarray(subjects)
    stimulus_keys = np.asarray(stimulus_keys)
    emb = np.asarray(gallery, dtype=np.float64)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)

    queries = np.where(subjects == holdout)[0]
    candidates = np.where(subjects != holdout)[0]
    ranks = np.full(queries.shape[0], -1, dtype=np.int64)
    if candidates.size == 0:
        return queries, ranks

    sims = np.nan_to_num(emb[queries] @ emb[candidates].T, nan=-1.0)  # (n_queries, n_candidates)
    for row, query in enumerate(queries):
        order = candidates[np.argsort(-sims[row], kind='stable')]
        same = stimulus_keys[order] == stimulus_keys[query]
        if same.any():
            ranks[row] = int(np.argmax(same)) + 1

    return queries, ranks


# ---- Aggregation ---- #


def _bootstrap_curves(curves: np.ndarray, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and percentile-bootstrap band over the reading axis of `(n_readings, width)` curves."""
    n = int(curves.shape[0])
    mean = curves.mean(axis=0)
    if n < 2:
        return mean, mean.copy(), mean.copy()

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    weights = np.zeros((n_boot, n), dtype=np.float64)
    np.add.at(weights, (np.repeat(np.arange(n_boot), n), draws.ravel()), 1.0)
    means = (weights / n) @ curves

    return mean, np.quantile(means, 0.025, axis=0), np.quantile(means, 0.975, axis=0)


def _n400_samples(window: int) -> tuple[int, int]:
    """The `[lo, hi)` sample span of the conventional N400 band inside a `window`-sample word-locked epoch."""
    lo = int(round(N400_WINDOW_MS[0] * SAMPLING_RATE_HZ / 1000.0))
    hi = int(round(N400_WINDOW_MS[1] * SAMPLING_RATE_HZ / 1000.0))

    return max(0, min(lo, window)), max(0, min(hi, window))


def _temporal_group(curves: np.ndarray, n_words: np.ndarray, seed: int) -> dict[str, Any]:
    """One group's temporal block from `(n_readings, n_layers, time_steps)` attention-received curves."""
    n_readings, n_layers, window = curves.shape
    lo, hi = _n400_samples(window)
    uniform_mass = (hi - lo) / window
    layers = []
    for index in range(n_layers):
        mean, ci_low, ci_high = _bootstrap_curves(curves[:, index], N_BOOT_CURVES, seed)
        layers.append({'layer': index, 'mean': mean.tolist(), 'ci_low': ci_low.tolist(), 'ci_high': ci_high.tolist()})

    # The last layer is the one the temporal pool averages, so its received attention is the weight each time
    # step's value carries into the word vector -- the curve to read the latency question on.
    last = curves[:, -1]
    mass = last[:, lo:hi].sum(axis=1)
    mass_mean, mass_low, mass_high = _bootstrap_ci(mass, seed=seed)
    peak = int(np.argmax(last.mean(axis=0)))
    peak_ms = _ms(peak + 0.5)

    return {
        'n_readings': int(n_readings),
        'n_words': int(n_words.sum()),
        'layers': layers,
        'headline_layer': n_layers - 1,
        'n400_mass': float(mass_mean),
        'n400_mass_ci': [float(mass_low), float(mass_high)],
        'n400_mass_uniform': float(uniform_mass),
        'n400_mass_per_reading': mass.tolist(),
        'peak_sample': peak,
        'peak_ms': peak_ms,
        'peak_in_n400_window': bool(N400_WINDOW_MS[0] <= peak_ms <= N400_WINDOW_MS[1]),
    }


def _spatial_group(received: np.ndarray, seed: int) -> dict[str, Any]:
    """One group's scalp block from `(n_readings, n_channels)` attention-received vectors."""
    mean, ci_low, ci_high = _bootstrap_curves(received, N_BOOT_CURVES, seed)

    return {
        'n_readings': int(received.shape[0]),
        'mean': mean.tolist(),
        'ci_low': ci_low.tolist(),
        'ci_high': ci_high.tolist(),
        'uniform': 1.0 / received.shape[1],
    }


def _difference_ci(a: np.ndarray, b: np.ndarray, seed: int, n_boot: int = 2000) -> list[float] | None:
    """Percentile bootstrap of `mean(a) - mean(b)` over two independent groups, or `None` when either is empty."""
    if a.size == 0 or b.size == 0:
        return None

    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    db = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    diff = da - db

    return [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]


def _ms(sample: float) -> float:
    """Sample offset from word onset in milliseconds."""
    return round(1000.0 * float(sample) / SAMPLING_RATE_HZ, 3)


# ---- The profile ---- #


def attention_profile(
    embedder: ZTEEmbedder,
    dataset: ZuCoDataset,
    subject: str,
    correct_top_k: int = 1,
    batch_size: int = 4,
    max_readings: int = 0,
    seed: int = 0,
    ckpt_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Profiles where the encoder's attention lands, in time and over the scalp, split by whether retrieval succeeded.

    Every reading of `subject` is scored against the other subjects' readings exactly as `held_out_retrieval`
    scores it, then run again through the hooked encoder. The intra-word transformer yields, per word, how much
    attention each of the raw window's time steps received (mean over heads and queries); the channel mixer yields
    the same per electrode. Both are averaged over a reading's words, then bootstrapped over readings, for the
    correctly retrieved readings, the rest, and all of them.

    Note:
        The channel mixer attends over electrodes with the whole raw window as each electrode's feature vector, so
        its weights carry no latency axis: the scalp map is the attention electrodes received across the entire
        word, and the N400 restriction applies to the temporal curve alone. With mean temporal pooling, attention
        received in the last layer is exactly the weight each time step's value carries into the word vector.

    Args:
        embedder (ZTEEmbedder): The loaded encoder; must read raw windows.
        dataset (ZuCoDataset): The built dataset the readings live in, carrying raw EEG windows.
        subject (str): The subject whose readings are profiled -- the checkpoint's holdout, for an honest split.
        correct_top_k (int, optional): A reading counts as retrieved when its sentence ranks within this cut-off
            among other subjects' readings. Defaults to 1.
        batch_size (int, optional): Readings per hooked forward pass; the per-head weights are quadratic in the
            window, so this stays small. Defaults to 4.
        max_readings (int, optional): Cap on readings run through the hooks, retrieved ones first; 0 profiles them
            all. Defaults to 0.
        seed (int, optional): Seed for the bootstrap intervals. Defaults to 0.
        ckpt_path (str | Path | None, optional): Checkpoint path, hashed into provenance. Defaults to None.

    Returns:
        dict[str, Any] | None: The `attention.json` payload, or `None` when the checkpoint reads no raw window or
            the dataset holds none, so there is no time axis to profile.

    Raises:
        ValueError: If the subject has no readings, or `correct_top_k` or `batch_size` is not positive.
    """
    if correct_top_k <= 0 or batch_size <= 0:
        raise ValueError('correct_top_k and batch_size must both be positive.')

    if not bool(embedder.model.uses_raw):
        _LOG.warning('Attention profile skipped: the model reads band-power features, which carry no time axis.')
        return None

    if dataset.raw_eeg is None:
        _LOG.warning('Attention profile skipped: the dataset holds no raw EEG windows.')
        return None

    torch_ds = ZuCoTorchDataset(dataset, subject_vocab=build_subject_vocab(dataset))
    keys = np.asarray(list(torch_ds.stimulus_keys))

    # Retrieval first, over the whole gallery and without hooks: the weights are quadratic in the window and would
    # not fit beside a full embedding batch.
    gallery, meta = embedder.embed(dataset, level='sentence')
    subjects = np.asarray([str(s) for s in meta['subject'].tolist()])
    queries, ranks = held_out_ranks(gallery, subjects, keys, subject)
    if queries.size == 0:
        raise ValueError(f'Subject {subject!r} has no reading in this dataset.')

    scorable = ranks > 0
    correct = scorable & (ranks <= correct_top_k)
    order = np.concatenate([queries[correct], queries[~correct]])
    is_correct = np.concatenate([np.ones(int(correct.sum()), bool), np.zeros(int((~correct).sum()), bool)])
    if max_readings > 0 and order.size > max_readings:
        order, is_correct = order[:max_readings], is_correct[:max_readings]

    # Then the hooked pass, in small batches, keeping only the valid (present, unpadded) words of each reading.
    spatial_module, temporal_modules, reasons = attention_modules(embedder.model)
    temporal_curves: list[np.ndarray] = []
    spatial_received: list[np.ndarray] = []
    n_words: list[int] = []
    objective = embedder.config.objective.name
    embedder.model.eval()
    with record_attention(embedder.model) as recorder, torch.no_grad():
        for lo in range(0, order.size, batch_size):
            positions = order[lo : lo + batch_size]
            batch = collate_sentences([torch_ds[int(p)] for p in positions])
            batch = {k: (v.to(embedder.device.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            recorder.clear()
            embedder.model.embed_sentence(batch, objective=objective)

            valid = (batch['pad_mask'] & batch['presence']).cpu().numpy()  # (b, L)
            b, length = valid.shape
            counts = valid.sum(axis=1)
            expected: list[tuple[str, list[list[np.ndarray]]]] = [('temporal', list(recorder.temporal.values()))]
            if spatial_module is not None:
                expected.append(('spatial', [recorder.spatial]))
            for kind, blocks in expected:
                for chunks in blocks:
                    if len(chunks) != 1 or chunks[0].shape[0] != b * length:
                        raise RuntimeError(
                            f'The {kind} attention fired {len(chunks)} time(s) over '
                            f'{[c.shape[0] for c in chunks]} tokens for {b * length} words; '
                            'one pass per batch is assumed, so this frontend cannot be read this way.'
                        )

            if temporal_modules:
                stack = np.stack([recorder.temporal[i][0] for i in range(len(temporal_modules))], axis=1)
                stack = stack.reshape(b, length, len(temporal_modules), -1)  # (b, L, layers, T)
                temporal_curves.extend(_masked_mean(stack, valid))
            if recorder.spatial:
                spatial_received.extend(_masked_mean(recorder.spatial[0].reshape(b, length, -1), valid))
            n_words.extend(int(c) for c in counts)

    words = np.asarray(n_words, dtype=np.int64)
    groups = {'correct': is_correct, 'incorrect': ~is_correct, 'all': np.ones(is_correct.size, dtype=bool)}
    window = int(dataset.raw_eeg.shape[-1])

    temporal: dict[str, Any] | None = None
    if temporal_curves:
        curves = np.stack(temporal_curves).astype(np.float64)  # (n_readings, layers, T)
        by_group = {
            name: _temporal_group(curves[mask], words[mask], seed) for name, mask in groups.items() if mask.any()
        }
        masses = {name: np.asarray(block['n400_mass_per_reading']) for name, block in by_group.items()}
        temporal = {
            'method': 'self_attention_received_mean_over_heads_and_queries',
            'n_layers': int(curves.shape[1]),
            'n_heads': recorder.n_heads.get('temporal'),
            'times_ms': [_ms(t + 0.5) for t in range(window)],
            'n400_window_ms': list(N400_WINDOW_MS),
            'n400_window_samples': list(_n400_samples(window)),
            'uniform': 1.0 / window,
            'groups': by_group,
            'contrast': {
                'n400_mass_difference': (
                    float(masses['correct'].mean() - masses['incorrect'].mean())
                    if 'correct' in masses and 'incorrect' in masses
                    else None
                ),
                'n400_mass_difference_ci': _difference_ci(
                    masses.get('correct', np.empty(0)), masses.get('incorrect', np.empty(0)), seed
                ),
            },
        }

    spatial: dict[str, Any] | None = None
    if spatial_received:
        received = np.stack(spatial_received).astype(np.float64)  # (n_readings, C)
        spatial = _spatial_block(embedder, dataset, received, groups, recorder.n_heads.get('spatial'), seed)

    from zte.training.init import file_sha256

    return {
        'schema': 'zte.lens.attention/1',
        'sampling_rate_hz': float(SAMPLING_RATE_HZ),
        'raw_window_samples': window,
        'window_ms': _ms(window),
        'subject': subject,
        'is_holdout': embedder.config.train.loso_holdout_subject == subject,
        'selection': {
            'criterion': f'held_out_top{correct_top_k}',
            'top_k': int(correct_top_k),
            'gallery': "other subjects' readings, unstratified",
            'postprocess_fit': 'none',
            'n_queries': int(queries.size),
            'n_scorable': int(scorable.sum()),
            'n_correct': int(correct.sum()),
            'n_incorrect': int((~correct).sum()),
            'n_profiled': int(order.size),
            'chance_top1': float(1.0 / max(len(set(keys[queries].tolist())), 1)),
            'rank_percentile': float(
                np.mean(1.0 - (ranks[scorable] - 1) / max(int((subjects != subject).sum()) - 1, 1))
            )
            if scorable.any()
            else None,
            'note': 'Selection only. The retrieval numbers here choose which readings are averaged and are never '
            'a result; the scoreboard and the length audit report retrieval.',
        },
        'temporal': temporal,
        'spatial': spatial,
        'absent': {kind: why for kind, why in reasons.items() if why},
        'caveat': ATTENTION_CAVEAT,
        'disclaimer': DISCLAIMER,
        'provenance': {
            'ckpt': None if ckpt_path is None else str(ckpt_path),
            'ckpt_sha256': None if ckpt_path is None else file_sha256(ckpt_path),
            'run_name': embedder.config.run_name,
            'git_commit': git_info()['commit'],
            'train_holdout': embedder.config.train.loso_holdout_subject,
        },
    }


def _masked_mean(values: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
    """Per-reading mean of `(b, L, ...)` over the words `valid` marks, one array per reading."""
    out: list[np.ndarray] = []
    for row in range(values.shape[0]):
        keep = valid[row]
        out.append(values[row][keep].mean(axis=0) if keep.any() else values[row].mean(axis=0))

    return out


def _spatial_block(
    embedder: ZTEEmbedder,
    dataset: ZuCoDataset,
    received: np.ndarray,
    groups: dict[str, np.ndarray],
    n_heads: int | None,
    seed: int,
) -> dict[str, Any]:
    """The scalp half of the payload: per-group attention received per electrode, with the montage when it exists."""
    n_channels = int(received.shape[1])
    blocks = {name: _spatial_group(received[mask], seed) for name, mask in groups.items() if mask.any()}

    mixer = getattr(embedder.model.frontend, 'spatial_mixer', None)
    approximate = bool(getattr(mixer, 'approximate_geometry', True))
    montage = _load_montage(dataset.config.montage_csv, n_channels) if dataset.config.montage_csv else None
    labels, regions, xyz = montage if montage is not None else ([f'ch{c:03d}' for c in range(n_channels)], None, None)

    for block in blocks.values():
        mean = np.asarray(block['mean'])
        top = np.argsort(-mean)[:N_TOP_CHANNELS]
        block['top_channels'] = [{'channel': int(c), 'label': labels[c], 'mean': float(mean[c])} for c in top]
        if regions is not None:
            names = list(dict.fromkeys(regions))
            block['region_mass'] = {
                name: float(mean[[c for c in range(n_channels) if regions[c] == name]].sum()) for name in names
            }

    return {
        'method': 'channel_mixer_attention_received_mean_over_heads_and_queries',
        'n_channels': n_channels,
        'n_heads': n_heads,
        'labels': labels,
        'regions': regions,
        'xy': None if xyz is None else _azimuthal_xy(xyz).tolist(),
        'xyz': None if xyz is None else xyz.tolist(),
        'approximate_geometry': approximate,
        'has_time_axis': False,
        'groups': blocks,
    }


# ---- Figures and prose ---- #


def write_figures(report: dict[str, Any], target: Path) -> dict[str, str | None]:
    """Draws the temporal curve and the scalp maps beside `attention.json`, returning what was written and why not.

    Args:
        report (dict[str, Any]): An `attention_profile` payload.
        target (Path): The directory the PNGs are written into.

    Returns:
        dict[str, str | None]: `temporal` and `topomap`, each the written path or `None`, plus `topomap_reason`
            when the scalp map was declined.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    written: dict[str, str | None] = {'temporal': None, 'topomap': None, 'topomap_reason': None}
    target.mkdir(parents=True, exist_ok=True)

    temporal = report.get('temporal')
    if temporal:
        path = target / 'attention_temporal.png'
        _draw_temporal(plt, temporal, path, report.get('subject', ''))
        written['temporal'] = str(path)

    spatial = report.get('spatial')
    if not spatial:
        written['topomap_reason'] = report.get('absent', {}).get('spatial') or 'no channel mixer attention recorded.'
    elif spatial.get('approximate_geometry'):
        written['topomap_reason'] = (
            'the checkpoint carries the approximate coordinate-free cap, so a scalp map would show array indices, '
            'not regions.'
        )
    elif spatial.get('xyz') is None:
        written['topomap_reason'] = 'no montage CSV was readable, so the electrodes have no scalp positions.'
    elif int(spatial.get('n_channels', 0)) < _MIN_TOPOMAP_CHANNELS:
        written['topomap_reason'] = f'fewer than {_MIN_TOPOMAP_CHANNELS} electrodes.'
    else:
        path = target / 'attention_topomap.png'
        _draw_topomaps(plt, spatial, path, report.get('subject', ''))
        written['topomap'] = str(path)

    if written['topomap_reason']:
        _LOG.warning('Scalp map not drawn: %s', written['topomap_reason'])

    return written


def _draw_temporal(plt: Any, temporal: dict[str, Any], path: Path, subject: str) -> None:
    """The attention-received curve per group over the word window, with the N400 band and the uniform floor."""
    times = np.asarray(temporal['times_ms'])
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    colours = {'correct': '#c0392b', 'incorrect': '#7f8c8d', 'all': '#2c3e50'}
    headline = int(temporal.get('n_layers', 1)) - 1
    for name in GROUPS:
        block = temporal['groups'].get(name)
        if not block:
            continue

        layer = block['layers'][headline]
        label = f'{name} (n={block["n_readings"]} readings, {block["n_words"]} words)'
        ax.plot(times, layer['mean'], color=colours[name], linewidth=1.6 if name == 'correct' else 1.1, label=label)
        ax.fill_between(times, layer['ci_low'], layer['ci_high'], color=colours[name], alpha=0.15, linewidth=0)

    lo, hi = temporal['n400_window_ms']
    ax.axvspan(lo, hi, color='#f39c12', alpha=0.12, label=f'N400 band {lo:.0f}-{hi:.0f} ms')
    ax.axhline(temporal['uniform'], color='k', linestyle=':', linewidth=0.9, label='uniform attention')
    ax.set_xlabel('ms from word onset')
    ax.set_ylabel(f'attention received, layer {headline}')
    ax.set_title(f'Intra-word attention, subject {subject} (mean over heads and queries)')
    ax.set_xlim(float(times[0]), float(times[-1]))
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _draw_topomaps(plt: Any, spatial: dict[str, Any], path: Path, subject: str) -> None:
    """Correct, incorrect and their difference on the scalp: mne's topomap when importable, the in-house map otherwise."""
    groups = spatial['groups']
    panels: list[tuple[str, np.ndarray]] = [
        (name, np.asarray(groups[name]['mean'])) for name in ('correct', 'incorrect') if name in groups
    ]
    if 'correct' in groups and 'incorrect' in groups:
        panels.append(
            ('correct - incorrect', np.asarray(groups['correct']['mean']) - np.asarray(groups['incorrect']['mean']))
        )
    if not panels:
        panels = [('all', np.asarray(groups['all']['mean']))]

    xyz = np.asarray(spatial['xyz'], dtype=np.float64)
    labels = list(spatial['labels'])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    axes = np.atleast_1d(axes)

    try:
        import mne

        info = mne.create_info(labels, sfreq=SAMPLING_RATE_HZ, ch_types='eeg')
        unit = xyz / np.clip(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-8, None)
        montage = mne.channels.make_dig_montage(
            ch_pos=dict(zip(labels, unit * _HEAD_RADIUS_M, strict=True)), coord_frame='head'
        )
        info.set_montage(montage)
        for ax, (title, values) in zip(axes, panels, strict=True):
            diverging = title.startswith('correct -')
            vlim: tuple[float | None, float | None] = (None, None)
            if diverging:
                limit = float(np.abs(values).max())
                vlim = (-limit, limit)
            image, _ = mne.viz.plot_topomap(
                values,
                info,
                axes=ax,
                show=False,
                cmap='RdBu_r' if diverging else 'magma',
                vlim=vlim,
                sphere=(0.0, 0.0, 0.0, _HEAD_RADIUS_M),
                contours=6,
            )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title)
    except ImportError:
        _LOG.warning('mne is not installed; drawing the scalp maps with the in-house projection instead.')
        from matplotlib import image as mimage

        from zte.evaluation.plots import scalp_topomap
        from zte.models.spatial import ScalpGeometry

        coords = ScalpGeometry(xyz=xyz).coords_2d
        plt.close(fig)
        figs = [scalp_topomap(values, coords, title=title, label='attention received') for title, values in panels]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
        for ax, single in zip(np.atleast_1d(axes), figs, strict=True):
            buffer = io.BytesIO()
            single.savefig(buffer, format='png', dpi=160)
            plt.close(single)
            buffer.seek(0)
            ax.imshow(mimage.imread(buffer))
            ax.axis('off')

    fig.suptitle(f'Channel-mixer attention received per electrode, subject {subject} (whole word window)')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_markdown(report: dict[str, Any], figures: dict[str, str | None] | None = None) -> str:
    """Renders an attention profile as the markdown block that ships beside `attention.json`.

    Args:
        report (dict[str, Any]): An `attention_profile` payload.
        figures (dict[str, str | None] | None, optional): What `write_figures` wrote, to name the PNGs. Defaults
            to None.

    Returns:
        str: Markdown: the selection, the per-group N400 mass, the most attended electrodes, and the caveat.
    """
    sel = report.get('selection') or {}
    lines = [
        '# Attention read-out',
        '',
        f'Subject **{report.get("subject")}**'
        f'{" (the checkpoint's held-out subject)" if report.get("is_holdout") else " (a TRAINING subject)"}: '
        f'{sel.get("n_profiled", 0)} readings profiled, of which **{sel.get("n_correct", 0)}** were retrieved at '
        f"`{sel.get('criterion')}` against other subjects' readings ({sel.get('n_scorable', 0)} scorable of "
        f'{sel.get("n_queries", 0)}; chance Top-1 {_fmt(sel.get("chance_top1"), 4)}; post-processing '
        f'`{sel.get("postprocess_fit")}`).',
        '',
        f'_{sel.get("note", "")}_',
        '',
    ]

    temporal = report.get('temporal')
    if temporal:
        lo, hi = temporal['n400_window_ms']
        lines += [
            f'## When in the word: attention received per time step (layer {temporal.get("headline_layer", temporal["n_layers"] - 1)} of '
            f'{temporal["n_layers"]}, {temporal.get("n_heads")} heads)',
            '',
            f'Mass inside {lo:.0f}-{hi:.0f} ms, where a uniform profile puts {_fmt(temporal["groups"]["all"]["n400_mass_uniform"], 4)}:',
            '',
            '| Group | Readings | Words | N400 mass | 95% CI | Peak (ms) | Peak in band |',
            '| --- | ---: | ---: | ---: | --- | ---: | --- |',
        ]
        for name in GROUPS:
            block = temporal['groups'].get(name)
            if not block:
                continue

            ci = block['n400_mass_ci']
            lines.append(
                f'| {name} | {block["n_readings"]} | {block["n_words"]} | {_fmt(block["n400_mass"], 4)} '
                f'| [{_fmt(ci[0], 4)}, {_fmt(ci[1], 4)}] | {_fmt(block["peak_ms"])} '
                f'| {"yes" if block["peak_in_n400_window"] else "no"} |'
            )

        contrast = temporal.get('contrast') or {}
        if contrast.get('n400_mass_difference') is not None:
            ci = contrast['n400_mass_difference_ci'] or [None, None]
            lines += [
                '',
                f'Correct minus incorrect N400 mass: **{_fmt(contrast["n400_mass_difference"], 4)}** '
                f'[{_fmt(ci[0], 4)}, {_fmt(ci[1], 4)}]. An interval containing zero means the retrieved readings '
                'were not attended differently in that band.',
            ]
        lines.append('')
    else:
        lines += ['## When in the word', '', f'Not available: {report.get("absent", {}).get("temporal")}', '']

    spatial = report.get('spatial')
    if spatial:
        lines += [
            f'## Where on the scalp: attention received per electrode ({spatial["n_channels"]} channels, '
            f'{spatial.get("n_heads")} heads, whole word window)',
            '',
            f'Geometry: **{"approximate cap -- array indices, not regions" if spatial["approximate_geometry"] else "exact montage"}**. '
            f'Uniform attention is {_fmt(1.0 / spatial["n_channels"], 4)} per electrode. The mixer attends over '
            "electrodes with the whole window as each electrode's features, so this map has no latency axis.",
            '',
        ]
        for name in GROUPS:
            block = spatial['groups'].get(name)
            if not block:
                continue

            top = ', '.join(f'{c["label"]} ({_fmt(c["mean"], 4)})' for c in block['top_channels'])
            lines.append(f'- **{name}** ({block["n_readings"]} readings): {top}')
            if block.get('region_mass'):
                regions = ', '.join(f'{r} {_fmt(v, 3)}' for r, v in block['region_mass'].items())
                lines.append(f'  - by region: {regions}')
        lines.append('')
    else:
        lines += ['## Where on the scalp', '', f'Not available: {report.get("absent", {}).get("spatial")}', '']

    if figures:
        lines += ['## Figures', '']
        for key in ('temporal', 'topomap'):
            if figures.get(key):
                lines.append(f'- `{Path(str(figures[key])).name}`')
        if figures.get('topomap_reason'):
            lines.append(f'- scalp map not drawn: {figures["topomap_reason"]}')
        lines.append('')

    lines += [f'{report.get("caveat", ATTENTION_CAVEAT)}', '', f'_{report.get("disclaimer", DISCLAIMER)}._']

    return '\n'.join(lines).rstrip() + '\n'


def _fmt(value: Any, digits: int = 1) -> str:
    """Formats a table cell, printing a dash rather than `None`."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '--'

    return f'{float(value):.{digits}f}'
