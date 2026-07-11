"""Typed, serialisable configuration objects for the whole ZTE pipeline.

Every tunable knob lives in a `dataclasses` object so configs are explicit, IDE-discoverable and round-trip cleanly to YAML.
The top-level `ZTEConfig` aggregates the dataset, model, objective and training sub-configs and is what the CLIs read and write.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args, get_type_hints

import yaml

from zte.data.schema import BANDS, Band, EyeTrackingMeasure, Task

type Granularity = Literal['word', 'sentence']
type Representation = Literal['band_power', 'raw', 'both']
type Normalization = Literal['zscore_channel', 'zscore_global', 'zscore_subject', 'minmax', 'none']
type MissingMethod = Literal[
    'zero',
    'row_mean',
    'col_mean',
    'global_mean',
    'median',
    'knn',
    'iterative',
    'ffill',
    'interpolate',
    'drop',
    'mask_only',
]
type SplitStrategy = Literal['random', 'by_sentence', 'by_stimulus', 'by_subject_loso', 'by_task']
type ObjectiveName = Literal['skipgram', 'cbow', 'masked', 'cpc']
type FrontendName = Literal['band_power_mlp', 'raw_conformer']
type PoolName = Literal['mean', 'attention', 'cls']
type SchedulerName = Literal['cosine', 'linear', 'constant']
type PosEncoding = Literal['rope', 'sinusoidal', 'learned', 'alibi', 'none']


@dataclass
class MissingConfig:
    """How missing word-level values (omitted words, rejected epochs) are filled.

    Attributes:
        method (MissingMethod): The imputation strategy. `mask_only` leaves NaNs in place and relies solely on the presence mask;
            `drop` removes incomplete rows; the remainder fill values in different ways.
        knn_neighbors (int): Neighbour count for the `knn` method.
        iterative_max_iter (int): Max rounds for the `iterative` (model-based) method.
        interpolate_method (Literal['linear', 'nearest', 'spline']): Pandas interpolation kind for `interpolate`.
        add_missing_indicator (bool): If `True`, emit a boolean presence mask alongside the imputed features so downstream losses can ignore filled entries.
    """

    method: MissingMethod = 'mask_only'
    knn_neighbors: int = 5
    iterative_max_iter: int = 10
    interpolate_method: Literal['linear', 'nearest', 'spline'] = 'linear'
    add_missing_indicator: bool = True


@dataclass
class DatasetConfig:
    """Everything that controls how raw ZuCo `.mat` files become tensors.

    Attributes:
        root: Directory holding extracted `.mat` files (searched recursively).
        tasks: Which reading tasks to include (`SR`, `NR`, `TSR`).
        subjects: Subject codes to include, or `None` for all discovered.
        granularity (Granularity): Token granularity. Only `'word'` is implemented; `'sentence'` is reserved
            (use `ZTEEmbedder(level='sentence')` for pooled sentence embeddings at inference time).
        representation (Representation): Use compact band-power vectors, raw time-series windows, or both.
        band_power_measures (tuple[EyeTrackingMeasure, ...]): Eye-tracking measures whose band features are used for the band-power representation.
        include_eye_tracking (bool): Whether eye-tracking *behaviour* (the fixation-duration scalars FFD/SFD/GD/GPT/TRT,
            `nFixations` and `meanPupilSize`) is appended to each word's band-power feature vector. `True` (default) enriches the
            reading-evoked representation; `False` yields an EEG-only representation better suited to imagined thought, where no eye-tracking exists.
            The EEG band-power itself is always kept -- this toggle only governs the extra gaze-behaviour dimensions.
        eye_tracking_measures (tuple[str, ...]): Which per-word eye-tracking scalars are appended when `include_eye_tracking` is `True`.
        bands (tuple[Band, ...]): Frequency bands used for the band-power representation.
        raw_field (str): Which raw EEG field to read (`rawEEG` per word or `rawData` per sentence).
        raw_window (int): Fixed time length (samples) raw EEG is padded/truncated to.
        normalize (Normalization): Feature normalisation. `zscore_channel`/`zscore_global` fit one mean/std
            across the whole cohort; `zscore_subject` fits and applies a **per-subject** mean/std, which removes the
            constant per-subject offset that otherwise makes subject identity the cheapest thing to encode
            (a direct attack on the "learns who, not what" failure mode). `minmax`/`none` as named.
        normalizer_fit (Literal['train', 'all']): Whether the normaliser (and imputer) statistics are fit on the
            training split only (`train`, default -- no leakage into val/test/held-out subject) or on the whole
            dataset before splitting (`all`, the legacy behaviour). `train` is required for honest held-out and LOSO numbers.
        montage_csv (str | None): Optional path to an electrode-montage CSV (`channel,region` or `channel,x,y,z`) used for
            scalp-region importance. When `None`, an *approximate* rostro-caudal channel partition is used and every
            region claim is flagged `approximate=True`.
        bandpass (tuple[float, float] | None): Optional `(low, high)` Hz Butterworth band-pass for raw EEG.
        missing (MissingConfig): Missing-value handling configuration.
        include_omitted (bool): Keep omitted words as masked tokens (preserves sentence sequence integrity).
            When `False`, omitted-word rows are dropped (as with `missing.method='drop'`).
        min_words (int): Drop sentences shorter than this many words.
        max_words (int | None): Drop sentences longer than this many words (`None` = no cap).
        cache_dir (str): Where processed artifacts are cached.
        cache_format (Literal['npz', 'parquet', 'hdf5']): On-disk cache format. Only `npz` is implemented; `parquet`/`hdf5` are reserved.
    """

    root: str = 'res/data/zuco_extracted'
    tasks: tuple[Task, ...] = ('SR', 'NR')
    subjects: tuple[str, ...] | None = None
    granularity: Granularity = 'word'
    representation: Representation = 'band_power'
    band_power_measures: tuple[EyeTrackingMeasure, ...] = ('TRT',)
    include_eye_tracking: bool = True
    eye_tracking_measures: tuple[str, ...] = (
        # SFD is dropped by default: it is ~60% missing and equals FFD wherever it is
        # present, so it adds a redundant, mostly-imputed column (see the performance
        # review). Re-add it explicitly if a run needs single-fixation duration.
        'FFD',
        'GD',
        'GPT',
        'TRT',
        'n_fixations',
        'mean_pupil',
    )
    bands: tuple[Band, ...] = BANDS
    raw_field: str = 'rawEEG'
    raw_window: int = 128
    normalize: Normalization = 'zscore_channel'
    normalizer_fit: Literal['train', 'all'] = 'train'
    montage_csv: str | None = None
    bandpass: tuple[float, float] | None = None
    missing: MissingConfig = field(default_factory=MissingConfig)
    include_omitted: bool = True
    min_words: int = 1
    max_words: int | None = None
    cache_dir: str = 'res/cache'
    cache_format: Literal['npz', 'parquet', 'hdf5'] = 'npz'


@dataclass
class ModelConfig:
    """ZTE encoder architecture.

    Attributes:
        frontend (FrontendName): `'band_power_mlp'` for band-power vectors or `'raw_conformer'` for raw time-series windows.
        embed_dim (int): Output embedding dimensionality (768 keeps it plug-compatible with the frozen LLM space used downstream in EEG-OT-CLIP).
        hidden_dim (int): Width of encoder hidden layers / transformer model dim.
        n_layers (int): Number of transformer / MLP blocks.
        n_heads (int): Attention heads (raw_conformer only).
        dropout (float): Dropout probability throughout the encoder.
        conformer_filters (int): Channel count after the temporal convolution.
        conformer_temporal_kernel (int): Temporal conv kernel size (acts as a learnable band-pass filter).
        pos_encoding (PosEncoding): Sequence positional-encoding scheme for the context transformer. `rope` (rotary, the default) injects relative
            position inside attention and generalises to any sentence length -- the current SOTA choice; `sinusoidal` adds the fixed Transformer encoding;
            `learned` adds an absolute learned table; `alibi` adds linear distance attention biases; `none` disables positional information (ablation). Each run records its scheme in the checkpoint config, so inference rebuilds the matching encoder.
        max_positions (int): Position table size for `learned`/`sinusoidal`.
        pool (PoolName): How per-word tokens are pooled into a sentence embedding.
        subject_conditioning (bool): Add a learned subject embedding (ZTE v1 is not yet subject-invariant; this exposes the knob for ablations).
        n_subjects (int): Vocabulary size for subject conditioning.
        projection_hidden (int): Width of the projection head's hidden layer.
    """

    frontend: FrontendName = 'band_power_mlp'
    embed_dim: int = 768
    hidden_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    conformer_filters: int = 40
    conformer_temporal_kernel: int = 25
    pos_encoding: PosEncoding = 'rope'
    max_positions: int = 512
    pool: PoolName = 'attention'
    subject_conditioning: bool = False
    n_subjects: int = 12
    n_tasks: int = 3
    projection_hidden: int = 512


@dataclass
class ObjectiveConfig:
    """Self-supervised training objective and its hyper-parameters.

    Attributes:
        name (ObjectiveName): Which objective to optimise --
            `skipgram`/`cbow` (word2vec analogues with InfoNCE),
            `masked` (BERT/data2vec/MAEEG style), or
            `cpc` (wav2vec/BENDR style).
        temperature (float): Softmax temperature for contrastive (InfoNCE) losses.
        context_window (int): Number of neighbouring words on each side used as context.
        mask_ratio (float): Fraction of tokens masked for the `'masked'` objective.
        masked_target (Literal['reconstruct', 'latent']): Reconstruct raw features (`reconstruct`) or predict an EMA-teacher latent (`latent`, the data2vec variant).
        ema_decay (float): Starting teacher EMA decay for `masked_target='latent'`.
        ema_decay_end (float): Final teacher EMA decay. The decay is ramped linearly from `ema_decay` to `ema_decay_end`
            over training (data2vec schedule): a fast-moving teacher early (more signal) that stabilises late. Set equal to `ema_decay` for a flat schedule.
        teacher_variance_floor (float): Minimum per-dimension std enforced when normalising the data2vec teacher target
            **across tokens** (not per-token). This is the anti-collapse fix for the masked objective -- a per-token
            LayerNorm target leaves between-token variance unconstrained and lets teacher and student co-collapse to a constant.
        cpc_steps (int): How many future steps CPC predicts.
        variance_weight (float): Weight of the VICReg variance-hinge term (0 disables). Penalises any embedding dimension
            whose batch std falls below `variance_target`, which is what prevents the InfoNCE/L1 objectives from collapsing
            into ~15 of 768 dimensions. The single biggest metric mover in the performance review.
        covariance_weight (float): Weight of the VICReg covariance term (0 disables). Pushes off-diagonal feature
            covariances toward zero so dimensions carry decorrelated information (higher effective rank).
        variance_target (float): Target per-dimension std (`gamma`) for the variance-hinge term.
        cross_subject_positives (bool): For skip-gram/CBOW, build contrastive positives from the **same stimulus read by
            different subjects** (using the batch's `content_id`) instead of same-subject neighbours. This turns subject
            identity from a shortcut into a nuisance the loss must remove. Requires a stimulus-grouped batch sampler to be
            effective; falls back to within-sentence neighbours when no cross-subject positive is present in the batch.
        subject_adversary_weight (float): Weight of a gradient-reversal subject-adversary loss (0 disables). An auxiliary
            head tries to classify the subject from the token hiddens; the reversed gradient trains the encoder to *hide*
            subject identity, directly lowering subject decodability toward chance.
        anisotropy_weight (float): Weight of an anti-cone *uniformity* penalty (0 disables). A Wang & Isola uniformity term spreads the
            L2-normalised embeddings over the sphere so their angular arrangement cannot degenerate. It complements `whiten`
            (which removes the shared-mean cone) by keeping directions well spread; pair both with VICReg (variance + covariance).
        whiten (bool): If `True`, the exported embeddings are ZCA-whitened at evaluation (centre + decorrelate + equalise variance). This is the
            direct fix for the "cone" (anisotropy ~0.997) and dimensional collapse the review found: centring removes the dominant shared direction
            (anisotropy -> ~0) and whitening spreads variance across all dimensions (effective rank -> full).  Because it is label-free, all downstream
            metrics are recomputed on the whitened space, so the report honestly shows whether content (retrieval, clustering) survives the fix.
        meaning_positives (bool): If `True` (skip-gram), also draw positive pairs from the *same content word occurring in
            different sentences* (subject-agnostic word identity), not only the same stimulus token. This gives the
            "same meaning across contexts" structure room to grow instead of memorising which passage a word came from.
        stimulus_adversary_weight (float): Weight of a second gradient-reversal adversary that tries to predict *which
            stimulus/passage* a token came from (0 disables). It removes the "which of the sentence-sets" shortcut so the
            model must encode content rather than passage identity. Requires `content_id` in the batch.
    """

    name: ObjectiveName = 'skipgram'
    temperature: float = 0.07
    context_window: int = 2
    mask_ratio: float = 0.5
    masked_target: Literal['reconstruct', 'latent'] = 'latent'
    ema_decay: float = 0.999
    ema_decay_end: float = 0.9999
    teacher_variance_floor: float = 1e-4
    cpc_steps: int = 4
    variance_weight: float = 0.0
    covariance_weight: float = 0.0
    variance_target: float = 1.0
    anisotropy_weight: float = 0.0
    whiten: bool = False
    cross_subject_positives: bool = False
    meaning_positives: bool = False
    subject_adversary_weight: float = 0.0
    stimulus_adversary_weight: float = 0.0


@dataclass
class TrainConfig:
    """Optimisation, scheduling, logging and checkpointing.

    Attributes:
        epochs (int): Number of passes over the training split.
        batch_size (int): Sentences (or words) per optimisation step.
        lr (float): Peak learning rate.
        weight_decay (float): AdamW weight decay.
        warmup_ratio: Fraction of total steps spent linearly warming up.
        scheduler (SchedulerName): Post-warmup learning-rate schedule.
        grad_accum_steps (int): Micro-batches accumulated per optimiser step.
        grad_clip (float): Global gradient-norm clip (`0` disables).
        device (DeviceKind | Literal['auto']): Backend preference passed to `resolve_device`.
        precision (PrecisionPreference): Mixed-precision preference.
        num_workers (int): DataLoader worker processes.
        split (SplitStrategy): Train/val split strategy.
        val_fraction (float): Validation fraction for random/by_sentence splits.
        test_fraction (float): Held-out test fraction (`0` disables). Defaults to `0.1` so evaluation reports on data the
            encoder never trained on (the review found the shipped runs were scored in-sample). For `random`/`by_sentence`/`by_stimulus`
            a disjoint test set is carved out and evaluation runs on it; for `by_subject_loso` the held-out subject is always the test set regardless of this value.
        loso_holdout_subject (str | None): Held-out subject for `by_subject_loso`.
        seed (int): Global RNG seed.
        deterministic (bool): Request deterministic cuDNN kernels for byte-for-byte reproducible CUDA runs (slower). Always seeds Python/NumPy/Torch.
        log_every (int): Log training metrics every N optimiser steps.
        eval_every (int): Run validation every N epochs.
        ckpt_dir (str): Local checkpoint directory.
        ckpt_keep_last (int): How many recent checkpoints to retain.
        tensorboard: Enable TensorBoard logging if installed.
        drive_backup_dir (str | None): Optional Google Drive folder id/path to mirror checkpoints to (see `zte.data.remote`).
        compile_model (bool): Apply `torch.compile` (skipped on MPS/CPU).

    """

    epochs: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    scheduler: SchedulerName = 'cosine'
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    device: Literal['auto', 'cpu', 'cuda', 'mps'] = 'auto'
    precision: Literal['auto', 'fp32', 'fp16', 'bf16'] = 'auto'
    num_workers: int = 0
    split: SplitStrategy = 'by_sentence'
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    loso_holdout_subject: str | None = None
    seed: int = 42
    deterministic: bool = False
    log_every: int = 10
    eval_every: int = 1
    ckpt_dir: str = 'res/checkpoints'
    ckpt_keep_last: int = 3
    tensorboard: bool = False
    drive_backup_dir: str | None = None
    compile_model: bool = False


@dataclass
class ZTEConfig:
    """Top-level configuration aggregating every sub-config.

    Attributes:
        dataset (DatasetConfig): Dataset construction options.
        model (ModelConfig): Encoder architecture.
        objective (ObjectiveConfig): Self-supervised objective.
        train (TrainConfig): Optimisation / logging / checkpointing.
        run_name (str): Identifier used in log/checkpoint paths.

    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run_name: str = 'zte-run'

    def to_dict(self) -> dict[str, Any]:
        """Returns a plain (YAML-safe) nested dict of the whole config."""
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> Path:
        """Writes the config to `path` as YAML and returns the path.

        Args:
            path (str | Path): Destination `.yaml` file (parent dirs are created).

        Returns:
            The written `pathlib.Path`.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding='utf-8')
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZTEConfig:
        """Builds a `ZTEConfig` from a nested dict, coercing tuples/types.

        Args:
            data (dict[str, Any]): A nested mapping such as one produced by :meth:`to_dict` or parsed from YAML.

        Returns:
            A fully constructed config with sub-dataclasses rebuilt `ZTEConfig`.

        """
        return cls(
            dataset=_build(DatasetConfig, data.get('dataset', {})),
            model=_build(ModelConfig, data.get('model', {})),
            objective=_build(ObjectiveConfig, data.get('objective', {})),
            train=_build(TrainConfig, data.get('train', {})),
            run_name=data.get('run_name', 'zte-run'),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ZTEConfig:
        """Loads a :class:`ZTEConfig` from a YAML file.

        Args:
            path (str | Path): Path to a YAML config previously written by :meth:`to_yaml`.

        Returns:
            ZTEConfig: The parsed config.

        """
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        return cls.from_dict(data)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Reconstructs a (possibly nested) dataclass, coercing lists back to tuples.

    Args:
        cls (type): The dataclass type to instantiate.
        data (dict[str, Any]): Field values, typically parsed from YAML where tuples became lists.

    Returns:
        An instance of `cls` with type-appropriate field values.

    """
    if not dataclasses.is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints.get(f.name)
        if dataclasses.is_dataclass(_strip_optional(hint)) and isinstance(value, dict):
            kwargs[f.name] = _build(_strip_optional(hint), value)
        elif _is_tuple_hint(hint) and isinstance(value, list):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _strip_optional(hint: Any) -> Any:
    """Returns the non-`None` member of an `X | None` hint, else `hint`."""
    args = [a for a in get_args(hint) if a is not type(None)]
    return args[0] if args and len(args) == 1 else hint


def _is_tuple_hint(hint: Any) -> bool:
    """Returns whether a type hint resolves to a `tuple[...]` type."""
    origin = getattr(hint, '__origin__', None)
    if origin is tuple:
        return True
    for arg in get_args(hint):
        if getattr(arg, '__origin__', None) is tuple:
            return True
    return False
