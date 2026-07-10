"""Typed, serialisable configuration for the EEG→language decode pipeline.

Mirrors the style of :mod:`zte.config`: every knob lives in a dataclass so configs
are explicit, IDE-discoverable and round-trip cleanly to YAML. The top-level
:class:`DecoderConfig` aggregates text-encoder, EEG-OT-CLIP alignment and
generative (prefix-LM) sub-configs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args, get_type_hints

import yaml

type TextPooling = Literal['cls', 'mean', 'pooler']
type TextBackend = Literal['auto', 'transformers', 'hash']
type AlignLevel = Literal['word', 'sentence']
type SchedulerName = Literal['cosine', 'linear', 'constant']
type SplitStrategy = Literal['random', 'by_sentence', 'by_subject_loso', 'by_task']
type DevicePref = Literal['auto', 'cpu', 'cuda', 'mps']
type PrecisionPref = Literal['auto', 'fp32', 'fp16', 'bf16']
type GenerativeBackend = Literal['auto', 'transformers', 'toy']
type DecodeMode = Literal['retrieval', 'prefix_lm', 'both']


@dataclass
class TextEncoderConfig:
    """Frozen (or lightly tuned) text encoder that defines the language side of the shared space.

    Attributes:
        model_name: HuggingFace model id, or ``'hash'`` for the deterministic fallback.
        embed_dim: Output embedding dimensionality (768 matches ZTE / RoBERTa / BART).
        max_length: Tokenizer truncation length.
        pooling: How token states become a single vector (``mean`` / ``cls`` / ``pooler``).
        normalize: L2-normalise embeddings before returning.
        cache_dir: Disk cache for precomputed text embeddings.
        backend: ``'auto'`` prefers transformers when installed, else hash; force with
            ``'transformers'`` or ``'hash'``.
        freeze: Freeze encoder parameters (typical for CLIP-style alignment).
        device: Device preference passed to :func:`zte.device.resolve_device`.
    """

    model_name: str = 'roberta-base'
    embed_dim: int = 768
    max_length: int = 128
    pooling: TextPooling = 'mean'
    normalize: bool = True
    cache_dir: str = 'res/cache/text_embeddings'
    backend: TextBackend = 'auto'
    freeze: bool = True
    device: DevicePref = 'auto'


@dataclass
class AlignConfig:
    """EEG-OT-CLIP alignment hyper-parameters.

    Attributes:
        eeg_dim: Dimensionality of ZTE embeddings (input to the EEG projector).
        text_dim: Dimensionality of text-encoder embeddings.
        proj_dim: Shared projection space dimensionality.
        proj_hidden: Hidden width of the MLP projectors.
        temperature: Softmax temperature for InfoNCE.
        lambda_infonce: Weight on the symmetric InfoNCE term.
        lambda_ot: Weight on the Sinkhorn OT term.
        ot_epsilon: Entropic regularisation for Sinkhorn.
        ot_iters: Number of Sinkhorn iterations.
        freeze_eeg_encoder: Keep the ZTE encoder frozen during alignment.
        level: Pair at ``'sentence'`` or ``'word'`` granularity.
        epochs: Alignment training epochs.
        batch_size: Paired samples per step.
        lr: Peak AdamW learning rate.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of steps spent warming up.
        scheduler: Post-warmup LR schedule.
        grad_clip: Global gradient-norm clip (``0`` disables).
        device: Device preference.
        precision: Mixed-precision preference.
        seed: RNG seed.
        split: Train/val split strategy (same vocabulary as ZTE training).
        val_fraction: Validation fraction for random / by-sentence splits.
        loso_holdout_subject: Held-out subject for ``by_subject_loso``.
        ckpt_dir: Checkpoint directory for the aligner.
        log_every: Log every N optimiser steps.
        eval_every: Validate every N epochs.
        tensorboard: Enable TensorBoard if installed.
        run_name: Identifier used in log / checkpoint paths.
    """

    eeg_dim: int = 768
    text_dim: int = 768
    proj_dim: int = 768
    proj_hidden: int = 512
    temperature: float = 0.07
    lambda_infonce: float = 1.0
    lambda_ot: float = 0.1
    ot_epsilon: float = 0.05
    ot_iters: int = 20
    freeze_eeg_encoder: bool = True
    level: AlignLevel = 'sentence'
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    scheduler: SchedulerName = 'cosine'
    grad_clip: float = 1.0
    device: DevicePref = 'auto'
    precision: PrecisionPref = 'auto'
    seed: int = 42
    split: SplitStrategy = 'by_subject_loso'
    val_fraction: float = 0.1
    loso_holdout_subject: str | None = None
    ckpt_dir: str = 'res/decode/alignment'
    log_every: int = 10
    eval_every: int = 1
    tensorboard: bool = False
    run_name: str = 'eeg-ot-clip'


@dataclass
class GenerativeConfig:
    """Autoregressive / prefix-LM decoder settings.

    Attributes:
        lm_name: HuggingFace causal LM id (e.g. ``'gpt2'``), ignored for ``backend='toy'``.
        backend: ``'auto'`` prefers transformers when installed, else the toy char LM.
        prefix_len: Number of soft-prompt tokens mapped from each EEG embedding.
        prefix_dim: Width of the EEG embedding fed into the prefix mapper.
        freeze_lm: Freeze the language-model backbone.
        max_new_tokens: Generation length cap.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling cutoff.
        epochs: Generative training epochs.
        batch_size: Sequences per step.
        lr: Peak AdamW learning rate.
        weight_decay: AdamW weight decay.
        warmup_ratio: Warmup fraction of total steps.
        device: Device preference.
        precision: Mixed-precision preference.
        seed: RNG seed.
        ckpt_dir: Checkpoint directory for the generative decoder.
        run_name: Identifier used in log / checkpoint paths.
    """

    lm_name: str = 'gpt2'
    backend: GenerativeBackend = 'auto'
    prefix_len: int = 8
    prefix_dim: int = 768
    freeze_lm: bool = True
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.9
    epochs: int = 10
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    device: DevicePref = 'auto'
    precision: PrecisionPref = 'auto'
    seed: int = 42
    ckpt_dir: str = 'res/decode/generative'
    run_name: str = 'eeg-prefix-lm'


@dataclass
class DecoderConfig:
    """Top-level decode pipeline configuration.

    Attributes:
        text: Text-encoder sub-config.
        align: EEG-OT-CLIP alignment sub-config.
        generative: Prefix-LM / generative sub-config.
        mode: Which decode path(s) to run (``retrieval``, ``prefix_lm``, or ``both``).
        zte_ckpt: Optional path to a trained ZTE checkpoint for EEG embeddings.
        bundle: Optional path to a saved :class:`~zte.data.dataset.ZuCoDataset` bundle.
        out_dir: Root output directory for decode artifacts.
        run_name: Identifier used in log / output paths.
    """

    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    generative: GenerativeConfig = field(default_factory=GenerativeConfig)
    mode: DecodeMode = 'retrieval'
    zte_ckpt: str | None = None
    bundle: str | None = None
    out_dir: str = 'res/decode'
    run_name: str = 'zte-decode'

    def to_dict(self) -> dict[str, Any]:
        """Returns a plain (YAML-safe) nested dict of the whole config."""
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> Path:
        """Writes the config to ``path`` as YAML and returns the path.

        Args:
            path: Destination ``.yaml`` file (parent dirs are created).

        Returns:
            The written :class:`pathlib.Path`.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding='utf-8')
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecoderConfig:
        """Builds a :class:`DecoderConfig` from a nested dict.

        Args:
            data: Nested mapping such as one produced by :meth:`to_dict` or parsed from YAML.

        Returns:
            A fully constructed config with sub-dataclasses rebuilt.
        """
        return cls(
            text=_build(TextEncoderConfig, data.get('text', {})),
            align=_build(AlignConfig, data.get('align', {})),
            generative=_build(GenerativeConfig, data.get('generative', {})),
            mode=data.get('mode', 'retrieval'),
            zte_ckpt=data.get('zte_ckpt'),
            bundle=data.get('bundle'),
            out_dir=data.get('out_dir', 'res/decode'),
            run_name=data.get('run_name', 'zte-decode'),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> DecoderConfig:
        """Loads a :class:`DecoderConfig` from a YAML file.

        Args:
            path: Path to a YAML config previously written by :meth:`to_yaml`.

        Returns:
            The parsed config.
        """
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        return cls.from_dict(data)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Reconstructs a (possibly nested) dataclass, coercing lists back to tuples.

    Args:
        cls: The dataclass type to instantiate.
        data: Field values, typically parsed from YAML where tuples became lists.

    Returns:
        An instance of ``cls`` with type-appropriate field values.
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
    """Returns the non-``None`` member of an ``X | None`` hint, else ``hint``."""
    args = [a for a in get_args(hint) if a is not type(None)]
    return args[0] if args and len(args) == 1 else hint


def _is_tuple_hint(hint: Any) -> bool:
    """Returns whether a type hint resolves to a ``tuple[...]`` type."""
    origin = getattr(hint, '__origin__', None)
    if origin is tuple:
        return True
    for arg in get_args(hint):
        if getattr(arg, '__origin__', None) is tuple:
            return True
    return False
