"""Training loops for EEG-OT-CLIP alignment and prefix-LM decoding.

:func:`run_alignment` trains an :class:`~zte.decode.alignment.OTCLIPAligner` on
precomputed EEG/text pairs. :func:`run_decode_training` trains a
:class:`~zte.decode.decoders.PrefixLanguageDecoder` (toy or HF) with teacher
forcing. Both keep the ZTE encoder frozen by default — only projectors / prefix
mappers are optimised.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from zte.decode.alignment import OTCLIPAligner
from zte.decode.config import AlignConfig, DecoderConfig, GenerativeConfig
from zte.decode.decoders import PrefixLanguageDecoder
from zte.decode.pairing import make_paired_loader
from zte.device import autocast, resolve_device, seed_everything
from zte.logging_utils import get_logger, progress
from zte.training.scheduler import build_scheduler

_LOG = get_logger('decode.train')


@dataclass(slots=True)
class AlignmentArtifacts:
    """Outputs of :func:`run_alignment`.

    Attributes:
        aligner: Trained aligner (eval mode).
        history: Per-epoch metric history.
        ckpt_path: Path to the best checkpoint, if saved.
    """

    aligner: OTCLIPAligner
    history: dict[str, list[float]]
    ckpt_path: Path | None = None


@dataclass(slots=True)
class GenerativeArtifacts:
    """Outputs of :func:`run_decode_training`.

    Attributes:
        decoder: Trained prefix-LM decoder.
        history: Per-epoch metric history.
        ckpt_path: Path to the best checkpoint, if saved.
    """

    decoder: PrefixLanguageDecoder
    history: dict[str, list[float]]
    ckpt_path: Path | None = None


def run_alignment(
    eeg: np.ndarray,
    text_emb: np.ndarray,
    texts: list[str] | None = None,
    config: AlignConfig | DecoderConfig | None = None,
    val_fraction: float | None = None,
) -> AlignmentArtifacts:
    """Trains an OT-CLIP aligner on precomputed paired embeddings.

    Args:
        eeg: EEG embeddings ``(N, eeg_dim)``.
        text_emb: Text embeddings ``(N, text_dim)``, row-aligned with ``eeg``.
        texts: Optional surface strings (stored in the checkpoint for retrieval).
        config: :class:`AlignConfig` or a :class:`DecoderConfig` (uses ``.align``).
        val_fraction: Override for the validation hold-out fraction.

    Returns:
        :class:`AlignmentArtifacts` with the trained aligner and history.
    """
    align_cfg = _align_config(config)
    seed_everything(align_cfg.seed)
    device = resolve_device(align_cfg.device, align_cfg.precision)
    texts = texts if texts is not None else [''] * len(eeg)

    n = len(eeg)
    if n == 0:
        aligner = OTCLIPAligner(align_cfg).to(device.device).eval()
        return AlignmentArtifacts(aligner=aligner, history={})

    rng = np.random.default_rng(align_cfg.seed)
    order = rng.permutation(n)
    frac = align_cfg.val_fraction if val_fraction is None else val_fraction
    n_val = int(round(n * frac)) if n > 1 else 0
    val_idx = order[:n_val]
    train_idx = order[n_val:] if n_val < n else order

    train_loader = make_paired_loader(
        eeg[train_idx],
        text_emb[train_idx],
        [texts[i] for i in train_idx],
        batch_size=align_cfg.batch_size,
        shuffle=True,
    )
    val_loader = None
    if len(val_idx) > 0:
        val_loader = make_paired_loader(
            eeg[val_idx],
            text_emb[val_idx],
            [texts[i] for i in val_idx],
            batch_size=align_cfg.batch_size,
            shuffle=False,
        )

    # Sync projector dims with actual arrays when they differ from defaults.
    align_cfg = AlignConfig(
        **{
            **asdict(align_cfg),
            'eeg_dim': int(eeg.shape[1]),
            'text_dim': int(text_emb.shape[1]),
        }
    )
    aligner = OTCLIPAligner(align_cfg).to(device.device)
    optimizer = torch.optim.AdamW(
        aligner.parameters(), lr=align_cfg.lr, weight_decay=align_cfg.weight_decay
    )
    total_steps = max(1, len(train_loader) * align_cfg.epochs)
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(total_steps * align_cfg.warmup_ratio),
        kind=align_cfg.scheduler,
    )

    history: dict[str, list[float]] = defaultdict(list)
    best_val = float('inf')
    ckpt_path: Path | None = None
    ckpt_dir = Path(align_cfg.ckpt_dir) / align_cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, align_cfg.epochs + 1):
        aligner.train()
        train_metrics = _run_align_epoch(
            aligner,
            train_loader,
            device.device,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=align_cfg.grad_clip,
            log_every=align_cfg.log_every,
            train=True,
            precision=align_cfg.precision,
            step_offset=global_step,
        )
        global_step += len(train_loader)
        history['train_loss'].append(train_metrics['loss'])
        history['train_loss_infonce'].append(train_metrics['loss_infonce'])
        history['train_loss_ot'].append(train_metrics['loss_ot'])
        history['train_alignment_diag_mean'].append(train_metrics['alignment_diag_mean'])

        val_metrics = train_metrics
        if val_loader is not None and (epoch % max(1, align_cfg.eval_every) == 0):
            aligner.eval()
            val_metrics = _run_align_epoch(
                aligner,
                val_loader,
                device.device,
                optimizer=None,
                scheduler=None,
                grad_clip=0.0,
                log_every=align_cfg.log_every,
                train=False,
                precision=align_cfg.precision,
            )
            history['val_loss'].append(val_metrics['loss'])
            history['val_alignment_diag_mean'].append(val_metrics['alignment_diag_mean'])

        _LOG.info(
            'align epoch %d/%d | train_loss=%.4f | val_loss=%.4f | diag=%.3f',
            epoch,
            align_cfg.epochs,
            train_metrics['loss'],
            val_metrics['loss'],
            val_metrics['alignment_diag_mean'],
        )
        if val_metrics['loss'] <= best_val:
            best_val = val_metrics['loss']
            ckpt_path = ckpt_dir / 'best.pt'
            torch.save(
                {
                    'aligner': aligner.state_dict(),
                    'config': asdict(align_cfg),
                    'texts': texts,
                    'text_bank_emb': np.asarray(text_emb, dtype=np.float32),
                    'epoch': epoch,
                },
                ckpt_path,
            )

    last_path = ckpt_dir / 'last.pt'
    torch.save(
        {
            'aligner': aligner.state_dict(),
            'config': asdict(align_cfg),
            'texts': texts,
            'text_bank_emb': np.asarray(text_emb, dtype=np.float32),
        },
        last_path,
    )
    if ckpt_path is None:
        ckpt_path = last_path
    aligner.eval()
    return AlignmentArtifacts(aligner=aligner, history=dict(history), ckpt_path=ckpt_path)


def run_decode_training(
    eeg: np.ndarray,
    texts: list[str],
    config: GenerativeConfig | DecoderConfig | None = None,
) -> GenerativeArtifacts:
    """Trains a prefix-LM decoder on EEG→text pairs.

    Args:
        eeg: EEG (or aligned) embeddings ``(N, D)``.
        texts: Target strings of length ``N``.
        config: :class:`GenerativeConfig` or :class:`DecoderConfig` (uses ``.generative``).

    Returns:
        :class:`GenerativeArtifacts` with the trained decoder and history.
    """
    gen_cfg = _generative_config(config)
    seed_everything(gen_cfg.seed)
    device = resolve_device(gen_cfg.device, gen_cfg.precision)

    # Ensure prefix_dim matches the EEG width.
    gen_cfg = GenerativeConfig(**{**asdict(gen_cfg), 'prefix_dim': int(eeg.shape[1])})
    decoder = PrefixLanguageDecoder(gen_cfg, texts=list(texts))
    decoder.to(device.device)

    # Only train parameters that require grad (prefix mapper; optionally LM).
    params = [p for p in decoder.parameters() if p.requires_grad]
    if not params:
        _LOG.warning('No trainable parameters in PrefixLanguageDecoder; skipping optimisation.')
        return GenerativeArtifacts(decoder=decoder.eval(), history={})

    optimizer = torch.optim.AdamW(params, lr=gen_cfg.lr, weight_decay=gen_cfg.weight_decay)
    n = len(texts)
    order = np.arange(n)
    rng = np.random.default_rng(gen_cfg.seed)
    history: dict[str, list[float]] = defaultdict(list)
    ckpt_dir = Path(gen_cfg.ckpt_dir) / gen_cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    ckpt_path: Path | None = None

    steps_per_epoch = max(1, (n + gen_cfg.batch_size - 1) // gen_cfg.batch_size)
    total_steps = steps_per_epoch * gen_cfg.epochs
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(total_steps * gen_cfg.warmup_ratio),
        kind='cosine',
    )

    for epoch in range(1, gen_cfg.epochs + 1):
        decoder.train()
        rng.shuffle(order)
        losses: list[float] = []
        for start in progress(
            range(0, n, gen_cfg.batch_size),
            description=f'decode epoch {epoch}',
        ):
            idx = order[start : start + gen_cfg.batch_size]
            batch_eeg = torch.from_numpy(eeg[idx]).to(device.device)
            batch_texts = [texts[i] for i in idx]
            input_ids, attention_mask = decoder.encode_texts(batch_texts)
            input_ids = input_ids.to(device.device)
            attention_mask = attention_mask.to(device.device)
            with autocast(device):  # DeviceSpec context
                loss, _ = decoder(batch_eeg, input_ids, attention_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        history['train_loss'].append(mean_loss)
        _LOG.info('decode epoch %d/%d | loss=%.4f', epoch, gen_cfg.epochs, mean_loss)
        if mean_loss <= best_loss:
            best_loss = mean_loss
            ckpt_path = ckpt_dir / 'best.pt'
            torch.save(
                {
                    'decoder': decoder.state_dict(),
                    'config': asdict(gen_cfg),
                    'backend': decoder.backend,
                    'char_to_id': getattr(decoder, 'char_to_id', {}),
                    'epoch': epoch,
                },
                ckpt_path,
            )

    decoder.eval()
    return GenerativeArtifacts(decoder=decoder, history=dict(history), ckpt_path=ckpt_path)


def _align_config(config: AlignConfig | DecoderConfig | None) -> AlignConfig:
    if config is None:
        return AlignConfig()
    if isinstance(config, DecoderConfig):
        return config.align
    return config


def _generative_config(config: GenerativeConfig | DecoderConfig | None) -> GenerativeConfig:
    if config is None:
        return GenerativeConfig()
    if isinstance(config, DecoderConfig):
        return config.generative
    return config


def _run_align_epoch(
    aligner: OTCLIPAligner,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    grad_clip: float,
    log_every: int,
    train: bool,
    precision: str,
    step_offset: int = 0,
) -> dict[str, float]:
    """Runs one alignment epoch (train or eval)."""
    del precision  # autocast handled via resolve_device at caller; keep signature stable
    totals: dict[str, float] = defaultdict(float)
    n_batches = 0
    for step, (eeg_batch, text_batch, _texts) in enumerate(loader):
        eeg_dev = eeg_batch.to(device)
        text_dev = text_batch.to(device)
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = aligner(eeg_dev, text_dev)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(aligner.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if (step_offset + step) % max(1, log_every) == 0:
                _LOG.debug(
                    'align step %d | loss=%.4f nce=%.4f ot=%.4f',
                    step_offset + step,
                    metrics['loss'],
                    metrics['loss_infonce'],
                    metrics['loss_ot'],
                )
        else:
            with torch.no_grad():
                _, metrics = aligner(eeg_dev, text_dev)
        for k, v in metrics.items():
            totals[k] += float(v)
        n_batches += 1
    if n_batches == 0:
        return {'loss': 0.0, 'loss_infonce': 0.0, 'loss_ot': 0.0, 'alignment_diag_mean': 0.0}
    return {k: v / n_batches for k, v in totals.items()}
