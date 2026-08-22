"""Per-subject whitening of raw EEG windows, and the covariance descriptor that identifies a brain.

Both are computed from the subject's own voltages alone -- no label, split or stimulus -- so both apply unchanged to
a person the model has never seen. See `docs/SUBJECT_ALIGNMENT.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from zte.logging_utils import get_logger, progress

_LOG = get_logger('data.alignment')

#: Trials below this count fall back to the cohort reference; a covariance from fewer is mostly noise.
_MIN_TRIALS: int = 8

#: Trials sampled per reference. A 105x105 covariance is well determined long before this, and the raw
#: tensor is tens of GB, so estimating from everything buys nothing and costs a runtime.
_MAX_REF_TRIALS: int = 4000

#: Windows converted to float64 at once. Bounds the working set to a few hundred MB per chunk.
_COV_CHUNK: int = 512


def _spd_power(cov: np.ndarray, power: float, shrinkage: float, eps: float) -> np.ndarray:
    """Raises a symmetric positive-definite matrix to `power` via its eigendecomposition."""
    d = cov.shape[0]
    mu = float(np.trace(cov) / max(d, 1))

    # Ledoit-Wolf style shrinkage toward a scaled identity keeps the root well-conditioned for short recordings.
    cov = (1.0 - shrinkage) * cov + shrinkage * mu * np.eye(d, dtype=np.float64)

    w, v = np.linalg.eigh(cov)
    w = np.clip(w, eps, None)
    return (v * (w**power)) @ v.T


def _accel_device() -> Any | None:
    """Returns a torch GPU device when one is available, else `None` for the numpy path.

    Alignment is all batched linear algebra over a tensor too big for RAM; running it on the accelerator
    that is otherwise idle during data prep is both faster and keeps the working set off system memory.
    """
    try:
        import torch
    except ImportError:
        return None

    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    return None


def _matmul_backend() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Returns a `(C,C) x (N,C,T) -> (N,C,T)` multiplier, on the GPU when one is available."""
    device = _accel_device()
    if device is None:
        return lambda w, x: np.einsum('cd,ndt->nct', w, x, optimize=True)

    import torch

    def _gpu(w: np.ndarray, x: np.ndarray) -> np.ndarray:
        wt = torch.from_numpy(np.ascontiguousarray(w)).to(device)
        xt = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        return torch.matmul(wt, xt).cpu().numpy()

    return _gpu


def _sample_rows(mask: np.ndarray, limit: int = _MAX_REF_TRIALS) -> np.ndarray:
    """Evenly-spaced row indices where `mask` is set, capped at `limit`.

    Indices are subsampled BEFORE any fancy-indexing, because `raw[mask]` on a boolean would materialise a
    copy of the whole (tens of GB) tensor first.
    """
    idx = np.flatnonzero(mask)
    if len(idx) > limit:
        idx = idx[np.linspace(0, len(idx) - 1, limit).astype(int)]
    return idx


def _channel_covariance(raw: np.ndarray, idx: np.ndarray, eps: float) -> tuple[np.ndarray, float]:
    """Mean per-trial channel covariance `(n_channels, n_channels)` over the rows `idx` of `raw`, and their RMS.

    Accumulated in chunks so the float64 working set stays bounded regardless of how many trials are
    requested -- the raw tensor does not fit in memory twice. The trace-normalised covariance keeps only the
    SHAPE of a subject's channel geometry, so the amplitude is carried out separately as the second return
    value rather than being recovered by a second pass over the tensor.
    """
    n_ch = int(raw.shape[1])
    device = _accel_device()
    if device is not None:
        import torch

        # MPS has no float64; accumulating in float32 there is fine for a 105x105 covariance.
        dtype = torch.float32 if device.type == 'mps' else torch.float64
        acc_t = torch.zeros((n_ch, n_ch), dtype=dtype, device=device)
        trace_acc = 0.0
        seen = 0
        for start in range(0, len(idx), _COV_CHUNK):
            block = np.ascontiguousarray(raw[idx[start : start + _COV_CHUNK]])
            x = torch.nan_to_num(torch.from_numpy(block).to(device, dtype))
            t = x.shape[2]

            # Per-trial trace normalisation stops a few high-amplitude trials owning the reference.
            cov = (x @ x.transpose(1, 2)) / max(t, 1)
            trace = cov.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None]
            acc_t += (cov / trace.clamp_min(eps)).sum(dim=0)
            trace_acc += float(trace.sum().item())
            seen += len(block)
        cov_mean = (acc_t / max(seen, 1)).cpu().numpy().astype(np.float64)
        return cov_mean, _rms_from_trace(trace_acc, seen, n_ch, eps)

    acc = np.zeros((n_ch, n_ch), dtype=np.float64)
    trace_acc = 0.0
    seen = 0
    for start in range(0, len(idx), _COV_CHUNK):
        x = np.nan_to_num(np.asarray(raw[idx[start : start + _COV_CHUNK]], dtype=np.float64), nan=0.0)
        t = x.shape[2]

        # Per-trial trace normalisation stops a few high-amplitude trials owning the reference.
        cov = np.einsum('nct,ndt->ncd', x, x) / max(t, 1)
        trace = np.trace(cov, axis1=1, axis2=2)[:, None, None]
        acc += (cov / np.clip(trace, eps, None)).sum(axis=0)
        trace_acc += float(trace.sum())
        seen += len(x)

    return acc / max(seen, 1), _rms_from_trace(trace_acc, seen, n_ch, eps)


def _rms_from_trace(trace_acc: float, seen: int, n_ch: int, eps: float) -> float:
    """Root-mean-square voltage implied by a summed covariance trace, with the mean square floored at `eps`."""
    mean_square = trace_acc / max(seen, 1) / max(n_ch, 1)

    return float(np.sqrt(max(mean_square, eps)))


class RawSubjectAligner:
    """Whitens each subject's raw windows to a shared reference, and describes what whitening cannot remove.

    Note:
        The reference is estimated from trace-normalised trials, so it equalises the shape of a subject's channel
        covariance and leaves their overall amplitude -- the largest single carrier of subject identity -- intact.
        `match_amplitude` divides that amplitude out too, making the map scale-equivariant; it changes every
        downstream number, so it is off unless a config asks for it.

    Attributes:
        references (dict[str, np.ndarray]): Per-subject whitening maps `R_s^-1/2`.
        signatures (dict[str, np.ndarray]): Per-subject descriptors, computed before whitening.
        scales (dict[str, float]): Per-subject RMS voltage, divided out only when `match_amplitude` is set.
    """

    def __init__(
        self, shrinkage: float = 0.1, eps: float = 1e-8, n_regions: int = 8, match_amplitude: bool = False
    ) -> None:
        self.shrinkage = shrinkage
        self.eps = eps
        self.n_regions = n_regions
        self.match_amplitude = match_amplitude
        self.references: dict[str, np.ndarray] = {}
        self.signatures: dict[str, np.ndarray] = {}
        self.scales: dict[str, float] = {}
        self._global_reference: np.ndarray | None = None
        self._global_signature: np.ndarray | None = None
        self._global_scale: float = 1.0
        self._region_index: np.ndarray | None = None
        self._sig_mean: np.ndarray | None = None
        self._sig_std: np.ndarray | None = None

    @property
    def signature_dim(self) -> int:
        """Width of a signature vector: per-channel log-scale plus the region covariance tangent vector."""
        return 0 if self._global_signature is None else int(len(self._global_signature))

    def fit(
        self,
        raw: np.ndarray,
        subjects: np.ndarray,
        present: np.ndarray | None = None,
        region_index: np.ndarray | None = None,
    ) -> RawSubjectAligner:
        """Estimates a whitening map and a descriptor per subject from their own windows.

        Args:
            raw (np.ndarray): `(n_words, n_channels, time_steps)` windows.
            subjects (np.ndarray): `(n_words,)` subject codes.
            present (np.ndarray | None): `(n_words,)` mask of real (non-imputed) tokens.
            region_index (np.ndarray | None): `(n_channels,)` scalp-region id per electrode; contiguous
                blocks are assumed when omitted.
        """
        n_ch = raw.shape[1]
        if region_index is None:
            region_index = np.minimum(np.arange(n_ch) * self.n_regions // max(n_ch, 1), self.n_regions - 1)
        self._region_index = np.asarray(region_index, dtype=int)

        subjects = np.asarray(subjects)
        mask = np.ones(len(raw), dtype=bool) if present is None else np.asarray(present, dtype=bool)

        _LOG.info(
            'Estimating per-subject references from %d windows on %s ...',
            int(mask.sum()),
            _accel_device() or 'cpu',
        )

        # The cohort reference doubles as the fallback for a subject with too few usable trials.
        pooled, self._global_scale = _channel_covariance(raw, _sample_rows(mask), self.eps)
        self._global_reference = _spd_power(pooled, -0.5, self.shrinkage, self.eps)
        self._global_signature = self._signature_from(pooled)

        for code in np.unique(subjects):
            rows = (subjects == code) & mask
            if int(rows.sum()) < _MIN_TRIALS:
                _LOG.warning(
                    'Subject %s has <%d usable trials; using the cohort reference.',
                    code,
                    _MIN_TRIALS,
                )
                continue
            cov, scale = _channel_covariance(raw, _sample_rows(rows), self.eps)
            self.references[str(code)] = _spd_power(cov, -0.5, self.shrinkage, self.eps).astype(np.float32)
            self.signatures[str(code)] = self._signature_from(cov)
            self.scales[str(code)] = scale

        # Standardise across the cohort so the hypernetwork sees a zero-mean, unit-scale descriptor.
        stack = np.stack(list(self.signatures.values())) if self.signatures else self._global_signature[None, :]
        self._sig_mean = stack.mean(axis=0).astype(np.float32)
        self._sig_std = np.clip(stack.std(axis=0), 1e-3, None).astype(np.float32)

        _LOG.info('Fitted Euclidean alignment for %d subjects (%d channels).', len(self.references), n_ch)
        return self

    def _signature_from(self, cov: np.ndarray) -> np.ndarray:
        """Per-channel log scale plus the log-Euclidean tangent vector of the region-averaged covariance."""
        assert self._region_index is not None

        # Per-channel scale: impedance and cap contact.
        diag = np.log(np.clip(np.diag(cov), self.eps, None))

        # Average within scalp regions, then take the matrix log to land in a flat tangent space.
        n_reg = int(self._region_index.max()) + 1
        onehot = np.zeros((cov.shape[0], n_reg), dtype=np.float64)
        onehot[np.arange(cov.shape[0]), self._region_index] = 1.0
        counts = np.clip(onehot.sum(axis=0), 1.0, None)
        region_cov = (onehot.T @ cov @ onehot) / np.outer(counts, counts)

        w, v = np.linalg.eigh(_spd_power(region_cov, 1.0, self.shrinkage, self.eps))
        log_region = (v * np.log(np.clip(w, self.eps, None))) @ v.T

        # Only the upper triangle is independent; sqrt(2) on the off-diagonals keeps the vector isometric.
        iu = np.triu_indices(n_reg, k=1)
        tangent = np.concatenate([np.diag(log_region), np.sqrt(2.0) * log_region[iu]])
        return np.concatenate([diag, tangent]).astype(np.float32)

    def signature_for(self, subject: str) -> np.ndarray:
        """Standardised signature of `subject`, falling back to the cohort descriptor for an unknown one."""
        assert self._global_signature is not None
        sig = self.signatures.get(str(subject), self._global_signature)
        if self._sig_mean is None or self._sig_std is None:
            return sig
        return ((sig - self._sig_mean) / self._sig_std).astype(np.float32)

    def transform(
        self,
        raw: np.ndarray,
        subjects: np.ndarray,
        out: np.ndarray | None = None,
        chunk: int = 1024,
    ) -> np.ndarray:
        """Whitens windows subject by subject, streaming through `chunk` trials at a time.

        Args:
            raw (np.ndarray): `(n_words, n_channels, time_steps)` windows; may be a read-only memmap.
            subjects (np.ndarray): `(n_words,)` subject codes.
            out (np.ndarray | None): Destination (typically a writable memmap); `None` writes in place.
            chunk (int): Trials per step. Bounds the working set; the full tensor is never duplicated.

        Returns:
            np.ndarray: The destination array.
        """
        assert self._global_reference is not None
        subjects = np.asarray(subjects)
        dest = raw if out is None else out
        mm = _matmul_backend()
        codes = np.unique(subjects)
        _LOG.info(
            'Whitening %d windows for %d subjects on %s ...',
            len(raw),
            len(codes),
            _accel_device() or 'cpu',
        )

        for n_done, code in enumerate(progress(codes, description='aligning subjects'), start=1):
            w = self.references.get(str(code), self._global_reference.astype(np.float32))

            # Folding the amplitude into the whitener costs nothing per trial, and a subject the fit never
            # saw borrows the cohort RMS along with the cohort reference.
            if self.match_amplitude:
                w = (w / self.scales.get(str(code), self._global_scale)).astype(np.float32)

            rows = np.flatnonzero(subjects == code)
            for start in range(0, len(rows), chunk):
                idx = rows[start : start + chunk]
                dest[idx] = mm(w, np.asarray(raw[idx], dtype=np.float32))
            _LOG.info('  aligned %s (%d windows) [%d/%d]', code, len(rows), n_done, len(codes))
        return dest

    def calibrate_subject(self, baseline_raw: np.ndarray, subject: str) -> RawSubjectAligner:
        """Registers a brand-new brain from an unlabelled baseline recording -- the zero-shot path.

        Args:
            baseline_raw (np.ndarray): `(n, n_channels, time_steps)` windows of the person reading anything.
            subject (str): Subject code to register the maps under.
        """
        if len(baseline_raw) < _MIN_TRIALS:
            _LOG.warning(
                'Calibration baseline for %s has %d trials; keeping the cohort reference.',
                subject,
                len(baseline_raw),
            )
            return self
        cov, scale = _channel_covariance(baseline_raw, _sample_rows(np.ones(len(baseline_raw), dtype=bool)), self.eps)
        self.references[str(subject)] = _spd_power(cov, -0.5, self.shrinkage, self.eps).astype(np.float32)
        self.signatures[str(subject)] = self._signature_from(cov)
        self.scales[str(subject)] = scale
        return self

    @property
    def state(self) -> dict[str, object]:
        """Picklable state, embedded in the checkpoint so inference reproduces the exact alignment."""
        return {
            'shrinkage': self.shrinkage,
            'eps': self.eps,
            'n_regions': self.n_regions,
            'match_amplitude': self.match_amplitude,
            'references': self.references,
            'signatures': self.signatures,
            'scales': self.scales,
            'global_reference': self._global_reference,
            'global_signature': self._global_signature,
            'global_scale': self._global_scale,
            'region_index': self._region_index,
            'sig_mean': self._sig_mean,
            'sig_std': self._sig_std,
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> RawSubjectAligner:
        """Rebuilds an aligner from `state`."""
        obj = cls(
            shrinkage=float(state['shrinkage']),  # type: ignore[arg-type]
            eps=float(state['eps']),  # type: ignore[arg-type]
            n_regions=int(state['n_regions']),  # type: ignore[arg-type]
            match_amplitude=bool(state.get('match_amplitude', False)),
        )
        obj.references = dict(state['references'])  # type: ignore[arg-type]
        obj.signatures = dict(state['signatures'])  # type: ignore[arg-type]

        # A state saved without amplitude matching carries no scales, and none are needed: the knob is off.
        obj.scales = dict(cast('dict[str, float]', state.get('scales', {})))
        obj._global_scale = float(cast('float', state.get('global_scale', 1.0)))
        obj._global_reference = state['global_reference']  # type: ignore[assignment]
        obj._global_signature = state['global_signature']  # type: ignore[assignment]
        obj._region_index = state['region_index']  # type: ignore[assignment]
        obj._sig_mean = state.get('sig_mean')  # type: ignore[assignment]
        obj._sig_std = state.get('sig_std')  # type: ignore[assignment]
        return obj
