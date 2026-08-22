"""Signal transforms: band-pass filtering and feature normalisation.

These operate on NumPy arrays before tensors are built. Normalisers are stateful so the statistics learned on the
training split can be re-applied verbatim to validation/test splits, preventing information leakage.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.signal import butter, filtfilt

from zte.config import Normalization
from zte.data.schema import BAND_RANGES_HZ, BANDS, SAMPLING_RATE_HZ, Band
from zte.logging_utils import get_logger

_LOG = get_logger('data.transforms')


def phase_scramble(x: np.ndarray, *, axis: int = -1, seed: int = 0) -> np.ndarray:
    """Phase-randomised surrogate: destroys temporal/phase structure, preserves the power spectrum.

    Args:
        x (np.ndarray): Real signal, e.g. raw EEG `(..., n_channels, n_times)`; scrambled along `axis`.
        axis (int): Time axis to scramble.
        seed (int): Random number generator seed.

    Returns:
        np.ndarray: Phase-scrambled real signal, same shape as `x` (float32).
    """
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(np.asarray(x, dtype=np.float64), axis=axis)
    phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=spec.shape))
    # Keep DC (and the Nyquist bin for an even-length signal) real so irfft is exactly real.
    sl0 = [slice(None)] * spec.ndim
    sl0[axis] = slice(0, 1)
    phases[tuple(sl0)] = 1.0
    if x.shape[axis] % 2 == 0:
        sln = [slice(None)] * spec.ndim
        sln[axis] = slice(-1, None)
        phases[tuple(sln)] = 1.0
    scrambled = np.fft.irfft(np.abs(spec) * phases, n=x.shape[axis], axis=axis)
    return scrambled.astype(np.float32)


def bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 50.0,
    fs: float = SAMPLING_RATE_HZ,
    order: int = 5,
) -> np.ndarray:
    """Applies a zero-phase Butterworth band-pass filter along the last axis.

    Args:
        data (np.ndarray): EEG array; filtering is applied over the final (time) axis.
        lowcut (float): Low cut-off frequency in Hz.
        highcut (float): High cut-off frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int): Filter order.

    Returns:
        np.ndarray: The filtered array, same shape as `data`.

    Raises:
        ValueError: If the requested band is not within `(0, fs/2)`.
    """
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    if not 0 < low < high < 1:
        raise ValueError(f'Invalid band {lowcut}-{highcut} Hz for fs={fs} Hz.')
    b, a = butter(order, [low, high], btype='band')  # type: ignore[assignment]
    # filtfilt needs more samples than the padding length; guard short segments.
    if data.shape[-1] <= 3 * max(len(a), len(b)):
        return data.astype(np.float32, copy=False)
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


def normalize_raw_epoch(epoch: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel z-scores raw EEG epochs, absorbing impedance and gain differences across electrodes.

    Statistics come from the time axis only, so leading dimensions broadcast.

    Args:
        epoch (np.ndarray): Array `(..., n_channels, time_steps)`.
        eps (float): Numerical floor added to the per-channel standard deviation.

    Returns:
        np.ndarray: The standardised epochs as float32.
    """
    mean = epoch.mean(axis=-1, keepdims=True)
    std = epoch.std(axis=-1, keepdims=True)
    return ((epoch - mean) / (std + eps)).astype(np.float32)


def band_power_from_raw(
    raw: np.ndarray,
    bands: tuple[Band, ...] = BANDS,
    fs: float = SAMPLING_RATE_HZ,
    chunk: int = 1024,
) -> np.ndarray:
    """Per-channel band power for each raw EEG window -> `(n_epochs, n_channels * n_bands)`.

    A real FFT per epoch with power integrated over each `BAND_RANGES_HZ` band, giving the classical-feature control
    for a raw frontend in ~840 dims; flattening the window instead yields the time-domain signal at ~36,750 dims, which
    no probe can consume. Bands with no FFT bin inside their range report zero power rather than a NaN empty mean.

    Args:
        raw (np.ndarray): Raw windows `(n_epochs, n_channels, time_steps)`; may be a read-only memmap.
        bands (tuple[Band, ...]): Bands to integrate, in output order.
        fs (float): Sampling rate in Hz.
        chunk (int): Epochs per block (bounds peak temporary memory).

    Returns:
        np.ndarray: `(n_epochs, n_channels * n_bands)` float32 band power, `(channel, band)`-major.
    """
    x = np.asarray(raw)
    n, n_ch, n_t = x.shape
    freqs = np.fft.rfftfreq(n_t, d=1.0 / fs)
    masks = [(freqs >= BAND_RANGES_HZ[b][0]) & (freqs <= BAND_RANGES_HZ[b][1]) for b in bands]
    out = np.empty((n, n_ch * len(bands)), dtype=np.float32)
    for start in range(0, n, chunk):
        block = np.asarray(x[start : start + chunk], dtype=np.float32)
        spec = np.fft.rfft(block, axis=-1)  # complex64 for float32 input
        power = np.square(spec.real) + np.square(spec.imag)  # (b, n_ch, n_freq)
        cols = [power[..., m].mean(axis=-1) if m.any() else np.zeros(power.shape[:-1], dtype=np.float32) for m in masks]
        out[start : start + len(block)] = np.stack(cols, axis=-1).reshape(len(block), -1)
    return out


def sanitize_raw_windows(raw: np.ndarray, eps: float = 1e-6, chunk: int = 4096) -> np.ndarray:
    """Makes raw EEG windows model-safe in place: NaN/inf -> 0, then per-channel z-score per epoch.

    Raw signals reach the frontend exactly as parsed, unlike imputed and normalised band power, so a single `NaN` from
    ZuCo's rejected samples propagates through the convolution and NaNs the whole batch. This is the one choke point
    every raw consumer relies on, and it is idempotent up to `eps`.

    Args:
        raw (np.ndarray): Float array `(n_epochs, n_channels, time_steps)`, modified in place when float32.
        eps (float): Numerical floor added to the per-channel standard deviation.
        chunk (int): Rows standardised per pass (bounds peak temporary memory).

    Returns:
        np.ndarray: The sanitised, per-channel z-scored windows (float32).
    """
    x = np.asarray(raw, dtype=np.float32)
    if not x.flags.writeable:  # a read-only memmap must be sanitised at write time, not here
        raise ValueError(
            'sanitize_raw_windows writes in place but got a read-only array. A bundle whose raw windows '
            'are already sanitised must skip this call rather than re-run it on a memory-mapped array.'
        )
    for start in range(0, len(x), chunk):
        block = x[start : start + chunk]
        # nan_to_num builds full-size boolean masks internally, so it must stay inside the chunk loop.
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        mean = block.mean(axis=-1, keepdims=True)
        std = block.std(axis=-1, keepdims=True)
        block -= mean
        block /= std + eps
    return x


class FeatureNormalizer:
    """Stateful normaliser for 2-D feature matrices `(n_words, n_features)`.

    The subject-keyed modes take an optional `subjects` array in `fit`/`transform` and apply each row its own subject's
    statistics, removing the constant offset that makes subject identity the cheapest thing to encode. A row whose
    subject was unseen at fit time falls back to the global pooled statistics.

    Attributes:
        mode (Normalization): The normalisation scheme.
        eps (float): Numerical floor for divisions.
    """

    # Above this width the O(d^3) per-subject covariance whitening is skipped and `riemannian` degrades to a z-score.
    _RIEMANN_MAX_DIM: ClassVar[int] = 2048

    def __init__(self, mode: Normalization = 'zscore_channel', eps: float = 1e-6, shrinkage: float = 0.1) -> None:
        """Initialises the normaliser.

        Args:
            mode (Normalization): `zscore_channel` (per-column), `zscore_global` (single mean/std), `zscore_subject`
                (per-subject per-column), `riemannian` (per-subject covariance whitening, which recentres each
                subject's feature covariance to a shared reference and attacks the forward-model fingerprint),
                `minmax` or `none`.
            eps (float): Numerical floor for divisions.
            shrinkage (float): Ledoit-Wolf-style shrinkage toward a scaled identity, keeping the `riemannian`
                covariance well-conditioned and invertible.
        """
        self.mode = mode
        self.eps = eps
        self.shrinkage = shrinkage
        self._a: np.ndarray | None = None  # mean or min (global stats)
        self._b: np.ndarray | None = None  # std or (max-min) (global stats)
        self._subject_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Per-subject `(mean, Sigma^-1/2, Sigma^1/2)`; `_global_map` covers subjects unseen at fit time.
        self._subject_maps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._global_map: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def fit(self, x: np.ndarray, subjects: np.ndarray | None = None) -> FeatureNormalizer:
        """Learns normalisation statistics from `x`.

        Args:
            x (np.ndarray): Training feature matrix `(n_words, n_features)`.
            subjects (np.ndarray | None): Per-row subject labels `(n_words,)`, required for
                `mode='zscore_subject'` (ignored by the other modes).

        Returns:
            FeatureNormalizer: `self`, for chaining.
        """
        if self.mode == 'none':
            return self
        if self.mode == 'minmax':
            self._a = np.nanmin(x, axis=0)
            self._b = np.nanmax(x, axis=0) - self._a
        elif self.mode == 'zscore_global':
            self._a = np.array(np.nanmean(x), dtype=np.float32)
            self._b = np.array(np.nanstd(x), dtype=np.float32)
        elif self.mode == 'zscore_subject':
            return self._fit_subject(x, subjects)
        elif self.mode == 'riemannian':
            return self._fit_riemannian(x, subjects)
        else:  # zscore_channel (per-column)
            self._a = np.nanmean(x, axis=0)
            self._b = np.nanstd(x, axis=0)
        self._b = np.where(np.abs(np.asarray(self._b)) < self.eps, 1.0, self._b)
        return self

    def _fit_subject(self, x: np.ndarray, subjects: np.ndarray | None) -> FeatureNormalizer:
        """Fits per-subject per-column mean/std plus a global pooled fallback."""
        # Global pooled stats double as the fallback for unseen subjects at transform time.
        self._a = np.nanmean(x, axis=0)
        self._b = self._clamp(np.nanstd(x, axis=0))
        self._subject_stats = {}
        if subjects is not None:
            subjects = np.asarray(subjects)
            for code in np.unique(subjects):
                rows = subjects == code
                mean = np.nanmean(x[rows], axis=0)
                std = self._clamp(np.nanstd(x[rows], axis=0))
                # NaN can appear if a subject never observes a column; fall back to global.
                mean = np.where(np.isnan(mean), self._a, mean)
                self._subject_stats[str(code)] = (mean, std)
        return self

    def _clamp(self, b: np.ndarray) -> np.ndarray:
        """Replaces near-zero scales with 1 so constant columns pass through unchanged."""
        b = np.asarray(b, dtype=np.float32)
        return np.where(np.abs(b) < self.eps, 1.0, b)

    # -- Riemannian (per-subject covariance whitening) ---------------------- #

    def _cov_maps(self, xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns `(Sigma^-1/2, Sigma^1/2)` for centred rows `xc`, via a symmetric eigendecomposition.

        The shrinkage toward a scaled identity keeps the covariance invertible when a subject has few words relative to
        the feature dimension.
        """
        d = xc.shape[1]
        cov = (xc.T @ xc) / max(len(xc) - 1, 1)
        mu = float(np.trace(cov) / d)
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * mu * np.eye(d, dtype=np.float64)
        w, v = np.linalg.eigh(cov)
        w = np.clip(w, self.eps, None)
        inv_sqrt = (v * (w**-0.5)) @ v.T
        sqrt = (v * (w**0.5)) @ v.T
        return inv_sqrt.astype(np.float32), sqrt.astype(np.float32)

    def _fit_riemannian(self, x: np.ndarray, subjects: np.ndarray | None) -> FeatureNormalizer:
        """Fits per-subject whitening maps plus a global fallback map."""
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
        if x.shape[1] > self._RIEMANN_MAX_DIM:
            _LOG.warning(
                'riemannian normalise skipped: %d features exceed the %d cap; falling back to zscore_subject.',
                x.shape[1],
                self._RIEMANN_MAX_DIM,
            )
            self.mode = 'zscore_subject'
            return self._fit_subject(x, subjects)
        gmean = x.mean(axis=0)
        g_inv, g_sqrt = self._cov_maps(x - gmean)
        self._global_map = (gmean.astype(np.float32), g_inv, g_sqrt)
        self._a = gmean.astype(np.float32)  # keeps `state`/`transform` guards happy
        self._b = np.ones(x.shape[1], dtype=np.float32)
        self._subject_maps = {}
        if subjects is not None:
            subjects = np.asarray(subjects)
            for code in np.unique(subjects):
                rows = subjects == code
                if int(rows.sum()) < 2:
                    continue
                mean = x[rows].mean(axis=0)
                inv_sqrt, sqrt = self._cov_maps(x[rows] - mean)
                self._subject_maps[str(code)] = (mean.astype(np.float32), inv_sqrt, sqrt)
        return self

    def _riemann_apply(self, x: np.ndarray, subjects: np.ndarray | None, inverse: bool) -> np.ndarray:
        """Applies (or inverts) per-subject whitening row-wise, global map for unknown subjects."""
        assert self._global_map is not None
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
        out = np.empty_like(x, dtype=np.float32)
        subj_arr = None if subjects is None else np.asarray(subjects)
        for i in range(len(x)):
            code = None if subj_arr is None else str(subj_arr[i])
            mean, inv_sqrt, sqrt = self._subject_maps.get(code, self._global_map)  # type: ignore[arg-type]
            if inverse:
                out[i] = (x[i] @ sqrt) + mean
            else:
                out[i] = (x[i] - mean) @ inv_sqrt
        return out

    def _subject_ab(self, n_rows: int, subjects: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Broadcasts per-subject (mean, std) to `(n_rows, n_features)`, global for unknowns."""
        a = np.tile(np.asarray(self._a, dtype=np.float32), (n_rows, 1))
        b = np.tile(np.asarray(self._b, dtype=np.float32), (n_rows, 1))
        if subjects is not None and self._subject_stats:
            subjects = np.asarray(subjects)
            for code, (mean, std) in self._subject_stats.items():
                rows = subjects == code
                if rows.any():
                    a[rows] = mean
                    b[rows] = std
        return a, b

    def transform(self, x: np.ndarray, subjects: np.ndarray | None = None) -> np.ndarray:
        """Applies learned statistics to `x`.

        Args:
            x (np.ndarray): Feature matrix `(n_words, n_features)`.
            subjects (np.ndarray | None): Per-row subject labels `(n_words,)` for
                `mode='zscore_subject'`; unknown/`None` subjects use the global pooled fallback.

        Returns:
            np.ndarray: The normalised matrix as float32, unchanged when `mode='none'`.
        """
        if self.mode == 'none' or self._a is None:
            return x.astype(np.float32, copy=False)
        if self.mode == 'riemannian':
            return self._riemann_apply(x, subjects, inverse=False)
        if self.mode == 'zscore_subject':
            a, b = self._subject_ab(len(x), subjects)
            return ((x - a) / b).astype(np.float32)
        if self.mode == 'minmax':
            return ((x - self._a) / (self._b + self.eps)).astype(np.float32)  # type: ignore[operator]
        return ((x - self._a) / self._b).astype(np.float32)

    def inverse_transform(self, x: np.ndarray, subjects: np.ndarray | None = None) -> np.ndarray:
        """Exact inverse of `transform`, re-deriving raw features before re-fitting on a train-only subset.

        Args:
            x (np.ndarray): A previously normalised matrix `(n_words, n_features)`.
            subjects (np.ndarray | None): Per-row subject labels for `mode='zscore_subject'`.

        Returns:
            np.ndarray: The de-normalised matrix as float32, unchanged when `mode='none'`.
        """
        if self.mode == 'none' or self._a is None:
            return np.asarray(x, dtype=np.float32)
        if self.mode == 'riemannian':
            return self._riemann_apply(x, subjects, inverse=True)
        if self.mode == 'zscore_subject':
            a, b = self._subject_ab(len(x), subjects)
            return (x * b + a).astype(np.float32)
        if self.mode == 'minmax':
            return (x * (self._b + self.eps) + self._a).astype(np.float32)  # type: ignore[operator]
        return (x * self._b + self._a).astype(np.float32)

    def fit_transform(self, x: np.ndarray, subjects: np.ndarray | None = None) -> np.ndarray:
        """Convenience for `fit` followed by `transform`.

        Args:
            x (np.ndarray): Training feature matrix `(n_words, n_features)`.
            subjects (np.ndarray | None): Per-row subject labels for `mode='zscore_subject'`.

        Returns:
            np.ndarray: The normalised matrix as float32.
        """
        return self.fit(x, subjects=subjects).transform(x, subjects=subjects)

    def calibrate_subject(self, baseline_x: np.ndarray, subject_code: str) -> FeatureNormalizer:
        """Registers per-subject statistics for a new subject from an unlabelled baseline.

        The zero-shot new-brain path: a short recording of the person reading anything yields their own statistics, so
        their words land on the shared scale without labels or retraining, and later `transform` calls naming that
        subject code use them instead of the population fallback. A no-op for modes carrying no per-subject state.

        Args:
            baseline_x (np.ndarray): The new subject's baseline feature matrix `(n, n_features)`.
            subject_code (str): The code to register the statistics under.

        Returns:
            FeatureNormalizer: `self`, for chaining.
        """
        code = str(subject_code)
        if self.mode == 'zscore_subject':
            mean = np.nanmean(baseline_x, axis=0)
            std = self._clamp(np.nanstd(baseline_x, axis=0))
            mean = np.where(np.isnan(mean), self._a, mean)
            self._subject_stats[code] = (mean.astype(np.float32), std)
        elif self.mode == 'riemannian':
            x = np.nan_to_num(np.asarray(baseline_x, dtype=np.float64), nan=0.0)
            mean = x.mean(axis=0)
            inv_sqrt, sqrt = self._cov_maps(x - mean)
            self._subject_maps[code] = (mean.astype(np.float32), inv_sqrt, sqrt)
        return self

    @property
    def state(self) -> dict[str, object]:
        """Returns a serialisable dict of fitted statistics for checkpointing.

        `a`/`b` hold the global pooled fallback; the subject-keyed modes add their per-subject statistics alongside.
        """
        out: dict[str, object] = {
            'mode': self.mode,
            'eps': self.eps,
            'a': None if self._a is None else np.asarray(self._a).tolist(),
            'b': None if self._b is None else np.asarray(self._b).tolist(),
        }
        if self.mode == 'zscore_subject':
            out['subject_stats'] = {
                code: {'a': np.asarray(mean).tolist(), 'b': np.asarray(std).tolist()}
                for code, (mean, std) in self._subject_stats.items()
            }
        if self.mode == 'riemannian':
            out['shrinkage'] = self.shrinkage
            if self._global_map is not None:
                gm, gi, gs = self._global_map
                out['global_map'] = {'mean': gm.tolist(), 'inv': gi.tolist(), 'sqrt': gs.tolist()}
            out['subject_maps'] = {
                code: {'mean': m.tolist(), 'inv': i.tolist(), 'sqrt': s.tolist()}
                for code, (m, i, s) in self._subject_maps.items()
            }
        return out

    @classmethod
    def from_state(cls, state: dict[str, object]) -> FeatureNormalizer:
        """Rebuilds a normaliser from `state`.

        Args:
            state (dict[str, object]): A dict previously produced by `state`.

        Returns:
            FeatureNormalizer: The restored normaliser.
        """
        norm = cls(mode=state['mode'], eps=float(state['eps']))  # type: ignore[arg-type]
        if state.get('shrinkage') is not None:
            norm.shrinkage = float(state['shrinkage'])  # type: ignore[arg-type]
        if state['a'] is not None:
            norm._a = np.asarray(state['a'], dtype=np.float32)
        if state['b'] is not None:
            norm._b = np.asarray(state['b'], dtype=np.float32)
        gm = state.get('global_map')
        if gm is not None:
            norm._global_map = (
                np.asarray(gm['mean'], dtype=np.float32),  # type: ignore[index]
                np.asarray(gm['inv'], dtype=np.float32),  # type: ignore[index]
                np.asarray(gm['sqrt'], dtype=np.float32),  # type: ignore[index]
            )
        norm._subject_maps = {
            str(code): (
                np.asarray(m['mean'], dtype=np.float32),
                np.asarray(m['inv'], dtype=np.float32),
                np.asarray(m['sqrt'], dtype=np.float32),
            )
            for code, m in (state.get('subject_maps') or {}).items()  # type: ignore[union-attr]
        }
        subject_stats = state.get('subject_stats') or {}
        norm._subject_stats = {
            str(code): (
                np.asarray(stats['a'], dtype=np.float32),
                np.asarray(stats['b'], dtype=np.float32),
            )
            for code, stats in subject_stats.items()  # type: ignore[union-attr]
        }
        return norm
