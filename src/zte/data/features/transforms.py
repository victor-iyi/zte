"""Signal transforms: band-pass filtering and feature normalisation.

These operate on NumPy arrays before tensors are built. Normalisers are stateful so the statistics learned on the training
split can be re-applied verbatim to validation/test splits, preventing information leakage.
"""

# pylint: disable=import-outside-toplevel,protected-access
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from zte.config import Normalization
from zte.data.schema import BAND_RANGES_HZ, BANDS, SAMPLING_RATE_HZ, Band
from zte.logging_utils import get_logger

_LOG = get_logger('data.transforms')


def phase_scramble(x: np.ndarray, *, axis: int = -1, seed: int = 0) -> np.ndarray:
    """Phase-randomised surrogate: destroys temporal/phase structure, preserves the power spectrum.

    FFTs each channel along `axis`, replaces the Fourier phases with i.i.d. uniform angles (keeping DC
    and the Nyquist bin real so the inverse is real-valued), and inverts. The result has the *same
    per-channel power spectrum* as the input but no meaningful temporal or cross-channel structure -- the
    honest "spectrum-matched but meaningless" null input. Embedding phase-scrambled EEG through the
    *trained* encoder shows whether the encoder invents structure from destroyed signal: a genuine
    temporal encoder should collapse toward noise on it, while a purely band-power feature is essentially
    unchanged (so this control is informative for raw frontends, and a no-op-by-construction for band power).

    Args:
        x (np.ndarray): Real signal, e.g. raw EEG `(..., n_channels, n_times)`; scrambled along `axis`.
        axis (int): Time axis to scramble.
        seed (int): RNG seed.

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
        The filtered array, same shape and dtype family as `data`.

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
    """Per-channel z-scores raw EEG epochs `(..., n_channels, time_steps)`.

    Per-channel standardisation absorbs impedance and gain differences across electrodes, as recommended for ZuCo raw EEG.
    Statistics are taken over the time axis only, so leading dimensions (a batch of epochs) broadcast.

    Args:
        epoch (np.ndarray): Array `(..., n_channels, time_steps)`.
        eps (float): Numerical floor added to the per-channel standard deviation.

    Returns:
        The standardised epochs as float32.

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

    Recomputes, straight from the raw signal, the classical feature the band-power representation carries:
    a real FFT per epoch with power integrated over each ZuCo band (:data:`~zte.data.schema.BAND_RANGES_HZ`).

    This is the honest *classical-feature control* for a raw frontend. The alternative -- flattening the
    window to `n_channels * time_steps` -- is not a band-power baseline at all (it is the time-domain signal),
    and at 105 x 350 = 36,750 dims no probe can consume it: ridge regression forms a `d x d` Gram matrix
    (~10.8 GB here) and standardisation copies the whole matrix to float64. Band power keeps the same
    information a band-power pipeline would extract, in ~840 dims.

    Bands too low to resolve in a short window (no FFT bin falls inside the range) yield a zero column
    rather than a `NaN` from an empty mean -- with `raw_window=32` at 500 Hz the resolution is ~15.6 Hz, so
    every band below beta is unresolvable and honestly reports zero power.

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
        cols = [
            power[..., m].mean(axis=-1) if m.any() else np.zeros(power.shape[:-1], dtype=np.float32)
            for m in masks
        ]
        out[start : start + len(block)] = np.stack(cols, axis=-1).reshape(len(block), -1)
    return out


def sanitize_raw_windows(raw: np.ndarray, eps: float = 1e-6, chunk: int = 4096) -> np.ndarray:
    """Makes raw EEG windows model-safe **in place**: NaN/inf -> 0, then per-channel z-score per epoch.

    ZuCo's `rawEEG` carries `NaN` for rejected samples/channels and is stored in unscaled microvolts.
    Unlike band power -- which is imputed and `FeatureNormalizer`-scaled -- raw signals reach the frontend exactly as parsed,
    so a single `NaN` propagates through the convolution and makes the whole contrastive batch (and any exported embedding) `NaN`.
    This applies the `normalize_raw_epoch` treatment plus sanitisation, and is the single choke point every raw consumer relies on.

    Idempotent: re-applying to already-standardised windows is a no-op up to `eps`. Operates in place and
    in chunks so multi-GB arrays never allocate a full-size temporary.

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
        # nan_to_num must stay INSIDE the loop: it builds full-size boolean masks internally, so calling
        # it on the whole array costs several GB of temporaries on a multi-GB bundle.
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        mean = block.mean(axis=-1, keepdims=True)
        std = block.std(axis=-1, keepdims=True)
        block -= mean
        block /= std + eps
    return x


class FeatureNormalizer:
    """Stateful normaliser for 2-D feature matrices `(n_words, n_features)`.

    Attributes:
        mode: The normalisation scheme.
        eps: Numerical floor for divisions.

    Note:
        Mode `zscore_subject` fits and stores a **per-subject** per-column mean/std keyed by
        subject code, and applies each row using its own subject's statistics. This removes the
        constant per-subject offset that otherwise makes subject identity the cheapest thing for
        the encoder to latch onto. `fit`/`transform` take an optional `subjects` array (one label
        per row); at transform time any row whose subject was unseen at fit (e.g. a default
        inference subject) falls back to a **global pooled** mean/std computed at fit time.
    """

    # Above this feature width, per-subject covariance whitening (O(d^3)) is skipped and
    # `riemannian` degrades to `zscore_subject` with a warning.
    _RIEMANN_MAX_DIM = 2048

    def __init__(
        self, mode: Normalization = 'zscore_channel', eps: float = 1e-6, shrinkage: float = 0.1
    ) -> None:
        """Initialises the normaliser.

        Args:
            mode (Normalization): One of `zscore_channel` (per-column z-score), `zscore_global` (single mean/std),
                `zscore_subject` (per-subject per-column z-score), `riemannian` (per-subject covariance whitening
                -- recentres each subject's feature covariance to a shared reference, the mechanistic attack on the
                forward-model fingerprint), `minmax` or `none`.
            eps (float): Numerical floor for divisions.
            shrinkage (float): Ledoit-Wolf-style shrinkage toward a scaled identity for the `riemannian` covariance,
                keeping it well-conditioned and invertible.
        """
        self.mode = mode
        self.eps = eps
        self.shrinkage = shrinkage
        self._a: np.ndarray | None = None  # mean or min (global stats)
        self._b: np.ndarray | None = None  # std or (max-min) (global stats)
        # Per-subject stats for mode 'zscore_subject': {subject_code: (mean, std)}.
        self._subject_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Per-subject (mean, W = Sigma^-1/2, W_inv = Sigma^1/2) for mode 'riemannian';
        # `_global_map` is the fallback for subjects unseen at fit time.
        self._subject_maps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._global_map: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def fit(self, x: np.ndarray, subjects: np.ndarray | None = None) -> FeatureNormalizer:
        """Learns normalisation statistics from `x`.

        Args:
            x (np.ndarray): Training feature matrix `(n_words, n_features)`.
            subjects (np.ndarray | None): Per-row subject labels `(n_words,)`, required for
                `mode='zscore_subject'` (ignored by the other modes).

        Returns:
            `self`, for chaining.

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
        """Returns `(Sigma^-1/2, Sigma^1/2)` for centred rows `xc`, with shrinkage.

        Shrinkage toward a scaled identity keeps the covariance invertible even when a subject has
        few words relative to the feature dimension; the matrix square roots come from a symmetric
        eigendecomposition.
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
                'riemannian normalise skipped: %d features exceed the %d cap; '
                'falling back to zscore_subject.',
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

    def _riemann_apply(
        self, x: np.ndarray, subjects: np.ndarray | None, inverse: bool
    ) -> np.ndarray:
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

    def _subject_ab(
        self, n_rows: int, subjects: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
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
            The normalised matrix as float32 (unchanged when `mode='none'`).
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
        """Reconstructs the pre-normalisation matrix from normalised features.

        This is the exact inverse of `transform` and is used to re-derive the raw feature
        matrix when re-fitting statistics on a train-only subset (see `ZuCoDataset.refit_normalizer`).

        Args:
            x (np.ndarray): A previously normalised matrix `(n_words, n_features)`.
            subjects (np.ndarray | None): Per-row subject labels for `mode='zscore_subject'`.

        Returns:
            The de-normalised matrix as float32 (unchanged when `mode='none'`).
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
            The normalised matrix as float32.

        """
        return self.fit(x, subjects=subjects).transform(x, subjects=subjects)

    def calibrate_subject(self, baseline_x: np.ndarray, subject_code: str) -> FeatureNormalizer:
        """Registers per-subject statistics for a *new* subject from an unlabelled baseline.

        This is the zero-shot new-brain path: a short recording of the new person reading
        anything yields their own normalisation (per-subject mean/std, or the Riemannian
        covariance-whitening map), so their words are placed on the shared scale without any
        labels or retraining. Subsequent `transform(..., subjects=[subject_code, ...])` uses
        the calibrated statistics instead of the population fallback. A no-op for modes that
        carry no per-subject state.

        Args:
            baseline_x (np.ndarray): The new subject's baseline feature matrix `(n, n_features)`.
            subject_code (str): The code to register the statistics under.

        Returns:
            `self`, for chaining.
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

        For `mode='zscore_subject'` the dict additionally carries `subject_stats`, a
        `{subject_code: {'a': [...], 'b': [...]}}` mapping of per-subject mean/std; `a`/`b` hold
        the global pooled fallback.
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
            A restored `FeatureNormalizer`.

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
