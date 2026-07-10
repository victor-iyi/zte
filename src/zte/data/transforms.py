"""Signal transforms: band-pass filtering and feature normalisation.

These operate on NumPy arrays before tensors are built. Normalisers are stateful so the statistics learned on the training
split can be re-applied verbatim to validation/test splits, preventing information leakage.
"""

# pylint: disable=import-outside-toplevel,protected-access
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from zte.config import Normalization
from zte.data.schema import SAMPLING_RATE_HZ


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
    """Per-channel z-scores a single raw EEG epoch `(n_channels, time_steps)`.

    Per-channel standardisation absorbs impedance and gain differences across electrodes, as recommended for ZuCo raw EEG.

    Args:
        epoch (np.ndarray): Array `(n_channels, time_steps)`.
        eps (float): Numerical floor added to the per-channel standard deviation.

    Returns:
        The standardised epoch as float32.

    """
    mean = epoch.mean(axis=-1, keepdims=True)
    std = epoch.std(axis=-1, keepdims=True)
    return ((epoch - mean) / (std + eps)).astype(np.float32)


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

    def __init__(self, mode: Normalization = 'zscore_channel', eps: float = 1e-6) -> None:
        """Initialises the normaliser.

        Args:
            mode (Normalization): One of `zscore_channel` (per-column z-score), `zscore_global`
                (single mean/std), `zscore_subject` (per-subject per-column z-score), `minmax` or `none`.
            eps (float): Numerical floor for divisions.

        """
        self.mode = mode
        self.eps = eps
        self._a: np.ndarray | None = None  # mean or min (global stats)
        self._b: np.ndarray | None = None  # std or (max-min) (global stats)
        # Per-subject stats for mode 'zscore_subject': {subject_code: (mean, std)}.
        self._subject_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

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
        if self.mode == 'zscore_subject':
            a, b = self._subject_ab(len(x), subjects)
            return ((x - a) / b).astype(np.float32)
        if self.mode == 'minmax':
            return ((x - self._a) / (self._b + self.eps)).astype(np.float32)  # type: ignore[operator]
        return ((x - self._a) / self._b).astype(np.float32)

    def inverse_transform(self, x: np.ndarray, subjects: np.ndarray | None = None) -> np.ndarray:
        """Reconstructs the pre-normalisation matrix from normalised features.

        This is the exact inverse of :meth:`transform` and is used to re-derive the raw feature
        matrix when re-fitting statistics on a train-only subset (see
        :meth:`~zte.data.dataset.ZuCoDataset.refit_normalizer`).

        Args:
            x (np.ndarray): A previously normalised matrix `(n_words, n_features)`.
            subjects (np.ndarray | None): Per-row subject labels for `mode='zscore_subject'`.

        Returns:
            The de-normalised matrix as float32 (unchanged when `mode='none'`).
        """
        if self.mode == 'none' or self._a is None:
            return np.asarray(x, dtype=np.float32)
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
        return out

    @classmethod
    def from_state(cls, state: dict[str, object]) -> FeatureNormalizer:
        """Rebuilds a normaliser from :attr:`state`.

        Args:
            state (dict[str, object]): A dict previously produced by `state`.

        Returns:
            A restored `FeatureNormalizer`.

        """
        norm = cls(mode=state['mode'], eps=float(state['eps']))  # type: ignore[arg-type]
        if state['a'] is not None:
            norm._a = np.asarray(state['a'], dtype=np.float32)
        if state['b'] is not None:
            norm._b = np.asarray(state['b'], dtype=np.float32)
        subject_stats = state.get('subject_stats') or {}
        norm._subject_stats = {
            str(code): (
                np.asarray(stats['a'], dtype=np.float32),
                np.asarray(stats['b'], dtype=np.float32),
            )
            for code, stats in subject_stats.items()  # type: ignore[union-attr]
        }
        return norm
