"""Signal transforms: band-pass filtering and feature normalisation.

These operate on NumPy arrays before tensors are built. Normalisers are stateful
so the statistics learned on the training split can be re-applied verbatim to
validation/test splits, preventing information leakage.
"""

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
        data: EEG array; filtering is applied over the final (time) axis.
        lowcut: Low cut-off frequency in Hz.
        highcut: High cut-off frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order.

    Returns:
        The filtered array, same shape and dtype family as ``data``.

    Raises:
        ValueError: If the requested band is not within ``(0, fs/2)``.
    """
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    if not 0 < low < high < 1:
        raise ValueError(f'Invalid band {lowcut}-{highcut} Hz for fs={fs} Hz.')
    b, a = butter(order, [low, high], btype='band')
    # filtfilt needs more samples than the padding length; guard short segments.
    if data.shape[-1] <= 3 * max(len(a), len(b)):
        return data.astype(np.float32, copy=False)
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


def normalize_raw_epoch(epoch: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel z-scores a single raw EEG epoch ``(channels, time)``.

    Per-channel standardisation absorbs impedance and gain differences across
    electrodes, as recommended for ZuCo raw EEG.

    Args:
        epoch: Array ``(channels, time)``.
        eps: Numerical floor added to the per-channel standard deviation.

    Returns:
        The standardised epoch as float32.
    """
    mean = epoch.mean(axis=-1, keepdims=True)
    std = epoch.std(axis=-1, keepdims=True)
    return ((epoch - mean) / (std + eps)).astype(np.float32)


class FeatureNormalizer:
    """Stateful normaliser for 2-D feature matrices ``(N, F)``.

    Attributes:
        mode: The normalisation scheme.
        eps: Numerical floor for divisions.
    """

    def __init__(self, mode: Normalization = 'zscore_channel', eps: float = 1e-6) -> None:
        """Initialises the normaliser.

        Args:
            mode: One of ``'zscore_channel'`` (per-column z-score),
                ``'zscore_global'`` (single mean/std), ``'minmax'`` or ``'none'``.
            eps: Numerical floor for divisions.
        """
        self.mode = mode
        self.eps = eps
        self._a: np.ndarray | None = None  # mean or min
        self._b: np.ndarray | None = None  # std or (max-min)

    def fit(self, x: np.ndarray) -> FeatureNormalizer:
        """Learns normalisation statistics from ``x``.

        Args:
            x: Training feature matrix ``(N, F)``.

        Returns:
            ``self``, for chaining.
        """
        if self.mode == 'none':
            return self
        if self.mode == 'minmax':
            self._a = np.nanmin(x, axis=0)
            self._b = np.nanmax(x, axis=0) - self._a
        elif self.mode == 'zscore_global':
            self._a = np.array(np.nanmean(x), dtype=np.float32)
            self._b = np.array(np.nanstd(x), dtype=np.float32)
        else:  # zscore_channel (per-column)
            self._a = np.nanmean(x, axis=0)
            self._b = np.nanstd(x, axis=0)
        self._b = np.where(np.abs(np.asarray(self._b)) < self.eps, 1.0, self._b)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Applies learned statistics to ``x``.

        Args:
            x: Feature matrix ``(N, F)``.

        Returns:
            The normalised matrix as float32 (unchanged when ``mode='none'``).
        """
        if self.mode == 'none' or self._a is None:
            return x.astype(np.float32, copy=False)
        if self.mode == 'minmax':
            return ((x - self._a) / (self._b + self.eps)).astype(np.float32)
        return ((x - self._a) / self._b).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Convenience for :meth:`fit` followed by :meth:`transform`.

        Args:
            x: Training feature matrix ``(N, F)``.

        Returns:
            The normalised matrix.
        """
        return self.fit(x).transform(x)

    @property
    def state(self) -> dict[str, object]:
        """Returns a serialisable dict of fitted statistics for checkpointing."""
        return {
            'mode': self.mode,
            'eps': self.eps,
            'a': None if self._a is None else np.asarray(self._a).tolist(),
            'b': None if self._b is None else np.asarray(self._b).tolist(),
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> FeatureNormalizer:
        """Rebuilds a normaliser from :attr:`state`.

        Args:
            state: A dict previously produced by :attr:`state`.

        Returns:
            A restored :class:`FeatureNormalizer`.
        """
        norm = cls(mode=state['mode'], eps=float(state['eps']))  # type: ignore[arg-type]
        if state['a'] is not None:
            norm._a = np.asarray(state['a'], dtype=np.float32)
        if state['b'] is not None:
            norm._b = np.asarray(state['b'], dtype=np.float32)
        return norm
