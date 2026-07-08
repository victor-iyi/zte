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
    """

    def __init__(self, mode: Normalization = 'zscore_channel', eps: float = 1e-6) -> None:
        """Initialises the normaliser.

        Args:
            mode (Normalization): One of `zscore_channel` (per-column z-score), `zscore_global` (single mean/std), `minmax` or `none`.
            eps (float): Numerical floor for divisions.

        """
        self.mode = mode
        self.eps = eps
        self._a: np.ndarray | None = None  # mean or min
        self._b: np.ndarray | None = None  # std or (max-min)

    def fit(self, x: np.ndarray) -> FeatureNormalizer:
        """Learns normalisation statistics from `x`.

        Args:
            x (np.ndarray): Training feature matrix `(n_words, n_features)`.

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
        else:  # zscore_channel (per-column)
            self._a = np.nanmean(x, axis=0)
            self._b = np.nanstd(x, axis=0)
        self._b = np.where(np.abs(np.asarray(self._b)) < self.eps, 1.0, self._b)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Applies learned statistics to `x`.

        Args:
            x (np.ndarray): Feature matrix `(n_words, n_features)`.

        Returns:
            The normalised matrix as float32 (unchanged when `mode='none'`).
        """
        if self.mode == 'none' or self._a is None:
            return x.astype(np.float32, copy=False)
        if self.mode == 'minmax':
            return ((x - self._a) / (self._b + self.eps)).astype(np.float32)  # type: ignore[operator]
        return ((x - self._a) / self._b).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Convenience for `fit` followed by `transform`.

        Args:
            x (np.ndarray): Training feature matrix `(n_words, n_features)`.

        Returns:
            The normalised matrix as float32.

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
            state (dict[str, object]): A dict previously produced by `state`.

        Returns:
            A restored `FeatureNormalizer`.

        """
        norm = cls(mode=state['mode'], eps=float(state['eps']))  # type: ignore[arg-type]
        if state['a'] is not None:
            norm._a = np.asarray(state['a'], dtype=np.float32)
        if state['b'] is not None:
            norm._b = np.asarray(state['b'], dtype=np.float32)
        return norm
