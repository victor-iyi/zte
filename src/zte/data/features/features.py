"""Reshape the `(n_words, n_bp_features, n_channels)` band-power tensor and rank its channel x band dimensions."""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from zte.logging_utils import get_logger

_LOG = get_logger('data.features')

type SelectionMethod = Literal['variance', 'f_score', 'mutual_info', 'rf_importance']
"""The method used to score features."""


def channel_mean_features(band_power: np.ndarray) -> np.ndarray:
    """Averages band power over channels into a compact per-(measure, band) summary.

    Args:
        band_power (np.ndarray): Array `(n_words, n_bp_features, n_channels)`.

    Returns:
        np.ndarray: NaN-aware channel means `(n_words, n_bp_features)`; omitted words are all-`NaN` by design.
    """
    import warnings

    with warnings.catch_warnings():
        # Omitted words are all-NaN slices; their NaN mean is intentional.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(band_power, axis=2).astype(np.float32)


def flatten_band_power(band_power: np.ndarray) -> np.ndarray:
    """Flattens `(n_words, n_bp_features, n_channels)` band power to 2-D.

    Args:
        band_power (np.ndarray): Array `(n_words, n_bp_features, n_channels)`.

    Returns:
        np.ndarray: `(n_words, n_bp_features * n_channels)`, C-contiguous and band-major.
    """
    n = band_power.shape[0]
    return band_power.reshape(n, -1).astype(np.float32)


def flat_feature_names(bp_feature_names: list[str], n_channels: int) -> list[str]:
    """Builds names for the flattened band-power matrix.

    Args:
        bp_feature_names (list[str]): The `(measure, band)` names, length `n_bp_features`.
        n_channels (int): Number of channels.

    Returns:
        list[str]: `n_bp_features * n_channels` names like `'TRT_t1::ch007'`, in flatten order.
    """
    return [f'{name}::ch{ch:03d}' for name in bp_feature_names for ch in range(n_channels)]


@dataclass(slots=True)
class SelectionResult:
    """Outcome of a feature-selection pass."""

    indices: np.ndarray
    """Selected column indices into the (flattened) feature matrix."""
    scores: np.ndarray
    """Importance score per input feature (one per input column, not per selected column)."""
    names: list[str] | None
    """Names of the selected features, if names were supplied."""
    method: SelectionMethod
    """The selection method used."""


class FeatureSelector:
    """Ranks and selects the most informative features for a target.

    Attributes:
        method (SelectionMethod): The scoring method.
        k (int | None): Number of features to keep (`None` keeps all, just ranks them).
        task (Literal['regression', 'classification']): Whether the target is continuous or categorical.
    """

    def __init__(
        self,
        method: SelectionMethod = 'mutual_info',
        k: int | None = 64,
        task: Literal['regression', 'classification'] = 'regression',
    ) -> None:
        """Initialises the selector.

        Args:
            method (SelectionMethod): One of `variance`, `f_score`, `mutual_info` or `rf_importance`.
            k (int | None): How many top features to keep, or `None` to rank only.
            task (Literal['regression', 'classification']): Whether the target is continuous or categorical.
        """
        self.method = method
        self.k = k
        self.task = task

    def select(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        names: list[str] | None = None,
        sample_mask: np.ndarray | None = None,
    ) -> SelectionResult:
        """Scores features and returns the top-`k` selection.

        Args:
            x (np.ndarray): Feature matrix `(n_words, n_features)` (NaNs are mean-filled before scoring).
            y (np.ndarray | None): Target `(n_words,)`; required for every method except `'variance'`.
            names (list[str] | None): Optional feature names aligned with `x`'s columns.
            sample_mask (np.ndarray | None): Optional boolean `(n_words,)` selecting valid rows (e.g. present tokens)
                so omitted words never drive the ranking.

        Returns:
            SelectionResult: The selected column indices, per-input scores, names and method.

        Raises:
            ValueError: If a supervised method is chosen without `y`.
        """
        x = np.asarray(x, dtype=np.float32)
        if sample_mask is not None:
            x = x[sample_mask]
            if y is not None:
                y = np.asarray(y)[sample_mask]

        # Column-mean fill, since the scorers reject NaN.
        col_mean = np.nan_to_num(np.nanmean(np.where(np.isnan(x), np.nan, x), axis=0))
        x = np.where(np.isnan(x), col_mean[np.newaxis, :], x)

        if self.method == 'variance':
            scores = x.var(axis=0)
        else:
            if y is None:
                raise ValueError(f'Method {self.method!r} requires a target y.')
            scores = self._supervised_scores(x, np.asarray(y))

        order = np.argsort(scores)[::-1]
        indices = order if self.k is None else order[: self.k]
        sel_names = None if names is None else [names[i] for i in indices]

        _LOG.info(
            'Selected %d/%d features via %s (top score=%.4g)',
            len(indices),
            x.shape[1],
            self.method,
            float(scores[indices[0]]) if len(indices) else float('nan'),
        )
        return SelectionResult(indices=indices, scores=scores, names=sel_names, method=self.method)

    def _supervised_scores(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes per-feature importance scores for a target."""
        if self.method == 'f_score':
            from sklearn.feature_selection import f_classif, f_regression

            func = f_regression if self.task == 'regression' else f_classif
            scores, _ = func(x, y)
            return np.nan_to_num(scores)
        if self.method == 'mutual_info':
            from sklearn.feature_selection import (
                mutual_info_classif,
                mutual_info_regression,
            )

            func = mutual_info_regression if self.task == 'regression' else mutual_info_classif
            return np.nan_to_num(func(x, y, random_state=0))

        # rf_importance
        if self.task == 'regression':
            from sklearn.ensemble import RandomForestRegressor

            model: object = RandomForestRegressor(
                n_estimators=64, max_depth=8, n_jobs=-1, random_state=0
            )
        else:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=64, max_depth=8, n_jobs=-1, random_state=0)
        model.fit(x, y)  # type: ignore[attr-defined]
        return np.asarray(model.feature_importances_)  # type: ignore[attr-defined]
