"""Missing-value imputation for word-level EEG features, per `MissingConfig.method`.

Readers skip words, and those words carry no EEG. Treating the resulting NaN rows as real signal is the biggest source
of leakage in word-level ZuCo modelling, so every strategy also returns a presence mask for objectives to gate on.
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import numpy as np

from zte.config import MissingConfig
from zte.logging_utils import get_logger

_LOG = get_logger('data.missing')


class MissingValueImputer:
    """Applies a configured missing-value strategy to a 2-D feature matrix.

    Stateful for the column/global statistics methods, so the fill learned on the training split can be re-applied to
    validation/test via `transform` without leaking.

    Attributes:
        config (MissingConfig): The configuration driving behaviour.
    """

    def __init__(self, config: MissingConfig) -> None:
        """Initialises the imputer.

        Args:
            config (MissingConfig): Missing-value configuration.
        """
        self.config = config
        self._stats: np.ndarray | None = None
        self._sklearn_imputer: object | None = None

    @staticmethod
    def presence_mask(x: np.ndarray) -> np.ndarray:
        """Computes a per-row presence mask (`True` = the token has real data).

        Args:
            x (np.ndarray): Feature matrix `(n_words, n_features)` possibly containing `NaN`.

        Returns:
            np.ndarray: A boolean array `(n_words,)` that is `False` for all-`NaN` rows.
        """
        return ~np.all(np.isnan(x), axis=1)

    def fit_transform(
        self, x: np.ndarray, group_ids: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fits any needed statistics and returns `(imputed, presence_mask)`.

        Args:
            x (np.ndarray): Feature matrix `(n_words, n_features)` with `NaN` for missing entries.
            group_ids (np.ndarray | None): Optional integer group id per row (e.g. a sentence id) used
                by sequence-aware methods (`ffill`/`interpolate`) so fills never cross sentence boundaries.

        Returns:
            tuple[np.ndarray, np.ndarray]: `(imputed (n_words, n_features) float32, presence_mask (n_words,) bool)`.
        """
        x = np.asarray(x, dtype=np.float32)
        mask = self.presence_mask(x)
        method = self.config.method

        if method in {'zero', 'mask_only'}:
            out = np.nan_to_num(x, nan=0.0)
        elif method == 'row_mean':
            out = self._fill_row_mean(x)
        elif method in {'col_mean', 'global_mean', 'median'}:
            out = self._fill_column_stat(x, method)
        elif method == 'knn':
            out = self._fill_sklearn(x, 'knn')
        elif method == 'iterative':
            out = self._fill_sklearn(x, 'iterative')
        elif method in {'ffill', 'interpolate'}:
            out = self._fill_sequence(x, group_ids, method)
        elif method == 'drop':
            out = np.nan_to_num(x, nan=0.0)  # mask drives row removal in the dataset
        else:  # pragma: no cover - guarded by the Literal type
            raise ValueError(f'Unknown missing-value method: {method!r}')

        return out.astype(np.float32), mask

    def transform(
        self, x: np.ndarray, group_ids: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-applies a previously fitted fill to new data where possible.

        Column/global/median and sklearn methods reuse their fitted statistics; sequence and row methods recompute.

        Args:
            x (np.ndarray): Feature matrix `(n_words, n_features)` with `NaN` for missing entries.
            group_ids (np.ndarray | None): Optional group ids for sequence-aware methods.

        Returns:
            tuple[np.ndarray, np.ndarray]: `(imputed, presence_mask)` as in `fit_transform`.
        """
        x = np.asarray(x, dtype=np.float32)
        mask = self.presence_mask(x)

        # Reuse the statistics fitted on the training split.
        if self._stats is not None and self.config.method in {
            'col_mean',
            'global_mean',
            'median',
        }:
            out = np.where(np.isnan(x), self._stats[np.newaxis, :], x)
            return np.nan_to_num(out, nan=0.0).astype(np.float32), mask

        if self._sklearn_imputer is not None and self.config.method in {'knn', 'iterative'}:
            out = self._sklearn_imputer.transform(x)  # type: ignore[attr-defined]
            return np.nan_to_num(out, nan=0.0).astype(np.float32), mask

        # Stateless methods have nothing to reuse.
        return self.fit_transform(x, group_ids)

    def _fill_row_mean(self, x: np.ndarray) -> np.ndarray:
        """Fills each row's NaNs with that row's mean of present features."""
        import warnings  # pylint: disable=import-outside-toplevel

        with warnings.catch_warnings():
            # All-NaN rows (omitted words) produce an intentional NaN row mean.
            warnings.simplefilter('ignore', category=RuntimeWarning)
            row_means = np.nanmean(x, axis=1)

        row_means = np.nan_to_num(row_means, nan=0.0)
        return np.where(np.isnan(x), row_means[:, np.newaxis], x)

    def _fill_column_stat(self, x: np.ndarray, method: str) -> np.ndarray:
        """Fills NaNs with per-column mean/median or a single global mean."""
        with np.errstate(invalid='ignore'):
            if method == 'median':
                stat = np.nanmedian(x, axis=0)
            elif method == 'col_mean':
                stat = np.nanmean(x, axis=0)
            else:  # global_mean
                stat = np.full(x.shape[1], np.nanmean(x), dtype=np.float32)
        stat = np.nan_to_num(stat, nan=0.0).astype(np.float32)
        self._stats = stat
        out = np.where(np.isnan(x), stat[np.newaxis, :], x)
        return out

    def _fill_sklearn(self, x: np.ndarray, kind: str) -> np.ndarray:
        """Fills NaNs with a fitted KNN or iterative (model-based) imputer."""
        try:
            if kind == 'knn':
                from sklearn.impute import KNNImputer

                imputer: object = KNNImputer(n_neighbors=self.config.knn_neighbors)
            else:
                # from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer

                imputer = IterativeImputer(max_iter=self.config.iterative_max_iter, random_state=0)
        except ImportError:  # pragma: no cover - sklearn is a hard dependency
            _LOG.warning('scikit-learn unavailable; falling back to column mean.')
            return self._fill_column_stat(x, 'col_mean')

        # All-NaN columns break sklearn imputers; pre-fill them with zeros.
        all_nan_cols = np.all(np.isnan(x), axis=0)
        safe = x.copy()
        safe[:, all_nan_cols] = 0.0
        out = imputer.fit_transform(safe)  # type: ignore[attr-defined]
        self._sklearn_imputer = imputer
        return np.asarray(out, dtype=np.float32)

    def _fill_sequence(
        self, x: np.ndarray, group_ids: np.ndarray | None, method: str
    ) -> np.ndarray:
        """Forward-fills or interpolates along reading order within each group."""
        import pandas as pd

        if group_ids is None:
            group_ids = np.zeros(len(x), dtype=np.int64)
        frame = pd.DataFrame(x)  # value columns only

        def _fill(block: pd.DataFrame) -> pd.DataFrame:
            if method == 'ffill':
                return block.ffill().bfill()
            return block.interpolate(method=self.config.interpolate_method, limit_direction='both')

        # Group by the id array directly so grouping keys never enter `apply`.
        filled = frame.groupby(group_ids, group_keys=False, sort=False).apply(_fill)
        out = filled.to_numpy(dtype=np.float32)
        return np.nan_to_num(out, nan=0.0)
