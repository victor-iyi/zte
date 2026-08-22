"""Train-fitted removal of the sentence-length subspace from exported embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np

from zte.logging_utils import get_logger

_LOG = get_logger('models.encoder.nuisance')

# Length enters retrieval through more than one route -- more words means more tokens to pool, a longer eye-tracking
# trace and a bigger pad mask -- so a straight line in n_words does not span it. This basis is small enough to fit on
# a few hundred sentences and rich enough to catch the saturating part.
_BASIS_NAMES: tuple[str, ...] = ('const', 'n', 'log_n', 'inv_n', 'n_squared')


class LengthProjector:
    """Removes the component of an embedding that word count alone can predict.

    The length confound on ZuCo is not a nuisance to be noted in a caption; it is 5.14 of the 9.45 bits needed to name
    a sentence, and a length-only oracle beats every encoder measured here on every top-k. Length-stratified
    evaluation answers "would this hit survive if length were held constant". This answers the stronger question:
    make the representation itself carry no length, then measure what is left.

    Note:
        The regression is fitted on the training split only and applied row by row afterwards, exactly like the
        decoder's modality-gap correction. Fitting it on anything a held-out row can see would be transductive, and
        `metrics['postprocess_fit']` has to keep saying `train split` for the number to mean anything.

    Attributes:
        dim (int): Embedding width.
        n_fit (int): Rows the current coefficients were fitted on.
        fitted (bool): Whether `fit` has run.
    """

    __slots__ = ('coefficients', 'dim', 'fitted', 'mean_basis', 'n_fit', 'ridge')

    def __init__(self, dim: int, ridge: float = 1e-3) -> None:
        """Builds an unfitted projector, which is the identity until `fit` is called.

        Args:
            dim (int): Embedding width.
            ridge (float, optional): Ridge penalty on the basis regression. Defaults to 1e-3.
        """
        self.dim = int(dim)
        self.ridge = float(ridge)
        self.coefficients = np.zeros((len(_BASIS_NAMES), self.dim), dtype=np.float32)
        self.mean_basis = np.zeros(len(_BASIS_NAMES), dtype=np.float32)
        self.n_fit = 0
        self.fitted = False

    def fit(self, embeddings: np.ndarray, n_words: np.ndarray) -> None:
        """Fits the length basis against training embeddings.

        Args:
            embeddings (np.ndarray): `(n, dim)` training-split embeddings.
            n_words (np.ndarray): `(n,)` word counts of the sentences those rows came from.

        Raises:
            ValueError: If the shapes disagree or there are too few rows to fit.
        """
        z = np.asarray(embeddings, dtype=np.float64)
        n = np.asarray(n_words, dtype=np.float64).reshape(-1)
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise ValueError(f'embeddings must be (n, {self.dim}), got {z.shape}.')
        if z.shape[0] != n.shape[0]:
            raise ValueError(f'{z.shape[0]} embeddings against {n.shape[0]} word counts.')
        if z.shape[0] <= len(_BASIS_NAMES):
            raise ValueError(f'need more than {len(_BASIS_NAMES)} rows to fit a length projector, got {z.shape[0]}.')

        basis = _length_basis(n)
        self.mean_basis = basis.mean(axis=0).astype(np.float32)
        centred = basis - basis.mean(axis=0)

        gram = centred.T @ centred + self.ridge * np.eye(len(_BASIS_NAMES))
        self.coefficients = np.linalg.solve(gram, centred.T @ (z - z.mean(axis=0))).astype(np.float32)
        self.n_fit = int(z.shape[0])
        self.fitted = True

        explained = _explained_variance(z, centred @ self.coefficients.astype(np.float64))
        _LOG.info(
            'Fitted LengthProjector on %d rows; word count linearly explained %.4f of embedding variance.',
            self.n_fit,
            explained,
        )

    def transform(self, embeddings: np.ndarray, n_words: np.ndarray) -> np.ndarray:
        """Subtracts the fitted length component from `embeddings`.

        Args:
            embeddings (np.ndarray): `(n, dim)` embeddings.
            n_words (np.ndarray): `(n,)` word counts.

        Returns:
            np.ndarray: `(n, dim)` embeddings with the length subspace removed; unchanged when unfitted.
        """
        if not self.fitted:
            _LOG.warning('LengthProjector is unfitted and is passing embeddings through.')
            return np.asarray(embeddings, dtype=np.float32)

        basis = _length_basis(np.asarray(n_words, dtype=np.float64).reshape(-1)) - self.mean_basis
        return (np.asarray(embeddings, dtype=np.float32) - (basis @ self.coefficients).astype(np.float32)).astype(
            np.float32
        )

    @property
    def state(self) -> dict[str, Any]:
        """Returns a serialisable dict of the fitted coefficients, for the run's provenance."""
        return {
            'dim': self.dim,
            'ridge': self.ridge,
            'basis': list(_BASIS_NAMES),
            'n_fit': self.n_fit,
            'fitted': self.fitted,
            'coefficients': self.coefficients,
            'mean_basis': self.mean_basis,
        }


def length_leakage(embeddings: np.ndarray, n_words: np.ndarray) -> float:
    """Returns the fraction of embedding variance that word count linearly explains.

    This is the number the projector is built to drive to zero, and reporting it before and after is the only way to
    show the projection did what it claims rather than merely shrinking the vectors.

    Args:
        embeddings (np.ndarray): `(n, dim)` embeddings.
        n_words (np.ndarray): `(n,)` word counts.

    Returns:
        float: Explained-variance fraction in `[0, 1]`, or 0.0 when there are too few rows to fit.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    n = np.asarray(n_words, dtype=np.float64).reshape(-1)
    if z.ndim != 2 or z.shape[0] <= len(_BASIS_NAMES) or np.ptp(n) == 0:
        return 0.0

    basis = _length_basis(n)
    centred = basis - basis.mean(axis=0)
    gram = centred.T @ centred + 1e-3 * np.eye(len(_BASIS_NAMES))
    coefficients = np.linalg.solve(gram, centred.T @ (z - z.mean(axis=0)))

    return _explained_variance(z, centred @ coefficients)


def _length_basis(n_words: np.ndarray) -> np.ndarray:
    """Returns the `(n, 5)` design matrix of length features for a vector of word counts."""
    n = np.clip(n_words, 1.0, None)
    return np.stack([np.ones_like(n), n, np.log(n), 1.0 / n, n**2], axis=1)


def _explained_variance(z: np.ndarray, fitted: np.ndarray) -> float:
    """Returns the share of `z`'s total variance the fitted component accounts for."""
    total = float(((z - z.mean(axis=0)) ** 2).sum())
    if total <= 0.0:
        return 0.0

    return float((fitted**2).sum() / total)
