"""A lightweight cosine nearest-neighbour index over a labelled embedding bank.

This is a *temporary* stand-in for a learned decoder: given a bank of ZTE embeddings with known labels (e.g. the word each EEG token corresponds to),
it answers "which known items is this new embedding closest to?" and turns that into predictions (majority vote / distance-weighted mean) or a crude
`decode` to the nearest word. It also powers the evaluation suite's kNN probes and content retrieval. Brute-force cosine is used (no FAISS dependency);
ZuCo-scale banks (thousands of tokens) are well within reach.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

type ProbeTask = Literal['auto', 'classification', 'regression']


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalises rows of `x`.

    Args:
        x (np.ndarray): Array `(n_samples, embed_dim)`.
        eps (float): Numerical floor (default `1e-8`).

    Returns:
        np.ndarray: Row-normalised float32 array.
    """
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


# Cap the transient (block x n_items) cosine-similarity tile so a large bank never
# forces one giant (n_queries x n_items) allocation -- at ZuCo word scale that matrix
# is 160k x 160k float32 (~96 GiB) and swaps the box to death.
_QUERY_SIM_BUDGET_BYTES = 128 * 1024 * 1024


def _query_block_rows(n_items: int) -> int:
    """Number of queries to score at once so the similarity tile stays under budget.

    Args:
        n_items (int): Bank size (columns of the similarity tile).

    Returns:
        int: Query-block height in `[1, 4096]`, bounding the tile to ~128 MiB.
    """
    if n_items <= 0:
        return 1
    return max(1, min(4096, _QUERY_SIM_BUDGET_BYTES // (n_items * 4)))


class NearestNeighborIndex:
    """Cosine kNN over a fixed bank of embeddings with aligned metadata.

    Attributes:
        bank (np.ndarray): L2-normalised bank embeddings `(n_items, embed_dim)`.
        metadata (pd.DataFrame): Per-bank-row labels, length `n_items`.

    """

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        *,
        csls: bool = False,
        csls_k: int = 10,
    ) -> None:
        """Builds the index from a bank of embeddings and aligned metadata.

        Args:
            embeddings (np.ndarray): Bank embeddings `(n_items, embed_dim)`.
            metadata (pd.DataFrame): Labels aligned row-for-row with `embeddings`.
            csls (bool): If `True`, correct similarities with Cross-domain Similarity Local Scaling
                (Conneau et al., 2018): `sim -> 2 * cos - r_query - r_bank`, where `r` is a point's mean
                cosine to its `csls_k` nearest bank neighbours. This penalises hub-dense neighbourhoods
                that are everyone's nearest neighbour -- the anisotropy/hubness that drives below-chance
                retrieval -- without adding any signal (a monotone re-ranking). Default `False` keeps
                plain cosine, so every existing caller is unchanged.
            csls_k (int): Neighbourhood size for the CSLS mean-shift term.

        Raises:
            ValueError: If `embeddings` and `metadata` lengths differ.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(embeddings) != len(metadata):
            raise ValueError(
                f'embeddings ({len(embeddings)}) and metadata ({len(metadata)}) must align.'
            )
        self.bank = _l2_normalize(embeddings)
        self.metadata = metadata.reset_index(drop=True)
        self.csls = bool(csls)
        self.csls_k = int(csls_k)
        # Per-bank neighbourhood mean r_bank (the CSLS "hubness" term); None when disabled.
        self.r_bank: np.ndarray | None = (
            self._neighbourhood_means(self.bank, self.csls_k) if self.csls else None
        )

    def _neighbourhood_means(self, points: np.ndarray, m: int) -> np.ndarray:
        """Mean of each row's top-`m` cosine similarities to the bank (excluding an exact self-match).

        Block-tiled so the `(n_items x n_items)` similarity matrix is never materialised -- the same
        memory guard `query` uses (a full self-similarity at ZuCo word scale would be ~96 GiB).

        Args:
            points (np.ndarray): L2-normalised points `(n, embed_dim)` scored against the bank. When
                `points is self.bank` the diagonal self-match is dropped.
            m (int): Neighbourhood size.

        Returns:
            np.ndarray: Per-row mean top-`m` cosine `(n,)`, float32.
        """
        n_items = self.bank.shape[0]
        m = max(1, min(m, n_items - 1))
        is_self = points is self.bank
        out = np.empty(len(points), dtype=np.float32)
        block = _query_block_rows(n_items)
        for start in range(0, len(points), block):
            stop = min(start + block, len(points))
            sims = points[start:stop] @ self.bank.T  # (b, n_items)
            if is_self:
                rows = np.arange(stop - start)
                sims[rows, np.arange(start, stop)] = -np.inf
            top = np.partition(sims, kth=n_items - m, axis=1)[:, -m:]
            out[start:stop] = top.mean(axis=1)
        return out

    def __len__(self) -> int:
        """Returns the number of bank entries."""
        return len(self.bank)

    def query(
        self,
        queries: np.ndarray,
        k: int = 5,
        self_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Finds the top-`k` most similar bank entries for each query.

        Args:
            queries (np.ndarray): Query embeddings `(n_queries, embed_dim)`.
            k (int): Number of neighbours to return.
            self_indices (np.ndarray | None): Optional `(n_queries,)` bank row to exclude per query (for leave-one-out when the queries *are* the bank).

        Returns:
            tuple[np.ndarray, np.ndarray]: `(indices (n_queries, k), similarities (n_queries, k))`, neighbours sorted by descending cosine similarity.

        """
        q = _l2_normalize(queries)
        n_items = self.bank.shape[0]
        k = min(k, n_items)
        n_queries = q.shape[0]
        idx_out = np.empty((n_queries, k), dtype=np.int64)
        sim_out = np.empty((n_queries, k), dtype=np.float32)

        # Score the bank in query-blocks so the full (n_queries x n_items) similarity
        # matrix is never materialised; peak memory stays flat as the bank grows.
        block = _query_block_rows(n_items)
        for start in range(0, n_queries, block):
            stop = min(start + block, n_queries)
            sims = q[start:stop] @ self.bank.T  # (b, n_items)
            if self.csls and self.r_bank is not None:
                # CSLS mean-shift: 2*cos - r_query - r_bank de-hubs the space so anisotropy /
                # hubness no longer let a few points win every top-1 (Conneau et al., 2018).
                m = max(1, min(self.csls_k, n_items - 1))
                r_query = np.partition(sims, kth=n_items - m, axis=1)[:, -m:].mean(axis=1)
                sims = 2.0 * sims - r_query[:, None] - self.r_bank[None, :]
            if self_indices is not None:
                rows = np.arange(stop - start)
                sims[rows, self_indices[start:stop]] = -np.inf
            part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            # Sort each block's top-k slice by similarity.
            order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
            part = np.take_along_axis(part, order, axis=1)
            idx_out[start:stop] = part
            sim_out[start:stop] = np.take_along_axis(sims, part, axis=1)
        return idx_out, sim_out

    def retrieve(self, queries: np.ndarray, k: int = 5) -> list[pd.DataFrame]:
        """Returns the metadata rows of the top-`k` neighbours per query.

        Args:
            queries (np.ndarray): Query embeddings `(n_queries, embed_dim)`.
            k (int): Number of neighbours.

        Returns:
            list[pd.DataFrame]: One DataFrame of neighbour rows per query (with a `similarity` column added).
        """
        idx, sims = self.query(queries, k)
        out: list[pd.DataFrame] = []
        for row, sim in zip(idx, sims, strict=True):
            frame = self.metadata.iloc[row].copy()
            frame['similarity'] = sim
            out.append(frame)
        return out

    def decode(self, queries: np.ndarray, column: str = 'word') -> list[Any]:
        """Temporary decode: returns the nearest bank entry's label per query.

        Args:
            queries (np.ndarray): Query embeddings `(n_queries, embed_dim)`.
            column (str): Metadata column to read (default `word`).

        Returns:
            list[Any]: The nearest-neighbour value of `column` for each query.
        """
        idx, _ = self.query(queries, k=1)
        return self.metadata[column].to_numpy()[idx[:, 0]].tolist()

    def predict(
        self,
        queries: np.ndarray,
        column: str,
        k: int = 5,
        task: ProbeTask = 'auto',
        self_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predicts a metadata column from neighbours (vote for clf, mean for reg).

        Args:
            queries (np.ndarray): Query embeddings `(n_queries, embed_dim)`.
            column (str): Metadata column to predict.
            k (int): Neighbour count.
            task (ProbeTask): `classification`, `regression` or `auto`.
            self_indices (np.ndarray | None): Per-query bank rows to exclude.

        Returns:
            np.ndarray: Predictions `(n_queries,)`.

        """
        values = self.metadata[column].to_numpy()
        if task == 'auto':
            task = 'classification' if _looks_categorical(values) else 'regression'
        idx, sims = self.query(queries, k=k, self_indices=self_indices)
        neighbour_vals = values[idx]  # (n_queries, k)

        if task == 'regression':
            weights = np.clip(sims, 0.0, None)
            weights = np.where(weights.sum(1, keepdims=True) > 0, weights, 1.0)
            return (neighbour_vals.astype(np.float64) * weights).sum(1) / weights.sum(1)
        return np.array([_majority(row) for row in neighbour_vals], dtype=object)


def _looks_categorical(values: np.ndarray) -> bool:
    """Heuristically decides whether label values are categorical.

    Args:
        values (np.ndarray): Candidate label values.

    Returns:
        bool: `True` for non-numeric dtypes or low-cardinality integer labels.
    """
    if values.dtype.kind in {'U', 'S', 'O', 'b'}:
        return True
    return values.dtype.kind in {'i', 'u'} and len(np.unique(values)) <= 20


def _majority(row: np.ndarray) -> Any:
    """Returns the most frequent value in `row` (ties broken by first seen).

    Args:
        row (np.ndarray): Neighbour label values for one query.

    Returns:
        Any: The majority label.
    """
    labels, counts = np.unique(row, return_counts=True)
    return labels[int(np.argmax(counts))]
