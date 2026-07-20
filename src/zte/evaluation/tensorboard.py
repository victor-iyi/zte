"""TensorBoard reporter for ZTE embeddings, metrics and figures; a no-op when `tensorboard` is missing."""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zte.logging_utils import get_logger

_LOG = get_logger('evaluation.tensorboard')

#: Scalar-friendly hyper-parameters surfaced to the HParams dashboard.
_HPARAM_KEYS = (
    ('objective', 'name'),
    ('model', 'frontend'),
    ('model', 'pos_encoding'),
    ('model', 'embed_dim'),
    ('model', 'hidden_dim'),
    ('model', 'n_layers'),
    ('dataset', 'representation'),
    ('dataset', 'include_eye_tracking'),
    ('train', 'lr'),
    ('train', 'epochs'),
)


class TensorBoardReporter:
    """Thin, optional-dependency wrapper over `SummaryWriter`.

    Attributes:
        writer (Any | None): The underlying `SummaryWriter`, or `None` when unavailable.
        log_dir (Path): Directory the events are written to.
    """

    def __init__(self, log_dir: str | Path) -> None:
        """Opens a writer at `log_dir` (no-op if TensorBoard is missing).

        Args:
            log_dir (str | Path): Destination directory for the event files.
        """
        self.log_dir = Path(log_dir)
        self.writer: Any | None = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:  # pragma: no cover - optional dependency
            _LOG.warning('TensorBoard unavailable; install `tensorboard` to enable logging.')

    @property
    def enabled(self) -> bool:
        """Whether a live writer is attached."""
        return self.writer is not None

    def log_scalars(self, prefix: str, values: dict[str, Any], step: int = 0) -> None:
        """Logs every finite numeric value in `values` under `prefix/`.

        Args:
            prefix (str): Scalar-tag prefix (e.g. `health`).
            values (dict[str, Any]): Mapping of name -> value (non-numeric entries are skipped).
            step (int): Global step for the scalars.
        """
        if not self.writer:
            return
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
                continue
            if np.isfinite(value):
                self.writer.add_scalar(f'{prefix}/{key}', float(value), step)

    def log_embeddings(
        self,
        emb: np.ndarray,
        meta: pd.DataFrame,
        tag: str = 'thought_embeddings',
        columns: tuple[str, ...] = ('subject', 'task', 'category', 'word'),
        max_points: int = 5000,
        seed: int = 0,
    ) -> None:
        """Writes an Embedding Projector checkpoint with per-point metadata.

        Args:
            emb (np.ndarray): Embeddings `(n_samples, embed_dim)`.
            meta (pd.DataFrame): Aligned metadata; the intersection with `columns` labels the points.
            tag (str): Projector tag.
            columns (tuple[str, ...]): Metadata columns to expose (missing ones are dropped).
            max_points (int): Subsample cap (the projector is slow above a few thousand).
            seed (int): Sampling seed.
        """
        if not self.writer or len(emb) == 0:
            return
        cols = [c for c in columns if c in meta.columns] or None
        idx = np.arange(len(emb))
        if len(emb) > max_points:
            idx = np.random.default_rng(seed).choice(len(emb), size=max_points, replace=False)
        if cols:
            frame = meta.iloc[idx][cols].astype(str)
            metadata = frame.to_numpy().tolist()
            header = list(cols)
        else:  # projector needs at least a single-column label
            metadata, header = [str(i) for i in idx], None
        import torch

        self.writer.add_embedding(
            torch.from_numpy(np.asarray(emb[idx], dtype=np.float32)),
            metadata=metadata,
            metadata_header=header,
            tag=tag,
        )
        _LOG.info('Logged %d points to the TensorBoard projector (tag=%s).', len(idx), tag)

    def log_histogram(self, tag: str, values: np.ndarray, step: int = 0) -> None:
        """Logs a histogram of arbitrary values.

        Args:
            tag (str): Histogram tag.
            values (np.ndarray): Array of values.
            step (int): Global step.
        """
        if not self.writer:
            return
        self.writer.add_histogram(tag, np.asarray(values, dtype=np.float32).ravel(), step)

    def log_embedding_stats(self, emb: np.ndarray) -> None:
        """Logs value + per-dimension-norm histograms (collapse diagnostics)."""
        if not self.writer or len(emb) == 0:
            return
        emb = np.asarray(emb, dtype=np.float32)
        self.log_histogram('embeddings/values', emb)
        self.log_histogram('embeddings/per_dim_std', emb.std(axis=0))
        self.log_histogram('embeddings/row_norm', np.linalg.norm(emb, axis=1))

    def log_figure(self, name: str, fig: Any, step: int = 0) -> None:
        """Logs a Matplotlib figure as an image.

        Args:
            name (str): Image tag.
            fig (Any): A Matplotlib figure.
            step (int): Global step.
        """
        if not self.writer:
            return
        self.writer.add_figure(name, fig, global_step=step, close=False)

    def log_image_file(self, name: str, path: str | Path, step: int = 0) -> None:
        """Logs a saved PNG/JPEG image file by tag.

        Args:
            name (str): Image tag.
            path (str | Path): Path to an image file.
            step (int): Global step.
        """
        if not self.writer or not Path(path).is_file():
            return
        import matplotlib.image as mpimg

        img = mpimg.imread(str(path))  # HxWxC in [0, 1]
        self.writer.add_image(name, np.transpose(img[..., :3], (2, 0, 1)), global_step=step)

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        """Logs a block of Markdown/text.

        Args:
            tag (str): Text tag.
            text (str): The (Markdown) content.
            step (int): Global step.
        """
        if self.writer:
            self.writer.add_text(tag, text, step)

    def log_table(self, tag: str, rows: list[dict[str, Any]], step: int = 0) -> None:
        """Logs a list-of-dicts as a Markdown table under `tag`."""
        if not self.writer or not rows:
            return
        cols = list(rows[0])
        head = '| ' + ' | '.join(cols) + ' |\n| ' + ' | '.join(['---'] * len(cols)) + ' |\n'
        body = ''.join('| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |\n' for r in rows)
        self.writer.add_text(tag, head + body, step)

    def log_hparams(self, config: Any, metrics: dict[str, float]) -> None:
        """Joins a run's key hyper-parameters to its headline metrics.

        Args:
            config (Any): A `ZTEConfig` (or anything with the same nested attributes).
            metrics (dict[str, float]): Scalar metrics to associate with this run.
        """
        if not self.writer:
            return
        hparams: dict[str, Any] = {}
        for section, field in _HPARAM_KEYS:
            try:
                value = getattr(getattr(config, section), field)
            except AttributeError:
                continue
            hparams[f'{section}.{field}'] = (
                value if isinstance(value, (int, float, str, bool)) else str(value)
            )
        clean = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
        }
        if clean:
            self.writer.add_hparams(hparams, clean)

    def close(self) -> None:
        """Flushes and closes the writer."""
        if self.writer:
            self.writer.flush()
            self.writer.close()

    def __enter__(self) -> TensorBoardReporter:
        """Context-manager entry."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Context-manager exit; closes the writer."""
        self.close()
