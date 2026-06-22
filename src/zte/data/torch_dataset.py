"""PyTorch `Dataset`/`DataLoader` adapters over a processed :class:`ZuCoDataset`.

The self-supervised objectives (skip-gram, CBOW, masked, CPC) all consume a
*sentence as a sequence of word-EEG tokens*, so the dataset's atomic item is a
sentence. Each item carries the per-word band-power vector and/or raw EEG window,
a **presence mask** (`False` for omitted words -- the key anti-leakage signal),
the subject id and the sequence length. The collate function pads variable-length
sentences into a batch and emits a padding mask; objectives build their own
positive/negative pairs and masking on top of this batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from zte.data.dataset import ZuCoDataset


def build_subject_vocab(ds: ZuCoDataset) -> dict[str, int]:
    """Maps each subject code to a contiguous integer id.

    Args:
        ds (ZuCoDataset): A built dataset.

    Returns:
        dict[str, int]: An ordered `{subject: id}` mapping (sorted by subject code).

    """
    return {subj: i for i, subj in enumerate(sorted(ds.words['subject'].unique()))}


@dataclass(slots=True)
class SentenceSample:
    """One sentence's tensors prior to collation.

    Attributes:
        features: torch.Tensor | None: `(L, D)` band-power token matrix, or `None`.
        raw: torch.Tensor | None: `(L, C, T)` raw EEG windows, or `None`.
        presence: torch.Tensor: `(L,)` boolean presence mask.
        subject: int: Integer subject id.
        length: int: Number of word tokens `L`.
    """

    features: torch.Tensor | None
    raw: torch.Tensor | None
    presence: torch.Tensor
    subject: int
    length: int


class ZuCoTorchDataset(Dataset[SentenceSample]):
    """Yields sentence-level sequences of word-EEG tokens for SSL training.

    Attributes:
        representation (str | None): Which tensors to emit (`'band_power'`, `'raw'` or `'both'`); defaults to the dataset's own setting.
        subject_vocab (dict[str, int] | None): The `{subject: id}` mapping in use.

    """

    def __init__(
        self,
        dataset: ZuCoDataset,
        indices: np.ndarray | None = None,
        representation: str | None = None,
        min_length: int = 1,
        subject_vocab: dict[str, int] | None = None,
    ) -> None:
        """Builds the sequence index over (a subset of) the dataset's words.

        Args:
            dataset (ZuCoDataset): A built :class:`ZuCoDataset`.
            indices (np.ndarray | None): Optional word-row indices selecting a split. A sentence is
                kept (restricted to its in-split words) when at least `min_length` of its words fall inside `indices`.
            representation (str | None): Override the emitted representation.
            min_length (int): Minimum surviving words for a sentence to be included.
            subject_vocab (dict[str, int] | None): Optional pre-built subject vocabulary (share across splits so IDs are consistent).

        """
        self._ds = dataset
        self.representation = representation or dataset.config.representation
        self.subject_vocab = subject_vocab or build_subject_vocab(dataset)
        allowed = None if indices is None else set(int(i) for i in indices)

        self._sequences: list[np.ndarray] = []
        self._subjects: list[int] = []
        for (subject, _task, _s_idx), rows in dataset.groups:
            kept = rows if allowed is None else np.array([r for r in rows if r in allowed])
            if len(kept) < min_length:
                continue
            self._sequences.append(kept)
            self._subjects.append(self.subject_vocab.get(subject, 0))

    @property
    def sequences(self) -> list[np.ndarray]:
        """Per-sentence arrays of original word-row indices, in dataset order (`sequences[i]`).

        Note:
            Iterating a non-shuffled DataLoader visits sentences in this order,
            so callers can map per-token model outputs back to word rows via `sequences[i]`.

        """
        return self._sequences

    @property
    def subject_ids(self) -> list[int]:
        """Integer subject id per sentence, aligned with :attr:`sequences`."""
        return self._subjects

    def __len__(self) -> int:
        """Returns the number of sentences in this split."""
        return len(self._sequences)

    def __getitem__(self, idx: int) -> SentenceSample:
        """Returns the tensors for sentence `idx`.

        Args:
            idx (int): Sentence position within this split.

        Returns:
            SentenceSample: A :class:`SentenceSample`.
        """
        rows = self._sequences[idx]
        ds = self._ds
        want_bp = self.representation in {'band_power', 'both'}
        want_raw = self.representation in {'raw', 'both'}

        features = None
        if want_bp and ds.features is not None:
            features = torch.from_numpy(np.ascontiguousarray(ds.features[rows])).float()
        raw = None
        if want_raw and ds.raw_eeg is not None:
            raw = torch.from_numpy(np.ascontiguousarray(ds.raw_eeg[rows])).float()
        if ds.presence is not None:
            presence = torch.from_numpy(np.ascontiguousarray(ds.presence[rows])).bool()
        else:
            presence = torch.ones(len(rows), dtype=torch.bool)
        return SentenceSample(
            features=features,
            raw=raw,
            presence=presence,
            subject=self._subjects[idx],
            length=len(rows),
        )


def collate_sentences(batch: list[SentenceSample]) -> dict[str, Any]:
    """Pads a list of :class:`SentenceSample` into a batched tensor dict.

    Args:
        batch (list[SentenceSample]): Sentence samples of varying length.

    Returns:
        dict[str, Any]: A dict with keys:

        * `features`: `(B, Lmax, D)` or `None`.
        * `raw`: `(B, Lmax, C, T)` or `None`.
        * `pad_mask`: `(B, Lmax)` bool, `True` at valid positions.
        * `presence`: `(B, Lmax)` bool, `True` for present (non-omitted) words.
        * `subject`: `(B,)` long.
        * `lengths`: `(B,)` long.

    """
    lengths = torch.tensor([s.length for s in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(batch)

    pad_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    presence = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, sample in enumerate(batch):
        pad_mask[i, : sample.length] = True
        presence[i, : sample.length] = sample.presence

    features = None
    if batch[0].features is not None:
        dim = batch[0].features.shape[-1]
        features = torch.zeros(batch_size, max_len, dim, dtype=torch.float32)
        for i, sample in enumerate(batch):
            features[i, : sample.length] = sample.features

    raw = None
    if batch[0].raw is not None:
        c, t = batch[0].raw.shape[-2:]
        raw = torch.zeros(batch_size, max_len, c, t, dtype=torch.float32)
        for i, sample in enumerate(batch):
            raw[i, : sample.length] = sample.raw

    return {
        'features': features,
        'raw': raw,
        'pad_mask': pad_mask,
        'presence': presence,
        'subject': torch.tensor([s.subject for s in batch], dtype=torch.long),
        'lengths': lengths,
    }


def make_dataloader(
    torch_dataset: ZuCoTorchDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader[SentenceSample]:
    """Creates a :class:`~torch.utils.data.DataLoader` with the padding collate.

    Args:
        torch_dataset (ZuCoTorchDataset): A :class:`ZuCoTorchDataset`.
        batch_size (int): Sentences per batch.
        shuffle (bool): Whether to shuffle each epoch.
        num_workers (int): Worker processes (0 = main process; safest cross-platform).
        pin_memory (bool): Pin host memory (helps CUDA throughput; ignored elsewhere).
        drop_last (bool): Drop a final short batch (useful for contrastive losses).

    Returns:
        DataLoader[SentenceSample]: A configured DataLoader yielding collated batch dicts.

    """
    return DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_sentences,
    )
