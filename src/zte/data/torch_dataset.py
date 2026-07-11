"""PyTorch `Dataset`/`DataLoader` adapters over a processed :class:`ZuCoDataset`.

The self-supervised objectives (skip-gram, CBOW, masked, CPC) all consume a *sentence as a sequence of word-EEG tokens*, so the dataset's atomic item is a
sentence. Each item carries the per-word band-power vector and/or raw EEG window, a **presence mask** (`False` for omitted words -- the key anti-leakage signal),
the subject id and the sequence length. The collate function pads variable-length sentences into a batch and emits a padding mask; objectives build their own
positive/negative pairs and masking on top of this batch.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

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


def _content_key_arrays(ds: ZuCoDataset) -> tuple[np.ndarray, np.ndarray]:
    """Returns positional `(stimulus_key, word_idx)` arrays aligned with `ds.words` rows.

    `stimulus_key` is the normalised sentence text (subject/task-agnostic); together with the
    within-sentence `word_idx` it identifies the *same word of the same sentence* regardless of
    who read it -- the key ingredient for cross-subject positives.

    Args:
        ds (ZuCoDataset): A built dataset.

    Returns:
        tuple[np.ndarray, np.ndarray]: `(stimulus_key[str], word_idx[int])`, each `(n_words,)`.
    """
    words = ds.words
    if 'stimulus_key' in words.columns:
        skey = words['stimulus_key'].fillna('').astype(str).to_numpy()
    else:  # fall back to a per-sentence key when text wasn't propagated
        skey = (words['task'].astype(str) + '|' + words['sentence_idx'].astype(str)).to_numpy()
    if 'word_idx' in words.columns:
        widx = words['word_idx'].to_numpy()
    else:
        widx = np.arange(len(words))
    return skey, widx


def _build_content_vocab(ds: ZuCoDataset) -> dict[tuple[str, int], int]:
    """Builds a stable `{(stimulus_key, word_idx): int}` id map over the whole dataset.

    Ids are contiguous and shared across every split so the same stimulus word maps to the same
    id in train and test. Different subjects reading the same word share an id; that is what lets
    an objective treat same-word-across-subjects as a positive.

    Args:
        ds (ZuCoDataset): A built dataset.

    Returns:
        dict[tuple[str, int], int]: The stimulus-id vocabulary.
    """
    skey, widx = _content_key_arrays(ds)
    vocab: dict[tuple[str, int], int] = {}
    for s, w in zip(skey, widx, strict=True):
        key = (str(s), int(w))
        if key not in vocab:
            vocab[key] = len(vocab)
    return vocab


@dataclass(slots=True)
class SentenceSample:
    """One sentence's tensors prior to collation.

    Attributes:
        features (torch.Tensor | None): `(seq_len, n_features)` band-power token matrix, or `None`.
        raw (torch.Tensor | None): `(seq_len, n_channels, time_steps)` raw EEG windows, or `None`.
        presence (torch.Tensor): `(seq_len,)` boolean presence mask.
        subject (int): Integer subject id.
        length (int): Number of word tokens (`seq_len`).
        content (torch.Tensor | None): `(seq_len,)` long, subject-agnostic stimulus id per token
            (the same word of the same sentence text shares an id across subjects/tasks); omitted
            tokens are `-1`. `None` when not provided (padding-collated to all `-1`).
    """

    features: torch.Tensor | None
    raw: torch.Tensor | None
    presence: torch.Tensor
    subject: int
    length: int
    content: torch.Tensor | None = None
    word_id: torch.Tensor | None = None
    task_id: int = 0


class ZuCoTorchDataset(Dataset[SentenceSample]):
    """Yields sentence-level sequences of word-EEG tokens for SSL training.

    Attributes:
        representation (str | None): Which tensors to emit (`band_power`, `raw` or `both`); defaults to the dataset's own setting.
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

        # Subject-agnostic stimulus-id table, shared across the WHOLE dataset (built once) so ids
        # are stable across splits: key = (normalised sentence text, within-sentence word index).
        self._content_vocab = _build_content_vocab(dataset)
        skey_arr, widx_arr = _content_key_arrays(dataset)
        presence = dataset.presence

        # Subject-agnostic *word identity* (for meaning positives: same word in different sentences)
        # and a *passage/task id* per sentence (for the passage adversary).
        word_arr = (
            dataset.words['word'].fillna('').astype(str).to_numpy()
            if 'word' in dataset.words.columns
            else None
        )
        self._word_vocab: dict[str, int] = (
            {w: i for i, w in enumerate(sorted(set(word_arr.tolist())))}
            if word_arr is not None
            else {}
        )
        self._task_vocab: dict[str, int] = (
            {t: i for i, t in enumerate(sorted(set(dataset.words['task'].astype(str).tolist())))}
            if 'task' in dataset.words.columns
            else {}
        )

        self._sequences: list[np.ndarray] = []
        self._subjects: list[int] = []
        self._content: list[np.ndarray] = []
        self._word_id: list[np.ndarray] = []
        self._task_id: list[int] = []
        self._stimulus_keys: list[str] = []
        for (subject, task, _s_idx), rows in dataset.groups:
            kept = rows if allowed is None else np.array([r for r in rows if r in allowed])
            if len(kept) < min_length:
                continue
            self._sequences.append(kept)
            self._subjects.append(self.subject_vocab.get(subject, 0))
            content = np.full(len(kept), -1, dtype=np.int64)
            word_id = np.full(len(kept), -1, dtype=np.int64)
            for j, r in enumerate(kept):
                if presence is None or bool(presence[r]):
                    content[j] = self._content_vocab[(skey_arr[r], int(widx_arr[r]))]
                    if word_arr is not None:
                        word_id[j] = self._word_vocab.get(word_arr[r], -1)
            self._content.append(content)
            self._word_id.append(word_id)
            self._task_id.append(self._task_vocab.get(str(task), 0))
            self._stimulus_keys.append(str(skey_arr[kept[0]]) if len(kept) else '')

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
        """Integer subject id per sentence, aligned with `sequences`."""
        return self._subjects

    @property
    def stimulus_keys(self) -> list[str]:
        """Normalised sentence-text key per sentence, aligned with `sequences`.

        Sentences sharing a key are the same stimulus read by (potentially) different subjects; the
        stimulus-grouped sampler uses this to co-locate them in a batch.
        """
        return self._stimulus_keys

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
            content=torch.from_numpy(np.ascontiguousarray(self._content[idx])).long(),
            word_id=torch.from_numpy(np.ascontiguousarray(self._word_id[idx])).long(),
            task_id=self._task_id[idx],
        )


def collate_sentences(batch: list[SentenceSample]) -> dict[str, Any]:
    """Pads a list of :class:`SentenceSample` into a batched tensor dict.

    Args:
        batch (list[SentenceSample]): Sentence samples of varying length.

    Returns:
        dict[str, Any]: A dict with keys:
        - `features`: `(batch_size, max_seq_len, n_features)` or `None`.
        - `raw`: `(batch_size, max_seq_len, n_channels, time_steps)` or `None`.
        - `pad_mask`: `(batch_size, max_seq_len)` bool, `True` at valid positions.
        - `presence`: `(batch_size, max_seq_len)` bool, `True` for present (non-omitted) words.
        - `subject`: `(batch_size,)` long.
        - `lengths`: `(batch_size,)` long.
        - `content_id`: `(batch_size, max_seq_len)` long, subject-agnostic stimulus id per token
          (`-1` at padding/omitted positions). Tokens sharing a `content_id` are the same word of
          the same sentence text; used to build cross-subject positives.

    """
    lengths = torch.tensor([s.length for s in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(batch)

    pad_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    presence = torch.zeros(batch_size, max_len, dtype=torch.bool)
    content_id = torch.full((batch_size, max_len), -1, dtype=torch.long)
    word_id = torch.full((batch_size, max_len), -1, dtype=torch.long)
    for i, sample in enumerate(batch):
        pad_mask[i, : sample.length] = True
        presence[i, : sample.length] = sample.presence
        if sample.content is not None:
            content_id[i, : sample.length] = sample.content
        if sample.word_id is not None:
            word_id[i, : sample.length] = sample.word_id

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
        'content_id': content_id,
        'word_id': word_id,
        'task_id': torch.tensor([s.task_id for s in batch], dtype=torch.long),
    }


class StimulusBatchSampler(Sampler[list[int]]):
    """Batch sampler that co-locates sentences of the same stimulus (across subjects).

    Sentences sharing a normalised-text `stimulus_key` are the same sentence read by different
    subjects. Grouping them into the same batch is what makes cross-subject positives actually
    available to the contrastive loss (otherwise `content_id` matches would rarely co-occur).

    Attributes:
        batch_size (int): Target sentences per batch.
        drop_last (bool): Drop a trailing short batch.
    """

    def __init__(
        self,
        stimulus_keys: list[str],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        """Groups sentence indices by stimulus and prepares shuffled batches.

        Args:
            stimulus_keys (list[str]): Per-sentence stimulus key (`ZuCoTorchDataset.stimulus_keys`).
            batch_size (int): Target sentences per batch.
            shuffle (bool): Reshuffle group order (and the epoch seed) each iteration.
            drop_last (bool): Drop a trailing batch smaller than `batch_size`.
            seed (int): Base RNG seed (incremented each epoch when `shuffle`).
        """
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._seed = seed
        self._epoch = 0
        groups: dict[str, list[int]] = defaultdict(list)
        for i, key in enumerate(stimulus_keys):
            groups[key].append(i)
        self._groups = list(groups.values())

    def __iter__(self) -> Iterator[list[int]]:
        """Yields batches of sentence indices with same-stimulus sentences kept adjacent."""
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        order = list(range(len(self._groups)))
        if self.shuffle:
            rng.shuffle(order)
        flat: list[int] = []
        for g in order:
            members = list(self._groups[g])
            if self.shuffle:
                rng.shuffle(members)
            flat.extend(members)
        batches = [flat[i : i + self.batch_size] for i in range(0, len(flat), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        yield from batches

    def __len__(self) -> int:
        """Returns the number of batches per epoch."""
        total = sum(len(g) for g in self._groups)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


def make_dataloader(
    torch_dataset: ZuCoTorchDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    group_by_stimulus: bool = False,
    seed: int = 0,
) -> DataLoader[SentenceSample]:
    """Creates a :class:`~torch.utils.data.DataLoader` with the padding collate.

    Args:
        torch_dataset (ZuCoTorchDataset): A :class:`ZuCoTorchDataset`.
        batch_size (int): Sentences per batch.
        shuffle (bool): Whether to shuffle each epoch.
        num_workers (int): Worker processes (0 = main process; safest cross-platform).
        pin_memory (bool): Pin host memory (helps CUDA throughput; ignored elsewhere).
        drop_last (bool): Drop a final short batch (useful for contrastive losses).
        group_by_stimulus (bool): When `True`, batches are built by :class:`StimulusBatchSampler` so
            that the same sentence read by different subjects co-occurs -- required for cross-subject
            positives to actually appear in a batch. Overrides `shuffle` (the sampler shuffles internally).
        seed (int): Base seed for the stimulus sampler's per-epoch shuffling.

    Returns:
        DataLoader[SentenceSample]: A configured DataLoader yielding collated batch dicts.

    """
    if group_by_stimulus:
        sampler = StimulusBatchSampler(
            torch_dataset.stimulus_keys,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
        return DataLoader(
            torch_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_sentences,
        )
    return DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_sentences,
    )
