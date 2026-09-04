"""Occlusion saliency and the neighbour gallery for one reading -- an inspection surface, never an evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import torch

from zte.data.dataset import ZuCoDataset
from zte.data.schema import N_CHANNELS
from zte.data.torch_dataset import ZuCoTorchDataset, build_subject_vocab, collate_sentences
from zte.inference.embed import ZTEEmbedder
from zte.lens.montage import azimuthal_xy, load_montage_csv
from zte.logging_utils import get_logger
from zte.utils.provenance import git_info

if TYPE_CHECKING:
    from zte.inference.decode import ZTEDecoder

_LOG = get_logger('lens.saliency')

DISCLAIMER: Final[str] = 'inspection, not a result -- no number here is a headline'
"""The mandatory sentence every lens artifact carries and every lens page renders."""

# Region occlusion always fits (~10 passes); the full per-channel pass runs only when the montage is small enough
# that the whole map stays interactive on CPU, otherwise only the winning regions are refined channel by channel.
MAX_OCCLUSION_PASSES: Final[int] = 130
"""Ceiling on occlusion forward passes for one channel-saliency map."""

# Occlusion variants embedded per forward pass; sentences are short, so this bounds memory, not quality.
_EMBED_CHUNK: Final[int] = 32
"""Occlusion variants per `embed_sentence` call."""


@dataclass(slots=True, frozen=True, kw_only=True)
class Reading:
    """One subject's reading of one sentence, addressed by its position in the dataset's deterministic sentence order.

    Attributes:
        subject (str): Subject code.
        task (str): ZuCo task the sentence was read under.
        text (str): The readable sentence text.
        words (tuple[str, ...]): The sentence's words, in reading order.
        stimulus_key (str): Normalised sentence-text key shared by every subject's reading of this sentence.
        row_indices (np.ndarray): Word-row indices into `dataset.words` for this reading.
        position (int): Index into the full dataset's sentence order (the row of this reading in a gallery embed).
    """

    subject: str
    task: str
    text: str
    words: tuple[str, ...]
    stimulus_key: str
    row_indices: np.ndarray
    position: int

    @property
    def n_words(self) -> int:
        """Number of word tokens in this reading."""
        return len(self.words)


def select_reading(
    dataset: ZuCoDataset,
    subject: str,
    index: int = 0,
    contains: str | None = None,
) -> Reading:
    """Picks the `index`-th sentence reading of `subject` in the dataset's deterministic sentence order.

    Args:
        dataset (ZuCoDataset): A built dataset.
        subject (str): Subject code whose readings are searched.
        index (int, optional): Which of the subject's (filtered) readings to take. Defaults to 0.
        contains (str | None, optional): Keep only sentences whose text contains this string (case-insensitive).
            Defaults to None.

    Returns:
        Reading: The selected reading.

    Raises:
        ValueError: If the subject has no matching reading, or `index` is out of range.
    """
    torch_ds = ZuCoTorchDataset(dataset, subject_vocab=build_subject_vocab(dataset))
    texts = torch_ds.stimulus_texts

    matches: list[int] = []
    for pos, rows in enumerate(torch_ds.sequences):
        first = dataset.words.iloc[int(rows[0])]
        if str(first['subject']) != subject:
            continue
        text = texts.get(torch_ds.stimulus_keys[pos], torch_ds.stimulus_keys[pos])
        if contains is not None and contains.lower() not in text.lower():
            continue
        matches.append(pos)

    if not matches:
        filt = f' containing {contains!r}' if contains else ''
        raise ValueError(f'Subject {subject!r} has no reading{filt} in this dataset.')
    if not 0 <= index < len(matches):
        raise ValueError(f'Subject {subject!r} has {len(matches)} matching readings; index {index} is out of range.')

    position = matches[index]
    rows = torch_ds.sequences[position]
    first = dataset.words.iloc[int(rows[0])]
    key = torch_ds.stimulus_keys[position]

    return Reading(
        subject=subject,
        task=str(first['task']),
        text=texts.get(key, key),
        words=tuple(str(w) for w in dataset.words['word'].iloc[rows]),
        stimulus_key=key,
        row_indices=np.asarray(rows, dtype=np.int64),
        position=position,
    )


# ---- Occlusion machinery ---- #


def _reading_batch(embedder: ZTEEmbedder, dataset: ZuCoDataset, reading: Reading) -> dict[str, Any]:
    """Collates the reading's single sentence and moves it to the embedder's device."""
    torch_ds = ZuCoTorchDataset(dataset, indices=reading.row_indices, subject_vocab=build_subject_vocab(dataset))
    batch = collate_sentences([torch_ds[0]])

    return {k: (v.to(embedder.device.device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _replicate(batch: dict[str, Any], n: int) -> dict[str, Any]:
    """Copies every batch tensor `n` times along the batch axis, so each row can be occluded independently."""
    return {k: (v.repeat(n, *([1] * (v.ndim - 1))) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def _embed_chunks(embedder: ZTEEmbedder, batch: dict[str, Any], chunk: int = _EMBED_CHUNK) -> np.ndarray:
    """Embeds a (possibly large) variant batch at sentence level in bounded slices."""
    objective = embedder.config.objective.name
    n = int(batch['pad_mask'].shape[0])
    out: list[np.ndarray] = []
    for lo in range(0, n, chunk):
        piece = {k: (v[lo : lo + chunk] if torch.is_tensor(v) else v) for k, v in batch.items()}
        out.append(embedder.model.embed_sentence(piece, objective=objective).float().cpu().numpy())

    return np.concatenate(out)


def _cosine_drops(full: np.ndarray, variants: np.ndarray) -> np.ndarray:
    """Per-variant `1 - cos(full, variant)`; a variant that embeds to nothing (NaN/zero) counts as a total drop."""
    f = full / max(float(np.linalg.norm(full)), 1e-8)
    norms = np.linalg.norm(variants, axis=1, keepdims=True)
    cos = np.nan_to_num((variants / np.clip(norms, 1e-8, None)) @ f, nan=0.0)

    return (1.0 - cos).astype(np.float64)


def word_saliency(embedder: ZTEEmbedder, dataset: ZuCoDataset, reading: Reading) -> dict[str, Any]:
    """Scores each word by how much masking it out of the pad mask moves the sentence embedding.

    Occlusion is model-agnostic and honest: word `j` is removed from attention and pooling, the sentence is
    re-embedded, and the score is the cosine drop against the full-sentence embedding, clamped at 0. The unclamped
    drops travel alongside under `raw`, so a word whose removal *helps* the embedding is still visible.

    Args:
        embedder (ZTEEmbedder): The loaded encoder.
        dataset (ZuCoDataset): The built dataset the reading lives in.
        reading (Reading): The reading to inspect.

    Returns:
        dict[str, Any]: `{'scores': [float per word], 'raw': [float per word], 'method': str}`.
    """
    base = _reading_batch(embedder, dataset, reading)
    full = _embed_chunks(embedder, base)[0]

    variants = _replicate(base, reading.n_words)
    idx = torch.arange(reading.n_words, device=variants['pad_mask'].device)
    variants['pad_mask'][idx, idx] = False
    variants['presence'][idx, idx] = False
    raw = _cosine_drops(full, _embed_chunks(embedder, variants))

    return {
        'scores': np.clip(raw, 0.0, None).tolist(),
        'raw': raw.tolist(),
        'method': 'word_occlusion_cosine_drop',
    }


def channel_saliency(
    embedder: ZTEEmbedder,
    dataset: ZuCoDataset,
    reading: Reading,
    max_passes: int = MAX_OCCLUSION_PASSES,
) -> dict[str, Any] | None:
    """Scores scalp channels by occlusion, grouped through the montage's regions, or `None` without a montage.

    Regions are occluded first (each region's channels zeroed together). When the budget allows, every channel is
    then occluded individually; otherwise only the channels of the highest-scoring regions are refined and the rest
    inherit their region's score.

    Args:
        embedder (ZTEEmbedder): The loaded encoder.
        dataset (ZuCoDataset): The built dataset the reading lives in.
        reading (Reading): The reading to inspect.
        max_passes (int, optional): Ceiling on occlusion forward passes. Defaults to `MAX_OCCLUSION_PASSES`.

    Returns:
        dict[str, Any] | None: `{'labels', 'regions', 'xy', 'xyz', 'scores', 'method'}` with one entry per channel,
            or `None` when the dataset carries no montage (or the feature layout cannot address channels).
    """
    if not dataset.config.montage_csv:
        _LOG.info('Channel saliency skipped: the dataset has no montage CSV.')
        return None

    uses_raw = bool(embedder.model.uses_raw)
    if uses_raw:
        if dataset.raw_eeg is None:
            _LOG.warning('Channel saliency skipped: the model is raw but the dataset holds no raw EEG.')
            return None
        n_channels = int(dataset.raw_eeg.shape[1])
        n_bp = 0
    else:
        n_channels = N_CHANNELS
        n_bp = len(dataset.bp_feature_names)
        width = 0 if dataset.features is None else int(dataset.features.shape[1])
        if n_bp == 0 or n_bp * n_channels > width:
            _LOG.warning('Channel saliency skipped: the band-power layout cannot address individual channels.')
            return None

    montage = load_montage_csv(dataset.config.montage_csv, n_channels)
    if montage is None:
        _LOG.warning('Channel saliency skipped: no usable montage at %s.', dataset.config.montage_csv)
        return None
    labels, regions, xyz = montage.labels, montage.regions, montage.xyz

    # Region membership in first-appearance order, so the map reads anterior -> posterior like the montage does.
    region_names = list(dict.fromkeys(regions))
    region_channels = {name: [c for c in range(n_channels) if regions[c] == name] for name in region_names}

    base = _reading_batch(embedder, dataset, reading)
    full = _embed_chunks(embedder, base)[0]

    def occlude(groups: list[list[int]]) -> np.ndarray:
        variants = _replicate(base, len(groups))
        for row, channels in enumerate(groups):
            if uses_raw:
                variants['raw'][row][:, channels, :] = 0.0
            else:
                cols = [f * n_channels + c for f in range(n_bp) for c in channels]
                variants['features'][row][:, cols] = 0.0
        return _cosine_drops(full, _embed_chunks(embedder, variants))

    region_drops = occlude([region_channels[name] for name in region_names])
    scores = np.empty(n_channels, dtype=np.float64)
    for r, name in enumerate(region_names):
        scores[region_channels[name]] = region_drops[r]

    if len(region_names) + n_channels <= max_passes:
        # The budget affords the exact map: one occlusion per channel.
        scores = occlude([[c] for c in range(n_channels)])
        method = 'channel_occlusion_cosine_drop'
    else:
        # Refine the winning regions channel by channel; everyone else keeps their region's score.
        budget = max_passes - len(region_names)
        refined = False
        for r in np.argsort(-region_drops):
            channels = region_channels[region_names[int(r)]]
            if len(channels) > budget:
                continue
            scores[channels] = occlude([[c] for c in channels])
            budget -= len(channels)
            refined = True
        method = 'region_occlusion_cosine_drop' + ('+channel_refined' if refined else '')

    return {
        'labels': labels,
        'regions': regions,
        'xy': azimuthal_xy(xyz).tolist(),
        'xyz': xyz.tolist(),
        'scores': np.clip(scores, 0.0, None).tolist(),
        'method': method,
    }


# ---- The neighbour gallery ---- #


def neighbors(
    embeddings: np.ndarray,
    query_idx: int,
    texts: list[str],
    subjects: list[str],
    stimulus_keys: list[str],
    k: int = 10,
) -> list[dict[str, Any]]:
    """Top-`k` cosine neighbours of one reading over every *other* reading in the gallery.

    The query row is excluded by construction and can never appear in its own gallery. Other subjects' readings of
    the same sentence do appear and are flagged `is_true_sentence`; every neighbour carries its subject, so a
    same-subject match is visible for what it is.

    Args:
        embeddings (np.ndarray): `(n_readings, embed_dim)` sentence embeddings over the full dataset order.
        query_idx (int): Row of the query reading.
        texts (list[str]): Readable sentence text per row.
        subjects (list[str]): Subject code per row.
        stimulus_keys (list[str]): Normalised sentence-text key per row (same key = same sentence).
        k (int, optional): Neighbours to return. Defaults to 10.

    Returns:
        list[dict[str, Any]]: `{'text', 'cosine', 'subject', 'is_true_sentence'}` per neighbour, most similar first.
    """
    normed = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)
    sims = np.nan_to_num(normed @ normed[query_idx], nan=-1.0)

    order = np.argsort(-sims)
    order = order[order != query_idx][: max(int(k), 0)]

    return [
        {
            'text': texts[i],
            'cosine': float(sims[i]),
            'subject': subjects[i],
            'is_true_sentence': stimulus_keys[i] == stimulus_keys[query_idx],
        }
        for i in order
    ]


# ---- Assembly ---- #


def lens_report(
    embedder: ZTEEmbedder,
    dataset: ZuCoDataset,
    reading: Reading,
    decoder: ZTEDecoder | None = None,
    ckpt_path: str | Path | None = None,
    top_k: int = 10,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Builds the full `lens.json` dict for one reading: embedding, saliency, neighbours and the optional decode.

    Args:
        embedder (ZTEEmbedder): The loaded encoder.
        dataset (ZuCoDataset): The built dataset the reading lives in.
        reading (Reading): The reading to inspect.
        decoder (ZTEDecoder | None, optional): A loaded decoder; switches the report to decode mode and adds the
            generation trace. Defaults to None.
        ckpt_path (str | Path | None, optional): Checkpoint path, hashed into provenance. Defaults to None.
        top_k (int, optional): Neighbours in the gallery. Defaults to 10.
        max_new_tokens (int, optional): Decode cap for decode mode. Defaults to None, which uses the configured value.

    Returns:
        dict[str, Any]: The `lens.json` payload, disclaimer included.
    """
    torch_ds = ZuCoTorchDataset(dataset, subject_vocab=build_subject_vocab(dataset))
    text_by_key = torch_ds.stimulus_texts
    keys = list(torch_ds.stimulus_keys)

    gallery, meta = embedder.embed(dataset, level='sentence')
    query = gallery[reading.position]
    gallery_texts = [text_by_key.get(key, key) for key in keys]
    gallery_subjects = [str(s) for s in meta['subject'].tolist()]

    holdout = embedder.config.train.loso_holdout_subject
    decode = None
    if decoder is not None:
        from zte.lens.trace import decode_trace

        decode = decode_trace(decoder, dataset, reading, max_new_tokens=max_new_tokens)

    from zte.training.init import file_sha256

    return {
        'mode': 'encode' if decoder is None else 'decode',
        'reading': {
            'subject': reading.subject,
            'task': reading.task,
            'text': reading.text,
            'words': list(reading.words),
            'n_words': reading.n_words,
            'is_holdout': bool(holdout is not None and reading.subject == holdout),
        },
        'embedding': {'dim': int(query.shape[0]), 'norm': float(np.linalg.norm(query))},
        'word_saliency': word_saliency(embedder, dataset, reading),
        'channel_saliency': channel_saliency(embedder, dataset, reading),
        'neighbors': neighbors(gallery, reading.position, gallery_texts, gallery_subjects, keys, k=top_k),
        'decode': decode,
        'disclaimer': DISCLAIMER,
        'provenance': {
            'ckpt': None if ckpt_path is None else str(ckpt_path),
            'ckpt_sha256': None if ckpt_path is None else file_sha256(ckpt_path),
            'run_name': embedder.config.run_name,
            'git_commit': git_info()['commit'],
            'train_holdout': holdout,
        },
    }
