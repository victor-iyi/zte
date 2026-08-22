"""The lens analysis core: occlusion saliency, the self-excluding neighbour gallery, and the decode trace."""

import csv
import os
from pathlib import Path
from typing import Any, Final, cast

# The whole decoder test path is offline: `lm_source='tiny'` builds its LM and tokeniser locally.
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import torch
from torch import nn

from zte.config import DecoderConfig, ModelConfig, ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.data.montage.regions import SCALP_REGIONS
from zte.data.schema import N_CHANNELS
from zte.device import resolve_device
from zte.inference.decode import ZTEDecoder
from zte.inference.embed import ZTEEmbedder
from zte.lens.saliency import (
    DISCLAIMER,
    Reading,
    channel_saliency,
    lens_report,
    neighbors,
    select_reading,
    word_saliency,
)
from zte.lens.trace import decode_trace
from zte.models.decoder import GapCorrector, build_bridge, build_lm
from zte.models.embedding import ZTEModel, build_model

_EMBED_DIM: Final[int] = 8
"""Width of the stub embedding: the first band-power feature over the first eight channels."""

_LENS_KEYS: Final[frozenset[str]] = frozenset(
    {
        'mode',
        'reading',
        'embedding',
        'word_saliency',
        'channel_saliency',
        'neighbors',
        'decode',
        'disclaimer',
        'provenance',
    }
)
"""Top-level keys of the lens.json contract."""


class _SumModel(nn.Module):
    """Embeds a sentence as the masked sum of its first feature columns, so every occlusion effect is exact."""

    uses_raw = False

    def __init__(self, embed_dim: int = _EMBED_DIM) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self._unused = nn.Parameter(torch.zeros(1), requires_grad=False)

    def embed_sentence(self, batch: dict[str, Any], objective: object = None) -> torch.Tensor:
        """Masked sum over valid word tokens, truncated to `embed_dim` columns."""
        feats = batch['features']
        mask = (batch['pad_mask'] & batch['presence']).unsqueeze(-1).to(feats.dtype)

        return (feats * mask).sum(dim=1)[:, : self.embed_dim]


def _stub_embedder(holdout: str | None = None) -> ZTEEmbedder:
    """An embedder whose model is the exact stub above, with the LOSO holdout planted in its config."""
    config = ZTEConfig()
    config.train.loso_holdout_subject = holdout

    # The lens only calls `embed_sentence`/`uses_raw`, which the stub provides exactly.
    return ZTEEmbedder(cast('ZTEModel', _SumModel()), config, resolve_device('cpu'))


def _plant_reading(dataset: ZuCoDataset, reading: Reading, carrier_word: int, carrier_channel: int) -> None:
    """Rigs one reading's features so `carrier_word` and `carrier_channel` carry the whole embedding signal."""
    rows = reading.row_indices
    assert dataset.features is not None
    block = np.zeros((len(rows), dataset.features.shape[1]), dtype=np.float32)
    block[:, 1] = 0.05
    block[carrier_word, carrier_channel] = 30.0
    dataset.features[rows] = block
    assert dataset.presence is not None
    dataset.presence[rows] = True


def _write_montage(path: Path) -> None:
    """Writes a 105-channel `channel,x,y,z,label,region` montage CSV over the eight scalp regions."""
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['channel', 'x', 'y', 'z', 'label', 'region'])
        for c in range(N_CHANNELS):
            angle = 2.0 * np.pi * c / N_CHANNELS
            x, y = 0.7 * np.cos(angle), 0.7 * np.sin(angle)
            z = float(np.sqrt(max(0.0, 1.0 - x * x - y * y)))
            region = SCALP_REGIONS[min(c * len(SCALP_REGIONS) // N_CHANNELS, len(SCALP_REGIONS) - 1)]
            writer.writerow([c, f'{x:.6f}', f'{y:.6f}', f'{z:.6f}', f'E{c + 1}', region])


def _tiny_decoder(dataset: ZuCoDataset) -> ZTEDecoder:
    """An untrained decoder over the offline tiny LM: tracing needs shapes and wiring, not learned weights."""
    torch.manual_seed(0)
    decoder_config = DecoderConfig(
        lm_source='tiny',
        tokenizer_source='tiny',
        max_target_tokens=24,
        prefix_slots=2,
        bottleneck=8,
        max_new_tokens=8,
    )
    model_config = ModelConfig(embed_dim=16, hidden_dim=16, n_layers=1, n_heads=2, projection_hidden=16)
    assert dataset.features is not None
    model = build_model(model_config, in_dim=int(dataset.features.shape[1]))
    lm = build_lm(decoder_config, encoder=model)
    bridge, _ = build_bridge(decoder_config, 16, 16, lm.hidden_dim)

    return ZTEDecoder(
        model=model,
        config=ZTEConfig(model=model_config, decoder=decoder_config),
        decoder_config=decoder_config,
        bridge=bridge,
        lm=lm,
        gap=GapCorrector(16, mode='none'),
        device=resolve_device('cpu'),
    )


# ---- Word saliency ---- #


def test_word_saliency_finds_the_word_that_carries_the_signal(small_dataset: ZuCoDataset) -> None:
    """The carrying word's occlusion drop strictly exceeds every other word's."""
    embedder = _stub_embedder()
    reading = select_reading(small_dataset, 'ZAB', index=0)
    assert reading.n_words >= 2
    carrier = 1
    _plant_reading(small_dataset, reading, carrier_word=carrier, carrier_channel=0)

    result = word_saliency(embedder, small_dataset, reading)

    assert result['method'] == 'word_occlusion_cosine_drop'
    scores = result['scores']
    assert len(scores) == reading.n_words == len(result['raw'])
    others = [s for j, s in enumerate(scores) if j != carrier]
    assert scores[carrier] > max(others)
    assert all(s >= 0.0 for s in scores)


# ---- Channel saliency ---- #


def test_channel_saliency_is_none_without_a_montage(small_dataset: ZuCoDataset) -> None:
    """No montage CSV means no scalp panel: the function degrades to `None` rather than guessing geometry."""
    reading = select_reading(small_dataset, 'ZAB', index=0)

    assert channel_saliency(_stub_embedder(), small_dataset, reading) is None


def test_channel_saliency_finds_the_carrying_channel_on_a_montage(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """With a montage, per-channel occlusion pins the planted carrier channel as the top score."""
    montage = tmp_path / 'montage.csv'
    _write_montage(montage)
    small_dataset.config.montage_csv = str(montage)
    embedder = _stub_embedder()
    reading = select_reading(small_dataset, 'ZAB', index=0)
    carrier_channel = 3
    _plant_reading(small_dataset, reading, carrier_word=0, carrier_channel=carrier_channel)

    result = channel_saliency(embedder, small_dataset, reading)

    assert result is not None
    assert result['method'] == 'channel_occlusion_cosine_drop'
    assert len(result['labels']) == len(result['regions']) == len(result['scores']) == N_CHANNELS
    assert len(result['xy']) == len(result['xyz']) == N_CHANNELS
    assert all(len(pt) == 2 for pt in result['xy'])
    assert int(np.argmax(result['scores'])) == carrier_channel
    assert result['labels'][carrier_channel] == f'E{carrier_channel + 1}'


def test_channel_saliency_respects_the_pass_budget(small_dataset: ZuCoDataset, tmp_path: Path) -> None:
    """A budget too small for per-channel occlusion falls back to region scores with partial refinement."""
    montage = tmp_path / 'montage.csv'
    _write_montage(montage)
    small_dataset.config.montage_csv = str(montage)
    reading = select_reading(small_dataset, 'ZAB', index=0)
    _plant_reading(small_dataset, reading, carrier_word=0, carrier_channel=3)

    result = channel_saliency(_stub_embedder(), small_dataset, reading, max_passes=40)

    assert result is not None
    assert result['method'].startswith('region_occlusion_cosine_drop')
    assert len(result['scores']) == N_CHANNELS


# ---- The neighbour gallery ---- #


def test_neighbors_never_contains_the_query_reading() -> None:
    """The query row is excluded by construction; nothing in the gallery can score a self-match's cosine of 1."""
    embeddings = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    texts = ['the query sentence', 'a', 'b', 'c']
    subjects = ['ZAB', 'ZDM', 'ZJN', 'ZKW']
    keys = ['s0', 's1', 's2', 's3']

    out = neighbors(embeddings, 0, texts, subjects, keys, k=3)

    assert len(out) == 3
    assert [n['text'] for n in out] == ['a', 'b', 'c']
    # The query itself would score cosine 1.0; every honest neighbour here sits strictly below it.
    assert all(n['cosine'] < 0.99 for n in out)


def test_neighbors_flags_true_sentences_and_carries_subjects() -> None:
    """Another subject's reading of the same sentence appears, flagged; a same-subject reading carries its subject."""
    embeddings = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    texts = ['shared sentence', 'shared sentence', 'another one', 'far away']
    subjects = ['ZAB', 'ZDM', 'ZAB', 'ZJN']
    keys = ['s0', 's0', 's5', 's2']

    out = neighbors(embeddings, 0, texts, subjects, keys, k=3)

    assert out[0]['subject'] == 'ZDM'
    assert out[0]['is_true_sentence'] is True
    assert out[0]['cosine'] > 0.999
    assert out[1]['subject'] == 'ZAB'
    assert out[1]['is_true_sentence'] is False


# ---- The assembled report ---- #


def test_is_holdout_reflects_the_checkpoint_loso_config(small_dataset: ZuCoDataset) -> None:
    """A reading is a held-out brain exactly when the checkpoint's LOSO holdout names its subject."""
    reading = select_reading(small_dataset, 'ZAB', index=0)

    held = lens_report(_stub_embedder(holdout='ZAB'), small_dataset, reading, top_k=3)
    trained = lens_report(_stub_embedder(holdout='ZDM'), small_dataset, reading, top_k=3)
    unset = lens_report(_stub_embedder(holdout=None), small_dataset, reading, top_k=3)

    assert held['reading']['is_holdout'] is True
    assert trained['reading']['is_holdout'] is False
    assert unset['reading']['is_holdout'] is False
    assert held['provenance']['train_holdout'] == 'ZAB'


def test_encode_report_matches_the_contract_and_carries_the_disclaimer(small_dataset: ZuCoDataset) -> None:
    """An encoder-only report fills every schema key, with `decode` and `channel_saliency` honestly null."""
    reading = select_reading(small_dataset, 'ZDM', index=1)

    report = lens_report(_stub_embedder(), small_dataset, reading, top_k=4)

    assert set(report) == set(_LENS_KEYS)
    assert report['mode'] == 'encode'
    assert report['decode'] is None
    assert report['channel_saliency'] is None
    assert report['disclaimer'] == DISCLAIMER == 'inspection, not a result -- no number here is a headline'
    assert report['reading']['subject'] == 'ZDM'
    assert report['reading']['n_words'] == reading.n_words == len(report['reading']['words'])
    assert report['embedding']['dim'] == _EMBED_DIM
    assert len(report['neighbors']) == 4
    assert set(report['provenance']) == {'ckpt', 'ckpt_sha256', 'run_name', 'git_commit', 'train_holdout'}


def test_select_reading_contains_filters_by_text(small_dataset: ZuCoDataset) -> None:
    """The `contains` filter narrows selection to sentences whose text carries the fragment."""
    first = select_reading(small_dataset, 'ZAB', index=0)
    fragment = first.words[0]

    filtered = select_reading(small_dataset, 'ZAB', index=0, contains=fragment)

    assert fragment.lower() in filtered.text.lower()


# ---- The decode trace ---- #


def test_decode_trace_produces_the_schema_on_the_tiny_decoder(small_dataset: ZuCoDataset) -> None:
    """The trace fills the decode contract: slot influence per prefix slot, tokens, and the null-prefix decode."""
    decoder = _tiny_decoder(small_dataset)
    reading = select_reading(small_dataset, 'ZAB', index=0)

    trace = decode_trace(decoder, small_dataset, reading, max_new_tokens=6)

    assert set(trace) == {'generated', 'tokens', 'slot_influence', 'word_evidence', 'null_prefix_generated', 'method'}
    assert isinstance(trace['generated'], str)
    assert isinstance(trace['null_prefix_generated'], str)
    assert all(isinstance(t, str) for t in trace['tokens'])
    assert len(trace['slot_influence']) == decoder.decoder_config.prefix_slots == 2
    assert all(v >= 0.0 for v in trace['slot_influence'])
    assert trace['word_evidence'] is None
    assert trace['method'] == 'slot_occlusion_token_logprob_divergence'


def test_decode_report_carries_the_trace_and_the_disclaimer(small_dataset: ZuCoDataset) -> None:
    """A decode-mode report keeps the full schema and the disclaimer travels with the decode block."""
    decoder = _tiny_decoder(small_dataset)
    embedder = ZTEEmbedder(decoder.model, decoder.config, decoder.device)
    reading = select_reading(small_dataset, 'ZDM', index=0)

    report = lens_report(embedder, small_dataset, reading, decoder=decoder, top_k=5, max_new_tokens=4)

    assert set(report) == set(_LENS_KEYS)
    assert report['mode'] == 'decode'
    assert report['disclaimer'] == DISCLAIMER
    assert report['decode'] is not None
    assert len(report['decode']['slot_influence']) == 2
    assert len(report['neighbors']) == 5
    assert report['embedding']['dim'] == 16
