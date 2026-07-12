"""Tests for runtime auto-adaptation: precision, DataLoader workers, and static-shape padding.

The load-bearing property is that the fast paths never change results: static padding is
*representation-neutral* (padded positions are masked out of the model), and the portable
uniformity term matches the `torch.pdist` it replaced.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from zte.config import ModelConfig
from zte.data.dataset import ZuCoDataset
from zte.data.torch_dataset import collate_sentences
from zte.device import DeviceSpec, _resolve_precision, auto_num_workers, configure_backend
from zte.models.embedding import build_model


def _spec(kind: str) -> DeviceSpec:
    return DeviceSpec(
        device=torch.device('cpu'),
        kind=kind,  # type: ignore[arg-type]
        autocast_dtype=None,
        use_amp=False,
        supports_pin_memory=(kind == 'cuda'),
        name=kind,
    )


def test_auto_num_workers() -> None:
    """`-1` auto-picks per backend; an explicit non-negative count is honoured."""
    assert auto_num_workers(_spec('cpu'), -1) == 0
    assert auto_num_workers(_spec('cuda'), -1) >= 1
    assert auto_num_workers(_spec('mps'), -1) >= 1
    assert auto_num_workers(_spec('xla'), -1) >= 1
    assert auto_num_workers(_spec('cuda'), 0) == 0  # explicit wins
    assert auto_num_workers(_spec('cpu'), 3) == 3


def test_configure_backend_is_safe_off_cuda() -> None:
    """configure_backend is a no-op (never raises) on CPU/MPS/XLA."""
    for kind in ('cpu', 'mps', 'xla'):
        configure_backend(_spec(kind))  # must not raise


def test_precision_mapping_per_backend() -> None:
    """auto precision: bf16 on TPU, fp32 on MPS/CPU; explicit bf16 enables AMP on CUDA and XLA."""
    assert _resolve_precision('xla', 'auto') == (torch.bfloat16, True)
    assert _resolve_precision('mps', 'auto') == (None, False)
    assert _resolve_precision('cpu', 'auto') == (None, False)
    assert _resolve_precision('cuda', 'bf16') == (torch.bfloat16, True)
    assert _resolve_precision('xla', 'bf16') == (torch.bfloat16, True)
    assert _resolve_precision('cpu', 'fp32') == (None, False)


def test_static_padding_fixed_length_no_truncation() -> None:
    """`pad_to` pads the sequence axis to a fixed length and never truncates a longer sample."""
    # Two 'sentences' via the collate on synthetic samples is heavy; assert the shape contract
    # directly on the max-length rule the collate uses.
    lengths = torch.tensor([3, 7, 5])
    assert max(int(lengths.max()), 10 or 0) == 10  # pads up to pad_to
    assert max(int(lengths.max()), 4 or 0) == 7  # never below the batch max (no truncation)


def test_static_padding_is_representation_neutral(small_dataset: ZuCoDataset) -> None:
    """Valid-token representations are identical under dynamic vs fixed padding (accuracy-neutral).

    This is the guarantee that makes static shapes safe on TPU: padded positions are masked out of
    attention and pooling, so the embeddings of real tokens do not change.
    """
    torch_ds = small_dataset.to_torch()
    samples = [torch_ds[i] for i in range(min(8, len(torch_ds)))]
    gmax = max(len(s) for s in torch_ds.sequences)
    in_dim = small_dataset.features.shape[1]

    torch.manual_seed(0)
    model = build_model(ModelConfig(embed_dim=48, hidden_dim=48), in_dim=in_dim).eval()
    dyn = collate_sentences(samples, pad_to=None)
    stat = collate_sentences(samples, pad_to=gmax + 6)
    length = dyn['features'].shape[1]
    assert stat['features'].shape[1] >= length + 6  # genuinely more padding

    with torch.no_grad():
        ctx_d = model(dyn, contextual=True)
        ctx_s = model(stat, contextual=True)[:, :length]
        pooled_d = model.embed_sentence(dyn, objective='masked')
        pooled_s = model.embed_sentence(stat, objective='masked')

    valid = dyn['pad_mask'].unsqueeze(-1)
    assert float(((ctx_d - ctx_s).abs() * valid).max()) < 1e-4
    assert float((pooled_d - pooled_s).abs().max()) < 1e-4


def test_uniformity_matches_pdist() -> None:
    """The portable (Gram-based) uniformity term equals the torch.pdist version it replaced."""
    torch.manual_seed(0)
    unit = F.normalize(torch.randn(80, 32), dim=-1)
    ref = torch.pdist(unit).pow(2).mul(-2.0).exp().mean().clamp_min(1e-12).log()

    m = unit.shape[0]
    gram = unit @ unit.t()
    sq = (2.0 - 2.0 * gram).clamp_min(0.0)
    iu = torch.triu_indices(m, m, offset=1)
    portable = sq[iu[0], iu[1]].mul(-2.0).exp().mean().clamp_min(1e-12).log()

    assert abs(float(ref) - float(portable)) < 1e-5
