"""Fast unit tests for the EEG→language decode stack (hash / toy backends only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zte.decode.alignment import OTCLIPAligner, info_nce_loss, sinkhorn_ot_loss
from zte.decode.config import AlignConfig, DecoderConfig, GenerativeConfig, TextEncoderConfig
from zte.decode.decoders import PrefixLanguageDecoder, RetrievalDecoder
from zte.decode.evaluate import evaluate_decoding
from zte.decode.metrics import bleu_score, exact_match, token_f1, wer_score
from zte.decode.text_encoder import HashTextEncoder, build_text_encoder
from zte.decode.train_align import run_alignment_from_embeddings


def test_hash_text_encoder_deterministic() -> None:
    """Identical strings map to identical vectors; different strings differ."""
    enc = HashTextEncoder(TextEncoderConfig(embed_dim=32, backend='hash', model_name='hash'))
    a = enc.embed_texts(['hello world', 'hello world'])
    b = enc.embed_texts(['hello world', 'goodbye'])
    assert a.shape == (2, 32)
    assert torch.allclose(a[0], a[1])
    assert not torch.allclose(b[0], b[1])
    # build_text_encoder respects backend='hash'
    enc2 = build_text_encoder(TextEncoderConfig(embed_dim=32, backend='hash', model_name='hash'))
    assert torch.allclose(enc.embed_texts(['x']), enc2.embed_texts(['x']))


def test_info_nce_paired_better_than_shuffled() -> None:
    """Paired (eeg≈text) batches yield lower InfoNCE than a shuffled pairing."""
    rng = np.random.default_rng(0)
    n, d = 32, 16
    text = rng.standard_normal((n, d)).astype(np.float32)
    eeg = text + 0.05 * rng.standard_normal((n, d)).astype(np.float32)
    eeg_t = torch.nn.functional.normalize(torch.from_numpy(eeg), dim=-1)
    text_t = torch.nn.functional.normalize(torch.from_numpy(text), dim=-1)
    loss_paired = info_nce_loss(eeg_t, text_t, temperature=0.07)
    perm = torch.randperm(n)
    loss_shuffled = info_nce_loss(eeg_t, text_t[perm], temperature=0.07)
    assert float(loss_paired) < float(loss_shuffled)


def test_sinkhorn_ot_finite() -> None:
    """Sinkhorn OT returns a finite non-negative scalar."""
    rng = np.random.default_rng(1)
    a = torch.nn.functional.normalize(torch.from_numpy(rng.standard_normal((8, 12)).astype(np.float32)))
    b = torch.nn.functional.normalize(torch.from_numpy(rng.standard_normal((8, 12)).astype(np.float32)))
    loss = sinkhorn_ot_loss(a, b, epsilon=0.05, n_iters=10)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_alignment_trains_one_epoch(tmp_path: Path) -> None:
    """``run_alignment_from_embeddings`` completes one tiny epoch and saves a ckpt."""
    rng = np.random.default_rng(2)
    n, d = 24, 16
    text = rng.standard_normal((n, d)).astype(np.float32)
    eeg = text + 0.1 * rng.standard_normal((n, d)).astype(np.float32)
    cfg = AlignConfig(
        eeg_dim=d,
        text_dim=d,
        proj_dim=d,
        proj_hidden=32,
        epochs=1,
        batch_size=8,
        lr=1e-2,
        device='cpu',
        ckpt_dir=str(tmp_path / 'align'),
        run_name='test',
        val_fraction=0.25,
        eval_every=1,
        log_every=100,
    )
    split = 18
    arts = run_alignment_from_embeddings(
        eeg[:split],
        text[:split],
        eeg[split:],
        text[split:],
        config=cfg,
        out_dir=tmp_path / 'align',
        texts_train=[f't{i}' for i in range(split)],
        texts_val=[f't{i}' for i in range(split, n)],
    )
    assert arts.aligner_path.is_file()
    assert len(arts.history) >= 1
    restored = OTCLIPAligner.from_checkpoint(str(arts.aligner_path))
    assert restored.config.eeg_dim == d


def test_retrieval_decoder_exact_on_identical() -> None:
    """When query == bank row, top-1 retrieval recovers the exact string."""
    rng = np.random.default_rng(3)
    emb = rng.standard_normal((10, 8)).astype(np.float32)
    texts = [f'sentence-{i}' for i in range(10)]
    dec = RetrievalDecoder(emb, texts, aligner=None, bank_already_aligned=True)
    hyps = dec.decode(emb, k=1)
    assert hyps == texts


def test_evaluate_decoding_writes_artifacts(tmp_path: Path) -> None:
    """``evaluate_decoding`` writes metrics.json, report.md, predictions.csv, figures."""
    rng = np.random.default_rng(4)
    n, d = 12, 8
    text = rng.standard_normal((n, d)).astype(np.float32)
    eeg = text + 0.01 * rng.standard_normal((n, d)).astype(np.float32)
    texts = [f'ref {i}' for i in range(n)]
    meta = pd.DataFrame({'subject': ['A'] * 6 + ['B'] * 6, 'task': ['SR'] * n})
    out = tmp_path / 'eval'
    metrics = evaluate_decoding(
        eeg_emb=eeg,
        text_emb=text,
        texts=texts,
        meta=meta,
        aligner=None,
        decoder=None,
        out_dir=out,
        run_name='unit',
    )
    assert (out / 'metrics.json').is_file()
    assert (out / 'report.md').is_file()
    assert (out / 'predictions.csv').is_file()
    assert (out / 'figures' / 'retrieval_curve.png').is_file()
    assert 'verdict' in metrics
    assert 'retrieval_above_chance' in metrics['verdict']


def test_bleu_and_wer_smoke() -> None:
    """BLEU / WER / exact_match / token_f1 return sane values on trivial pairs."""
    refs = ['the cat sat on the mat', 'hello world']
    hyps = ['the cat sat on the mat', 'hello there']
    assert exact_match(hyps, refs) == 0.5
    assert 0.0 < bleu_score(hyps, refs) <= 1.0
    assert 0.0 < token_f1(hyps, refs) <= 1.0
    assert 0.0 <= wer_score(hyps, refs) < 1.0
    assert bleu_score(refs, refs) == 1.0 or bleu_score(refs, refs) > 0.99


def test_prefix_toy_decoder_forward() -> None:
    """Toy PrefixLanguageDecoder forward returns finite loss and correct logit shape."""
    texts = ['abc', 'xyz', 'hello']
    cfg = GenerativeConfig(
        backend='toy',
        prefix_dim=16,
        prefix_len=4,
        max_new_tokens=8,
        device='cpu',
        epochs=1,
    )
    dec = PrefixLanguageDecoder(cfg, texts=texts)
    eeg = torch.randn(2, 16)
    ids, mask = dec.encode_texts(texts[:2], max_length=16)
    loss, logits = dec(eeg, ids, mask)
    assert torch.isfinite(loss)
    assert logits.shape[0] == 2
    assert logits.shape[-1] == dec.vocab_size
    gens = dec.generate(eeg.numpy(), max_new_tokens=4)
    assert len(gens) == 2
    assert isinstance(gens[0], str)


def test_decoder_config_roundtrip(tmp_path: Path) -> None:
    """DecoderConfig YAML round-trip preserves mode and dims."""
    cfg = DecoderConfig()
    cfg.mode = 'retrieval'
    cfg.align.eeg_dim = 64
    path = tmp_path / 'cfg.yaml'
    cfg.to_yaml(path)
    loaded = DecoderConfig.from_yaml(path)
    assert loaded.mode == 'retrieval'
    assert loaded.align.eeg_dim == 64
