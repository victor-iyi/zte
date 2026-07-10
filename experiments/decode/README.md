# EEG → language decode experiments

Configs for the `zte.decode` stack (EEG-OT-CLIP alignment + retrieval / prefix-LM).

## Quick synthetic smoke

```bash
zte-decode-run --synthetic --epochs 3 --align-epochs 5 --backend hash --out res/decode/demo
```

## Alignment

```bash
# LOSO sentence-level OT-CLIP (needs a trained ZTE ckpt + ZuCo bundle/root)
zte-align --ckpt res/checkpoints/best.pt --bundle res/bundle \
  --config experiments/decode/align_otclip_loso.yaml \
  --out res/decode/alignment

# Word-level (hash text encoder — no transformers download)
zte-align --ckpt res/checkpoints/best.pt --bundle res/bundle \
  --config experiments/decode/align_otclip_word.yaml --backend hash
```

## Decode

```bash
zte-decode --align-ckpt res/decode/alignment/<run>/best.pt \
  --ckpt res/checkpoints/best.pt --bundle res/bundle \
  --config experiments/decode/decode_retrieval.yaml \
  --mode retrieval --out res/decode/predictions

zte-decode --align-ckpt ... --ckpt ... --bundle ... \
  --config experiments/decode/decode_prefix_lm.yaml \
  --mode prefix_lm --backend toy --out res/decode/generative
```

## Evaluate

```bash
zte-decode-eval --align-ckpt res/decode/alignment/<run>/best.pt \
  --ckpt res/checkpoints/best.pt --bundle res/bundle \
  --backend hash --out res/decode/eval
```

See [`docs/DECODING.md`](../../docs/DECODING.md) for architecture, losses, and the noise-anchored LOSO protocol.

**Optional deps:** `uv sync --group decode` installs `transformers` / `accelerate` / `safetensors`. The hash / toy backends work without that group.
