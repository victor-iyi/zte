# The lens — one reading, walked through the model

The lens is a hands-on inspection surface: pick **one reading** — one subject reading one sentence — run it through a trained checkpoint, and see what the model did with it. The thought embedding, which words and which scalp regions drove it, its neighborhood in the sentence gallery, and, for decoder checkpoints, the generated text with a trace of what parts of the input influenced it. It exists so that a number in a report can be chased down to a single concrete example, and so that a surprising behaviour can be looked at rather than argued about.

It is an inspection tool, **never an evaluation**. Every lens artifact carries, and every lens page renders, the disclaimer verbatim:

> inspection, not a result -- no number here is a headline

No lens output is phrased as a metric. The honest numbers live in the scoreboard, rebaseline and parallax reports; the lens is the magnifying glass held up beside them.

Code: `zte.lens` (`saliency`, `trace`, `page`). CLI: `zte-lens`. Notebook: §8 of `notebooks/zte_parallax.ipynb`. This document owns the definitions of what the lens computes and the contract every artifact obeys.

## Picking the reading

A reading is addressed as the `--index`-th sentence reading of `--subject` in the built dataset's deterministic order, optionally filtered to sentences whose text contains `--contains`. When `--subject` is omitted it defaults to the checkpoint's own LOSO holdout, read from the run config — the one brain the model's parameters never saw.

That default is deliberate. `reading.is_holdout` is computed against the checkpoint's `train.loso_holdout_subject` and rendered prominently: a non-holdout reading is a **training brain**, and whatever the lens shows there is memory as much as reach. Inspecting a training brain is legitimate — seeing what the model memorised is informative — but the page must and does say which kind of brain it is looking at.

## What the lens computes

All saliency in the lens is **occlusion-based**: remove a piece of the input, run the unchanged model again, and measure how far the output moved. No gradients, no attention rollout, no method-specific attribution — occlusion is model-agnostic and honest to the model's actual forward pass, at the price of measuring marginal influence rather than causal structure (correlated inputs share credit unpredictably).

### Word saliency

For a reading with word positions $1 \ldots n$ and full-sentence embedding $z$, position $i$ is occluded by masking it out of the pad mask and re-embedding, giving $z_{\setminus i}$. The score is the cosine drop against the full-sentence embedding:

$$
s_i \;=\; 1 - \cos(z,\, z_{\setminus i})
$$

A large $s_i$ means the embedding depended on that word's EEG epoch; a near-zero $s_i$ means the model would have produced essentially the same thought vector without it. Scores are comparable **within one reading only** — across readings or across checkpoints they are not on a common scale, which is one of the reasons they are never a metric.

### Channel saliency

The same occlusion, over space instead of time: electrode groups are zeroed and the cosine drop recorded, grouped by montage region (roughly ten groups). Per-channel scores come from a second occlusion pass over the winning regions, or from full per-channel occlusion when the channel count permits it. The artifact carries the electrode labels, their region assignments, and both 2-d and 3-d coordinates so the page can draw an honest scalp map.

When the checkpoint was trained without a montage, `channel_saliency` is `null` and the page drops the scalp panel with a note saying why — it does not invent a layout.

### The neighborhood

The reading's embedding is scored by cosine against the gallery of embedded readings and the top-k neighbors are listed, each with its text, cosine, subject, and an `is_true_sentence` flag. Two rules keep this honest, and both are mutation-tested:

- **The query reading itself never appears.** A gallery containing the query would put a cosine of 1.0 at rank one by construction and the panel would flatter every checkpoint equally.
- **Everything else does.** Other subjects' readings of the *same* sentence appear, flagged `is_true_sentence` — the true sentence surfacing through a stranger's brain is the interesting event. The same subject's readings of *other* sentences appear too, flagged by subject — a neighborhood dominated by same-subject readings is the signature of an entangled space, and the lens must be able to show it.

The neighborhood is geometry, not accuracy: the absence of the true sentence among the neighbors is not a score, and its presence is not a hit rate.

### The decode trace

For a checkpoint with a decoder, `zte-lens decode` adds the generation view; for one without, it refuses loudly, naming the reason. The trace has four parts:

- **The greedy generation** itself, token by token.
- **Per-slot occlusion of the prefix**: one prefix slot is zeroed and the already-generated tokens are *re-scored* teacher-forced under the occluded prefix — the slot's influence is the mean absolute shift in the generated tokens' log-probabilities. Nothing is regenerated; the trace measures how much each slot propped up the text the full prefix produced.
- **Word-synchronous evidence**, when the checkpoint uses the WordEvidence or MonotonicPointer mechanisms (`zte.models.decoder.evidence`): the (token, word, weight) triples those mechanisms expose. One thing must be read into every ribbon: the pointer walk is a *fixed monotonic schedule* — a function of step count and word count, never of content, precisely so that brain-free controls inherit it. The diagonal band is the schedule, not a discovered alignment. Absent from checkpoints without those mechanisms, and the page says so rather than approximating.
- **The null-prefix control, side by side.** The same generation under the bridge's *learned unconditional (null) prefix* — the LM with no brain attached. Not literal zeros: a zeroed prefix is off-manifold and would make the control look artificially degenerate, flattering the real generation. This is the single most important panel on a decode page: fluent text that the null prefix also produces came from the language model, not from the EEG. It is rendered beside the real generation, always.

## What the lens is not

- **Not a metric.** The `disclaimer` field is mandatory in every `lens.json` and rendered on every page. No lens number is a headline, appears in `docs/RESULTS.md`, or is quoted in a claim. The powered readouts and their statistics live in the scoreboard (`zte-audit`, `held_out_retrieval`), the length audit (`zte-rebaseline`) and the parallax report — the lens cites them and replaces none of them.
- **Not an average.** One reading is an anecdote by construction. The lens makes the anecdote inspectable; it does not aggregate, and adding aggregation to it would turn it into a worse copy of the scoreboard.
- **Not causal.** Occlusion measures what moved the output of this model on this input. It does not establish what the brain encoded, and correlated words or regions share credit unpredictably.

## Operations

```sh
# One reading through an encoder checkpoint, with the HTML page beside the JSON.
uv run zte-lens encode --ckpt res/experiments/<run>/checkpoints/best.pt \
    --root res/data/zuco_extracted --subject ZAB --index 0 --out res/analysis/lens --html

# The same reading through a decoder checkpoint; refuses a checkpoint with no decoder.
uv run zte-lens decode --ckpt <decoder-ckpt> --root <data> --contains "movie" --out res/analysis/lens --html
```

Both subcommands accept `--root`, `--bundle` or `--synthetic` as the data source, `--subject` / `--index` / `--contains` to pick the reading, `--top-k` for the neighborhood size, and `--device`. `decode` adds `--max-new-tokens`. Output lands in `<out>/<run_name>_<subject>_<index>/lens.json`, with `LENS.html` beside it when `--html` is passed.

`lens.json` is one self-describing object: `mode`, the `reading` (subject, task, text, words, `is_holdout`), the `embedding` summary, `word_saliency`, `channel_saliency` (or `null`), the `neighbors` list, `decode` (or `null` for encode mode), the mandatory `disclaimer`, and `provenance` (checkpoint path and sha256, run name, git commit, the run's holdout). The page is built by `build_lens_page` in `zte.lens.page` — one `lens.json` in, one self-contained HTML file out, no server and no network. It degrades honestly: an encode-mode artifact renders without the decode panel, and a `null` `channel_saliency` drops the scalp panel with a note.

On Colab, §8 of `notebooks/zte_parallax.ipynb` drives both subcommands and writes every artifact to Drive under the session's `analysis/lens/`, staging only the IFrame copy on the VM disk.

## How to read a lens page honestly

- **Saliency is occlusion, not causality.** A high-scoring word or region is one whose removal moved this model's embedding of this reading — a statement about the model, not about the brain.
- **Neighbors are geometry, not accuracy.** The true sentence surfacing through another subject's reading is worth noticing; its absence is not a failure score, and neither observation generalises beyond this reading.
- **The decode trace is inspection.** Read the generation against its null-prefix control before finding any of it impressive, and remember the bit budget: the encoder supplies a few bits of sentence identity, free generation needs hundreds, and the honest generation verdict lives behind the controls in `docs/DECODER.md`, not on a lens page.
- **One reading proves nothing.** The lens shows what happened, on one input, once. When it makes something look interesting, the next step is a pre-registered measurement in the reports — never a screenshot of the page.
