"""Greedy decode tracing for one reading: what the decoder wrote, and which prefix slots and words drove it."""

from typing import TYPE_CHECKING, Any, Final

import torch

from zte.data.dataset import ZuCoDataset
from zte.lens.saliency import Reading

if TYPE_CHECKING:
    from zte.inference.decode import ReadingBatch, ZTEDecoder

# Pointer weights below this are numerical dust; keeping them would triple the JSON for no visible arc.
_EVIDENCE_EPS: Final[float] = 1e-3
"""Smallest pointer weight kept in the `word_evidence` triples."""


def decode_trace(
    decoder: ZTEDecoder,
    dataset: ZuCoDataset,
    reading: Reading,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Decodes one reading greedily and records what drove each part of the generation.

    The trace covers: the generated text and its token pieces; per-prefix-slot occlusion influence (slot zeroed, the
    generated tokens re-scored, mean absolute token log-probability divergence); the word-synchronous pointer weights
    when the checkpoint uses the evidence path; and the null-prefix generation from the trained unconditional branch,
    decoded side by side so a brain-independent output is visible for what it is.

    Args:
        decoder (ZTEDecoder): A loaded decoder checkpoint.
        dataset (ZuCoDataset): The built dataset the reading lives in.
        reading (Reading): The reading to decode.
        max_new_tokens (int, optional): Decode cap. Defaults to None, which uses the configured value.

    Returns:
        dict[str, Any]: `{'generated', 'tokens', 'slot_influence', 'word_evidence', 'null_prefix_generated',
            'method'}` per the `lens.json` decode contract.

    Raises:
        ValueError: If the reading's rows do not select exactly one sentence.
    """
    readings = decoder.conditioning(dataset, indices=reading.row_indices)
    if len(readings) != 1:
        raise ValueError(f'Expected the rows of one reading, got {len(readings)} sentences.')

    steps = max_new_tokens or decoder.decoder_config.max_new_tokens
    record = decoder.decode_trace(readings, max_new_tokens=steps)[0]
    generated = str(record['hypothesis'])
    tokens = [str(step['piece']) for step in record['steps']]

    return {
        'generated': generated,
        'tokens': tokens,
        'slot_influence': _slot_influence(decoder, readings, generated),
        'word_evidence': _word_evidence(record['pointer'], n_steps=len(tokens)),
        'null_prefix_generated': decoder.generate_from_prefix(decoder.null_prefix(1), max_new_tokens=steps)[0],
        'method': 'slot_occlusion_token_logprob_divergence',
    }


@torch.no_grad()
def _slot_influence(decoder: ZTEDecoder, readings: ReadingBatch, generated: str) -> list[float]:
    """Mean |delta log p| over the generated tokens when each prefix slot is zeroed in turn."""
    prefix = decoder.prefix_from_z(readings.z)
    slots = int(prefix.shape[1])
    if not generated:
        return [0.0] * slots

    ids, mask = decoder.tokenise([generated])
    real = mask.to(torch.bool)
    if not bool(real.any()):
        return [0.0] * slots

    evidence = decoder.evidence_fn(readings, 0, 1)
    base = decoder.lm.target_token_logprobs(prefix, ids, mask, evidence=evidence)

    out: list[float] = []
    for slot in range(slots):
        occluded = prefix.clone()
        occluded[:, slot, :] = 0.0
        scored = decoder.lm.target_token_logprobs(occluded, ids, mask, evidence=evidence)
        out.append(float((base - scored)[real].abs().float().mean()))

    return out


def _word_evidence(pointer: list[list[float]] | None, n_steps: int) -> list[list[float]] | None:
    """Flattens the pointer's walk into `[token_idx, word_idx, weight]` triples, or `None` without an evidence path."""
    if pointer is None:
        return None

    triples: list[list[float]] = []
    for t, row in enumerate(pointer[:n_steps]):
        for w, weight in enumerate(row):
            if weight > _EVIDENCE_EPS:
                triples.append([t, w, round(float(weight), 6)])

    return triples
