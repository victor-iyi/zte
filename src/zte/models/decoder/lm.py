"""The frozen causal LM: prompt assembly, teacher-forced scoring, rescoring and free-running decode."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys

from zte.config import DecoderConfig
from zte.data.targets.tokens import (
    TINY_SOURCE,
    TinyByteTokenizer,
    fingerprint_tokenizer,
    load_tokenizer,
)
from zte.logging_utils import get_logger

_LOG = get_logger('models.decoder.lm')

# The offline test LM: 22,688 parameters, trains and generates with no network access.
_TINY_KWARGS: dict[str, Any] = {
    'vocab_size': 64,
    'hidden_size': 32,
    'intermediate_size': 64,
    'num_hidden_layers': 2,
    'num_attention_heads': 4,
    'num_key_value_heads': 2,
    'max_position_embeddings': 128,
    'bos_token_id': 1,
    'eos_token_id': 2,
    'pad_token_id': 0,
}


class FrozenLM(nn.Module):
    """A causal language model held frozen, driven only by a soft prompt.

    Nothing inside is trainable and nothing inside is checkpointed: `state_dict` is empty, every parameter has
    `requires_grad=False`, and `train()` leaves the module in eval. The empty state dict is a hard requirement -- the
    trainer writes `objective.state_dict()` into every epoch checkpoint, so a 0.5B LM would add roughly a gigabyte per
    epoch, three times over. Freezing is also the scientific claim: with no LM weight able to move, generated text
    cannot be corpus recall stored in the decoder.

    Attributes:
        lm (nn.Module): The wrapped causal LM.
        tokenizer (Any): The matching tokeniser, or `TinyByteTokenizer` for `'tiny'`.
        hidden_dim (int): Token-embedding width, which is the prefix width the bridge must produce.
        vocab_size (int): Token count.
        bos_id (int): Beginning-of-sequence id opening every prompt.
        eos_id (int): End-of-sequence id that stops free-running decode.
        pad_id (int): Padding id, excluded from the loss and from the attention mask.
    """

    lm: Any
    scaffold: torch.Tensor

    def __init__(
        self,
        source: str,
        revision: str | None = None,
        cache_dir: str | None = None,
        prompt_template: str = '\nSentence: ',
        tokenizer_source: str | None = None,
    ) -> None:
        """Loads and freezes the LM.

        Args:
            source (str): HuggingFace model id, or `'tiny'` to build the offline test LM locally.
            revision (str | None, optional): Pinned commit SHA. Defaults to None, which resolves to `main` and is
                logged as a reproducibility risk because HuggingFace repositories are mutable.
            cache_dir (str | None, optional): Local snapshot directory. Defaults to None.
            prompt_template (str, optional): Fixed scaffold between the prefix and the target. Defaults to
                '\\nSentence: '.
            tokenizer_source (str | None, optional): Tokeniser id. Defaults to None, which uses `source`; anything
                else must share the model's vocabulary or the ids address the wrong embeddings.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        super().__init__()
        self.source = source
        self.revision = revision
        self.prompt_template = prompt_template
        self.tokenizer_name = tokenizer_source or source
        self.lm = _build_causal_lm(source, revision, cache_dir)
        self.tokenizer = load_tokenizer(self.tokenizer_name, revision, cache_dir)

        config = self.lm.config
        self.hidden_dim = int(config.hidden_size)
        self.vocab_size = int(config.vocab_size)
        self.bos_id = int(_first_id(config.bos_token_id, 1))
        self.eos_id = int(_first_id(config.eos_token_id, 2))
        self.pad_id = int(_first_id(config.pad_token_id, self.eos_id))
        self.tokenizer_fingerprint = fingerprint_tokenizer(
            self.tokenizer, self.tokenizer_name, revision
        )
        self.register_buffer(
            'scaffold',
            torch.as_tensor(self._encode(prompt_template), dtype=torch.long),
            persistent=False,
        )

        self.lm.requires_grad_(False)
        self.lm.eval()
        if revision is None and source != TINY_SOURCE:
            _LOG.warning(
                'decoder.lm_revision is unset for %s; HuggingFace repositories are mutable, so this run is not '
                'reproducible from the manifest alone.',
                source,
            )
        _LOG.info(
            'Frozen LM %s: hidden %d, vocab %d, %d parameters (all frozen).',
            source,
            self.hidden_dim,
            self.vocab_size,
            sum(p.numel() for p in self.lm.parameters()),
        )

    # ---- Frozen-module contract ---- #

    def train(self, mode: bool = True) -> FrozenLM:
        """Keeps the module in eval whatever the trainer asks for, so decoding is deterministic.

        Args:
            mode (bool, optional): Ignored. Defaults to True.

        Returns:
            FrozenLM: `self`.
        """
        super().train(False)
        return self

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Returns an empty state dict, keeping the frozen weights out of every checkpoint.

        Returns:
            dict[str, Any]: An empty dict. The parent module's `destination` is deliberately left untouched.
        """
        return {}

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Ignores `state_dict`, since the weights come from the pinned `source`/`revision` and never from a run.

        Args:
            state_dict (Mapping[str, Any]): Ignored.
            strict (bool, optional): Ignored. Defaults to True.
            assign (bool, optional): Ignored. Defaults to False.

        Returns:
            Any: An empty `_IncompatibleKeys`.
        """
        return _IncompatibleKeys([], [])

    def provenance(self) -> dict[str, Any]:
        """Returns the identity of the frozen weights, for the run manifest.

        Returns:
            dict[str, Any]: Source, revision, sizes, parameter count and the tokeniser fingerprint.
        """
        return {
            'source': self.source,
            'revision': self.revision,
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_dim,
            'n_parameters': int(sum(p.numel() for p in self.lm.parameters())),
            'tokenizer': self.tokenizer_name,
            'tokenizer_fingerprint': self.tokenizer_fingerprint,
            'prompt_template': self.prompt_template,
        }

    # ---- Prompt assembly ---- #

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Embeds token ids with the frozen input embedding.

        Args:
            ids (torch.Tensor): Token ids `(..., seq_len)`.

        Returns:
            torch.Tensor: Embeddings `(..., seq_len, hidden_dim)`.
        """
        return self.lm.get_input_embeddings()(ids)

    def assemble(
        self,
        prefix: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        scaffold_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Builds `[bos, prefix, scaffold, target]` embeddings and the matching attention mask.

        Args:
            prefix (torch.Tensor): Soft prompt `(batch_size, slots, hidden_dim)`.
            target_ids (torch.Tensor | None, optional): Target token ids `(batch_size, n_target)`. Defaults to None,
                which assembles the prompt alone for generation.
            target_mask (torch.Tensor | None, optional): Boolean `(batch_size, n_target)`; `True` at real target
                tokens. Defaults to None, meaning every target token is real.
            scaffold_ids (torch.Tensor | None, optional): Scaffold ids `(n_scaffold,)`. Defaults to None, which uses
                `prompt_template`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, int]: `(inputs_embeds, attention_mask, target_start)` where
                `target_start` is the position of the first target token.
        """
        batch_size = prefix.shape[0]
        device = prefix.device
        scaffold = self.scaffold if scaffold_ids is None else scaffold_ids.to(device)
        bos = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        parts = [
            self.embed_tokens(bos),
            prefix,
            self.embed_tokens(scaffold.unsqueeze(0).expand(batch_size, -1)),
        ]
        start = int(1 + prefix.shape[1] + scaffold.shape[0])
        mask = torch.ones(batch_size, start, dtype=torch.long, device=device)
        if target_ids is not None:
            parts.append(self.embed_tokens(target_ids))
            real = (
                torch.ones_like(target_ids, dtype=torch.long)
                if target_mask is None
                else target_mask.to(torch.long)
            )
            mask = torch.cat([mask, real], dim=1)
        return torch.cat(parts, dim=1), mask, start

    # ---- Scoring ---- #

    def target_token_logprobs(
        self,
        prefix: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        scaffold_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the per-token log-probability of each target token under its own prefix.

        Differentiable: this is the primitive under both the cross-entropy loss and the in-batch grounding loss.

        Args:
            prefix (torch.Tensor): Soft prompt `(batch_size, slots, hidden_dim)`.
            target_ids (torch.Tensor): Target ids `(batch_size, n_target)`.
            target_mask (torch.Tensor): Boolean `(batch_size, n_target)`; `True` at real target tokens.
            scaffold_ids (torch.Tensor | None, optional): Scaffold ids. Defaults to None.

        Returns:
            torch.Tensor: `(batch_size, n_target)` log-probabilities, zeroed at padded positions.
        """
        embeds, mask, start = self.assemble(prefix, target_ids, target_mask, scaffold_ids)
        logits = self.lm(inputs_embeds=embeds, attention_mask=mask).logits
        n_target = target_ids.shape[1]
        pred = logits[:, start - 1 : start - 1 + n_target].float()
        logprob = F.log_softmax(pred, dim=-1)
        picked = logprob.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return picked * target_mask.to(picked.dtype)

    def forward_with_prefix(
        self,
        prefix: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        scaffold_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the token-mean teacher-forced cross-entropy of the targets under their prefixes.

        Args:
            prefix (torch.Tensor): Soft prompt `(batch_size, slots, hidden_dim)`.
            target_ids (torch.Tensor): Target ids `(batch_size, n_target)`.
            target_mask (torch.Tensor): Boolean `(batch_size, n_target)`; `True` at real target tokens.
            scaffold_ids (torch.Tensor | None, optional): Scaffold ids. Defaults to None.

        Returns:
            torch.Tensor: Scalar cross-entropy, 0 when no target token is real.
        """
        logprob = self.target_token_logprobs(prefix, target_ids, target_mask, scaffold_ids)
        n_tokens = target_mask.sum().clamp_min(1).to(logprob.dtype)
        return -logprob.sum() / n_tokens

    def candidate_logprobs(
        self,
        prefix: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_mask: torch.Tensor,
        length_normalise: bool = True,
        chunk: int = 64,
    ) -> torch.Tensor:
        """Scores several candidate sentences against each prefix.

        Differentiable, so the in-batch grounding loss can force a prefix to prefer its own reference; `sequence_logprob`
        is the no-grad wrapper used at evaluation time.

        Args:
            prefix (torch.Tensor): Soft prompts `(batch_size, slots, hidden_dim)`.
            cand_ids (torch.Tensor): Candidate ids, `(n_cand, n_target)` for a gallery shared by every row or
                `(batch_size, n_cand, n_target)` for per-row candidates.
            cand_mask (torch.Tensor): Boolean mask with the same shape as `cand_ids`.
            length_normalise (bool, optional): Divide by the candidate's token count, which stops the score being a
                sentence-length ranking. Defaults to True.
            chunk (int, optional): Rows per forward pass, bounding peak memory. Defaults to 64.

        Returns:
            torch.Tensor: `(batch_size, n_cand)` sequence log-probabilities.
        """
        batch_size, slots, hidden = prefix.shape
        if cand_ids.ndim == 2:
            cand_ids = cand_ids.unsqueeze(0).expand(batch_size, -1, -1)
            cand_mask = cand_mask.unsqueeze(0).expand(batch_size, -1, -1)
        n_cand, n_target = cand_ids.shape[1], cand_ids.shape[2]

        flat_prefix = prefix.unsqueeze(1).expand(-1, n_cand, -1, -1).reshape(-1, slots, hidden)
        flat_ids = cand_ids.reshape(-1, n_target)
        flat_mask = cand_mask.reshape(-1, n_target)

        scores: list[torch.Tensor] = []
        for lo in range(0, flat_ids.shape[0], max(chunk, 1)):
            hi = lo + max(chunk, 1)
            part = self.target_token_logprobs(flat_prefix[lo:hi], flat_ids[lo:hi], flat_mask[lo:hi])
            total = part.sum(dim=1)
            if length_normalise:
                total = total / flat_mask[lo:hi].sum(dim=1).clamp_min(1).to(total.dtype)
            scores.append(total)
        return torch.cat(scores).reshape(batch_size, n_cand)

    @torch.no_grad()
    def sequence_logprob(
        self,
        prefix: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_mask: torch.Tensor,
        length_normalise: bool = True,
        chunk: int = 64,
    ) -> torch.Tensor:
        """No-grad `candidate_logprobs`, the scoring path behind decoder-rescoring retrieval.

        Args:
            prefix (torch.Tensor): Soft prompts `(batch_size, slots, hidden_dim)`.
            cand_ids (torch.Tensor): Candidate ids `(n_cand, n_target)` or `(batch_size, n_cand, n_target)`.
            cand_mask (torch.Tensor): Boolean mask with the same shape as `cand_ids`.
            length_normalise (bool, optional): Divide by the candidate's token count. Defaults to True.
            chunk (int, optional): Rows per forward pass. Defaults to 64.

        Returns:
            torch.Tensor: `(batch_size, n_cand)` sequence log-probabilities.
        """
        return self.candidate_logprobs(prefix, cand_ids, cand_mask, length_normalise, chunk)

    @torch.no_grad()
    def next_token_logits(
        self, prefix: torch.Tensor, scaffold_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Returns the logits for the first generated token.

        Args:
            prefix (torch.Tensor): Soft prompts `(batch_size, slots, hidden_dim)`.
            scaffold_ids (torch.Tensor | None, optional): Scaffold ids. Defaults to None.

        Returns:
            torch.Tensor: `(batch_size, vocab_size)` logits.
        """
        embeds, mask, _ = self.assemble(prefix, scaffold_ids=scaffold_ids)
        return self.lm(inputs_embeds=embeds, attention_mask=mask).logits[:, -1].float()

    @torch.no_grad()
    def next_token_kl(
        self,
        prefix: torch.Tensor,
        other: torch.Tensor,
        scaffold_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the per-row KL in nats between two prefixes' next-token distributions.

        Against another row's prefix this is the collapse detector: a bridge whose output does not depend on its
        conditioning vector emits one prompt for every reading and scores exactly 0 here whatever its loss curve looks
        like, which is the most likely training failure of a 227k-parameter head driven by a conditioning vector worth a
        handful of bits. Against the null prefix it measures something else entirely -- the distance from the learned
        unconditional prompt, which is a free parameter and stays nonzero under exactly that collapse.

        Args:
            prefix (torch.Tensor): Soft prompts `(batch_size, slots, hidden_dim)`.
            other (torch.Tensor): Comparison prompts of the same shape.
            scaffold_ids (torch.Tensor | None, optional): Scaffold ids. Defaults to None.

        Returns:
            torch.Tensor: `(batch_size,)` KL divergences `KL(p_prefix || p_other)`.
        """
        p = F.log_softmax(self.next_token_logits(prefix, scaffold_ids), dim=-1)
        q = F.log_softmax(self.next_token_logits(other, scaffold_ids), dim=-1)
        return (p.exp() * (p - q)).sum(dim=-1)

    # ---- Free-running decode ---- #

    @torch.no_grad()
    def generate_from_prefix(
        self,
        prefix: torch.Tensor,
        scaffold_ids: torch.Tensor | None = None,
        max_new_tokens: int = 96,
        beams: int = 1,
    ) -> list[str]:
        """Decodes free-running text from a soft prompt, with no reference and no candidate set.

        Args:
            prefix (torch.Tensor): Soft prompts `(batch_size, slots, hidden_dim)`.
            scaffold_ids (torch.Tensor | None, optional): Scaffold ids. Defaults to None.
            max_new_tokens (int, optional): Decode cap; the reference length is never supplied. Defaults to 96.
            beams (int, optional): Beam width; 1 is greedy and deterministic. Defaults to 1.

        Returns:
            list[str]: One detokenised hypothesis per row.
        """
        embeds, mask, _ = self.assemble(prefix, scaffold_ids=scaffold_ids)
        out = self.lm.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            num_beams=max(beams, 1),
            do_sample=False,
            bos_token_id=self.bos_id,
            eos_token_id=self.eos_id,
            pad_token_id=self.pad_id,
        )
        # Generating from `inputs_embeds` returns only the new ids, so there is no prompt to strip.
        return [self.decode(row) for row in out]

    def decode(self, ids: torch.Tensor | Sequence[int]) -> str:
        """Detokenises generated ids, dropping the specials.

        Args:
            ids (torch.Tensor | Sequence[int]): Token ids.

        Returns:
            str: The decoded string.
        """
        row = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        if isinstance(self.tokenizer, TinyByteTokenizer):
            return self.tokenizer.decode(row)
        return str(self.tokenizer.decode(row, skip_special_tokens=True))

    def _encode(self, text: str) -> list[int]:
        """Encodes `text` without special tokens, for the fixed scaffold."""
        if isinstance(self.tokenizer, TinyByteTokenizer):
            return self.tokenizer.encode(text, add_eos=False)
        return [int(i) for i in self.tokenizer(text, add_special_tokens=False)['input_ids']]


def _first_id(value: Any, fallback: int) -> int:
    """Returns a single token id from a config field that may be a list, or `fallback` when it is unset."""
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else fallback
    return fallback if value is None else int(value)


def _build_causal_lm(source: str, revision: str | None, cache_dir: str | None) -> nn.Module:
    """Builds the causal LM, constructing the offline test model locally for `'tiny'`.

    Args:
        source (str): HuggingFace model id, or `'tiny'`.
        revision (str | None): Pinned commit SHA.
        cache_dir (str | None): Local snapshot directory.

    Returns:
        nn.Module: The loaded model.

    Raises:
        RuntimeError: If `transformers` is missing or the weights cannot be resolved.
    """
    if source == TINY_SOURCE:
        try:
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                'The prefix decoder needs `transformers`; install the `decoder` dependency group.'
            ) from exc
        return LlamaForCausalLM(LlamaConfig(**_TINY_KWARGS))
    try:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(source, revision=revision, cache_dir=cache_dir)
    except (ImportError, OSError, ValueError) as exc:
        raise RuntimeError(
            f'Could not load the frozen decoder LM {source!r} (revision={revision!r}): {exc!r}. '
            "Install `transformers` and pre-download the weights, or use lm_source='tiny' for offline runs."
        ) from exc


def build_lm(config: DecoderConfig) -> FrozenLM:
    """Constructs the frozen LM named by a decoder configuration.

    Args:
        config (DecoderConfig): Decoder configuration (uses `lm_source`, `lm_revision`, `lm_cache_dir`,
            `prompt_template`).

    Returns:
        FrozenLM: The frozen LM.
    """
    return FrozenLM(
        config.lm_source,
        revision=config.lm_revision,
        cache_dir=config.lm_cache_dir,
        prompt_template=config.prompt_template,
        tokenizer_source=config.tokenizer_source,
    )
