"""The trainable surface of the prefix decoder: the pooled-vector bridge and the word resampler over it."""

from __future__ import annotations

import torch
from torch import nn

from zte.config import DecoderConfig
from zte.logging_utils import get_logger

_LOG = get_logger('models.decoder.bridge')


class _ResidualBlock(nn.Module):
    """One pre-norm residual MLP block inside the bridge's bottleneck."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the block."""
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(x))))


class PrefixBridge(nn.Module):
    """Maps one sentence vector to a soft prompt of `slots` frozen-LM embeddings.

    The whole map is rank-limited: the sentence vector is compressed to `bottleneck` once and every slot is a learned
    FiLM view of that single code, so slot count buys prompt length rather than capacity. At 768/896/8/128 this is
    226,560 parameters against roughly 120k supervised target tokens -- the smallest surface that can still write, and
    deliberately far too small to memorise a 700-sentence corpus.

    Attributes:
        slots (int): Number of prefix positions produced.
        lm_dim (int): Frozen-LM embedding width.
        null_prefix (nn.Parameter): The `(slots, lm_dim)` unconditional prefix used by null-prefix dropout, the
            `null_prefix` control and the prefix-influence diagnostic.
    """

    def __init__(self, z_dim: int, lm_dim: int, slots: int = 8, bottleneck: int = 128, depth: int = 1) -> None:
        """Builds the bridge.

        Args:
            z_dim (int): Width of the conditioning vector.
            lm_dim (int): Frozen-LM embedding width.
            slots (int, optional): Prefix positions produced. Defaults to 8.
            bottleneck (int, optional): Rank of the shared low-rank map. Defaults to 128.
            depth (int, optional): Residual blocks inside the bottleneck; `1` is the plain linear map. Defaults to 1.
        """
        super().__init__()
        self.slots = slots
        self.lm_dim = lm_dim
        self.norm_in = nn.LayerNorm(z_dim)
        self.to_bottleneck = nn.Linear(z_dim, bottleneck)
        # Zero-initialised second layers make every extra block the identity at step 0, so raising `depth` never
        # changes where a run starts -- only where it can get to.
        self.blocks = nn.ModuleList(_ResidualBlock(bottleneck) for _ in range(max(depth - 1, 0)))
        self.film_gamma = nn.Parameter(torch.empty(slots, bottleneck))
        self.film_beta = nn.Parameter(torch.empty(slots, bottleneck))
        self.to_lm = nn.Linear(bottleneck, lm_dim)
        self.norm_out = nn.LayerNorm(lm_dim)
        self.null_prefix = nn.Parameter(torch.empty(slots, lm_dim))

        # Zero-initialised FiLM would make every slot the same vector and leave the prompt one position wide.
        nn.init.trunc_normal_(self.film_gamma, std=0.02)
        nn.init.trunc_normal_(self.film_beta, std=0.02)
        nn.init.trunc_normal_(self.null_prefix, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Produces the soft prompt for a batch of conditioning vectors.

        Args:
            z (torch.Tensor): Conditioning vectors `(batch_size, z_dim)`.

        Returns:
            torch.Tensor: Prefix embeddings `(batch_size, slots, lm_dim)`.
        """
        u = self.to_bottleneck(self.norm_in(z))  # (batch_size, bottleneck)
        for block in self.blocks:
            u = block(u)
        u = u.unsqueeze(1) * (1.0 + self.film_gamma) + self.film_beta
        return self.norm_out(self.to_lm(u))  # (batch_size, slots, lm_dim)

    def null(self, batch_size: int) -> torch.Tensor:
        """Returns the unconditional prefix broadcast over a batch.

        Args:
            batch_size (int): Number of rows.

        Returns:
            torch.Tensor: `(batch_size, slots, lm_dim)` view of `null_prefix`.
        """
        return self.null_prefix.unsqueeze(0).expand(batch_size, -1, -1)

    def dropout_null(
        self,
        prefix: torch.Tensor,
        prob: float,
        generator: torch.Generator | None = None,
        null: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replaces whole rows of `prefix` with the null prefix at probability `prob`.

        Args:
            prefix (torch.Tensor): Conditional prefixes `(batch_size, slots, lm_dim)`.
            prob (float): Per-row replacement probability.
            generator (torch.Generator | None, optional): Sampling generator. Defaults to None.
            null (torch.Tensor | None, optional): The unconditional prefix to substitute, which must be as wide as
                `prefix`. Defaults to None, meaning this bridge's own `null_prefix`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(prefix, replaced)` where `replaced` is a `(batch_size,)` bool mask.

        Raises:
            ValueError: If `null` is narrower than `prefix`, which would leave conditioned slots inside the
                unconditional branch and quietly break the independence the null prefix exists to give.
        """
        batch_size = prefix.shape[0]
        if prob <= 0.0:
            return prefix, prefix.new_zeros(batch_size, dtype=torch.bool)

        unconditional = self.null(batch_size) if null is None else null
        if unconditional.shape[1:] != prefix.shape[1:]:
            raise ValueError(
                f'null prefix is {tuple(unconditional.shape[1:])} against a {tuple(prefix.shape[1:])} prefix; a '
                'partial null leaves the brain inside the unconditional branch.'
            )

        draw = torch.rand(batch_size, device=prefix.device, generator=generator)
        replaced = draw < prob
        mixed = torch.where(replaced[:, None, None], unconditional, prefix)
        return mixed, replaced


class WordResampler(nn.Module):
    """Cross-attends learned latents over the word-token hiddens to extra prefix slots.

    The ablation arm for `conditioning='pooled_plus_words'`. It is an ablation and not the default because a length-L
    memory hands the decoder the word count, which alone carries 5.14 bits of sentence identity, and because the
    word-synchronous evidence path reads the same tokens through a schedule the controls also get.

    Attributes:
        slots (int): Number of prefix positions produced.
        lm_dim (int): Frozen-LM embedding width.
        null_prefix (nn.Parameter): The `(slots, lm_dim)` unconditional continuation of `PrefixBridge.null_prefix`.
            Without one, null-prefix dropout would replace the pooled slots and leave the word slots -- and with them
            the brain and the word count -- inside the branch that is supposed to be independent of both.
    """

    def __init__(
        self,
        hidden_dim: int,
        lm_dim: int,
        slots: int = 8,
        n_blocks: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """Builds the resampler.

        Args:
            hidden_dim (int): Width of the encoder's token hiddens.
            lm_dim (int): Frozen-LM embedding width.
            slots (int, optional): Latents, and therefore prefix positions produced. Defaults to 8.
            n_blocks (int, optional): Cross-attention blocks. Defaults to 2.
            n_heads (int, optional): Attention heads per block. Defaults to 4.
            dropout (float, optional): Attention dropout. Defaults to 0.0.
        """
        super().__init__()
        self.slots = slots
        self.lm_dim = lm_dim
        self.latents = nn.Parameter(torch.empty(slots, hidden_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)
        heads = _largest_head_count(hidden_dim, n_heads)
        self.blocks = nn.ModuleList(
            nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True) for _ in range(n_blocks)
        )
        self.norms_q = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(n_blocks))
        self.norms_kv = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(n_blocks))
        self.to_lm = nn.Linear(hidden_dim, lm_dim)
        self.norm_out = nn.LayerNorm(lm_dim)
        self.null_prefix = nn.Parameter(torch.empty(slots, lm_dim))
        nn.init.trunc_normal_(self.null_prefix, std=0.02)

    def null(self, batch_size: int) -> torch.Tensor:
        """Returns the unconditional word slots broadcast over a batch.

        Args:
            batch_size (int): Number of rows.

        Returns:
            torch.Tensor: `(batch_size, slots, lm_dim)` view of `null_prefix`.
        """
        return self.null_prefix.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Resamples the token sequence into a fixed number of prefix slots.

        Args:
            hidden (torch.Tensor): Token hiddens `(batch_size, seq_len, hidden_dim)`.
            valid (torch.Tensor): Boolean `(batch_size, seq_len)`; `True` at attendable positions.

        Returns:
            torch.Tensor: Prefix embeddings `(batch_size, slots, lm_dim)`.
        """
        # A row with no attendable key yields NaN from the softmax, so it falls back to attending everything.
        empty = ~valid.any(dim=1)
        if bool(empty.any()):
            valid = valid.clone()
            valid[empty] = True
        pad = ~valid

        q = self.latents.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        for attn, norm_q, norm_kv in zip(self.blocks, self.norms_q, self.norms_kv, strict=True):
            kv = norm_kv(hidden)
            out, _ = attn(norm_q(q), kv, kv, key_padding_mask=pad, need_weights=False)
            q = q + out
        return self.norm_out(self.to_lm(q))


def _largest_head_count(dim: int, requested: int) -> int:
    """Returns the largest head count no greater than `requested` that divides `dim`."""
    for heads in range(min(requested, dim), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


def build_bridge(
    config: DecoderConfig, z_dim: int, hidden_dim: int, lm_dim: int
) -> tuple[PrefixBridge, WordResampler | None]:
    """Constructs the trainable decoder surface for a configuration.

    Args:
        config (DecoderConfig): Decoder configuration (uses `prefix_slots`, `bottleneck`, `bridge_depth`,
            `word_slots`, `conditioning`).
        z_dim (int): Width of the conditioning vector.
        hidden_dim (int): Width of the encoder's token hiddens, consumed by the resampler.
        lm_dim (int): Frozen-LM embedding width.

    Returns:
        tuple[PrefixBridge, WordResampler | None]: The bridge, and the resampler when `conditioning` asks for words.
    """
    bridge = PrefixBridge(
        z_dim,
        lm_dim,
        slots=config.prefix_slots,
        bottleneck=config.bottleneck,
        depth=config.bridge_depth,
    )
    resampler = (
        WordResampler(hidden_dim, lm_dim, slots=config.word_slots)
        if config.conditioning == 'pooled_plus_words'
        else None
    )
    n_trainable = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    n_trainable += sum(p.numel() for p in (resampler.parameters() if resampler else ()))
    _LOG.info(
        'Prefix bridge: %d -> %d x %d slots (%s), depth %d, %d trainable parameters.',
        z_dim,
        config.prefix_slots + (config.word_slots if resampler else 0),
        lm_dim,
        config.conditioning,
        config.bridge_depth,
        n_trainable,
    )
    return bridge, resampler
