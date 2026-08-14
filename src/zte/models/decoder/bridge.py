"""The trainable surface of the prefix decoder: pooled-vector bridge, word resampler, modality-gap correction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from zte.config import DecoderConfig, GapCorrection
from zte.logging_utils import get_logger

_LOG = get_logger('models.decoder.bridge')


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

    def __init__(self, z_dim: int, lm_dim: int, slots: int = 8, bottleneck: int = 128) -> None:
        """Builds the bridge.

        Args:
            z_dim (int): Width of the conditioning vector.
            lm_dim (int): Frozen-LM embedding width.
            slots (int, optional): Prefix positions produced. Defaults to 8.
            bottleneck (int, optional): Rank of the shared low-rank map. Defaults to 128.
        """
        super().__init__()
        self.slots = slots
        self.lm_dim = lm_dim
        self.norm_in = nn.LayerNorm(z_dim)
        self.to_bottleneck = nn.Linear(z_dim, bottleneck)
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
        self, prefix: torch.Tensor, prob: float, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replaces whole rows of `prefix` with the null prefix at probability `prob`.

        Args:
            prefix (torch.Tensor): Conditional prefixes `(batch_size, slots, lm_dim)`.
            prob (float): Per-row replacement probability.
            generator (torch.Generator | None, optional): Sampling generator. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(prefix, replaced)` where `replaced` is a `(batch_size,)` bool mask.
        """
        batch_size = prefix.shape[0]
        if prob <= 0.0:
            return prefix, prefix.new_zeros(batch_size, dtype=torch.bool)
        draw = torch.rand(batch_size, device=prefix.device, generator=generator)
        replaced = draw < prob
        mixed = torch.where(replaced[:, None, None], self.null(batch_size), prefix)
        return mixed, replaced


class WordResampler(nn.Module):
    """Cross-attends learned latents over the word-token hiddens to extra prefix slots.

    The ablation arm for `conditioning='pooled_plus_words'`. It is an ablation and not the default because
    cross-subject word-level content is measurably absent on ZuCo, and because a length-L memory hands the decoder the
    word count, which alone carries 5.14 bits of sentence identity.

    Attributes:
        slots (int): Number of prefix positions produced.
        lm_dim (int): Frozen-LM embedding width.
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


class GapCorrector(nn.Module):
    """Train-fitted affine map from the EEG vector cloud onto the text vector cloud.

    An EEG sentence vector aligned to a text space by a contrastive head still sits on its own shell: a systematic
    mean and scale offset that a frozen LM reads as out-of-distribution. The correction is fitted once on the training
    split and then applied row by row, so it is a property of the model and not of the evaluation set. Fitting it on
    anything a held-out row can see is transductive and is exactly the contamination this class exists to avoid.

    Attributes:
        dim (int): Vector width.
        mode (GapCorrection): `'none'`, `'mean_scale'` (per-dimension) or `'whiten'` (full covariance).
        n_fit (int): Rows the current statistics were fitted on, carried into `state` as provenance.
    """

    mu_eeg: torch.Tensor
    sigma_eeg: torch.Tensor
    mu_txt: torch.Tensor
    sigma_txt: torch.Tensor
    fitted: torch.Tensor
    whiten_eeg: torch.Tensor
    colour_txt: torch.Tensor

    def __init__(self, dim: int, mode: GapCorrection = 'mean_scale', eps: float = 1e-6) -> None:
        """Builds an unfitted corrector, which is the identity until `fit` is called.

        Args:
            dim (int): Vector width.
            mode (GapCorrection, optional): Correction family. Defaults to 'mean_scale'.
            eps (float, optional): Numerical floor for divisions and eigenvalues. Defaults to 1e-6.
        """
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.eps = eps
        self.n_fit = 0
        self.register_buffer('mu_eeg', torch.zeros(dim))
        self.register_buffer('sigma_eeg', torch.ones(dim))
        self.register_buffer('mu_txt', torch.zeros(dim))
        self.register_buffer('sigma_txt', torch.ones(dim))
        self.register_buffer('fitted', torch.zeros((), dtype=torch.bool))
        if mode == 'whiten':
            self.register_buffer('whiten_eeg', torch.eye(dim))
            self.register_buffer('colour_txt', torch.eye(dim))

    @torch.no_grad()
    def fit(self, z_eeg: torch.Tensor, z_txt: torch.Tensor) -> None:
        """Fits the correction from paired clouds drawn from the training split only.

        Args:
            z_eeg (torch.Tensor): EEG vectors `(n_eeg, dim)`.
            z_txt (torch.Tensor): Text vectors `(n_txt, dim)`; the rows need not correspond to `z_eeg`.

        Raises:
            ValueError: If either cloud has fewer than two rows or the wrong width.
        """
        if self.mode == 'none':
            self.fitted.fill_(True)
            self.n_fit = int(z_eeg.shape[0])
            return
        for name, cloud in (('z_eeg', z_eeg), ('z_txt', z_txt)):
            if cloud.ndim != 2 or cloud.shape[1] != self.dim:
                raise ValueError(f'{name} must be (n, {self.dim}), got {tuple(cloud.shape)}.')
            if cloud.shape[0] < 2:
                raise ValueError(f'{name} needs at least 2 rows to fit a gap correction.')

        eeg = z_eeg.detach().to(torch.float32)
        txt = z_txt.detach().to(torch.float32)
        self.mu_eeg.copy_(eeg.mean(0))
        self.mu_txt.copy_(txt.mean(0))
        self.sigma_eeg.copy_(eeg.std(0).clamp_min(self.eps))
        self.sigma_txt.copy_(txt.std(0).clamp_min(self.eps))
        if self.mode == 'whiten':
            self.whiten_eeg.copy_(_matrix_power(eeg - self.mu_eeg, -0.5, self.eps))
            self.colour_txt.copy_(_matrix_power(txt - self.mu_txt, 0.5, self.eps))
        self.fitted.fill_(True)
        self.n_fit = int(eeg.shape[0])
        _LOG.info(
            'Fitted GapCorrector(%s) on %d EEG and %d text vectors; mean offset %.4f.',
            self.mode,
            self.n_fit,
            int(txt.shape[0]),
            float((self.mu_txt - self.mu_eeg).abs().mean()),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Applies the fitted correction.

        Args:
            z (torch.Tensor): EEG vectors `(..., dim)`.

        Returns:
            torch.Tensor: Corrected vectors of the same shape; `z` unchanged when the mode is `'none'` or unfitted.
        """
        if self.mode == 'none':
            return z
        if not bool(self.fitted):
            _LOG.warning('GapCorrector(%s) is unfitted and is passing vectors through.', self.mode)
            return z
        centred = z - self.mu_eeg
        if self.mode == 'whiten':
            return (centred @ self.whiten_eeg) @ self.colour_txt + self.mu_txt
        return centred / self.sigma_eeg * self.sigma_txt + self.mu_txt

    @property
    def state(self) -> dict[str, Any]:
        """Returns a serialisable dict of the fitted statistics, for `extra['gap_correction']`."""
        out: dict[str, Any] = {
            'mode': self.mode,
            'dim': self.dim,
            'eps': self.eps,
            'fitted': bool(self.fitted),
            'n_fit': self.n_fit,
            'mu_eeg': self.mu_eeg.detach().cpu().numpy(),
            'sigma_eeg': self.sigma_eeg.detach().cpu().numpy(),
            'mu_txt': self.mu_txt.detach().cpu().numpy(),
            'sigma_txt': self.sigma_txt.detach().cpu().numpy(),
        }
        if self.mode == 'whiten':
            out['whiten_eeg'] = self.whiten_eeg.detach().cpu().numpy()
            out['colour_txt'] = self.colour_txt.detach().cpu().numpy()
        return out

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> GapCorrector:
        """Rebuilds a corrector from `state`.

        Args:
            state (dict[str, Any]): A dict previously produced by `state`.

        Returns:
            GapCorrector: The restored corrector.
        """
        gap = cls(int(state['dim']), mode=state['mode'], eps=float(state['eps']))
        gap.n_fit = int(state.get('n_fit', 0))
        for name in ('mu_eeg', 'sigma_eeg', 'mu_txt', 'sigma_txt'):
            getattr(gap, name).copy_(torch.as_tensor(np.asarray(state[name], dtype=np.float32)))
        if gap.mode == 'whiten':
            for name in ('whiten_eeg', 'colour_txt'):
                getattr(gap, name).copy_(torch.as_tensor(np.asarray(state[name], dtype=np.float32)))
        gap.fitted.fill_(bool(state.get('fitted', False)))
        return gap


def _largest_head_count(dim: int, requested: int) -> int:
    """Returns the largest head count no greater than `requested` that divides `dim`."""
    for heads in range(min(requested, dim), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


def _matrix_power(centred: torch.Tensor, power: float, eps: float) -> torch.Tensor:
    """Returns `Sigma ** power` for the covariance of centred rows, via a symmetric eigendecomposition."""
    n = max(centred.shape[0] - 1, 1)
    cov = (centred.t() @ centred) / n
    cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    values, vectors = torch.linalg.eigh(cov.double())
    values = values.clamp_min(eps)
    return ((vectors * values.pow(power)) @ vectors.t()).to(centred.dtype)


def build_bridge(
    config: DecoderConfig, z_dim: int, hidden_dim: int, lm_dim: int
) -> tuple[PrefixBridge, WordResampler | None]:
    """Constructs the trainable decoder surface for a configuration.

    Args:
        config (DecoderConfig): Decoder configuration (uses `prefix_slots`, `bottleneck`, `word_slots`, `conditioning`).
        z_dim (int): Width of the conditioning vector.
        hidden_dim (int): Width of the encoder's token hiddens, consumed by the resampler.
        lm_dim (int): Frozen-LM embedding width.

    Returns:
        tuple[PrefixBridge, WordResampler | None]: The bridge, and the resampler when `conditioning` asks for words.
    """
    bridge = PrefixBridge(z_dim, lm_dim, slots=config.prefix_slots, bottleneck=config.bottleneck)
    resampler = (
        WordResampler(hidden_dim, lm_dim, slots=config.word_slots)
        if config.conditioning == 'pooled_plus_words'
        else None
    )
    n_trainable = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    n_trainable += sum(p.numel() for p in (resampler.parameters() if resampler else ()))
    _LOG.info(
        'Prefix bridge: %d -> %d x %d slots (%s), %d trainable parameters.',
        z_dim,
        config.prefix_slots + (config.word_slots if resampler else 0),
        lm_dim,
        config.conditioning,
        n_trainable,
    )
    return bridge, resampler
