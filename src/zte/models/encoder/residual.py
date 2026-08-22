"""Predictive residual coding: keep the part of a word's EEG that its context did not already predict."""

from __future__ import annotations

import torch
from torch import nn

from zte.config import ModelConfig
from zte.logging_utils import get_logger
from zte.models.transformer import ZTETransformerEncoder

_LOG = get_logger('models.encoder.residual')


class PredictiveResidual(nn.Module):
    """Subtracts the left context's expectation of a token from the token itself.

    Reading is predictive, and the largest language-related EEG deflection is a surprisal response: the reaction to
    word `w` is dominated by how unexpected `w` was, not by `w` in isolation. Everything that is *not* surprising about
    a moment of reading -- the reader's tonic state, cap impedance, the 1/f background, the drift of the last few
    seconds -- is predictable from the preceding tokens and therefore cancels in the residual, while the word-specific
    response does not.

    Note:
        The expectation head trains on its own regression loss against a detached target, and the encoder sees the
        expectation only as a detached offset. Otherwise the encoder could cut the regression loss by making itself
        predictable, which is collapse wearing a disguise.

    Attributes:
        gate (nn.Parameter): Scalar mixing weight; the module is the identity at `gate = 0`.
    """

    def __init__(self, dim: int, n_layers: int = 1, n_heads: int = 4, dropout: float = 0.0, gate: float = 1.0) -> None:
        """Builds the expectation head.

        Args:
            dim (int): Token hidden width.
            n_layers (int, optional): Depth of the causal expectation encoder. Defaults to 1.
            n_heads (int, optional): Attention heads in that encoder. Defaults to 4.
            dropout (float, optional): Dropout inside the expectation encoder. Defaults to 0.0.
            gate (float, optional): Initial value of the subtraction gate. Defaults to 1.0.
        """
        super().__init__()
        self.bos = nn.Parameter(torch.zeros(1, 1, dim))
        self.context = ZTETransformerEncoder(
            dim=dim, n_heads=_divisor(dim, n_heads), n_layers=n_layers, dropout=dropout, pos='rope'
        )
        self.readout = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(float(gate)))
        nn.init.trunc_normal_(self.bos, std=0.02)

    def expectation(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Predicts each token from the tokens before it.

        Args:
            hidden (torch.Tensor): Token hiddens `(batch_size, seq_len, dim)`.
            valid (torch.Tensor): Boolean `(batch_size, seq_len)` of real positions.

        Returns:
            torch.Tensor: Predicted hiddens `(batch_size, seq_len, dim)`; position 0 sees only the learned BOS.
        """
        b = hidden.shape[0]
        shifted = torch.cat([self.bos.expand(b, 1, hidden.shape[-1]).to(hidden.dtype), hidden[:, :-1]], dim=1)
        shifted_valid = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=valid.device), valid[:, :-1]], dim=1)

        return self.readout(self.context(shifted, shifted_valid, causal=True))

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Returns `(residual, prediction_loss, metrics)` for a batch of token hiddens.

        Args:
            hidden (torch.Tensor): Token hiddens `(batch_size, seq_len, dim)`.
            valid (torch.Tensor): Boolean `(batch_size, seq_len)` of real, usable positions.

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict[str, float]]: The de-trended hiddens, the expectation head's own
                regression loss, and the metrics naming how much of a token turned out to be predictable.
        """
        # The head reads a detached input and regresses a detached target, so this loss reaches the head and nothing
        # else; `hidden` keeps exactly the gradient its downstream objective gives it, minus a constant offset.
        predicted = self.expectation(hidden.detach(), valid)
        mask = valid.unsqueeze(-1)
        n = mask.sum().clamp_min(1)
        predict_loss = (((predicted - hidden.detach()) ** 2) * mask).sum() / (n * hidden.shape[-1])

        residual = hidden - self.gate * predicted.detach()

        with torch.no_grad():
            var_hidden = _masked_variance(hidden, mask)
            var_residual = _masked_variance(residual, mask)
            explained = float(1.0 - (var_residual / var_hidden.clamp_min(1e-8)))
        return (
            residual,
            predict_loss,
            {
                'residual_predict_loss': float(predict_loss.detach()),
                'residual_context_explained': explained,
                'residual_gate': float(self.gate.detach()),
            },
        )


def _masked_variance(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Returns the scalar variance of `x` over the positions `mask` marks true."""
    n = mask.sum().clamp_min(1) * x.shape[-1]
    total = (x * mask).sum() / n
    return ((((x - total) ** 2) * mask).sum() / n).detach()


def _divisor(value: int, target: int) -> int:
    """Returns the largest divisor of `value` that is at most `target`, so the head count always divides the width."""
    for h in range(min(target, value), 0, -1):
        if value % h == 0:
            return h
    return 1


def build_predictive_residual(config: ModelConfig, dim: int) -> PredictiveResidual | None:
    """Constructs the residual coder a model configuration asks for, or `None` when it is off.

    Args:
        config (ModelConfig): Model configuration (reads the `residual_*` fields).
        dim (int): Token hidden width.

    Returns:
        PredictiveResidual | None: The coder, or `None`.
    """
    if not config.residual_coding:
        return None
    _LOG.info(
        'Predictive residual coding on: %d-layer causal expectation head over %d dims, gate init %.2f.',
        config.residual_layers,
        dim,
        config.residual_gate,
    )
    return PredictiveResidual(
        dim,
        n_layers=config.residual_layers,
        n_heads=config.n_heads,
        dropout=config.dropout,
        gate=config.residual_gate,
    )


__all__ = ['PredictiveResidual', 'build_predictive_residual']
