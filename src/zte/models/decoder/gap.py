"""The train-fitted affine map from the EEG vector cloud onto the frozen text cloud."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from zte.config import GapCorrection
from zte.logging_utils import get_logger

_LOG = get_logger('models.decoder.gap')


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


def _matrix_power(centred: torch.Tensor, power: float, eps: float) -> torch.Tensor:
    """Returns `Sigma ** power` for the covariance of centred rows, via a symmetric eigendecomposition."""
    n = max(centred.shape[0] - 1, 1)
    cov = (centred.t() @ centred) / n
    cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)

    # MPS has neither float64 nor `_linalg_eigh`, and this is a once-per-run fit, so the round trip costs nothing.
    values, vectors = torch.linalg.eigh(cov.detach().cpu().double())
    values = values.clamp_min(eps)

    return ((vectors * values.pow(power)) @ vectors.t()).to(device=centred.device, dtype=centred.dtype)
