"""Device, dtype and autocast helpers for portable CPU / CUDA / MPS execution.

The package is meant to run unchanged on a CPU box, an Apple-silicon Mac (the `mps` backend, e.g. an M-series chip) and an Nvidia GPU (`cuda`).
All device-specific decisions -- which accelerator to use, whether mixed precision is safe, which autocast dtype to pick,
whether to pin host memory -- are centralised here so model and training code stays backend-agnostic.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import torch

type DeviceKind = Literal['cpu', 'cuda', 'mps']
type PrecisionPreference = Literal['auto', 'fp32', 'fp16', 'bf16']


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """Resolved description of the compute device and how to use it.

    Attributes:
        device (torch.device): The concrete `torch.device` to place tensors/modules on.
        kind (DeviceKind): The backend family (`'cpu'`, `'cuda'` or `'mps'`).
        autocast_dtype (torch.dtype | None): The dtype to use under `torch.autocast`, or `None` when mixed precision is disabled.
        use_amp (bool): Whether automatic mixed precision should be enabled.
        supports_pin_memory (bool): Whether `DataLoader(pin_memory=True)` helps.
        name (str): A human-readable device name for logging.

    """

    device: torch.device
    kind: DeviceKind
    autocast_dtype: torch.dtype | None
    use_amp: bool
    supports_pin_memory: bool
    name: str

    @property
    def is_cuda(self) -> bool:
        """Returns `True` when the resolved device is an Nvidia GPU."""
        return self.kind == 'cuda'

    @property
    def is_mps(self) -> bool:
        """Returns `True` when the resolved device is Apple-silicon MPS."""
        return self.kind == 'mps'


def _cuda_supports_bf16() -> bool:
    """Returns whether the current CUDA device supports bfloat16 natively."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except (AssertionError, RuntimeError):
        return False


def resolve_device(
    prefer: DeviceKind | Literal['auto'] = 'auto',
    precision: PrecisionPreference = 'auto',
) -> DeviceSpec:
    """Selects the best available device and a safe autocast configuration.

    Selection order for ``prefer='auto'`` is CUDA, then MPS, then CPU. Mixed
    precision defaults are chosen conservatively: bfloat16 on capable CUDA
    devices, float16 on older CUDA devices, and full precision on MPS/CPU where
    autocast support is partial or unhelpful.

    Args:
        prefer (DeviceKind | Literal['auto']): Force a backend, or `'auto'` to pick the best available.
        precision (PrecisionPreference): `'auto'` to choose per backend, or force one of `'fp32'`, `'fp16'` or `'bf16'`.

    Returns:
        A fully populated `DeviceSpec`.

    Raises:
        RuntimeError: If a specific backend is requested but unavailable.

    Example:
        >>> spec = resolve_device('auto')
        >>> spec.device.type in {'cpu', 'cuda', 'mps'}
        True
    """
    kind = _select_kind(prefer)
    device = torch.device(kind if kind != 'cuda' else 'cuda:0')
    autocast_dtype, use_amp = _resolve_precision(kind, precision)
    name = _device_name(kind, device)
    return DeviceSpec(
        device=device,
        kind=kind,
        autocast_dtype=autocast_dtype,
        use_amp=use_amp,
        supports_pin_memory=(kind == 'cuda'),
        name=name,
    )


def _select_kind(prefer: DeviceKind | Literal['auto']) -> DeviceKind:
    """Resolves the requested preference to a concrete, available backend."""
    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available() and torch.backends.mps.is_built()

    if prefer == 'cuda':
        if not cuda_ok:
            raise RuntimeError('CUDA requested but torch.cuda.is_available() is False.')
        return 'cuda'
    if prefer == 'mps':
        if not mps_ok:
            raise RuntimeError('MPS requested but the MPS backend is unavailable.')
        return 'mps'
    if prefer == 'cpu':
        return 'cpu'

    # auto
    if cuda_ok:
        return 'cuda'
    if mps_ok:
        return 'mps'
    return 'cpu'


def _resolve_precision(
    kind: DeviceKind, precision: PrecisionPreference
) -> tuple[torch.dtype | None, bool]:
    """Maps a (backend, precision) request to an (autocast dtype, use_amp) pair."""
    if precision == 'fp32':
        return None, False
    if precision == 'fp16':
        return torch.float16, kind in {'cuda', 'mps'}
    if precision == 'bf16':
        return torch.bfloat16, kind == 'cuda'

    # auto: be conservative outside CUDA.
    if kind == 'cuda':
        return (torch.bfloat16, True) if _cuda_supports_bf16() else (torch.float16, True)
    # MPS autocast is still maturing; CPU AMP rarely helps -> default to fp32.
    return None, False


def _device_name(kind: DeviceKind, device: torch.device) -> str:
    """Builds a readable device name for logs."""
    if kind == 'cuda':
        try:
            return f'cuda:{device.index or 0} ({torch.cuda.get_device_name(device)})'
        except (AssertionError, RuntimeError):
            return 'cuda'
    if kind == 'mps':
        return 'mps (Apple Silicon)'
    return f'cpu ({os.cpu_count() or 1} cores)'


@contextlib.contextmanager
def autocast(spec: DeviceSpec) -> Iterator[None]:
    """Context manager that enables autocast when the device spec allows it.

    Args:
        spec (DeviceSpec): The resolved device specification.

    Yields:
        Control inside a `torch.autocast` block when AMP is enabled, or a
        no-op context otherwise.

    Example:
        >>> spec = resolve_device('cpu')
        >>> with autocast(spec):
        ...     pass
    """
    if spec.use_amp and spec.autocast_dtype is not None:
        with torch.autocast(device_type=spec.kind, dtype=spec.autocast_dtype):
            yield
    else:
        yield


# pylint: disable=import-outside-toplevel
def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seeds Python, NumPy and Torch RNGs for reproducible runs.

    Args:
        seed (int): The base seed applied to every RNG.
        deterministic (bool): If `True`, also request deterministic cuDNN kernels (slower but reproducible on CUDA).

    Example:
        >>> seed_everything(42)
        >>> torch.randn(10).mean()
        tensor(0.0000)
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
