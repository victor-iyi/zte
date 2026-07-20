"""Device, dtype and autocast helpers, so model and training code stays backend-agnostic across CPU/CUDA/MPS/XLA."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import torch

type DeviceKind = Literal['cpu', 'cuda', 'mps', 'xla']
type PrecisionPreference = Literal['auto', 'fp32', 'fp16', 'bf16']


def _xla_device() -> torch.device | None:
    """Returns a Cloud-TPU XLA device if `torch_xla` is installed and a TPU is present, else `None`."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

        return xm.xla_device()
    except Exception:  # noqa: BLE001 -- any import/runtime failure just means "no TPU here".
        return None


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """Resolved description of the compute device and how to use it.

    Attributes:
        device (torch.device): The concrete `torch.device` to place tensors/modules on.
        kind (DeviceKind): The backend family (`cpu`, `cuda` or `mps`).
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
    except AssertionError, RuntimeError:
        return False


def resolve_device(
    prefer: DeviceKind | Literal['auto'] = 'auto',
    precision: PrecisionPreference = 'auto',
) -> DeviceSpec:
    """Selects the best available device and a safe autocast configuration.

    Mixed-precision defaults are conservative: bf16 on capable CUDA and XLA, fp16 on older CUDA, and full precision on
    MPS/CPU where autocast support is partial or unhelpful.

    Args:
        prefer (DeviceKind | Literal['auto']): Force a backend, or 'auto' to pick the best available.
        precision (PrecisionPreference): 'auto' to choose per backend, or force one of `fp32`, `fp16` or `bf16`.

    Returns:
        DeviceSpec: A fully populated device specification.

    Raises:
        RuntimeError: If a specific backend is requested but unavailable.

    Example:
        >>> spec = resolve_device('auto')
        >>> spec.device.type in {'cpu', 'cuda', 'mps'}
        True
    """
    kind = _select_kind(prefer)
    if kind == 'xla':
        xla = _xla_device()
        if xla is None:
            raise RuntimeError('XLA/TPU requested but torch_xla is unavailable.')
        device = xla
    else:
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
    if prefer in ('xla', 'tpu'):
        if _xla_device() is None:
            raise RuntimeError('XLA/TPU requested but torch_xla is unavailable.')
        return 'xla'
    if prefer == 'cpu':
        return 'cpu'

    # auto: prefer a discrete accelerator, then a Cloud TPU, then Apple MPS, then CPU.
    if cuda_ok:
        return 'cuda'
    if not cuda_ok and not mps_ok and _xla_device() is not None:
        return 'xla'
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
        return torch.bfloat16, kind in {'cuda', 'xla'}

    # auto: the fastest accuracy-safe mixed precision per backend.
    if kind == 'cuda':
        # bf16 on Ampere+ keeps an fp32 master copy; older CUDA needs fp16 plus a GradScaler.
        return (torch.bfloat16, True) if _cuda_supports_bf16() else (torch.float16, True)
    if kind == 'xla':
        return torch.bfloat16, True  # Cloud TPUs are bf16-native

    # MPS autocast still NaNs on some ops and CPU AMP rarely helps, so both stay fp32.
    return None, False


def configure_backend(spec: DeviceSpec) -> None:
    """Applies backend-global performance settings that do not change training accuracy.

    Idempotent; call once at pipeline setup. Only CUDA has a global switch worth flipping -- TF32 for fp32 matmuls and
    cuDNN, a large Ampere+ speedup and a no-op below it. MPS/CPU/XLA handle their precision through `autocast`.

    Args:
        spec (DeviceSpec): The resolved device specification.
    """
    if spec.kind != 'cuda':
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    except AttributeError, RuntimeError:  # pragma: no cover -- depends on the CUDA build
        pass


def auto_num_workers(spec: DeviceSpec, requested: int) -> int:
    """Resolves the DataLoader worker count, auto-picking when `requested < 0`.

    Auto uses a few workers on an accelerator, where input pipelining matters, and stays single-process on CPU, where
    extra workers rarely help and hurt reproducibility.

    Args:
        spec (DeviceSpec): The resolved device specification.
        requested (int): The configured worker count, or a negative value to request `auto`.

    Returns:
        int: The effective number of DataLoader workers.
    """
    if requested >= 0:
        return requested
    if spec.kind in {'cuda', 'mps', 'xla'}:
        return min(4, max(1, (os.cpu_count() or 2) - 1))
    return 0


def _device_name(kind: DeviceKind, device: torch.device) -> str:
    """Builds a readable device name for logs."""
    if kind == 'cuda':
        try:
            return f'cuda:{device.index or 0} ({torch.cuda.get_device_name(device)})'
        except AssertionError, RuntimeError:
            return 'cuda'
    if kind == 'mps':
        return 'mps (Apple Silicon)'
    if kind == 'xla':
        return f'xla / TPU ({device})'
    return f'cpu ({os.cpu_count() or 1} cores)'


@contextlib.contextmanager
def autocast(spec: DeviceSpec) -> Iterator[None]:
    """Context manager that enables autocast when the device spec allows it.

    Args:
        spec (DeviceSpec): The resolved device specification.

    Yields:
        None: Control inside a `torch.autocast` block when AMP is enabled, or a no-op context otherwise.

    Example:
        >>> spec = resolve_device('cpu')
        >>> with autocast(spec):
        ...     pass
    """
    if not (spec.use_amp and spec.autocast_dtype is not None):
        yield
        return
    if spec.kind == 'xla':
        # TPU autocast lives in torch_xla; degrade through the generic device_type to fp32 rather than crash.
        try:
            import torch_xla.amp  # type: ignore[import-untyped]

            with torch_xla.amp.autocast(spec.device):
                yield
            return
        except Exception:  # noqa: BLE001 -- try the generic path next.
            pass
        try:
            with torch.autocast(device_type='xla', dtype=spec.autocast_dtype):
                yield
            return
        except RuntimeError, ValueError:
            yield
            return
    with torch.autocast(device_type=spec.kind, dtype=spec.autocast_dtype):
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
