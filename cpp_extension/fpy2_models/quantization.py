"""Quantization adapters for the native MMA model wrappers."""

import torch
from torch import Tensor
from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    f32_to_f4_unpacked,
)


def quantize_e2m1(x: Tensor) -> Tensor:
    """Round *x* to unpacked OCP E2M1 values using TorchAO.

    TorchAO's E2M1 encoding has no representation for NaN.  The native MMA
    models still need NaNs to reach their special-value handling, so preserve
    them after delegating all finite rounding and saturation to TorchAO.
    """
    x = x.to(dtype=torch.float32).contiguous()
    quantized = f4_unpacked_to_f32(f32_to_f4_unpacked(x))
    return torch.where(torch.isnan(x), x, quantized)
