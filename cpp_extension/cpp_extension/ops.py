import torch
from torch import Tensor

__all__ = ["amd_cdna2_bf16", "amd_cdna2_f16", "fp64_fma"]


def fp64_fma(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Apply the four-element FP64 FMA dot-product model."""
    return torch.ops.fpy2_models.fp64_fma.default(a, b, c)


@torch.library.register_fake("fpy2_models::fp64_fma")
def _fp64_fma_fake(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float64)
    torch._check(b.dtype == torch.float64)
    torch._check(a.device == b.device)
    torch._check(a.ndim >= 1)
    torch._check(a.shape[-1] == 4)
    return a.new_empty(a.shape[:-1])


def amd_cdna2_bf16(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Apply the four-element CDNA2 BF16 FTZ dot-product model."""
    return torch.ops.fpy2_models.amd_cdna2_bf16.default(a, b, c)


@torch.library.register_fake("fpy2_models::amd_cdna2_bf16")
def _amd_cdna2_bf16_fake(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.bfloat16)
    torch._check(b.dtype == torch.bfloat16)
    torch._check(a.device == b.device)
    torch._check(a.ndim >= 1)
    torch._check(a.shape[-1] == 4)
    return a.new_empty(a.shape[:-1], dtype=torch.float32)


def amd_cdna2_f16(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Apply the four-element CDNA2 FP16 FTZ dot-product model."""
    return torch.ops.fpy2_models.amd_cdna2_f16.default(a, b, c)


@torch.library.register_fake("fpy2_models::amd_cdna2_f16")
def _amd_cdna2_f16_fake(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float16)
    torch._check(b.dtype == torch.float16)
    torch._check(a.device == b.device)
    torch._check(a.ndim >= 1)
    torch._check(a.shape[-1] == 4)
    return a.new_empty(a.shape[:-1], dtype=torch.float32)
