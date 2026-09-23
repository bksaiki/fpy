import torch as _torch  # Load LibTorch before the extension on platforms using rpath.

from . import _C as _C
from .ops import amd_cdna2_bf16, amd_cdna2_f16, fp64_fma

__all__ = ["amd_cdna2_bf16", "amd_cdna2_f16", "fp64_fma"]
