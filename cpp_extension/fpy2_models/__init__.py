import torch as _torch  # Load LibTorch before the extension on platforms using rpath.

from . import _C as _C
from .ops import (
    amd_cdna2_bf16,
    amd_cdna2_f16,
    amd_cdna3_bf8,
    amd_cdna3_bf16,
    amd_cdna3_f16,
    fp64_fma,
    nv_ada_e5m2_f32,
    nv_ampere_bf16_f32,
    nv_ampere_tf32_f32,
    nv_blackwell_mxfp8,
    nv_blackwell_nvfp4,
    nv_hopper_f16_f32,
    nv_turing_f16_f32,
    nv_volta_f16_f32,
)

__all__ = [
    "amd_cdna2_bf16",
    "amd_cdna2_f16",
    "amd_cdna3_bf8",
    "amd_cdna3_bf16",
    "amd_cdna3_f16",
    "fp64_fma",
    "nv_ada_e5m2_f32",
    "nv_ampere_bf16_f32",
    "nv_ampere_tf32_f32",
    "nv_blackwell_mxfp8",
    "nv_blackwell_nvfp4",
    "nv_hopper_f16_f32",
    "nv_turing_f16_f32",
    "nv_volta_f16_f32",
]
