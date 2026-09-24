import torch
from torch import Tensor

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


def _quantize_inputs(
    a: Tensor, b: Tensor, storage_dtype: torch.dtype, compute_dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Quantize inputs to the model format, then expose its C++ compute type."""
    return (
        a.to(dtype=storage_dtype).to(dtype=compute_dtype),
        b.to(dtype=storage_dtype).to(dtype=compute_dtype),
    )


def _float32(value: float) -> float:
    """Round a Python scalar to the accumulator format used by FP32 models."""
    return torch.tensor(value, dtype=torch.float32).item()


def _quantize_scalar(value: float, storage_dtype: torch.dtype) -> float:
    """Quantize a scalar to a model storage format and return its FP32 value."""
    return (
        torch.tensor(value, dtype=torch.float32)
        .to(dtype=storage_dtype)
        .to(dtype=torch.float32)
        .item()
    )


def _quantize_tf32(x: Tensor) -> Tensor:
    """Apply the operand truncation used by NVIDIA TF32 instructions."""
    x = x.to(dtype=torch.float32).contiguous()
    return ((x.view(torch.int32) >> 13) << 13).view(torch.float32)


def _quantize_e2m1(x: Tensor) -> Tensor:
    """Round to unpacked OCP E2M1 values using nearest, ties-to-even."""
    x = x.to(dtype=torch.float32)
    magnitude = x.abs()
    quantized = torch.where(
        magnitude <= 0.25,
        0.0,
        torch.where(
            magnitude < 0.75,
            0.5,
            torch.where(
                magnitude <= 1.25,
                1.0,
                torch.where(
                    magnitude < 1.75,
                    1.5,
                    torch.where(
                        magnitude <= 2.5,
                        2.0,
                        torch.where(
                            magnitude < 3.5,
                            3.0,
                            torch.where(magnitude <= 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    quantized = torch.copysign(quantized, x)
    return torch.where(torch.isnan(x), x, quantized)


def fp64_fma(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Quantize to FP64 and apply the four-element FP64 FMA model."""
    a, b = _quantize_inputs(a, b, torch.float64, torch.float64)
    return torch.ops.fpy2_models.fp64_fma.default(a, b, c)


def amd_cdna2_bf16(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Quantize to BF16 and apply the four-element CDNA2 FTZ model."""
    a, b = _quantize_inputs(a, b, torch.bfloat16, torch.float32)
    return torch.ops.fpy2_models.amd_cdna2_bf16.default(a, b, _float32(c))


def amd_cdna2_f16(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Quantize to FP16 and apply the four-element CDNA2 FTZ model."""
    a, b = _quantize_inputs(a, b, torch.float16, torch.float32)
    return torch.ops.fpy2_models.amd_cdna2_f16.default(a, b, _float32(c))


def amd_cdna3_bf16(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Quantize to BF16 and apply the CDNA3 truncated FDPA model."""
    a, b = _quantize_inputs(a, b, torch.bfloat16, torch.float32)
    return torch.ops.fpy2_models.amd_cdna3_bf16.default(a, b, _float32(c))


def amd_cdna3_bf8(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Quantize to S1E5M2 and apply the CDNA3 grouped FDPA model."""
    a, b = _quantize_inputs(a, b, torch.float8_e5m2fnuz, torch.float32)
    return torch.ops.fpy2_models.amd_cdna3_bf8.default(a, b, _float32(c))


def amd_cdna3_f16(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Quantize to FP16 and apply the CDNA3 truncated FDPA model."""
    a, b = _quantize_inputs(a, b, torch.float16, torch.float32)
    return torch.ops.fpy2_models.amd_cdna3_f16.default(a, b, _float32(c))


def nv_ada_e5m2_f32(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Quantize to OCP E5M2 and apply the Ada FP32-accumulate model."""
    a, b = _quantize_inputs(a, b, torch.float8_e5m2, torch.float32)
    return torch.ops.fpy2_models.nv_ada_e5m2_f32.default(a, b, _float32(c))


def nv_ampere_bf16_f32(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Quantize to BF16 and apply the Ampere FP32-accumulate model."""
    a, b = _quantize_inputs(a, b, torch.bfloat16, torch.float32)
    return torch.ops.fpy2_models.nv_ampere_bf16_f32.default(a, b, _float32(c))


def nv_ampere_tf32_f32(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Truncate FP32 operands to TF32 and apply the Ampere model."""
    a, b = _quantize_tf32(a), _quantize_tf32(b)
    return torch.ops.fpy2_models.nv_ampere_tf32_f32.default(a, b, _float32(c))


def nv_blackwell_mxfp8(
    a: Tensor, b: Tensor, c: float, alpha: float, beta: float
) -> Tensor:
    """Apply the Blackwell MXFP8 model with E8M0 scalar scales."""
    a, b = _quantize_inputs(a, b, torch.float8_e5m2, torch.float32)
    alpha = _quantize_scalar(alpha, torch.float8_e8m0fnu)
    beta = _quantize_scalar(beta, torch.float8_e8m0fnu)
    return torch.ops.fpy2_models.nv_blackwell_mxfp8.default(
        a, b, _float32(c), alpha, beta
    )


def nv_blackwell_nvfp4(
    a: Tensor, b: Tensor, c: float, alphas: Tensor, betas: Tensor
) -> Tensor:
    """Apply the Blackwell NVFP4 model to unpacked E2M1 operands."""
    a, b = _quantize_e2m1(a), _quantize_e2m1(b)
    alphas, betas = _quantize_inputs(
        alphas, betas, torch.float8_e4m3fn, torch.float32
    )
    return torch.ops.fpy2_models.nv_blackwell_nvfp4.default(
        a, b, _float32(c), alphas, betas
    )


def nv_hopper_f16_f32(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Quantize to FP16 and apply the Hopper FP32-accumulate model."""
    a, b = _quantize_inputs(a, b, torch.float16, torch.float32)
    return torch.ops.fpy2_models.nv_hopper_f16_f32.default(a, b, _float32(c))


def nv_turing_f16_f32(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Quantize to FP16 and apply the Turing FP32-accumulate model."""
    a, b = _quantize_inputs(a, b, torch.float16, torch.float32)
    return torch.ops.fpy2_models.nv_turing_f16_f32.default(a, b, _float32(c))


def nv_volta_f16_f32(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Quantize to FP16 and apply the Volta FP32-accumulate model."""
    a, b = _quantize_inputs(a, b, torch.float16, torch.float32)
    return torch.ops.fpy2_models.nv_volta_f16_f32.default(a, b, _float32(c))
