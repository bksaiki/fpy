import fpy2_models
import pytest
import torch
from fpy2_models.ops import _quantize_tf32
from fpy2_models.quantization import quantize_e2m1
from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    f32_to_f4_unpacked,
)


def _inputs(dtype):
    a = torch.tensor([-2.0**13, -0.5, -0.25, -0.125], dtype=dtype)
    b = torch.tensor([2.0**10, 1.0, 1.0, 1.0], dtype=dtype)
    return a, b


def _padded_inputs(k):
    a = torch.zeros(k, dtype=torch.float64)
    b = torch.zeros(k, dtype=torch.float64)
    base_a, base_b = _inputs(torch.float64)
    a[:4] = base_a
    b[:4] = base_b
    return a, b


def test_table8_values():
    a, b = _inputs(torch.float64)
    assert fpy2_models.fp64_fma(a, b, 2.0**23).item() == -0.875

    assert fpy2_models.amd_cdna2_bf16(a, b, 2.0**23).item() == -0.375

    assert fpy2_models.amd_cdna2_f16(a, b, 2.0**23).item() == 0.0


@pytest.mark.parametrize(
    ("name", "k", "expected"),
    [
        ("amd_cdna3_bf16", 8, -0.5),
        ("amd_cdna3_bf8", 16, -1.0),
        ("amd_cdna3_f16", 8, -0.5),
        ("nv_ada_e5m2_f32", 16, 0.0),
        ("nv_ampere_bf16_f32", 16, -0.5),
        ("nv_ampere_tf32_f32", 8, -0.5),
        ("nv_hopper_f16_f32", 32, -0.75),
        ("nv_turing_f16_f32", 16, -0.5),
        ("nv_volta_f16_f32", 8, 0.0),
    ],
)
def test_additional_table8_values(name, k, expected):
    a, b = _padded_inputs(k)
    output = getattr(fpy2_models, name)(a, b, 2.0**23)
    assert output.dtype == torch.float32
    assert output.item() == expected


def test_blackwell_scaled_models():
    a, b = _padded_inputs(32)
    output = fpy2_models.nv_blackwell_mxfp8(a, b, 2.0**23, 1.0, 1.0)
    assert output.item() == -0.75

    a = torch.zeros(64)
    b = torch.zeros(64)
    a[0] = b[0] = 1.0
    scales = torch.ones(4)
    output = fpy2_models.nv_blackwell_nvfp4(a, b, 0.0, scales, scales)
    assert output.item() == 1.0


def test_blackwell_group_scaled_batch():
    a = torch.zeros(2, 64)
    b = torch.zeros(2, 64)
    a[:, 0] = b[:, 0] = 1.0
    scales = torch.ones(2, 4)
    output = fpy2_models.nv_blackwell_nvfp4(a, b, 0.0, scales, scales)
    torch.testing.assert_close(output, torch.ones(2))


def test_e2m1_quantization_uses_torchao_numerics():
    inputs = torch.tensor(
        [
            -float("inf"),
            -7.0,
            -5.0,
            -3.5,
            -2.5,
            -1.75,
            -1.25,
            -0.75,
            -0.25,
            -0.0,
            0.0,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            7.0,
            float("inf"),
        ],
        dtype=torch.float64,
    )
    expected = f4_unpacked_to_f32(f32_to_f4_unpacked(inputs.float()))

    actual = quantize_e2m1(inputs)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.signbit(actual[9])
    assert not torch.signbit(actual[10])


def test_e2m1_quantization_preserves_nan_for_model_special_values():
    inputs = torch.tensor([float("nan"), -float("nan")])

    actual = quantize_e2m1(inputs)

    assert torch.isnan(actual).all()
    assert not torch.signbit(actual[0])
    assert torch.signbit(actual[1])


def test_e2m1_quantization_accepts_noncontiguous_batches():
    inputs = torch.tensor([[0.25, 0.75], [2.5, 5.0]]).t()
    assert not inputs.is_contiguous()

    actual = quantize_e2m1(inputs)

    torch.testing.assert_close(
        actual,
        torch.tensor([[0.0, 2.0], [1.0, 4.0]]),
        atol=0,
        rtol=0,
    )


def test_blackwell_nvfp4_quantizes_operands_to_e2m1():
    a = torch.zeros(64)
    b = torch.zeros(64)
    a[0] = 0.75
    b[0] = 1.25
    scales = torch.ones(4)

    actual = fpy2_models.nv_blackwell_nvfp4(a, b, 0.0, scales, scales)

    # Both halfway cases round to 1.0 under E2M1 round-to-nearest-even.
    assert actual.item() == 1.0


def test_tf32_truncation_remains_an_explicit_local_exception():
    inputs = torch.tensor([1.00048828125, -1.00048828125, float("nan")])

    actual = _quantize_tf32(inputs)

    torch.testing.assert_close(actual[:2], torch.tensor([1.0, -1.0]))
    assert torch.isnan(actual[2])


def test_batched_shape():
    a, b = _inputs(torch.float64)
    output = fpy2_models.fp64_fma(a.repeat(3, 1), b.repeat(3, 1), 2.0**23)
    assert output.shape == (3,)
    torch.testing.assert_close(
        output, torch.full((3,), -0.875, dtype=torch.float64)
    )


def test_raw_operator_accepts_noncontiguous_batches():
    a, b = _inputs(torch.float64)
    a = a.repeat(3, 1).t().contiguous().t()
    b = b.repeat(3, 1).t().contiguous().t()
    assert not a.is_contiguous()
    assert not b.is_contiguous()

    output = torch.ops.fpy2_models.fp64_fma.default(a, b, 2.0**23)

    torch.testing.assert_close(
        output, torch.full((3,), -0.875, dtype=torch.float64)
    )


def test_operator_registration():
    a, b = _inputs(torch.float64)
    output = torch.ops.fpy2_models.fp64_fma.default(a, b, 2.0**23)
    assert output.item() == -0.875


def test_python_quantizes_before_calling_cpp():
    a = torch.tensor([1.003, 2.007, 3.011, 4.015], dtype=torch.float64)
    b = torch.ones(4, dtype=torch.float64)

    expected_bf16 = torch.ops.fpy2_models.amd_cdna2_bf16.default(
        a.to(torch.bfloat16).to(torch.float32),
        b.to(torch.bfloat16).to(torch.float32),
        0.0,
    )
    expected_f16 = torch.ops.fpy2_models.amd_cdna2_f16.default(
        a.to(torch.float16).to(torch.float32),
        b.to(torch.float16).to(torch.float32),
        0.0,
    )

    torch.testing.assert_close(fpy2_models.amd_cdna2_bf16(a, b, 0.0), expected_bf16)
    torch.testing.assert_close(fpy2_models.amd_cdna2_f16(a, b, 0.0), expected_f16)
