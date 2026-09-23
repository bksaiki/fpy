import fpy2_models
import torch


def _inputs(dtype):
    a = torch.tensor([-2.0**13, -0.5, -0.25, -0.125], dtype=dtype)
    b = torch.tensor([2.0**10, 1.0, 1.0, 1.0], dtype=dtype)
    return a, b


def test_table8_values():
    a, b = _inputs(torch.float64)
    assert fpy2_models.fp64_fma(a, b, 2.0**23).item() == -0.875

    a, b = _inputs(torch.bfloat16)
    assert fpy2_models.amd_cdna2_bf16(a, b, 2.0**23).item() == -0.375

    a, b = _inputs(torch.float16)
    assert fpy2_models.amd_cdna2_f16(a, b, 2.0**23).item() == 0.0


def test_batched_shape():
    a, b = _inputs(torch.float64)
    output = fpy2_models.fp64_fma(a.repeat(3, 1), b.repeat(3, 1), 2.0**23)
    assert output.shape == (3,)
    torch.testing.assert_close(
        output, torch.full((3,), -0.875, dtype=torch.float64)
    )


def test_operator_registration():
    a, b = _inputs(torch.float64)
    torch.library.opcheck(
        torch.ops.fpy2_models.fp64_fma.default,
        (a, b, 2.0**23),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
