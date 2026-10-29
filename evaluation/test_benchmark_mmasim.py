from pathlib import Path

import pytest

from evaluation.benchmark_mmasim import (
    DEFAULT_MMASIM_ROOT,
    _instruction,
    _manifest,
    build_cases,
    prepare_case,
    validate,
)


@pytest.fixture(scope="module")
def cases():
    mmasim_root = Path(DEFAULT_MMASIM_ROOT)
    if not mmasim_root.is_dir():
        pytest.skip(f"local MMA-Sim checkout not found at {mmasim_root}")
    return build_cases(mmasim_root)


def test_registry_contains_all_compiled_designs(cases):
    # Every model built into fpy2_models has an MMA-Sim reference.
    assert len(cases) == len(_manifest()["models"]) == 60
    assert len({case.name for case in cases}) == len(cases)
    assert len({case.op_name for case in cases}) == len(cases)
    # validation.json records each NVIDIA and AMD instruction.
    assert all(_instruction(case.reference) for case in cases)


@pytest.mark.parametrize("distribution", ["normal", "uniform"])
def test_all_cases_are_bitwise_equivalent(cases, distribution):
    failures = {}
    for index, case in enumerate(cases):
        prepared = prepare_case(
            case, batch=4, seed=1000 + index, distribution=distribution
        )
        result = validate(prepared)
        if not result["valid"]:
            failures[case.name] = result
    assert not failures


@pytest.mark.parametrize("length", [1024, 2048, 8192])
def test_lengths_and_repeat_calls(cases, length):
    import torch

    for case in cases:
        prepared = prepare_case(case, batch=2, seed=123, length=length)
        assert validate(prepared)["valid"], case.name
        assert torch.equal(prepared.fpy(), prepared.fpy()), case.name
        assert torch.equal(prepared.mmasim(), prepared.mmasim()), case.name


def test_native_chain_matches_individual_model_calls(cases):
    import torch

    for case in cases:
        length = 3 * case.k
        # Distinct rows and chunks detect lost carry, incorrect offsets, and
        # accidentally sharing accumulators between independent dot products.
        a = (
            torch.arange(2 * length, dtype=case.compute_dtype).reshape(2, length)
            / 128
        )
        b = torch.full_like(a, 0.5)
        scales = ()
        per_chunk = 0
        if case.scale_block:
            # Distinct power-of-two scales per row and block detect scale
            # offset errors as well as operand offset errors.
            per_chunk = case.k // case.scale_block
            count = length // case.scale_block
            exponents = torch.arange(2 * count).reshape(2, count) % 5 - 2
            scales = (2.0**exponents, 2.0 ** exponents.flip(-1))
        native = getattr(torch.ops.fpy2_models, case.op_name + "_dot")
        single = getattr(torch.ops.fpy2_models, case.op_name)
        actual = native(a, b, *scales)
        for row in range(2):
            acc = 0.0
            for chunk, offset in enumerate(range(0, length, case.k)):
                chunk_scales = tuple(
                    s[row : row + 1, chunk * per_chunk : (chunk + 1) * per_chunk]
                    for s in scales
                )
                if case.kind == "scaled":
                    chunk_scales = tuple(s.item() for s in chunk_scales)
                acc = single(
                    a[row : row + 1, offset : offset + case.k],
                    b[row : row + 1, offset : offset + case.k],
                    acc,
                    *chunk_scales,
                ).item()
            assert actual[row].item() == acc, case.name
        assert torch.equal(
            native(torch.zeros_like(a), b, *scales), torch.zeros(2, dtype=a.dtype)
        )
        with pytest.raises(RuntimeError):
            native(a[:, :-1], b[:, :-1], *scales)
        if scales:
            with pytest.raises(RuntimeError):
                native(a, b, *(s[:, :-1] for s in scales))
        with pytest.raises(ValueError):
            prepare_case(case, 1, 1, length=case.k + 1)


def test_summary_separates_lengths():
    from evaluation.benchmark_mmasim import enrich_measurements, parse_args, summarize

    rows = [
        {
            "design": "example",
            "implementation": "fpy-core",
            "batch_size": 2,
            "dot_length": length,
            "model_width": 8,
            "threads": 1,
            "seconds_per_call": 0.5,
        }
        for length in (1024, 2048)
    ]
    summaries = summarize(rows)
    assert len(summaries) == 2
    assert [row["terms_per_second"] for row in summaries] == [4096, 8192]
    enrich_measurements(rows, summaries)
    assert all(row["dots_per_second"] == 4 for row in rows)
    args = parse_args([])
    assert args.dot_lengths == [1024, 2048, 8192]
    assert args.batch_sizes == [1, 16, 256]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_validation_preserves_signed_zero_and_reports_first_mismatch(dtype):
    import torch

    from evaluation.benchmark_mmasim import Prepared

    actual = torch.tensor([1.0, 0.0], dtype=getattr(torch, dtype))
    expected = torch.tensor([1.0, -0.0], dtype=getattr(torch, dtype))
    result = validate(Prepared(lambda: actual, lambda: expected))
    assert not result["valid"]
    assert result["reason"] == "output bit mismatch"
    assert result["first_difference"] == 1
    assert int(result["fpy_bits"], 16) == 0
    assert int(result["mmasim_bits"], 16) == 1 << (31 if dtype == "float32" else 63)
