from pathlib import Path

import pytest

from evaluation.performance.benchmark_mmasim import (
    DEFAULT_MMASIM_ROOT,
    build_cases,
    instruction,
    manifest,
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
    assert len(cases) == len(manifest()["models"]) == 60
    assert len({case.name for case in cases}) == len(cases)
    assert len({case.op_name for case in cases}) == len(cases)
    # validation.json records each NVIDIA and AMD instruction.
    assert all(instruction(case.reference) for case in cases)


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


def test_batched_calls_match_one_row_at_a_time(cases):
    import torch

    for case in cases:
        # Distinct rows, accumulators and scales detect row offset errors and
        # rows accidentally sharing an accumulator or scale.
        a = torch.arange(3 * case.k, dtype=case.compute_dtype).reshape(3, case.k) / 128
        b = torch.full_like(a, 0.5)
        c = torch.tensor([0.0, 1.0, -2.0], dtype=case.compute_dtype)
        scales = ()
        if case.scale_block:
            per_row = case.k // case.scale_block
            exponents = torch.arange(3 * per_row).reshape(3, per_row) % 5 - 2
            scales = (2.0**exponents, 2.0 ** exponents.flip(-1))
            scales = tuple(s.to(case.compute_dtype) for s in scales)
        dpa = getattr(torch.ops.fpy2_models, case.op_name)
        batched = dpa(a, b, c, *scales)
        for row in range(3):
            single = dpa(a[row], b[row], c[row], *(s[row] for s in scales))
            assert batched[row].item() == single.item(), case.name
        with pytest.raises(RuntimeError):
            dpa(a[:, :-1], b[:, :-1], c, *scales)
        with pytest.raises(RuntimeError):
            dpa(a, b, c[:-1], *scales)
        if scales:
            with pytest.raises(RuntimeError):
                dpa(a, b, c, *(s[:, :-1] for s in scales))
        with pytest.raises(ValueError):
            prepare_case(case, 1, 1, length=case.k + 1)


def test_summary_separates_lengths():
    from evaluation.performance.benchmark_mmasim import parse_args, summarize

    rows = [
        {
            "design": "example",
            "implementation": "fpy-core",
            "batch_size": 2,
            "dot_length": length,
            "threads": 1,
            "seconds_per_call": 0.5,
        }
        for length in (1024, 2048)
    ]
    summaries = summarize(rows)
    assert len(summaries) == 2
    assert [row["dot_length"] for row in summaries] == [1024, 2048]
    assert all(row["dots_per_second"] == 4 for row in summaries)
    args = parse_args([])
    assert args.dot_lengths == [256, 1024, 4096]
    assert args.batch_sizes == [1024, 4096, 16384]
    assert not args.compile


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_validation_preserves_signed_zero_and_reports_first_mismatch(dtype):
    import torch

    from evaluation.performance.benchmark_mmasim import Prepared

    actual = torch.tensor([1.0, 0.0], dtype=getattr(torch, dtype))
    expected = torch.tensor([1.0, -0.0], dtype=getattr(torch, dtype))
    result = validate(Prepared(lambda: actual, lambda: expected))
    assert not result["valid"]
    assert result["reason"] == "output bit mismatch"
    assert result["first_difference"] == 1
    assert int(result["fpy_bits"], 16) == 0
    assert int(result["mmasim_bits"], 16) == 1 << (31 if dtype == "float32" else 63)
