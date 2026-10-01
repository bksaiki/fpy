from pathlib import Path

import pytest

from evaluation.benchmark_mmasim import (
    DEFAULT_MMASIM_ROOT,
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
    assert len(cases) == 14
    assert len({case.name for case in cases}) == len(cases)
    assert len({case.op_name for case in cases}) == len(cases)


@pytest.mark.parametrize("workload", ["normal", "stress"])
def test_all_cases_are_bitwise_equivalent(cases, workload):
    failures = {}
    for index, case in enumerate(cases):
        prepared = prepare_case(case, batch=4, workload=workload, seed=1000 + index)
        result = validate(prepared)
        if not result["valid"]:
            failures[case.name] = result
    assert not failures
