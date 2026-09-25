"""
The Triton kernels against the interpreter on hard test cases: zeros of both
signs, the smallest and largest magnitudes, the least normal, and the
specials each format has.  Needs a GPU; skipped without one.

    pytest tests/test_triton.py
"""

import functools
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')


@functools.cache
def _kernel(name: str):
    """`(kernel, design, arg_types)` for the design `name`."""
    import compile_triton as ct
    from compile import DESIGNS

    return ct.compile_matmul(dict(DESIGNS)[name], None)


@pytest.mark.parametrize('name', ['nv.volta.f16.f32', 'amd.cdna3.bf8'])
def test_hard_cases_agree(name: str) -> None:
    import compile_triton as ct

    kernel, design, arg_types = _kernel(name)
    m, n, trials = 4, 4, 8
    agree = ct.run_matmul(kernel, design, arg_types, m, n, trials, seed=0, hard_every=1)
    assert agree == m * n * trials


@pytest.mark.parametrize('name', ['nv.volta.f16.f32', 'nv.hopper.f16.f32'])
def test_all_negative_zeros_agree(name: str) -> None:
    """Every product and the accumulator `-0`: the terms are summed as
    integers, which have no `-0`, so the sign is the rounding's to give."""
    import compile_triton as ct
    import torch

    from fpy2.backend.triton import launch

    kernel, design, arg_types = _kernel(name)
    a, b, c = arg_types[:3]
    m = n = 2
    A = [[-0.0] * ct._length(a) for _ in range(m)]
    BT = [[0.0] * ct._length(b) for _ in range(n)]
    C = [[-0.0] * n for _ in range(m)]
    out = torch.zeros(m, n, dtype=ct._dtype(ct._fmt(c))).cuda()
    launch(kernel, [ct._tensor(A, a), ct._tensor(BT, b), ct._tensor(C, c), out], block=64)
    got = out.cpu().tolist()
    for i in range(m):
        for j in range(n):
            want = float(design(A[i], BT[j], C[i][j]))
            assert math.copysign(1.0, got[i][j]) == math.copysign(1.0, want) and got[i][j] == want
