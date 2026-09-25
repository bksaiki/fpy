"""
The Triton kernels against the interpreter on hard test cases: zeros of both
signs, the smallest and largest magnitudes, the least normal, and the
specials each format has.  Needs a GPU; skipped without one.

    pytest tests/test_triton.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')


@pytest.mark.parametrize('name', ['nv.volta.f16.f32', 'amd.cdna3.bf8'])
def test_hard_cases_agree(name: str) -> None:
    import compile_triton as ct
    from compile import DESIGNS

    kernel, design, arg_types = ct.compile_matmul(dict(DESIGNS)[name], None)
    m, n, trials = 4, 4, 8
    agree = ct.run_matmul(kernel, design, arg_types, m, n, trials, seed=0, hard_every=1)
    assert agree == m * n * trials
