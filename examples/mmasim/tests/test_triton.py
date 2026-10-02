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


def _remixes():
    """A design of each structure the catalog has, with what no model has:
    NVIDIA's FP8 chain joined as CDNA3 joins, or flushing subnormals; a wider
    tree of exact products; CDNA2's tree rounding toward zero; an exact sum of
    FP8 rounded to E8M13; CDNA3's even-odd groups over NVIDIA's E4M3 at 13
    bits; MX scales at 13 bits; groups of 32; an FMA rounding toward zero;
    CDNA2's tree over 16; Hopper's FP8 and CDNA3's BF16 rounding to nearest
    below FP32's precision; trees toward zero, of FP8 to E8M13 and of exact
    BF16 products to FP32."""
    from dataclasses import replace

    from generator import Design
    from generator.catalog import CATALOG
    from models.nv import RZ_E8M13, RZ_FP32

    import fpy2 as fp

    hopper = CATALOG['nv.hopper.e4m3.f32']
    return {
        'hopper-separate': replace(hopper, join='separate', F2=20, e_zero=None),
        'hopper-ftz': replace(hopper, ftz=True),
        'tree-8-exact': Design(fp.BF16, fp.BF16, fp.FP32, 8, fp.FP32, 16, sum='tree'),
        'cdna2-rz': replace(CATALOG['amd.cdna2.f16'], rho=RZ_FP32),
        'exact-e8m13': Design(fp.MX_E4M3, fp.MX_E4M3, fp.FP32, 16, RZ_E8M13, 32, sum='exact'),
        'even-odd-e4m3': Design(fp.MX_E4M3, fp.MX_E4M3, fp.FP32, 16, fp.FP32, 32, F=13,
                                join='separate', groups='even-odd'),
        'mx-13': replace(CATALOG['nv.blackwell.mx.e4m3'], F=13, rho=RZ_E8M13),
        'nvfp4-g32': replace(CATALOG['nv.blackwell.nvfp4'], G=32),
        'fma-rz': replace(CATALOG['fp32 (fma)'], rho=RZ_FP32),
        'tree-16': replace(CATALOG['amd.cdna2.f16'], L=16, k=32),
        'hopper-rne': replace(hopper, rho=fp.IEEEContext(8, 22, fp.RM.RNE)),
        'cdna3-rne-7': replace(CATALOG['amd.cdna3.bf16'], rho=fp.IEEEContext(8, 15, fp.RM.RNE)),
        'hopper-tree': replace(hopper, sum='tree'),
        'ampere-tree': replace(CATALOG['nv.ampere.bf16.f32'], sum='tree'),
    }


REFUSED = {
    'exact-e8m13': '`sum` is exact, and no storage holds',
    'tree-8-exact': 'narrow tl.float64 to tl.float32',
}
"""The remixes the backend refuses, and why: no storage for an exact sum of
FP8 products and an FP32 accumulator (CDNA1's too), and a sum of exact BF16
products, held in FP64, rounded to nearest FP32: through FP64 it would round
twice, which toward zero composes and to nearest does not."""


@pytest.mark.parametrize('remix', list(REFUSED))
def test_a_backend_gap_is_refused(remix: str) -> None:
    """Refused at compile time, never computed wrong; a gap closed fails
    here, to move its remix to the agreeing ones."""
    import compile_triton as ct
    from generator import build

    from fpy2.backend.triton.emitter import TritonEmitError

    with pytest.raises(TritonEmitError, match=REFUSED[remix]):
        ct.compile_matmul(lambda: build(_remixes()[remix]), None)


@pytest.mark.parametrize('tile', [(16, 1), (64, 4)])
@pytest.mark.parametrize('remix', [r for r in _remixes() if r not in REFUSED])
def test_a_remix_agrees_with_the_interpreter(remix: str, tile: tuple[int, int]) -> None:
    """Every output bit for bit, a quarter of the draws hard cases, at two
    tiles: a tile changes the launch, not the result."""
    import compile_triton as ct
    from generator import build

    kernel, design, arg_types = ct.compile_matmul(lambda: build(_remixes()[remix]), None)
    m, n, trials = 4, 4, 8
    agree = ct.run_matmul(kernel, design, arg_types, m, n, trials, seed=0, hard_every=2,
                          block=tile[0], block_m=tile[1])
    assert agree == m * n * trials


@pytest.mark.parametrize('remix', ['hopper-separate', 'mx-13'])
def test_a_remix_fuses(remix: str) -> None:
    """Over a static `k` in one kernel: chained per instruction (scaled) or
    promoted every 128, as `serve` runs a scaled scheme."""
    import compile_triton as ct
    from generator import build

    ct.compile_fused(lambda: build(_remixes()[remix]), 256)
