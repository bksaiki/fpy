"""What the launcher checks and passes, run against the interpreter.

Needs a GPU, as `test_launch.py` does; see there for `FPY_REQUIRE_GPU`.
"""

import os

import pytest

import fpy2 as fp
from fpy2.backend.triton import KernelSource, TritonCompiler, launch, unavailable
from fpy2.types import ListType, RealType
from fpy2.utils import NamedId

_WHY = unavailable()

if os.environ.get('FPY_REQUIRE_GPU') not in (None, '', '0') and _WHY is not None:
    raise RuntimeError(
        f'FPY_REQUIRE_GPU is set but the Triton runtime is unusable: {_WHY}'
    )

if _WHY is None:
    import torch

pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

_N = NamedId('n')
_R = RealType(fp.FP32)


@fp.fpy(ctx=fp.FP32)
def _floor(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(xs)):
        out[i] = fp.floor(xs[i])
    return out


@fp.fpy(ctx=fp.FP32)
def _scale(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real, c: fp.Real):
    for i in range(len(xs)):
        out[i] = xs[i] * c
    return out


def _compile(f: fp.Function, *extra: RealType) -> KernelSource:
    return TritonCompiler(drop_asserts=True).compile(
        f, ctx=fp.FP32,
        arg_types=[ListType(_R, _N), ListType(_R, _N), RealType(fp.INTEGER), *extra])


@pytest.mark.parametrize('block', [8, None], ids=['fixed', 'tuned'])
def test_libdevice_keeps_subnormals(block):
    """`floor` goes through libdevice, which flushes subnormals by default."""
    xs = [1e-45, -1e-45, 1e-40, -1e-40, -1.5]
    want = [float(v) for v in _floor(xs, [0.0] * len(xs), 0)]
    xt = torch.tensor(xs, dtype=torch.float32).cuda()
    ot = torch.zeros(len(xs), dtype=torch.float32).cuda()
    launch(_compile(_floor), [xt, ot], block=block)
    assert ot.cpu().tolist() == want
    assert want[1] == -1.0, 'a flushed input floors to -0'


@pytest.mark.parametrize('block', [8, None], ids=['fixed', 'tuned'])
def test_the_block_need_not_be_last(block):
    xt = torch.arange(8, dtype=torch.float32).cuda()
    ot = torch.zeros(8, dtype=torch.float32).cuda()
    launch(_compile(_scale, _R), [xt, ot, 3.0], block=block)
    assert ot.cpu().tolist() == [3.0 * i for i in range(8)]


def test_int32_offsets_refuse_a_larger_tensor():
    """Checked before the device is touched, so storage-less tensors do."""
    src = _compile(_floor)
    for n, block in ((2 ** 31, 8), (2 ** 31 - 4, 8)):
        t = torch.empty(n, device='meta')
        with pytest.raises(ValueError, match='int32'):
            launch(src, [t, t], block=block)


def test_a_tensor_of_another_dtype_is_refused():
    xt = torch.zeros(4, dtype=torch.float64).cuda()
    ot = torch.zeros(4, dtype=torch.float32).cuda()
    with pytest.raises(ValueError, match='compiled for torch.float32'):
        launch(_compile(_floor), [xt, ot], block=8)
