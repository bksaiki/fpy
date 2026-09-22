"""Running an emitted kernel, and checking it against the interpreter.

**Everything here needs a GPU.**  Triton compiles for `amd` and `nvidia` only
-- there is no CPU target in mainline -- so `triton` importing is not enough;
`unavailable()` also checks for a device, and these skip without one.

The contract this closes: *if the compiler succeeds, the emitted code must
behave as the FPy interpreter does*.  Unlike the C++ harness, this one has an
empty non-correctly-rounded exclusion list -- the op table omits every
transcendental -- so every function it compiles it can check bit-for-bit.
"""

import os

import pytest

import fpy2 as fp
from fpy2.backend.triton import TritonCompiler, launch, unavailable
from fpy2.types import ListType, RealType

_WHY = unavailable()

_REQUIRE = os.environ.get('FPY_REQUIRE_GPU') not in (None, '', '0')
"""Turn a skip into a failure.

Without a GPU these tests skip, and a skip is indistinguishable from a pass in
a summary line -- so a broken install looks exactly like a machine with no
card, and this backend looks tested when nothing ran.  CI cannot fix that by
acquiring hardware, so anyone who *has* a card sets this and the skip becomes
an error instead.
"""

if _REQUIRE and _WHY is not None:
    raise RuntimeError(
        f'FPY_REQUIRE_GPU is set but the Triton runtime is unusable: {_WHY}'
    )

pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

FP16 = fp.IEEEContext(5, 16)
K = 8


@fp.fpy(ctx=fp.REAL)
def _batched_dot(xss: list[list[fp.Real]], yss: list[list[fp.Real]],
                 out: list[fp.Real], BLOCK: fp.Real):
    """FP16 in, exact products, FP32 accumulation -- the running example."""
    for r in range(len(xss)):
        acc = fp.round(0)
        for k in range(K):
            with fp.FP32:
                acc = acc + xss[r][k] * yss[r][k]
        out[r] = acc
    return out


def _compile(n: int):
    return TritonCompiler(drop_asserts=True).compile(
        _batched_dot, ctx=fp.REAL, arg_types=[
            ListType(ListType(RealType(FP16), K), n),
            ListType(ListType(RealType(FP16), K), n),
            ListType(RealType(fp.FP32), n),
            RealType(fp.INTEGER)])


def _run(n: int, block: int, seed: int = 0):
    """The emitted kernel and the interpreter, on the same inputs."""
    import torch

    src = _compile(n)
    torch.manual_seed(seed)
    xt = (torch.randn(n, K) * 4).half().cuda()
    yt = (torch.randn(n, K) * 4).half().cuda()
    ot = torch.zeros(n, dtype=torch.float32).cuda()
    launch(src, [xt, yt, ot], block=block)

    xs = [[float(v) for v in row] for row in xt.cpu().tolist()]
    ys = [[float(v) for v in row] for row in yt.cpu().tolist()]
    want = _batched_dot(xs, ys, [0.0] * n, block)
    return [float(v) for v in want], ot.cpu().tolist()


class TestDifferential:
    @pytest.mark.parametrize('n,block', [
        (8, 8),      # exactly one full tile
        (8, 4),      # two full tiles
        (6, 4),      # a partial tile: the mask does the work
        (1, 4),      # one element, most of the tile masked off
        (9, 4),      # two full tiles and a remainder of one
    ])
    def test_it_agrees_bit_for_bit(self, n, block):
        want, got = _run(n, block)
        assert got == want, f'n={n} block={block}'

    @pytest.mark.parametrize('seed', [0, 1, 2, 3])
    def test_it_agrees_across_inputs(self, seed):
        want, got = _run(6, 4, seed=seed)
        assert got == want


class TestLauncher:
    def test_the_grid_is_derived_from_the_proven_extent(self):
        assert _compile(6).grid_extent == 6

    def test_fusion_is_taken_from_the_kernel_not_the_caller(self):
        """Whether contracting a multiply-add is observable is a property of
        the program, so the launcher does not get to choose."""
        assert _compile(6).enable_fp_fusion

    def test_a_kernel_with_no_tile_needs_an_explicit_grid(self):
        from fpy2.backend.backend import CompileError
        from fpy2.backend.triton.emitter import KernelSource

        src = KernelSource(
            name='nothing', source='', params=(),
            grid_extent=None, enable_fp_fusion=False)
        with pytest.raises(CompileError, match='no extent'):
            launch(src, [], block=4)


@pytest.mark.skipif(False, reason='')
def test_the_guard_reports_a_reason(capsys):
    """Runs with or without hardware: it *is* the guard.

    It prints the reason so a CI log says why the rest did not run.  A silent
    skip is the failure mode that matters here -- the suite passes, the
    backend is untested, and nothing says so.
    """
    why = unavailable()
    with capsys.disabled():
        print(f'\n  [triton] {"runnable" if why is None else f"skipped: {why}"}')
    assert why is None or isinstance(why, str)


@fp.fpy(ctx=fp.REAL)
def _row_bound(xss: list[list[fp.Real]], yss: list[list[fp.Real]],
               out: list[fp.Real], BLOCK: fp.Real):
    """`_batched_dot` with the rows bound to names first.

    What inlining a call produces: a design takes its vector as a parameter,
    so the wrapper's `design(Ass[r], ...)` becomes `A = Ass[r]`.
    """
    for r in range(len(xss)):
        xs = xss[r]
        ys = yss[r]
        acc = fp.round(0)
        for k in range(K):
            with fp.FP32:
                acc = acc + xs[k] * ys[k]
        out[r] = acc
    return out


class TestRowBinding:
    """A row bound to a name used to emit a `_ptr` that was not a parameter.

    The kernel failed to JIT at all -- `NameError: xs_ptr is not defined` --
    so this is the regression net for it actually running.
    """

    @pytest.mark.parametrize('n,block', [(8, 8), (6, 4), (1, 4), (9, 4)])
    def test_it_agrees_with_the_interpreter(self, n, block):
        import torch

        src = TritonCompiler(drop_asserts=True).compile(
            _row_bound, ctx=fp.REAL, arg_types=[
                ListType(ListType(RealType(FP16), K), n),
                ListType(ListType(RealType(FP16), K), n),
                ListType(RealType(fp.FP32), n),
                RealType(fp.INTEGER)])
        torch.manual_seed(0)
        xt = (torch.randn(n, K) * 4).half().cuda()
        yt = (torch.randn(n, K) * 4).half().cuda()
        ot = torch.zeros(n, dtype=torch.float32).cuda()
        launch(src, [xt, yt, ot], block=block)

        xs = [[float(v) for v in row] for row in xt.cpu().tolist()]
        ys = [[float(v) for v in row] for row in yt.cpu().tolist()]
        want = [float(v) for v in _row_bound(xs, ys, [0.0] * n, block)]
        assert ot.cpu().tolist() == want, f'n={n} block={block}'

    def test_it_matches_the_inline_form(self):
        """Naming the row changes nothing: same loads, same order."""
        _, got_inline = _run(8, 4)
        import torch

        src = TritonCompiler(drop_asserts=True).compile(
            _row_bound, ctx=fp.REAL, arg_types=[
                ListType(ListType(RealType(FP16), K), 8),
                ListType(ListType(RealType(FP16), K), 8),
                ListType(RealType(fp.FP32), 8),
                RealType(fp.INTEGER)])
        torch.manual_seed(0)
        xt = (torch.randn(8, K) * 4).half().cuda()
        yt = (torch.randn(8, K) * 4).half().cuda()
        ot = torch.zeros(8, dtype=torch.float32).cuda()
        launch(src, [xt, yt, ot], block=4)
        assert ot.cpu().tolist() == got_inline
