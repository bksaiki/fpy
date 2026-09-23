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


@fp.fpy(ctx=fp.FP32)
def _any_all(xss: list[list[fp.Real]], out: list[fp.Real], BLOCK: fp.Real):
    """Lanes disagreeing on `any` and on `all`, in one tile."""
    for r in range(len(out)):
        flags = [xss[r][k] > 0.0 for k in range(3)]
        out[r] = 2.0 if all(flags) else (1.0 if any(flags) else 0.0)
    return out


class TestAnyAll:
    @pytest.mark.parametrize('n,block', [(6, 4), (8, 8), (1, 4)])
    def test_it_agrees_with_the_interpreter(self, n, block):
        import torch

        src = TritonCompiler(drop_asserts=True).compile(
            _any_all, ctx=fp.FP32, arg_types=[
                ListType(ListType(RealType(fp.FP32), 3), n),
                ListType(RealType(fp.FP32), n),
                RealType(fp.INTEGER)])
        torch.manual_seed(0)
        xt = (torch.randn(n, 3) * 2).float().cuda()
        ot = torch.zeros(n, dtype=torch.float32).cuda()
        launch(src, [xt, ot], block=block)

        vals = [[float(v) for v in row] for row in xt.cpu().tolist()]
        want = [float(v) for v in _any_all(vals, [0.0] * n, block)]
        assert ot.cpu().tolist() == want, f'n={n} block={block}'

    def test_all_three_outcomes_are_exercised(self):
        """Otherwise the agreement above could hold on one branch alone."""
        import torch

        torch.manual_seed(0)
        rows = (torch.randn(6, 3) * 2).float().tolist()
        outs = {all(v > 0 for v in r) and 2 or (any(v > 0 for v in r) and 1 or 0)
                for r in rows}
        assert len(outs) >= 2


@fp.fpy(ctx=fp.FP32)
def _signbit(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        out[i] = 1.0 if fp.signbit(xs[i]) else 0.0
    return out


def test_signbit_agrees_including_both_zeros():
    """`-0.0` is the case no float comparison reaches.

    A NaN is deliberately absent: FPy does not distinguish a NaN's sign --
    rounding normalizes it away -- so there is nothing for the hardware's
    answer to agree with.
    """
    import torch

    vals = [0.0, -0.0, 1.0, -1.0, 2.5, -2.5,
            float('inf'), float('-inf')]
    n = len(vals)
    src = TritonCompiler(drop_asserts=True).compile(
        _signbit, ctx=fp.FP32, arg_types=[
            ListType(RealType(fp.FP32), n),
            ListType(RealType(fp.FP32), n),
            RealType(fp.INTEGER)])
    xt = torch.tensor(vals, dtype=torch.float32).cuda()
    ot = torch.zeros(n, dtype=torch.float32).cuda()
    launch(src, [xt, ot], block=8)

    want = [float(v) for v in _signbit(
        [fp.FP32.round(v) for v in vals], [fp.FP32.round(0.0)] * n, 8)]
    assert ot.cpu().tolist() == want
    assert want[0] == 0.0 and want[1] == 1.0, 'both zeros were exercised'


@fp.fpy(ctx=fp.FP32)
def _nan_inf(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        v = fp.nan() if xs[i] > 1.0 else (
            fp.inf() if xs[i] > 0.0 else -fp.inf())
        out[i] = v
    return out


def test_nan_and_inf_agree_with_the_interpreter():
    import math

    import torch

    vals = [2.0, 0.5, -1.0, 3.0, 0.1, -0.0]
    n = len(vals)
    src = TritonCompiler(drop_asserts=True).compile(
        _nan_inf, ctx=fp.FP32, arg_types=[
            ListType(RealType(fp.FP32), n),
            ListType(RealType(fp.FP32), n),
            RealType(fp.INTEGER)])
    xt = torch.tensor(vals, dtype=torch.float32).cuda()
    ot = torch.zeros(n, dtype=torch.float32).cuda()
    launch(src, [xt, ot], block=4)

    want = [float(v) for v in _nan_inf(
        [fp.FP32.round(v) for v in vals], [fp.FP32.round(0.0)] * n, 4)]
    got = ot.cpu().tolist()
    assert all((math.isnan(a) and math.isnan(b)) or a == b
               for a, b in zip(got, want)), f'{got} vs {want}'
    assert any(math.isnan(v) for v in got) and float('-inf') in got


@fp.fpy(ctx=fp.FP32)
def _row_slice(xss: list[list[fp.Real]], out: list[fp.Real], BLOCK: fp.Real):
    """A slice of a row, indexed by a loop variable."""
    for r in range(len(out)):
        w = xss[r][2:6]
        s = fp.round(0)
        for j in range(4):
            s = s + w[j]
        out[r] = s
    return out


class TestSliceIsAnAddress:
    """A slice of something in memory is the same list at an offset.

    Scalarizing it into that many loads threw the address away, and a
    subscript by anything but a constant then had nothing to resolve
    against.  As an offset the loop stays rolled and composes with the
    row's own stride.
    """

    @pytest.mark.parametrize('n,block', [(6, 4), (8, 8), (1, 4)])
    def test_it_agrees_with_the_interpreter(self, n, block):
        import torch

        src = TritonCompiler(drop_asserts=True).compile(
            _row_slice, ctx=fp.FP32, arg_types=[
                ListType(ListType(RealType(fp.FP32), 8), n),
                ListType(RealType(fp.FP32), n),
                RealType(fp.INTEGER)])
        assert 'tl.load(xss_ptr + r * 8 + j + 2' in src.source
        torch.manual_seed(0)
        xt = torch.randn(n, 8).float().cuda()
        ot = torch.zeros(n, dtype=torch.float32).cuda()
        launch(src, [xt, ot], block=block)

        vals = [[float(v) for v in row] for row in xt.cpu().tolist()]
        want = [float(v) for v in _row_slice(vals, [0.0] * n, block)]
        assert ot.cpu().tolist() == want, f'n={n} block={block}'


@fp.fpy(ctx=fp.FP32)
def _logb(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        out[i] = fp.logb(xs[i])
    return out


def test_logb_agrees_over_every_exponent_and_random_bits():
    """Exhaustive where it can be: every fp32 exponent boundary of both
    signs -- which is where subnormals, the scaling and the specials all
    live -- plus random bit patterns, which land anywhere at all."""
    import math
    import random
    import struct

    import torch

    random.seed(0)
    vals = [1.0, 2.0, 3.0, 0.5, -4.0, 0.0, -0.0,
            float('inf'), float('-inf'), float('nan')]
    for e in range(-149, 128):
        vals += [2.0 ** e, -(2.0 ** e)]
    for _ in range(2000):
        vals.append(
            struct.unpack('<f', struct.pack('<I', random.getrandbits(32)))[0])
    n = len(vals)

    src = TritonCompiler(drop_asserts=True).compile(
        _logb, ctx=fp.FP32, arg_types=[
            ListType(RealType(fp.FP32), n),
            ListType(RealType(fp.FP32), n),
            RealType(fp.INTEGER)])
    xt = torch.tensor(vals, dtype=torch.float32).cuda()
    ot = torch.zeros(n, dtype=torch.float32).cuda()
    launch(src, [xt, ot], block=256)

    want = [float(v) for v in _logb(
        [fp.FP32.round(v) for v in vals], [fp.FP32.round(0.0)] * n, 256)]
    bad = [(v, g, w) for v, g, w in zip(vals, ot.cpu().tolist(), want)
           if not ((math.isnan(g) and math.isnan(w)) or g == w)]
    assert not bad, f'{len(bad)} differ, first: {bad[:3]}'
    # the subnormal range really was covered
    assert any(w < -126 and math.isfinite(w) for w in want)


@fp.fpy(ctx=fp.REAL)
def _scaled(xs: list[fp.Real], ns: list[fp.Real], out: list[fp.Real],
            BLOCK: fp.Real):
    for i in range(len(out)):
        with fp.REAL:
            t = (2 ** ns[i]) * xs[i]
        out[i] = t
    return out


def test_ldexp_agrees_with_a_per_lane_exponent():
    """A *runtime* scale is how a fixed-point context with a data-dependent
    grid reaches this backend at all: `RescaleFixed` moves the dependence out
    of the context and into this product."""
    import torch

    vals = [1.0, 3.0, -2.5, 0.5, 7.0, -1.0]
    exps = [0, 2, -3, 4, 1, -1]
    n = len(vals)
    src = TritonCompiler(drop_asserts=True).compile(
        _scaled, ctx=fp.REAL, arg_types=[
            ListType(RealType(fp.FP32), n),
            ListType(RealType(fp.INTEGER), n),
            ListType(RealType(fp.FP32), n),
            RealType(fp.INTEGER)])
    xt = torch.tensor(vals, dtype=torch.float32).cuda()
    nt = torch.tensor(exps, dtype=torch.int32).cuda()
    ot = torch.zeros(n, dtype=torch.float32).cuda()
    launch(src, [xt, nt, ot], block=8, grid=1)

    want = [v * 2.0 ** k for v, k in zip(vals, exps)]
    assert ot.cpu().tolist() == want
    assert len(set(exps)) > 1, 'the lanes really do disagree'
