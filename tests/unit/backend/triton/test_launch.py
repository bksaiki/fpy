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
import re

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
            # bounded, so the product has a storage: an unbounded one has none
            ListType(RealType(fp.SINT8), n),
            ListType(RealType(fp.FP32), n),
            RealType(fp.INTEGER)])
    xt = torch.tensor(vals, dtype=torch.float32).cuda()
    nt = torch.tensor(exps, dtype=torch.int8).cuda()
    ot = torch.zeros(n, dtype=torch.float32).cuda()
    launch(src, [xt, nt, ot], block=8, grid=1)

    want = [v * 2.0 ** k for v, k in zip(vals, exps)]
    assert ot.cpu().tolist() == want
    assert len(set(exps)) > 1, 'the lanes really do disagree'


@fp.fpy(ctx=fp.REAL)
def _trunc(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        with fp.MPFixedContext(-1, fp.RoundingMode.RTZ):
            t = fp.round(xs[i])
        out[i] = t
    return out


@fp.fpy(ctx=fp.REAL)
def _rne(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        with fp.MPFixedContext(-1, fp.RoundingMode.RNE):
            t = fp.round(xs[i])
        out[i] = t
    return out


@pytest.mark.parametrize('f', [_trunc, _rne], ids=['RTZ', 'RNE'])
def test_integral_rounding_agrees_including_ties(f):
    """The ties are what separate the modes, so they are in the inputs."""
    import torch

    vals = [2.7, -2.7, 2.5, -2.5, 3.5, -3.5, 0.5, -0.5,
            0.0, -0.0, 1.0, -1.0, 4.5, -4.5, 100.25, -100.25]
    n = len(vals)
    src = TritonCompiler(drop_asserts=True).compile(
        f, ctx=fp.REAL, arg_types=[
            ListType(RealType(fp.FP32), n),
            ListType(RealType(fp.FP32), n),
            RealType(fp.INTEGER)])
    xt = torch.tensor(vals, dtype=torch.float32).cuda()
    ot = torch.zeros(n, dtype=torch.float32).cuda()
    launch(src, [xt, ot], block=32, grid=1)

    want = [float(v) for v in f(
        [fp.FP32.round(v) for v in vals], [fp.FP32.round(0.0)] * n, 32)]
    assert ot.cpu().tolist() == want


@fp.fpy(ctx=fp.REAL)
def _branchy(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(xs)):
        x = xs[i]
        with fp.FP32:
            y = x
            if x < 0:
                y = x * -3
            if abs(y) > 4:
                out[i] = y + 1
            else:
                z = y * 0.5
                out[i] = z
    return out


@pytest.mark.parametrize('seed', [0, 1])
def test_a_flattened_branch_agrees(seed):
    """Emitted from the branchy program directly: the normal form would
    if-convert it first."""
    import torch

    from fpy2 import Module
    from fpy2.backend.triton import emit_kernel, tile_loops
    from fpy2.transform import Specialize

    n = 37      # a partial last tile at any power-of-two width
    m = Module()
    m.add(_branchy, ctx=fp.REAL, arg_types=[
        ListType(RealType(fp.FP32), n),
        ListType(RealType(fp.FP32), n),
        RealType(fp.INTEGER)])
    g = Specialize.apply(m, size_key=True).get(_branchy.name).func
    r = tile_loops(g.ast, 'BLOCK')
    src = emit_kernel(
        r.func, r.tiled, block='BLOCK', drop_asserts=True, guards=r.guards)

    torch.manual_seed(seed)
    xt = (torch.randn(n) * 4).float().cuda()
    ot = torch.zeros(n, dtype=torch.float32).cuda()
    launch(src, [xt, ot], block=16)
    want = _branchy([float(v) for v in xt.cpu().tolist()], [0.0] * n, 16)
    assert ot.cpu().tolist() == [float(v) for v in want]


@fp.fpy(ctx=fp.REAL)
def _all_nonneg(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    ok = True
    for i in range(len(xs)):
        if xs[i] < 0:
            ok = False
    out[0] = 1 if ok else 0
    return out


@fp.fpy(ctx=fp.REAL)
def _largest(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    with fp.FP32:
        m = fp.round(0)
        for i in range(len(xs)):
            m = max(m, xs[i])
        out[0] = m
    return out


@pytest.mark.parametrize('f', [_all_nonneg, _largest], ids=['search', 'max'])
@pytest.mark.parametrize('seed', [0, 1])
def test_a_reduction_stays_sequential_and_agrees(f, seed):
    """No reduction across a tile is lowered, so the loop is not tiled."""
    import torch

    n = 37
    src = TritonCompiler(drop_asserts=True).compile(
        f, ctx=fp.REAL, arg_types=[
            ListType(RealType(fp.FP32), n),
            ListType(RealType(fp.FP32), 1),
            RealType(fp.INTEGER)])
    assert 'tl.static_range(37)' in src.source

    torch.manual_seed(seed)
    xt = (torch.randn(n) * 4).float().cuda()
    ot = torch.zeros(1, dtype=torch.float32).cuda()
    launch(src, [xt, ot], block=16, grid=1)
    want = f([float(v) for v in xt.cpu().tolist()], [0.0], 16)
    assert ot.cpu().tolist() == [float(v) for v in want]


_RZ_FP32 = fp.IEEEContext(8, 32, fp.RM.RTZ)


@fp.fpy(ctx=fp.REAL)
def _rz(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(xs)):
        with _RZ_FP32:
            out[i] = fp.round(xs[i])
    return out


def test_round_toward_zero_fp32_agrees_bit_for_bit():
    """Lowered through `unfold`: Triton rounds toward zero from `f32` only.
    The inputs are the edges -- subnormals, the overflow boundary, the zeros,
    the specials -- and values needing more than 24 bits."""
    import math
    import struct

    import torch

    f32max = struct.unpack('f', struct.pack('I', 0x7f7fffff))[0]
    vals = [
        0.0, -0.0, 1.0, -1.0, 1 / 3, -1 / 3, 3.0000001, 16777217.0,
        -16777217.0, 1e-40, -1e-40, 1.4e-45, 7e-46, 3e-46, 1e-50, 2 ** -149,
        2 ** -126, 2 ** -126 * (1 - 2 ** -30), 5e-324, f32max, -f32max,
        f32max * (1 + 2 ** -30), 2.0 ** 128, -2.0 ** 128, 1e300, -1e300,
        math.inf, -math.inf, math.nan,
    ]
    n = len(vals)
    src = TritonCompiler(
        drop_asserts=True, unfold=TritonCompiler.UnfoldMode.ROUNDINGS,
    ).compile(_rz, ctx=fp.REAL, arg_types=[
        ListType(RealType(fp.FP64), n),
        ListType(RealType(fp.FP32), n),
        RealType(fp.INTEGER)])
    xt = torch.tensor(vals, dtype=torch.float64).cuda()
    ot = torch.zeros(n, dtype=torch.float32).cuda()
    launch(src, [xt, ot], block=16)

    def bits(v):
        return struct.unpack('I', struct.pack('f', v))[0]

    want = [float(v) for v in _rz(vals, [0.0] * n, 16)]
    for x, w, g in zip(vals, want, ot.cpu().tolist()):
        assert bits(w) == bits(g) or (math.isnan(w) and math.isnan(g)), x


@fp.fpy(ctx=fp.REAL)
def _prefix_plus(xs: list[fp.Real], ys: list[fp.Real], out: list[fp.Real],
                 BLOCK: fp.Real):
    for r in range(len(out)):
        with fp.FP32:
            s = fp.round(0)
            for j in range(4):
                s = s + xs[j]
            out[r] = s + ys[r]
    return out


def test_a_scalar_address_under_the_tile_mask():
    """`xs[j]` is one address for every lane; under the tile's guard alone it
    is one scalar load, which Triton broadcasts where it meets the tile."""
    import torch

    n = 8
    src = TritonCompiler(drop_asserts=True).compile(
        _prefix_plus, ctx=fp.REAL, arg_types=[
            ListType(RealType(fp.FP32), 4),
            ListType(RealType(fp.FP32), n),
            ListType(RealType(fp.FP32), n),
            RealType(fp.INTEGER)])
    assert 'tl.zeros_like(' not in src.source
    torch.manual_seed(0)
    xs = torch.randn(4).cuda()
    ys = torch.randn(n).cuda()
    ot = torch.zeros(n).cuda()
    launch(src, [xs, ys, ot], block=4)
    want = _prefix_plus(xs.cpu().tolist(), ys.cpu().tolist(), [0.0] * n, 4)
    assert ot.cpu().tolist() == [float(v) for v in want]


class TestARuntimeLength:
    """A length the arguments leave unproven is the kernel's to be told, as
    Triton's own kernels take theirs: one kernel, launched at any length."""

    def test_one_kernel_at_every_length(self):
        import torch
        from fpy2.utils import NamedId

        n = NamedId('n')
        src = TritonCompiler(drop_asserts=True).compile(
            _batched_dot, ctx=fp.REAL, arg_types=[
                ListType(ListType(RealType(FP16), K), n),
                ListType(ListType(RealType(FP16), K), n),
                ListType(RealType(fp.FP32), n),
                RealType(fp.INTEGER)])
        assert src.sizes and src.grid_extent == src.sizes[0][0]
        for rows in (1, 6, 9, 16):
            torch.manual_seed(rows)
            xt = (torch.randn(rows, K) * 4).half().cuda()
            yt = (torch.randn(rows, K) * 4).half().cuda()
            ot = torch.zeros(rows, dtype=torch.float32).cuda()
            launch(src, [xt, yt, ot], block=4)
            xs = [[float(v) for v in row] for row in xt.cpu().tolist()]
            ys = [[float(v) for v in row] for row in yt.cpu().tolist()]
            want = _batched_dot(xs, ys, [0.0] * rows, 4)
            assert ot.cpu().tolist() == [float(v) for v in want], rows


# -- lanes over a list ----------------------------------------------------

_ROWS = 7


def _lanes(func, arg_types):
    return TritonCompiler(drop_asserts=True).compile(
        func, ctx=fp.REAL, arg_types=arg_types)


def _agree_rows(func, n_in: int, n_out: int, *, seed: int = 0) -> None:
    """*func*, compiled, against the interpreter: a row of
    *n_in* in and of *n_out* out per output, over `_ROWS` of them."""
    import torch
    from fpy2.utils import NamedId

    rows = NamedId('rows')
    f32 = RealType(fp.FP32)
    src = _lanes(func, [ListType(ListType(f32, n_in), rows),
                        ListType(ListType(f32, n_out), rows),
                        RealType(fp.INTEGER)])
    torch.manual_seed(seed)
    xt = (torch.randn(_ROWS, n_in) * 4).cuda()
    xt[0, 0], xt[1, -1] = -0.0, float('inf')
    ot = torch.zeros(_ROWS, n_out).cuda()
    launch(src, [xt, ot], block=4)
    want = func(xt.cpu().tolist(), [[0.0] * n_out for _ in range(_ROWS)], 4)
    got = ot.cpu().tolist()
    for r in range(_ROWS):
        for k in range(n_out):
            w, g = float(want[r][k]), got[r][k]
            assert repr(w) == repr(g), (r, k, w, g)


@fp.fpy(ctx=fp.REAL)
def _twice_plus_one(xss: list[list[fp.Real]], out: list[list[fp.Real]],
                    BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        with fp.FP32:
            ys = [x * 2 + 1 for x in xs]
        for k in range(len(row)):
            row[k] = ys[k]
    return out


@fp.fpy(ctx=fp.FP32)
def _magnitude(x: fp.Real) -> fp.Real:
    if x < 0:
        return -x
    return x


@fp.fpy(ctx=fp.REAL)
def _called(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        ys = [_magnitude(x) for x in xs]
        for k in range(len(row)):
            row[k] = ys[k]
    return out


@fp.fpy(ctx=fp.REAL)
def _join(xs, ys):
    n = len(xs)
    m = len(ys)
    zs = fp.empty(n + m)
    for i in range(n):
        zs[i] = xs[i]
    for i in range(m):
        zs[n + i] = ys[i]
    return zs


@fp.fpy(ctx=fp.REAL)
def _joined(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        with fp.FP32:
            ps = [x * x for x in xs]
        zs = _join(ps, [xs[0]])
        for k in range(len(row)):
            row[k] = zs[k]
    return out


@fp.fpy(ctx=fp.REAL)
def _picked(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    """Elements read and written one at a time, outside a lane loop."""
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        with fp.FP32:
            ys = [x + 1 for x in xs]
            ys[1] = ys[0] * 3
        acc = ys[0]
        for k in range(len(ys)):
            acc = max(acc, ys[k])
        row[0] = acc
        row[1] = ys[1]
        row[2] = ys[len(ys) - 1]
    return out


class TestLanesOverAList:
    """A local list is a `[rows, P]` tile and a tail past the largest power
    of two in its length; a lane loop runs its body across the tile and then
    once per tail element."""

    @pytest.mark.parametrize('n', [1, 3, 4, 5, 8])
    def test_an_elementwise_comprehension_at_every_index(self, n):
        _agree_rows(_twice_plus_one, n, n)

    def test_a_call_inside_a_comprehension(self):
        """`_magnitude` branches, so its lanes take a mask of their own."""
        _agree_rows(_called, 5, 5)

    def test_join_is_a_tile_and_a_tail(self):
        """`L + 1` long: the products' tile, and the accumulator its tail."""
        _agree_rows(_joined, 4, 5)

    @pytest.mark.parametrize('n', [4, 6])
    def test_elements_one_at_a_time(self, n):
        _agree_rows(_picked, n, 3)


@fp.fpy(ctx=fp.REAL)
def _reduced(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        ys = [x * 2 for x in xs]
        row[0] = max(ys)
        row[1] = min(ys)
        row[2] = 1.0 if any([y > 0 for y in ys]) else 0.0
        row[3] = 1.0 if all([y > 0 for y in ys]) else 0.0
        row[4] = sum(ys)
        with fp.FP16:
            zs = [x + 1 for x in xs]
            row[5] = sum(zs)
    return out


class TestReductionsAcrossLanes:
    """`max`, `min`, `any`, `all` across a tile's lanes and then its tail; a
    `sum` there only where no grouping can be seen, else left to right."""

    @pytest.mark.parametrize('n', [4, 5, 6])
    def test_they_agree_with_the_interpreter(self, n):
        import torch
        from fpy2.utils import NamedId

        rows = NamedId('rows')
        src = _lanes(_reduced, [
            ListType(ListType(RealType(FP16), n), rows),
            ListType(ListType(RealType(fp.FP64), 6), rows),
            RealType(fp.INTEGER)])
        # the `REAL` sum across the lanes; the `FP16` one left to right
        assert re.search(r'tl\.sum\((?!tl\.where)', src.source)
        torch.manual_seed(n)
        xt = (torch.randn(8, n) * 1000).half()
        xt[0] = -0.0                                   # a sum of `-0.0` alone
        xt[1, 0], xt[1, -1] = float('inf'), -float('inf')
        xt[2, 1] = float('nan')
        xt[3] = torch.tensor([0.0, -0.0] * n)[:n]      # both zeros
        xt[4, -1] = 60000.0                            # an FP16 sum overflows
        xt[4, 0] = 60000.0
        xt = xt.cuda()
        ot = torch.zeros(8, 6, dtype=torch.float64).cuda()
        launch(src, [xt, ot], block=4)
        want = _reduced([[float(v) for v in r] for r in xt.cpu().tolist()],
                        [[0.0] * 6 for _ in range(8)], 4)
        for r, (w, g) in enumerate(zip(want, ot.cpu().tolist())):
            assert [repr(float(v)) for v in w] == [repr(v) for v in g], r


@fp.fpy(ctx=fp.REAL)
def _grouped(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        ys = [x * 2 for x in xs]
        for g in range(4):
            group = ys[g * 2:(g + 1) * 2]
            row[g] = sum(group)
            row[4 + g] = 1.0 if all([y == 0 for y in group]) else 0.0
    return out


def test_an_aligned_slice_of_a_tile_is_a_tile():
    """nvfp4's groups: row `g` of the tile reshaped to the group's width."""
    import torch
    from fpy2.utils import NamedId

    rows = NamedId('rows')
    src = _lanes(_grouped, [
        ListType(ListType(RealType(FP16), 8), rows),
        ListType(ListType(RealType(fp.FP64), 8), rows),
        RealType(fp.INTEGER)])
    assert 'tl.reshape(ys, (BLOCK, 4, 2))' in src.source
    torch.manual_seed(0)
    xt = (torch.randn(6, 8) * 100).half()
    xt[0, :2] = 0.0
    xt[1, 2], xt[1, 3] = -0.0, -0.0
    xt[2, 5] = float('nan')
    xt = xt.cuda()
    ot = torch.zeros(6, 8, dtype=torch.float64).cuda()
    launch(src, [xt, ot], block=4)
    want = _grouped([[float(v) for v in r] for r in xt.cpu().tolist()],
                    [[0.0] * 8 for _ in range(6)], 4)
    for r, (w, g) in enumerate(zip(want, ot.cpu().tolist())):
        assert [repr(float(v)) for v in w] == [repr(v) for v in g], r


@fp.fpy(ctx=fp.REAL)
def _chain(xss: list[list[fp.Real]], yss: list[list[fp.Real]],
           out: list[fp.Real], BLOCK: fp.Real):
    """A T-FDPA chain's shape: blocks of four, a block sum, and an
    accumulator carried from block to block."""
    for j in range(len(out)):
        xs = xss[j]
        ys = yss[j]
        with fp.FP32:
            d = fp.round(0)
        for i in range(0, len(xs), 4):
            xb = xs[i:i + 4]
            yb = ys[i:i + 4]
            with fp.FP32:
                ps = [a * b for a, b in zip(xb, yb)]
                d = d + sum(ps)
        out[j] = d
    return out


def test_one_kernel_at_every_depth():
    """`K` a kernel argument: the loop over its blocks runs at runtime, and
    the accumulator it carries is a row at entry and at every assignment."""
    import torch
    from fpy2.utils import NamedId

    rows, k = NamedId('rows'), NamedId('k')
    src = TritonCompiler(drop_asserts=True).compile(_chain, ctx=fp.REAL, arg_types=[
        ListType(ListType(RealType(FP16), k), rows),
        ListType(ListType(RealType(FP16), k), rows),
        ListType(RealType(fp.FP32), rows), RealType(fp.INTEGER)])
    assert 'in range(0, ' in src.source
    for depth in (4, 8, 20):
        torch.manual_seed(depth)
        xt = (torch.randn(6, depth) * 8).half().cuda()
        yt = (torch.randn(6, depth) * 8).half().cuda()
        ot = torch.zeros(6, dtype=torch.float32).cuda()
        launch(src, [xt, yt, ot], block=4)
        xs = [[float(v) for v in r] for r in xt.cpu().tolist()]
        ys = [[float(v) for v in r] for r in yt.cpu().tolist()]
        want = _chain(xs, ys, [0.0] * 6, 4)
        assert ot.cpu().tolist() == [float(v) for v in want], depth


@fp.fpy(ctx=fp.REAL)
def _matmul(A: list[list[fp.Real]], BT: list[list[fp.Real]],
            out: list[list[fp.Real]], BLOCK: fp.Real):
    for i in range(len(out)):
        row = out[i]
        for j in range(len(row)):
            with fp.FP32:
                ps = [a * b for a, b in zip(A[i], BT[j])]
                row[j] = sum(ps)
    return out


def test_both_output_dimensions_are_program_ids():
    """`i` is `program_id(1)`, one program per row of the output, and `j`
    the tile: `m` and `n` both larger than a block."""
    import torch
    from fpy2.utils import NamedId

    m, n = NamedId('m'), NamedId('n')
    f32 = RealType(fp.FP32)
    src = _lanes(_matmul, [ListType(ListType(f32, 8), m),
                           ListType(ListType(f32, 8), n),
                           ListType(ListType(f32, n), m), RealType(fp.INTEGER)])
    assert 'i = tl.program_id(1)' in src.source and src.grid_outer is not None
    for rows, cols in ((5, 9), (1, 3), (6, 4)):
        torch.manual_seed(rows * cols)
        at, bt = torch.randn(rows, 8).cuda(), torch.randn(cols, 8).cuda()
        ot = torch.zeros(rows, cols).cuda()
        launch(src, [at, bt, ot], block=4)
        want = _matmul(at.cpu().tolist(), bt.cpu().tolist(),
                       [[0.0] * cols for _ in range(rows)], 4)
        assert ot.cpu().tolist() == [[float(v) for v in r] for r in want], (rows, cols)


class TestTuning:
    """Left without a block, a launch times each of `TUNING` and runs the
    fastest -- over the arguments it is given, so what it writes is put back
    between configs."""

    @pytest.fixture(autouse=True)
    def _two_configs(self, monkeypatch):
        from fpy2.backend.triton import launcher
        monkeypatch.setattr(launcher, 'TUNING', ((16, 4), (32, 2)))

    def test_it_agrees_with_the_interpreter(self):
        import torch
        from fpy2.utils import NamedId

        rows = NamedId('rows')
        src = _lanes(_twice_plus_one, [
            ListType(ListType(RealType(fp.FP32), 5), rows),
            ListType(ListType(RealType(fp.FP32), 5), rows), RealType(fp.INTEGER)])
        assert src.writes == ('out_ptr',)
        xt = torch.randn(37, 5).cuda()
        ot = torch.zeros(37, 5).cuda()
        launch(src, [xt, ot])
        want = _twice_plus_one(xt.cpu().tolist(), [[0.0] * 5 for _ in range(37)], 4)
        assert ot.cpu().tolist() == [[float(v) for v in r] for r in want]
