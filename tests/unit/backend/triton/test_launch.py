"""Running an emitted kernel, and checking it against the interpreter.

**Everything here needs a GPU.**  Triton compiles for `amd` and `nvidia` only,
so `triton` importing is not enough; `unavailable()` also checks for a device,
and these skip without one.

If the compiler succeeds, the kernel must behave as the FPy interpreter does,
bit for bit: the op table omits every transcendental, so nothing is exempt.
"""

import math
import os
import random
import re
import struct

import pytest

import fpy2 as fp
from fpy2 import Module
from fpy2.backend.backend import CompileError
from fpy2.backend.triton import (
    KernelSource,
    TritonCompiler,
    emit_kernel,
    launch,
    tile_loops,
    unavailable,
)
from fpy2.backend.triton.launcher import _torch_dtype
from fpy2.transform import Specialize
from fpy2.types import ListType, RealType
from fpy2.utils import NamedId

from .programs import (
    K,
    aligned_sum,
    any_all,
    arm_writes,
    batched_dot,
    carried_index,
    interleaved,
    logb,
    logb_clamped,
    logb_finite,
    logb_guarded,
    mask_rebound,
    nan_inf,
    pow2_logb,
    rare_arm,
    rare_cell,
    reversed_row,
    round_to_int,
    row_first,
    scale_joined,
    scale_rebound,
    scale_stored_over,
    scaled,
    signbit,
)

_WHY = unavailable()

_REQUIRE = os.environ.get('FPY_REQUIRE_GPU') not in (None, '', '0')
"""Turn a skip into a failure.

A skip is indistinguishable from a pass in a summary line, so a broken install
looks exactly like a machine with no card.  Anyone who *has* a card sets this
and the skip becomes an error instead.
"""

if _REQUIRE and _WHY is not None:
    raise RuntimeError(
        f'FPY_REQUIRE_GPU is set but the Triton runtime is unusable: {_WHY}'
    )

if _WHY is None:
    import torch

pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

FP16 = fp.IEEEContext(5, 16)
F32 = RealType(fp.FP32)
INT = RealType(fp.INTEGER)


def _compile(func: fp.Function, arg_types: list, *, ctx=fp.REAL, **opts) -> KernelSource:
    return TritonCompiler(drop_asserts=True, **opts).compile(
        func, ctx=ctx, arg_types=arg_types)


def _reprs(v) -> list | str:
    """*v*, nested, as the `repr` of each float: exact, and a NaN equals one."""
    return [_reprs(x) for x in v] if isinstance(v, list) else repr(float(v))


def _agree(src: KernelSource, func: fp.Function, tensors: list, *,
           block: int | None = 4, block_m: int = 1, grid: int | None = None) -> list:
    """Launches *src* on *tensors* and asserts the last, the output, is what
    *func* returns on the same inputs; returns that."""
    args = [t.cpu().tolist() for t in tensors]
    launch(src, tensors, block=block, block_m=block_m, grid=grid)
    want = func(*args, block)
    assert _reprs(tensors[-1].cpu().tolist()) == _reprs(want)
    return want


def _out_like(src: KernelSource, n: int, pos: int = 1) -> 'torch.Tensor':
    """*n* zeros in the storage of *src*'s list argument at *pos*."""
    return torch.zeros(n, dtype=_torch_dtype(dict(src.dtypes)[pos])).cuda()


def _f32(vals: list) -> 'torch.Tensor':
    return torch.tensor(vals, dtype=torch.float32).cuda()


def _halves(rows: int, cols: int, scale: float = 4) -> 'torch.Tensor':
    return (torch.randn(rows, cols) * scale).half().cuda()


def _dot_types(n: int | NamedId) -> list:
    return [ListType(ListType(RealType(FP16), K), n)] * 2 + [ListType(F32, n), INT]


def _dot_agrees(src: KernelSource, n: int, block: int, seed: int) -> None:
    torch.manual_seed(seed)
    _agree(src, batched_dot, [_halves(n, K), _halves(n, K), torch.zeros(n).cuda()],
           block=block)


class TestDifferential:
    @pytest.mark.parametrize('n,block', [
        (8, 8),      # exactly one full tile
        (8, 4),      # two full tiles
        (6, 4),      # a partial tile: the mask does the work
        (1, 4),      # one element, most of the tile masked off
        (9, 4),      # two full tiles and a remainder of one
    ])
    def test_it_agrees_bit_for_bit(self, n, block):
        _dot_agrees(_compile(batched_dot, _dot_types(n)), n, block, 0)

    @pytest.mark.parametrize('seed', [0, 1, 2, 3])
    def test_it_agrees_across_inputs(self, seed):
        _dot_agrees(_compile(batched_dot, _dot_types(6)), 6, 4, seed)

    def test_one_kernel_at_every_length(self):
        """A length left unproven is a kernel argument, as in Triton's own."""
        src = _compile(batched_dot, _dot_types(NamedId('n')))
        assert src.sizes and src.grid_extent == src.sizes[0][0]
        for rows in (1, 6, 9, 16):
            _dot_agrees(src, rows, 4, rows)


def test_a_kernel_with_no_tile_needs_an_explicit_grid():
    src = KernelSource(
        name='nothing', source='', params=(),
        grid_extent=None, enable_fp_fusion=False)
    with pytest.raises(CompileError, match='no extent'):
        launch(src, [], block=4)


@pytest.mark.parametrize('n,block', [(6, 4), (8, 8), (1, 4)])
def test_any_and_all_agree(n, block):
    src = _compile(any_all, [ListType(ListType(F32, 3), n), ListType(F32, n), INT],
                   ctx=fp.FP32)
    torch.manual_seed(0)
    want = _agree(src, any_all, [(torch.randn(n, 3) * 2).cuda(), torch.zeros(n).cuda()],
                  block=block)
    if n == 8:
        assert len({float(v) for v in want}) == 3, 'every outcome is exercised'


def _map_types(n: int, ctx=fp.FP32) -> list:
    """An `xs`, an `out`, and a block."""
    return [ListType(RealType(ctx), n), ListType(F32, n), INT]


def test_signbit_agrees_including_both_zeros():
    """A NaN is absent: FPy does not distinguish a NaN's sign."""
    vals = [0.0, -0.0, 1.0, -1.0, 2.5, -2.5, math.inf, -math.inf]
    src = _compile(signbit, _map_types(len(vals)), ctx=fp.FP32)
    want = _agree(src, signbit, [_f32(vals), torch.zeros(len(vals)).cuda()], block=8)
    assert want[0] == 0.0 and want[1] == 1.0, 'both zeros were exercised'


def test_nan_and_inf_agree_with_the_interpreter():
    vals = [2.0, 0.5, -1.0, 3.0, 0.1, -0.0]
    src = _compile(nan_inf, _map_types(len(vals)), ctx=fp.FP32)
    want = [float(v) for v in _agree(
        src, nan_inf, [_f32(vals), torch.zeros(len(vals)).cuda()])]
    assert any(math.isnan(v) for v in want) and -math.inf in want


@fp.fpy(ctx=fp.FP32)
def _row_slice(xss: list[list[fp.Real]], out: list[fp.Real], BLOCK: fp.Real):
    for r in range(len(out)):
        w = xss[r][2:6]
        s = fp.round(0)
        for j in range(4):
            s = s + w[j]
        out[r] = s
    return out


@pytest.mark.parametrize('n,block', [(6, 4), (8, 8), (1, 4)])
def test_a_slice_is_an_offset(n, block):
    """A slice of a row in memory is the row at an offset, so a loop over it
    stays rolled and composes with the row's own stride."""
    src = _compile(_row_slice, [ListType(ListType(F32, 8), n), ListType(F32, n), INT],
                   ctx=fp.FP32)
    assert 'tl.load(xss_ptr + r * 8 + j + 2' in src.source
    torch.manual_seed(0)
    _agree(src, _row_slice, [torch.randn(n, 8).cuda(), torch.zeros(n).cuda()],
           block=block)


def test_logb_agrees_over_every_exponent_and_random_bits():
    """Every fp32 exponent boundary of both signs -- where subnormals, the
    scaling and the specials live -- plus random bit patterns."""
    rng = random.Random(0)
    vals = [1.0, 2.0, 3.0, 0.5, -4.0, 0.0, -0.0, math.inf, -math.inf, math.nan]
    for e in range(-149, 128):
        vals += [2.0 ** e, -(2.0 ** e)]
    vals += [struct.unpack('<f', struct.pack('<I', rng.getrandbits(32)))[0]
             for _ in range(2000)]
    src = _compile(logb, _map_types(len(vals)), ctx=fp.FP32)
    want = _agree(src, logb, [_f32(vals), torch.zeros(len(vals)).cuda()], block=256)
    assert any(w < -126 and math.isfinite(w) for w in want), 'subnormals covered'


def test_ldexp_agrees_with_a_per_lane_exponent():
    """A runtime scale is how a fixed-point context with a data-dependent grid
    reaches this backend: `RescaleFixed` moves it into this product."""
    vals = [1.0, 3.0, -2.5, 0.5, 7.0, -1.0]
    exps = [0, 2, -3, 4, 1, -1]
    n = len(vals)
    # a bounded exponent, so the product has a storage
    src = _compile(scaled, [ListType(F32, n), ListType(RealType(fp.SINT8), n),
                            ListType(F32, n), INT])
    xt, nt = _f32(vals), torch.tensor(exps, dtype=torch.int8).cuda()
    # the product is exact, so the kernel stores it as `float64`
    ot = torch.zeros(n, dtype=torch.float64).cuda()
    _agree(src, scaled, [xt, nt, ot], block=8, grid=1)
    assert ot.cpu().tolist() == [v * 2.0 ** k for v, k in zip(vals, exps)]


@pytest.mark.parametrize('rm', [fp.RM.RTZ, fp.RM.RNE], ids=['RTZ', 'RNE'])
def test_integral_rounding_agrees_including_ties(rm):
    """The ties are what separate the modes, so they are in the inputs."""
    vals = [2.7, -2.7, 2.5, -2.5, 3.5, -3.5, 0.5, -0.5,
            0.0, -0.0, 1.0, -1.0, 4.5, -4.5, 100.25, -100.25]
    f = round_to_int(rm)
    src = _compile(f, _map_types(len(vals)))
    _agree(src, f, [_f32(vals), torch.zeros(len(vals)).cuda()], block=32, grid=1)


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
    n = 37      # a partial last tile at any power-of-two width
    m = Module()
    m.add(_branchy, ctx=fp.REAL, arg_types=_map_types(n))
    g = Specialize.apply(m, size_key=True).get(_branchy.name).func
    r = tile_loops(g.ast, 'BLOCK')
    src = emit_kernel(
        r.func, r.tiled, block='BLOCK', drop_asserts=True, guards=r.guards)
    torch.manual_seed(seed)
    _agree(src, _branchy, [(torch.randn(n) * 4).cuda(), torch.zeros(n).cuda()],
           block=16)


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
    n = 37
    src = _compile(f, [ListType(F32, n), ListType(F32, 1), INT])
    assert 'tl.static_range(37)' in src.source
    torch.manual_seed(seed)
    _agree(src, f, [(torch.randn(n) * 4).cuda(), torch.zeros(1).cuda()],
           block=16, grid=1)


@pytest.mark.parametrize('rctx', [
    fp.IEEEContext(8, 32, fp.RM.RTZ),
    fp.IEEEContext(8, 32, fp.RM.RTN),
    fp.IEEEContext(8, 32, fp.RM.RTP),
    fp.IEEEContext(8, 22, fp.RM.RTZ),
    fp.IEEEContext(8, 16, fp.RM.RTZ),
], ids=['rtz', 'rtn', 'rtp', 'e8m13-rtz', 'bf16-rtz'])
def test_a_directed_round_from_f64_agrees_bit_for_bit(rctx):
    """libdevice's directed conversion into `f32`, then truncation onto a
    narrower format with its exponents.  The inputs are the edges --
    subnormals, the overflow boundary, the zeros, the specials -- and values
    needing more than 24 bits."""
    @fp.fpy(ctx=fp.REAL)
    def rnd(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
        for i in range(len(xs)):
            with rctx:
                out[i] = fp.round(xs[i])
        return out

    f32max = struct.unpack('f', struct.pack('I', 0x7f7fffff))[0]
    vals = [
        0.0, -0.0, 1.0, -1.0, 1 / 3, -1 / 3, 3.0000001, 16777217.0,
        -16777217.0, 1e-40, -1e-40, 1.4e-45, 7e-46, 3e-46, 1e-50, 2 ** -149,
        2 ** -126, 2 ** -126 * (1 - 2 ** -30), 5e-324, f32max, -f32max,
        f32max * (1 + 2 ** -30), 2.0 ** 128, -2.0 ** 128, 1e300, -1e300,
        math.inf, -math.inf, math.nan,
    ]
    src = _compile(rnd, _map_types(len(vals), fp.FP64))
    assert 'libdevice.double2float' in src.source
    xt = torch.tensor(vals, dtype=torch.float64).cuda()
    _agree(src, rnd, [xt, torch.zeros(len(vals)).cuda()], block=16)


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
    n = 8
    src = _compile(_prefix_plus, [ListType(F32, 4), ListType(F32, n), ListType(F32, n), INT])
    assert 'tl.zeros_like(' not in src.source
    torch.manual_seed(0)
    _agree(src, _prefix_plus, [torch.randn(4).cuda(), torch.randn(n).cuda(),
                               torch.zeros(n).cuda()])


# -- lanes over a list ----------------------------------------------------

def _agree_on(func: fp.Function, xt: 'torch.Tensor', n_out: int, *, ctx_in=fp.FP32,
              ctx_out=fp.FP32, n_in: int | NamedId | None = None, **opts) -> str:
    """*func*, compiled with *opts*, against the interpreter on the rows of
    *xt*; its source, for a caller that checks the spelling."""
    dtypes = {FP16: torch.float16, fp.FP32: torch.float32,
              fp.FP64: torch.float64, fp.SINT32: torch.int32}
    rows = NamedId('rows')
    src = _compile(func, [
        ListType(ListType(RealType(ctx_in), n_in or xt.shape[1]), rows),
        ListType(ListType(RealType(ctx_out), n_out), rows), INT], **opts)
    xt = xt.to(dtypes[ctx_in]).cuda()
    ot = torch.zeros(xt.shape[0], n_out, dtype=dtypes[ctx_out]).cuda()
    # a kernel that tiled nothing loops over the rows itself
    _agree(src, func, [xt, ot], grid=None if src.grid_extent else 1)
    return src.source


def _rows(n: int) -> 'torch.Tensor':
    """Seven rows of *n*, with a `-0.0` and an infinity."""
    torch.manual_seed(0)
    xt = torch.randn(7, n) * 4
    xt[0, 0], xt[1, -1] = -0.0, math.inf
    return xt


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
        _agree_on(_twice_plus_one, _rows(n), n)

    def test_a_call_inside_a_comprehension(self):
        """`_magnitude` branches, so its lanes take a mask of their own."""
        _agree_on(_called, _rows(5), 5)

    def test_join_is_a_tile_and_a_tail(self):
        """`L + 1` long: the products' tile, and the accumulator its tail."""
        _agree_on(_joined, _rows(4), 5)

    @pytest.mark.parametrize('n', [4, 6])
    def test_elements_one_at_a_time(self, n):
        _agree_on(_picked, _rows(n), 3)


@fp.fpy(ctx=fp.REAL)
def _reduced(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        ys = [x * 2 for x in xs]
        row[0] = max(ys)
        row[1] = min(ys)
        row[2] = 1.0 if any([y > 0 for y in ys]) else 0.0  # noqa: C419
        row[3] = 1.0 if all([y > 0 for y in ys]) else 0.0  # noqa: C419
        row[4] = sum(ys)
        with fp.FP16:
            zs = [x + 1 for x in xs]
            row[5] = sum(zs)
    return out


@pytest.mark.parametrize('n', [4, 5, 6])
def test_reductions_across_lanes_agree(n):
    """`max`, `min`, `any`, `all` across a tile's lanes and then its tail; a
    `sum` there only where no grouping can be seen, else left to right."""
    torch.manual_seed(n)
    xt = torch.randn(8, n) * 1000
    xt[0] = -0.0                                   # a sum of `-0.0` alone
    xt[1, 0], xt[1, -1] = math.inf, -math.inf
    xt[2, 1] = math.nan
    xt[3] = torch.tensor([0.0, -0.0] * n)[:n]      # both zeros
    xt[4, -1] = 60000.0                            # an FP16 sum overflows
    xt[4, 0] = 60000.0
    src = _agree_on(_reduced, xt, 6, ctx_in=FP16, ctx_out=fp.FP64)
    # the `REAL` sum across the lanes; the `FP16` one left to right
    assert re.search(r'tl\.sum\((?!tl\.where)', src)


@fp.fpy(ctx=fp.REAL)
def _grouped(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        ys = [x * 2 for x in xs]
        for g in range(4):
            group = ys[g * 2:(g + 1) * 2]
            row[g] = sum(group)
            row[4 + g] = 1.0 if all([y == 0 for y in group]) else 0.0  # noqa: C419
    return out


def test_an_aligned_slice_of_a_tile_is_a_tile():
    """nvfp4's groups: row `g` of the tile reshaped to the group's width."""
    torch.manual_seed(0)
    xt = torch.randn(6, 8) * 100
    xt[0, :2] = 0.0
    xt[1, 2], xt[1, 3] = -0.0, -0.0
    xt[2, 5] = math.nan
    src = _agree_on(_grouped, xt, 8, ctx_in=FP16, ctx_out=fp.FP64)
    assert 'tl.reshape(ys, (BLOCK, 4, 2))' in src


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
    rows, k = NamedId('rows'), NamedId('k')
    src = _compile(_chain, [ListType(ListType(RealType(FP16), k), rows)] * 2
                   + [ListType(F32, rows), INT])
    assert 'in range(0, ' in src.source
    for depth in (4, 8, 20):
        torch.manual_seed(depth)
        _agree(src, _chain, [_halves(6, depth, 8), _halves(6, depth, 8),
                             torch.zeros(6).cuda()])


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


@pytest.fixture(scope='module')
def matmul_src() -> KernelSource:
    m, n = NamedId('m'), NamedId('n')
    return _compile(_matmul, [ListType(ListType(F32, 8), m), ListType(ListType(F32, 8), n),
                              ListType(ListType(F32, n), m), INT])


def test_both_output_dimensions_are_program_ids(matmul_src):
    """With a lane loop, `i` is `program_id(1)`, one program per row of the
    output, and `j` the tile: `m` and `n` both larger than a block."""
    assert 'i = tl.program_id(1)\n' in matmul_src.source
    assert matmul_src.grid_outer is not None and matmul_src.block_m is None
    for rows, cols in ((5, 9), (1, 3), (6, 4)):
        torch.manual_seed(rows * cols)
        # a tile height is ignored where there is no tile of rows
        _agree(matmul_src, _matmul, [torch.randn(rows, 8).cuda(), torch.randn(cols, 8).cuda(),
                                     torch.zeros(rows, cols).cuda()], block_m=4)


@fp.fpy(ctx=fp.FP32)
def _accumulate(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        for k in range(len(row)):
            row[k] = row[k] + xs[k]
    return out


def test_tuning_restores_the_output_between_configs(monkeypatch):
    """Left without a block, a launch times each of `TUNING` and runs the
    fastest over the arguments it is given, so an output it reads is put back
    between configs and the update lands once."""
    from fpy2.backend.triton import launcher
    monkeypatch.setattr(launcher, 'TUNING', ((16, 4), (32, 2)))
    rows = NamedId('rows')
    src = _compile(_accumulate, [ListType(ListType(F32, 5), rows)] * 2 + [INT])
    assert src.writes == ('out_ptr',)
    torch.manual_seed(0)
    _agree(src, _accumulate, [torch.randn(37, 5).cuda(), torch.randn(37, 5).cuda()],
           block=None)


class TestLayout:
    """The kernel addresses each list row major from the shape it was compiled
    for, so a tensor laid out any other way is refused rather than read and
    written in the wrong places."""

    @staticmethod
    def _args(m: int = 3, n: int = 5, k: int = 8) -> list:
        return [torch.randn(m, k).cuda(), torch.randn(n, k).cuda(),
                torch.zeros(m, n).cuda()]

    def test_a_strided_view_is_refused(self, matmul_src):
        a, bt, _ = self._args()
        out = torch.zeros(3, 10).cuda()[:, :5]
        with pytest.raises(ValueError, match='`out_ptr` is not contiguous'):
            launch(matmul_src, [a, bt, out], block=4)

    def test_a_proven_length_must_match(self, matmul_src):
        with pytest.raises(ValueError, match='compiled for 8'):
            launch(matmul_src, self._args(k=16), block=4)

    def test_a_shared_length_must_agree(self, matmul_src):
        """`out`'s rows are `A`'s, one parameter read off `A`."""
        a, bt, _ = self._args()
        with pytest.raises(ValueError, match='where the kernel has one length'):
            launch(matmul_src, [a, bt, torch.zeros(4, 5).cuda()], block=4)

    def test_the_rank_must_match(self, matmul_src):
        a, bt, _ = self._args()
        with pytest.raises(ValueError, match='has 1 dimensions, not 2'):
            launch(matmul_src, [a, bt, torch.zeros(15).cuda()], block=4)

    def test_a_list_takes_a_tensor(self, matmul_src):
        _, bt, out = self._args()
        with pytest.raises(TypeError, match='takes a tensor'):
            launch(matmul_src, [[[0.0] * 8] * 3, bt, out], block=4)


# -- edge cases -----------------------------------------------------------

@fp.fpy(ctx=fp.REAL)
def _carried_max(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        acc = xs[0]
        for i in range(len(xs)):
            acc = max(acc, xs[i])
        row[0] = acc
    return out


def test_a_carried_negative_zero_stays_negative():
    """A runtime loop's value, broadcast to a row: `-0.0 + 0.0` is `+0.0`."""
    _agree_on(_carried_max, torch.tensor([[-0.0, -0.0, -0.0], [-1.0, -0.0, -3.0]]),
              1, n_in=NamedId('k'))


@fp.fpy(ctx=fp.REAL)
def _whole_tile(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        ys = [x * 2 for x in xs]
        zs = ys[0:4]
        zs[0] = 7
        with fp.FP32:
            row[0] = sum(zs)
        row[1] = ys[0]
        row[2] = ys[5]
    return out


def test_a_slice_the_whole_tile_wide_is_a_copy_without_the_tail():
    _agree_on(_whole_tile, torch.tensor([[1.0, 2, 3, 4, 100, 200]]), 3, ctx_in=FP16)


@fp.fpy(ctx=fp.REAL)
def _stale_index(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        row = out[j]
        t = j
        acc = xss[0][0] * 0
        for i in range(2):
            xs = xss[t]
            t = 0
            with fp.FP32:
                acc = acc + xs[i]
        row[0] = acc
    return out


def test_a_row_keeps_the_index_it_was_bound_at():
    """`xs[1]` is row `j`'s, not row `0`'s: `t` changed after the binding."""
    _agree_on(_stale_index, torch.tensor([[1.0, 2], [3.0, 4], [5.0, 6]]), 1)


@fp.fpy(ctx=fp.REAL)
def _destructured(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        a = xs[0] * xs[0]
        b = xs[0]
        c = a
        if xs[1] > 0:
            a, b = (xs[1], xs[2])
            with fp.FP32:
                c = a * a
        row[0] = c
        row[1] = a
        row[2] = b
    return out


def test_a_destructured_name_is_held_in_its_class():
    """`a` is an exact `f16` product, so `f32`: `a * a` rounds there, not in
    the `f16` a load would leave it in."""
    _agree_on(_destructured, torch.tensor([[1.0, 1.0009765625, 3.0], [2.0, -1.0, 3.0]]),
              3, ctx_in=FP16)


@fp.fpy(ctx=fp.REAL)
def _int_compare(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        row[0] = 1.0 if xs[0] > 16777216.5 else 0.0
        row[1] = 1.0 if xs[0] == xs[1] * 0.5 else 0.0
    return out


def test_an_integer_compares_with_a_float_exactly():
    """Not in `fp32`, where `16777217` is `16777216`."""
    _agree_on(_int_compare, torch.tensor([[16777217, 33554434], [16777216, 5]]),
              2, ctx_in=fp.SINT32)


@fp.fpy(ctx=fp.REAL)
def _wide_literal(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        row = out[j]
        row[0] = fp.rational(4503599627370497, 4503599627370496)
    return out


def test_a_literal_fp32_cannot_hold_is_typed():
    _agree_on(_wide_literal, torch.tensor([[1.0], [-1.0]]), 1, ctx_out=fp.FP64)


@fp.fpy(ctx=fp.REAL)
def _rounded_literal(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        with FP16:
            z = fp.round(fp.rational(12582911, 4194304))
        with fp.FP32:
            row[0] = xs[0] + z
    return out


def test_a_rounded_literal_is_rounded():
    """Retyping it would give `2.9999998`, where `FP16` rounds to `3`."""
    _agree_on(_rounded_literal, torch.tensor([[0.0], [1.0]]), 1, optimize=False)


@fp.fpy(ctx=fp.REAL)
def _to_integer(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        with fp.INTEGER:
            t = fp.round(xs[0])
        row[0] = t
    return out


def test_rounding_to_the_integers_has_no_negative_zero():
    _agree_on(_to_integer, torch.tensor([[-0.5], [-0.25], [-0.0], [-1.5], [0.5]]), 1)


@fp.fpy(ctx=fp.REAL)
def _to_e4m3(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        with fp.MX_E4M3:
            row[0] = fp.round(xs[0])
    return out


def test_an_unfolded_rounding_with_a_nan_compiles_and_agrees():
    """The unfolded rounding negates a NaN, which as a Python float has no
    `.to` to cast it with."""
    vals = [0.0, -0.0, 1.0, -1.0, 448.0, 464.0, 1e6, -1e6, math.inf, 2**-9,
            3 * 2**-10, -2**-10, 0.3, 17.0, 19.0, 1.1875]
    _agree_on(_to_e4m3, torch.tensor([[v] for v in vals]), 1,
              unfold=TritonCompiler.UnfoldMode.ROUNDINGS)


@pytest.mark.parametrize('rm', [fp.RM.RTZ, fp.RM.RNE, fp.RM.RNA, fp.RM.RTN, fp.RM.RTP])
def test_an_aligned_sum_agrees_on_hard_cases(rm: fp.RM) -> None:
    """Zeros, subnormals and the largest magnitudes put the grid anywhere in
    its range, a row of zeros at its lowest; one half and just below it where
    the scale is largest."""
    hard = [0.0, -0.0, 2.0 ** -149, -(2.0 ** -149), 2.0 ** -126, 3.4028234663852886e38,
            -3.4028234663852886e38, 1.0, -1.5, 2.0 ** -140 * 3, 1e-30, -7e20]
    rng = random.Random(0)
    half = (0.5 - 2.0 ** -25) * 2.0 ** 104
    rows = [[0.0, -0.0, 0.0, 0.0], [2.0 ** -149, 0.0, -(2.0 ** -148), 0.0],
            # scaled down by 2 ** 104: one half, and just below it
            [2.0 ** 127, half, -half, 0.5 * 2.0 ** 104],
            # scaled up by 2 ** 149
            [2.0 ** -126, 3 * 2.0 ** -149, 2.0 ** -149, -(2.0 ** -148)]]
    rows += [[rng.choice(hard) if rng.random() < 0.5 else rng.uniform(-100, 100)
              for _ in range(4)] for _ in range(60)]
    f = aligned_sum(rm)
    n = len(rows)
    src = _compile(f, [ListType(ListType(F32, 4), n), ListType(RealType(fp.FP64), n), INT],
                   unfold=TritonCompiler.UnfoldMode.ROUNDINGS)
    _agree(src, f, [torch.tensor(rows, dtype=torch.float32).cuda(), _out_like(src, n)], block=64)


_LOGB_HARD = {
    'fp16': [0.0, -0.0, 2.0 ** -24, -(2.0 ** -24), 2.0 ** -14 * (1 - 2 ** -10), 2.0 ** -14,
             65504.0, -65504.0, math.inf, -math.inf, math.nan, 1.0, -3.5, 1000.0],
    'fp32': [0.0, -0.0, 2.0 ** -149, -(2.0 ** -149), 2.0 ** -126 * (1 - 2 ** -23), 2.0 ** -126,
             3.4028234663852886e38, -3.4028234663852886e38, math.inf, -math.inf, math.nan,
             1.0, -3.5, 1e30],
}


@pytest.mark.parametrize('prog, c', [
    (logb_clamped, -14), (logb_clamped, -20), (logb_clamped, -126), (logb_clamped, -140),
    (logb_guarded, -14), (logb_guarded, -126), (logb_finite, None),
])
@pytest.mark.parametrize('fmt', ['fp16', 'fp32'])
def test_logb_agrees_on_hard_cases(prog, c: int | None, fmt: str) -> None:
    f = prog if c is None else prog(c)
    ctx, dtype = (FP16, torch.float16) if fmt == 'fp16' else (fp.FP32, torch.float32)
    vals = _LOGB_HARD[fmt]
    n = len(vals)
    src = _compile(f, [ListType(RealType(ctx), n), ListType(RealType(ctx), n), INT])
    _agree(src, f, [torch.tensor(vals, dtype=dtype).cuda(), _out_like(src, n)], block=16)


@pytest.mark.parametrize('prog, width', [(rare_arm, 4), (arm_writes, 5)])
def test_a_skipped_arm_agrees_where_one_row_takes_it(prog, width: int) -> None:
    """Block 0 has one row with an infinity, block 1 none, so the arm runs for
    one and is skipped for the other; a list the arm writes keeps the writes,
    as a list merges by them."""
    rng = random.Random(0)
    rows = [[rng.uniform(-4, 4) for _ in range(width)] for _ in range(32)]
    rows[3][2] = math.inf
    n = len(rows)
    src = _compile(prog, [ListType(ListType(F32, width), n), ListType(F32, n), INT])
    assert 'if tl.max(' in src.source
    _agree(src, prog, [torch.tensor(rows).cuda(), torch.zeros(n).cuda()], block=16)


@pytest.mark.parametrize('prog, n_in', [(interleaved, 8), (reversed_row, 4)])
def test_a_gather_agrees_on_hard_cases(prog, n_in: int) -> None:
    """Signed zeros, subnormals, the largest magnitudes, and the specials:
    a gather moves each bit pattern unchanged."""
    hard = [0.0, -0.0, 2.0 ** -24, -(2.0 ** -14), 65504.0, -65504.0,
            math.inf, -math.inf, math.nan, 1.5]
    rng = random.Random(0)
    rows = [[rng.choice(hard) for _ in range(n_in)] for _ in range(32)]
    _agree_on(prog, torch.tensor(rows), 4, ctx_in=FP16)


@fp.fpy(ctx=fp.FP32)
def _dot_matmul(A: list[list[fp.Real]], BT: list[list[fp.Real]],
                out: list[list[fp.Real]], BLOCK: fp.Real):
    for i in range(len(out)):
        row = out[i]
        for j in range(len(row)):
            a = A[i]
            b = BT[j]
            acc = 0.0
            for k in range(len(a)):
                acc = acc + a[k] * b[k]
            row[j] = acc
    return out


@pytest.mark.parametrize('block_m', [1, 4])
def test_a_tile_of_rows_agrees_past_its_end(block_m: int) -> None:
    """With no lanes, the rows of the output are a tile too: counts including
    ones not a multiple of either tile's size."""
    m, n, k = NamedId('m'), NamedId('n'), NamedId('k')
    src = _compile(_dot_matmul, [ListType(ListType(F32, k), m), ListType(ListType(F32, k), n),
                                 ListType(ListType(F32, n), m), INT])
    assert src.block_m == 'BLOCK_M'
    for rows, cols in ((5, 9), (1, 3), (6, 4)):
        torch.manual_seed(rows * cols)
        _agree(src, _dot_matmul, [torch.randn(rows, 7).cuda(), torch.randn(cols, 7).cuda(),
                                  torch.zeros(rows, cols).cuda()], block_m=block_m)


def test_a_skipped_arm_agrees_in_a_tile_of_rows() -> None:
    """The skip reduces a `[BM, BLOCK]` mask, and the merged name's
    placeholder has that shape."""
    m, n = NamedId('m'), NamedId('n')
    src = _compile(rare_cell, [ListType(F32, m), ListType(F32, n), ListType(ListType(F32, n), m), INT])
    assert src.block_m is not None and 'if tl.max(' in src.source
    rng = random.Random(0)
    xs = [rng.uniform(-4, 4) for _ in range(9)]
    ys = [rng.uniform(-4, 4) for _ in range(7)]
    xs[5] = math.inf
    _agree(src, rare_cell, [_f32(xs), _f32(ys), torch.zeros(9, 7).cuda()], block_m=4)


def test_a_loop_ahead_of_the_column_tile_carries_a_column() -> None:
    """Under a tile of rows alone, a carried value is `[BM, 1]`."""
    m, n, k = NamedId('m'), NamedId('n'), NamedId('k')
    src = _compile(row_first, [ListType(ListType(F32, k), m), ListType(F32, n),
                               ListType(ListType(F32, n), m), INT])
    assert 'acc = tl.broadcast_to(acc, (BLOCK_M, 1))' in src.source
    torch.manual_seed(0)
    _agree(src, row_first, [torch.randn(5, 3).cuda(), torch.randn(9).cuda(),
                            torch.zeros(5, 9).cuda()], block_m=4)


def test_a_power_of_two_agrees_on_special_exponents() -> None:
    """`2 ** logb(x)`: `+0` at a zero, `inf` at an infinity, NaN at NaN."""
    vals = [0.0, -0.0, 2.0 ** -24, 2.0 ** -14, 1.0, -3.0, 65504.0, math.inf, -math.inf, math.nan]
    want = [0.0, 0.0, 2.0 ** -24, 2.0 ** -14, 1.0, 2.0, 2.0 ** 15, math.inf, math.inf, math.nan]
    n = len(vals)
    src = _compile(pow2_logb, [ListType(RealType(FP16), n), ListType(F32, n), INT])
    out = torch.zeros(n).cuda()
    launch(src, [torch.tensor(vals, dtype=torch.float16).cuda(), out], block=16)
    assert _reprs(out.cpu().tolist()) == _reprs(want)


_S3 = RealType(fp.FixedContext(True, 0, 3, fp.RM.RTZ, fp.OV.WRAP))
"""A small signed integer: a scale's exponent."""


def test_a_load_is_not_reused_past_a_rebound_mask() -> None:
    n = NamedId('n')
    src = _compile(mask_rebound, [ListType(F32, n)] * 3 + [INT])
    _agree(src, mask_rebound, [_f32([10.0, 20.0, 30.0, 40.0]), _f32([0.5, 2.0, 0.5, 2.0]),
                               _f32([0.0] * 4)])


def test_a_carried_name_is_not_its_value_before_the_loop() -> None:
    n = NamedId('n')
    src = _compile(carried_index, [ListType(ListType(F32, 4), n), ListType(RealType(fp.SINT8), n),
                                   ListType(F32, n), INT])
    _agree(src, carried_index, [_f32([[1.0, 2, 3, 4], [5, 6, 7, 8]]),
                                torch.tensor([1, 2], dtype=torch.int8).cuda(), _f32([0.0] * 2)])


def test_a_fused_scale_in_rebound_before_its_round_agrees() -> None:
    n = NamedId('n')
    src = _compile(scale_rebound, [ListType(F32, n), ListType(_S3, n), ListType(F32, n),
                                   ListType(F32, n), INT])
    _agree(src, scale_rebound, [_f32([1536.0, 3000.0, 4096.5, 999.0]),
                                torch.tensor([2, 1, -1, 3], dtype=torch.int8).cuda(),
                                _f32([0.0] * 4), _f32([0.0] * 4)])


def test_a_fused_scale_in_a_merge_reads_agrees() -> None:
    n = NamedId('n')
    src = _compile(scale_joined, [ListType(F32, n), ListType(_S3, n), ListType(RealType(fp.FP64), n),
                                  INT])
    _agree(src, scale_joined, [_f32([1536.0, -2.0, 3000.0, 4096.5]),
                               torch.tensor([2, 1, -1, 3], dtype=torch.int8).cuda(),
                               torch.zeros(4, dtype=torch.float64).cuda()])


def test_a_fused_scale_in_of_an_element_stored_over_agrees() -> None:
    n = NamedId('n')
    src = _compile(scale_stored_over, [ListType(ListType(F32, 2), n), ListType(_S3, n),
                                       ListType(F32, n), INT])
    _agree(src, scale_stored_over, [_f32([[1536.0, 1], [3000.0, 1], [4096.5, 2], [999.0, 3]]),
                                    torch.tensor([2, 1, -1, 3], dtype=torch.int8).cuda(),
                                    _f32([0.0] * 4)])


def test_an_aligned_sum_of_fp16_agrees() -> None:
    """The fused round is held in `fp32`: libdevice has no `fp16` one."""
    n = NamedId('n')
    f = aligned_sum(fp.RM.RTZ, digits=10, emin=-14)
    src = _compile(f, [ListType(ListType(RealType(FP16), 4), n), ListType(RealType(fp.FP64), n), INT])
    torch.manual_seed(0)
    _agree(src, f, [(torch.randn(8, 4) * 100).half().cuda(), torch.zeros(8, dtype=torch.float64).cuda()])
