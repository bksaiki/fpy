"""`TritonCompiler`: the pipeline in one call."""

import ast as pyast

import pytest

import fpy2 as fp
from fpy2 import Module
from fpy2.backend.backend import CompileError
from fpy2.backend.triton import TritonCompiler, unavailable
from fpy2.types import ListType, RealType

FP16 = fp.IEEEContext(5, 16)
K = 8


@fp.fpy(ctx=fp.REAL)
def batched_dot(xss: list[list[fp.Real]], yss: list[list[fp.Real]],
                out: list[fp.Real], BLOCK: fp.Real):
    for r in range(len(xss)):
        acc = fp.round(0)
        for k in range(K):
            with fp.FP32:
                acc = acc + xss[r][k] * yss[r][k]
        out[r] = acc
    return out


_ARGS = [
    ListType(ListType(RealType(FP16), K), 4),
    ListType(ListType(RealType(FP16), K), 4),
    ListType(RealType(fp.FP32), 4),
    RealType(fp.INTEGER),
]


def test_unavailable_is_none_or_a_reason():
    why = unavailable()
    assert why is None or (isinstance(why, str) and why)


def test_a_list_is_a_pointer_and_the_tile_width_a_constexpr():
    k = TritonCompiler(drop_asserts=True).compile(
        batched_dot, ctx=fp.REAL, arg_types=_ARGS)
    assert k.params == ('xss_ptr', 'yss_ptr', 'out_ptr', 'BLOCK: tl.constexpr')


_V4 = ListType(RealType(fp.FP32), 4)


class TestAbi:
    def test_a_missing_block_parameter_is_refused(self):
        """A kernel's tile width is a `constexpr` the launcher picks, so the
        program has to take it."""
        @fp.fpy(ctx=fp.FP32)
        def no_block(xs: list[fp.Real], out: list[fp.Real]):
            for i in range(len(xs)):
                out[i] = xs[i]
            return out

        with pytest.raises(CompileError, match='no `BLOCK` parameter'):
            TritonCompiler().compile(no_block, ctx=fp.FP32, arg_types=[_V4, _V4])

    def test_the_block_name_is_configurable(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], out: list[fp.Real], TILE: fp.Real):
            for i in range(len(xs)):
                out[i] = xs[i]
            return out

        k = TritonCompiler(block='TILE', drop_asserts=True).compile(
            f, ctx=fp.FP32, arg_types=[_V4, _V4, RealType(fp.INTEGER)])
        assert 'TILE: tl.constexpr' in k.source

    def test_an_assert_is_refused_by_default(self):
        """Dropping one is a semantic change, so it is asked for."""
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
            for i in range(len(xs)):
                out[i] = xs[i]
            return out

        with pytest.raises(CompileError, match='cannot raise'):
            TritonCompiler().compile(
                f, ctx=fp.FP32, arg_types=[_V4, _V4, RealType(fp.INTEGER)])


class TestModule:
    def test_every_entry_becomes_a_kernel(self):
        m = Module()
        m.add(batched_dot, ctx=fp.REAL, arg_types=_ARGS)
        out = TritonCompiler(drop_asserts=True).compile_module(m)
        assert len(out) == 1
        assert out[0].name == 'batched_dot'


class TestOptimize:
    """`optimize` gates `ConstFold` and `Simplify`, as the cpp backend's flag
    of the same name gates its optimizing transforms."""

    def test_both_settings_compile_this_program(self):
        for optimize in (True, False):
            k = TritonCompiler(drop_asserts=True, optimize=optimize).compile(
                batched_dot, ctx=fp.REAL, arg_types=_ARGS)
            pyast.parse(k.source)

    def test_optimizing_removes_the_debris(self):
        """`FreeVarElim` materializes a captured value and `ConstFold` then
        inlines past it, leaving a binding nothing reads."""
        SCALE = 2.5

        @fp.fpy(ctx=fp.FP32)
        def scaled(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
            for i in range(len(xs)):
                out[i] = xs[i] * SCALE
            return out

        argt = [_V4, _V4, RealType(fp.INTEGER)]
        on = TritonCompiler(drop_asserts=True).compile(
            scaled, ctx=fp.FP32, arg_types=argt).source
        off = TritonCompiler(drop_asserts=True, optimize=False).compile(
            scaled, ctx=fp.FP32, arg_types=argt).source
        assert 'SCALE' not in on
        assert 'SCALE = ' in off
        # the value still reaches the multiply either way
        assert '2.5' in on and '2.5' in off


RZ_FP16 = fp.IEEEContext(5, 16, fp.RM.RTZ)
_RZ_ARGT = [ListType(RealType(fp.FP64), 8), ListType(RealType(fp.FP16), 8),
            RealType(fp.INTEGER)]


@fp.fpy(ctx=fp.REAL)
def _rz(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(xs)):
        with RZ_FP16:
            out[i] = fp.round(xs[i])
    return out


class TestUnfold:
    """A rounding Triton cannot spell -- round-toward-zero into FP16 from
    `f64` -- is lowered to integer arithmetic where `unfold` asks for it."""

    def test_without_it_the_rounding_is_refused(self):
        with pytest.raises(CompileError, match='no cast spelling'):
            TritonCompiler(drop_asserts=True).compile(
                _rz, ctx=fp.REAL, arg_types=_RZ_ARGT)

    def test_with_it_the_rounding_compiles(self):
        k = TritonCompiler(
            drop_asserts=True, unfold=TritonCompiler.UnfoldMode.ROUNDINGS,
        ).compile(_rz, ctx=fp.REAL, arg_types=_RZ_ARGT)
        assert 'libdevice.trunc' in k.source
        pyast.parse(k.source)
