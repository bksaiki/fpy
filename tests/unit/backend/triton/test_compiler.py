"""`TritonCompiler`: the pipeline in one call.

What it is really for is the *order*: three of the steps' orderings are not
obvious and each was found the hard way, so the driver states them once rather
than leaving a caller to rediscover them.
"""

import ast as pyast

import pytest

import fpy2 as fp
from fpy2 import Module
from fpy2.backend.backend import CompileError
from fpy2.backend.triton import TritonCompiler
from fpy2.types import ListType, RealType

FP16 = fp.IEEEContext(5, 16)
K = 8
"""A *foreign* constant.  `Specialize` monomorphizes contexts and types, not
closure values, so reaching `tl.static_range` at all needs `ConstFold`."""


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


def _compile(func=batched_dot, argt=None, **kwargs):
    return TritonCompiler(drop_asserts=True, **kwargs).compile(
        func, ctx=fp.REAL, arg_types=argt or _ARGS)


class TestPipeline:
    def test_one_call_produces_a_kernel(self):
        k = _compile()
        assert k.source.startswith('@triton.jit\ndef batched_dot(')
        assert k.params == (
            'xss_ptr', 'yss_ptr', 'out_ptr', 'BLOCK: tl.constexpr')
        pyast.parse(k.source)

    def test_a_foreign_constant_is_folded(self):
        """`range(K)` names a closure value, which only `ConstFold` resolves
        -- and `tl.static_range` needs the count as a constant."""
        assert 'tl.static_range(8)' in _compile().source

    def test_the_tile_and_the_sequential_fold_are_distinguished(self):
        """The batch tiles; the accumulation rounds, so it stays per-lane."""
        src = _compile().source
        assert 'tl.program_id(0)' in src and 'tl.arange(0, ' in src
        assert src.count('tl.static_range') == 1

    def test_fusion_is_derived(self):
        assert _compile().enable_fp_fusion


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
            TritonCompiler().compile(
                no_block, ctx=fp.FP32,
                arg_types=[ListType(RealType(fp.FP32), 4),
                           ListType(RealType(fp.FP32), 4)])

    def test_the_block_name_is_configurable(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], out: list[fp.Real], TILE: fp.Real):
            for i in range(len(xs)):
                out[i] = xs[i]
            return out

        k = TritonCompiler(block='TILE', drop_asserts=True).compile(
            f, ctx=fp.FP32,
            arg_types=[ListType(RealType(fp.FP32), 4),
                       ListType(RealType(fp.FP32), 4),
                       RealType(fp.INTEGER)])
        assert 'TILE: tl.constexpr' in k.source

    def test_an_assert_is_refused_by_default(self):
        """Dropping one is a semantic change, so it is asked for."""
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
            for i in range(len(xs)):
                out[i] = xs[i]
            return out

        argt = [ListType(RealType(fp.FP32), 4),
                ListType(RealType(fp.FP32), 4), RealType(fp.INTEGER)]
        with pytest.raises(CompileError, match='cannot raise'):
            TritonCompiler().compile(f, ctx=fp.FP32, arg_types=argt)


class TestModule:
    def test_every_entry_becomes_a_kernel(self):
        m = Module()
        m.add(batched_dot, ctx=fp.REAL, arg_types=_ARGS)
        out = TritonCompiler(drop_asserts=True).compile_module(m)
        assert len(out) == 1
        assert out[0].name == 'batched_dot'
