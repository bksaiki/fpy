"""Unit tests for :func:`fpy2.strategies.monomorphize`."""

import pytest

import fpy2 as fp

from fpy2.ast import ListTypeAnn, RealTypeAnn
from fpy2.strategies import monomorphize
from fpy2.types import ListType, RealType


@fp.fpy
def _add_third(x: fp.Real) -> fp.Real:
    return x + fp.rational(1, 3)


@fp.fpy(ctx=fp.FP32)
def _add_third_fp32(x: fp.Real) -> fp.Real:
    return x + fp.rational(1, 3)


@fp.fpy
def _first(xs: list[fp.Real]) -> fp.Real:
    return xs[0]


@fp.fpy
def _mul(x: fp.Real, y: fp.Real) -> fp.Real:
    return x * y


class TestMonomorphizeCtx:

    def test_pins_context(self):
        out = monomorphize(_add_third, fp.FP32)
        assert out.ast.ctx is not None
        assert out.ast.ctx.is_equiv(fp.FP32)
        # the input is not mutated
        assert _add_third.ast.ctx is None

    def test_pinned_context_rounds(self):
        out = monomorphize(_add_third, fp.FP32)
        # behaves exactly like the natively-pinned FP32 function
        for x in (0.0, 1.0, -2.5):
            assert out(x) == _add_third_fp32(x)
        # and differs from the default (FP64) rounding of 1/3
        assert out(0.0) != _add_third(0.0)

    def test_equivalent_pin_ok(self):
        out = monomorphize(_add_third_fp32, fp.FP32)
        assert out.ast.ctx is not None
        assert out.ast.ctx.is_equiv(fp.FP32)

    def test_conflicting_pin_raises(self):
        with pytest.raises(ValueError):
            monomorphize(_add_third_fp32, fp.FP64)


class TestMonomorphizeArgs:

    def test_real_arg(self):
        out = monomorphize(_add_third, args=[RealType(fp.FP32)])
        ann = out.ast.args[0].type
        assert isinstance(ann, RealTypeAnn)
        assert ann.ctx is not None
        assert ann.ctx.is_equiv(fp.FP32)

    def test_list_arg(self):
        out = monomorphize(_first, args=[ListType(RealType(fp.FP32))])
        ann = out.ast.args[0].type
        assert isinstance(ann, ListTypeAnn)
        assert isinstance(ann.elt, RealTypeAnn)
        assert ann.elt.ctx is not None
        assert ann.elt.ctx.is_equiv(fp.FP32)

    def test_none_entry_unchanged(self):
        out = monomorphize(_mul, args=[RealType(fp.FP32), None])
        ann0 = out.ast.args[0].type
        assert isinstance(ann0, RealTypeAnn) and ann0.ctx is not None
        assert out.ast.args[1].type.is_equiv(_mul.ast.args[1].type)

    def test_arity_mismatch_raises(self):
        with pytest.raises(ValueError):
            monomorphize(_mul, args=[RealType(fp.FP32)])

    def test_runtime_preserved(self):
        # annotating argument formats does not change evaluation
        out = monomorphize(_mul, args=[RealType(fp.FP32), RealType(fp.FP32)])
        for x, y in ((0.0, 1.0), (1.5, 2.5), (-3.25, 0.5)):
            assert _mul(x, y) == out(x, y)


