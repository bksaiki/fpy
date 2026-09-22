"""The op table's omissions are the design, so they are pinned.

The contract: if compilation succeeds,
the emitted kernel must behave as the FPy interpreter does.  On a target whose
defaults trade numerical agreement for throughput, holding that means declining
to name an operation whose Triton spelling is not the one FPy specifies.  Every
absence below is such a decline, and each would be easy to "fix" by adding a
signature that quietly changes what a program computes.
"""

import pytest

import fpy2 as fp
from fpy2.ast.fpyast import (
    Add,
    Cos,
    Div,
    Erf,
    Exp,
    Fma,
    Log,
    Mul,
    Sin,
    Sqrt,
    Sub,
    Tan,
)
from fpy2.backend.triton.target import is_native_ctx, make_op_table
from fpy2.backend.triton.types import TritonScalar as T

_TABLE = make_op_table()
_FP16 = fp.IEEEContext(5, 16, fp.RM.RNE)
_FP32 = fp.IEEEContext(8, 32, fp.RM.RNE)


def _ctxs(table, op):
    return [s.out_ctx for s in table.get(op, [])]


class TestDivision:
    """Two separate reasons division is restricted."""

    def test_no_fp16_divide(self):
        """Triton types `fp16 / fp16` as fp32 -- no hardware divide below 32
        bits -- so the result is an fp32 quotient narrowed back.  That is a
        double rounding, not FP16's division."""
        assert _FP16 not in _ctxs(_TABLE.binary, Div)

    def test_fp32_and_fp64_divide_exist(self):
        assert _FP32 in _ctxs(_TABLE.binary, Div)

    def test_no_integer_divide(self):
        """FPy's integer contexts round toward zero; Triton's `//` floors.
        They disagree on every negative quotient."""
        for ctx in (fp.SINT32, fp.UINT32, fp.INTEGER):
            assert ctx not in _ctxs(_TABLE.binary, Div)

    def test_divide_uses_the_correctly_rounded_spelling(self):
        """`/` is `fdiv`, the fast variant."""
        for sig in _TABLE.binary[Div]:
            assert sig.name == 'tl.div_rn'


class TestSquareRoot:
    def test_uses_the_correctly_rounded_spelling(self):
        """`tl.sqrt` is the fast variant; `tl.sqrt_rn` is IEEE."""
        for sig in _TABLE.unary[Sqrt]:
            assert sig.name == 'tl.sqrt_rn'


class TestNoTranscendentals:
    """Not an oversight, and it is what makes the validation gate total.

    Triton has no correctly-rounded `exp`/`log`/`sin`/`erf`.  The cpp backend
    lives with the same exposure by excluding 27 operators from its bit-exact
    differential check; emitting none of them instead makes that exclusion list
    empty, so every function this backend compiles can also be checked
    bit-for-bit.
    """

    @pytest.mark.parametrize('op', [Exp, Log, Sin, Cos, Tan, Erf])
    def test_absent(self, op):
        assert op not in _TABLE.unary
        assert op not in _TABLE.binary
        assert op not in _TABLE.ternary


class TestNativeContexts:
    def test_rne_only(self):
        """Triton exposes no per-instruction rounding modifier, so every other
        mode reaches codegen through `unfold_round` or not at all."""
        assert is_native_ctx(_FP32)
        for rm in (fp.RM.RTZ, fp.RM.RTP, fp.RM.RTN):
            assert not is_native_ctx(fp.IEEEContext(8, 32, rm))

    def test_fp16_is_native(self):
        """`x.to(tl.float16)` *is* FP16's round-to-nearest-even, which is what
        this predicate is asked about at a `Round`/`Cast` site."""
        assert is_native_ctx(_FP16)

    def test_out_of_scope_formats_are_not_native(self):
        for ctx in (fp.BF16, fp.S1E4M3, fp.MX_E4M3, fp.TF32):
            assert not is_native_ctx(ctx)

    def test_native_is_not_the_same_as_dispatchable(self):
        """The predicate is context-level while op coverage is not uniform over
        a context -- `Add` at FP16 dispatches and `Div` at FP16 does not.  The
        cast reading is the one that must stay; see `is_native_ctx`'s docstring
        for why, and for what the mismatch costs (a diagnostic, not an
        outcome)."""
        assert is_native_ctx(_FP16)
        assert _FP16 in _ctxs(_TABLE.binary, Add)
        assert _FP16 not in _ctxs(_TABLE.binary, Div)


class TestFp16Arithmetic:
    """What FP16 *does* dispatch: Triton keeps `fp16 op fp16` in fp16 for
    everything but division."""

    @pytest.mark.parametrize('op', [Add, Sub, Mul])
    def test_present(self, op):
        assert _FP16 in _ctxs(_TABLE.binary, op)

    def test_fma_is_present(self):
        assert _FP16 in _ctxs(_TABLE.ternary, Fma)

    def test_spellings(self):
        sig = next(s for s in _TABLE.binary[Add] if s.out_ctx == _FP16)
        assert sig.in_tys == (T.F16, T.F16)
        assert sig.format('a', 'b') == '(a + b)'
