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
    Expr,
    Fma,
    Log,
    Mul,
    Sin,
    Sqrt,
    Sub,
    Tan,
)
from fpy2.backend.triton.target import OpTable, is_native_ctx, make_op_table
from fpy2.backend.triton.types import TritonScalar as T

_TABLE = make_op_table()
_FP16 = fp.IEEEContext(5, 16, fp.RM.RNE)
_FP32 = fp.IEEEContext(8, 32, fp.RM.RNE)
_SUFFIX = {fp.RM.RTZ: 'rz', fp.RM.RTN: 'rd', fp.RM.RTP: 'ru'}


def _spelling(op: str, rn: str, rm: fp.RM) -> str:
    if rm is fp.RM.RNE:
        return rn
    return f'_fpy_{op}_ro' if rm is fp.RM.RTO else f'libdevice.{op}_{_SUFFIX[rm]}'
_FP64 = fp.IEEEContext(11, 64, fp.RM.RNE)


def _ctxs(table: OpTable, op: type[Expr]) -> list[fp.Context]:
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
        assert _FP64 in _ctxs(_TABLE.binary, Div)

    def test_no_integer_divide(self):
        """FPy's integer contexts round toward zero; Triton's `//` floors.
        They disagree on every negative quotient."""
        for ctx in (fp.SINT32, fp.UINT32, fp.INTEGER):
            assert ctx not in _ctxs(_TABLE.binary, Div)

    def test_divide_uses_the_correctly_rounded_spelling(self):
        """Triton's `/` is `fdiv`, the fast variant; `tl.div_rn` is IEEE,
        libdevice's `div_rz` and the rest its directed modes, and the
        launcher's `_fpy_div_ro` round to odd."""
        for sig in _TABLE.binary[Div]:
            assert sig.name == _spelling('div', 'tl.div_rn', sig.out_ctx.rm)


class TestSquareRoot:
    def test_uses_the_correctly_rounded_spelling(self):
        """`tl.sqrt` is the fast variant; `tl.sqrt_rn` is IEEE,
        libdevice's `sqrt_rz` and the rest its directed modes, and the
        launcher's `_fpy_sqrt_ro` round to odd."""
        for sig in _TABLE.unary[Sqrt]:
            assert sig.name == _spelling('sqrt', 'tl.sqrt_rn', sig.out_ctx.rm)


class TestNoTranscendentals:
    """Not an oversight, and it is what makes the validation gate total.

    Triton has no correctly-rounded `exp`/`log`/`sin`/`erf`.  The cpp backend
    lives with the same exposure by excluding them from its bit-exact
    differential check; emitting none of them makes that exclusion list empty,
    so every function this backend compiles can be checked bit-for-bit.
    """

    @pytest.mark.parametrize('op', [Exp, Log, Sin, Cos, Tan, Erf])
    def test_absent(self, op):
        assert op not in _TABLE.unary
        assert op not in _TABLE.binary
        assert op not in _TABLE.ternary


class TestDirectedArithmetic:
    """libdevice's directed `add_rz` and the rest, at FP32 and FP64 only."""

    _DIRECTED = tuple(fp.IEEEContext(es, nbits, rm) for es, nbits in ((8, 32), (11, 64))
                      for rm in (fp.RM.RTZ, fp.RM.RTN, fp.RM.RTP, fp.RM.RTO))

    @pytest.mark.parametrize('op', [Add, Sub, Mul, Div])
    def test_binary_present(self, op):
        assert set(self._DIRECTED) <= set(_ctxs(_TABLE.binary, op))

    def test_fma_and_sqrt_present(self):
        assert set(self._DIRECTED) <= set(_ctxs(_TABLE.ternary, Fma))
        assert set(self._DIRECTED) <= set(_ctxs(_TABLE.unary, Sqrt))

    def test_not_at_fp16(self):
        """libdevice has no directed half arithmetic."""
        for rm in (fp.RM.RTZ, fp.RM.RTN, fp.RM.RTP, fp.RM.RTO):
            assert fp.IEEEContext(5, 16, rm) not in _ctxs(_TABLE.binary, Add)

    def test_spellings(self):
        rz = fp.IEEEContext(8, 32, fp.RM.RTZ)
        add = next(s for s in _TABLE.binary[Add] if s.out_ctx == rz)
        sub = next(s for s in _TABLE.binary[Sub] if s.out_ctx == rz)
        assert add.in_tys == (T.F32, T.F32)
        assert add.format('a', 'b') == 'libdevice.add_rz(a, b)'
        assert sub.format('a', 'b') == 'libdevice.add_rz(a, -(b))'

    def test_odd_spellings(self):
        """Round to odd is the launcher's helper, from rounding down and up."""
        ro = fp.IEEEContext(11, 64, fp.RM.RTO)
        add = next(s for s in _TABLE.binary[Add] if s.out_ctx == ro)
        sub = next(s for s in _TABLE.binary[Sub] if s.out_ctx == ro)
        assert add.format('a', 'b') == '_fpy_add_ro(a, b)'
        assert sub.format('a', 'b') == '_fpy_add_ro(a, -(b))'


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
