"""
MPFX backend: target-specific instructions.
"""

from enum import Enum
from typing import NoReturn

from ...ast.fpyast import FuncDef
from ...number import RM, Context, INTEGER, FP64, REAL
from .format import AbstractFormat, SupportedContext
from .utils import MPFXCompileError

__all__ = [
    'AddInstr',
    'NegInstr',
    'AbsInstr',
    'SqrtInstr',
    'SubInstr',
    'MulInstr',
    'DivInstr',
    'FMAInstr',
]

def _cvt_context(ctx: Context):
    if isinstance(ctx, SupportedContext):
        return AbstractFormat.from_context(ctx)
    else:
        return None

def _round_mode(ctx: Context):
    if isinstance(ctx, SupportedContext):
        return ctx.rm
    else:
        raise ValueError("Unsupported context for rounding mode extraction.")

def _fits_in_integer(ty: AbstractFormat) -> bool:
    """Does this type fit in an integer?"""
    return ty <= AbstractFormat.from_context(INTEGER)

def _fits_in_double(ty: AbstractFormat) -> bool:
    """Does this type fit in a double?"""
    return ty <= AbstractFormat.from_context(FP64)


def _rto_is_valid(ctx: Context) -> bool:
    """
    Can we double round under `ctx` after rounding to double using RTO?

    This determination is made based on the number format
    corresponding to `ctx` and the rounding mode.
    """
    fmt = _cvt_context(ctx)
    if fmt is None:
        return False

    rm = _round_mode(ctx)
    match rm:
        case RM.RNE | RM.RNA:
            # need at least 2 more bits of precision
            eff_fmt = fmt.with_prec_offset(2).with_exp_offset(-2)
        case RM.RTZ | RM.RAZ | RM.RTN | RM.RTP:
            # need at least 1 more bit of precision
            eff_fmt = fmt.with_prec_offset(1).with_exp_offset(-1)
        case RM.RTO:
            # need at least as much precision
            pass
        case _:
            raise ValueError(f"Unsupported rounding mode: {rm}")

    return eff_fmt <= AbstractFormat.from_context(FP64)

#####################################################################
# Instruction generator

class FP64Engine(Enum):
    """MPFX engine type."""
    FP_RTO = 0
    EFT = 1
    SOFTFLOAT = 2
    FLOPPYFLOAT = 3

    def to_cpp(self) -> str:
        match self:
            case FP64Engine.FP_RTO:
                return 'mpfx::Engine::FP_RTO'
            case FP64Engine.EFT:
                return 'mpfx::Engine::EFT'
            case FP64Engine.SOFTFLOAT:
                return 'mpfx::Engine::SOFTFLOAT'
            case FP64Engine.FLOPPYFLOAT:
                return 'mpfx::Engine::FFLOAT'
            case _:
                raise ValueError(f"Unsupported FP64 engine: {self}")


class InstrGenerator:
    """Instruction generator."""
    func: FuncDef
    engine: FP64Engine
    allow_exact: bool

    def __init__(self, func: FuncDef, engine: FP64Engine = FP64Engine.EFT, allow_exact: bool = True):
        self.func = func
        self.engine = engine
        self.allow_exact = allow_exact

    def raise_error(self, msg: str) -> NoReturn:
        raise MPFXCompileError(self.func, msg)

    def cast(
        self,
        arg_str: str,
        ctx_str: str | None,
        arg_ty: AbstractFormat,
        ctx: Context
    ):
        if arg_ty <= _cvt_context(ctx):
            # no cast needed
            return arg_str
        self.raise_error(f'cannot compile `cast({arg_str})`: {arg_ty} under context {ctx}')

    def neg(
        self,
        arg_str: str,
        ctx_str: str | None,
        arg_ty: AbstractFormat,
        ctx: Context
    ):
        if _fits_in_double(arg_ty) and _rto_is_valid(ctx):
            # use the FP-RTO backed implementation
            if ctx_str is None:
                self.raise_error(f'unsupported context `-{arg_str}`: {ctx}')
            return f'mpfx::neg({arg_str}, {ctx_str})'

        self.raise_error(f'cannot compile `-{arg_str}`: {arg_ty} under context {ctx}')

    def abs(
        self,
        arg_str: str,
        ctx_str: str | None,
        arg_ty: AbstractFormat,
        ctx: Context
    ):
        if _fits_in_double(arg_ty) and _rto_is_valid(ctx):
            # use the FP-RTO backed implementation
            if ctx_str is None:
                self.raise_error(f'unsupported context `abs({arg_str})`: {ctx}')
            return f'mpfx::abs({arg_str}, {ctx_str})'

        self.raise_error(f'cannot compile `abs({arg_str})`: {arg_ty} under context {ctx}')

    def sqrt(
        self,
        arg_str: str,
        ctx_str: str | None,
        arg_ty: AbstractFormat,
        ctx: Context
    ):
        if _fits_in_double(arg_ty) and _rto_is_valid(ctx):
            # use the FP-RTO backed implementation
            if ctx_str is None:
                self.raise_error(f'unsupported context `sqrt({arg_str})`: {ctx}')
            engine_str = self.engine.to_cpp()
            return f'mpfx::sqrt<{engine_str}>({arg_str}, {ctx_str})'

        self.raise_error(f'cannot compile `sqrt({arg_str})`: {arg_ty} under context {ctx}')

    def add(
        self,
        lhs_str: str,
        rhs_str: str,
        ctx_str: str | None,
        lhs_ty: AbstractFormat,
        rhs_ty: AbstractFormat,
        ctx: Context
    ):
        if ctx is REAL:
            # exact addition
            if _fits_in_integer(lhs_ty) and _fits_in_integer(rhs_ty) and ctx is REAL:
                # use integer addition for real addition under INTEGER context
                return f'({lhs_str} + {rhs_str})'
            if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty):
                exact_ty = lhs_ty + rhs_ty
                if exact_ty <= AbstractFormat.from_context(FP64):
                    # addition is exact
                    return f'({lhs_str} + {rhs_str})'
        else:
            if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty):
                exact_ty = lhs_ty + rhs_ty
                if self.allow_exact and exact_ty <= AbstractFormat.from_context(FP64):
                    # use direct addition if the result fits in double
                    if ctx_str is None:
                        self.raise_error(f'unsupported context `{lhs_str} + {rhs_str}`: {ctx}')
                    return f'mpfx::add<mpfx::Engine::FP_EXACT>({lhs_str}, {rhs_str}, {ctx_str})'
                if _rto_is_valid(ctx):
                    # use the FP-RTO backed implementation
                    if ctx_str is None:
                        self.raise_error(f'unsupported context `{lhs_str} + {rhs_str}`: {ctx}')
                    engine_str = self.engine.to_cpp()
                    return f'mpfx::add<{engine_str}>({lhs_str}, {rhs_str}, {ctx_str})'

        self.raise_error(f'cannot compile `{lhs_str} + {rhs_str}`: {lhs_ty} + {rhs_ty} under context {ctx}')

    def sub(
        self,
        lhs_str: str,
        rhs_str: str,
        ctx_str: str | None,
        lhs_ty: AbstractFormat,
        rhs_ty: AbstractFormat,
        ctx: Context
    ):
        if ctx is REAL:
            # exact subtraction
            if _fits_in_integer(lhs_ty) and _fits_in_integer(rhs_ty) and ctx is REAL:
                # use integer subtraction for real subtraction under INTEGER context
                return f'({lhs_str} - {rhs_str})'
            if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty):
                exact_ty = lhs_ty - rhs_ty
                if exact_ty <= AbstractFormat.from_context(FP64):
                    # subtraction is exact
                    return f'({lhs_str} - {rhs_str})'
        else:
            if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty):
                exact_ty = lhs_ty - rhs_ty
                if self.allow_exact and exact_ty <= AbstractFormat.from_context(FP64):
                    # use direct subtraction if the result fits in double
                    if ctx_str is None:
                        self.raise_error(f'unsupported context `{lhs_str} - {rhs_str}`: {ctx}')
                    return f'mpfx::sub<mpfx::Engine::FP_EXACT>({lhs_str}, {rhs_str}, {ctx_str})'
                if _rto_is_valid(ctx):
                    # use the FP-RTO backed implementation
                    if ctx_str is None:
                        self.raise_error(f'unsupported context `{lhs_str} - {rhs_str}`: {ctx}')
                    engine_str = self.engine.to_cpp()
                    return f'mpfx::sub<{engine_str}>({lhs_str}, {rhs_str}, {ctx_str})'

        self.raise_error(f'cannot compile `{lhs_str} - {rhs_str}`: {lhs_ty} - {rhs_ty} under context {ctx}')

    def mul(
        self,
        lhs_str: str,
        rhs_str: str,
        ctx_str: str | None,
        lhs_ty: AbstractFormat,
        rhs_ty: AbstractFormat,
        ctx: Context
    ):
        if ctx is REAL:
            # exact multiplication
            if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty):
                exact_ty = lhs_ty * rhs_ty
                if exact_ty <= AbstractFormat.from_context(FP64):
                    # multiplication is exact
                    return f'({lhs_str} * {rhs_str})'
        else:
            if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty):
                exact_ty = lhs_ty * rhs_ty
                if self.allow_exact and exact_ty <= AbstractFormat.from_context(FP64):
                    # use direct multiplication if the result fits in double
                    if ctx_str is None:
                        self.raise_error(f'unsupported context `{lhs_str} * {rhs_str}`: {ctx}')
                    return f'mpfx::mul<mpfx::Engine::FP_EXACT>({lhs_str}, {rhs_str}, {ctx_str})'
                if _rto_is_valid(ctx):
                    # use the FP-RTO backed implementation
                    if ctx_str is None:
                        self.raise_error(f'unsupported context `{lhs_str} * {rhs_str}`: {ctx}')
                    engine_str = self.engine.to_cpp()
                    return f'mpfx::mul<{engine_str}>({lhs_str}, {rhs_str}, {ctx_str})'

        self.raise_error(f'cannot compile `{lhs_str} * {rhs_str}`: {lhs_ty} * {rhs_ty} under context {ctx}')

    def div(
        self,
        lhs_str: str,
        rhs_str: str,
        ctx_str: str | None,
        lhs_ty: AbstractFormat,
        rhs_ty: AbstractFormat,
        ctx: Context
    ):
        if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty) and _rto_is_valid(ctx):
            # use the FP-RTO backed implementation
            if ctx_str is None:
                self.raise_error(f'unsupported context `{lhs_str} / {rhs_str}`: {ctx}')
            engine_str = self.engine.to_cpp()
            return f'mpfx::div<{engine_str}>({lhs_str}, {rhs_str}, {ctx_str})'

        self.raise_error(f'cannot compile `{lhs_str} / {rhs_str}`: {lhs_ty} / {rhs_ty} under context {ctx}')

    def fma(
        self,
        a_str: str,
        b_str: str,
        c_str: str,
        ctx_str: str | None,
        a_ty: AbstractFormat,
        b_ty: AbstractFormat,
        c_ty: AbstractFormat,
        ctx: Context
    ):
        if _fits_in_double(a_ty) and _fits_in_double(b_ty) and _fits_in_double(c_ty):
            # exact_ty = a_ty * b_ty + c_ty
            # if self.allow_exact and exact_ty <= AbstractFormat.from_context(FP64):
            #     # use direct FMA if the result fits in double
            #     if ctx_str is None:
            #         self.raise_error(f'unsupported context `fma({a_str}, {b_str}, {c_str})`: {ctx}')
            #     return f'mpfx::fma<mpfx::Engine::FP_EXACT>({a_str}, {b_str}, {c_str}, {ctx_str})'
            if _rto_is_valid(ctx):
                # use the FP-RTO backed implementation
                if ctx_str is None:
                    self.raise_error(f'unsupported context `fma({a_str}, {b_str}, {c_str})`: {ctx}')
                engine_str = self.engine.to_cpp()
                return f'mpfx::fma<{engine_str}>({a_str}, {b_str}, {c_str}, {ctx_str})'

        self.raise_error(f'cannot compile `fma({a_str}, {b_str}, {c_str})`: fma({a_ty}, {b_ty}, {c_ty}) under context {ctx}')
