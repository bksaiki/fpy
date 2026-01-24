"""
C++/FPy backend: target-specific instructions.
"""

from ...number import RM, RealContext, Context, INTEGER, FP64
from .format import AbstractFormat, SupportedContext

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
    return ty.contained_in(AbstractFormat.from_context(INTEGER))

def _fits_in_double(ty: AbstractFormat) -> bool:
    """Does this type fit in a double?"""
    return ty.contained_in(AbstractFormat.from_context(FP64))


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

    return eff_fmt.contained_in(AbstractFormat.from_context(FP64))



#####################################################################
# Numerical instructions

class AddInstr:
    """C++ instruction: addition."""

    @staticmethod
    def generator(lhs_ty: AbstractFormat, rhs_ty: AbstractFormat, ctx: Context):
        """
        Generate C++ code for addition.

        Args:
            lhs_ty: Type of the left-hand side operand.
            rhs_ty: Type of the right-hand side operand.
            ctx: Number context.
        Returns:
            A function that generates C++ code for addition.
        """
        if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty) and _rto_is_valid(ctx):
            # use the FP-RTO backed implementation
            return lambda lhs, rhs, ctx: f'fpy::add({lhs}, {rhs}, {ctx})'
        elif _fits_in_integer(lhs_ty) and _fits_in_integer(rhs_ty) and isinstance(ctx, RealContext):
            # use integer addition for real addition under INTEGER context
            return lambda lhs, rhs, ctx: f'({lhs} + {rhs})'

        # no suitable implementation found
        return None


class NegInstr:
    """C++ instruction: negation."""

    @staticmethod
    def generator(arg_ty: AbstractFormat, ctx: Context):
        """Generate C++ code for negation."""
        ret_ty = _cvt_context(ctx)
        if _fits_in_double(arg_ty) and _rto_is_valid(ctx):
            # use the FP-RTO backed implementation
            return lambda operand, ctx: f'fpy::neg({operand}, {ctx})'

        # no suitable implementation found
        return None


class AbsInstr:
    """C++ instruction: absolute value."""

    @staticmethod
    def generator(arg_ty: AbstractFormat, ctx: Context):
        """Generate C++ code for absolute value."""
        if _fits_in_double(arg_ty) and _rto_is_valid(ctx):
            # use the FP-RTO backed implementation
            return lambda operand, ctx: f'fpy::abs({operand}, {ctx})'

        # no suitable implementation found
        return None


class SqrtInstr:
    """C++ instruction: square root."""

    @staticmethod
    def generator(arg_ty: AbstractFormat, ctx: Context):
        """Generate C++ code for square root."""
        ret_ty = _cvt_context(ctx)
        if _fits_in_double(arg_ty) and _rto_is_valid(ctx):
            # use the FP-RTO backed implementation
            return lambda operand, ctx: f'fpy::sqrt({operand}, {ctx})'

        # no suitable implementation found
        return None


class SubInstr:
    """C++ instruction: subtraction."""

    @staticmethod
    def generator(lhs_ty: AbstractFormat, rhs_ty: AbstractFormat, ctx: Context):
        """Generate C++ code for subtraction."""
        ret_ty = _cvt_context(ctx)
        if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty) and _rto_is_valid(ctx):
            # use the FP-RTO backed implementation
            return lambda lhs, rhs, ctx: f'fpy::sub({lhs}, {rhs}, {ctx})'

        # no suitable implementation found
        return None


class MulInstr:
    """C++ instruction: multiplication."""

    @staticmethod
    def generator(lhs_ty: AbstractFormat, rhs_ty: AbstractFormat, ctx: Context):
        """Generate C++ code for multiplication."""

        if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty):
            exact_ty = lhs_ty * rhs_ty
            ctx_ty = _cvt_context(ctx)
            if ctx_ty is not None and exact_ty.contained_in(ctx_ty):
                # multiplication is exact
                return lambda lhs, rhs, ctx: f'{lhs} * {rhs}'
            if exact_ty.contained_in(AbstractFormat.from_context(FP64)):
                # use direct multiplication if the result fits in double
                return lambda lhs, rhs, ctx: f'fpy::mul<fpy::EngineType::EXACT>({lhs}, {rhs}, {ctx})'
            if _rto_is_valid(ctx):
                # use the FP-RTO backed implementation
                return lambda lhs, rhs, ctx: f'fpy::mul({lhs}, {rhs}, {ctx})'

        # no suitable implementation found
        return None


class DivInstr:
    """C++ instruction: division."""

    @staticmethod
    def generator(lhs_ty: AbstractFormat, rhs_ty: AbstractFormat, ctx: Context):
        """Generate C++ code for division."""
        if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty) and _rto_is_valid(ctx):
            # use the FP-RTO backed implementation
            return lambda lhs, rhs, ctx: f'fpy::div({lhs}, {rhs}, {ctx})'

        # no suitable implementation found
        return None


class FMAInstr:
    """C++ instruction: fused multiply-add."""

    @staticmethod
    def generator(a_ty: AbstractFormat, b_ty: AbstractFormat, c_ty: AbstractFormat, ctx: Context):
        """Generate C++ code for fused multiply-add."""
        if (
            _fits_in_double(a_ty)
            and _fits_in_double(b_ty)
            and _fits_in_double(c_ty)
            and _rto_is_valid(ctx)
        ):
            # use the FP-RTO backed implementation
            return lambda a, b, c, ctx: f'fpy::fma({a}, {b}, {c}, {ctx})'

        # no suitable implementation found
        return None
