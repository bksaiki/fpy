"""
C++/FPy backend: target-specific instructions.
"""

from ...ast import FuncDef
from ...number import Context, INTEGER, FP64

from .format import AbstractFormat, SupportedContext
from .utils import CppFpyCompileError




def _fits_in_integer(ty: AbstractFormat) -> bool:
    return ty.contained_in(AbstractFormat.from_context(INTEGER))

def _fits_in_double(ty: AbstractFormat) -> bool:
    return ty.contained_in(AbstractFormat.from_context(FP64))

def _cvt_context(ctx: Context):
    if isinstance(ctx, SupportedContext):
        return AbstractFormat.from_context(ctx)
    else:
        return None


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
        ret_ty = _cvt_context(ctx)
        if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty) and _fits_in_double(ret_ty):
            # use the FP-RTO backed implementation
            return lambda lhs, rhs, ctx: f'fpy::add({lhs}, {rhs}, {ctx})'

        # no suitable implementation found
        return None


class NegInstr:
    """C++ instruction: negation."""

    @staticmethod
    def generator(operand_ty: AbstractFormat, ctx: Context):
        """
        Generate C++ code for negation.

        Args:
            operand_ty: Type of the operand.
            ctx: Number context.
        Returns:
            A function that generates C++ code for negation.
        """
        ret_ty = _cvt_context(ctx)
        if _fits_in_double(operand_ty) and _fits_in_double(ret_ty):
            # use the FP-RTO backed implementation
            return lambda operand, ctx: f'fpy::neg({operand}, {ctx})'

        # no suitable implementation found
        return None


class AbsInstr:
    """C++ instruction: absolute value."""

    @staticmethod
    def generator(operand_ty: AbstractFormat, ctx: Context):
        """
        Generate C++ code for absolute value.

        Args:
            operand_ty: Type of the operand.
            ctx: Number context.
        Returns:
            A function that generates C++ code for absolute value.
        """
        ret_ty = _cvt_context(ctx)
        if _fits_in_double(operand_ty) and _fits_in_double(ret_ty):
            # use the FP-RTO backed implementation
            return lambda operand, ctx: f'fpy::abs({operand}, {ctx})'

        # no suitable implementation found
        return None


class SqrtInstr:
    """C++ instruction: square root."""

    @staticmethod
    def generator(operand_ty: AbstractFormat, ctx: Context):
        """
        Generate C++ code for square root.

        Args:
            operand_ty: Type of the operand.
            ctx: Number context.
        Returns:
            A function that generates C++ code for square root.
        """
        ret_ty = _cvt_context(ctx)
        if _fits_in_double(operand_ty) and _fits_in_double(ret_ty):
            # use the FP-RTO backed implementation
            return lambda operand, ctx: f'fpy::sqrt({operand}, {ctx})'

        # no suitable implementation found
        return None


class SubInstr:
    """C++ instruction: subtraction."""

    @staticmethod
    def generator(lhs_ty: AbstractFormat, rhs_ty: AbstractFormat, ctx: Context):
        """
        Generate C++ code for subtraction.

        Args:
            lhs_ty: Type of the left-hand side operand.
            rhs_ty: Type of the right-hand side operand.
            ctx: Number context.
        Returns:
            A function that generates C++ code for subtraction.
        """
        ret_ty = _cvt_context(ctx)
        if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty) and _fits_in_double(ret_ty):
            # use the FP-RTO backed implementation
            return lambda lhs, rhs, ctx: f'fpy::sub({lhs}, {rhs}, {ctx})'

        # no suitable implementation found
        return None


class MulInstr:
    """C++ instruction: multiplication."""

    @staticmethod
    def generator(lhs_ty: AbstractFormat, rhs_ty: AbstractFormat, ctx: Context):
        """
        Generate C++ code for multiplication.

        Args:
            lhs_ty: Type of the left-hand side operand.
            rhs_ty: Type of the right-hand side operand.
            ctx: Number context.
        Returns:
            A function that generates C++ code for multiplication.
        """
        ret_ty = _cvt_context(ctx)
        if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty) and _fits_in_double(ret_ty):
            # use the FP-RTO backed implementation
            return lambda lhs, rhs, ctx: f'fpy::mul({lhs}, {rhs}, {ctx})'

        # no suitable implementation found
        return None


class DivInstr:
    """C++ instruction: division."""

    @staticmethod
    def generator(lhs_ty: AbstractFormat, rhs_ty: AbstractFormat, ctx: Context):
        """
        Generate C++ code for division.

        Args:
            lhs_ty: Type of the left-hand side operand.
            rhs_ty: Type of the right-hand side operand.
            ctx: Number context.
        Returns:
            A function that generates C++ code for division.
        """
        ret_ty = _cvt_context(ctx)
        if _fits_in_double(lhs_ty) and _fits_in_double(rhs_ty) and _fits_in_double(ret_ty):
            # use the FP-RTO backed implementation
            return lambda lhs, rhs, ctx: f'fpy::div({lhs}, {rhs}, {ctx})'

        # no suitable implementation found
        return None


class FMAInstr:
    """C++ instruction: fused multiply-add."""

    @staticmethod
    def generator(a_ty: AbstractFormat, b_ty: AbstractFormat, c_ty: AbstractFormat, ctx: Context):
        """
        Generate C++ code for fused multiply-add.

        Args:
            a_ty: Type of the first operand.
            b_ty: Type of the second operand.
            c_ty: Type of the third operand.
            ctx: Number context.
        Returns:
            A function that generates C++ code for fused multiply-add.
        """
        ret_ty = _cvt_context(ctx)
        if (
            _fits_in_double(a_ty)
            and _fits_in_double(b_ty)
            and _fits_in_double(c_ty)
            and _fits_in_double(ret_ty)
        ):
            # use the FP-RTO backed implementation
            return lambda a, b, c, ctx: f'fpy::fma({a}, {b}, {c}, {ctx})'

        # no suitable implementation found
        return None
