"""
C++ backend: target description
"""

import dataclasses

from typing import TypeAlias

from ...ast import *
from ...libraries.core import logb
from ...primitive import Primitive

from .types import CppType, CppScalar, ALL_SCALARS, FLOAT_TYPES, INT_TYPES


@dataclasses.dataclass
class UnaryCppOp:
    name: str
    arg: CppType
    ret: CppType

    def matches(self, arg: CppType, ret: CppType) -> bool:
        return self.arg == arg and self.ret == ret

    def format(self, arg: str) -> str:
        return f'{self.name}({arg})'


@dataclasses.dataclass
class BinaryCppOp:
    name: str
    is_infix: bool
    lhs: CppType
    rhs: CppType
    ret: CppType

    def matches(self, lhs: CppType, rhs: CppType, ret: CppType) -> bool:
        return self.lhs == lhs and self.rhs == rhs and self.ret == ret

    def format(self, lhs: str, rhs: str) -> str:
        if self.is_infix:
            return f'({lhs} {self.name} {rhs})'
        else:
            return f'{self.name}({lhs}, {rhs})'


@dataclasses.dataclass
class TernaryCppOp:
    name: str
    arg1: CppType
    arg2: CppType
    arg3: CppType
    ret: CppType

    def matches(self, arg1: CppType, arg2: CppType, arg3: CppType, ret: CppType) -> bool:
        return self.arg1 == arg1 and self.arg2 == arg2 and self.arg3 == arg3 and self.ret == ret

    def format(self, arg1: str, arg2: str, arg3: str) -> str:
        return f'{self.name}({arg1}, {arg2}, {arg3})'


UnaryOpTable: TypeAlias = dict[type[Expr], list[UnaryCppOp]]
BinaryOpTable: TypeAlias = dict[type[Expr], list[BinaryCppOp]]
TernaryOpTable: TypeAlias = dict[type[Expr], list[TernaryCppOp]]
PrimitiveTable: TypeAlias = dict[Primitive, list[UnaryCppOp | BinaryCppOp | TernaryCppOp]]

@dataclasses.dataclass
class ScalarOpTable:
    unary: UnaryOpTable
    binary: BinaryOpTable
    ternary: TernaryOpTable
    prims: PrimitiveTable


def _make_unary_table() -> UnaryOpTable:
    return {
        # Sign operations
        Neg: [
            UnaryCppOp('-', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('-', CppScalar.F32, CppScalar.F32),
        ],
        Fabs: [
            UnaryCppOp('std::abs', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::abs', CppScalar.F32, CppScalar.F32),
        ],

        # Rounding and truncation
        Ceil: [
            UnaryCppOp('std::ceil', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::ceil', CppScalar.F32, CppScalar.F32),
        ],
        Floor: [
            UnaryCppOp('std::floor', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::floor', CppScalar.F32, CppScalar.F32),
        ],
        Trunc: [
            UnaryCppOp('std::trunc', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::trunc', CppScalar.F32, CppScalar.F32),
        ],
        RoundInt: [
            UnaryCppOp('std::round', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::round', CppScalar.F32, CppScalar.F32),
        ],
        NearbyInt: [
            UnaryCppOp('std::nearbyint', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::nearbyint', CppScalar.F32, CppScalar.F32),
        ],

        # Square root and cube root
        Sqrt: [
            UnaryCppOp('std::sqrt', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::sqrt', CppScalar.F32, CppScalar.F32),
        ],
        Cbrt: [
            UnaryCppOp('std::cbrt', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::cbrt', CppScalar.F32, CppScalar.F32),
        ],

        # Trigonometric functions
        Sin: [
            UnaryCppOp('std::sin', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::sin', CppScalar.F32, CppScalar.F32),
        ],
        Cos: [
            UnaryCppOp('std::cos', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::cos', CppScalar.F32, CppScalar.F32),
        ],
        Tan: [
            UnaryCppOp('std::tan', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::tan', CppScalar.F32, CppScalar.F32),
        ],
        Asin: [
            UnaryCppOp('std::asin', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::asin', CppScalar.F32, CppScalar.F32),
        ],
        Acos: [
            UnaryCppOp('std::acos', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::acos', CppScalar.F32, CppScalar.F32),
        ],
        Atan: [
            UnaryCppOp('std::atan', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::atan', CppScalar.F32, CppScalar.F32),
        ],
        
        # Hyperbolic functions
        Sinh: [
            UnaryCppOp('std::sinh', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::sinh', CppScalar.F32, CppScalar.F32),
        ],
        Cosh: [
            UnaryCppOp('std::cosh', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::cosh', CppScalar.F32, CppScalar.F32),
        ],
        Tanh: [
            UnaryCppOp('std::tanh', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::tanh', CppScalar.F32, CppScalar.F32),
        ],
        Asinh: [
            UnaryCppOp('std::asinh', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::asinh', CppScalar.F32, CppScalar.F32),
        ],
        Acosh: [
            UnaryCppOp('std::acosh', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::acosh', CppScalar.F32, CppScalar.F32),
        ],
        Atanh: [
            UnaryCppOp('std::atanh', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::atanh', CppScalar.F32, CppScalar.F32),
        ],
        
        # Exponential and logarithmic functions
        Exp: [
            UnaryCppOp('std::exp', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::exp', CppScalar.F32, CppScalar.F32),
        ],
        Exp2: [
            UnaryCppOp('std::exp2', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::exp2', CppScalar.F32, CppScalar.F32),
        ],
        Expm1: [
            UnaryCppOp('std::expm1', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::expm1', CppScalar.F32, CppScalar.F32),
        ],
        Log: [
            UnaryCppOp('std::log', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::log', CppScalar.F32, CppScalar.F32),
        ],
        Log10: [
            UnaryCppOp('std::log10', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::log10', CppScalar.F32, CppScalar.F32),
        ],
        Log2: [
            UnaryCppOp('std::log2', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::log2', CppScalar.F32, CppScalar.F32),
        ],
        Log1p: [
            UnaryCppOp('std::log1p', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::log1p', CppScalar.F32, CppScalar.F32),
        ],
        
        # Special functions
        Erf: [
            UnaryCppOp('std::erf', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::erf', CppScalar.F32, CppScalar.F32),
        ],
        Erfc: [
            UnaryCppOp('std::erfc', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::erfc', CppScalar.F32, CppScalar.F32),
        ],
        Lgamma: [
            UnaryCppOp('std::lgamma', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::lgamma', CppScalar.F32, CppScalar.F32),
        ],
        Tgamma: [
            UnaryCppOp('std::tgamma', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::tgamma', CppScalar.F32, CppScalar.F32),
        ],
        
        # Classification functions (return bool)
        IsFinite: [
            UnaryCppOp('std::isfinite', CppScalar.F64, CppScalar.BOOL),
            UnaryCppOp('std::isfinite', CppScalar.F32, CppScalar.BOOL),
        ],
        IsInf: [
            UnaryCppOp('std::isinf', CppScalar.F64, CppScalar.BOOL),
            UnaryCppOp('std::isinf', CppScalar.F32, CppScalar.BOOL),
        ],
        IsNan: [
            UnaryCppOp('std::isnan', CppScalar.F64, CppScalar.BOOL),
            UnaryCppOp('std::isnan', CppScalar.F32, CppScalar.BOOL),
        ],
        IsNormal: [
            UnaryCppOp('std::isnormal', CppScalar.F64, CppScalar.BOOL),
            UnaryCppOp('std::isnormal', CppScalar.F32, CppScalar.BOOL),
        ],
        Signbit: [
            UnaryCppOp('std::signbit', CppScalar.F64, CppScalar.BOOL),
            UnaryCppOp('std::signbit', CppScalar.F32, CppScalar.BOOL),
        ],

        # Rounding operations
        Round: [
            UnaryCppOp(f'static_cast<{ret_ty.format()}>', arg_ty, ret_ty)
            for arg_ty in ALL_SCALARS
            for ret_ty in ALL_SCALARS
        ],

        # Logical operations
        Not: [
            UnaryCppOp('!', CppScalar.BOOL, CppScalar.BOOL),
        ],
    }

def _make_binary_table() -> BinaryOpTable:
    return {
        # Basic arithmetic
        Add: [
            BinaryCppOp('+', True, ty, ty, ty)
            for ty in FLOAT_TYPES + INT_TYPES
        ],
        Sub: [
            BinaryCppOp('-', True, ty, ty, ty)
            for ty in FLOAT_TYPES + INT_TYPES
        ],
        Mul: [
            BinaryCppOp('*', True, ty, ty, ty)
            for ty in FLOAT_TYPES + INT_TYPES
        ],
        Div: [
            BinaryCppOp('/', True, ty, ty, ty)
            for ty in FLOAT_TYPES + INT_TYPES
        ],

        # Min/Max operations
        # these are technically N-ary operations, but the
        # C++ backend reduces them down to 2-ary
        Min: [
            BinaryCppOp('std::fmin', False, ty, ty, ty)
            for ty in FLOAT_TYPES
        ] + [
            BinaryCppOp('std::min', False, ty, ty, ty)
            for ty in INT_TYPES
        ],
        Max: [
            BinaryCppOp('std::fmax', False, ty, ty, ty)
            for ty in FLOAT_TYPES
        ] + [
            BinaryCppOp('std::max', False, ty, ty, ty)
            for ty in INT_TYPES
        ],

        # Power operations
        Pow: [
            BinaryCppOp('std::pow', False, CppScalar.F64, CppScalar.F64, CppScalar.F64),
            BinaryCppOp('std::pow', False, CppScalar.F32, CppScalar.F32, CppScalar.F32),
        ],

        # Modulus operations
        Fmod: [
            BinaryCppOp('std::fmod', False, CppScalar.F64, CppScalar.F64, CppScalar.F64),
            BinaryCppOp('std::fmod', False, CppScalar.F32, CppScalar.F32, CppScalar.F32),
        ],
        Remainder: [
            BinaryCppOp('std::remainder', False, CppScalar.F64, CppScalar.F64, CppScalar.F64),
            BinaryCppOp('std::remainder', False, CppScalar.F32, CppScalar.F32, CppScalar.F32),
        ],

        # Sign operations
        Copysign: [
            BinaryCppOp('std::copysign', False, CppScalar.F64, CppScalar.F64, CppScalar.F64),
            BinaryCppOp('std::copysign', False, CppScalar.F32, CppScalar.F32, CppScalar.F32),
        ],

        # Composite arithmetic
        Fdim: [
            BinaryCppOp('std::fdim', False, CppScalar.F64, CppScalar.F64, CppScalar.F64),
            BinaryCppOp('std::fdim', False, CppScalar.F32, CppScalar.F32, CppScalar.F32),
        ],
        Hypot: [
            BinaryCppOp('std::hypot', False, CppScalar.F64, CppScalar.F64, CppScalar.F64),
            BinaryCppOp('std::hypot', False, CppScalar.F32, CppScalar.F32, CppScalar.F32),
        ],

        # Trigonometric functions
        Atan2: [
            BinaryCppOp('std::atan2', False, CppScalar.F64, CppScalar.F64, CppScalar.F64),
            BinaryCppOp('std::atan2', False, CppScalar.F32, CppScalar.F32, CppScalar.F32),
        ],
    }

def _make_ternary_table() -> TernaryOpTable:
    return {
        # Fused multiply-add
        Fma: [
            TernaryCppOp('std::fma', CppScalar.F64, CppScalar.F64, CppScalar.F64, CppScalar.F64),
            TernaryCppOp('std::fma', CppScalar.F32, CppScalar.F32, CppScalar.F32, CppScalar.F32),
        ],
    }


class IlogbOp(UnaryCppOp):
    """`std::ilogb` returns `int`, so we need to cast it to `int64_t`."""

    def __init__(self, arg_ty: CppType, ret_ty: CppType):
        super().__init__('std::ilogb', arg_ty, ret_ty)

    def format(self, arg: str) -> str:
        return f'static_cast<int64_t>(std::ilogb({arg}))'

def _make_primitive_table() -> PrimitiveTable:
    return {
        logb: [
            UnaryCppOp('std::logb', CppScalar.F64, CppScalar.F64),
            UnaryCppOp('std::logb', CppScalar.F32, CppScalar.F32),
            IlogbOp(CppScalar.F64, CppScalar.S64), # technically returns `int`
            IlogbOp(CppScalar.F32, CppScalar.S64),
        ]
    }

def make_op_table() -> ScalarOpTable:
    return ScalarOpTable(
        unary=_make_unary_table(),
        binary=_make_binary_table(),
        ternary=_make_ternary_table(),
        prims=_make_primitive_table()
    )
