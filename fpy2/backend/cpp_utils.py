"""
C++ compilation utilities.
"""

import enum
import dataclasses

from typing import TypeAlias

from ..ast import *
from ..utils import enum_repr

###########################################################
# C++ (scalar) type

@enum_repr
class CppType(enum.Enum):
    """
    C++ types.

    Each type represents either

    t ::= bool | real R

    where R is a concrete rounding context.
    """

    BOOL = 0
    FLOAT = 1
    DOUBLE = 2
    U8 = 3
    U16 = 4
    U32 = 5
    U64 = 6
    S8 = 7
    S16 = 8
    S32 = 9
    S64 = 10

    @property
    def cpp_name(self):
        match self:
            case CppType.BOOL:
                return 'bool'
            case CppType.FLOAT:
                return 'float'
            case CppType.DOUBLE:
                return 'double'
            case CppType.U8:
                return 'uint8_t'
            case CppType.U16:
                return 'uint16_t'
            case CppType.U32:
                return 'uint32_t'
            case CppType.U64:
                return 'uint64_t'
            case CppType.S8:
                return 'int8_t'
            case CppType.S16:
                return 'int16_t'
            case CppType.S32:
                return 'int32_t'
            case CppType.S64:
                return 'int64_t'

_FLOAT_TYPES = [
    CppType.FLOAT,
    CppType.DOUBLE
]

_INT_TYPES = [
    CppType.S8,
    CppType.S16,
    CppType.S32,
    CppType.S64,
    CppType.U8,
    CppType.U16,
    CppType.U32,
    CppType.U64
]

_ALL_TYPES = [CppType.BOOL] + _FLOAT_TYPES + _INT_TYPES

###########################################################
# C++ operation table

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

@dataclasses.dataclass
class ScalarOpTable:
    unary: UnaryOpTable
    binary: BinaryOpTable
    ternary: TernaryOpTable


def _make_unary_table() -> UnaryOpTable:
    return {
        # Sign operations
        Neg: [
            UnaryCppOp('-', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('-', CppType.FLOAT, CppType.FLOAT),
        ],
        Fabs: [
            UnaryCppOp('std::abs', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::abs', CppType.FLOAT, CppType.FLOAT),
        ],

        # Rounding and truncation
        Ceil: [
            UnaryCppOp('std::ceil', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::ceil', CppType.FLOAT, CppType.FLOAT),
        ],
        Floor: [
            UnaryCppOp('std::floor', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::floor', CppType.FLOAT, CppType.FLOAT),
        ],
        Trunc: [
            UnaryCppOp('std::trunc', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::trunc', CppType.FLOAT, CppType.FLOAT),
        ],
        RoundInt: [
            UnaryCppOp('std::round', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::round', CppType.FLOAT, CppType.FLOAT),
        ],
        NearbyInt: [
            UnaryCppOp('std::nearbyint', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::nearbyint', CppType.FLOAT, CppType.FLOAT),
        ],
        
        # Square root and cube root
        Sqrt: [
            UnaryCppOp('std::sqrt', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::sqrt', CppType.FLOAT, CppType.FLOAT),
        ],
        Cbrt: [
            UnaryCppOp('std::cbrt', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::cbrt', CppType.FLOAT, CppType.FLOAT),
        ],
        
        # Trigonometric functions
        Sin: [
            UnaryCppOp('std::sin', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::sin', CppType.FLOAT, CppType.FLOAT),
        ],
        Cos: [
            UnaryCppOp('std::cos', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::cos', CppType.FLOAT, CppType.FLOAT),
        ],
        Tan: [
            UnaryCppOp('std::tan', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::tan', CppType.FLOAT, CppType.FLOAT),
        ],
        Asin: [
            UnaryCppOp('std::asin', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::asin', CppType.FLOAT, CppType.FLOAT),
        ],
        Acos: [
            UnaryCppOp('std::acos', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::acos', CppType.FLOAT, CppType.FLOAT),
        ],
        Atan: [
            UnaryCppOp('std::atan', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::atan', CppType.FLOAT, CppType.FLOAT),
        ],
        
        # Hyperbolic functions
        Sinh: [
            UnaryCppOp('std::sinh', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::sinh', CppType.FLOAT, CppType.FLOAT),
        ],
        Cosh: [
            UnaryCppOp('std::cosh', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::cosh', CppType.FLOAT, CppType.FLOAT),
        ],
        Tanh: [
            UnaryCppOp('std::tanh', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::tanh', CppType.FLOAT, CppType.FLOAT),
        ],
        Asinh: [
            UnaryCppOp('std::asinh', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::asinh', CppType.FLOAT, CppType.FLOAT),
        ],
        Acosh: [
            UnaryCppOp('std::acosh', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::acosh', CppType.FLOAT, CppType.FLOAT),
        ],
        Atanh: [
            UnaryCppOp('std::atanh', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::atanh', CppType.FLOAT, CppType.FLOAT),
        ],
        
        # Exponential and logarithmic functions
        Exp: [
            UnaryCppOp('std::exp', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::exp', CppType.FLOAT, CppType.FLOAT),
        ],
        Exp2: [
            UnaryCppOp('std::exp2', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::exp2', CppType.FLOAT, CppType.FLOAT),
        ],
        Expm1: [
            UnaryCppOp('std::expm1', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::expm1', CppType.FLOAT, CppType.FLOAT),
        ],
        Log: [
            UnaryCppOp('std::log', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::log', CppType.FLOAT, CppType.FLOAT),
        ],
        Log10: [
            UnaryCppOp('std::log10', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::log10', CppType.FLOAT, CppType.FLOAT),
        ],
        Log2: [
            UnaryCppOp('std::log2', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::log2', CppType.FLOAT, CppType.FLOAT),
        ],
        Log1p: [
            UnaryCppOp('std::log1p', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::log1p', CppType.FLOAT, CppType.FLOAT),
        ],
        
        # Special functions
        Erf: [
            UnaryCppOp('std::erf', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::erf', CppType.FLOAT, CppType.FLOAT),
        ],
        Erfc: [
            UnaryCppOp('std::erfc', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::erfc', CppType.FLOAT, CppType.FLOAT),
        ],
        Lgamma: [
            UnaryCppOp('std::lgamma', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::lgamma', CppType.FLOAT, CppType.FLOAT),
        ],
        Tgamma: [
            UnaryCppOp('std::tgamma', CppType.DOUBLE, CppType.DOUBLE),
            UnaryCppOp('std::tgamma', CppType.FLOAT, CppType.FLOAT),
        ],
        
        # Classification functions (return bool)
        IsFinite: [
            UnaryCppOp('std::isfinite', CppType.DOUBLE, CppType.BOOL),
            UnaryCppOp('std::isfinite', CppType.FLOAT, CppType.BOOL),
        ],
        IsInf: [
            UnaryCppOp('std::isinf', CppType.DOUBLE, CppType.BOOL),
            UnaryCppOp('std::isinf', CppType.FLOAT, CppType.BOOL),
        ],
        IsNan: [
            UnaryCppOp('std::isnan', CppType.DOUBLE, CppType.BOOL),
            UnaryCppOp('std::isnan', CppType.FLOAT, CppType.BOOL),
        ],
        IsNormal: [
            UnaryCppOp('std::isnormal', CppType.DOUBLE, CppType.BOOL),
            UnaryCppOp('std::isnormal', CppType.FLOAT, CppType.BOOL),
        ],
        Signbit: [
            UnaryCppOp('std::signbit', CppType.DOUBLE, CppType.BOOL),
            UnaryCppOp('std::signbit', CppType.FLOAT, CppType.BOOL),
        ],

        # Rounding operations
        Round: [
            UnaryCppOp(f'static_cast<{ret_ty.cpp_name}>', arg_ty, ret_ty)
            for arg_ty in _ALL_TYPES
            for ret_ty in _ALL_TYPES
        ],

        # Logical operations
        Not: [
            UnaryCppOp('!', CppType.BOOL, CppType.BOOL),
        ],
    }

def _make_binary_table() -> BinaryOpTable:
    return {
        # Basic arithmetic
        Add: [
            BinaryCppOp('+', True, ty, ty, ty)
            for ty in _FLOAT_TYPES + _INT_TYPES
        ],
        Sub: [
            BinaryCppOp('-', True, ty, ty, ty)
            for ty in _FLOAT_TYPES + _INT_TYPES
        ],
        Mul: [
            BinaryCppOp('*', True, ty, ty, ty)
            for ty in _FLOAT_TYPES + _INT_TYPES
        ],
        Div: [
            BinaryCppOp('/', True, ty, ty, ty)
            for ty in _FLOAT_TYPES + _INT_TYPES
        ],

        # Min/Max operations
        # NOTE: std::min and std::max don't handle NaN properly for IEEE 754,
        # so we use std::fmin and std::fmax
        Min: [
            BinaryCppOp('std::fmin', False, ty, ty, ty)
            for ty in _FLOAT_TYPES
        ] + [
            BinaryCppOp('std::min', False, ty, ty, ty)
            for ty in _INT_TYPES
        ],
        Max: [
            BinaryCppOp('std::fmax', False, ty, ty, ty)
            for ty in _FLOAT_TYPES
        ] + [
            BinaryCppOp('std::max', False, ty, ty, ty)
            for ty in _INT_TYPES
        ],

        # Power operations
        Pow: [
            BinaryCppOp('std::pow', False, CppType.DOUBLE, CppType.DOUBLE, CppType.DOUBLE),
            BinaryCppOp('std::pow', False, CppType.FLOAT, CppType.FLOAT, CppType.FLOAT),
        ],

        # Modulus operations
        Fmod: [
            BinaryCppOp('std::fmod', False, CppType.DOUBLE, CppType.DOUBLE, CppType.DOUBLE),
            BinaryCppOp('std::fmod', False, CppType.FLOAT, CppType.FLOAT, CppType.FLOAT),
        ],
        Remainder: [
            BinaryCppOp('std::remainder', False, CppType.DOUBLE, CppType.DOUBLE, CppType.DOUBLE),
            BinaryCppOp('std::remainder', False, CppType.FLOAT, CppType.FLOAT, CppType.FLOAT),
        ],

        # Sign operations
        Copysign: [
            BinaryCppOp('std::copysign', False, CppType.DOUBLE, CppType.DOUBLE, CppType.DOUBLE),
            BinaryCppOp('std::copysign', False, CppType.FLOAT, CppType.FLOAT, CppType.FLOAT),
        ],

        # Composite arithmetic
        Fdim: [
            BinaryCppOp('std::fdim', False, CppType.DOUBLE, CppType.DOUBLE, CppType.DOUBLE),
            BinaryCppOp('std::fdim', False, CppType.FLOAT, CppType.FLOAT, CppType.FLOAT),
        ],
        Hypot: [
            BinaryCppOp('std::hypot', False, CppType.DOUBLE, CppType.DOUBLE, CppType.DOUBLE),
            BinaryCppOp('std::hypot', False, CppType.FLOAT, CppType.FLOAT, CppType.FLOAT),
        ],

        # Trigonometric functions
        Atan2: [
            BinaryCppOp('std::atan2', False, CppType.DOUBLE, CppType.DOUBLE, CppType.DOUBLE),
            BinaryCppOp('std::atan2', False, CppType.FLOAT, CppType.FLOAT, CppType.FLOAT),
        ],
    }

def _make_ternary_table() -> TernaryOpTable:
    return {
        # Fused multiply-add
        Fma: [
            TernaryCppOp('std::fma', CppType.DOUBLE, CppType.DOUBLE, CppType.DOUBLE, CppType.DOUBLE),
            TernaryCppOp('std::fma', CppType.FLOAT, CppType.FLOAT, CppType.FLOAT, CppType.FLOAT),
        ],
    }

def make_op_table() -> ScalarOpTable:
    return ScalarOpTable(
        unary=_make_unary_table(),
        binary=_make_binary_table(),
        ternary=_make_ternary_table(),
    )
