"""
Custom generators over FPy types for Hypothesis tests.
"""

__all__ = [
    # Context strategies
    'mp_float_contexts', 'mps_float_contexts', 'ieee_contexts',
    'efloat_contexts', 'exp_contexts',
    'fixed_contexts', 'mp_fixed_contexts', 'sm_fixed_contexts',
    'common_contexts',
    # Rounding-mode / overflow strategies
    'overflow_modes', 'rounding_modes',
    # Number strategies
    'real_floats', 'floats',
    # FPy program generators
    'TypeEnv',
    'expr', 'real_expr', 'bool_expr', 'list_expr', 'tuple_expr', 'context_expr',
    'stmt_block',
    'fpy_funcdef', 'fpy_function',
    'fpy_real_funcdef', 'fpy_real_function',
    'arbitrary_type', 'value_for_type',
    # Grammar surface
    'Grammar', 'DEFAULT_GRAMMAR',
    'RealProd', 'BoolProd', 'ListProd', 'TupleProd', 'ContextProd', 'StmtProd',
]

from .context import (
    # floating-point contexts
    mp_float_contexts,
    mps_float_contexts,
    ieee_contexts,
    efloat_contexts,
    exp_contexts,
    # fixed-point contexts
    fixed_contexts,
    mp_fixed_contexts,
    sm_fixed_contexts,
    # common contexts
    common_contexts,
)

from .round import overflow_modes, rounding_modes
from .number import real_floats, floats

from .fpy_program import (
    TypeEnv,
    expr,
    real_expr,
    bool_expr,
    list_expr,
    tuple_expr,
    context_expr,
    stmt_block,
    fpy_funcdef,
    fpy_function,
    fpy_real_funcdef,
    fpy_real_function,
    arbitrary_type,
    value_for_type,
    # Grammar surface
    Grammar,
    DEFAULT_GRAMMAR,
    RealProd,
    BoolProd,
    ListProd,
    TupleProd,
    ContextProd,
    StmtProd,
)
