"""
Custom generators over FPy types for Hypothesis tests.
"""

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
    stmt_block,
    fpy_real_funcdef,
    fpy_real_function,
)
