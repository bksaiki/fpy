"""
Named :class:`Grammar` configurations for common test scenarios.

Each profile narrows :data:`DEFAULT_GRAMMAR` to the subset of productions a
particular backend pass or analysis exercises.  Tests apply a profile via
``grammar=PROFILE``; new profiles should live next to their consumer
tests or here when they're broadly reusable.
"""

import fpy2 as fp

from .fpy_program import (
    BoolProd,
    DEFAULT_GRAMMAR,
    Grammar,
    ListProd,
    RealProd,
    StmtProd,
    TupleProd,
)


MINIMAL_PROFILE: Grammar = DEFAULT_GRAMMAR.narrow(
    real_prods=RealProd.INTEGER | RealProd.VAR | RealProd.ARITH,
    bool_prods=BoolProd.LEAVES,
    list_prods=ListProd.LITERAL | ListProd.VAR,
    tuple_prods=TupleProd.LITERAL | TupleProd.VAR,
    stmt_prods=StmtProd.ASSIGN,
    contexts=(fp.FP64,),
)
"""Smallest viable grammar — integer arithmetic + variable refs, no
control flow, single ``FP64`` context.  Useful for fast smoke tests that
just need *some* well-typed FPy program."""


ROUND_ELIM_PROFILE: Grammar = DEFAULT_GRAMMAR.narrow(
    real_prods=(
        RealProd.INTEGER | RealProd.VAR
        | RealProd.ADD | RealProd.SUB | RealProd.MUL
        | RealProd.NEG | RealProd.ABS
        | RealProd.ROUND
    ),
    bool_prods=BoolProd.LEAVES,
    list_prods=ListProd.LITERAL | ListProd.VAR,
    tuple_prods=TupleProd.LITERAL | TupleProd.VAR,
    stmt_prods=StmtProd.ASSIGN | StmtProd.WITH,
    contexts=(fp.FP32, fp.FP64),
)
"""Exercises :class:`fpy2.transform.RoundElim`.  Round-heavy real
arithmetic under FP32/FP64 ``with`` blocks; no ``IfExpr``, no
transcendentals, no division, no loops."""


WIDENING_PROFILE: Grammar = DEFAULT_GRAMMAR.narrow(
    real_prods=(
        RealProd.INTEGER | RealProd.VAR
        | RealProd.ADD | RealProd.SUB | RealProd.MUL
        | RealProd.ROUND
    ),
    bool_prods=BoolProd.LEAVES,
    list_prods=ListProd.LITERAL | ListProd.VAR,
    tuple_prods=TupleProd.LITERAL | TupleProd.VAR,
    stmt_prods=StmtProd.ASSIGN | StmtProd.WITH,
    contexts=(fp.SINT8, fp.SINT16, fp.FP32),
)
"""Exercises the cpp emitter's lossless-widening dispatch.  Integer-typed
fixed-point contexts plus a wider FP32 the products can fit into.  No
``div`` (would explode into rationals); no transcendentals (would
require wider storage than any cpp scalar)."""


LOOP_HEAVY_PROFILE: Grammar = DEFAULT_GRAMMAR.narrow(
    real_prods=RealProd.INTEGER | RealProd.VAR | RealProd.ARITH | RealProd.ROUND,
    bool_prods=BoolProd.LEAVES | BoolProd.COMPARE,
    list_prods=ListProd.LITERAL | ListProd.VAR | ListProd.RANGE,
    tuple_prods=TupleProd.LITERAL | TupleProd.VAR,
    stmt_prods=StmtProd.ASSIGN | StmtProd.FOR | StmtProd.WHILE,
    contexts=(fp.FP32, fp.FP64),
)
"""For- and while-heavy bodies — exercises loop-fixpoint convergence in
:mod:`fpy2.analysis.format_infer` and the cpp emitter's per-iteration
storage stability check.  ``IF`` is intentionally omitted to keep the
loop-vs-other-control-flow signal clean."""


CONTROL_FLOW_PROFILE: Grammar = DEFAULT_GRAMMAR.narrow(
    real_prods=RealProd.INTEGER | RealProd.VAR | RealProd.ARITH | RealProd.IF_EXPR,
    bool_prods=BoolProd.LEAVES | BoolProd.COMPARE | BoolProd.NOT | BoolProd.AND_OR,
    list_prods=ListProd.LITERAL | ListProd.VAR,
    tuple_prods=TupleProd.LITERAL | TupleProd.VAR,
    stmt_prods=StmtProd.ASSIGN | StmtProd.IF | StmtProd.FOR | StmtProd.WHILE,
    contexts=(fp.FP32, fp.FP64),
)
"""Branches, predicates, and loops — exercises def-use phi merging,
context-use scope tracking, and the storage selector's hoist-before
logic for ``IfStmt`` introductions."""


FRACTIONAL_PROFILE: Grammar = DEFAULT_GRAMMAR.narrow(
    real_prods=(
        DEFAULT_GRAMMAR.real_prods
        | RealProd.DECNUM | RealProd.HEXNUM | RealProd.RATIONAL
    ),
)
"""Opts every literal kind in — ``Decnum``, ``Hexnum``, ``Rational`` join
``Integer``.  Use for tests that want to surface non-dyadic / non-finite
literal-handling paths.  Pays the ~3× draw-time overhead from
:data:`_SLOW_REAL` in exchange for that coverage."""


__all__ = [
    'MINIMAL_PROFILE',
    'ROUND_ELIM_PROFILE',
    'WIDENING_PROFILE',
    'LOOP_HEAVY_PROFILE',
    'CONTROL_FLOW_PROFILE',
    'FRACTIONAL_PROFILE',
]
