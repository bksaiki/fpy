"""Digit-bound inference: a relational analysis over integer exponents.

Answers one question -- how many binades separate a value from the position it
is rounded at -- which is the precision of the resulting fixed-point value.
See ``docs/todos/digit-bound.md``.

:mod:`solver` is the constraint language, the interface to a solver and
what every backend shares; :mod:`z3_solver` the backend,
:mod:`backends` which one a store gets, :mod:`store` the domain
and its queries, and :mod:`infer` the pass over a program.
"""

from .backends import default_solver, solver_factory
from .infer import (
    Bounds,
    DigitBoundAnalysis,
    DigitBoundInfer,
    DigitBoundParams,
    FormatView,
    Terms,
)
from .solver import Constraint, MeasuredSolver, QueryRecord, Solver, SolverStats, Term
from .store import DigitBoundStore
from .z3_solver import Z3Solver

__all__ = [
    'Bounds',
    'Constraint',
    'DigitBoundAnalysis',
    'DigitBoundInfer',
    'DigitBoundParams',
    'DigitBoundStore',
    'FormatView',
    'MeasuredSolver',
    'QueryRecord',
    'Solver',
    'SolverStats',
    'Term',
    'Terms',
    'Z3Solver',
    'default_solver',
    'solver_factory',
]
