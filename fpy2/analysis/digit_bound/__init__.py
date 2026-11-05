"""Digit-bound inference: a relational analysis over integer exponents.

Answers one question -- how many binades separate a value from the position it
is rounded at -- which is the precision of the resulting fixed-point value.
See ``docs/todos/digit-bound.md``.

:mod:`solver` is the constraint language and its backends, :mod:`backends`
which one a store gets, :mod:`store` the domain and its queries, and
:mod:`infer` the pass over a program.
"""

from .backends import BACKENDS, default_solver, make_solver, solver_backend
from .infer import (
    Bounds,
    DigitBoundAnalysis,
    DigitBoundInfer,
    DigitBoundParams,
    FormatView,
    Terms,
)
from .solver import Constraint, Solver, SolverStats, Term, Z3Solver
from .store import DigitBoundStore

__all__ = [
    'BACKENDS',
    'Bounds',
    'Constraint',
    'DigitBoundAnalysis',
    'DigitBoundInfer',
    'DigitBoundParams',
    'DigitBoundStore',
    'FormatView',
    'Solver',
    'SolverStats',
    'Term',
    'Terms',
    'Z3Solver',
    'default_solver',
    'make_solver',
    'solver_backend',
]
