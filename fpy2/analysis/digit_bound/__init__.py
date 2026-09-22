"""Digit-bound inference: a relational analysis over integer exponents.

Answers one question -- how many binades separate a value from the position it
is rounded at -- which is the precision of the resulting fixed-point value.
See ``docs/todos/digit-bound-inference.md``.

:mod:`solver` is the constraint language and the backend that answers about
it; :mod:`store` is the domain and its queries, and walks no AST; :mod:`infer`
is the pass over a program, which reads ranges from the format domain through
a protocol and hands back precisions it does not itself turn into formats.
"""

from .infer import (
    Bounds,
    DigitBoundAnalysis,
    DigitBoundInfer,
    DigitBoundParams,
    FormatView,
    Terms,
)
from .solver import Constraint, Solver, Term, Z3Solver
from .store import DigitBoundStore

__all__ = [
    'Bounds',
    'Constraint',
    'DigitBoundAnalysis',
    'DigitBoundInfer',
    'DigitBoundParams',
    'DigitBoundStore',
    'FormatView',
    'Solver',
    'Term',
    'Terms',
    'Z3Solver',
]
