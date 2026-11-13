"""The digit-bound solver backends the sensitivity evaluation compares, by name."""

from collections.abc import Callable

from cvc5_solver import Cvc5Solver
from z3_opt_solver import Z3OptSolver

from fpy2.analysis.digit_bound import MeasuredSolver, Z3Solver

__all__ = ['BACKENDS']

BACKENDS: dict[str, Callable[..., MeasuredSolver]] = {
    'z3': Z3Solver,
    'z3-opt': Z3OptSolver,
    'cvc5': Cvc5Solver,
}
"""Each backend's constructor: `z3`, the default, finds a maximum by
bisection; `z3-opt` by z3's `Optimize`; `cvc5` by bisection."""
