"""
Which solver backend a store gets.

z3 is the default.  ``z3-opt``, z3 finding each maximum with ``Optimize``
rather than by bisection, exists to be measured against it; so may others a
caller adds to :data:`BACKENDS`.
The environment variable ``FPY_DIGIT_BOUND_SOLVER`` names the default, so a
test suite can run under any backend unchanged; :func:`solver_backend`
chooses one for a block of code, and keeps what it makes.
"""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from .solver import MeasuredSolver, Solver
from .z3_solver import Z3Solver

__all__ = [
    'BACKENDS',
    'ENV_VAR',
    'default_solver',
    'make_solver',
    'solver_backend',
]


def _z3_opt(**opts: Any) -> MeasuredSolver:
    return Z3Solver(bisect=False, **opts)


BACKENDS: dict[str, Callable[..., MeasuredSolver]] = {
    'z3': Z3Solver,
    'z3-opt': _z3_opt,
}
"""Each backend's constructor, by name."""

ENV_VAR = 'FPY_DIGIT_BOUND_SOLVER'


def make_solver(name: str, **opts: Any) -> MeasuredSolver:
    """A fresh solver of backend *name*, built with *opts*."""
    make = BACKENDS.get(name)
    if make is None:
        raise ValueError(f'unknown solver backend {name!r}; expected one of {sorted(BACKENDS)}')
    return make(**opts)


_factory: ContextVar[Callable[[], MeasuredSolver] | None] = ContextVar('_factory', default=None)
"""What :func:`solver_backend` chose, where it is in effect."""


def default_solver() -> Solver:
    """The solver a store gets when it is not handed one: what
    :func:`solver_backend` chose, else what ``FPY_DIGIT_BOUND_SOLVER`` names,
    else z3."""
    factory = _factory.get()
    if factory is not None:
        return factory()
    return make_solver(os.environ.get(ENV_VAR) or 'z3')


@contextmanager
def solver_backend(name: str, **opts: Any) -> Iterator[list[MeasuredSolver]]:
    """Give every store made inside the block a solver of backend *name*.

    Yields the list of solvers made, which grows as they are; their
    :attr:`~MeasuredSolver.stats` together are what the block cost::

        with solver_backend('z3-opt') as solvers:
            compile_design(build)
        total = sum((s.stats for s in solvers), SolverStats())
    """
    made: list[MeasuredSolver] = []

    def factory() -> MeasuredSolver:
        s = make_solver(name, **opts)
        made.append(s)
        return s

    token = _factory.set(factory)
    try:
        yield made
    finally:
        _factory.reset(token)
