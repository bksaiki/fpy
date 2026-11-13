"""
Which solver a store gets when it is not handed one: z3, unless
:func:`solver_factory` chooses otherwise for a block of code.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar

from .solver import Solver
from .z3_solver import Z3Solver

__all__ = ['default_solver', 'solver_factory']

_S = TypeVar('_S', bound=Solver)

_factory: ContextVar[Callable[[], Solver] | None] = ContextVar('_factory', default=None)
"""What :func:`solver_factory` chose, where it is in effect."""


def default_solver() -> Solver:
    """What :func:`solver_factory` chose, else z3."""
    make = _factory.get()
    return make() if make is not None else Z3Solver()


@contextmanager
def solver_factory(make: Callable[[], _S]) -> Generator[list[_S], None, None]:
    """Give every store made inside the block a solver from *make*.

    Yields the list of solvers made, which grows as they are; their
    :attr:`~MeasuredSolver.stats` together are what the block cost::

        with solver_factory(Z3Solver) as solvers:
            compile_design(build)
        total = sum((s.stats for s in solvers), SolverStats())
    """
    made: list[_S] = []

    def factory() -> _S:
        s = make()
        made.append(s)
        return s

    token = _factory.set(factory)
    try:
        yield made
    finally:
        _factory.reset(token)
