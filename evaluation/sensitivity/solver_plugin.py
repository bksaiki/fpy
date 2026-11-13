"""
A pytest plugin that runs a test suite with every digit-bound store on
another backend.  From `evaluation/`:

    PYTHONPATH=sensitivity uv run pytest -p solver_plugin --digit-bound-solver cvc5 ...
"""

import pytest
from backends import BACKENDS

from fpy2.analysis.digit_bound import solver_factory


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption('--digit-bound-solver', choices=list(BACKENDS),
                     help='the backend every digit-bound store gets, unless handed one')


@pytest.fixture(autouse=True)
def _digit_bound_solver(request: pytest.FixtureRequest):
    name = request.config.getoption('--digit-bound-solver')
    if name is None:
        yield
        return
    with solver_factory(BACKENDS[name]):
        yield
