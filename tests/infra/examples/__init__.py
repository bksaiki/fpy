import fpy2 as fp

from . import lod
from . import misc
from . import unit

__all__ = [
    'all_unit_tests',
    'all_example_tests',
    'all_tests',
]

_example_modules = [
    lod,
    misc,
]

_unit_tests: list[fp.Function] = []
_example_tests: list[fp.Function] = []

def _load_tests():
    global _unit_tests
    for v in unit.__dict__.values():
        if isinstance(v, fp.Function):
            _unit_tests.append(v)

    global _example_tests
    for m in _example_modules:
        for v in m.__dict__.values():
            if isinstance(v, fp.Function) and not v.name.startswith('_'):
                _example_tests.append(v)


def all_unit_tests():
    return list(_unit_tests)

def all_example_tests():
    return list(_example_tests)

def all_tests():
    return _unit_tests + _example_tests


_load_tests()
