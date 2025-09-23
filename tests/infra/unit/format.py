from fpy2 import Function
from .unit_tests import tests, examples

def test_format():
    for core in tests + examples:
        assert isinstance(core, Function)
        print(core.format())
