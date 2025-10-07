from fpy2 import Function
from ..examples import all_tests

def test_format():
    for core in all_tests():
        assert isinstance(core, Function)
        print(core.format())
