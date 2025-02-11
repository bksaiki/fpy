from fpy2 import Function
from .defs import tests, examples

def test_eval():
    for core in tests + examples:
        fn = core
        assert isinstance(core, Function)
        args = [1.0 for _ in range(len(core.args))]
        print(core.name, fn(*args))

