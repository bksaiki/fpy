from fpy2 import Function, FPCoreCompiler
from .defs import tests, examples

def test_compile_fpc():
    comp = FPCoreCompiler()
    for core in tests + examples:
        assert isinstance(core, Function)
        fpc = comp.compile(core)
        print(fpc)
