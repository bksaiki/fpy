from fpy2 import Function, FPCoreCompiler
from .defs import tests, examples

_ignore = [
    'test_context_expr1',
    'test_context_expr2',
    'test_context6',
    'test_context7',
    'test_context8',
    'keep_p_1',
]

def test_compile_fpc():
    comp = FPCoreCompiler()
    for core in tests + examples:
        assert isinstance(core, Function)
        if core.name not in _ignore:
            print(f'Compiling {core.name}')
            fpc = comp.compile(core)
            print(fpc)

if __name__ == '__main__':
    test_compile_fpc()
