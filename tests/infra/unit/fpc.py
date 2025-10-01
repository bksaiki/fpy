from fpy2 import Function, FPCoreCompiler
from .unit_tests import tests, examples

_ignore = [
    # unrounded constant
    'test_decnum2',
    'test_hexnum1',
    'test_rational1',
    'test_digits4',
    'test_context4',

    # context values
    'test_context_expr1',
    'test_context_expr2',
    'test_context6',
    'test_context7',
    'test_context8',
    'example_static_context1',
    'example_static_context2',
    'keep_p_1',
]

def test_compile_fpc():
    comp = FPCoreCompiler(unsafe_int_cast=True)
    for core in tests + examples:
        assert isinstance(core, Function)
        if core.name not in _ignore:
            print(f'Compiling {core.name}')
            fpc = comp.compile(core)
            print(fpc)

if __name__ == '__main__':
    test_compile_fpc()
