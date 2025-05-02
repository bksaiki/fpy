from fpy2 import Function, FPYCompiler
from .defs import tests, examples

def test_compile_fpy():
    comp = FPYCompiler()
    for core in tests + examples:
        assert isinstance(core, Function)
        ast = comp.compile(core.to_ir())
        print(ast.format())

if __name__ == '__main__':
    test_compile_fpy()
