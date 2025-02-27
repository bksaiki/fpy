from fpy2 import Function, set_default_interpreter, RealInterpreter
from .defs import tests, examples

_banned_tests = [
    'test_tuple1',
    'test_tuple2',
    'test_tuple3',
    'test_tuple4',
    'test_list1',
    'test_list2',
    'test_list_comp1',
    'test_list_comp2',
    'test_list_ref1',
    'test_list_ref2',
    'test_list_ref3',
    'test_list_set1',
    'test_list_set2',
    'test_list_set3',
    'test_while3',
    'test_while4',
    'test_while5',
    'test_for1',
    'test_for2',
    'test_for3',
]

def test_eval():
    for core in tests + examples:
        fn = core
        assert isinstance(core, Function)
        if core.name not in _banned_tests:
            args = [1.0 for _ in range(len(core.args))]
            print(core.name, fn(*args))

if __name__ == '__main__':
    set_default_interpreter(RealInterpreter(True))
    test_eval()
