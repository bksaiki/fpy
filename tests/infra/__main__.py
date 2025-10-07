"""
Integration tests
"""

from .analysis import *
from .backend import *
from .eval import test_eval
from .format import test_format
from .transform import *
from .strategies import *

# from .fpc import test_compile_fpc

def _run_tests():
    # formatting
    test_format()
    # evaluation
    test_eval()
    # analyses
    test_defs()
    test_reaching_defs()
    test_define_use()
    test_tcheck()
    test_context_infer()
    test_purity()
    # transformations
    test_const_prop()
    test_copy_prop()
    test_const_fold()
    test_dead_code()
    test_while_unroll()
    test_split_loop()
    # strategies
    test_simplify()
    # compilation
    test_compile_fpc()
    test_compile_cpp()


_run_tests()

