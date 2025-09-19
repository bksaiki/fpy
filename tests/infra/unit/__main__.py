from .context_infer import test_context_infer
from .eval import test_eval
from .format import test_format
from .fpc import test_compile_fpc
from .tcheck import test_tcheck
from .purity import test_purity
from .dead_code import test_dead_code

def _run_tests():
    # formatting
    test_format()
    # evaluation
    test_eval()
    # analyses
    test_tcheck()
    test_context_infer()
    test_purity()
    # transformations
    test_dead_code()
    # compilation
    test_compile_fpc()


_run_tests()
