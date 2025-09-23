from .context_infer import test_context_infer
from .const_prop import test_const_prop
from .copy_prop import test_copy_prop
from .dead_code import test_dead_code
from .define_use import test_define_use
from .defs import test_defs
from .eval import test_eval
from .format import test_format
from .fpc import test_compile_fpc
from .reaching_defs import test_reaching_defs
from .purity import test_purity
from .tcheck import test_tcheck

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
    test_dead_code()
    # compilation
    test_compile_fpc()


_run_tests()
