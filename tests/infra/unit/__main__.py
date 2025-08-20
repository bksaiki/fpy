from .context_infer import test_context_infer
from .eval import test_eval
from .format import test_format
from .fpc import test_compile_fpc
from .tcheck import test_tcheck


test_format()
test_eval()
test_tcheck()
test_context_infer()
test_compile_fpc()
