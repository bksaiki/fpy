"""
FPBench infastructure tests.

- Tests conversion of FPCore functions to FPy functions.
- Tests evaluation of FPy functions from FPCore
- Checks that FPy and FPCore functions are equivalent.

Only runs if FPBENCH_PATH is set.
"""

import os, sys

# only runs if FPBENCH_PATH is set.
if not os.getenv("FPBENCH_PATH"):
    print("FPBENCH_PATH not set, skipping fpbench tests")
    sys.exit(0)

from .context_infer import test_context_infer
from .eval import test_eval
from .parse import test_parse
from .round_trip import test_round_trip
from .tcheck import test_tcheck

test_parse()
test_eval()
test_round_trip()
test_tcheck()
test_context_infer()
