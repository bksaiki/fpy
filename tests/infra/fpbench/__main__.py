"""
FPBench infastructure tests.

- Tests conversion of FPCore functions to FPy functions.
- Tests evaluation of FPy functions from FPCore
- Checks that FPy and FPCore functions are equivalent.
"""

from .eval import test_eval
from .parse import test_parse
from .round_trip import test_round_trip
from .tcheck import test_tcheck

test_parse()
test_eval()
test_round_trip()
test_tcheck()
