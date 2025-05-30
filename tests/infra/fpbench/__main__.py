"""
FPBench infastructure tests.

- Tests conversion of FPCore functions to FPy functions.
- Tests evaluation of FPy functions from FPCore
- Checks that FPy and FPCore functions are equivalent.
"""

from .eval import test_eval
from .parse import test_parse

test_parse()
test_eval()
