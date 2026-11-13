"""
FPCore infastructure tests.

- Tests conversion of FPCore functions to FPy functions.
- Tests evaluation of FPy functions from FPCore
- Checks that FPy and FPCore functions are equivalent.
- Checks the FPBench ports against FPBench (see `diff_test.py`).

Only runs if FPBENCH is set.
"""

import os, sys

# only runs if FPBENCH is set.
if not os.getenv("FPBENCH"):
    print("FPBENCH not set, skipping fpbench tests")
    sys.exit(0)

from .eval import test_eval
from .parse import test_parse
from .diff_test import test_fpbench_examples
from .round_trip import test_round_trip
from .tcheck import test_tcheck

# guarded: `test_fpbench_examples` spawns workers, which re-run this module
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='FPCore infrastructure tests')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='worker processes for the FPBench port checks (default 1)')
    args = parser.parse_args()
    test_parse()
    test_eval()
    test_round_trip()
    test_tcheck()
    test_fpbench_examples(jobs=args.jobs)
