from fpy2 import Function
from titanfp.fpbench.fpcast import FPCore

from .fetch import fetch_cores

def _parse(cores: list[FPCore]):
    for core in cores:
        func = Function.from_fpcore(core)
        print(func.format())

def test_parse():
    fpbench = fetch_cores()
    _parse(fpbench.sanity_cores)
    _parse(fpbench.tests_cores)
    _parse(fpbench.benchmark_cores)
    _parse(fpbench.tensor_cores)
