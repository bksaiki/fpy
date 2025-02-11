import os

from dataclasses import dataclass
from pathlib import Path
from titanfp.fpbench.fpcast import FPCore
from titanfp.fpbench.fpcparser import compfile, FPCoreParserError

fpbench_env = os.environ.get("FPBENCH", None)
if fpbench_env is None:
    raise ValueError("Set $FPBENCH to the root of the FPBench repository")

fpbench_dir = Path(fpbench_env)
benchmarks_dir = Path(fpbench_dir, 'benchmarks')
tests_dir = Path(fpbench_dir, 'tests')
sanity_dir = Path(tests_dir, 'sanity')
tensor_dir = Path(tests_dir, 'tensor')

def _read_file(f: Path) -> list[FPCore]:
    try:
        return compfile(f)
    except FPCoreParserError:
        return []

def _read_dir(dir: Path):
    cores: list[FPCore] = []
    for path in dir.iterdir():
        if path.is_file() and path.name.endswith('.fpcore'):
            cores += _read_file(path)
    return cores

@dataclass
class FPBenchCores:
    sanity_cores: list[FPCore]
    tests_cores: list[FPCore]
    benchmark_cores: list[FPCore]
    tensor_cores: list[FPCore]

def fetch_cores():
    sanity_cores = _read_dir(sanity_dir)
    tests_cores = _read_dir(tests_dir)
    benchmark_cores = _read_dir(benchmarks_dir)
    tensor_cores = _read_dir(tensor_dir)
    return FPBenchCores(sanity_cores, tests_cores, benchmark_cores, tensor_cores)
