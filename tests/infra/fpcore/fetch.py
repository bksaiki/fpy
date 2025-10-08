import os

from dataclasses import dataclass
from pathlib import Path
from titanfp.fpbench.fpcast import FPCore
from infra.fpbench import read_dir, read_file

fpbench_env = os.environ.get("FPBENCH", None)
if fpbench_env is None:
    raise ValueError("Set $FPBENCH to the root of the FPBench repository")

fpbench_dir = Path(fpbench_env)
benchmarks_dir = Path(fpbench_dir, 'benchmarks')
tests_dir = Path(fpbench_dir, 'tests')
sanity_dir = Path(tests_dir, 'sanity')
tensor_dir = Path(tests_dir, 'tensor')


@dataclass
class FPBenchCores:
    sanity_cores: list[FPCore]
    tests_cores: list[FPCore]
    benchmark_cores: list[FPCore]
    tensor_cores: list[FPCore]

    def all_cores(self) -> list[FPCore]:
        return (self.sanity_cores + self.tests_cores +
                self.benchmark_cores + self.tensor_cores)

def fetch_cores():
    sanity_cores = read_dir(sanity_dir)
    tests_cores = read_dir(tests_dir)
    benchmark_cores = read_dir(benchmarks_dir)
    tensor_cores = read_dir(tensor_dir)
    return FPBenchCores(sanity_cores, tests_cores, benchmark_cores, tensor_cores)
