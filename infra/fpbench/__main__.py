from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from titanfp.fpbench.fpcast import FPCore
from fpy2 import Function, FPYCompiler

from .fetch import read_dir, read_file

def _resolve_path(s: str):
    path = Path(s).resolve()
    if not path.exists():
        print(f"error: {path} does not exist")
        exit(1)
    return path

# parse command line arguments
parser = ArgumentParser(description="converts FPCore benchmarks to FPy")
parser.add_argument('paths', help='paths to FPCore files', nargs='+', type=str)
parser.add_argument('-o', '--output', help='output directory', type=str)
args = parser.parse_args()

path_strs: list[str] = args.paths
output: Optional[str] = args.output

# convert paths to Path objects
input_paths = [_resolve_path(s) for s in path_strs]
output_path = _resolve_path(output) if output is not None else None

# read FPCores from files
cores: list[FPCore] = []
for path in input_paths:
    if path.is_dir():
        cores += read_dir(path)
    elif path.is_file():
        cores += read_file(path)
    else:
        print(f"error: {path} is not a file or directory")
        exit(1)

# write FPy functions to files
if output_path is None:
    for core in cores:
        func = Function.from_fpcore(core)
        ast = FPYCompiler().compile(func)
        print(ast.format())
else:
    with open(output_path, 'w') as f:
        for core in cores:
            func = Function.from_fpcore(core)
            ast = FPYCompiler().compile(func)
            print(ast.format(), file=f)
