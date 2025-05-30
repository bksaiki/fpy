import re

from argparse import ArgumentParser
from sys import stdout
from pathlib import Path
from typing import Optional

from titanfp.fpbench.fpcast import FPCore
from fpy2 import Function
from fpy2.utils import pythonize_id

from .fetch import read_dir, read_file

VALID_NAME_CHAR = re.compile('[a-zA-Z0-9_]')

def _resolve_path(s: str):
    return Path(s).resolve()

def _fpcore_name(core: FPCore, default_name: str):
    if core.ident is not None:
        return core.ident
    elif 'name' in core.props:
        name_data = core.props['name']
        return pythonize_id(name_data.value.value)
    else:
        return default_name

def _write_cores(cores: list[FPCore], f):
    for i, core in enumerate(cores):
        name = _fpcore_name(core, f'f{i}')
        func = Function.from_fpcore(core, default_name=name, ignore_unknown=True)
        print(func.format(), file=f)
        print('', file=f)

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
    _write_cores(cores, stdout)
else:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        print('from fpy2 import *', file=f)
        print('from fpy2.typing import *', file=f)
        print('', file=f)
        _write_cores(cores, f)
