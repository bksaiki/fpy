from argparse import ArgumentParser
from pathlib import Path

from .fetch import read_cores
from .write import module_name, write_ports

# parse command line arguments
parser = ArgumentParser(description='converts FPCore files to FPy, one module per file')
parser.add_argument('paths', help='paths to FPCore files', nargs='+', type=Path)
parser.add_argument('-o', '--output', help='output directory', type=Path, required=True)
args = parser.parse_args()

paths: list[Path] = args.paths
output: Path = args.output

# write one FPy module per FPCore file
output.mkdir(parents=True, exist_ok=True)
for path in paths:
    with open(output / f'{module_name(path)}.py', 'w') as f:
        write_ports(path, read_cores(path), f)
