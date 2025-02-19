from pathlib import Path
from titanfp.fpbench.fpcast import FPCore
from titanfp.fpbench.fpcparser import compfile, FPCoreParserError

def read_file(f: Path) -> list[FPCore]:
    try:
        return compfile(f)
    except FPCoreParserError:
        return []

def read_dir(dir: Path):
    cores: list[FPCore] = []
    for path in dir.iterdir():
        if path.is_file() and path.name.endswith('.fpcore'):
            cores += read_file(path)
    return cores
