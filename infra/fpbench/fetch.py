from pathlib import Path
from titanfp.fpbench.fpcast import FPCore
from titanfp.fpbench.fpcparser import Visitor, compfile, parse, FPCoreParserError

class _Visitor(Visitor):
    """
    titanfp's FPCore visitor, except that it skips `:alt`,
    which FPCore allows to repeat but titanfp does not.
    """

    def _parse_props(self, props):
        return super()._parse_props([prop for prop in props if prop.name.text != ':alt'])

def read_cores(f: Path) -> list[FPCore]:
    """Reads the FPCores in the file `f`, without `:alt`."""
    return _Visitor().visit(parse(f.read_text()))

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
