import sys
import importlib.util

from pathlib import Path
from fpy2 import Function

THIS_PATH = Path(__file__).resolve()
TOP_PATH = THIS_PATH.parent.parent.parent

def _import_module(path: Path):
    rel_path = path.relative_to(TOP_PATH).with_suffix('')
    module_name = str(rel_path).replace('/', '.')
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import module: {module_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _load_funs_from_file(path: Path):
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    m = _import_module(path)
    return [v for k, v in m.__dict__.items() if isinstance(v, Function)]

def load_funs(paths: list[Path]):
    funs: list[Function] = []
    for path in paths:
        funs += _load_funs_from_file(path)
    return funs
