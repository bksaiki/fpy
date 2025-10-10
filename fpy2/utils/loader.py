"""
Modified import loader that caches original source code of modules as they are loaded.

FPy reparses function source code to extract the original AST.
"""

import sys
import types

from importlib.machinery import SourceFileLoader, PathFinder
from importlib.util import source_from_cache

_SOURCE: dict[str, bytes] = {}  # path -> bytes
_LINES: dict[str, list[str]] = {}  # path -> lines

class CachingSourceFileLoader(SourceFileLoader):
    """Source file loader that caches the original source code as read."""

    def get_data(self, path):
        if path not in _SOURCE:
            # not cached yet, load and cache
            if path.endswith('.py'):
                # loading the source file and cache it
                data = super().get_data(path)
                _SOURCE[path] = data
                return data
            elif path.endswith('.pyc'):
                # loading a cached file
                # find the original source if we can find it
                src_path = source_from_cache(path)
                try:
                    data = super().get_data(src_path)
                    _SOURCE[src_path] = data
                except FileNotFoundError:
                    pass

        return super().get_data(path)


class CachingPathFinder:
    """Source file finder that uses `CachingSourceFileLoader`."""

    def find_spec(self, fullname, path, target=None):
        # Ask the normal machinery for a spec (skip ourselves)
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or not isinstance(spec.loader, SourceFileLoader):
            return spec
        # wrap the loader with our caching subclass
        spec.loader = CachingSourceFileLoader(spec.loader.name, spec.loader.path)
        return spec

_PATH_FINDER = CachingPathFinder()
"""Modified path finder that uses `CachingSourceFileLoader`."""

def install_caching_loader():
    """Install the caching loader into sys.meta_path."""
    sys.meta_path.insert(0, _PATH_FINDER)

def get_module_source(mod: types.ModuleType) -> list[str] | None:
    """
    Gets the original source code of a module, if available.

    This returns the exact source code as read from the file,
    before any transformations by import hooks.
    If the source is not available, returns `None`.
    """
    path = getattr(mod, '__file__', None)
    if path is None:
        return None

    # look up the cached source by the source path
    lines = _LINES.get(path)
    if lines is None:
        # not in cache yet, try to load from _SOURCE
        data = _SOURCE.get(path)
        if data is None:
            return None

        lines = data.decode('utf-8').splitlines(keepends=True)
        _LINES[path] = lines

    return lines
