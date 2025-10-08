"""
Modified import loader that caches original source code of modules as they are loaded.

FPy reparses function source code to extract the original AST.
"""

import os
import sys
import types

from importlib.machinery import SourceFileLoader, PathFinder
from importlib.util import source_from_cache

_SOURCE: dict[str, list[str]] = {}  # path -> lines

class CachingSourceFileLoader(SourceFileLoader):
    """Source file loader that caches the original source code as read."""

    def get_data(self, path):
        if 'site-packages' in path:
            # don't cache files from site-packages
            return super().get_data(path)
        elif path.endswith('.py'):
            # loading a source file
            data = super().get_data(path)
            _SOURCE[path] = data.decode('utf-8').splitlines(keepends=True)
            return data
        elif path.endswith('.pyc'):
            # loading a cached file
            # find the original source if we can find it
            src_path = source_from_cache(path)
            try:
                data = super().get_data(src_path)
                _SOURCE[src_path] = data.decode('utf-8').splitlines(keepends=True)
            except FileNotFoundError:
                pass

            return super().get_data(path)
        else:
            # unknown file type, just load normally
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
    lines = _SOURCE.get(path)
    if lines is None:
        # not in cache yet
        if not os.path.exists(path):
            # can't even find source file
            return None

        # try to read the source file directly
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.read().splitlines(keepends=True)
            _SOURCE[path] = lines

    return lines
