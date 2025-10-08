"""
Modified import loader that caches original source code of modules as they are loaded.

FPy reparses function source code to extract the original AST.
"""

import sys
import types

from importlib.machinery import SourceFileLoader, PathFinder
from importlib.util import source_from_cache

_ORIG_BY_PATH: dict[str, bytes] = {}  # path -> bytes

class CachingSourceFileLoader(SourceFileLoader):
    """Source file loader that caches the original source code as read."""

    def get_data(self, path):
        if 'site-packages' in path:
            # don't cache files from site-packages
            return super().get_data(path)
        elif path.endswith('.py'):
            # loading a source file
            data = super().get_data(path)
            _ORIG_BY_PATH[path] = data
            return data
        elif path.endswith('.pyc'):
            # loading a cached file
            # find the original source if we can find it
            src_path = source_from_cache(path)
            try:
                data = super().get_data(src_path)
                _ORIG_BY_PATH[src_path] = data
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

def install_caching_loader():
    """Install the caching loader into sys.meta_path."""
    sys.meta_path.insert(0, CachingPathFinder())

def get_original_source(module: types.ModuleType) -> str | None:
    """
    Get the original source code of a module, if available.

    This returns the exact source code as read from the file, before any
    transformations by import hooks. If the source is not available, returns `None`.
    """
    # get the source file path
    path = getattr(module, '__file__', None)
    if path is None:
        return None

    # look up the cached source by the source path
    data = _ORIG_BY_PATH.get(path, None)
    if data is None:
        return None

    return data.decode('utf-8')


