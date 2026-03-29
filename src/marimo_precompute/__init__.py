"""marimo-precompute: Extend mo.persistent_cache with WASM loading and parameter sweeps."""

from marimo_precompute.patch import install as _install

# Register our NumpyJsonLoader in marimo's PERSISTENT_LOADERS on import
_install()

from marimo_precompute.decorator import persistent_cache
from marimo_precompute.registry import PrecomputeRegistry, get_registry
from marimo_precompute.wasm_store import PrecomputeStore
from marimo_precompute.numpy_json import NumpyJsonLoader

# Backward compat alias
cached = persistent_cache

__all__ = [
    "persistent_cache",
    "cached",
    "PrecomputeRegistry",
    "get_registry",
    "PrecomputeStore",
    "NumpyJsonLoader",
]
