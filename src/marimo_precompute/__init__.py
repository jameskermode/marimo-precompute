"""marimo-precompute: Extend mo.persistent_cache with WASM loading and parameter sweeps."""

from marimo_precompute.patch import install as _install

# Register LazyPrecomputeLoader in marimo's PERSISTENT_LOADERS on import
# (no-op if marimo is not installed, e.g. in Pyodide before marimo is loaded)
_install()

from marimo_precompute.registry import PrecomputeRegistry, get_registry  # noqa: E402

# Lazy imports for modules that depend on marimo at runtime
try:
    from marimo_precompute.decorator import persistent_cache
    from marimo_precompute.wasm_store import PrecomputeStore, prefetch_all
    cached = persistent_cache
except ImportError:
    persistent_cache = None  # type: ignore[assignment]
    PrecomputeStore = None  # type: ignore[assignment]
    cached = None  # type: ignore[assignment]

    async def prefetch_all() -> None:  # type: ignore[misc]
        pass

__all__ = [
    "persistent_cache",
    "cached",
    "PrecomputeRegistry",
    "get_registry",
    "PrecomputeStore",
    "prefetch_all",
]
