"""Wrapper around mo.persistent_cache that adds parameter grid support and WASM loading."""

from __future__ import annotations

from typing import Any, Callable, Sequence, Union

from marimo_precompute.registry import get_registry
from marimo_precompute.wasm_store import PrecomputeStore


def persistent_cache(
    name: Union[str, Callable, None] = None,
    *,
    save_path: str | None = None,
    method: str = "lazy_precompute",
    pin_modules: bool = False,
    params: dict[str, Sequence] | None = None,
    store: Any = None,
) -> Any:
    """Drop-in extension of ``mo.persistent_cache`` with precompute support.

    Supports the same three usage patterns as ``mo.persistent_cache``:

    1. **Bare decorator**::

        @persistent_cache
        def expensive(x, y): ...

    2. **Decorator with arguments**::

        @persistent_cache(params={"x": [1, 2, 3], "y": [10, 20]})
        def expensive(x, y): ...

    3. **Context manager**::

        with persistent_cache(name="my_calc"):
            result = expensive(x, y)

    Parameters
    ----------
    name : str or callable, optional
        For context manager: cache name. For bare decorator: the function.
    save_path : str, optional
        Cache directory. Defaults to ``public/__marimo_precompute__/``
        alongside the notebook (following marimo's WASM data convention).
    method : str
        Serialization method. Defaults to ``"lazy_precompute"`` — our thin
        subclass of marimo's ``LazyLoader`` that tolerates cross-environment
        hash mismatches (needed for WASM). ``"lazy"`` is the upstream
        marimo loader (same formats: ``.npy``/``.arrow``/``.pickle``).
        ``"pickle"`` and ``"json"`` are also supported.
    pin_modules : bool
        If True, invalidate cache when module versions change.
    params : dict[str, Sequence], optional
        **New.** Parameter grid for offline precomputation via the sweep CLI.
        Keys are parameter names, values are sequences of values to sweep.
    store : Store, optional
        Custom store. Defaults to ``PrecomputeStore`` which writes to
        ``public/__marimo_precompute__/`` (bundled by ``marimo export
        html-wasm``) and reads via ``mo.notebook_location()`` in WASM.
    """
    # Ensure our loader is registered
    from marimo_precompute.patch import install
    install()

    import marimo as mo

    # Default store: PrecomputeStore (public/ convention for WASM bundling)
    if store is None:
        store = PrecomputeStore(save_path=save_path)

    # Case 1: Bare decorator — @persistent_cache (name is the function)
    if callable(name):
        fn = name
        wrapped = mo.persistent_cache(
            fn, save_path=save_path, method=method,
            pin_modules=pin_modules, store=store,
        )
        _register_params(fn.__name__, wrapped, params)
        return wrapped

    # Case 2: Context manager — with persistent_cache(name="..."):
    if isinstance(name, str):
        return mo.persistent_cache(
            name, save_path=save_path, method=method,
            pin_modules=pin_modules, store=store,
        )

    # Case 3: Decorator with arguments — @persistent_cache(params={...})
    def decorator(fn: Callable) -> Any:
        wrapped = mo.persistent_cache(
            fn, save_path=save_path, method=method,
            pin_modules=pin_modules, store=store,
        )
        _register_params(fn.__name__, wrapped, params)
        return wrapped
    return decorator


def _register_params(
    name: str,
    func: Callable,
    params: dict[str, Sequence] | None,
) -> None:
    """Register a function's parameter grid in the global registry for CLI sweep.

    ``func`` should be the cache-wrapped callable so the sweep benefits
    from marimo's caching (hits skip computation, misses write blobs).
    """
    if params is not None:
        registry = get_registry()
        registry.register(name, func, params)
