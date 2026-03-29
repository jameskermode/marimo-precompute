"""Wrapper around mo.persistent_cache that adds parameter grid support and WASM loading."""

from __future__ import annotations

from typing import Any, Callable, Sequence, Union

import marimo as mo

from marimo_precompute.registry import get_registry
from marimo_precompute.wasm_store import PrecomputeStore


def persistent_cache(
    name: Union[str, Callable, None] = None,
    *,
    save_path: str | None = None,
    method: str = "numpy_json",
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
        Serialization method. Defaults to ``"numpy_json"`` (our extension
        that handles numpy arrays). Also supports ``"json"`` and ``"pickle"``.
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

    # Default store: PrecomputeStore (public/ convention for WASM bundling)
    if store is None:
        store = PrecomputeStore(save_path=save_path)

    # Case 1: Bare decorator — @persistent_cache (name is the function)
    if callable(name):
        fn = name
        _register_params(fn, params)
        return mo.persistent_cache(
            fn, save_path=save_path, method=method,
            pin_modules=pin_modules, store=store,
        )

    # Case 2: Context manager — with persistent_cache(name="..."):
    if isinstance(name, str):
        # Pass through to marimo's context manager, but with our store/method
        return mo.persistent_cache(
            name, save_path=save_path, method=method,
            pin_modules=pin_modules, store=store,
        )

    # Case 3: Decorator with arguments — @persistent_cache(params={...})
    def decorator(fn: Callable) -> Any:
        _register_params(fn, params)
        return mo.persistent_cache(
            fn, save_path=save_path, method=method,
            pin_modules=pin_modules, store=store,
        )
    return decorator


def _register_params(
    fn: Callable,
    params: dict[str, Sequence] | None,
) -> None:
    """Register a function's parameter grid in the global registry for CLI sweep."""
    if params is not None:
        registry = get_registry()
        registry.register(fn.__name__, fn, params)
