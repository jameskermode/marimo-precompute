"""Register LazyPrecomputeLoader in marimo's PERSISTENT_LOADERS.

LazyPrecomputeLoader is a thin subclass of marimo's LazyLoader that skips
the bytecode-hash integrity check, so caches produced in native CPython
can be loaded in a different Python environment (e.g. Pyodide inside
a WASM notebook).

All marimo imports are inside the install() function, so this module is
safe to import even when marimo is not installed (e.g. in Pyodide before
marimo is loaded).
"""

from __future__ import annotations

import weakref
from typing import Any

_installed = False
_ACTIVE_LOADERS: "weakref.WeakSet[Any]" = weakref.WeakSet()


def install() -> None:
    """Register LazyPrecomputeLoader. Safe to call multiple times."""
    global _installed
    if _installed:
        return
    _installed = True

    import sys

    try:
        import time

        from marimo._save.cache import Cache
        from marimo._save.loaders import PERSISTENT_LOADERS
        from marimo._save.loaders.lazy import LazyLoader
        from marimo._save.loaders.loader import LoaderError

        class LazyPrecomputeLoader(LazyLoader):
            """LazyLoader that tolerates cross-environment hash mismatches.

            Precompute runs in native CPython; WASM reads in Pyodide. Python
            bytecode hashes differ between environments, so marimo's
            integrity check in the base ``cache_attempt`` would reject the
            loaded cache. We override it to skip the hash check while
            keeping the variable-set check.
            """

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                _ACTIVE_LOADERS.add(self)

            def cache_attempt(self, defs, key, stateful_refs):  # type: ignore[no-untyped-def]
                start_time = time.time()
                loaded = self.load_cache(key)
                if not loaded:
                    return Cache.empty(
                        defs=defs, key=key, stateful_refs=stateful_refs
                    )
                load_time = time.time() - start_time

                if loaded.hash != key.hash:
                    print(
                        f"[marimo-precompute] patching hash:"
                        f" {loaded.hash[:20]}... -> {key.hash[:20]}...",
                        file=sys.stderr,
                    )

                if (defs | stateful_refs) != set(loaded.defs):
                    raise LoaderError("Variable mismatch in loaded cache.")
                self._hits += 1
                runtime = loaded.meta.get("runtime", 0)
                if runtime > 0:
                    self._time_saved += max(0, runtime - load_time)
                return Cache.new(
                    loaded=loaded, key=key, stateful_refs=stateful_refs
                )

        PERSISTENT_LOADERS["lazy_precompute"] = LazyPrecomputeLoader

        try:
            from importlib.metadata import version as _pkg_version
            _ver = _pkg_version("marimo-precompute")
        except Exception:
            _ver = "?"
        print(
            f"[marimo-precompute {_ver}] registered method='lazy_precompute' "
            f"(loaders: {sorted(PERSISTENT_LOADERS)})",
            file=sys.stderr,
        )

    except ImportError as e:
        print(
            f"[marimo-precompute] install failed: {e}",
            file=sys.stderr,
        )


def flush_pending_caches() -> None:
    """Wait for all LazyPrecomputeLoader background write threads to finish.

    LazyLoader writes cache blobs from daemon threads; the CLI sweep must
    call this before exiting or files may be truncated.
    """
    for loader in list(_ACTIVE_LOADERS):
        try:
            loader.flush()
        except Exception:
            pass
