"""Register LazyPrecomputeLoader in marimo's PERSISTENT_LOADERS.

LazyPrecomputeLoader is a thin subclass of marimo's LazyLoader adapted
for cross-environment static hosting. It (1) skips the bytecode-hash
integrity check so caches produced in native CPython load in Pyodide,
and (2) loads blobs sequentially in ``restore_cache`` because Pyodide's
pthread-less runtime refuses ``threading.Thread.start()``.

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

    try:
        import io
        import pickle
        import time
        from pathlib import Path

        import msgspec
        from marimo._save.cache import MARIMO_CACHE_VERSION, Cache
        from marimo._save.loaders import PERSISTENT_LOADERS
        from marimo._save.loaders.lazy import LazyLoader, from_item
        from marimo._save.loaders.loader import LoaderError
        from marimo._save.stubs.lazy_stub import Cache as CacheSchema

        def _deserialize(data: bytes, ext: str, type_hint=None):
            """Extension-based blob deserializer.

            Implemented inline to support both marimo 0.23.1 (pickle only)
            and 0.23.2+ (multi-format). In 0.23.2+ marimo exposes an
            equivalent ``BLOB_DESERIALIZERS`` table; we avoid importing
            it so a single build works across versions.
            """
            del type_hint
            if ext == ".npy":
                import numpy as np
                return np.load(io.BytesIO(data), allow_pickle=False)
            if ext == ".arrow":
                try:
                    import pyarrow.feather as feather
                    return feather.read_table(io.BytesIO(data))
                except Exception:
                    return pickle.loads(data)
            return pickle.loads(data)

        class LazyPrecomputeLoader(LazyLoader):
            """LazyLoader with two Pyodide-compat tweaks.

            1. ``cache_attempt`` skips the bytecode-hash integrity check so
               caches produced in native CPython load in a Pyodide runtime
               (the two environments hash bytecode differently).
            2. ``restore_cache`` loads blobs sequentially. Upstream spawns
               a thread per blob; Pyodide's pthread-less runtime refuses
               ``Thread.start()``.
            """

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                _ACTIVE_LOADERS.add(self)

            def restore_cache(self, _key, blob):  # type: ignore[no-untyped-def]
                """Sequential version of LazyLoader.restore_cache.

                Loads blobs one at a time instead of spawning a thread per
                blob, so the method works under Pyodide's pthread-less
                runtime. Same outputs; slower on wide cache entries.
                """
                cache_data = msgspec.json.decode(blob, type=CacheSchema)

                ref_vars: dict = {}
                ref_type_hints: dict = {}
                variable_hashes: dict = {}
                base = Path(self.name) / cache_data.hash
                for var_name, item in cache_data.defs.items():
                    if var_name in cache_data.ui_defs:
                        ref_vars[var_name] = (base / "ui.pickle").as_posix()
                    elif item.reference is not None:
                        ref_vars[var_name] = item.reference
                        ref_type_hints[item.reference] = item.type_hint
                    if item.hash:
                        variable_hashes[var_name] = item.hash

                return_ref = None
                return_type_hint = None
                if (
                    cache_data.meta.return_value
                    and cache_data.meta.return_value.reference
                ):
                    return_ref = cache_data.meta.return_value.reference
                    return_type_hint = cache_data.meta.return_value.type_hint

                unique_keys = set(ref_vars.values())
                if return_ref:
                    unique_keys.add(return_ref)

                unpickled: dict = {}
                for ref_key in unique_keys:
                    data = self.store.get(ref_key)
                    if not data:
                        raise FileNotFoundError(
                            f"Incomplete cache: missing blob {ref_key}"
                        )
                    ext = Path(ref_key).suffix
                    type_hint = ref_type_hints.get(ref_key) or (
                        return_type_hint if ref_key == return_ref else None
                    )
                    unpickled[ref_key] = _deserialize(data, ext, type_hint)

                defs: dict = {}
                for var_name, item in cache_data.defs.items():
                    if var_name in ref_vars:
                        ref_key = ref_vars[var_name]
                        val = unpickled.get(ref_key)
                        if var_name in cache_data.ui_defs and isinstance(val, dict):
                            defs[var_name] = val[var_name]
                        else:
                            defs[var_name] = val
                    else:
                        defs[var_name] = from_item(item)

                if return_ref and return_ref in unpickled:
                    return_item = unpickled[return_ref]
                elif cache_data.meta.return_value:
                    return_item = from_item(cache_data.meta.return_value)
                else:
                    return_item = None

                return Cache(
                    hash=cache_data.hash,
                    cache_type=cache_data.cache_type.value,
                    stateful_refs=set(cache_data.stateful_refs),
                    defs=defs,
                    meta={
                        "version": cache_data.meta.version or MARIMO_CACHE_VERSION,
                        "return": return_item,
                        "variable_hashes": variable_hashes,
                    },
                    hit=True,
                )

            def load_cache(self, key):  # type: ignore[no-untyped-def]
                # Don't silently swallow restore_cache errors — the base
                # class's LOGGER.warning doesn't surface in Pyodide, which
                # makes WASM cache failures invisible.
                blob = self.store.get(str(self.build_path(key)))
                if not blob:
                    return None
                try:
                    return self.restore_cache(key, blob)
                except Exception as e:
                    print(
                        f"[marimo-precompute] restore_cache failed for "
                        f"{self.name}: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    return None

            def cache_attempt(self, defs, key, stateful_refs):  # type: ignore[no-untyped-def]
                start_time = time.time()
                loaded = self.load_cache(key)
                if not loaded:
                    return Cache.empty(
                        defs=defs, key=key, stateful_refs=stateful_refs
                    )
                load_time = time.time() - start_time

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

    except Exception as e:
        print(
            f"[marimo-precompute] install failed: {type(e).__name__}: {e}",
            flush=True,
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
