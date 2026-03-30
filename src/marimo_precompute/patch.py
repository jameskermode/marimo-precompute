"""Monkey-patch marimo to register our numpy-aware JSON loader.

All marimo imports are inside the try block, so this module is safe
to import even when marimo is not installed (e.g. in Pyodide).
"""

from __future__ import annotations

_installed = False


def install() -> None:
    """Register NumpyJsonLoader in marimo's PERSISTENT_LOADERS.

    Safe to call multiple times — only patches once.
    No-op if marimo is not installed.
    """
    global _installed
    if _installed:
        return
    _installed = True

    try:
        import dataclasses
        import json
        from typing import Optional

        from marimo._save.cache import Cache
        from marimo._save.hash import HashKey
        from marimo._save.loaders import PERSISTENT_LOADERS
        from marimo._save.loaders.json import JsonLoader
        from marimo._save.loaders.loader import LoaderError

        from marimo_precompute.numpy_json import NumpyEncoder, numpy_object_hook

        class NumpyJsonLoader(JsonLoader):
            """JsonLoader that handles numpy arrays via tagged encoding.

            In WASM, the cache may have been produced in a different Python
            environment (different bytecode hash).  We patch the loaded hash
            to match the local key so marimo's integrity check passes.
            """

            def restore_cache(self, key: HashKey, blob: bytes) -> Cache:
                cache = json.loads(blob, object_hook=numpy_object_hook)
                cache["stateful_refs"] = set(cache["stateful_refs"])
                # Replace stored hash with the local hash so the
                # cross-environment integrity check in cache_attempt passes.
                import sys
                if cache["hash"] != key.hash:
                    print(
                        f"[marimo-precompute] patching hash:"
                        f" {cache['hash'][:20]}... -> {key.hash[:20]}...",
                        file=sys.stderr,
                    )
                cache["hash"] = key.hash
                try:
                    hash_key = cache.pop("key", {})
                    return Cache(**hash_key, **cache)
                except TypeError as e:
                    raise LoaderError(
                        "Invalid json object for cache restoration"
                    ) from e

            def cache_attempt(
                self,
                defs: set,
                key: HashKey,
                stateful_refs: set,
            ) -> Cache:
                """Override to skip hash verification for cross-env precompute."""
                import time
                start_time = time.time()
                loaded = self.load_cache(key)
                if not loaded:
                    return Cache.empty(defs=defs, key=key, stateful_refs=stateful_refs)
                load_time = time.time() - start_time

                # Skip hash check — cross-environment precompute produces
                # different hashes (Python bytecode is version-dependent).
                if (defs | stateful_refs) != set(loaded.defs):
                    raise LoaderError("Variable mismatch in loaded cache.")
                self._hits += 1
                runtime = loaded.meta.get("runtime", 0)
                if runtime > 0:
                    self._time_saved += max(0, runtime - load_time)
                return Cache.new(loaded=loaded, key=key, stateful_refs=stateful_refs)

            def to_blob(self, cache: Cache) -> Optional[bytes]:
                dump = dataclasses.asdict(cache)
                dump["stateful_refs"] = list(dump["stateful_refs"])
                return json.dumps(dump, indent=4, cls=NumpyEncoder).encode("utf-8")

        PERSISTENT_LOADERS["numpy_json"] = NumpyJsonLoader  # type: ignore[assignment]

    except ImportError:
        # marimo not installed — patch is a no-op.
        pass
