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
            """JsonLoader that handles numpy arrays via tagged encoding."""

            def restore_cache(self, key: HashKey, blob: bytes) -> Cache:
                del key
                cache = json.loads(blob, object_hook=numpy_object_hook)
                cache["stateful_refs"] = set(cache["stateful_refs"])
                try:
                    hash_key = cache.pop("key", {})
                    return Cache(**hash_key, **cache)
                except TypeError as e:
                    raise LoaderError(
                        "Invalid json object for cache restoration"
                    ) from e

            def to_blob(self, cache: Cache) -> Optional[bytes]:
                dump = dataclasses.asdict(cache)
                dump["stateful_refs"] = list(dump["stateful_refs"])
                return json.dumps(dump, indent=4, cls=NumpyEncoder).encode("utf-8")

        PERSISTENT_LOADERS["numpy_json"] = NumpyJsonLoader  # type: ignore[assignment]

    except ImportError:
        # marimo not installed — patch is a no-op.
        pass
