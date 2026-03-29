"""JSON loader with numpy array support, extending marimo's JsonLoader."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Optional

import numpy as np

from marimo._save.cache import Cache
from marimo._save.hash import HashKey
from marimo._save.loaders.json import JsonLoader


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {"__numpy__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _numpy_object_hook(obj: dict) -> Any:
    """Reconstruct numpy arrays from JSON."""
    if obj.get("__numpy__"):
        return np.array(obj["data"], dtype=obj["dtype"])
    return obj


class NumpyJsonLoader(JsonLoader):
    """JsonLoader that handles numpy arrays via tagged encoding.

    Numpy arrays are serialized as {"__numpy__": true, "data": [...], "dtype": "float64"}
    and reconstructed on load. All other types use standard JSON encoding.
    """

    def restore_cache(self, key: HashKey, blob: bytes) -> Cache:
        del key
        cache = json.loads(blob, object_hook=_numpy_object_hook)
        cache["stateful_refs"] = set(cache["stateful_refs"])
        try:
            hash_key = cache.pop("key", {})
            return Cache(**hash_key, **cache)
        except TypeError as e:
            from marimo._save.loaders.loader import LoaderError
            raise LoaderError(
                "Invalid json object for cache restoration"
            ) from e

    def to_blob(self, cache: Cache) -> Optional[bytes]:
        dump = dataclasses.asdict(cache)
        dump["stateful_refs"] = list(dump["stateful_refs"])
        return json.dumps(dump, indent=4, cls=_NumpyEncoder).encode("utf-8")
