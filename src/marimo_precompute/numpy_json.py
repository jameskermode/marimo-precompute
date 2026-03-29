"""Numpy-aware JSON encoding and decoding.

This module has NO marimo dependency and works standalone in Pyodide.
The marimo Loader integration (NumpyJsonLoader) lives in patch.py.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types.

    Arrays are tagged as ``{"__numpy__": true, "data": [...], "dtype": "float64"}``
    so they can be reconstructed on decode.
    """

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


def numpy_object_hook(obj: dict) -> Any:
    """Reconstruct numpy arrays from JSON produced by NumpyEncoder."""
    if obj.get("__numpy__"):
        return np.array(obj["data"], dtype=obj["dtype"])
    return obj


# Backward compat aliases (used by tests and other modules)
_NumpyEncoder = NumpyEncoder
_numpy_object_hook = numpy_object_hook
