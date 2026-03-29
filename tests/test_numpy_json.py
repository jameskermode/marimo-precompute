"""Tests for NumpyJsonLoader — numpy array serialization roundtrip."""

import json

import numpy as np
import pytest

from marimo_precompute.numpy_json import NumpyJsonLoader, _NumpyEncoder, _numpy_object_hook


def test_numpy_encoder_array():
    arr = np.array([1.0, 2.0, 3.0])
    encoded = json.dumps(arr, cls=_NumpyEncoder)
    decoded = json.loads(encoded, object_hook=_numpy_object_hook)
    np.testing.assert_array_equal(decoded, arr)


def test_numpy_encoder_2d_array():
    arr = np.array([[1, 2], [3, 4]])
    encoded = json.dumps(arr, cls=_NumpyEncoder)
    decoded = json.loads(encoded, object_hook=_numpy_object_hook)
    np.testing.assert_array_equal(decoded, arr)


def test_numpy_encoder_integer():
    val = np.int64(42)
    encoded = json.dumps(val, cls=_NumpyEncoder)
    assert json.loads(encoded) == 42


def test_numpy_encoder_float():
    val = np.float64(3.14)
    encoded = json.dumps(val, cls=_NumpyEncoder)
    assert json.loads(encoded) == pytest.approx(3.14)


def test_numpy_encoder_bool():
    val = np.bool_(True)
    encoded = json.dumps(val, cls=_NumpyEncoder)
    assert json.loads(encoded) is True


def test_numpy_encoder_nested():
    """Numpy arrays inside dicts/lists should roundtrip correctly."""
    data = {
        "positions": np.array([[0.0, 1.0], [2.0, 3.0]]),
        "energy": np.float64(-1.5),
        "labels": ["a", "b"],
    }
    encoded = json.dumps(data, cls=_NumpyEncoder)
    decoded = json.loads(encoded, object_hook=_numpy_object_hook)
    np.testing.assert_array_equal(decoded["positions"], data["positions"])
    assert decoded["energy"] == -1.5
    assert decoded["labels"] == ["a", "b"]


def test_numpy_encoder_preserves_dtype():
    arr = np.array([1, 2, 3], dtype=np.int32)
    encoded = json.dumps(arr, cls=_NumpyEncoder)
    decoded = json.loads(encoded, object_hook=_numpy_object_hook)
    assert decoded.dtype == np.int32
