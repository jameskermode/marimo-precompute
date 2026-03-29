"""Tests for the PrecomputeRegistry."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from marimo_precompute.registry import PrecomputeRegistry, _param_key


def test_param_key_deterministic():
    assert _param_key({"a": 1, "b": 2}) == _param_key({"b": 2, "a": 1})


def test_param_key_float_rounding():
    # Floats that differ only beyond 12 decimals should produce same key
    k1 = _param_key({"x": 0.1 + 0.2})
    k2 = _param_key({"x": 0.3})
    assert k1 == k2


def test_register_and_grid():
    reg = PrecomputeRegistry()
    reg.register("f", lambda x, y: x + y, {"x": [1, 2], "y": [10, 20, 30]})

    entry = reg.entries["f"]
    assert entry.param_names == ["x", "y"]
    assert entry.grid_sizes == [2, 3]
    assert entry.total_combinations == 6


def test_sweep_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = PrecomputeRegistry(cache_dir=tmpdir)
        reg.register("add", lambda x, y: x + y, {"x": [1, 2], "y": [10, 20]})

        results = reg.sweep(verbose=False)
        assert "add" in results
        assert len(results["add"]) == 4

        # Check a specific result
        key = _param_key({"x": 1, "y": 10})
        assert results["add"][key] == 11


def test_sweep_caching():
    """Sweeping twice should skip already-computed values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        call_count = 0

        def counting_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        reg = PrecomputeRegistry(cache_dir=tmpdir)
        reg.register("double", counting_func, {"x": [1, 2, 3]})

        reg.sweep(verbose=False)
        assert call_count == 3

        # Second sweep should not call the function again
        call_count = 0
        reg2 = PrecomputeRegistry(cache_dir=tmpdir)
        reg2.register("double", counting_func, {"x": [1, 2, 3]})
        reg2.sweep(verbose=False)
        assert call_count == 0


def test_sweep_exceeds_limit():
    reg = PrecomputeRegistry()
    reg.register("big", lambda x, y: 0, {"x": range(100), "y": range(200)})

    with pytest.raises(ValueError, match="20,000 combinations exceeds limit"):
        reg.sweep(max_combinations=10_000, verbose=False)


def test_dry_run():
    reg = PrecomputeRegistry()
    reg.register("f", lambda x: x, {"x": [1, 2, 3, 4, 5]})
    reg.register("g", lambda a, b: a, {"a": range(10), "b": range(20)})

    reports = reg.dry_run()
    assert len(reports) == 2

    f_report = next(r for r in reports if r["name"] == "f")
    assert f_report["total_combinations"] == 5

    g_report = next(r for r in reports if r["name"] == "g")
    assert g_report["total_combinations"] == 200


def test_numpy_serialization():
    """Results containing numpy arrays should serialize to JSON correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = PrecomputeRegistry(cache_dir=tmpdir)
        reg.register(
            "np_func",
            lambda x: {"arr": np.array([x, x * 2]), "val": np.float64(x)},
            {"x": [1.0, 2.0]},
        )

        results = reg.sweep(verbose=False)

        # Verify the cache file is valid JSON
        cache_path = Path(tmpdir) / "np_func.json"
        with open(cache_path) as f:
            loaded = json.load(f)
        assert "results" in loaded


def test_lookup():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = PrecomputeRegistry(cache_dir=tmpdir)
        reg.register("f", lambda x: x ** 2, {"x": [2, 3, 4]})
        reg.sweep(verbose=False)

        assert reg.lookup("f", {"x": 3}) == 9
        assert reg.lookup("f", {"x": 99}) is None
        assert reg.lookup("nonexistent", {"x": 1}) is None
