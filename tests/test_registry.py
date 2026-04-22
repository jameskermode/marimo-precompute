"""Tests for the PrecomputeRegistry."""

import pytest

from marimo_precompute.registry import PrecomputeRegistry


def test_register_and_grid():
    reg = PrecomputeRegistry()
    reg.register("f", lambda x, y: x + y, {"x": [1, 2], "y": [10, 20, 30]})

    entry = reg.entries["f"]
    assert entry.param_names == ["x", "y"]
    assert entry.grid_sizes == [2, 3]
    assert entry.total_combinations == 6


def test_iter_combinations():
    reg = PrecomputeRegistry()
    reg.register("f", lambda x, y: x + y, {"x": [1, 2], "y": [10, 20]})
    combos = list(reg.entries["f"].iter_combinations())
    assert len(combos) == 4
    assert {"x": 1, "y": 10} in combos
    assert {"x": 2, "y": 20} in combos


def test_sweep_invokes_each_combination():
    """Sweep should invoke the function once per grid point and return counts."""
    calls = []
    reg = PrecomputeRegistry()
    reg.register(
        "f",
        lambda x, y: calls.append((x, y)) or x + y,
        {"x": [1, 2], "y": [10, 20]},
    )

    counts = reg.sweep(verbose=False)
    assert counts == {"f": 4}
    assert len(calls) == 4


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
