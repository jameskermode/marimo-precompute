"""Tests for persistent_cache wrapper — parameter grid registration."""

from marimo_precompute.registry import PrecomputeRegistry
from marimo_precompute.decorator import _register_params


def test_register_params():
    """_register_params should add function to the global registry."""
    registry = PrecomputeRegistry()

    def my_func(x, y):
        return x + y

    from marimo_precompute import registry as reg_module
    old = reg_module._global_registry
    reg_module._global_registry = registry

    try:
        _register_params(my_func, {"x": [1, 2], "y": [10, 20]})
        assert "my_func" in registry.entries
        assert registry.entries["my_func"].total_combinations == 4
    finally:
        reg_module._global_registry = old


def test_register_params_none():
    """_register_params with params=None should not register."""
    registry = PrecomputeRegistry()

    def my_func(x):
        return x

    from marimo_precompute import registry as reg_module
    old = reg_module._global_registry
    reg_module._global_registry = registry

    try:
        _register_params(my_func, None)
        assert "my_func" not in registry.entries
    finally:
        reg_module._global_registry = old


def test_patch_installed():
    """Importing marimo_precompute should register numpy_json in PERSISTENT_LOADERS."""
    from marimo._save.loaders import PERSISTENT_LOADERS
    import marimo_precompute  # noqa: F401

    assert "numpy_json" in PERSISTENT_LOADERS
