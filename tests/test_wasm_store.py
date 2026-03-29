"""Tests for WasmStore — native mode (FileStore delegation)."""

import tempfile
from pathlib import Path

from marimo_precompute.wasm_store import WasmStore


def test_native_put_get():
    """In native mode, WasmStore delegates to FileStore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = WasmStore(save_path=tmpdir)
        assert store.put("test/data.json", b'{"hello": "world"}')
        assert store.hit("test/data.json")
        assert store.get("test/data.json") == b'{"hello": "world"}'


def test_native_miss():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = WasmStore(save_path=tmpdir)
        assert not store.hit("nonexistent.json")
        assert store.get("nonexistent.json") is None


def test_native_clear():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = WasmStore(save_path=tmpdir)
        store.put("test/data.json", b"data")
        assert store.hit("test/data.json")
        assert store.clear("test/data.json")
        assert not store.hit("test/data.json")
