"""Tests for PrecomputeStore — native mode (local filesystem)."""

import tempfile

from marimo_precompute.wasm_store import PrecomputeStore


def test_native_put_get():
    """In native mode, PrecomputeStore writes to local filesystem."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = PrecomputeStore(save_path=tmpdir)
        assert store.put("test/data.json", b'{"hello": "world"}')
        assert store.hit("test/data.json")
        assert store.get("test/data.json") == b'{"hello": "world"}'


def test_native_miss():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = PrecomputeStore(save_path=tmpdir)
        assert not store.hit("nonexistent.json")
        assert store.get("nonexistent.json") is None


def test_native_clear():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = PrecomputeStore(save_path=tmpdir)
        store.put("test/data.json", b"data")
        assert store.hit("test/data.json")
        assert store.clear("test/data.json")
        assert not store.hit("test/data.json")


def test_native_multi_file_cache_entry():
    """LazyLoader writes one manifest + multiple per-variable blobs per entry.

    PrecomputeStore must accept arbitrary nested keys with any extension.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        store = PrecomputeStore(save_path=tmpdir)

        # Manifest file (jsonl extension — actually JSON, per marimo convention)
        assert store.put("qs_run/C_abc123.jsonl", b'{"hash":"abc123"}')
        # Per-variable blobs under the hash directory
        assert store.put("qs_run/abc123/forces.npy", b"\x93NUMPY...")
        assert store.put("qs_run/abc123/separations.npy", b"\x93NUMPY...")
        assert store.put("qs_run/abc123/return.pickle", b"pickle bytes")

        assert store.hit("qs_run/C_abc123.jsonl")
        assert store.hit("qs_run/abc123/forces.npy")
        assert store.get("qs_run/abc123/forces.npy") == b"\x93NUMPY..."
