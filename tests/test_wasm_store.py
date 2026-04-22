"""Tests for PrecomputeStore — native mode (local filesystem)."""

import tempfile

import marimo_precompute.wasm_store as wasm_store
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


def test_vfs_manifest_hash_insensitive_fallback(monkeypatch, tmp_path):
    """Cross-env hash mismatch: the key uses the local bytecode hash, but
    the bundled file was produced with a different hash. For .jsonl
    manifests we fall back to any C_*.jsonl sibling."""
    monkeypatch.setattr(wasm_store, "_VFS_ROOT", tmp_path)
    bundled = tmp_path / "demo1_qs" / "C_CPYTHONHASH.jsonl"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b'{"hash":"CPYTHONHASH"}')

    key = "demo1_qs/C_PYODIDE_DIFFERENT_HASH.jsonl"
    assert wasm_store._fetch_bytes_wasm(key) == b'{"hash":"CPYTHONHASH"}'
    assert wasm_store._head_wasm(key)


def test_vfs_blob_requires_exact_match(monkeypatch, tmp_path):
    """Blob references come from inside the manifest (already using the
    bundled hash), so they must resolve exactly — no fuzzy fallback for
    .pickle/.npy."""
    monkeypatch.setattr(wasm_store, "_VFS_ROOT", tmp_path)
    bundled = tmp_path / "demo1_qs" / "CPYTHONHASH" / "qs_data.pickle"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"data")

    assert wasm_store._fetch_bytes_wasm("demo1_qs/CPYTHONHASH/qs_data.pickle") == b"data"
    assert wasm_store._fetch_bytes_wasm("demo1_qs/OTHER_HASH/qs_data.pickle") is None


def test_vfs_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(wasm_store, "_VFS_ROOT", tmp_path)
    assert wasm_store._fetch_bytes_wasm("nothing_here/C_abc.jsonl") is None
    assert not wasm_store._head_wasm("nothing_here/C_abc.jsonl")
