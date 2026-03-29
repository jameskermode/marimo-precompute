"""A Store that reads from HTTP in WASM (Pyodide) and local filesystem in native Python."""

from __future__ import annotations

from typing import Optional

from marimo._save.stores.store import Store


def _is_wasm() -> bool:
    try:
        import pyodide  # noqa: F401
        return True
    except ImportError:
        return False


def _sync_fetch_bytes(url: str) -> Optional[bytes]:
    """Synchronously fetch bytes from a URL in Pyodide using XMLHttpRequest.

    XMLHttpRequest is synchronous in Pyodide web workers, which is what
    marimo uses. This avoids the async/sync mismatch with Store.get().
    """
    from js import XMLHttpRequest  # type: ignore[import]

    xhr = XMLHttpRequest.new()
    xhr.open("GET", url, False)  # synchronous
    xhr.responseType = "arraybuffer"
    xhr.send()
    if xhr.status == 200:
        return bytes(xhr.response.to_py())
    return None


def _sync_head(url: str) -> bool:
    """Synchronously check if a URL exists via HEAD request in Pyodide."""
    from js import XMLHttpRequest  # type: ignore[import]

    xhr = XMLHttpRequest.new()
    xhr.open("HEAD", url, False)  # synchronous
    xhr.send()
    return xhr.status == 200


class WasmStore(Store):
    """Store that reads from HTTP in WASM, delegates to FileStore in native.

    In WASM/Pyodide:
      - get() fetches cache files via synchronous XMLHttpRequest
      - put() is a no-op (read-only — cache is pre-built offline)
      - hit() checks existence via HEAD request

    In native Python:
      - Delegates all operations to marimo's FileStore
    """

    def __init__(
        self,
        save_path: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._wasm = _is_wasm()
        if self._wasm:
            # In WASM: resolve base URL relative to worker location
            if base_url is None:
                from js import self as _js_self  # type: ignore[import]
                base_url = str(_js_self.location.href).rsplit("/", 2)[0]
            self._base_url = base_url.rstrip("/")
        else:
            from marimo._save.stores.file import FileStore
            self._file_store = FileStore(save_path)

    def get(self, key: str) -> Optional[bytes]:
        if self._wasm:
            return _sync_fetch_bytes(f"{self._base_url}/{key}")
        return self._file_store.get(key)

    def put(self, key: str, value: bytes) -> bool:
        if self._wasm:
            return False  # read-only in WASM
        return self._file_store.put(key, value)

    def hit(self, key: str) -> bool:
        if self._wasm:
            return _sync_head(f"{self._base_url}/{key}")
        return self._file_store.hit(key)

    def clear(self, key: str) -> bool:
        if self._wasm:
            return False
        return self._file_store.clear(key)
