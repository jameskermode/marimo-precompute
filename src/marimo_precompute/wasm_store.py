"""A Store that uses marimo's public/ convention for WASM-compatible data loading.

Follows the pattern from https://docs.marimo.io/guides/wasm/#including-data:
- Precomputed cache files are written to ``public/`` alongside the notebook
- Read via ``mo.notebook_location() / "public" / ...`` which resolves to
  a local path in native Python and an HTTP URL in WASM/Pyodide
- ``marimo export html-wasm`` bundles the ``public/`` folder automatically
"""

from __future__ import annotations

import pathlib
from typing import Optional

from marimo._save.stores.store import Store


# Subdirectory inside public/ for our cache files
CACHE_SUBDIR = "__marimo_precompute__"


def _is_wasm() -> bool:
    try:
        import pyodide  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_base_path() -> pathlib.PurePath:
    """Get the base path for cache files using mo.notebook_location().

    Returns notebook_location / "public" / CACHE_SUBDIR, which resolves to:
    - Native: local filesystem path (e.g. /home/user/project/public/__marimo_precompute__/)
    - WASM: URL (e.g. https://site.com/public/__marimo_precompute__/)
    """
    import marimo as mo
    loc = mo.notebook_location()
    if loc is None:
        return pathlib.PurePosixPath("public") / CACHE_SUBDIR
    return loc / "public" / CACHE_SUBDIR


def _fetch_bytes_wasm(url: str) -> Optional[bytes]:
    """Fetch bytes from a URL in Pyodide using the marimo WASM transport.

    marimo's Pyodide environment patches ``open_url`` so that it works
    from within the web-worker.  We use that, falling back to
    ``XMLHttpRequest`` if it is unavailable.
    """
    try:
        from pyodide.http import open_url  # type: ignore[import]
        # open_url returns a StringIO — for JSON payloads this is fine
        text = open_url(url).read()
        return text.encode("utf-8")
    except Exception:
        pass
    # Fallback: try sync XHR (may fail in some worker contexts)
    try:
        from js import XMLHttpRequest  # type: ignore[import]
        xhr = XMLHttpRequest.new()
        xhr.open("GET", url, False)
        xhr.overrideMimeType("text/plain; charset=x-user-defined")
        xhr.send()
        if xhr.status == 200:
            return bytes(ord(c) & 0xFF for c in xhr.response)
    except Exception:
        pass
    return None


def _head_wasm(url: str) -> bool:
    """Check if a URL exists in Pyodide."""
    try:
        from pyodide.http import open_url  # type: ignore[import]
        open_url(url)
        return True
    except Exception:
        pass
    try:
        from js import XMLHttpRequest  # type: ignore[import]
        xhr = XMLHttpRequest.new()
        xhr.open("HEAD", url, False)
        xhr.send()
        return xhr.status == 200
    except Exception:
        return False


class PrecomputeStore(Store):
    """Store that writes to ``public/__marimo_precompute__/`` for WASM bundling.

    In **native Python** (precomputation and local development):
      - ``put()`` writes cache files to ``public/__marimo_precompute__/``
        relative to the notebook, so ``marimo export html-wasm`` bundles them.
      - ``get()``/``hit()`` read from the same local path.

    In **WASM/Pyodide** (deployed notebook):
      - ``get()``/``hit()`` use ``mo.notebook_location()`` to resolve the
        URL of the bundled cache files, then fetch via XMLHttpRequest.
      - ``put()`` is a no-op (cache is pre-built offline).
    """

    def __init__(self, save_path: Optional[str] = None) -> None:
        self._wasm = _is_wasm()
        if not self._wasm:
            # In native mode: write to public/ alongside the notebook
            if save_path is not None:
                self._local_root = pathlib.Path(save_path)
            else:
                import marimo as mo
                loc = mo.notebook_dir()
                if loc is None:
                    self._local_root = pathlib.Path("public") / CACHE_SUBDIR
                else:
                    self._local_root = pathlib.Path(loc) / "public" / CACHE_SUBDIR

    def _local_path(self, key: str) -> pathlib.Path:
        return self._local_root / key

    def _wasm_url(self, key: str) -> str:
        return str(_resolve_base_path() / key)

    def get(self, key: str) -> Optional[bytes]:
        if self._wasm:
            return _fetch_bytes_wasm(self._wasm_url(key))
        path = self._local_path(key)
        if not path.exists() or path.stat().st_size == 0:
            return None
        return path.read_bytes()

    def put(self, key: str, value: bytes) -> bool:
        if self._wasm:
            return False  # read-only in WASM
        path = self._local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return True

    def hit(self, key: str) -> bool:
        if self._wasm:
            return _head_wasm(self._wasm_url(key))
        path = self._local_path(key)
        return path.exists() and path.stat().st_size > 0

    def clear(self, key: str) -> bool:
        if self._wasm:
            return False
        path = self._local_path(key)
        if not path.exists():
            return False
        path.unlink()
        return True
