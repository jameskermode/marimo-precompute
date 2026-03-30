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
    """Fetch bytes from a URL in Pyodide.

    Uses the Pyodide virtual filesystem as a cache: an async ``prefetch()``
    downloads files via ``pyfetch`` and writes them to the VFS.  Subsequent
    sync ``get()`` calls read from the VFS path.
    """
    vfs = _vfs_path_for_url(url)
    if vfs.exists() and vfs.stat().st_size > 0:
        return vfs.read_bytes()
    return None


def _head_wasm(url: str) -> bool:
    """Check if a URL exists — looks at the VFS cache."""
    vfs = _vfs_path_for_url(url)
    return vfs.exists() and vfs.stat().st_size > 0


def _vfs_path_for_url(url: str) -> pathlib.Path:
    """Map a cache URL to a deterministic VFS path."""
    # URL looks like http://host/public/__marimo_precompute__/name/hash.json
    # Store under /tmp/__marimo_precompute_cache__/<name>/<hash.json>
    import hashlib
    key = hashlib.md5(url.encode()).hexdigest()
    return pathlib.Path("/tmp/__marimo_precompute_cache__") / key


async def prefetch_all() -> None:
    """Pre-fetch all precomputed cache files into the Pyodide virtual filesystem.

    Call ``await prefetch_all()`` in an async cell **before** any cells that
    use ``persistent_cache``.  This downloads every JSON file under
    ``public/__marimo_precompute__/`` via ``pyfetch`` and writes them to
    ``/tmp/`` so that the synchronous ``Store.get()`` can read them.

    No-op outside WASM/Pyodide.
    """
    if not _is_wasm():
        return

    from pyodide.http import pyfetch  # type: ignore[import]
    import json

    base = str(_resolve_base_path())

    # Fetch the directory listing.  Python's http.server returns an HTML page;
    # we parse href links that look like cache subdirectories.
    try:
        resp = await pyfetch(base + "/")
        html = await resp.string()
    except Exception:
        return

    import re
    # Links look like <a href="demo1_qs/">demo1_qs/</a>
    dirs = re.findall(r'href="([^"]+/)"', html)

    for d in dirs:
        dir_url = base + "/" + d
        try:
            resp2 = await pyfetch(dir_url)
            html2 = await resp2.string()
        except Exception:
            continue
        files = re.findall(r'href="([^"]+\.json)"', html2)
        for f in files:
            file_url = dir_url + f
            full_key = d + f
            vfs = _vfs_path_for_url(file_url)
            if vfs.exists() and vfs.stat().st_size > 0:
                continue
            try:
                resp3 = await pyfetch(file_url)
                data = await resp3.bytes()
                vfs.parent.mkdir(parents=True, exist_ok=True)
                vfs.write_bytes(data)
            except Exception:
                pass


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
