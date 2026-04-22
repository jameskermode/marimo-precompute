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

CACHE_SUBDIR = "__marimo_precompute__"

_VFS_ROOT = pathlib.Path("/tmp/__marimo_precompute_cache__")

# Files written by LazyPrecomputeLoader / LazyLoader. The manifest glob
# excludes any *manifest.json* that write_manifest() itself produces.
_CACHE_EXTENSIONS = ("*.jsonl", "*.pickle", "*.npy", "*.arrow")


def _is_wasm() -> bool:
    try:
        import pyodide  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_base_path() -> pathlib.PurePath:
    """Get the base path for cache files using mo.notebook_location().

    Returns notebook_location / "public" / CACHE_SUBDIR, which resolves to:
    - Native: local filesystem path
    - WASM: URL (e.g. https://site.com/public/__marimo_precompute__/)
    """
    import marimo as mo
    loc = mo.notebook_location()
    if loc is None:
        return pathlib.PurePosixPath("public") / CACHE_SUBDIR
    return loc / "public" / CACHE_SUBDIR


def _rel_from_base(key: str) -> Optional[str]:
    """Extract the path relative to the cache root from a store key.

    Keys look like ``"<hash>/forces.npy"`` (LazyLoader blobs) or
    ``"<name>/C_<hash>.jsonl"`` (manifest files), optionally prefixed with
    the public/__marimo_precompute__ path when coming from a full URL.
    """
    parts = pathlib.PurePosixPath(key).parts
    for i, part in enumerate(parts):
        if part == CACHE_SUBDIR:
            return "/".join(parts[i + 1:]) if i + 1 < len(parts) else None
    return key


def _fetch_bytes_wasm(key: str) -> Optional[bytes]:
    """Read precomputed cache bytes from the Pyodide VFS, keyed by the
    cache-relative path.

    Path structure (preserved verbatim from the remote): e.g.
    ``<name>/C_<hash>.jsonl`` or ``<name>/<hash>/forces.npy``.
    """
    rel = _rel_from_base(key)
    if rel is None:
        return None
    vfs = _VFS_ROOT / rel
    if vfs.exists() and vfs.stat().st_size > 0:
        return vfs.read_bytes()
    return None


def _head_wasm(key: str) -> bool:
    rel = _rel_from_base(key)
    if rel is None:
        return False
    vfs = _VFS_ROOT / rel
    return vfs.exists() and vfs.stat().st_size > 0


async def prefetch_all() -> None:
    """Pre-fetch all precomputed cache files into the Pyodide VFS.

    Call ``await prefetch_all()`` in an async cell **before** any cells that
    use ``persistent_cache``.  Reads a ``manifest.json`` listing all cache
    file paths, then downloads each via ``pyfetch`` and writes them to
    ``/tmp/`` preserving the relative path so the synchronous ``Store.get()``
    can read them.

    No-op outside WASM/Pyodide.
    """
    if not _is_wasm():
        return

    import json

    from pyodide.http import pyfetch  # type: ignore[import]

    base = str(_resolve_base_path())

    try:
        resp = await pyfetch(base + "/manifest.json")
        manifest = json.loads(await resp.string())
    except Exception:
        return

    for rel_path in manifest.get("files", []):
        vfs = _VFS_ROOT / rel_path
        if vfs.exists() and vfs.stat().st_size > 0:
            continue
        file_url = base + "/" + rel_path
        try:
            resp2 = await pyfetch(file_url)
            data = await resp2.bytes()
            vfs.parent.mkdir(parents=True, exist_ok=True)
            vfs.write_bytes(data)
        except Exception:
            pass


def write_manifest(cache_dir: Optional[str] = None) -> None:
    """Write a ``manifest.json`` listing all cache files under ``cache_dir``.

    Called automatically after ``put()`` operations. The manifest lets
    ``prefetch_all()`` discover files on static hosts (like GitHub Pages)
    where directory listings aren't available.
    """
    import json

    if cache_dir is None:
        import marimo as mo
        loc = mo.notebook_dir()
        if loc is None:
            root = pathlib.Path("public") / CACHE_SUBDIR
        else:
            root = pathlib.Path(loc) / "public" / CACHE_SUBDIR
    else:
        root = pathlib.Path(cache_dir)

    if not root.exists():
        return

    files: list[str] = []
    for pattern in _CACHE_EXTENSIONS:
        for p in sorted(root.rglob(pattern)):
            if p.name == "manifest.json":
                continue
            files.append(str(p.relative_to(root)))

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({"files": files}, indent=2))


class PrecomputeStore(Store):
    """Store that writes to ``public/__marimo_precompute__/`` for WASM bundling.

    In **native Python** (precomputation and local development):
      - ``put()`` writes cache files to ``public/__marimo_precompute__/``
        relative to the notebook, so ``marimo export html-wasm`` bundles them.
      - ``get()``/``hit()`` read from the same local path.

    In **WASM/Pyodide** (deployed notebook):
      - ``get()``/``hit()`` read from the Pyodide VFS, populated by
        ``prefetch_all()``.
      - ``put()`` is a no-op (cache is pre-built offline).
    """

    def __init__(self, save_path: Optional[str] = None) -> None:
        self._wasm = _is_wasm()
        if not self._wasm:
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

    def get(self, key: str) -> Optional[bytes]:
        if self._wasm:
            return _fetch_bytes_wasm(key)
        path = self._local_path(key)
        if not path.exists() or path.stat().st_size == 0:
            return None
        return path.read_bytes()

    def put(self, key: str, value: bytes) -> bool:
        if self._wasm:
            return False
        path = self._local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        write_manifest(str(self._local_root))
        return True

    def hit(self, key: str) -> bool:
        if self._wasm:
            return _head_wasm(key)
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
