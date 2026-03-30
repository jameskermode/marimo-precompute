# marimo-precompute

[![PyPI](https://img.shields.io/pypi/v/marimo-precompute)](https://pypi.org/project/marimo-precompute/)
[![CI](https://github.com/jameskermode/marimo-precompute/actions/workflows/ci.yml/badge.svg)](https://github.com/jameskermode/marimo-precompute/actions/workflows/ci.yml)
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/jameskermode/marimo-precompute/blob/main/examples/lj_dimer_precompute.py/wasm)

Extends [marimo](https://github.com/marimo-team/marimo)'s [`mo.persistent_cache`](https://docs.marimo.io/api/caching/) with **parameter grid sweeps** and **WASM-compatible loading**, enabling expensive notebook computations to be precomputed offline and served as static WASM apps.

Inspired by Julia's [PlutoSliderServer.jl](https://github.com/JuliaPluto/PlutoSliderServer.jl), which pre-runs Pluto notebooks for all slider combinations and serves results without per-user sessions.

## What this adds over `mo.persistent_cache`

| Feature | `mo.persistent_cache` | `marimo-precompute` |
|---|---|---|
| Cache to disk | Yes | Yes (delegates to marimo) |
| Numpy array serialization | No (JSON fails on arrays) | Yes (`NumpyJsonLoader`) |
| Parameter grid declaration | No | `params={"x": [1,2,3]}` |
| Batch precomputation CLI | No | `marimo-precompute notebook.py` |
| Dry-run feasibility check | No | `--dry-run` reports grid sizes |
| WASM-bundled cache | No | `PrecomputeStore` writes to `public/` |

## Installation

```bash
pip install marimo-precompute
```

This also works with `micropip` in WASM/Pyodide environments:

```python
import micropip
await micropip.install("marimo-precompute")
```

For development:

```bash
git clone https://github.com/jameskermode/marimo-precompute.git
cd marimo-precompute
pip install -e ".[dev]"
```

## Usage

### Decorator (wraps `mo.persistent_cache`)

```python
from marimo_precompute import persistent_cache

@persistent_cache(params={
    "force": [0.0, 1.0, 2.0, 3.0],
    "temperature": [300, 400, 500],
})
def run_simulation(force, temperature):
    return {"energy": ..., "positions": ...}
```

All three `mo.persistent_cache` forms are supported:

```python
@persistent_cache                              # bare decorator
@persistent_cache(method="json")               # with args
with persistent_cache(name="my_block"): ...    # context manager
```

### Precompute CLI

```bash
# Check feasibility (grid sizes, estimated storage)
marimo-precompute notebook.py --dry-run

# Run all parameter combinations
marimo-precompute notebook.py

# Custom cache directory
marimo-precompute notebook.py --cache-dir public/__marimo_precompute__
```

### WASM deployment

Cache files are written to `public/__marimo_precompute__/` alongside the notebook, following marimo's [convention for including data in WASM notebooks](https://docs.marimo.io/guides/wasm/#including-data). This means `marimo export html-wasm` bundles them automatically:

```bash
# Precompute all parameter combinations
marimo-precompute notebook.py

# Export — public/ is bundled into the WASM app
marimo export html-wasm notebook.py -o dist/
```

At runtime, the `PrecomputeStore` uses [`mo.notebook_location()`](https://docs.marimo.io/api/notebook_location/) to resolve cache paths, which returns a local filesystem path in native Python and an HTTP URL in WASM/Pyodide.

## Architecture

A thin integration layer hooking into three marimo extension points:

- **`PrecomputeStore`** — implements marimo's [`Store`](https://github.com/marimo-team/marimo/blob/main/marimo/_save/stores/store.py) ABC. Writes to `public/__marimo_precompute__/` in native Python; reads via `mo.notebook_location()` + `XMLHttpRequest` in WASM.
- **`NumpyJsonLoader`** — extends marimo's [`JsonLoader`](https://github.com/marimo-team/marimo/blob/main/marimo/_save/loaders/json.py) with tagged numpy encoding (`{"__numpy__": true, "data": [...], "dtype": "float64"}`).
- **`persistent_cache`** — wraps `mo.persistent_cache`, adding `params=` for sweep grid registration.

On `import marimo_precompute`, the `NumpyJsonLoader` is registered in marimo's `PERSISTENT_LOADERS` as `"numpy_json"`.

## Related marimo issues

- [#5535 — Automatically bundling data into WASM notebooks](https://github.com/marimo-team/marimo/issues/5535)
- [#3194 — Recommendations for including data in WASM notebooks](https://github.com/marimo-team/marimo/issues/3194)
- [#7849 — Add cached outputs to run mode](https://github.com/marimo-team/marimo/issues/7849)
- [#1831 — Stateless deployment for multi-container scaling](https://github.com/marimo-team/marimo/issues/1831)
- [#3176 — Persistent cache with auto-cleanup](https://github.com/marimo-team/marimo/issues/3176)
- [#2661 — Persistent cache with non-pickle types](https://github.com/marimo-team/marimo/issues/2661)

## License

MIT
