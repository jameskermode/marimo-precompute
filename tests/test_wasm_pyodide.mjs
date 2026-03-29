/**
 * Test marimo-precompute components in Pyodide (Node.js).
 *
 * Verifies that the parts which run in WASM work correctly:
 *  1. NumpyJsonLoader roundtrips numpy arrays
 *  2. Registry and param_key work
 *  3. Precomputed cache files are valid and readable
 *
 * Note: The full mo.persistent_cache integration can't be tested here
 * because marimo itself doesn't run in Pyodide. These tests cover the
 * data layer that WASM notebooks consume.
 *
 * Requires: npm install pyodide (see package.json)
 * Run:      node tests/test_wasm_pyodide.mjs
 */

import { loadPyodide } from "pyodide";
import { existsSync } from "fs";
import { resolve, join } from "path";
import { fileURLToPath } from "url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const ROOT = resolve(__dirname, "..");

// ── Helpers ──────────────────────────────────────────────────────────────

let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (!condition) {
    console.error(`  FAIL: ${message}`);
    failed++;
  } else {
    console.log(`  PASS: ${message}`);
    passed++;
  }
}

// ── Main ────────────────────────────────────────────────────────────────

async function main() {
  console.log("Loading Pyodide...");
  const pyodide = await loadPyodide();

  // Mount the repo so Pyodide can access source files
  pyodide.FS.mkdirTree("/repo");
  pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: ROOT }, "/repo");

  // Install numpy (Pyodide built-in)
  console.log("Installing numpy...");
  await pyodide.loadPackage("numpy");

  // Add our source to Python path (bypasses __init__.py which needs marimo)
  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/repo/src")
  `);

  // ── Test 1: NumpyJsonLoader roundtrip ───────────────────────────────
  console.log("\nTest 1: NumpyJsonLoader roundtrip in Pyodide");

  await pyodide.runPythonAsync(`
import json
import numpy as np
from marimo_precompute.numpy_json import NumpyEncoder, numpy_object_hook

# Roundtrip a 1D float array
arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
encoded = json.dumps(arr, cls=NumpyEncoder)
decoded = json.loads(encoded, object_hook=numpy_object_hook)
assert isinstance(decoded, np.ndarray), f"Expected ndarray, got {type(decoded)}"
assert np.array_equal(decoded, arr), f"Arrays not equal"
assert decoded.dtype == np.float64, f"Wrong dtype: {decoded.dtype}"

# Roundtrip a 2D int array
arr2d = np.array([[1, 2], [3, 4]], dtype=np.int32)
encoded2 = json.dumps(arr2d, cls=NumpyEncoder)
decoded2 = json.loads(encoded2, object_hook=numpy_object_hook)
assert np.array_equal(decoded2, arr2d), "2D array roundtrip failed"
assert decoded2.dtype == np.int32, f"Wrong dtype: {decoded2.dtype}"

# Nested structure with mixed types
data = {"positions": np.array([[0.0, 1.0], [2.0, 3.0]]), "energy": np.float64(-1.5)}
enc = json.dumps(data, cls=NumpyEncoder)
dec = json.loads(enc, object_hook=numpy_object_hook)
assert np.array_equal(dec["positions"], data["positions"]), "Nested array failed"
assert dec["energy"] == -1.5, "Nested scalar failed"
  `);
  assert(true, "NumpyJsonLoader roundtrips numpy arrays in Pyodide");

  // ── Test 2: Registry works in Pyodide ──────────────────────────────
  console.log("\nTest 2: Registry param_key and grid enumeration");

  await pyodide.runPythonAsync(`
from marimo_precompute.registry import PrecomputeRegistry, _param_key

# Deterministic keys regardless of insertion order
assert _param_key({"a": 1, "b": 2}) == _param_key({"b": 2, "a": 1}), "Keys not deterministic"

# Float rounding avoids floating point noise
k1 = _param_key({"x": 0.1 + 0.2})
k2 = _param_key({"x": 0.3})
assert k1 == k2, f"Float keys differ: {k1} vs {k2}"

# Grid registration and enumeration
reg = PrecomputeRegistry()
reg.register("f", lambda x, y: x + y, {"x": [1, 2], "y": [10, 20, 30]})
entry = reg.entries["f"]
assert entry.total_combinations == 6, f"Expected 6 combos, got {entry.total_combinations}"
combos = list(entry.iter_combinations())
assert len(combos) == 6, f"Expected 6 combos, got {len(combos)}"
assert {"x": 1, "y": 10} in combos, "Missing expected combo"
  `);
  assert(true, "Registry param_key and grid enumeration work in Pyodide");

  // ── Test 3: Precomputed cache files readable in Pyodide ────────────
  console.log("\nTest 3: Precomputed cache files readable in Pyodide");

  const publicDir = join(ROOT, "public", "__marimo_precompute__");
  if (!existsSync(publicDir)) {
    console.log("  SKIP: No precomputed data. Run marimo-precompute first.");
  } else {
    await pyodide.runPythonAsync(`
import json
import os
import numpy as np
from marimo_precompute.numpy_json import numpy_object_hook

cache_dir = "/repo/public/__marimo_precompute__"
assert os.path.isdir(cache_dir), f"Cache dir not found: {cache_dir}"

# Find all JSON cache files
cache_files = []
for root, dirs, files in os.walk(cache_dir):
    for f in files:
        if f.endswith(".json"):
            cache_files.append(os.path.join(root, f))

assert len(cache_files) > 0, "No cache files found"

# Verify each file is valid JSON with results
for path in cache_files:
    with open(path) as f:
        data = json.load(f)
    assert "results" in data, f"Missing 'results' in {path}"
    assert len(data["results"]) > 0, f"Empty results in {path}"

# Verify numpy arrays deserialize correctly from cache
for path in cache_files:
    with open(path) as fh:
        data = json.loads(fh.read(), object_hook=numpy_object_hook)
    for key, val in data["results"].items():
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, (list, np.ndarray)):
                    arr = np.asarray(v)
                    assert arr.size > 0, f"Empty array in {path}:{key}:{k}"
    `);
    assert(true, "Precomputed cache files readable and deserializable in Pyodide");
  }

  // ── Summary ─────────────────────────────────────────────────────────
  console.log(`\n${"=".repeat(60)}`);
  console.log(`Results: ${passed} passed, ${failed} failed`);
  if (failed > 0) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
