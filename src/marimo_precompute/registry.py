"""Global registry for precomputed functions and their parameter grids."""

from __future__ import annotations

import hashlib
import json
import os
from itertools import product
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


class FunctionEntry:
    """A registered function with its parameter grid and metadata."""

    __slots__ = ("name", "func", "params", "param_names", "param_values")

    def __init__(
        self,
        name: str,
        func: Callable,
        params: dict[str, Sequence],
    ):
        self.name = name
        self.func = func
        self.params = params
        self.param_names = list(params.keys())
        self.param_values = [list(params[k]) for k in self.param_names]

    @property
    def grid_sizes(self) -> list[int]:
        return [len(v) for v in self.param_values]

    @property
    def total_combinations(self) -> int:
        n = 1
        for s in self.grid_sizes:
            n *= s
        return n

    def iter_combinations(self):
        """Yield dicts of parameter values for every grid point."""
        for combo in product(*self.param_values):
            yield dict(zip(self.param_names, combo))


def _param_key(params: dict) -> str:
    """Deterministic string key for a parameter combination."""
    # Round floats to avoid floating point noise in keys
    parts = []
    for k in sorted(params):
        v = params[k]
        if isinstance(v, float):
            v = round(v, 12)
        parts.append(f"{k}={v!r}")
    return ",".join(parts)


class PrecomputeRegistry:
    """Stores registered functions, their grids, and cached results."""

    def __init__(self, cache_dir: str | Path = "__marimo__/cache"):
        self.entries: dict[str, FunctionEntry] = {}
        self.cache_dir = Path(cache_dir)

    def register(
        self,
        name: str,
        func: Callable,
        params: dict[str, Sequence],
    ) -> FunctionEntry:
        entry = FunctionEntry(name, func, params)
        self.entries[name] = entry
        return entry

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.json"

    def load_cache(self, name: str) -> dict[str, Any] | None:
        """Load cached results for a function. Returns {param_key: result} or None."""
        path = self._cache_path(name)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def save_cache(self, name: str, results: dict[str, Any]) -> None:
        """Save results dict to cache file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(name)
        with open(path, "w") as f:
            json.dump(results, f, cls=_NumpyEncoder)

    def lookup(self, name: str, params: dict) -> Any | None:
        """Look up a single cached result. Returns None if not cached."""
        cache = self.load_cache(name)
        if cache is None:
            return None
        key = _param_key(params)
        results = cache.get("results", {})
        if key not in results:
            return None
        return results[key]

    def dry_run(self) -> list[dict[str, Any]]:
        """Report on all registered functions and their grid sizes.

        Returns a list of dicts with keys: name, param_names, grid_sizes,
        total_combinations, estimated_bytes (if a sample result exists).
        """
        reports = []
        for name, entry in self.entries.items():
            report: dict[str, Any] = {
                "name": name,
                "param_names": entry.param_names,
                "grid_sizes": entry.grid_sizes,
                "total_combinations": entry.total_combinations,
            }

            # Try to estimate size from a single cached result or by running once
            cache = self.load_cache(name)
            if cache and cache.get("results"):
                sample_key = next(iter(cache["results"]))
                sample_json = json.dumps(
                    cache["results"][sample_key], cls=_NumpyEncoder
                )
                report["estimated_bytes_per_result"] = len(sample_json.encode())
                report["estimated_total_bytes"] = (
                    report["estimated_bytes_per_result"] * entry.total_combinations
                )

            reports.append(report)
        return reports

    def sweep(
        self,
        name: str | None = None,
        *,
        max_combinations: int = 10_000,
        verbose: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Run precomputation for registered functions.

        Parameters
        ----------
        name : str, optional
            If given, only sweep this function. Otherwise sweep all.
        max_combinations : int
            Abort if total combinations exceed this threshold.
        verbose : bool
            Print progress.

        Returns
        -------
        dict mapping function names to {param_key: result} dicts.
        """
        entries = (
            {name: self.entries[name]} if name else self.entries
        )
        all_results = {}

        for func_name, entry in entries.items():
            total = entry.total_combinations
            if total > max_combinations:
                raise ValueError(
                    f"{func_name}: {total:,} combinations exceeds limit "
                    f"of {max_combinations:,}. Use --dry-run to inspect, "
                    f"or increase --max-combinations."
                )

            # Load existing cache to skip already-computed values
            existing = self.load_cache(func_name)
            existing_results = existing.get("results", {}) if existing else {}

            results = dict(existing_results)
            computed = 0
            skipped = 0

            for i, params in enumerate(entry.iter_combinations()):
                key = _param_key(params)
                if key in results:
                    skipped += 1
                    continue
                result = entry.func(**params)
                results[key] = result
                computed += 1
                if verbose and (computed % 100 == 0 or computed + skipped == total):
                    print(
                        f"  {func_name}: {computed + skipped}/{total} "
                        f"({computed} computed, {skipped} cached)"
                    )

            cache_data = {
                "name": func_name,
                "param_names": entry.param_names,
                "param_values": [
                    [_serialize_value(v) for v in vals]
                    for vals in entry.param_values
                ],
                "results": results,
            }
            self.save_cache(func_name, cache_data)
            all_results[func_name] = results

            if verbose:
                print(
                    f"  {func_name}: done ({computed} computed, "
                    f"{skipped} cached, {total} total)"
                )

        return all_results


def _serialize_value(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# Global singleton registry
_global_registry: PrecomputeRegistry | None = None


def get_registry(cache_dir: str | Path = "__marimo__/cache") -> PrecomputeRegistry:
    """Get or create the global PrecomputeRegistry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PrecomputeRegistry(cache_dir)
    return _global_registry
