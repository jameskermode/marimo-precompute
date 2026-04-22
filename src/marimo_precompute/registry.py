"""Global registry for precomputed functions and their parameter grids.

The registry only tracks *which* functions have parameter grids and what those
grids look like — it no longer caches results itself. Result caching is
delegated to marimo's LazyLoader via ``mo.persistent_cache``: when the CLI
sweep calls each registered function, the wrapped callable writes cache
files under ``public/__marimo_precompute__/`` automatically.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Callable, Sequence


class FunctionEntry:
    """A registered function with its parameter grid."""

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


class PrecomputeRegistry:
    """Stores registered functions and their grids."""

    def __init__(self, cache_dir: str | Path = "public/__marimo_precompute__"):
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

    def dry_run(self) -> list[dict[str, Any]]:
        """Report grid sizes and total combinations for every registered function."""
        return [
            {
                "name": name,
                "param_names": entry.param_names,
                "grid_sizes": entry.grid_sizes,
                "total_combinations": entry.total_combinations,
            }
            for name, entry in self.entries.items()
        ]

    def sweep(
        self,
        name: str | None = None,
        *,
        max_combinations: int = 10_000,
        verbose: bool = True,
    ) -> dict[str, int]:
        """Invoke each registered function across its parameter grid.

        The wrapped function (returned by ``persistent_cache``) handles
        caching transparently: each call either hits an existing cache
        file or computes-then-writes. Returns a dict mapping function
        names to the number of combinations invoked.
        """
        entries = {name: self.entries[name]} if name else self.entries
        counts: dict[str, int] = {}

        for func_name, entry in entries.items():
            total = entry.total_combinations
            if total > max_combinations:
                raise ValueError(
                    f"{func_name}: {total:,} combinations exceeds limit "
                    f"of {max_combinations:,}. Use --dry-run to inspect, "
                    f"or increase --max-combinations."
                )

            for i, params in enumerate(entry.iter_combinations(), start=1):
                entry.func(**params)
                if verbose and (i % 100 == 0 or i == total):
                    print(f"  {func_name}: {i}/{total}")

            counts[func_name] = total
            if verbose:
                print(f"  {func_name}: done ({total} combinations)")

        return counts


_global_registry: PrecomputeRegistry | None = None


def get_registry(cache_dir: str | Path = "public/__marimo_precompute__") -> PrecomputeRegistry:
    """Get or create the global PrecomputeRegistry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PrecomputeRegistry(cache_dir)
    return _global_registry
