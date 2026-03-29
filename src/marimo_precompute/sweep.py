"""CLI for running parameter sweeps and dry-run analysis."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path


def _format_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _load_notebook(path: str) -> None:
    """Import a marimo notebook file to trigger @cached registrations.

    We exec the module-level code which includes the @cached decorators.
    The decorators register functions in the global registry as a side effect.
    """
    p = Path(path).resolve()
    if not p.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    # Add the notebook's directory to sys.path so relative imports work
    notebook_dir = str(p.parent)
    if notebook_dir not in sys.path:
        sys.path.insert(0, notebook_dir)

    spec = importlib.util.spec_from_file_location("__notebook__", p)
    if spec is None or spec.loader is None:
        print(f"Error: could not load {path}", file=sys.stderr)
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        # Marimo notebooks may fail to fully execute outside marimo runtime,
        # but the @cached decorators at module level should still register.
        # Only warn if the registry is empty.
        pass


def do_dry_run(registry, verbose: bool = True) -> list[dict]:
    """Print a dry-run report and return the reports list."""
    reports = registry.dry_run()

    if not reports:
        print("No @cached functions registered.")
        return reports

    total_combos = 0
    total_bytes = 0

    for r in reports:
        combos = r["total_combinations"]
        total_combos += combos

        print(f"\n  {r['name']}:")
        for pname, gsize in zip(r["param_names"], r["grid_sizes"]):
            print(f"    {pname}: {gsize} values")
        print(f"    Total combinations: {combos:,}")

        if "estimated_total_bytes" in r:
            est = r["estimated_total_bytes"]
            total_bytes += est
            print(f"    Estimated storage: {_format_bytes(est)}")
            per = r["estimated_bytes_per_result"]
            print(f"    ({_format_bytes(per)} per result)")

    print(f"\n  Grand total: {total_combos:,} combinations")
    if total_bytes > 0:
        print(f"  Estimated total storage: {_format_bytes(total_bytes)}")

    # Warnings
    if total_combos > 100_000:
        print(
            "\n  WARNING: >100,000 combinations. This will likely be slow "
            "and produce large files."
        )
        print("  Consider:")
        print("    - Reducing grid resolution")
        print("    - Identifying independent parameter groups")
        print("    - Using a live server instead (Approach B)")
    elif total_combos > 10_000:
        print(
            "\n  NOTE: >10,000 combinations. This is feasible but may take "
            "a while and produce sizable output."
        )

    return reports


def do_sweep(registry, args) -> None:
    """Run the actual precomputation sweep."""
    name = args.name if hasattr(args, "name") and args.name else None
    max_combos = args.max_combinations

    print(f"Starting sweep (max {max_combos:,} combinations)...")
    t0 = time.time()

    try:
        results = registry.sweep(
            name=name,
            max_combinations=max_combos,
            verbose=True,
        )
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("Use --dry-run to inspect, or increase --max-combinations.")
        sys.exit(1)

    elapsed = time.time() - t0
    total = sum(len(v) for v in results.values())
    print(f"\nDone: {total} results in {elapsed:.1f}s")
    print(f"Cache directory: {registry.cache_dir}")


def main():
    parser = argparse.ArgumentParser(
        prog="marimo-precompute",
        description="Precompute cached function results for WASM deployment",
    )
    parser.add_argument(
        "notebook",
        help="Path to the marimo notebook (.py file)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report parameter grid sizes and estimated storage without computing",
    )
    parser.add_argument(
        "--name",
        help="Only sweep this specific @cached function",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=10_000,
        help="Abort if total combinations exceed this (default: 10,000)",
    )
    parser.add_argument(
        "--cache-dir",
        default="public/__marimo_precompute__",
        help="Directory for cached results (default: public/__marimo_precompute__)",
    )
    parser.add_argument(
        "--time-one",
        action="store_true",
        help="With --dry-run, time a single function call to estimate total time",
    )

    args = parser.parse_args()

    # Set up registry with the specified cache dir
    from marimo_precompute.registry import get_registry
    registry = get_registry(args.cache_dir)

    # Load notebook to trigger @cached registrations
    print(f"Loading {args.notebook}...")
    _load_notebook(args.notebook)

    if not registry.entries:
        print("No @cached functions found in notebook.")
        sys.exit(0)

    print(f"Found {len(registry.entries)} @cached function(s)")

    if args.time_one:
        # Time a single call for estimation
        for name, entry in registry.entries.items():
            first_combo = next(entry.iter_combinations())
            print(f"\n  Timing one call to {name}({first_combo})...")
            t0 = time.time()
            entry.func(**first_combo)
            elapsed = time.time() - t0
            total_est = elapsed * entry.total_combinations
            print(f"  Single call: {elapsed:.3f}s")
            print(f"  Estimated total: {total_est:.1f}s ({total_est/60:.1f}min)")

    if args.dry_run:
        do_dry_run(registry, verbose=True)
    else:
        do_sweep(registry, args)


if __name__ == "__main__":
    main()
