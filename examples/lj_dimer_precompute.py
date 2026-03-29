"""Example: LJ dimer force-extension using marimo-precompute.

This demonstrates how the LACT demo1 pattern can be simplified using
persistent_cache with parameter grids.

Usage:
    # Dry run to check feasibility:
    marimo-precompute examples/lj_dimer_precompute.py --dry-run

    # Precompute all results:
    marimo-precompute examples/lj_dimer_precompute.py

    # Run the notebook:
    marimo run examples/lj_dimer_precompute.py
"""

import numpy as np
from marimo_precompute import persistent_cache


def _lj_force(r):
    """Analytical LJ force at separation r."""
    return 24.0 * (1.0 / r**7 - 2.0 / r**13)


@persistent_cache(params={
    "force_max": [3.0],
    "n_steps": [30],
})
def quasi_static_run(force_max, n_steps):
    """Simulate quasi-static loading of LJ dimer."""
    forces = np.linspace(0, force_max, n_steps)
    r_vals = np.linspace(1.0, 3.0, 10000)
    f_vals = _lj_force(r_vals)

    seps, applied = [], []
    for f in forces:
        mask = r_vals < 1.3
        idx = np.argmin(np.abs(f_vals[mask] - f))
        seps.append(float(r_vals[mask][idx]))
        applied.append(float(f))

    return {"separations": seps, "forces": applied}


@persistent_cache(params={
    "ds_default": [0.01],
    "n_iter": [100],
})
def continuation_run(ds_default, n_iter):
    """Simulate arclength continuation of LJ dimer."""
    r_vals = np.linspace(1.05, 2.5, n_iter)
    forces = [float(_lj_force(r)) for r in r_vals]
    seps = [float(r) for r in r_vals]
    return {"separations": seps, "forces": forces}
