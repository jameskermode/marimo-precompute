# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "marimo-precompute>=0.2.3",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from marimo_precompute import persistent_cache, prefetch_all

    return mo, np, persistent_cache, plt, prefetch_all


@app.cell
async def _(prefetch_all):
    await prefetch_all()
    return


@app.cell
def _(np):
    def lj_force(r):
        """Analytical LJ force at separation r."""
        return 24.0 * (1.0 / r**7 - 2.0 / r**13)

    r_anal = np.linspace(1.05, 2.5, 500)
    F_anal = lj_force(r_anal)
    return F_anal, lj_force, r_anal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LJ Dimer: Force-Extension Curve

    This notebook computes the force-extension curve for a Lennard-Jones
    dimer using quasi-static loading and arclength continuation (simulated).

    The computations are wrapped in `persistent_cache` so they can be
    precomputed offline and loaded instantly in WASM.
    """)
    return


@app.cell
def _(lj_force, np, persistent_cache):
    def _run_qs():
        forces = np.linspace(0, 3.0, 30)
        r_vals = np.linspace(1.0, 3.0, 10000)
        f_vals = lj_force(r_vals)
        seps = []
        for f in forces:
            mask = r_vals < 1.3
            idx = np.argmin(np.abs(f_vals[mask] - f))
            seps.append(float(r_vals[mask][idx]))
        return {"separations": np.array(seps), "forces": forces}

    with persistent_cache(name="qs_run"):
        qs_data = _run_qs()
    return (qs_data,)


@app.cell
def _(lj_force, np, persistent_cache):
    def _run_cont():
        r_vals = np.linspace(1.05, 2.5, 100)
        forces = np.array([lj_force(r) for r in r_vals])
        return {"separations": r_vals, "forces": forces}

    with persistent_cache(name="cont_run"):
        cont_data = _run_cont()
    return (cont_data,)


@app.cell(hide_code=True)
def _(F_anal, cont_data, plt, qs_data, r_anal):
    _fig, _ax = plt.subplots(figsize=(7, 5))
    _ax.plot(F_anal, r_anal, "-", lw=4, color="C1", alpha=0.5, label="Analytical")
    _ax.plot(qs_data["forces"], qs_data["separations"], "s", ms=5, color="C2", label="Quasi-static")
    _ax.plot(cont_data["forces"], cont_data["separations"], "-", lw=1.5, color="C0", label="Continuation")
    _ax.set_xlabel("Applied force F")
    _ax.set_ylabel("Separation r")
    _ax.set_title("LJ Dimer: Force-Extension Curve")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
