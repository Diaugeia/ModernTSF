"""Lazy single-run and sweep execution entry points.

Keeping this package initializer import-free prevents the evaluation and runner
layers from pulling each other in while their focused submodules are loading.
"""

__all__ = ["run_one", "run_sweep"]


def __getattr__(name: str):
    if name == "run_one":
        from moderntsf.benchmark.runner.run_one import run_one

        globals()[name] = run_one
        return run_one
    if name == "run_sweep":
        from moderntsf.benchmark.runner.run_sweep import run_sweep

        globals()[name] = run_sweep
        return run_sweep
    raise AttributeError(name)
