"""Command-line entrypoint for running benchmarks.

This module wires together config loading, registry setup, and
the sweep runner.
"""

from __future__ import annotations

import argparse

from moderntsf.benchmark.config import load_config
from moderntsf.benchmark.registry.loader import register_from_config
from moderntsf.benchmark.runner.run_sweep import run_sweep


def main() -> None:
    """Parse arguments and run the configured sweep.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="ModernTSF runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Path to config TOML")
    group.add_argument("--payload", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.payload:
        # A detached queue worker may disappear. Never leave unmonitored compute
        # alive after its budget controller dies; inherited GPU leases stay held
        # until this process exits, allowing the queue to resume safely.
        import os
        import signal
        import threading
        import time
        controller = os.environ.get("MODERNTSF_CONTROLLER_PID")
        if controller:
            if os.getppid() != int(controller):
                os.killpg(os.getpgrp(), signal.SIGTERM)
                raise SystemExit(1)
            def watch_controller():
                while True:
                    if os.getppid() != int(controller):
                        os.killpg(os.getpgrp(), signal.SIGTERM)
                        return
                    time.sleep(.2)
            threading.Thread(target=watch_controller, daemon=True).start()
        import signal
        from moderntsf.benchmark.infra.runs import read_run
        from moderntsf.benchmark.infra.execution import validated_config
        from moderntsf.benchmark.runner.run_one import run_one
        def interrupted(signum, frame):
            raise KeyboardInterrupt("execution cancelled")
        signal.signal(signal.SIGTERM, interrupted)
        saved = read_run(args.payload)
        config = validated_config(saved["config"])
        register_from_config(config)
        run_one(config, saved["raw"], saved["sweep_keys"])
        return
    configs = load_config(args.config)
    for loaded in configs:
        register_from_config(loaded.config)
    run_sweep(configs)


if __name__ == "__main__":
    main()
