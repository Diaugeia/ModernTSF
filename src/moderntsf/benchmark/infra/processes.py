"""Owned child-process launch and termination, independent of run accounting."""

import os
import signal
import subprocess
import sys
import time

from moderntsf.benchmark.infra.storage import write_json


def terminate(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def launch_run(path, policy, gpus, attempt, log):
    env = os.environ.copy()
    env.pop("MODERNTSF_RESEARCH_ROUND", None)
    env.update(
        MODERNTSF_RUN_DIR=str(path),
        MODERNTSF_POLICY=policy.model_dump_json(),
        MODERNTSF_ATTEMPT=str(attempt),
        MODERNTSF_CONTROLLER_PID=str(os.getpid()),
    )
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        env["MODERNTSF_ASSIGNED_GPUS"] = str(len(gpus))
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "moderntsf.benchmark.run_config", "--payload", str(path)],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        pass_fds=getattr(gpus, "descriptors", ()),
    )


def monitor(process, path, gpus, started, timeout_minutes, reason):
    """Return a terminal status; caller supplies cancellation/budget decisions."""
    while process.poll() is None:
        outcome = reason()
        if timeout_minutes and time.monotonic() - started >= timeout_minutes * 60:
            outcome = "timed_out"
        write_json(
            path / "runtime.json",
            {
                "pid": process.pid,
                "heartbeat": time.time(),
                "gpus": gpus,
                "elapsed_sec": time.monotonic() - started,
            },
        )
        if outcome:
            terminate(process)
            return outcome
        time.sleep(0.2)
    return "succeeded" if process.returncode == 0 else "failed"
