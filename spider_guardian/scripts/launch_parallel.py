"""Launch multiple Spider Guardian processes in parallel."""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

CommandSpec = Dict[str, object]

COMMANDS: Dict[str, CommandSpec] = {
    "guardian": {
        "cmd": [
            PYTHON,
            "-m",
            "spider_guardian.scripts.guardian_orchestrator",
            "--log-level",
            "INFO",
            "--selenium-driver",
            "firefox",
        ],
    },
    "bot-live": {
        "cmd": [
            PYTHON,
            "-m",
            "spider_guardian",
            "--respond",
            "-1",
            "--stream-posts",
            "0",
            "--log-level",
            "INFO",
            "--selenium-driver",
            "firefox",
        ],
    },
    "autoposter": {
        "cmd": [
            PYTHON,
            "-m",
            "spider_guardian.scripts.auto_poster",
            "--daily-rate",
            "2.5",
            "--selenium-driver",
            "firefox",
        ],
    },
    "followups": {
        "cmd": [
            PYTHON,
            "spider_replies_to_replies.py",
            "--respond",
            "-1",
            "--selenium-driver",
            "firefox",
            "--log-level",
            "INFO",
        ],
    },
}


def _spawn(name: str, spec: CommandSpec) -> subprocess.Popen[str]:
    env = os.environ.copy()
    extra_env = spec.get("env")
    if isinstance(extra_env, dict):
        env.update({str(k): str(v) for k, v in extra_env.items()})

    cwd = spec.get("cwd")
    proc_cwd = Path(cwd).resolve() if isinstance(cwd, (str, os.PathLike)) else REPO_ROOT

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    return subprocess.Popen(
        spec["cmd"],
        cwd=proc_cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )


def _forward_output(name: str, proc: subprocess.Popen[str]) -> None:
    prefix = f"[{name}] "
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(prefix + line)
    sys.stdout.flush()


def _terminate(procs: Dict[str, subprocess.Popen[str]]) -> None:
    if not procs:
        return
    for name, proc in procs.items():
        if proc.poll() is not None:
            continue
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                proc.terminate()
        except Exception:
            pass
    for proc in procs.values():
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)


def _validate_commands(requested: Iterable[str]) -> List[str]:
    chosen: List[str] = []
    for name in requested:
        if name not in COMMANDS:
            raise SystemExit(f"Unknown command '{name}'. Use --list to view options.")
        chosen.append(name)
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Spider Guardian scripts in parallel")
    parser.add_argument(
        "commands",
        nargs="*",
        default=list(COMMANDS.keys()),
        help="Subset of command names to launch (default: all)",
    )
    parser.add_argument("--list", action="store_true", help="List available command names and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print("Available commands:")
        for name, spec in COMMANDS.items():
            cmd = " ".join(spec["cmd"])  # type: ignore[index]
            print(f"  {name:12s} -> {cmd}")
        return

    names = _validate_commands(args.commands)
    procs: Dict[str, subprocess.Popen[str]] = {}
    threads: Dict[str, threading.Thread] = {}

    try:
        for name in names:
            spec = COMMANDS[name]
            proc = _spawn(name, spec)
            procs[name] = proc
            thread = threading.Thread(target=_forward_output, args=(name, proc), daemon=True)
            thread.start()
            threads[name] = thread
            print(f"Started '{name}' (pid={proc.pid})")

        while procs:
            for name, proc in list(procs.items()):
                code = proc.poll()
                if code is None:
                    continue
                print(f"Process '{name}' exited with code {code}")
                _terminate({k: v for k, v in procs.items() if k != name})
                return
            time.sleep(1)
    except KeyboardInterrupt:
        print("Ctrl+C received, terminating child processes...")
        _terminate(procs)
    finally:
        for thread in threads.values():
            thread.join(timeout=2)


__all__ = ["COMMANDS", "main"]


if __name__ == "__main__":
    main()
