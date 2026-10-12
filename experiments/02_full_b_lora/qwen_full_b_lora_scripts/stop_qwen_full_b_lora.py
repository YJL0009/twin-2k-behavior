#!/usr/bin/env python3
"""Stop one verified all-B Qwen LoRA job and its own descendants."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import sys
import time


DEFAULT_PROJECT = Path("/root/autodl-tmp/twin_qwen")
ALLOWED_SCRIPTS = {"run_qwen_full_b_lora_pipeline.py", "train_qwen_full_b_lora.py"}


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    start_time: int
    state: str


def process_info(pid: int) -> Process | None:
    """Linux start-time ticks distinguish a reused PID from the original job."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rindex(")") + 2:].split()
        return Process(pid, int(fields[1]), int(fields[19]), fields[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_live_process(expected: Process) -> bool:
    current = process_info(expected.pid)
    return (current is not None and current.start_time == expected.start_time
            and current.state not in {"Z", "X", "x"})


def python_script_argument(argv: list[str]) -> int:
    """Find the actual Python script, rather than a script name inside -c text."""
    if not argv or not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(argv[0]).name):
        raise ValueError("The PID is not a Python LoRA pipeline or trainer.")
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            index += 1
            break
        if not argument.startswith("-"):
            break
        if argument in {"-c", "-m"} or argument.startswith(("-c", "-m")):
            raise ValueError("Refusing a Python -c or -m process.")
        if argument in {"-W", "-X"}:
            index += 2
            continue
        if argument.startswith(("-W", "-X")) or argument in {
            "-u", "-B", "-E", "-s", "-S", "-I", "-O", "-OO", "-q", "-v", "-b", "-bb", "-x"
        }:
            index += 1
            continue
        raise ValueError(f"Unrecognized Python invocation option: {argument}")
    if index >= len(argv) or Path(argv[index]).name not in ALLOWED_SCRIPTS:
        raise ValueError("The PID does not execute run_qwen_full_b_lora_pipeline.py or train_qwen_full_b_lora.py.")
    return index


def verify_root(expected: Process, project: Path) -> str:
    try:
        argv = Path(f"/proc/{expected.pid}/cmdline").read_bytes().decode(errors="surrogateescape").split("\0")
        argv = [argument for argument in argv if argument]
        cwd = Path(os.readlink(f"/proc/{expected.pid}/cwd")).resolve()
    except (FileNotFoundError, ProcessLookupError) as error:
        raise ValueError("The process exited during verification; run this command again.") from error
    script_index = python_script_argument(argv)
    arguments = argv[script_index + 1:]
    root_values = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--root":
            if index + 1 >= len(arguments):
                raise ValueError("The process has an incomplete --root argument.")
            root_values.append(arguments[index + 1])
            index += 1
        elif argument.startswith("--root="):
            root_values.append(argument.split("=", 1)[1])
        elif argument.startswith("--roo"):
            raise ValueError("Use an explicit --root for unambiguous process verification.")
        index += 1
    if root_values:
        for value in root_values:
            path = Path(value)
            resolved = (path if path.is_absolute() else cwd / path).resolve()
            if resolved != project:
                raise ValueError("The process --root belongs to a different project; nothing was stopped.")
    elif cwd != project or project != DEFAULT_PROJECT.resolve():
        raise ValueError("No matching explicit --root was found and the process is not in the default project.")
    if not same_live_process(expected):
        raise ValueError("The PID changed or exited during verification; run this command again.")
    return Path(argv[script_index]).name


def add_descendants(tracked: dict[int, Process]) -> None:
    """Add children only while their already-verified parent still has its identity."""
    active = {pid for pid, item in tracked.items() if same_live_process(item)}
    snapshot = {}
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit():
            item = process_info(int(directory.name))
            if item is not None:
                snapshot[item.pid] = item
    while active:
        additions = []
        for item in snapshot.values():
            if item.pid in tracked or item.ppid not in active:
                continue
            parent = tracked[item.ppid]
            if same_live_process(parent) and same_live_process(item):
                additions.append(item)
        active = set()
        for item in additions:
            tracked[item.pid] = item
            active.add(item.pid)


def signal_tracked(tracked: dict[int, Process], root_pid: int, sig: int) -> None:
    # Children go first; the pipeline root is signalled last. No process groups.
    order = [item for pid, item in tracked.items() if pid != root_pid]
    order.reverse()
    if root_pid in tracked:
        order.append(tracked[root_pid])
    for item in order:
        if not same_live_process(item):
            continue
        try:
            if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                # The descriptor pins this process across any PID-reuse race.
                descriptor = os.pidfd_open(item.pid)
                try:
                    if same_live_process(item):
                        signal.pidfd_send_signal(descriptor, sig)
                finally:
                    os.close(descriptor)
            elif same_live_process(item):
                os.kill(item.pid, sig)
        except ProcessLookupError:
            pass


def wait_for_stop(tracked: dict[int, Process], root_pid: int, seconds: float, sig: int) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        old_pids = set(tracked)
        add_descendants(tracked)
        newly_seen = {pid: item for pid, item in tracked.items() if pid not in old_pids}
        if newly_seen:
            signal_tracked(newly_seen, root_pid, sig)
        if not any(same_live_process(item) for item in tracked.values()):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("QWEN_PROJECT_ROOT", str(DEFAULT_PROJECT))))
    parser.add_argument("--pid", type=int, help="An explicit observed pipeline or trainer PID.")
    args = parser.parse_args()
    project = args.root.resolve()
    pid_file = project / "logs/qwen_full_b_lora.pid"
    if args.pid is None:
        if not pid_file.is_file():
            print("No recorded LoRA PID was found. For a foreground job use Ctrl+C; otherwise rerun with --pid followed by its observed pipeline or trainer PID.", file=sys.stderr)
            return 1
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            print("The recorded LoRA PID is invalid; supply the observed PID using --pid.", file=sys.stderr)
            return 1
    else:
        pid = args.pid
    if pid <= 1 or pid == os.getpid():
        print("Refusing an invalid or protected PID.", file=sys.stderr)
        return 1
    try:
        root = process_info(pid)
        if root is None or root.state in {"Z", "X", "x"}:
            print(f"LoRA PID {pid} is already stopped. Nothing was signalled.")
            return 0
        script = verify_root(root, project)
        tracked = {pid: root}
        add_descendants(tracked)
        print(f"Stopping verified {script} PID {pid} and {len(tracked) - 1} descendants.", flush=True)
        signal_tracked(tracked, pid, signal.SIGINT)
        if not wait_for_stop(tracked, pid, 3.0, signal.SIGINT):
            print("Some verified job processes remain; sending SIGTERM.", flush=True)
            signal_tracked(tracked, pid, signal.SIGTERM)
            if not wait_for_stop(tracked, pid, 5.0, signal.SIGTERM):
                remaining = [item.pid for item in tracked.values() if same_live_process(item)]
                print(f"Job processes remain: {remaining}. No SIGKILL was sent. Do not start another job until they stop.", file=sys.stderr)
                return 1
        print("The verified LoRA job is stopped. Existing checkpoint files were left unchanged.")
        return 0
    except (OSError, ValueError) as error:
        print(f"Refusing to stop this PID: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
