#!/usr/bin/env python3
"""Preflight, then start a detached resumable comparison job with a durable log."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    args, _ = p.parse_known_args()
    script = Path(__file__).with_name("run_comparisons.py")
    command = [sys.executable, "-u", str(script), *sys.argv[1:]]
    if any(option in sys.argv for option in ("--help", "-h", "--check-only", "--prepare-only", "--score-only")):
        raise SystemExit(subprocess.run(command).returncode)
    subprocess.run([*command, "--check-only"], check=True)
    logs = args.root.resolve() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    log = logs / f"qwen_epoch1_comparisons_{stamp}.log"
    with log.open("xb") as handle:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    latest = logs / "qwen_epoch1_comparisons_latest.log"
    temporary = logs / (latest.name + f".{os.getpid()}.tmp")
    temporary.symlink_to(log.name)
    temporary.replace(latest)
    print(f"Comparison job started (PID {process.pid}). You can close this terminal.")
    print(f"Log: {log}")
    print(f"Monitor: tail -f {latest}")
    print("A stopped or interrupted job resumes with the same command and unchanged settings.")


if __name__ == "__main__":
    main()
