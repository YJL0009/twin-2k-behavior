#!/usr/bin/env python3
"""Run all-B LoRA training, epoch selection, and full validation."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import train_qwen_full_b_lora as training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qwen_full_b_lora_v4_b16_a2"))
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--prepared-from", type=Path, default=Path("outputs/qwen_full_b_lora_v1"))
    parser.add_argument("--fresh-preparation", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out if args.out.is_absolute() else root / args.out).resolve()
    directory = Path(__file__).resolve().parent
    train_command = [sys.executable, "-u", str(directory / "train_qwen_full_b_lora.py"),
                     "--root", str(root), "--out", str(out),
                     "--micro-batch-size", str(args.micro_batch_size),
                     "--gradient-accumulation", str(args.gradient_accumulation)]
    if not args.fresh_preparation:
        train_command += ["--prepared-from", str(args.prepared_from)]
    # Direct trainer launches may not hold the shell launcher's lock. Hold the
    # prior run's own locks as well before reusing its data or taking the GPU.
    source_locks = []
    if not args.fresh_preparation:
        source = (args.prepared_from if args.prepared_from.is_absolute()
                  else root / args.prepared_from).resolve()
        if source != out:
            for filename in (".pipeline.lock", ".training.lock"):
                path = source / filename
                if path.is_file():
                    handle = path.open("a+")
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close()
                        raise RuntimeError("The previous LoRA run is still running; stop it before starting the new run")
                    source_locks.append(handle)
    if args.check_only:
        subprocess.run(train_command + ["--check-only"], check=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".pipeline.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("The full-B LoRA pipeline is already running")
    status = out / "pipeline_status.json"
    stages = [("training_lora", train_command), ("validation", [sys.executable, "-u",
        str(directory / "evaluate_qwen_full_b_lora.py"), "--root", str(root), "--run", str(out)])]
    for name, command in stages:
        training.write_json(status, {"state": name, "started_unix": time.time(), "test_evaluated": False,
                                    "pid": os.getpid()})
        print(f"Starting full-B LoRA {name} stage", flush=True)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            training.write_json(status, {"state": "failed", "stage": name, "exit_code": error.returncode,
                                        "updated_unix": time.time(), "test_evaluated": False})
            raise
    training.write_json(status, {"state": "complete", "updated_unix": time.time(), "test_evaluated": False})
    print("FULL-B LORA TRAINING AND VALIDATION COMPLETE", flush=True)
    print("Report:", out / "full_b_lora_validation_report.txt", flush=True)
    print("Selected base model and adapter:", out / "best_checkpoint.json", flush=True)
    print("The 309 test participants were not evaluated.", flush=True)


if __name__ == "__main__":
    main()
