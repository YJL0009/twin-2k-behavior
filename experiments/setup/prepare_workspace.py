#!/usr/bin/env python3
"""Stage the unchanged full-B engines at their required runtime location.

This does not download data, create training metadata, load weights, or train.
Existing different files are refused; no run's source hashes are bypassed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "experiments/02_full_b_lora/qwen_full_b_lora_scripts"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    if workspace == REPO or REPO in workspace.parents:
        raise SystemExit("Choose a private workspace outside this public source tree.")
    if not SOURCE.is_dir():
        raise SystemExit(f"Missing packaged engines: {SOURCE}")
    target = workspace / "qwen_full_b_lora_scripts"
    files = sorted(p for p in SOURCE.rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts)
    # Check all collisions before copying any file.
    for source in files:
        dest = target / source.relative_to(SOURCE)
        if dest.exists() and (not dest.is_file() or digest(dest) != digest(source)):
            raise SystemExit(f"Different runtime source already exists: {dest}. "
                             "Use its exact historical source or a fresh workspace.")
    for source in files:
        dest = target / source.relative_to(SOURCE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(source, dest)
    print(json.dumps({"workspace": str(workspace), "engine_directory": str(target),
                      "files_verified": len(files), "training_launched": False}, indent=2))


if __name__ == "__main__":
    main()
