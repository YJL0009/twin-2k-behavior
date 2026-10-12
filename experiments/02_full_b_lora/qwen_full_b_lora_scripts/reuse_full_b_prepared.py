#!/usr/bin/env python3
"""Reuse only verified all-B preparation artifacts in a separate run directory.

The data preparation modules deliberately remain unchanged. Their usual cache
validation still runs after this helper, including the current input, tokenizer,
dependency and script fingerprints. This helper never copies training state.
"""
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


PROTOCOL = "twin-full-b-train-B-validation-inputs-v1"
ARTIFACTS = {
    "train.arrow": "prepared/train.arrow",
    "validation.arrow": "prepared/validation.arrow",
    "training_manifest.jsonl": "training_manifest.jsonl",
    "validation_manifest.jsonl": "prepared/validation_manifest.jsonl",
    "training_baselines.json": "training_baselines.json",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _exists(path):
    return os.path.lexists(path)


def _regular(path):
    return path.is_file() and not path.is_symlink()


def _rename_no_replace(source, destination):
    """Atomic Linux fallback when the destination filesystem lacks hardlinks."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOTSUP, "Atomic no-clobber rename is unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                       ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    # AT_FDCWD=-100; RENAME_NOREPLACE=1. Never replace an unrelated file.
    result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def _verify_existing(path, expected):
    _require(_regular(path), f"Refusing a nonregular or symlink destination: {path}")
    _require(_digest(path) == expected,
             f"Destination artifact differs; choose a new output directory: {path}")


def _install(source, destination, expected):
    """Publish a verified temporary hardlink/copy atomically, without replacing."""
    if _exists(destination):
        _verify_existing(destination, expected)
        return "existing"
    temporary = destination.with_name(f".{destination.name}.reuse-{uuid.uuid4().hex}.tmp")
    method = "hardlink"
    try:
        try:
            os.link(source, temporary)
        except OSError:
            method = "copy"
            # Exclusive creation also avoids following an existing symlink.
            with temporary.open("xb") as output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        _require(_digest(temporary) == expected,
                 f"Source artifact changed during preparation reuse: {source}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            _verify_existing(destination, expected)
            return "existing"
        except OSError:
            try:
                _rename_no_replace(temporary, destination)
            except FileExistsError:
                _verify_existing(destination, expected)
                return "existing"
        return method
    finally:
        if _exists(temporary):
            temporary.unlink()


def reuse_prepared(source, destination):
    """Reuse seven explicit files; return a small action/count report.

    A destination with completed preparation is left untouched: the caller's
    unchanged ``prepare_training`` validates it next. A partial destination is
    resumed only when every already-present mapped file exactly matches the
    verified source. The completion summary is installed last.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    _require(source != destination, "Preparation reuse source and destination must differ")
    _require(source not in destination.parents and destination not in source.parents,
             "Preparation reuse directories must not contain one another")
    completed = destination / "preparation_summary.json"
    if _exists(completed):
        _require(_regular(completed), "Completed destination summary must be a regular file")
        return {"action": "destination_already_prepared", "installed_files": 0}

    config_path = source / "preparation_config.json"
    summary_path = source / "preparation_summary.json"
    _require(_regular(config_path) and _regular(summary_path),
             "Reuse source needs completed preparation_config.json and preparation_summary.json")
    config_bytes, summary_bytes = config_path.read_bytes(), summary_path.read_bytes()
    config, summary = json.loads(config_bytes), json.loads(summary_bytes)
    _require(isinstance(config, dict) and isinstance(summary, dict),
             "Source preparation metadata must contain JSON objects")
    _require(config.get("protocol") == PROTOCOL and summary.get("protocol") == PROTOCOL,
             "Source preparation protocol does not match the all-B data protocol")
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, allow_nan=False)
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _require(summary.get("config_hash") == config_hash,
             "Source preparation configuration hash does not match")
    hashes = summary.get("artifacts_sha256")
    _require(isinstance(hashes, dict) and set(hashes) == set(ARTIFACTS),
             "Source preparation must list exactly the five expected artifact hashes")
    files = []
    for name, relative in ARTIFACTS.items():
        path, expected = source / relative, hashes[name]
        _require(isinstance(expected, str) and len(expected) == 64
                 and all(character in "0123456789abcdef" for character in expected),
                 f"Source preparation has an invalid SHA256 for {name}")
        _require(_regular(path) and _digest(path) == expected,
                 f"Source prepared artifact changed or is missing: {name}")
        files.append((path, destination / relative, expected))
    # Metadata is byte-for-byte preserved; no output-relative paths are rewritten.
    files.extend((path, destination / path.name, hashlib.sha256(value).hexdigest())
                 for path, value in ((config_path, config_bytes), (summary_path, summary_bytes)))

    # Validate the full destination before copying anything, even on partial retry.
    _require(not destination.exists() or destination.is_dir(),
             "Preparation destination must be a directory")
    prepared = destination / "prepared"
    _require(not _exists(prepared) or (prepared.is_dir() and not prepared.is_symlink()),
             "Preparation destination prepared/ must be a real directory")
    for _, path, expected in files:
        if _exists(path):
            _verify_existing(path, expected)
    destination.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(exist_ok=True)
    counts = {"hardlink": 0, "copy": 0, "existing": 0}
    for source_path, destination_path, expected in files:
        counts[_install(source_path, destination_path, expected)] += 1
    return {"action": "reused_verified_preparation", "installed_files":
            counts["hardlink"] + counts["copy"], **counts,
            "source": str(source), "destination": str(destination)}
