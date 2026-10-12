#!/usr/bin/env python3
"""Plan and reproduce the Twin-2K-500 development experiments, sequentially.

No training or downloads occur for list/plan/collect. Final-test evaluation is
deliberately absent. The run command executes real preparation/inference/training.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

PACKAGE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def digest_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024), b""):
            sha.update(block)
    return sha.hexdigest()


def external_fingerprint(args, selected):
    """Raw-file and original-weight integrity, without analyzing held-out answers."""
    from download_data import verify_dataset
    dataset = args.dataset_dir or args.workspace / "data/Twin-2K-500"
    result = {}
    if any(s['id']!='qwen_controls' for s in selected):
        result["dataset"] = verify_dataset(dataset)
    if args.split_manifest:
        result["supplied_split_sha256"] = digest_file(args.split_manifest)
    for key, model in (("qwen_model",args.qwen_model),("liquid_model",args.liquid_model)):
        if model.is_dir():
            result[key] = {p.name:digest_file(p) for p in sorted(model.iterdir())
                if p.is_file() and p.suffix in {".json",".safetensors",".bin",".model",".jinja",".txt"}}
    return result


def stage_inputs(stage, inputs):
    keys = set()
    stage_id = stage['id']
    if stage_id!='qwen_controls': keys.update(('dataset','supplied_split_sha256'))
    if stage['gpu']:
        if stage_id.startswith('qwen') or stage_id.startswith('full_b') or stage_id=='batch32_probe':
            keys.add('qwen_model')
        if stage_id.startswith('liquid') or stage_id.startswith('full_b') or stage_id=='batch32_probe' or stage_id in ('qwen_full_train','qwen_matched_baseline','qwen_full_eval') or stage_id.startswith('qwen_cv_fold'):
            keys.add('liquid_model')
    return {key:inputs[key] for key in sorted(keys) if key in inputs}


def load_config():
    config = read_json(PACKAGE / "configs/experiments.json")
    ids = [s["id"] for s in config["stages"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate stage IDs")
    known = set()
    for stage in config["stages"]:
        if not set(stage["depends_on"]) <= known:
            raise ValueError(f"Stages must be in dependency order: {stage['id']}")
        known.add(stage["id"])
    for alias, stages in config["groups"].items():
        if not set(stages) <= known:
            raise ValueError(f"Unknown stage in group {alias}")
    return config


def resolve_selection(config, experiments, stages=None, include_dependencies=True):
    by_id = {s["id"]: s for s in config["stages"]}
    wanted = set(stages or [])
    for name in experiments or []:
        if name not in config["groups"]:
            raise ValueError(f"Unknown experiment group {name!r}; use list")
        wanted.update(config["groups"][name])
    if not wanted:
        wanted.update(config["groups"]["all"])
    if not wanted <= by_id.keys():
        raise ValueError(f"Unknown stages: {sorted(wanted - by_id.keys())}")
    def add(stage_id):
        for dependency in by_id[stage_id]["depends_on"]:
            if dependency not in wanted:
                wanted.add(dependency)
                add(dependency)
    if include_dependencies:
        for stage_id in list(wanted):
            add(stage_id)
    return [s for s in config["stages"] if s["id"] in wanted]


def variables(args):
    root = args.workspace.expanduser().resolve()
    return {
        "python": str(args.python.expanduser().resolve()),
        "root": str(root), "package": str(PACKAGE),
        "legacy": str(PACKAGE / "engines/legacy"),
        "full_b": str(PACKAGE / "engines/qwen_full_b_lora_scripts"),
        "qwen_model": str(args.qwen_model.expanduser().resolve()),
        "liquid_model": str(args.liquid_model.expanduser().resolve()),
        "profile_out": str(root / "outputs/qwen_profile_diagnostic_reproduction"),
    }


def command_for(stage, values, args):
    command = [arg.format_map(values) for arg in stage["command"]]
    if stage["id"] == "prepare_data":
        if args.dataset_dir:
            command += ["--dataset-dir", str(args.dataset_dir.expanduser().resolve())]
        if args.split_manifest:
            command += ["--split-manifest", str(args.split_manifest.expanduser().resolve())]
    return command


def source_fingerprint():
    """Fingerprint all release sources, excluding mutable runs/data/results."""
    hashes = {}
    for path in sorted(PACKAGE.rglob("*")):
        rel = path.relative_to(PACKAGE)
        if any(p in {"runs", "data", "outputs", "logs", "__pycache__", ".git"} for p in rel.parts):
            continue
        if path.is_file() and path.suffix in {".py", ".json", ".sh", ".ipynb", ".txt", ".md"}:
            hashes[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest(hashes)


def check_runtime(args, selected):
    """Check the chosen Python without importing models or touching labels."""
    script = '''import importlib.metadata as m, json, sys
packages = ["torch", "transformers", "datasets", "accelerate", "peft", "pandas", "pyarrow", "numpy"]
versions = {}
for p in packages:
    try: versions[p] = m.version(p)
    except m.PackageNotFoundError: versions[p] = None
print(json.dumps({"python":sys.version,"executable":sys.executable,"packages":versions}))
'''
    result = subprocess.run([str(args.python), "-c", script], check=True, capture_output=True, text=True)
    runtime = json.loads(result.stdout.strip())
    if any(s["gpu"] for s in selected):
        needed = ["torch", "transformers", "datasets", "accelerate", "peft", "pandas", "pyarrow"]
        missing = [p for p in needed if not runtime["packages"][p]]
        if missing:
            raise ValueError("Missing packages: " + ", ".join(missing) + "; see README.md")
        if any(s["id"].startswith("full_b_") for s in selected):
            for p, expected in {"transformers":"5.2.0", "accelerate":"1.12.0", "peft":"0.18.0"}.items():
                if runtime["packages"][p] != expected:
                    raise ValueError(f"Full-B requires {p}=={expected}; got {runtime['packages'][p]}")
        check = subprocess.run([str(args.python), "-c",
            "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'"], capture_output=True, text=True)
        if check.returncode:
            raise ValueError(check.stderr.strip())
        required_models = []
        if any("{qwen_model}" in arg for stage in selected for arg in stage["command"]):
            required_models.append(args.qwen_model)
        if any("{liquid_model}" in arg for stage in selected for arg in stage["command"]):
            required_models.append(args.liquid_model)
        for model in required_models:
            if not (model / "config.json").is_file():
                raise ValueError(f"Original model files missing at {model}; see README.md")
    return runtime


def gpu_is_free():
    """Do not compete with the user's current job. No process is killed."""
    check = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    if check.returncode:
        raise ValueError("Cannot inspect GPU processes: " + check.stderr.strip())
    pids = [v.strip() for v in check.stdout.splitlines() if v.strip().isdigit()]
    if pids:
        raise ValueError("GPU work is already running (PIDs " + ", ".join(pids) +
                         "). Wait for it to finish, then rerun this command; no job was stopped.")


def run_selected(config, args, selected):
    values = variables(args)
    root = Path(values["root"])
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock_path = root / "logs/experiment_queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another experiment queue holds this workspace lock")
        runtime = check_runtime(args, selected)
        fingerprint = source_fingerprint()
        inputs = external_fingerprint(args, selected)
        plan = [{"id":s["id"], "command":command_for(s, values, args)} for s in selected]
        write_json(root / "logs/last_launch_plan.json", {
            "created_utc":timestamp(), "sources_sha256":fingerprint, "stages":plan,
            "runtime":runtime, "external_inputs_sha256":digest(inputs), "test_evaluated":False})
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        env["PYTHONPATH"] = str(PACKAGE / "engines/legacy") + os.pathsep + env.get("PYTHONPATH", "")
        for stage in selected:
            command = command_for(stage, values, args)
            identity = digest({"command":command, "sources":fingerprint, "runtime":runtime,
                               "inputs":stage_inputs(stage, inputs)})
            marker = root / "logs/stages" / (stage["id"] + ".json")
            if marker.exists():
                previous = read_json(marker)
                if previous["identity"] != identity:
                    raise ValueError(f"{stage['id']}: source/arguments changed; choose a new --workspace")
                if previous["state"] == "complete":
                    artifacts = previous.get("artifacts", [])
                    hashes = previous.get("artifact_sha256", {})
                    if artifacts and hashes and all(Path(p).is_file() and digest_file(p)==hashes.get(p) for p in artifacts):
                        print(f"SKIP complete: {stage['id']}", flush=True)
                        continue
                    raise ValueError(f"{stage['id']}: completed outputs are missing or changed; inspect workspace")
            if stage["gpu"]:
                gpu_is_free()
            logfile = root / "logs" / (stage["id"] + ".log")
            state = {"stage":stage["id"], "identity":identity, "command":command,
                     "state":"running", "started_utc":timestamp(), "log":str(logfile)}
            write_json(marker, state)
            print(f"RUN {stage['id']}: {shlex.join(command)}", flush=True)
            try:
                with logfile.open("a", encoding="utf-8") as log:
                    log.write(f"\n[{timestamp()}] {shlex.join(command)}\n"); log.flush()
                    proc = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        code = proc.wait()
                    except KeyboardInterrupt:
                        proc.send_signal(__import__('signal').SIGINT)
                        proc.wait()
                        raise
                if code:
                    raise subprocess.CalledProcessError(code, command)
                artifacts = []
                for pattern in stage["artifacts"]:
                    paths = list(root.glob(pattern.format_map(values)))
                    if not paths:
                        raise ValueError(f"Stage returned successfully but required output is missing: {pattern}")
                    artifacts += [str(p) for p in paths if p.is_file()]
                state.update(state="complete", finished_utc=timestamp(), artifacts=artifacts,
                             artifact_sha256={p:digest_file(p) for p in artifacts})
                write_json(marker, state)
                print(f"DONE {stage['id']} · {logfile}", flush=True)
            except BaseException as exc:
                state.update(state="failed", finished_utc=timestamp(), error=str(exc))
                write_json(marker, state)
                raise
        collect(root, config)


def collect(root, config):
    """Collect actual completed output summaries; never substitute observed scores."""
    rows = []
    for item in config["result_sources"]:
        path = root / item["path"]
        row = {"experiment":item["experiment"], "scope":item["scope"], "status":"not_run",
               "macro_accuracy":"", "pooled_accuracy":"", "invalid_outputs":"",
               "majority_macro_accuracy":"", "gain_over_majority_pp":"", "artifact":str(path)}
        if path.exists():
            data = read_json(path)
            for key in item.get("nested", []):
                data = data[key]
            row["status"] = "completed"
            for output, field in item["fields"].items():
                row[output] = data.get(field, "")
            if isinstance(row["macro_accuracy"], (int,float)) and isinstance(row["majority_macro_accuracy"], (int,float)):
                row["gain_over_majority_pp"] = 100 * (row["macro_accuracy"] - row["majority_macro_accuracy"])
        rows.append(row)
    output = root / "reports/experiment_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(root / "reports/experiment_summary.json", {"created_utc":timestamp(), "rows":rows,
        "test_evaluated_by_this_runner":False,
        "note":"Actual output files only. Full-B has heterogeneous family metrics; inspect its validation report."})
    print("Collected actual results:", output)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("list", "plan", "run", "collect"))
    p.add_argument("--workspace", type=Path, default=PACKAGE / "runs/twin_experiments")
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--qwen-model", type=Path, default=Path("/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct"))
    p.add_argument("--liquid-model", type=Path, default=Path("/root/autodl-tmp/models/LFM2.5-350M"))
    p.add_argument("--dataset-dir", type=Path)
    p.add_argument("--split-manifest", type=Path)
    p.add_argument("--experiments", nargs="+", help="Experiment groups from list; default all")
    p.add_argument("--stages", nargs="+", help="Exact stage IDs instead of groups")
    p.add_argument("--no-deps", action="store_true", help="Use only explicit stages; caller supplies their prerequisites")
    return p


def main():
    args = parser().parse_args()
    config = load_config()
    if args.action == "list":
        print("Experiment groups:")
        for name, stages in config["groups"].items(): print(f"  {name}: {', '.join(stages)}")
        print("\nStages:")
        for s in config["stages"]: print(f"  {s['id']}: {'GPU' if s['gpu'] else 'CPU'} · {s['description']}")
        return
    if args.action == "collect":
        collect(args.workspace.resolve(), config); return
    selected = resolve_selection(config, args.experiments, args.stages, not args.no_deps)
    if args.action == "plan":
        values = variables(args)
        for i, s in enumerate(selected, 1):
            print(f"{i:02d}. {s['id']} [{'GPU' if s['gpu'] else 'CPU'}]")
            print("    " + shlex.join(command_for(s, values, args)))
        print("\nPlan only: no data, outputs or training modified. Final test is excluded.")
        return
    run_selected(config, args, selected)


if __name__ == "__main__":
    try: main()
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
