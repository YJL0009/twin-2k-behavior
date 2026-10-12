#!/usr/bin/env python3
"""Finish Qwen full_v1, run three fresh full-FT participant folds, write report.

Launch via start_qwen_overnight.sh to survive terminal disconnects. Existing
training processes are waited on, never terminated. No final-test evaluation.
"""
import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PARAMETERS = 494_032_768
SCRIPT_DIR = Path(__file__).resolve().parent
CURRENT_SCRIPTS = ("run_qwen_full.sh", "train_qwen_full.py", "evaluate_qwen_full.py",
                   "train_liquid_full.py", "evaluate_liquid_full.py", "run_liquid_full.sh",
                   "train_qwen_lora.py", "train_liquid_lora.py", "evaluate_liquid_lora.py",
                   "qwen_cv_fold.py")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_bytes(folder):
    if not folder.exists():
        return 0
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and not p.is_symlink())


def check_disk(root, source_run, cv_root):
    # Three retained current checkpoints, one retained checkpoint per CV fold,
    # plus one temporary checkpoint during rotation and prepared data/headroom.
    expected = 7 * PARAMETERS * 12 + 3_000_000_000
    existing = checkpoint_bytes(source_run / "checkpoints")
    existing += sum(checkpoint_bytes(cv_root / f"fold_{fold}" / "checkpoints") for fold in (1, 2, 3))
    needed = max(2_000_000_000, expected - existing)
    free = shutil.disk_usage(root).free
    print(f"Disk check: {free / 2**30:.1f} GiB free; approximately {needed / 2**30:.1f} GiB "
          "needed for remaining checkpoints and preparation.", flush=True)
    require(free >= needed,
            f"Insufficient disk space for the full overnight queue: need about {needed / 2**30:.1f} GiB "
            f"free, have {free / 2**30:.1f} GiB. Free or expand storage, then rerun. "
            "No existing experiment files have been removed.")


def ignored_pids():
    pids, pid = set(), os.getpid()
    while pid > 0 and pid not in pids:
        pids.add(pid)
        try:
            # stat comm can contain spaces and parentheses; fields after last ')'.
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def external_jobs(proc_root=Path("/proc")):
    ignored = ignored_pids()
    jobs = []
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name) in ignored:
            continue
        try:
            argv = [s.decode(errors="replace") for s in (process / "cmdline").read_bytes().split(b"\0") if s]
        except OSError:
            continue
        matches = [Path(arg).name for arg in argv if Path(arg).name in CURRENT_SCRIPTS]
        if matches:
            jobs.append({"pid": int(process.name), "script": matches[0]})
    # Also wait for other GPU work; a parent shell remains visible across gaps
    # between original launcher stages via the named-process check above.
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    require(query.returncode == 0, "Cannot query GPU processes: " + query.stderr.strip())
    known = {job["pid"] for job in jobs}
    for line in query.stdout.splitlines():
        value = line.strip()
        if value.isdigit() and int(value) not in ignored | known:
            jobs.append({"pid": int(value), "script": "other GPU process"})
            known.add(int(value))
    return sorted(jobs, key=lambda job: job["pid"])


def wait_for_existing(status_path):
    while True:
        jobs = external_jobs()
        if not jobs:
            return
        print("Waiting for existing work:", jobs, flush=True)
        write_json(status_path, {"state": "waiting", "updated_utc": utc_now(),
                                 "pid": os.getpid(), "existing_jobs": jobs})
        time.sleep(30)


def run_step(name, command, status_path, history):
    print(f"\n[{utc_now()}] {name}", flush=True)
    write_json(status_path, {"state": "running", "stage": name, "updated_utc": utc_now(),
                             "pid": os.getpid(), "completed_stages_this_launch": history})
    subprocess.run(command, check=True)
    history.append(name)


def write_reports(root, source_run, cv_root):
    import pandas as pd
    import qwen_cv_data as data
    config, _, manifest, folds = data.load_cv(cv_root)
    current = read_json(source_run / "best_checkpoint.json")
    summaries, frames = [], []
    label_source_hashes = None
    for fold in (1, 2, 3):
        folder = cv_root / f"fold_{fold}"
        summary = read_json(folder / "validation/summary.json")
        if label_source_hashes is None:
            label_source_hashes = summary["label_source_files_sha256"]
        require(summary["label_source_files_sha256"] == label_source_hashes,
                "Response files changed between CV folds; results cannot be combined")
        fold_config = read_json(folder / "fold_config.json")
        scores = pd.read_csv(folder / "validation/scored_predictions.csv",
                             dtype={"pid": "string", "qid": "string"})
        scores["fold"] = fold
        require(set(scores.pid).issubset(set(fold_config["val_pids"])), "Unexpected scored participant")
        frames.append(scores)
        summaries.append({"fold": fold, "train_participants": len(fold_config["train_pids"]),
            "validation_participants": len(fold_config["val_pids"]),
            "examples": summary["examples"], "question_macro_accuracy": summary["question_macro_accuracy"],
            "pooled_accuracy": summary["pooled_accuracy"],
            "majority_macro_accuracy": summary["majority_macro_accuracy"],
            "gain_over_majority_pp": 100 * (summary["question_macro_accuracy"] - summary["majority_macro_accuracy"]),
            "invalid_outputs": summary["invalid_outputs"]})
    table = pd.DataFrame(summaries)
    oof = pd.concat(frames, ignore_index=True)
    require(not oof[["pid", "qid"]].duplicated().any(), "Duplicate out-of-fold examples")
    require("574" not in set(oof.pid), "Participant 574 must remain training-only")
    split = pd.read_csv(root / "data/participant_split.csv", dtype={"pid": "string"})
    test_pids = set(split.loc[split["split"].eq("test"), "pid"])
    require(not set(oof.pid) & test_pids, "Final test participant entered CV report")
    # Scored files retain all prompts; exclude missing C from accuracy explicitly.
    required_pids = set(folds.loc[folds.validation_fold.ne(-1), "pid"].astype(str))
    require(set(oof.pid) == required_pids, "Unexpected or missing out-of-fold participant")
    expected_keys = set(manifest.loc[manifest.pid.isin(required_pids), ["pid", "qid"]]
                        .itertuples(index=False, name=None))
    require(set(oof[["pid", "qid"]].itertuples(index=False, name=None)) == expected_keys,
            "Out-of-fold question coverage changed")
    has_c = oof.target_c.notna()
    valid = oof.prediction.isin([1, 2])
    correct = (valid & oof.prediction.eq(oof.target_c)).astype(float).where(has_c)
    majority_correct = oof.majority_prediction.eq(oof.target_c).astype(float).where(has_c)
    per_question = pd.DataFrame({"qid": oof.qid, "correct": correct, "majority_correct": majority_correct})
    per_question = per_question.groupby("qid").agg(examples=("correct", "count"),
        model_accuracy=("correct", "mean"), majority_accuracy=("majority_correct", "mean"))
    per_question["gain_over_majority_pp"] = 100 * (per_question.model_accuracy - per_question.majority_accuracy)
    report = {"protocol": "fixed-three-epoch-participant-development-cross-validation",
        "cv_epochs": 3, "folds": summaries, "training_only_anchor_pid": "574",
        "oof_participants_expected": len(required_pids), "oof_participants_scored": int(oof.loc[has_c, "pid"].nunique()),
        "oof_prompts": len(oof), "oof_examples_scored": int(has_c.sum()),
        "oof_excluded_missing_c": int((~has_c).sum()),
        "fold_macro_accuracy_mean": float(table.question_macro_accuracy.mean()),
        "fold_macro_accuracy_sample_std": float(table.question_macro_accuracy.std(ddof=1)),
        "fold_gain_over_majority_pp_mean": float(table.gain_over_majority_pp.mean()),
        "fold_gain_over_majority_pp_sample_std": float(table.gain_over_majority_pp.std(ddof=1)),
        "oof_question_macro_accuracy": float(per_question.model_accuracy.mean()),
        "oof_majority_macro_accuracy": float(per_question.majority_accuracy.mean()),
        "oof_pooled_accuracy": float(correct.mean()), "oof_invalid_outputs": int((~valid).sum()),
        "current_fixed_split_best_checkpoint": current["best"], "test_evaluated": False,
        "interpretation": "Development robustness check after prior validation-guided experimentation; "
            "folds use 3 fixed epochs, unlike the current run's best-of-3 checkpoint selection. "
            "Fold standard deviation is descriptive, not a confidence interval."}
    table.to_csv(cv_root / "fold_comparison.csv", index=False)
    oof.to_csv(cv_root / "oof_predictions.csv", index=False)
    per_question.to_csv(cv_root / "oof_metrics.csv")
    write_json(cv_root / "cv_summary.json", report)
    lines = ["Qwen overnight results", "", "Current fixed train/validation split:",
        f"Selected checkpoint: {current['best']['checkpoint']}",
        f"Validation question-macro accuracy: {current['best']['question_macro_accuracy']:.2%}", "",
        "Development CV: 3 fresh models, 3 fixed epochs each; PID 574 always training.",
        table.to_string(index=False), "",
        f"Fold macro accuracy: {report['fold_macro_accuracy_mean']:.2%} +/- "
        f"{100 * report['fold_macro_accuracy_sample_std']:.2f} percentage points (sample SD)",
        f"Mean gain over fold-specific majority: {report['fold_gain_over_majority_pp_mean']:+.2f} pp",
        f"Out-of-fold macro accuracy: {report['oof_question_macro_accuracy']:.2%}",
        f"Out-of-fold majority macro accuracy: {report['oof_majority_macro_accuracy']:.2%}",
        f"Scored OOF participants: {report['oof_participants_scored']}/{len(required_pids)}", "",
        report["interpretation"], "The final 309-participant test set was not evaluated."]
    (cv_root / "overnight_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--source-run", type=Path, default=Path("outputs/qwen_full_v1"))
    parser.add_argument("--cv-out", type=Path, default=Path("outputs/qwen_cv_v1"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    source_run = (args.source_run if args.source_run.is_absolute() else root / args.source_run).resolve()
    cv_root = (args.cv_out if args.cv_out.is_absolute() else root / args.cv_out).resolve()
    require(source_run != cv_root and source_run not in cv_root.parents
            and cv_root not in source_run.parents, "Current and CV output directories must be separate")
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    status_path = logs / "qwen_overnight_status.json"
    lock = (logs / "qwen_overnight.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another overnight launcher is already running. Check logs/qwen_overnight_status.json.")
    history = []
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if args.report_only:
            write_reports(root, source_run, cv_root)
            return
        required = [root / "data/participant_split.csv",
            root / "data/Twin-2K-500/question_catalog_and_human_response_csv/wave4_response.csv",
            root / "outputs/liquid_lora_v1/training_manifest.csv",
            Path("/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct/model.safetensors")]
        for path in required:
            require(path.is_file(), f"Required input missing: {path}")
        import importlib.metadata
        require(importlib.metadata.version("transformers") == "5.2.0",
                "Use /root/autodl-tmp/venvs/liquid_tf52/bin/python")
        for package in ("torch", "datasets", "accelerate", "pandas", "pyarrow"):
            print(f"{package}: {importlib.metadata.version(package)}", flush=True)
        require(shutil.which("nvidia-smi") is not None, "nvidia-smi is unavailable")
        check_disk(root, source_run, cv_root)
        if args.check_only:
            print("Preflight passed. No training launched.", flush=True)
            return
        wait_for_existing(status_path)
        check_disk(root, source_run, cv_root)
        # Preserve every byte of the previous package's scripts and configuration.
        current = SCRIPT_DIR
        env = os.environ.copy()
        env.update(QWEN_PROJECT_ROOT=str(root), QWEN_PYTHON=sys.executable, QWEN_RUN_DIR=str(source_run))
        print(f"\n[{utc_now()}] Finish current Qwen full_v1 experiment", flush=True)
        write_json(status_path, {"state": "running", "stage": "current Qwen experiment",
                                 "updated_utc": utc_now(), "pid": os.getpid()})
        subprocess.run(["bash", str(current / "run_qwen_full.sh")], check=True, env=env)
        history.append("current Qwen experiment")
        run_step("Prepare three participant folds",
            [sys.executable, "-u", str(SCRIPT_DIR / "qwen_cv_data.py"), "--root", str(root),
             "--source-run", str(source_run), "--out", str(cv_root)], status_path, history)
        for fold in (1, 2, 3):
            run_step(f"CV fold {fold}: fresh full fine-tuning and final-epoch evaluation",
                [sys.executable, "-u", str(SCRIPT_DIR / "qwen_cv_fold.py"),
                 "--cv-root", str(cv_root), "--fold", str(fold)], status_path, history)
        write_reports(root, source_run, cv_root)
        write_json(status_path, {"state": "complete", "updated_utc": utc_now(), "pid": os.getpid(),
            "completed_stages_this_launch": history, "report": str(cv_root / "overnight_report.txt"),
            "test_evaluated": False})
        print("\nALL STAGES COMPLETE. Report:", cv_root / "overnight_report.txt", flush=True)
    except BaseException as exc:
        write_json(status_path, {"state": "failed", "updated_utc": utc_now(), "pid": os.getpid(),
            "completed_stages_this_launch": history, "error": str(exc), "exception": type(exc).__name__})
        traceback.print_exc()
        print("Stopped with saved progress. Correct the reported issue and rerun the same launcher.", flush=True)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
