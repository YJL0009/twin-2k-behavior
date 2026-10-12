#!/usr/bin/env python3
"""Generate and score full-B validation predictions using the retained epoch-1 LoRA.

Save this file in /root/autodl-tmp/twin_qwen, stop the training job, then run:
  /root/autodl-tmp/venvs/liquid_tf52/bin/python infer_qwen_full_b_epoch1.py

Uses the existing native engine directory and already frozen prepared inputs.
Does not require/create training_complete.json, modify training, select another
epoch, or evaluate the reserved test. Outputs/resume: RUN/validation_epoch1/.
This is epoch 1 of the existing three-epoch learning-rate schedule.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import gc
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
PROTOCOL = "qwen-full-b-fixed-epoch1-validation-v1"
DEFAULT_ROOT = Path("/root/autodl-tmp/twin_qwen")
DEFAULT_RUN = Path("outputs/qwen_full_b_lora_v4_b16_a2")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def import_engines(directory):
    directory = Path(directory).resolve()
    names = {
        "base": "run_liquid_baseline", "data": "full_b_train_data",
        "inputs": "full_b_data", "schema": "full_b_schema",
        "metrics": "full_b_metrics", "evaluator": "evaluate_qwen_full_b_lora",
        "generation": "run_qwen_full_b_test",
    }
    require(all((directory / (name + ".py")).is_file() for name in names.values()),
            f"Native full-B engine files missing in {directory}; use --scripts-dir if needed")
    sys.path.insert(0, str(directory))
    modules = {key: importlib.import_module(name) for key, name in names.items()}
    require(all(Path(module.__file__).resolve().parent == directory for module in modules.values()),
            "An unrelated engine module was imported; use a clean Python process")
    return SimpleNamespace(directory=directory, **modules)


def verify_named_files(folder, hashes, base):
    require(isinstance(hashes, dict) and bool(hashes), "Missing frozen file hashes")
    folder = Path(folder).resolve()
    for name, expected in hashes.items():
        path = (folder / name).resolve()
        require(folder in path.parents and path.is_file() and base.digest_file(path) == expected,
                f"Frozen source/model file changed: {path}")


def inspect_run(root, run, engine):
    """Read frozen development metadata; no validation/test answer cells parsed."""
    root, run = Path(root).resolve(), Path(run).resolve()
    base, ev = engine.base, engine.evaluator
    training = read_json(run / "run_config.json")
    require(training.get("protocol") == ev.TRAINING_PROTOCOL and training.get("epochs") == 3,
            "Expected the existing three-epoch full-B LoRA recipe; this evaluator uses its epoch 1 only")
    require(Path(training["root"]).resolve() == root, "Training project root changed")
    for field, filename in (("preparation_config_sha256", "preparation_config.json"),
                            ("preparation_summary_sha256", "preparation_summary.json")):
        require(base.digest_file(run / filename) == training[field], "Training preparation provenance changed")
    preparation_config = read_json(run / "preparation_config.json")
    require(Path(preparation_config["root"]).resolve() == root,
            "Prepared inputs belong to a different project")
    for path, expected in preparation_config["inputs_sha256"].items():
        require(Path(path).is_file() and base.digest_file(Path(path)) == expected,
                f"Frozen source data or split changed: {path}")
    verify_named_files(engine.directory, preparation_config["scripts_sha256"], base)
    verify_named_files(engine.directory, training["scripts_sha256"], base)
    model_path = Path(training["model"]).resolve()
    require(Path(preparation_config["qwen_model"]).resolve() == model_path,
            "Prepared Qwen model differs from training")
    verify_named_files(model_path, training["model_files_sha256"], base)
    verify_named_files(model_path, preparation_config["qwen_tokenizer_sha256"], base)
    verify_named_files(Path(training["selection_model"]),
                       preparation_config["selection_tokenizer_sha256"], base)
    require(training.get("base_parameter_dtype") in {"float32", "bfloat16", "float16"}
            and training.get("adapter_parameter_dtype") == "float32",
            "Missing or unsupported recorded training precision")
    require(training["base_parameters_expected"] == ev.EXPECTED_PARAMETERS
            and training["adapter_parameters_expected"] == ev.EXPECTED_ADAPTER_PARAMETERS
            and training["total_training_parameters_expected"] == ev.EXPECTED_TOTAL_PARAMETERS < 500_000_000,
            "Training parameter budget differs from the rank-8 all-projection recipe")
    descriptor = run / "epoch_adapters/epoch-1.json"
    require(descriptor.is_file(), "Epoch 1 is not yet retained; wait for 'Retained epoch-1 LoRA adapter'")
    candidate = read_json(descriptor)
    adapter = (run / candidate["path"]).resolve()
    require(candidate["name"] == "epoch-1" and candidate["epoch"] == 1
            and candidate["artifact_type"] == "lora_adapter"
            and adapter == (run / "epoch_adapters/epoch-1").resolve()
            and run in adapter.parents, "Not the frozen retained epoch-1 adapter")
    epoch_steps = math.ceil(math.ceil(training["training_examples"] / training["micro_batch_size"])
                            / training["gradient_accumulation"])
    require(candidate["global_step"] == training["optimizer_steps_per_epoch"] == epoch_steps,
            "Adapter was not saved at the first complete epoch boundary")
    engine.generation.verify_hashes(adapter, candidate["model_files_sha256"])
    ev.validate_adapter_config(adapter)
    train, validation, manifest, preparation = engine.data.load_prepared(run)
    require(len(train) == preparation["train_examples"] == training["training_examples"]
            and len(validation) == len(manifest) == training["validation_examples"],
            "Training/validation example counts changed")
    del train
    split = engine.inputs.read_split(root)
    require(set(preparation_config["train_pids"]) == split["train"]
            and set(preparation_config["validation_pids"]) == split["validation"],
            "Preparation participant membership changed")
    pids = {str(item["pid"]) for item in manifest}
    require(pids == split["validation"] and not pids & (split["train"] | split["test"]),
            "Prepared inference participants must be exactly the 309 validation participants")
    require(set(validation.column_names) == {"input_ids", "attention_mask"},
            "Inference dataset must contain label-free prefixes only")
    require(len({(str(s["pid"]), s["item_id"]) for s in manifest}) == len(manifest),
            "Duplicate prepared participant/item")
    item_specs = {}
    for index, spec in enumerate(manifest):
        engine.data.validate_prefix(validation[index], spec)
        previous = item_specs.get(spec["item_id"])
        require(previous is None or engine.metrics._schema_signature(previous)
                == engine.metrics._schema_signature(spec), "Inconsistent prepared item schema")
        item_specs[spec["item_id"]] = spec
    trained_qids = set(training["trained_question_ids"])
    require(trained_qids == set(preparation["coverage"]["train"]["qid_examples"]),
            "Training question coverage changed")
    baseline = read_json(run / "training_baselines.json")
    require(set(item_specs) <= set(baseline) and all(baseline[key]["n"] > 0 for key in item_specs),
            "Missing frozen train-B-only baseline")
    return training, candidate, adapter, validation, manifest, preparation, baseline, item_specs, split


def check_versions(training):
    names = ("torch", "transformers", "tokenizers", "datasets", "pyarrow", "pandas", "numpy", "peft")
    versions = {name: importlib.metadata.version(name) for name in names}
    require(versions["transformers"] == "5.2.0", "Use the existing liquid_tf52 environment")
    for name in ("torch", "transformers", "tokenizers", "peft"):
        require(versions[name] == training["versions"][name], f"Training/evaluation {name} versions differ")
    return versions


def gpu_is_free():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    others = [p.strip() for p in result.stdout.splitlines()
              if p.strip().isdigit() and int(p.strip()) != os.getpid()]
    require(not others, "GPU work is still active (PIDs " + ", ".join(others)
            + "). Stop the training job before inference. No process was stopped by this evaluator.")


def make_report(summary, training, candidate, adapter, scores_dir):
    import pandas as pd
    family = pd.read_csv(scores_dir / "metrics_by_family.csv")
    questions = pd.read_csv(scores_dir / "metrics_by_question.csv")
    columns = ["reference", "family", "valid_labels", "prediction_valid_rate_on_valid_labels",
               "question_macro_categorical_accuracy", "question_macro_mode_categorical_accuracy",
               "question_macro_ordinal_nmae_invalid_penalty", "question_macro_median_ordinal_nmae",
               "question_macro_bounded_nmae_invalid_penalty", "question_macro_median_bounded_nmae"]
    numeric = questions[questions.family.isin(["bounded_numeric", "unbounded_numeric"])][
        ["reference", "qid", "numeric_valid_rate", "numeric_mae_valid_only",
         "median_numeric_mae_on_model_valid", "numeric_fallback_mae_all", "median_numeric_mae_all"]]
    return "\n".join([
        "Fixed epoch-1 full-B LoRA validation inference complete",
        f"Adapter: {adapter}", f"Optimizer step at retained epoch: {candidate['global_step']}",
        f"Training examples per epoch: {training['training_examples']:,}",
        "This is epoch 1 of the existing three-epoch learning-rate schedule; no new training occurred.",
        "Epoch 1 was fixed by user request. No checkpoint selection or validation-NLL selection occurred.",
        f"Validation participants: {summary['participants']}; scalar generations: {summary['prompts']}",
        "C is the primary reference; B is secondary; baselines were frozen using training B only.", "",
        "Family metrics (accuracy higher is better; normalized errors lower):",
        family[columns].round(6).to_string(index=False), "",
        "Numeric errors by question (original units; lower is better):",
        numeric.round(6).to_string(index=False), "",
        "All question families were trained. No mixed-unit full-B score is reported.",
        "Human B/C references, matrix exact matches and detailed metrics are in scores/.",
        "No reserved test participant inputs or response cells were used.",
        f"Detailed scores: {scores_dir}",
    ]) + "\n"


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--scripts-dir", type=Path)
    p.add_argument("--batch-size", type=int, default=1, help="Default 1 matches the original validation generation")
    actions = p.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true", help="Verify cached artifacts; no model or answer values loaded")
    actions.add_argument("--score-only", action="store_true", help="Score existing complete epoch-1 predictions without inference")
    return p


def main():
    args = parser().parse_args()
    require(args.batch_size > 0, "Batch size must be positive")
    root = args.root.resolve()
    run = (args.run if args.run.is_absolute() else root / args.run).resolve()
    require(run.is_dir(), f"Training run directory missing: {run}")
    scripts = args.scripts_dir or root / "qwen_full_b_lora_scripts"
    engine = import_engines(scripts)
    with ExitStack() as locks:
        for name in (".pipeline.lock", ".training.lock"):
            handle = locks.enter_context((run / name).open("a+"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Training/pipeline is still running. Stop it before epoch-1 inference.")
        artifacts = inspect_run(root, run, engine)
        training, candidate, adapter, dataset, manifest, preparation, baselines, item_specs, split = artifacts
        versions = check_versions(training)
        print(f"Verified epoch 1: step {candidate['global_step']}; 309 validation participants; "
              f"{len(manifest):,} scalar predictions; batch {args.batch_size}", flush=True)
        if args.check_only:
            print("Check passed. No GPU model, validation answer values or test answer values were loaded.")
            return
        out = run / "validation_epoch1"
        out.mkdir(exist_ok=True)
        response = root / "data/Twin-2K-500/question_catalog_and_human_response_csv"
        b_path, c_path = response / "wave1_3_response.csv", response / "wave4_response.csv"
        base = engine.base
        config = {"protocol": PROTOCOL, "evaluation_split": "validation", "epoch": 1,
            "checkpoint_policy": "fixed retained epoch 1; no selection; original three-epoch LR schedule",
            "training_config_sha256": base.digest_file(run / "run_config.json"),
            "preparation_summary_sha256": base.digest_file(run / "preparation_summary.json"),
            "epoch_descriptor_sha256": base.digest_file(run / "epoch_adapters/epoch-1.json"),
            "candidate": candidate, "model": training["model"],
            "base_parameter_dtype": training["base_parameter_dtype"], "adapter_parameter_dtype": "float32",
            "generation": {"do_sample": False, "batch_size": args.batch_size,
                           "output_caps": "8 for original binary9; 16 for other scalar items"},
            "helper_sha256": base.digest_file(Path(__file__)),
            "evaluator_sources_sha256": {p.name: base.digest_file(p) for p in
                (Path(engine.evaluator.__file__), Path(engine.generation.__file__), Path(engine.metrics.__file__))},
            "labels_sha256": {"B": base.digest_file(b_path), "C": base.digest_file(c_path)},
            "versions": versions, "primary_reference": "C", "secondary_reference": "B", "test_evaluated": False}
        raw_path = out / "raw_predictions.jsonl"
        if not (out / "evaluation_config.json").exists():
            require(not raw_path.exists(), "Untracked epoch-1 predictions exist; do not mix runs")
        engine.evaluator.freeze_json(out / "evaluation_config.json", config)
        records = engine.generation.read_resume(raw_path)
        seen = engine.generation.validate_records(records, manifest)
        if len(seen) < len(manifest):
            require(not args.score_only, "Cannot score incomplete predictions; rerun without --score-only")
            gpu_is_free()
            import torch
            from transformers import AutoTokenizer
            require(torch.cuda.is_available(), "CUDA is unavailable")
            original = AutoTokenizer.from_pretrained(training["model"], local_files_only=True, padding_side="left")
            model, tokenizer = engine.evaluator._load_model(adapter, original, training["model"],
                training["model_files_sha256"], base_parameter_dtype=training["base_parameter_dtype"])
            try:
                records = engine.generation.infer(model, tokenizer, dataset, manifest, raw_path,
                    batch_size=args.batch_size, status_path=out / "status.json")
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
        require(len(engine.generation.validate_records(records, manifest)) == len(manifest),
                "Generation coverage incomplete; scoring refused")
        for name, path in (("B", b_path), ("C", c_path)):
            require(base.digest_file(path) == config["labels_sha256"][name], "Response CSV changed during inference")
        engine.generation.verify_hashes(adapter, candidate["model_files_sha256"])
        engine.generation.verify_retest_metadata(root, split["validation"], out)
        # Only now parse eligible validation B/C cells. C never enters generation.
        labels_b = engine.inputs.read_selected_labels(b_path, split["validation"], list(item_specs))
        labels_c = engine.inputs.read_selected_labels(c_path, split["validation"], list(item_specs))
        summary = engine.metrics.score_full(records, manifest, labels_b, labels_c, baselines, out / "scores",
            trained_qids=set(training["trained_question_ids"]), trained_scope="trained_full_b_lora_epoch1")
        summary.update(evaluation_protocol=PROTOCOL, evaluation_split="validation", fixed_epoch=1,
            adapter_path=str(adapter), optimizer_step=candidate["global_step"],
            learning_rate_schedule="first epoch of the original planned three epochs",
            checkpoint_selection_performed=False, test_evaluated=False,
            evaluation_config_sha256=base.digest_file(out / "evaluation_config.json"),
            raw_predictions_sha256=base.digest_file(raw_path))
        base.write_json(out / "summary.json", summary)
        report = make_report(summary, training, candidate, adapter, out / "scores")
        (out / "report.txt").write_text(report, encoding="utf-8")
        base.write_json(out / "evaluation_complete.json", {"protocol": PROTOCOL, "epoch": 1,
            "examples": len(records), "predictions_sha256": summary["raw_predictions_sha256"],
            "evaluation_config_sha256": summary["evaluation_config_sha256"], "test_evaluated": False})
        base.write_json(out / "status.json", {"state": "complete", "completed": len(records),
            "total": len(manifest), "validation_response_labels_read": True, "test_response_cells_read": False})
        print(report, flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError, importlib.metadata.PackageNotFoundError) as exc:
        raise SystemExit(f"Epoch-1 evaluation stopped: {exc}")
