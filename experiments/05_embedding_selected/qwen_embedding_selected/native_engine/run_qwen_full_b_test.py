#!/usr/bin/env python3
"""Frozen Qwen evaluation on all B target items for the 309 test participants.

Expands matrix rows and slider statements into scalar prompts. Train-B-only
baselines are fixed before inference; test B/C answers are opened only after
all saved model predictions are complete. Never fine-tunes or selects on test.
"""
import argparse
import fcntl
import gc
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import run_liquid_baseline as base

PROTOCOL = "qwen-frozen-full-b-test-scalar-items-v1"
EXPECTED_PARAMETERS = 494_032_768
TRAINED_QIDS = set(base.QIDS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_hashes(folder, hashes):
    folder = Path(folder).resolve()
    require(hashes, "Missing checkpoint/tokenizer fingerprints")
    for name, digest in hashes.items():
        path = (folder / name).resolve()
        require(folder in path.parents and path.is_file() and base.digest_file(path) == digest,
                f"Frozen model/tokenizer artifact changed: {path}")


def load_frozen_run(root, run, selection_model, verify_weights=True):
    selection = read_json(run / "best_checkpoint.json")
    training = read_json(run / "run_config.json")
    completion = read_json(run / "training_complete.json")
    require(training["protocol"] == "qwen-nine-binary-a-to-b-full-fixed-content-v1"
            and selection["selection_split"] == "validation"
            and selection["selection_labels"] == "C"
            and selection["test_evaluated"] is False, "Expected the existing validation-selected Qwen run")
    require(completion["config_hash"] == base.digest_text(base.dumps(training))
            and selection["training_complete_sha256"] == base.digest_file(run / "training_complete.json"),
            "Qwen training/selection provenance mismatch")
    require(completion["base_parameters"] == completion["trainable_parameters"]
            == completion["total_training_parameters"] == EXPECTED_PARAMETERS,
            "Expected the completed 494M full-parameter Qwen run")
    best = selection["best"]
    candidates = [c for c in completion["candidate_checkpoints"] if c["name"] == best["checkpoint"]]
    require(len(candidates) == 1, "Selected checkpoint is not a completed candidate")
    candidate = candidates[0]
    checkpoint = (run / candidate["path"]).resolve()
    require(run in checkpoint.parents and checkpoint.name == best["checkpoint"], "Invalid checkpoint path")
    require(Path(best["model_path"]).resolve() == checkpoint, "Selected model path changed")
    original = Path(training["model"]).resolve()
    require(not (checkpoint / "adapter_config.json").exists(), "Expected a complete model, not a LoRA adapter")
    require(training["system"] == base.SYSTEM and training["source_liquid_selection_budget"] == 8192
            and set(training["qids"]) == TRAINED_QIDS, "Original prompt/training subset changed")
    split_path = root / "data/participant_split.csv"
    require(training["inputs_sha256"][str(split_path)] == base.digest_file(split_path), "Participant split changed")
    source_liquid = read_json(Path(training["source_run"]) / "run_config.json")
    require(Path(source_liquid["model"]).resolve() == selection_model.resolve(),
            "Use the original Liquid tokenizer to preserve source A selection")
    if verify_weights:
        verify_hashes(checkpoint, candidate["model_files_sha256"])
        # Only tokenizer files are needed from the original model for equality.
        original_hashes = {k: v for k, v in training["model_files_sha256"].items()
                           if not (k.endswith(".safetensors") or k == "pytorch_model.bin")}
        verify_hashes(original, original_hashes)
        selection_hashes = {k: v for k, v in source_liquid["model_files_sha256"].items()
                            if not (k.endswith(".safetensors") or k == "pytorch_model.bin")}
        verify_hashes(selection_model, selection_hashes)
    versions = {key: importlib.metadata.version(key) for key in
                ("torch", "transformers", "tokenizers", "datasets", "pyarrow", "pandas", "numpy")}
    require(versions["transformers"] == "5.2.0", "Use the existing liquid_tf52 Python environment")
    for key in ("torch", "transformers", "tokenizers"):
        require(versions[key] == training["versions"][key], f"Training/evaluation {key} versions differ")
    return training, selection, candidate, checkpoint, original, versions


def source_files(root, run):
    response = root / "data/Twin-2K-500/question_catalog_and_human_response_csv"
    files = [root / "data/participant_split.csv", run / "run_config.json", run / "best_checkpoint.json",
             run / "training_complete.json", response / "wave1_3_response.csv", response / "wave4_response.csv"]
    parquets = sorted((root / "data/Twin-2K-500/wave_split/chunks").glob("*.parquet"))
    require(len(parquets) == 7, "Expected seven wave_split dataset chunks")
    files += parquets
    require(all(p.is_file() for p in files), "Missing split, completed model metadata, response CSV or parquet files")
    return files


def preflight(args):
    import full_b_data as data
    import full_b_schema as schema
    import full_b_metrics
    split = data.read_split(args.root)
    require(len(split["train"]) == 1440 and len(split["validation"]) == 309 and len(split["test"]) == 309,
            "Unexpected train/validation/test participant counts")
    source_files(args.root, args.run)
    load_frozen_run(args.root, args.run, args.selection_model, verify_weights=False)
    catalog_path = Path(__file__).with_name("full_b_question_catalog.json")
    catalog = read_json(catalog_path)
    require(len(schema.catalog_items(catalog)) == 126, "Bundled full-B catalog schema changed")
    require(shutil.disk_usage(args.root).free >= 6 * 2**30,
            "Need at least 6 GiB free for tokenized inputs, manifests and prediction records")
    import torch
    require(torch.cuda.is_available(), "CUDA is unavailable")
    print("Preflight passed: 309 test participants; frozen Qwen checkpoint; full-B schema supported.", flush=True)


def read_resume(path):
    if not path.exists():
        return []
    contents = path.read_bytes()
    records, consumed = [], 0
    lines = contents.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            require(index == len(lines) - 1 and not line.endswith(b"\n"), "Corrupt saved prediction record")
            backup = path.with_name(path.name + ".incomplete_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
            backup.write_bytes(line)
            with path.open("r+b") as handle:
                handle.truncate(consumed)
            print("Recovered interrupted final line; preserved it at", backup, flush=True)
            break
        records.append(record)
        consumed += len(line)
    if records and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return records


def validate_records(records, manifest):
    from full_b_schema import parse_prediction
    specs = {(str(p["pid"]), p["item_id"]): p for p in manifest}
    require(len(specs) == len(manifest), "Duplicate prepared target item")
    seen = set()
    for row in records:
        key = (str(row["pid"]), row["item_id"])
        require(key in specs and key not in seen, "Duplicate or unexpected prediction")
        spec = specs[key]
        require(row["qid"] == spec["qid"] and row["prompt_hash"] == spec["prompt_hash"]
                and row["input_tokens"] == spec["input_tokens"], "Saved prediction prompt changed")
        require(row["prediction"] == parse_prediction(row["raw_output"], spec), "Saved prediction parser changed")
        require(row["max_new_tokens"] == (8 if spec["qid"] in TRAINED_QIDS else 16), "Saved output budget changed")
        seen.add(key)
    return seen


def verify_retest_metadata(root, test_pids, out):
    """After prediction, ensure C describes the same target stimuli as B."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    paths = sorted((root / "data/Twin-2K-500/wave_split/chunks").glob("*.parquet"))
    source = ds.dataset([str(path) for path in paths], format="parquet")
    scanner = source.scanner(columns=["pid", "wave4_Q_wave1_3_A", "wave4_Q_wave4_A"],
        filter=ds.field("pid").cast(pa.string()).isin(sorted(test_pids)), batch_size=1)
    seen, mismatches = set(), []
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            pid = str(row["pid"])
            require(pid in test_pids and pid not in seen, "Retest metadata participant mismatch")
            seen.add(pid)
            b = dict(base.survey_units(row["wave4_Q_wave1_3_A"], include_answers=False))
            c = dict(base.survey_units(row["wave4_Q_wave4_A"], include_answers=False))
            for qid in sorted(set(b) | set(c)):
                if b.get(qid) != c.get(qid):
                    mismatches.append({"pid": pid, "qid": qid})
    require(seen == test_pids, "Retest metadata coverage incomplete")
    write_json(out / "retest_metadata_check.json", {"participants": len(seen), "mismatches": mismatches,
        "policy": "B and C target text, options, rows/statements and display instructions must match."})
    require(not mismatches, "B/C question metadata differ; C scoring stopped. See retest_metadata_check.json. "
            "Saved model predictions remain intact.")


def make_report(summary, score_dir, checkpoint, n_items):
    import pandas as pd
    scope = pd.read_csv(score_dir / "metrics_by_training_scope.csv")
    family = pd.read_csv(score_dir / "metrics_by_family.csv")
    questions = pd.read_csv(score_dir / "metrics_by_question.csv")
    columns = ["reference", "training_scope", "family", "valid_labels",
        "prediction_valid_rate_on_valid_labels", "question_macro_categorical_accuracy",
        "question_macro_mode_categorical_accuracy"]
    categories = scope.loc[scope.family.isin(["mc", "ordinal", "matrix_nominal", "matrix_ordinal"]), columns]
    errors = family[["reference", "family", "valid_labels", "prediction_valid_rate_on_valid_labels",
        "question_macro_ordinal_nmae_invalid_penalty", "question_macro_median_ordinal_nmae",
        "question_macro_bounded_nmae_invalid_penalty", "question_macro_median_bounded_nmae"]]
    errors = errors[errors.family.isin(["ordinal", "matrix_ordinal", "bounded_numeric"])]
    numeric = questions.loc[questions.family.isin(["bounded_numeric", "unbounded_numeric"]),
        ["reference", "qid", "family", "valid_labels", "numeric_valid_rate", "numeric_mae_valid_only",
         "median_numeric_mae_on_model_valid", "numeric_fallback_mae_all", "median_numeric_mae_all"]]
    return "\n".join(["Qwen full-B test evaluation complete", f"Checkpoint: {checkpoint}",
        f"Participants: 309; scalar generations: {n_items}", "",
        "Categorical accuracy (question macro; higher is better; invalid answers are wrong):",
        categories.round(6).to_string(index=False), "",
        "Ordinal/bounded normalized absolute error (lower is better; invalid predictions receive error 1):",
        errors.round(6).to_string(index=False), "",
        "Numeric errors by question (original units; lower is better):",
        numeric.round(6).to_string(index=False), "",
        "C is primary; B is secondary. Scores use valid reference labels and disclose missing/invalid references.",
        "The model was trained on nine binary QIDs; new_qids are expanded-task evaluation.",
        "No single overall number mixes categorical accuracy and numeric errors.",
        "Raw numeric MAE is valid-prediction-only; its baseline uses the identical subset.",
        "Median-fallback MAE is a separate system metric substituting the training median for invalid outputs.",
        "Human B-vs-C reference and matrix-vector exact match are saved separately; human consistency is not a strict ceiling.",
        "This test set has now been evaluated. Further changes informed by these scores are exploratory.",
        f"Detailed scores: {score_dir}"])+"\n"


def infer(model, tokenizer, dataset, manifest, raw_path, batch_size=1, device="cuda", status_path=None):
    import torch
    from transformers import GenerationConfig
    from full_b_schema import parse_prediction
    records = read_resume(raw_path)
    seen = validate_records(records, manifest)
    pending = [i for i, spec in enumerate(manifest) if (str(spec["pid"]), spec["item_id"]) not in seen]
    started, initial = time.monotonic(), len(records)
    # Group only by fixed output cap, with recorded original indices. No labels
    # are involved and each group preserves source order.
    for cap in (8, 16):
        indices = [i for i in pending if (8 if manifest[i]["qid"] in TRAINED_QIDS else 16) == cap]
        generation = GenerationConfig(do_sample=False, max_new_tokens=cap,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id, use_cache=True)
        with raw_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                rows = [dataset[i] for i in batch_indices]
                specs = [manifest[i] for i in batch_indices]
                for row, spec in zip(rows, specs):
                    require(len(row["input_ids"]) == len(row["attention_mask"]) == spec["input_tokens"]
                            and all(int(v) == 1 for v in row["attention_mask"]), "Prepared token/mask mismatch")
                inputs = tokenizer.pad(rows, padding=True, return_tensors="pt").to(device)
                width = inputs["input_ids"].shape[1]
                require(width + cap <= model.config.max_position_embeddings, "Input plus output exceeds context limit")
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=generation)
                for spec, output in zip(specs, generated):
                    generated_ids = output[width:].tolist()
                    eos = tokenizer.eos_token_id in generated_ids
                    n_generated = generated_ids.index(tokenizer.eos_token_id) + 1 if eos else len(generated_ids)
                    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    prediction = parse_prediction(raw, spec)
                    record = {k: spec[k] for k in ("pid", "item_id", "qid", "family", "prompt_hash", "input_tokens")}
                    record.update(raw_output=raw, prediction=prediction, valid=prediction is not None,
                        max_new_tokens=cap, generated_tokens=n_generated, eos_reached=eos,
                        hit_token_cap=not eos and n_generated >= cap)
                    handle.write(base.dumps(record) + "\n")
                    handle.flush()
                    records.append(record)
                completed = len(records) - initial
                elapsed = time.monotonic() - started
                speed = completed / elapsed if elapsed else 0
                if completed % 50 < batch_size or len(records) == len(manifest):
                    remaining = (len(manifest) - len(records)) / speed if speed else None
                    print(f"Generated {len(records)}/{len(manifest)} items | {speed:.2f} items/s | "
                          f"estimated remaining {(remaining or 0)/3600:.2f} h", flush=True)
                    if status_path:
                        write_json(status_path, {"state": "generating", "updated_utc": now(),
                            "completed": len(records), "total": len(manifest), "test_answers_read": False})
                del inputs, generated
    require(len(validate_records(records, manifest)) == len(manifest), "Predictions incomplete; scoring refused")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--run", type=Path, default=Path("outputs/qwen_full_v1"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qwen_full_b_test_v1"))
    parser.add_argument("--selection-model", type=Path, help="Original Liquid tokenizer; inferred from training config by default")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true")
    actions.add_argument("--prepare-only", action="store_true")
    actions.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.run = (args.run if args.run.is_absolute() else args.root / args.run).resolve()
    args.out = (args.out if args.out.is_absolute() else args.root / args.out).resolve()
    if args.selection_model is None:
        source = read_json(args.run / "run_config.json")["source_run"]
        args.selection_model = Path(read_json(Path(source) / "run_config.json")["model"])
    args.selection_model = args.selection_model.resolve()
    require(args.workers >= 1 and args.batch_size >= 1, "workers and batch-size must be positive")
    require(args.out != args.run and args.run not in args.out.parents, "Use a separate test output folder")
    preflight(args)
    if args.check_only:
        print("No test question inputs or test answer values were loaded.", flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".run.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("This full-B evaluation is already running")
    status_path = args.out / "status.json"
    write_json(status_path, {"state": "preparing", "updated_utc": now(), "test_answers_read": False})
    import full_b_data as data
    import full_b_metrics as metrics
    import full_b_schema as schema
    training, selection, candidate, checkpoint, original, versions = load_frozen_run(
        args.root, args.run, args.selection_model)
    source_paths = source_files(args.root, args.run)
    source_hashes = {str(path): base.digest_file(path) for path in source_paths}
    directory = Path(__file__).resolve().parent
    script_paths = [directory / name for name in ("run_qwen_full_b_test.py", "full_b_data.py", "full_b_schema.py",
        "full_b_metrics.py", "run_liquid_baseline.py", "diagnose_liquid_outputs.py", "full_b_question_catalog.json")]
    config = {"protocol": PROTOCOL, "created_by_user_request": "full B evaluation for test participants",
        "model_path": str(checkpoint), "selected_checkpoint": candidate, "training_run": str(args.run),
        "training_selection": selection, "trained_question_ids": sorted(TRAINED_QIDS),
        "evaluation_split": "test", "primary_labels": "C", "secondary_labels": "B",
        "generation": {"do_sample": False, "batch_size": args.batch_size,
            "trained_binary_max_new_tokens": 8, "other_scalar_max_new_tokens": 16},
        "selection_tokenizer": str(args.selection_model), "selection_token_budget": 8192,
        "qwen_input_safety_ceiling": 32760, "workers": args.workers,
        "source_files_sha256": source_hashes, "script_files_sha256": {p.name: base.digest_file(p) for p in script_paths},
        "versions": versions, "test_policy": "One frozen expanded-question evaluation. No training, calibration, "
            "threshold fitting or checkpoint selection from test results. Nine trained QIDs reported separately.",
        "metric_policy": "Categorical accuracy; ordinal/bounded normalized absolute error; invalid bounded error=1; "
            "unbounded valid-only MAE with coverage and same-mask baseline plus separately named median-fallback MAE. "
            "Human B-vs-C reference uses paired valid labels. No mixed-unit global score."}
    config_path = args.out / "config.json"
    if config_path.exists():
        require(read_json(config_path) == config, "Frozen test configuration changed; keep original settings to resume")
    else:
        require(not (args.out / "raw_predictions.jsonl").exists(), "Untracked prior test predictions")
        write_json(config_path, config)
    # This stage reads A and sanitized question metadata, never held-out labels.
    dataset, manifest, preparation = data.prepare_inputs(args.root, args.out, original,
        args.selection_model, workers=args.workers, max_input_tokens=32760, selection_budget=8192)
    require(len(dataset) == len(manifest), "Prepared test inputs/metadata mismatch")
    split = data.read_split(args.root)
    require(set(str(p["pid"]) for p in manifest) == split["test"], "Prepared participants differ from the test split")
    require(set(dataset.column_names) == {"input_ids", "attention_mask"}, "Tokenized inference data contains unexpected fields")
    specs_by_item = {}
    for item in manifest:
        require(item["input_tokens"] + (8 if item["qid"] in TRAINED_QIDS else 16) <= 32768,
                "A prepared input exceeds the Qwen context after reserving output tokens")
        key = item["item_id"]
        if key in specs_by_item:
            previous = specs_by_item[key]
            require(all(item[k] == previous[k] for k in ("qid", "family", "options", "lower", "upper")),
                    "A target item has inconsistent answer schemas")
        else:
            specs_by_item[key] = item
    catalog_items = schema.catalog_items(read_json(directory / "full_b_question_catalog.json"))
    require(set(specs_by_item).issubset({p["item_id"] for p in catalog_items}), "Prepared target item is outside full B")
    response_dir = args.root / "data/Twin-2K-500/question_catalog_and_human_response_csv"
    b_path, c_path = response_dir / "wave1_3_response.csv", response_dir / "wave4_response.csv"
    # Only training-participant B cells are parsed to freeze the baselines.
    train_b = data.read_selected_labels(b_path, split["train"], set(specs_by_item))
    values = {item_id: [value for (pid, key), value in train_b.items() if key == item_id]
              for item_id in specs_by_item}
    baseline = metrics.build_baselines(values, specs_by_item)
    require(all(baseline[item_id]["n"] > 0 for item_id in specs_by_item),
            "At least one target item has no valid training-B labels; a baseline cannot be defined")
    baseline_path = args.out / "training_baselines.json"
    if baseline_path.exists():
        require(read_json(baseline_path) == baseline, "Training-only baselines changed")
    else:
        write_json(baseline_path, baseline)
    del train_b, values
    print(f"Frozen checkpoint: {candidate['name']}; test participants: 309; "
          f"scalar predictions required: {len(manifest)}", flush=True)
    print("Matrix rows and multi-slider statements are separate scalar predictions.", flush=True)
    if args.prepare_only:
        write_json(status_path, {"state": "prepared", "updated_utc": now(), "test_answers_read": False,
                                 "total": len(manifest)})
        print("Preparation complete. No test B/C answer values were opened.", flush=True)
        return
    raw_path = args.out / "raw_predictions.jsonl"
    records = read_resume(raw_path)
    seen = validate_records(records, manifest)
    if len(seen) != len(manifest):
        require(not args.score_only, "Cannot score an incomplete test prediction file")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
        original_tokenizer = AutoTokenizer.from_pretrained(original, local_files_only=True, padding_side="left")
        for tok in (tokenizer, original_tokenizer):
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
        require(tokenizer.get_vocab() == original_tokenizer.get_vocab()
                and tokenizer.chat_template == original_tokenizer.chat_template
                and all(getattr(tokenizer,k) == getattr(original_tokenizer,k)
                        for k in ("bos_token_id", "eos_token_id", "pad_token_id")), "Checkpoint tokenizer changed")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
            dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
        require(model.config.model_type == "qwen2" and sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS,
                "Unexpected full-model architecture or parameter count")
        write_json(args.out / "test_evaluation_started.json", {"protocol": PROTOCOL,
            "updated_utc": now(), "checkpoint": candidate["name"], "configuration_sha256": base.digest_file(config_path),
            "note": "Test answers will be scored only after all frozen predictions are complete."})
        records = infer(model, tokenizer, dataset, manifest, raw_path, args.batch_size, status_path=status_path)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    require(len(validate_records(records, manifest)) == len(manifest), "Incomplete predictions; scoring refused")
    require(all(base.digest_file(Path(path)) == digest for path, digest in source_hashes.items()),
            "Dataset or selection changed during inference; scoring refused")
    # This is the first stage allowed to parse test B/C answer values.
    write_json(status_path, {"state": "scoring", "updated_utc": now(), "test_answers_read": True,
                            "completed": len(records), "total": len(manifest)})
    verify_retest_metadata(args.root, split["test"], args.out)
    b_labels = data.read_selected_labels(b_path, split["test"], set(specs_by_item))
    c_labels = data.read_selected_labels(c_path, split["test"], set(specs_by_item))
    score_dir = args.out / "scores"
    score_dir.mkdir(exist_ok=True)
    summary = metrics.score_full(records, manifest, b_labels, c_labels, baseline, score_dir)
    summary.update(protocol=PROTOCOL, model=str(checkpoint), checkpoint=candidate["name"],
        test_participants=309, generated_scalar_items=len(manifest), trained_question_ids=sorted(TRAINED_QIDS),
        heldout_test_evaluated=True, training_scope="Fine-tuned on nine binary QIDs only; other QIDs are expanded-task evaluation.",
        configuration_sha256=base.digest_file(config_path), raw_predictions_sha256=base.digest_file(raw_path))
    write_json(args.out / "summary.json", summary)
    report = make_report(summary, score_dir, candidate["name"], len(manifest))
    (args.out / "report.txt").write_text(report, encoding="utf-8")
    write_json(status_path, {"state": "complete", "updated_utc": now(), "test_answers_read": True,
                            "completed": len(records), "total": len(manifest)})
    print(report, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Full-B evaluation stopped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
