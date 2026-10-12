#!/usr/bin/env python3
"""Resumable validation comparisons for the retained full-B epoch-1 adapter.

All model inputs and training-only baselines are frozen before validation B/C
labels are parsed. This script never trains, selects a checkpoint, or evaluates
reserved test participants. Run start_comparisons.sh on the original AutoDL.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import gc
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import infer_qwen_full_b_epoch1 as fixed
import baselines as methods
import data_io
import prompts
import reporting
from collect_results import collect

PROTOCOL = "qwen-epoch1-comparisons-validation-v1"
require = fixed.require
read_json = fixed.read_json


def freeze(engine, path, value):
    engine.evaluator.freeze_json(path, value)


def source_hashes(engine):
    return {p.name: engine.base.digest_file(p)
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def write_jsonl(path, rows, engine):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(engine.base.dumps(row) + "\n")
    temporary.replace(path)


class PreparedWriter:
    """Stream token rows to disk, keeping only the much smaller manifest in RAM."""
    def __init__(self, folder, engine):
        import pyarrow as pa
        from datasets import Features, Sequence, Value
        self.pa, self.engine, self.folder = pa, engine, Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "inputs.arrow"
        self.tmp = self.folder / "inputs.arrow.tmp"
        features = Features({"input_ids": Sequence(Value("int32")),
                             "attention_mask": Sequence(Value("int8"))})
        self.schema = features.arrow_schema
        self.sink = pa.OSFile(str(self.tmp), "wb")
        self.writer = pa.ipc.new_stream(self.sink, self.schema)
        self.manifest, self.buffer = [], []

    def add(self, record, encoded):
        self.engine.data.validate_prefix(encoded, record)
        self.manifest.append(record)
        self.buffer.append(encoded)
        if len(self.buffer) >= 64:
            self.flush()

    def flush(self):
        if self.buffer:
            self.writer.write_table(self.pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()

    def finish(self):
        self.flush()
        self.writer.close()
        self.sink.close()
        self.tmp.replace(self.path)
        write_jsonl(self.folder / "manifest.jsonl", self.manifest, self.engine)
        self.engine.base.write_json(self.folder / "prepared_complete.json", {
            "rows": len(self.manifest), "artifacts_sha256": {
                name: self.engine.base.digest_file(self.folder / name)
                for name in ("inputs.arrow", "manifest.jsonl")}})


def prepared(folder, engine):
    from datasets import Dataset
    folder = Path(folder)
    info = read_json(folder / "prepared_complete.json")
    fixed.verify_named_files(folder, info["artifacts_sha256"], engine.base)
    manifest = engine.base.read_jsonl(folder / "manifest.jsonl")
    dataset = Dataset.from_file(str(folder / "inputs.arrow"))
    require(len(dataset) == len(manifest) == info["rows"], "Prepared stage coverage changed")
    require(set(dataset.column_names) == {"input_ids", "attention_mask"}, "Stage contains answer labels")
    for row, spec in zip(dataset, manifest):
        engine.data.validate_prefix(row, spec)
    return dataset, manifest


def preserve_targets(original, changed):
    require(len(original) == len(changed), "Intervention removed validation targets")
    for left, right in zip(original, changed):
        require((left["pid"], left["item_id"]) == (right["pid"], right["item_id"])
                and prompts.target_signature(left) == prompts.target_signature(right),
                "Intervention changed recipient question, price, options or scoring schema")


def unique_inputs(dataset, manifest, engine):
    """Reuse only exact identical token sequences with identical output cap.

    Useful for no-A targets, which repeat across people. No label or donor
    information is used. Full equality guards hash collisions.
    """
    first, indices, mapping = {}, [], []
    for i, spec in enumerate(manifest):
        cap = engine.data.response_cap(spec)
        key = (spec["prompt_hash"], cap)
        if key not in first:
            first[key] = i
            indices.append(i)
        j = first[key]
        require(dataset[i] == dataset[j], "Equal prompt hashes have unequal tokens")
        mapping.append(j)
    return indices, mapping


class RowView:
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def generate_stage(folder, dataset, manifest, engine, model, tokenizer, batch_size):
    folder = Path(folder)
    raw = folder / "raw_predictions.jsonl"
    if (folder / "generation_complete.json").exists():
        info = read_json(folder / "generation_complete.json")
        require(engine.base.digest_file(raw) == info["raw_sha256"], "Completed predictions changed")
        rows = engine.base.read_jsonl(raw)
        require(len(engine.generation.validate_records(rows, manifest)) == len(manifest), "Incomplete stage")
        return rows
    indices, mapping = unique_inputs(dataset, manifest, engine)
    unique_manifest = [manifest[i] for i in indices]
    rows = engine.generation.infer(model, tokenizer, RowView(dataset, indices), unique_manifest,
        folder / "unique_predictions.jsonl", batch_size=batch_size, status_path=folder / "status.json")
    by_key = {(str(r["pid"]), r["item_id"]): r for r in rows}
    expanded = []
    for spec, source_index in zip(manifest, mapping):
        source = manifest[source_index]
        row = dict(by_key[(str(source["pid"]), source["item_id"])])
        row.update({k: spec[k] for k in ("pid", "item_id", "qid", "family", "prompt_hash", "input_tokens")})
        row["reused_identical_prompt"] = (str(spec["pid"]), spec["item_id"]) != (str(source["pid"]), source["item_id"])
        expanded.append(row)
    require(len(engine.generation.validate_records(expanded, manifest)) == len(manifest), "Expansion coverage failed")
    write_jsonl(raw, expanded, engine)
    engine.base.write_json(folder / "generation_complete.json", {
        "rows": len(expanded), "unique_prompts": len(indices),
        "raw_sha256": engine.base.digest_file(raw), "test_evaluated": False})
    return expanded


def verify_original_results(run, engine, training, candidate, manifest):
    folder = run / "validation_epoch1"
    config = read_json(folder / "evaluation_config.json")
    done = read_json(folder / "evaluation_complete.json")
    require(config["protocol"] == fixed.PROTOCOL and config["epoch"] == 1
            and config["candidate"] == candidate
            and config["training_config_sha256"] == engine.base.digest_file(run / "run_config.json")
            and config["epoch_descriptor_sha256"] == engine.base.digest_file(run / "epoch_adapters/epoch-1.json"),
            "Existing validation predictions do not belong to this fixed epoch-1 adapter")
    require(done["predictions_sha256"] == engine.base.digest_file(folder / "raw_predictions.jsonl")
            and done["evaluation_config_sha256"] == engine.base.digest_file(folder / "evaluation_config.json"),
            "Original evaluation artifacts changed")
    rows = engine.base.read_jsonl(folder / "raw_predictions.jsonl")
    require(len(engine.generation.validate_records(rows, manifest)) == len(manifest),
            "Run infer_qwen_full_b_epoch1.py first: original predictions are incomplete")
    return config, rows


def load_base(training, engine):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    engine.generation.verify_hashes(Path(training["model"]), training["model_files_sha256"])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(training["model"], local_files_only=True, padding_side="left")
    model, loading = AutoModelForCausalLM.from_pretrained(training["model"], local_files_only=True,
        dtype=dtype, attn_implementation="sdpa", output_loading_info=True)
    require(all(not loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
            "Base model loading mismatch")
    require(sum(p.numel() for p in model.parameters()) == engine.evaluator.EXPECTED_PARAMETERS,
            "Wrong base model size")
    model.requires_grad_(False)
    return model.to("cuda").eval(), tokenizer


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=fixed.DEFAULT_ROOT)
    p.add_argument("--run", type=Path, default=fixed.DEFAULT_RUN)
    p.add_argument("--out", type=Path, default=Path("outputs/qwen_epoch1_comparisons_v1"))
    p.add_argument("--scripts-dir", type=Path)
    p.add_argument("--batch-size", type=int, default=1,
                   help="1 matches saved original; other sizes also rerun the original adapter prompts")
    p.add_argument("--shuffle-seed", type=int, default=20261012)
    actions = p.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true")
    actions.add_argument("--prepare-only", action="store_true")
    actions.add_argument("--score-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    require(args.batch_size > 0, "Batch size must be positive")
    root = args.root.resolve()
    run = (args.run if args.run.is_absolute() else root / args.run).resolve()
    out = (args.out if args.out.is_absolute() else root / args.out).resolve()
    require(out != run and run not in out.parents, "Use a separate comparison output directory")
    require(run.is_dir(), f"Original training run missing: {run}")
    out.mkdir(parents=True, exist_ok=True)
    engine = fixed.import_engines(args.scripts_dir or root / "qwen_full_b_lora_scripts")
    with ExitStack() as locks:
        for path in (run / ".pipeline.lock", run / ".training.lock", out / ".comparisons.lock"):
            handle = locks.enter_context(path.open("a+"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f"Another training/comparison job owns {path}")
        artifacts = fixed.inspect_run(root, run, engine)
        training, candidate, adapter, original_data, original_manifest, prep, train_baselines, item_specs, split = artifacts
        versions = fixed.check_versions(training)
        original_config, original_rows = verify_original_results(run, engine, training, candidate, original_manifest)
        dataset_dir = root / "data/Twin-2K-500"
        responses = dataset_dir / "question_catalog_and_human_response_csv"
        b_path, c_path = responses / "wave1_3_response.csv", responses / "wave4_response.csv"
        labels_sha = {"B": engine.base.digest_file(b_path), "C": engine.base.digest_file(c_path)}
        require(labels_sha == original_config["labels_sha256"], "Original/current response sources differ")
        config = {"protocol": PROTOCOL, "fixed_epoch": 1, "candidate": candidate,
            "root": str(root), "run": str(run), "evaluation_split": "validation", "test_evaluated": False,
            "training_config_sha256": engine.base.digest_file(run / "run_config.json"),
            "original_predictions_sha256": engine.base.digest_file(run / "validation_epoch1/raw_predictions.jsonl"),
            "preparation_summary_sha256": engine.base.digest_file(run / "preparation_summary.json"),
            "source_sha256": source_hashes(engine), "labels_sha256": labels_sha, "versions": versions,
            "batch_size": args.batch_size, "shuffle_seed": args.shuffle_seed,
            "knn_k": 5, "demonstrations": 3, "retrieval_budget": 4096,
            "retrieval_own_budget": 1800, "retrieval_donor_budget": 450,
            "price_method": "per-product training-B monotone isotonic purchase curve",
            "retrieval_rule": "train-A-only similarity; exact target/options/bounds; pricing matches actual product and price across QID positions; other items match item ID",
            "selection": "no model, threshold or hyperparameter selection",
            "original_batch_size": original_config["generation"]["batch_size"]}
        freeze(engine, out / "comparison_config.json", config)
        print(f"Epoch-1 comparison preflight passed: {len(original_manifest):,} validation items; "
              f"adapter step {candidate['global_step']}; batch {args.batch_size}", flush=True)
        if args.check_only:
            print("No model inference or validation answer parsing performed.")
            return

        stage_names = ["adapter_no_a", f"adapter_shuffled_a_seed{args.shuffle_seed}",
                       "base_compact_no_demo", "base_retrieved_3"]
        all_prepared = all((out / name / "prepared_complete.json").exists() for name in stage_names)
        cpu_done = out / "cpu_predictions_complete.json"
        if not (all_prepared and cpu_done.exists()):
            require(not args.score_only, "Preparation incomplete; rerun without --score-only")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(training["model"], local_files_only=True, padding_side="left")
            selection = AutoTokenizer.from_pretrained(training["selection_model"], local_files_only=True, padding_side="left")
            print("Loading training/validation A profiles and answer-free target metadata...", flush=True)
            people = data_io.load_people(dataset_dir, split, engine)
            verification = prompts.verify_originals(engine, tokenizer, original_data, original_manifest, people)
            engine.base.write_json(out / "original_prompt_verification.json", verification)
            train_labels = engine.inputs.read_selected_labels(b_path, split["train"], list(item_specs))
            print("Fitting training-only price curve and ranking A-profile neighbors...", flush=True)
            fitted = methods.build_baselines(people, train_labels, split["train"], split["validation"], k=5, demonstrations=3)
            engine.base.write_json(out / "retrieval_and_price_audit.json", fitted["report"])
            for name, outputs in (("knn_5", fitted["knn"]), ("product_price", fitted["price"])):
                folder = out / name
                folder.mkdir(exist_ok=True)
                specs = [dict(s) for s in original_manifest if (str(s["pid"]), s["item_id"]) in outputs]
                require(specs and len(specs) == len(outputs), "CPU baseline coverage mismatch")
                for s in specs:
                    s["input_tokens"] = 0
                    s["prompt_hash"] = engine.base.digest_text(engine.base.dumps({"method": name,
                        "config": engine.base.digest_file(out / "comparison_config.json"),
                        "pid": s["pid"], "item_id": s["item_id"], "target": s["target_text"]}))
                records = data_io.make_records(outputs, specs, engine, name)
                write_jsonl(folder / "manifest.jsonl", specs, engine)
                write_jsonl(folder / "raw_predictions.jsonl", records, engine)
            engine.base.write_json(cpu_done, {"artifacts_sha256": {
                f"{name}/{filename}": engine.base.digest_file(out / name / filename)
                for name in ("knn_5", "product_price") for filename in ("manifest.jsonl", "raw_predictions.jsonl")}})
            for name, mode in zip(stage_names[:2], ("no_a", "shuffled_a")):
                folder = out / name
                if (folder / "prepared_complete.json").exists():
                    continue
                print(f"Preparing {name}...", flush=True)
                writer = PreparedWriter(folder, engine)
                for record, encoded in prompts.iter_ablation(engine, tokenizer, selection, original_manifest,
                        people, mode=mode, seed=args.shuffle_seed):
                    writer.add(record, encoded)
                preserve_targets(original_manifest, writer.manifest)
                writer.finish()
            if not all((out / n / "prepared_complete.json").exists() for n in stage_names[2:]):
                require(not any((out / n / "raw_predictions.jsonl").exists() for n in stage_names[2:]),
                        "Cannot rebuild retrieval inputs after inference; choose a fresh output")
                writers = [PreparedWriter(out / n, engine) for n in stage_names[2:]]
                donors = {key: [dict(d) for d in row["donors"][:3]] for key, row in fitted["knn"].items()}
                print("Preparing matched compact prompts with and without retrieved examples...", flush=True)
                for pair in prompts.iter_retrieval_pairs(engine, tokenizer, original_manifest, people,
                        donors, train_labels, split["train"], budget=4096, k=3, own_budget=1800, donor_budget=450):
                    for writer, (record, encoded) in zip(writers, pair):
                        writer.add(record, encoded)
                for writer in writers:
                    preserve_targets(original_manifest, writer.manifest)
                    writer.finish()
            del people, fitted, train_labels, tokenizer, selection
            gc.collect()

        fixed.verify_named_files(out, read_json(cpu_done)["artifacts_sha256"], engine.base)
        stages = {}
        for name in stage_names:
            dataset, manifest = prepared(out / name, engine)
            preserve_targets(original_manifest, manifest)
            stages[name] = (dataset, manifest)
        if args.prepare_only:
            print("Preparation complete. No GPU inference or validation answer parsing performed.", flush=True)
            return

        original_out = out / "adapter_original"
        original_out.mkdir(exist_ok=True)
        if args.batch_size == original_config["generation"]["batch_size"]:
            write_jsonl(original_out / "raw_predictions.jsonl", original_rows, engine)
            original_predictions = original_rows
        else:
            stages = {"adapter_original": (original_data, original_manifest), **stages}
            original_predictions = None
        write_jsonl(original_out / "manifest.jsonl", original_manifest, engine)
        for model_kind in ("adapter", "base"):
            selected = {n: pair for n, pair in stages.items() if n.startswith(model_kind + "_")}
            pending = [n for n in selected if not (out / n / "generation_complete.json").exists()]
            model = tokenizer = None
            if pending:
                require(not args.score_only, "GPU predictions incomplete; rerun without --score-only")
                fixed.gpu_is_free()
                import torch
                from transformers import AutoTokenizer
                require(torch.cuda.is_available(), "CUDA unavailable")
                if model_kind == "adapter":
                    original_tokenizer = AutoTokenizer.from_pretrained(training["model"], local_files_only=True, padding_side="left")
                    model, tokenizer = engine.evaluator._load_model(adapter, original_tokenizer,
                        training["model"], training["model_files_sha256"], base_parameter_dtype=training["base_parameter_dtype"])
                else:
                    model, tokenizer = load_base(training, engine)
            try:
                for name, (dataset, manifest) in selected.items():
                    print(f"Stage: {name}", flush=True)
                    generated = generate_stage(out / name, dataset, manifest, engine, model, tokenizer, args.batch_size)
                    if name == "adapter_original":
                        original_predictions = generated
            finally:
                if model is not None:
                    del model, tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()

        # Scoring starts only after every requested method has complete predictions.
        score_jobs = {"adapter_original": (original_predictions, original_manifest)}
        for name, (_, manifest) in stages.items():
            records = engine.base.read_jsonl(out / name / "raw_predictions.jsonl")
            require(len(engine.generation.validate_records(records, manifest)) == len(manifest), "Incomplete prediction stage")
            score_jobs[name] = (records, manifest)
        for name in ("knn_5", "product_price"):
            score_jobs[name] = (engine.base.read_jsonl(out / name / "raw_predictions.jsonl"),
                                engine.base.read_jsonl(out / name / "manifest.jsonl"))
        require(source_hashes(engine) == config["source_sha256"], "Comparison code changed during execution")
        require(engine.base.digest_file(run / "run_config.json") == config["training_config_sha256"]
                and engine.base.digest_file(run / "preparation_summary.json") == config["preparation_summary_sha256"],
                "Original training provenance changed during execution")
        for path, expected in read_json(run / "preparation_config.json")["inputs_sha256"].items():
            require(engine.base.digest_file(path) == expected, "Frozen original data changed during execution")
        for name in stage_names:
            fixed.verify_named_files(out / name,
                read_json(out / name / "prepared_complete.json")["artifacts_sha256"], engine.base)
        engine.generation.verify_hashes(adapter, candidate["model_files_sha256"])
        for name, path in (("B", b_path), ("C", c_path)):
            require(engine.base.digest_file(path) == labels_sha[name], "Response source changed during execution")
        engine.generation.verify_retest_metadata(root, split["validation"], out)
        labels_b = engine.inputs.read_selected_labels(b_path, split["validation"], list(item_specs))
        labels_c = engine.inputs.read_selected_labels(c_path, split["validation"], list(item_specs))
        score_dirs = {}
        for name, (records, manifest) in score_jobs.items():
            require(records and len(records) == len(manifest), "Missing method predictions")
            scores = out / name / "scores"
            engine.metrics.score_full(records, manifest, labels_b, labels_c, train_baselines, scores,
                trained_qids=set(training["trained_question_ids"]), trained_scope="fixed_epoch1_comparisons")
            score_dirs[name] = scores
        metadata = {
            "adapter_original": {"checkpoint": str(adapter), "epoch": 1,
                "source": "saved original predictions" if args.batch_size == config["original_batch_size"]
                          else "original prompts rerun at the matched batch size"},
            "adapter_no_a": {"interpretation": "empty A; original system and target retained"},
            stage_names[1]: {"seed": args.shuffle_seed,
                "interpretation": "one global no-self participant permutation; a single diagnostic, not a confidence interval"},
            "knn_5": {"source": "training-only B; A-only neighbor ranking; exact target matching"},
            "product_price": {"scope": "40 pricing QIDs only", "uses_A": False,
                "fitting": "training B only; fixed monotonic curve and 0.5 threshold"},
        }
        for name in stage_names[2:]:
            manifest = stages[name][1]
            metadata[name] = {
                "model": "Qwen2.5-0.5B-Instruct base without adapter",
                "actual_demonstration_count_histogram": dict(sorted(Counter(
                    int(s["demonstrations_in_prompt"]) for s in manifest).items())),
                "requested_demonstrations": 0 if name == "base_compact_no_demo" else 3,
                "max_input_tokens": max(s["input_tokens"] for s in manifest),
                "budget_including_generation": 4096,
            }
        reporting.make_comparison_report(out, score_dirs, stage_metadata=metadata)
        engine.base.write_json(out / "complete.json", {"protocol": PROTOCOL, "stages": list(score_jobs),
            "config_sha256": engine.base.digest_file(out / "comparison_config.json"), "test_evaluated": False})
        print((out / "report.txt").read_text(), flush=True)
        print(f"Aggregate results ZIP: {collect(out)}", flush=True)
        print(f"ALL COMPARISONS COMPLETE: {out}", flush=True)


if __name__ == "__main__":
    main()
