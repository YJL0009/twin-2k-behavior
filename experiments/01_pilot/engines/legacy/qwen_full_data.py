#!/usr/bin/env python3
"""Rebuild Liquid's exact text examples with Qwen token IDs; never reuse Arrow IDs.

Training labels are B from training participants only. The source manifest fixes
whole A units and verifies their rendered message hashes. Validation messages
are frozen Liquid-v3 messages, retokenized with Qwen without truncation/refill.
"""
import csv
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import run_liquid_baseline as base
import run_liquid_plain_baseline as plain
import diagnose_liquid_outputs as diag
import train_liquid_lora as legacy

PROTOCOL = "qwen-nine-binary-a-to-b-full-fixed-content-v1"
EXPECTED_PARAMETERS = 494_032_768
OFFICIAL_WEIGHTS_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
require = base.require


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_files_sha256(folder):
    files = sorted(p for p in Path(folder).iterdir() if p.is_file() and
                   (p.suffix in {".json", ".safetensors", ".model", ".jinja"}
                    or p.name in {"merges.txt", "vocab.txt", "pytorch_model.bin"}))
    return {p.name: base.digest_file(p) for p in files
            if p.name not in {"trainer_state.json", "training_args.json"}}


def reconstruct_prompt(row, record, tokenizer, budget):
    """Reproduce both recorded Liquid hashes using source text, then tokenize Qwen."""
    a_units = base.survey_units(row["wave1_3_persona_json"], True)
    metadata = base.survey_units(row["wave4_Q_wave1_3_A"], False)
    target = base.target_for(metadata, record["qid"])
    require(not diag.has_answers(target), "Target metadata contains answer fields")
    indices = json.loads(record["a_unit_indices"])
    require(isinstance(indices, list) and indices and
            all(type(i) is int and 0 <= i < len(a_units) for i in indices) and
            indices == sorted(set(indices)), "Invalid recorded A selection")
    require(len(a_units) == record["a_units_total"] and len(indices) == record["a_units_selected"],
            "A selection counts changed")
    context = "\n\n".join(base.dumps(a_units[i][1]) for i in indices)
    original = {"pid": str(row["pid"]), "qid": record["qid"],
                "messages": base.messages_for(context, target),
                "input_tokens": int(record["source_input_tokens"])}
    original["prompt_hash"] = base.digest_text(base.dumps(original["messages"]))
    require(original["prompt_hash"] == record["source_prompt_hash"],
            f"Original JSON prompt changed: {original['pid']}/{record['qid']}")
    messages = diag.make_case("qwen_same_content", original, plain=True)["messages"]
    prompt_hash = base.digest_text(base.dumps(messages))
    require(prompt_hash == record["prompt_hash"],
            f"Original plain prompt changed: {original['pid']}/{record['qid']}")
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_dict=False, truncation=False)
    require(isinstance(tokens, list) and all(type(i) is int for i in tokens), "Invalid token IDs")
    require(0 < len(tokens) <= budget,
            f"Qwen context too long ({len(tokens)} > {budget}); no text was truncated")
    prompt = {"pid": original["pid"], "qid": record["qid"], "messages": messages,
              "prompt_hash": prompt_hash, "input_tokens": len(tokens)}
    encoded = legacy.encode_supervised(tokenizer, prompt, int(record["label_b"]))
    result = dict(record)
    result.update(liquid_input_tokens=int(record["input_tokens"]),
                  input_tokens=len(tokens), total_tokens=len(encoded["input_ids"]))
    return result, encoded


_TOKENIZER = None
_BUDGET = None


def init_worker(model, budget):
    global _TOKENIZER, _BUDGET
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    _TOKENIZER = AutoTokenizer.from_pretrained(model, local_files_only=True, padding_side="right")
    _BUDGET = budget


def prepare_job(job):
    row, records = job
    require(_TOKENIZER is not None, "Tokenizer worker not initialized")
    return str(row["pid"]), [reconstruct_prompt(row, r, _TOKENIZER, _BUDGET) for r in records]


def stream_prepared(args, tokenizer, jobs):
    if args.preprocess_workers == 1:
        for row, records in jobs:
            yield str(row["pid"]), [reconstruct_prompt(row, r, tokenizer, args.max_input_tokens)
                                    for r in records]
    else:
        with ProcessPoolExecutor(max_workers=args.preprocess_workers,
                mp_context=mp.get_context("spawn"), initializer=init_worker,
                initargs=(str(args.model), args.max_input_tokens)) as pool:
            yield from legacy.ordered_bounded_map(pool, prepare_job, jobs,
                                                 limit=2 * args.preprocess_workers)


def prepare_artifacts(args, tokenizer, manifest, train_pids, all_pids, parquet_paths,
                      validation_prompts, config_hash):
    import pyarrow as pa
    import pyarrow.dataset as ds
    from datasets import Dataset, Features, Sequence, Value
    prepared = args.out / "prepared"
    prepared.mkdir(exist_ok=True)
    artifacts = {"train.arrow": prepared / "train.arrow",
                 "training_manifest.csv": args.out / "training_manifest.csv",
                 "mask_audit.json": args.out / "mask_audit.json",
                 "validation_prompts.jsonl": prepared / "validation_prompts.jsonl"}
    summary_path = args.out / "preprocessing_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        require(summary["config_hash"] == config_hash, "Prepared configuration changed")
        for name, path in artifacts.items():
            require(path.is_file() and base.digest_file(path) == summary["artifacts_sha256"][name],
                    f"Prepared artifact changed: {name}")
        dataset = Dataset.from_file(str(artifacts["train.arrow"]))
        require(len(dataset) == len(manifest) == summary["examples"], "Prepared count changed")
        print(f"Reusing {len(dataset)} Qwen-tokenized training examples.", flush=True)
        return dataset
    source = ds.dataset([str(p) for p in parquet_paths], format="parquet")
    ids = list(map(str, source.to_table(columns=["pid"])["pid"].to_pylist()))
    require(len(ids) == len(set(ids)) == 2058 and set(ids) == all_pids, "Dataset IDs differ")
    fields = ["pid", "wave1_3_persona_json", "wave4_Q_wave1_3_A"]
    scanner = source.scanner(columns=fields, batch_size=1,
                            filter=ds.field("pid").cast(pa.string()).isin(sorted(train_pids)))
    grouped = {str(pid): group.to_dict("records") for pid, group in manifest.groupby("pid", sort=False)}
    expected_keys = list(manifest[["pid", "qid"]].itertuples(index=False, name=None))

    def jobs():
        for batch in scanner.to_batches():
            for row in batch.to_pylist():
                pid = str(row["pid"])
                require(pid in train_pids, "Unexpected participant in training scan")
                yield row, grouped[pid]

    features = Features({"input_ids": Sequence(Value("int32")),
                         "attention_mask": Sequence(Value("int8")), "labels": Sequence(Value("int32"))})
    columns = list(manifest.columns) + ["liquid_input_tokens"]
    arrow_tmp = prepared / "train.arrow.tmp"
    manifest_tmp = args.out / "training_manifest.csv.tmp"
    seen, counts, audit, lengths, example_count, buffer = set(), Counter(), [], [], 0, []
    with pa.OSFile(str(arrow_tmp), "wb") as sink, manifest_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        with pa.ipc.new_stream(sink, features.arrow_schema) as arrow_writer:
            for pid, examples in stream_prepared(args, tokenizer, jobs()):
                require(pid in train_pids and pid not in seen, "Duplicate training participant")
                seen.add(pid)
                for record, encoded in examples:
                    require((pid, record["qid"]) == expected_keys[example_count],
                            "Training order differs from Liquid; inspect source manifests")
                    writer.writerow(record)
                    buffer.append(encoded)
                    example_count += 1
                    counts[(record["qid"], int(record["label_b"]))] += 1
                    lengths.append(record["input_tokens"])
                    if len(audit) < 2:
                        audit.append({"pid": pid, "qid": record["qid"], "label_b": record["label_b"],
                            "prompt_tokens_masked": record["input_tokens"],
                            "supervised_token_ids": encoded["labels"][-2:],
                            "supervised_text": tokenizer.decode(encoded["labels"][-2:], skip_special_tokens=False),
                            "prompt_hash_identical_to_liquid": True})
                    if len(buffer) == 16:
                        arrow_writer.write_table(pa.Table.from_pylist(buffer, schema=features.arrow_schema))
                        buffer.clear()
                if len(seen) % 100 == 0:
                    print(f"Retokenized {len(seen)}/1440 participants; {example_count} examples", flush=True)
            if buffer:
                arrow_writer.write_table(pa.Table.from_pylist(buffer, schema=features.arrow_schema))
    require(seen == train_pids and example_count == len(manifest), "Incomplete training preparation")
    val_tmp = prepared / "validation_prompts.jsonl.tmp"
    with val_tmp.open("w", encoding="utf-8") as handle:
        for source_prompt in validation_prompts:
            prompt = dict(source_prompt)
            tokens = tokenizer.apply_chat_template(prompt["messages"], tokenize=True,
                        add_generation_prompt=True, return_dict=False, truncation=False)
            require(isinstance(tokens, list) and all(type(i) is int for i in tokens), "Invalid validation IDs")
            require(0 < len(tokens) <= args.max_input_tokens, "Validation prompt exceeds Qwen context")
            prompt.update(liquid_input_tokens=source_prompt["input_tokens"],
                          liquid_source_input_tokens=source_prompt.get("source_input_tokens"),
                          source_input_tokens=source_prompt["input_tokens"], input_tokens=len(tokens))
            # Text/hash stays exactly the same. Only tokenizer-dependent length changes.
            handle.write(base.dumps(prompt) + "\n")
    arrow_tmp.replace(artifacts["train.arrow"])
    manifest_tmp.replace(artifacts["training_manifest.csv"])
    val_tmp.replace(artifacts["validation_prompts.jsonl"])
    base.write_json(artifacts["mask_audit.json"], audit)
    val_lengths = [p["input_tokens"] for p in base.read_jsonl(artifacts["validation_prompts.jsonl"])]
    summary = {"config_hash": config_hash, "participants": 1440, "examples": example_count,
        "training_labels": "B", "validation_examples": len(val_lengths),
        "identical_training_message_hashes_to_liquid": True,
        "identical_validation_messages_to_liquid": True, "reused_liquid_token_ids": False,
        "counts_by_question_label": {qid: {str(label): counts[(qid, label)] for label in (1, 2)} for qid in base.QIDS},
        "input_tokens": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
        "validation_input_tokens": {"min": min(val_lengths), "max": max(val_lengths),
                                    "mean": sum(val_lengths) / len(val_lengths)},
        "artifacts_sha256": {name: base.digest_file(path) for name, path in artifacts.items()}}
    base.write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("Answer-mask audit:", json.dumps(audit, indent=2), flush=True)
    return Dataset.from_file(str(artifacts["train.arrow"]))


def prepare_run(args):
    import pandas as pd
    from transformers import AutoTokenizer
    source = read_json(args.source_run / "run_config.json")
    complete = read_json(args.source_run / "training_complete.json")
    prep = read_json(args.source_run / "preprocessing_summary.json")
    source_hash = base.digest_text(base.dumps(source))
    require(source["protocol"] == "liquid-nine-binary-a-to-b-lora-plain-target-v1"
            and complete["config_hash"] == prep["config_hash"] == source_hash,
            "Source Liquid training/configuration mismatch")
    require(source["max_input_tokens"] == 8192 and prep["training_labels"] == "B"
            and prep["participants"] == 1440, "Unexpected source training protocol")
    helper_hashes = {Path(p).name: base.digest_file(p) for p in (base.__file__, plain.__file__, diag.__file__)}
    require(helper_hashes == source["helper_scripts_sha256"]
            and base.digest_file(legacy.__file__) == source["script_sha256"],
            "Prompt/training helpers differ from original Liquid run")
    split_path = args.root / "data/participant_split.csv"
    labels_path = args.root / "data/Twin-2K-500/question_catalog_and_human_response_csv/wave1_3_response.csv"
    majority_path = args.root / "outputs/majority_baseline_v1/predictions.csv"
    parquet_paths = sorted((args.root / "data/Twin-2K-500/wave_split/chunks").glob("*.parquet"))
    require(len(parquet_paths) == 7, "Expected seven dataset chunks")
    input_hashes = {str(p): base.digest_file(p) for p in (split_path, labels_path, *parquet_paths)}
    require(input_hashes == source["inputs_sha256"], "Source data or participant split changed")
    source_manifest_path = args.source_run / "training_manifest.csv"
    require(base.digest_file(source_manifest_path) == prep["artifacts_sha256"]["training_manifest.csv"],
            "Source training manifest changed")
    manifest = pd.read_csv(source_manifest_path, dtype={"pid": "string", "qid": "string"})
    labels, train_pids, all_pids = legacy.read_train_labels(split_path, labels_path)
    require(not manifest[["pid", "qid"]].duplicated().any() and set(manifest.pid) == train_pids
            and set(manifest.qid) == set(base.QIDS) and manifest.label_b.isin([1, 2]).all(),
            "Invalid source training manifest")
    require(len(manifest) == prep["examples"] == complete["training_examples"]
            == int(labels.notna().sum().sum()), "Training coverage changed")
    require(all(not pd.isna(labels.at[r.pid, r.qid]) and int(labels.at[r.pid, r.qid]) == r.label_b
                for r in manifest.itertuples()), "Training B labels differ from manifest")
    counts = {qid: {str(label): int((manifest.qid.eq(qid) & manifest.label_b.eq(label)).sum())
                    for label in (1, 2)} for qid in base.QIDS}
    require(counts == prep["counts_by_question_label"], "Training B counts changed")
    anchor = source["validation_baseline"]
    baseline_run = Path(anchor["path"])
    for filename, key in (("config.json", "config_sha256"), ("summary.json", "summary_sha256"),
                          ("prepared_prompts.jsonl", "prepared_prompts_sha256")):
        require(base.digest_file(baseline_run / filename) == anchor[key], "Liquid validation anchor changed")
    baseline_config = read_json(baseline_run / "config.json")
    require(baseline_config["protocol"] == "liquid-nine-binary-fixed-a-plain-target-v3"
            and baseline_config["max_new_tokens"] == 8, "Unexpected Liquid validation protocol")
    input_hashes[str(majority_path)] = base.digest_file(majority_path)
    require(all(input_hashes[str(p)] == baseline_config["inputs_sha256"][str(p)]
                for p in (split_path, majority_path)), "Validation manifest/split changed")
    validation_ids = pd.read_csv(majority_path, usecols=["pid", "qid"], dtype="string")
    split = pd.read_csv(split_path, dtype={"pid": "string"})
    require(len(validation_ids) == 1545 and not validation_ids.duplicated().any()
            and set(validation_ids.qid) == set(base.QIDS)
            and set(validation_ids.pid) == set(split.loc[split["split"].eq("validation"), "pid"]),
            "Validation IDs differ")
    validation_prompts = base.read_jsonl(baseline_run / "prepared_prompts.jsonl")
    require([(p["pid"], p["qid"]) for p in validation_prompts]
            == list(validation_ids.itertuples(index=False, name=None)), "Validation prompt coverage/order changed")
    for p in validation_prompts:
        require(base.digest_text(base.dumps(p["messages"])) == p["prompt_hash"], "Validation prompt hash changed")
        require([m["role"] for m in p["messages"]] == ["system", "user"]
                and p["messages"][0]["content"] == source["system"], "Validation message roles/system differ")
    require(not (args.model / "adapter_config.json").exists(), "Use original Qwen weights, not a LoRA adapter")
    model_config = read_json(args.model / "config.json")
    require(model_config["model_type"] == "qwen2"
            and args.max_input_tokens + 8 <= model_config["max_position_embeddings"], "Unexpected model/context")
    model_hashes = model_files_sha256(args.model)
    require(model_hashes.get("model.safetensors") == OFFICIAL_WEIGHTS_SHA256,
            "Use the previously verified original Qwen2.5-0.5B-Instruct model.safetensors")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="right")
    require(tokenizer.eos_token_id is not None and isinstance(tokenizer.chat_template, str), "Tokenizer template/EOS missing")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "tokenizers", "datasets", "accelerate", "pandas", "pyarrow")}
    require(versions["transformers"] == "5.2.0", "Use /root/autodl-tmp/venvs/liquid_tf52/bin/python (Transformers 5.2.0)")
    total_steps = args.epochs * math.ceil(math.ceil(len(manifest) / args.micro_batch_size) / args.gradient_accumulation)
    config = {"protocol": PROTOCOL, "root": str(args.root), "source_run": str(args.source_run),
        "model": str(args.model), "initialization": "original Qwen2.5-0.5B-Instruct weights; all parameters trainable",
        "qids": list(base.QIDS), "system": source["system"], "epochs": args.epochs,
        "max_input_tokens": args.max_input_tokens, "max_new_tokens": 8,
        "context_policy": "exact Liquid training/validation message content; retokenize with Qwen; no refill or truncation",
        "source_liquid_selection_budget": 8192, "max_input_tokens_is_safety_ceiling_only": True,
        "micro_batch_size": args.micro_batch_size, "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate, "seed": 42, "optimizer": "adamw_torch", "weight_decay": 0.0,
        "warmup_fraction": 0.05, "total_optimizer_steps": total_steps,
        "optimizer_steps_per_epoch": total_steps // args.epochs,
        "warmup_steps": math.ceil(0.05 * total_steps),
        "lr_scheduler_type": "linear", "max_grad_norm": 1.0,
        "parameter_dtype": "torch.float32", "precision": "bf16-if-supported-otherwise-fp16 autocast",
        "gradient_checkpointing": {"use_reentrant": False}, "trainable_parameters_expected": EXPECTED_PARAMETERS,
        "loss": "answer digit plus EOS only; prompt and padding labels -100",
        "preprocess_workers": args.preprocess_workers, "dataloader_num_workers": args.num_workers,
        "multiprocessing_start_method": "spawn", "save_strategy": "epoch", "save_total_limit": args.epochs,
        "chat_template_hash": base.digest_text(tokenizer.chat_template), "model_files_sha256": model_hashes,
        "inputs_sha256": input_hashes, "helper_scripts_sha256": helper_hashes,
        "source_training_files_sha256": {name: base.digest_file(args.source_run / name)
            for name in ("run_config.json", "training_complete.json", "preprocessing_summary.json", "training_manifest.csv")},
        "validation_baseline": anchor, "training_examples": len(manifest), "versions": versions,
        "script_files_sha256": {Path(p).name: base.digest_file(p)
            for p in (__file__, Path(__file__).with_name("train_qwen_full.py"), legacy.__file__)}}
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "run_config.json"
    if config_path.exists():
        require(read_json(config_path) == config, "Immutable configuration differs; choose a new --out")
    else:
        require(not any(args.out.iterdir()), "Output directory is not empty; choose a new --out")
        base.write_json(config_path, config)
    dataset = prepare_artifacts(args, tokenizer, manifest, train_pids, all_pids, parquet_paths,
                                validation_prompts, base.digest_text(base.dumps(config)))
    return dataset, tokenizer, config
