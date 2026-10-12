#!/usr/bin/env python3
"""Train Liquid on training participants' A -> B with baseline-v3 plain targets.

Run in the Liquid environment: python -u train_liquid_lora.py
Preparation only: python -u train_liquid_lora.py --prepare-only
Smoke run: python -u train_liquid_lora.py --out outputs/liquid_lora_smoke --max-steps 2
Serial CPU work: python -u train_liquid_lora.py --preprocess-workers 1 --num-workers 0
Select complete A units with the original JSON-target token budget, then change
only target presentation to plain text, exactly as in Liquid baseline v3.
Evaluate the merged model separately with evaluate_liquid_lora.py.
This program never loads C or held-out participants' response values.
"""
import argparse
import csv
import importlib.metadata
import json
import multiprocessing as mp
import os
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import run_liquid_baseline as baseline
import run_liquid_plain_baseline as plain

QIDS = baseline.QIDS
require = baseline.require
write_json = baseline.write_json
digest_file = baseline.digest_file


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--model", type=Path,
                        default=Path("/root/autodl-tmp/models/LFM2.5-350M"))
    parser.add_argument("--out", type=Path, default=Path("outputs/liquid_lora_v1"))
    parser.add_argument("--baseline-run", type=Path, default=Path("outputs/liquid_plain_target_v3"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--preprocess-workers", type=int, default=4,
                        help="CPU processes preparing participant prompts; 1 runs serially")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Training DataLoader CPU workers; 0 loads in the main process")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    require(min(args.epochs, args.micro_batch_size, args.gradient_accumulation,
                args.learning_rate, args.max_input_tokens) > 0, "Arguments must be positive")
    require(args.max_steps == -1 or args.max_steps > 0, "--max-steps must be -1 or positive")
    require(args.preprocess_workers >= 1, "--preprocess-workers must be at least 1")
    require(args.num_workers >= 0, "--num-workers must be nonnegative")
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1,
            "Use plain python on one GPU for this controlled run")
    args.root, args.model = args.root.resolve(), args.model.resolve()
    args.out = (args.out if args.out.is_absolute() else args.root / args.out).resolve()
    args.baseline_run = (args.baseline_run if args.baseline_run.is_absolute()
                         else args.root / args.baseline_run).resolve()
    return args


def read_train_labels(split_path, response_path):
    """Read all IDs, then skip non-training CSV rows BEFORE loading answer columns."""
    import pandas as pd
    split = pd.read_csv(split_path, dtype={"pid": "string"})
    require(split.pid.notna().all() and split.pid.is_unique, "Invalid split IDs")
    require(split["split"].value_counts().to_dict() ==
            {"train": 1440, "validation": 309, "test": 309}, "Unexpected split sizes")
    require(split.set_index("pid").loc["574", "split"] == "train", "574 must remain in training")
    train_pids = set(split.loc[split["split"] == "train", "pid"])
    ids = pd.read_csv(response_path, usecols=["pid"], dtype={"pid": "string"})["pid"]
    require(ids.notna().all() and ids.is_unique and set(ids) == set(split.pid),
            "Response CSV and split participant IDs differ")
    skipped_rows = [index + 1 for index, pid in enumerate(ids) if pid not in train_pids]
    labels = pd.read_csv(response_path, skiprows=skipped_rows,
                         usecols=["pid", *QIDS], dtype={"pid": "string"}).set_index("pid")
    require(labels.index.is_unique and set(labels.index) == train_pids,
            "Training-only label selection failed")
    for qid in QIDS:
        answers = labels[qid].dropna()
        require(len(answers) > 0 and answers.isin([1, 2]).all(), f"Invalid B labels: {qid}")
    return labels, train_pids, set(split.pid)


def encode_supervised(tokenizer, prompt, label):
    """Check chat boundaries, then supervise precisely the answer digit and EOS."""
    prefix = tokenizer.apply_chat_template(prompt["messages"], tokenize=True,
                                           add_generation_prompt=True, return_dict=False,
                                           truncation=False)
    completed = tokenizer.apply_chat_template(
        prompt["messages"] + [{"role": "assistant", "content": str(label)}],
        tokenize=True, add_generation_prompt=False, return_dict=False, truncation=False)
    require(isinstance(prefix, list) and isinstance(completed, list)
            and all(isinstance(token, int) for token in prefix + completed),
            "Chat template must return flat token ID lists")
    require(completed[:len(prefix)] == prefix, "Assistant template changed the prompt prefix")
    digit = tokenizer.encode(str(label), add_special_tokens=False)
    require(len(digit) == 1 and label in [1, 2], "Expected a single answer digit token")
    suffix = digit + [tokenizer.eos_token_id]
    require(completed[len(prefix):len(prefix) + 2] == suffix,
            "Assistant response must begin with digit + EOS")
    tail = tokenizer.decode(completed[len(prefix) + 2:], skip_special_tokens=False)
    require(not tail.strip(), "Unexpected template content after assistant EOS")
    require(len(prefix) == prompt["input_tokens"], "Prompt token count changed")
    return {"input_ids": prefix + suffix,
            "attention_mask": [1] * (len(prefix) + 2),
            "labels": [-100] * len(prefix) + suffix}


class AnswerOnlyCollator:
    """Right-pad inputs; never replace masked labels with input tokens."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        import torch
        size = max(len(row["input_ids"]) for row in examples)
        ids, masks, labels = [], [], []
        for row in examples:
            padding = size - len(row["input_ids"])
            require(sum(token != -100 for token in row["labels"]) == 2,
                    "Each example must supervise only digit + EOS")
            ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            masks.append(row["attention_mask"] + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


_WORKER_TOKENIZER = None
_WORKER_BUDGET = None


def init_preprocess_worker(model_path, budget):
    """Each spawned CPU process loads one tokenizer, never a GPU model."""
    global _WORKER_TOKENIZER, _WORKER_BUDGET
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, padding_side="right")
    _WORKER_BUDGET = budget


def prepare_participant(job, tokenizer, budget):
    """Pure per-participant preparation; preserve baseline question/unit order."""
    row, targets = job
    pid = str(row["pid"])
    a_units = baseline.survey_units(row["wave1_3_persona_json"], True)
    b_metadata = baseline.survey_units(row["wave4_Q_wave1_3_A"], False)
    rendered = baseline.render_a_units(tokenizer, a_units)
    examples = []
    for qid, label in targets:
        target = baseline.target_for(b_metadata, qid)
        source_prompt = baseline.prepare_prompt(tokenizer, pid, qid, a_units, target, budget, rendered)
        prompt = plain.prepare_plain_prompt(tokenizer, source_prompt, budget)
        encoded = encode_supervised(tokenizer, prompt, label)
        record = {key: prompt[key] for key in
                  ["pid", "qid", "prompt_hash", "source_prompt_hash", "input_tokens",
                   "source_input_tokens", "a_units_total", "a_units_selected"]}
        record.update(label_b=label, total_tokens=len(encoded["input_ids"]),
                      a_unit_indices=json.dumps(prompt["a_unit_indices"]))
        examples.append((record, encoded))
    return pid, examples


def prepare_worker_job(job):
    require(_WORKER_TOKENIZER is not None, "Preprocessing worker was not initialized")
    return prepare_participant(job, _WORKER_TOKENIZER, _WORKER_BUDGET)


def ordered_bounded_map(executor, function, jobs, limit):
    """Consume futures in submission order with at most `limit` jobs in flight."""
    require(limit > 0, "In-flight job limit must be positive")
    iterator, pending = iter(jobs), deque()
    for _ in range(limit):
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        yield pending.popleft().result()
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            pass


def prepared_participants(args, tokenizer, jobs):
    if args.preprocess_workers == 1:
        for job in jobs:
            yield prepare_participant(job, tokenizer, args.max_input_tokens)
        return
    with ProcessPoolExecutor(max_workers=args.preprocess_workers,
            mp_context=mp.get_context("spawn"), initializer=init_preprocess_worker,
            initargs=(str(args.model), args.max_input_tokens)) as executor:
        yield from ordered_bounded_map(executor, prepare_worker_job, jobs,
                                       limit=2 * args.preprocess_workers)


def prepare_training(args, tokenizer, labels, train_pids, all_pids, parquet_paths, config_hash):
    """Stream participant-by-participant into an Arrow file, avoiding a Python token corpus."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds
    from datasets import Dataset, Features, Sequence, Value
    prepared = args.out / "prepared"
    prepared.mkdir(exist_ok=True)
    arrow_path = prepared / "train.arrow"
    manifest_path = args.out / "training_manifest.csv"
    summary_path = args.out / "preprocessing_summary.json"
    audit_path = args.out / "mask_audit.json"
    artifacts = [arrow_path, manifest_path, audit_path]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        require(summary["config_hash"] == config_hash, "Prepared training configuration differs")
        for path in artifacts:
            require(path.exists() and digest_file(path) == summary["artifacts_sha256"][path.name],
                    f"Prepared artifact missing or changed: {path}")
        dataset = Dataset.from_file(str(arrow_path))
        require(len(dataset) == summary["examples"], "Prepared example count mismatch")
        print(f"Reusing {len(dataset)} prepared training examples.", flush=True)
        return dataset
    # Failed preparation leaves only reproducible temporary files. Completed
    # artifacts without their completion marker are safely rebuilt in this run.
    source = ds.dataset([str(p) for p in parquet_paths], format="parquet")
    source_ids = source.to_table(columns=["pid"])["pid"].to_pylist()
    require(len(source_ids) == len(set(map(str, source_ids))) == 2058 and
            set(map(str, source_ids)) == all_pids, "Parquet/split ID mismatch")
    fields = ["pid", "wave1_3_persona_json", "wave4_Q_wave1_3_A"]
    scanner = source.scanner(columns=fields, batch_size=1,
        filter=ds.field("pid").cast(pa.string()).isin(sorted(train_pids)))

    def training_jobs():
        for batch in scanner.to_batches():
            for row in batch.to_pylist():
                pid = str(row["pid"])
                require(pid in train_pids, "Unexpected training participant")
                targets = [(qid, int(labels.at[pid, qid])) for qid in QIDS
                           if not pd.isna(labels.at[pid, qid])]
                yield row, targets

    features = Features({"input_ids": Sequence(Value("int32")),
                         "attention_mask": Sequence(Value("int8")),
                         "labels": Sequence(Value("int32"))})
    columns = ["pid", "qid", "label_b", "prompt_hash", "source_prompt_hash",
               "input_tokens", "source_input_tokens", "total_tokens",
               "a_units_total", "a_units_selected", "a_unit_indices"]
    arrow_tmp, manifest_tmp = arrow_path.with_suffix(".arrow.tmp"), manifest_path.with_suffix(".csv.tmp")
    counts, audit, seen, token_counts, selected_counts = Counter(), [], set(), [], []
    buffer, example_count = [], 0
    with pa.OSFile(str(arrow_tmp), "wb") as sink, manifest_tmp.open("w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=columns)
        writer.writeheader()
        with pa.ipc.new_stream(sink, features.arrow_schema) as arrow_writer:
            for pid, examples in prepared_participants(args, tokenizer, training_jobs()):
                require(pid in train_pids and pid not in seen, "Unexpected/duplicate training participant")
                seen.add(pid)
                for record, encoded in examples:
                    buffer.append(encoded)
                    writer.writerow(record)
                    counts[(record["qid"], record["label_b"])] += 1
                    token_counts.append(record["input_tokens"])
                    selected_counts.append(record["a_units_selected"])
                    example_count += 1
                    if len(audit) < 2:
                        audit.append({"pid": pid, "qid": record["qid"], "label_b": record["label_b"],
                            "prompt_tokens_masked": record["input_tokens"],
                            "supervised_token_ids": encoded["labels"][-2:],
                            "supervised_text": tokenizer.decode(encoded["labels"][-2:], skip_special_tokens=False),
                            "prompt_tail": tokenizer.decode(encoded["input_ids"][-42:-2], skip_special_tokens=False)})
                    if len(buffer) == 16:
                        arrow_writer.write_table(pa.Table.from_pylist(buffer, schema=features.arrow_schema))
                        buffer.clear()
                if len(seen) % 50 == 0:
                    print(f"Prepared {len(seen)}/1440 participants, {example_count} examples", flush=True)
            if buffer:
                arrow_writer.write_table(pa.Table.from_pylist(buffer, schema=features.arrow_schema))
    require(seen == train_pids, "Missing training participants")
    require(example_count == int(labels.notna().sum().sum()), "Training example coverage mismatch")
    arrow_tmp.replace(arrow_path)
    manifest_tmp.replace(manifest_path)
    write_json(audit_path, audit)
    summary = {"config_hash": config_hash, "participants": len(seen), "examples": example_count,
        "training_labels": "B", "context": "A only; JSON-budget selection followed by fixed-A plain-target conversion",
        "counts_by_question_label": {qid: {str(label): counts[(qid, label)] for label in [1, 2]} for qid in QIDS},
        "input_tokens": {"min": min(token_counts), "max": max(token_counts), "mean": sum(token_counts) / example_count},
        "a_units_selected": {"min": min(selected_counts), "max": max(selected_counts), "mean": sum(selected_counts) / example_count},
        "artifacts_sha256": {path.name: digest_file(path) for path in artifacts}}
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("Answer-mask audit:", json.dumps(audit, indent=2), flush=True)
    return Dataset.from_file(str(arrow_path))


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.num_workers > 0:
        # Trainer constructs DataLoaders after CUDA initialization. Spawn workers
        # instead of forking an initialized CUDA runtime; never override a caller.
        start_method = mp.get_start_method(allow_none=True)
        require(start_method in (None, "spawn"), "DataLoader workers require the spawn start method")
        if start_method is None:
            mp.set_start_method("spawn")
    import torch
    from transformers import AutoTokenizer
    split_path = args.root / "data/participant_split.csv"
    response_path = args.root / "data/Twin-2K-500/question_catalog_and_human_response_csv/wave1_3_response.csv"
    parquet_paths = sorted((args.root / "data/Twin-2K-500/wave_split/chunks").glob("*.parquet"))
    require(len(parquet_paths) == 7, "Expected seven dataset chunks")
    require((args.model / "config.json").exists(), "Local model config missing")
    reference_paths = {"config_sha256": args.baseline_run / "config.json",
                       "prepared_prompts_sha256": args.baseline_run / "prepared_prompts.jsonl",
                       "summary_sha256": args.baseline_run / "summary.json"}
    require(all(path.exists() for path in reference_paths.values()),
            "Complete Liquid plain-target validation baseline v3 before training")
    reference_config = json.loads(reference_paths["config_sha256"].read_text())
    reference_summary = json.loads(reference_paths["summary_sha256"].read_text())
    require(reference_config.get("protocol") == "liquid-nine-binary-fixed-a-plain-target-v3",
            "Expected the controlled Liquid plain-target baseline v3")
    require(reference_summary.get("examples") == 1545 and reference_summary.get("invalid_outputs") == 0
            and reference_summary.get("invalid_rate") == 0.0
            and reference_summary.get("evaluation_split") == "validation"
            and reference_summary.get("question_subset") == list(QIDS),
            "Reference baseline must complete all 1545 validation examples with 100% valid outputs")
    require(args.max_input_tokens == reference_config["max_input_tokens"] == 8192,
            "Keep the established baseline input budget")
    require(reference_config["inputs_sha256"].get(str(split_path)) == digest_file(split_path),
            "Participant split changed since the validation baseline")
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, padding_side="right")
    require(tokenizer.eos_token_id is not None, "Tokenizer EOS missing")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_files = sorted(p for p in args.model.iterdir() if p.is_file() and
                         (p.suffix in {".json", ".safetensors", ".model", ".jinja"} or p.name == "merges.txt"))
    model_hashes = {path.name: digest_file(path) for path in model_files}
    require(model_hashes == reference_config["model_files_sha256"],
            "Initial model/tokenizer files differ from the recorded Liquid baseline")
    helper_hashes = {Path(path).name: digest_file(path) for path in
                     (baseline.__file__, plain.__file__, plain.diag.__file__)}
    require(helper_hashes == reference_config["script_files_sha256"],
            "Prompt helper scripts changed since the validation baseline")
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "transformers", "tokenizers", "datasets", "accelerate", "peft", "pandas", "pyarrow"]}
    for name in ("torch", "transformers", "tokenizers"):
        require(versions[name] == reference_config["versions"][name],
                f"Use the same {name} version as the recorded Liquid baseline")
    config = {"protocol": "liquid-nine-binary-a-to-b-lora-plain-target-v1", "root": str(args.root),
        "model": str(args.model), "qids": list(QIDS), "system": baseline.SYSTEM,
        "context_policy": "JSON-target-budget whole-A-unit selection; then plain-target conversion without refill",
        "max_input_tokens": args.max_input_tokens, "epochs": args.epochs, "max_steps": args.max_steps,
        "micro_batch_size": args.micro_batch_size, "gradient_accumulation": args.gradient_accumulation,
        "preprocess_workers": args.preprocess_workers, "preprocess_start_method": "spawn",
        "dataloader_num_workers": args.num_workers, "dataloader_pin_memory": True,
        "dataloader_persistent_workers": args.num_workers > 0,
        "dataloader_prefetch_factor": 2 if args.num_workers > 0 else None,
        "learning_rate": args.learning_rate, "seed": 42, "warmup_fraction": 0.05,
        "warmup_steps_argument": 0.05,
        "optimizer": "adamw_torch", "weight_decay": 0.0, "max_grad_norm": 1.0,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "targets": ["q_proj", "v_proj"], "bias": "none"},
        "gradient_checkpointing": {"use_reentrant": False}, "precision": "bf16-if-supported-otherwise-fp16",
        "loss": "answer digit plus EOS only; prompt and padding labels -100",
        "chat_template_hash": baseline.digest_text(tokenizer.chat_template),
        "inputs_sha256": {str(path): digest_file(path) for path in [split_path, response_path, *parquet_paths]},
        "model_files_sha256": model_hashes,
        "script_sha256": digest_file(__file__), "helper_scripts_sha256": helper_hashes,
        "validation_baseline": {"path": str(args.baseline_run),
                                **{key: digest_file(path) for key, path in reference_paths.items()}},
        "versions": versions}
    config_path = args.out / "run_config.json"
    if config_path.exists():
        require(json.loads(config_path.read_text()) == config,
                "Existing training configuration differs. Choose a new --out; nothing was overwritten.")
    else:
        require(not any(args.out.iterdir()), "Output directory is not empty; choose a new --out")
        write_json(config_path, config)
    config_hash = baseline.digest_text(baseline.dumps(config))
    labels, train_pids, all_pids = read_train_labels(split_path, response_path)
    dataset = prepare_training(args, tokenizer, labels, train_pids, all_pids, parquet_paths, config_hash)
    if args.prepare_only:
        print("Preparation complete. No training or validation scoring performed.")
        return
    done_path = args.out / "training_complete.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        require(done["config_hash"] == config_hash, "Completed-run configuration mismatch")
        for relative, digest in done["artifacts_sha256"].items():
            path = args.out / relative
            require(path.exists() and digest_file(path) == digest, f"Completed artifact changed: {relative}")
        print("Training already complete. Merged model:", args.out / "merged")
        return
    require(torch.cuda.is_available(), "CUDA is required for training")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    set_seed(42)
    bf16 = torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
        dtype=torch.bfloat16 if bf16 else torch.float16, attn_implementation="sdpa")
    require(args.max_input_tokens + 2 <= model.config.max_position_embeddings, "Training context too long")
    base_params = sum(p.numel() for p in model.parameters())
    require(model.config.model_type == "lfm2" and base_params == 354_483_968,
            "Expected the native LFM2.5-350M base model")
    require(model.config.layer_types.count("full_attention") == 6, "Expected six attention layers")
    projections = {name: module for name, module in model.named_modules()
                   if name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}}
    require(Counter(name.rsplit(".", 1)[-1] for name in projections) == {"q_proj": 6, "v_proj": 6},
            "Unexpected LoRA target modules")
    for name, module in projections.items():
        require(module.in_features == 1024 and module.out_features == (1024 if name.endswith("q_proj") else 512),
                f"Unexpected projection shape: {name}")
    expected_adapter_params = 8 * sum(module.in_features + module.out_features for module in projections.values())
    require(expected_adapter_params == 172_032, "Unexpected rank-8 adapter size")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none"))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(total_params == 354_656_000 < 500_000_000 and trainable_params == expected_adapter_params,
            "Unexpected adapter parameter count")
    model.print_trainable_parameters()
    checkpoints = args.out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    checkpoint = get_last_checkpoint(str(checkpoints))
    if checkpoint:
        require((Path(checkpoint) / "trainer_state.json").exists() and
                (Path(checkpoint) / "optimizer.pt").exists(),
                "Latest checkpoint is incomplete; inspect it before resuming")
        print("Resuming:", checkpoint, flush=True)
    loader_kwargs = {"dataloader_num_workers": args.num_workers,
                     "dataloader_pin_memory": True,
                     "dataloader_persistent_workers": args.num_workers > 0}
    if args.num_workers > 0:
        loader_kwargs["dataloader_prefetch_factor"] = 2
    training_args = TrainingArguments(output_dir=str(checkpoints),
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, warmup_steps=0.05, weight_decay=0.0,
        optim="adamw_torch", lr_scheduler_type="linear", max_grad_norm=1.0,
        bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no", save_strategy="steps", save_steps=100, save_total_limit=2,
        logging_strategy="steps", logging_steps=10, logging_first_step=True,
        seed=42, data_seed=42, report_to=[], **loader_kwargs,
        remove_unused_columns=False, label_names=["labels"])
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=AnswerOnlyCollator(tokenizer.pad_token_id), processing_class=tokenizer)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    adapter_dir = args.out / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    merged = model.merge_and_unload(safe_merge=True).eval()
    require(sum(p.numel() for p in merged.parameters()) == base_params, "Merged model size changed")
    merged_dir = args.out / "merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    saved_tokenizer = AutoTokenizer.from_pretrained(merged_dir, local_files_only=True)
    require(saved_tokenizer.chat_template == tokenizer.chat_template, "Saved chat template changed")
    artifacts = [path for folder in [adapter_dir, merged_dir] for path in sorted(folder.iterdir()) if path.is_file()]
    write_json(done_path, {"config_hash": config_hash, "training_examples": len(dataset),
        "global_step": trainer.state.global_step, "base_parameters": base_params,
        "adapter_parameters": trainable_params, "total_training_parameters": total_params,
        "dtype": str(merged.dtype), "training_metrics": result.metrics,
        "artifacts_sha256": {str(path.relative_to(args.out)): digest_file(path) for path in artifacts}})
    print("Training complete. Merged model:", merged_dir, flush=True)
    print("Evaluate separately with evaluate_liquid_lora.py, using the frozen plain-target baseline prompts.", flush=True)


if __name__ == "__main__":
    main()
