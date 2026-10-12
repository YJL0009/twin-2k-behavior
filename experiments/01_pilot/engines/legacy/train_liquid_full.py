#!/usr/bin/env python3
"""Full-parameter Liquid A -> B fine-tuning from original pretrained weights.

Reuse the exact, audited training cache from liquid_lora_v1. No LoRA/merged
weights are loaded. All 354,483,968 parameters are optimized in FP32 with CUDA
mixed precision. Default: three epochs, save every epoch for separate validation
selection. This program never reads held-out answer values or test responses.
Run in the existing liquid_tf52 environment with plain python on one GPU.
"""
import argparse
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import run_liquid_baseline as base
import run_liquid_plain_baseline as plain
from diagnose_liquid_fit import prefix_from_training_example
from train_liquid_lora import AnswerOnlyCollator

PROTOCOL = "liquid-nine-binary-a-to-b-full-plain-target-v1"
EXPECTED_PARAMETERS = 354_483_968
require = base.require


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--source-run", type=Path, default=Path("outputs/liquid_lora_v1"))
    parser.add_argument("--out", type=Path, default=Path("outputs/liquid_full_v1"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Validate all cache rows and write configuration; do not load/train a model")
    args = parser.parse_args()
    require(min(args.epochs, args.learning_rate, args.micro_batch_size,
                args.gradient_accumulation) > 0 and math.isfinite(args.learning_rate),
            "Epochs, learning rate, batch size and accumulation must be positive")
    require(args.num_workers >= 0, "--num-workers must be nonnegative")
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "Run plain python on one GPU")
    args.root = args.root.resolve()
    for key in ("source_run", "out"):
        value = getattr(args, key)
        setattr(args, key, (value if value.is_absolute() else args.root / value).resolve())
    require(args.out != args.source_run and args.source_run not in args.out.parents,
            "Use a separate output directory, outside the source LoRA run")
    return args


def read_json(path):
    return json.loads(Path(path).read_text())


def model_files_sha256(folder):
    """Inference artifacts only; optimizer/RNG/trainer states are deliberately excluded."""
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.is_file() and
                   (p.suffix in {".json", ".safetensors", ".model", ".jinja"}
                    or p.name in {"merges.txt", "vocab.txt", "pytorch_model.bin"}))
    return {p.name: base.digest_file(p) for p in files
            if p.name not in {"trainer_state.json", "training_args.json"}}


def require_checkpoint_state(path):
    """Never silently restart from a partial checkpoint."""
    path = Path(path)
    require(all((path / name).is_file() for name in
                ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json")),
            f"Checkpoint state is incomplete; inspect before resuming: {path}")
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    require(weights and all(p.stat().st_size > 0 for p in weights), f"Checkpoint weights missing: {path}")
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (path / filename).exists():
            for relative in set(read_json(path / filename)["weight_map"].values()):
                target = (path / relative).resolve()
                require(path.resolve() in target.parents and target.is_file() and target.stat().st_size > 0,
                        f"Checkpoint weight shard missing: {relative}")


def load_verified_training_cache(args, tokenizer, source, completion):
    """Verify provenance and every B digit/EOS mask without reopening response CSVs."""
    import pandas as pd
    from datasets import Dataset
    source_hash = base.digest_text(base.dumps(source))
    require(source["protocol"] == "liquid-nine-binary-a-to-b-lora-plain-target-v1"
            and completion["config_hash"] == source_hash,
            "Source LoRA completion/configuration mismatch")
    require(completion["base_parameters"] == EXPECTED_PARAMETERS
            and completion["total_training_parameters"] < 500_000_000, "Unexpected source model")
    prep_path = args.source_run / "preprocessing_summary.json"
    prep = read_json(prep_path)
    require(prep["config_hash"] == source_hash and prep["training_labels"] == "B",
            "Source cache is not the recorded A -> B training cache")
    cache_paths = {"train.arrow": args.source_run / "prepared/train.arrow",
                   "training_manifest.csv": args.source_run / "training_manifest.csv",
                   "mask_audit.json": args.source_run / "mask_audit.json"}
    for name, path in cache_paths.items():
        require(path.is_file() and base.digest_file(path) == prep["artifacts_sha256"][name],
                f"Source cache artifact changed: {name}")
    split_path = args.root / "data/participant_split.csv"
    require(base.digest_file(split_path) == source["inputs_sha256"][str(split_path)],
            "Participant split differs from the source training run")
    split = pd.read_csv(split_path, dtype={"pid": "string"})
    require(split.pid.notna().all() and split.pid.is_unique and
            split["split"].value_counts().to_dict() == {"train": 1440, "validation": 309, "test": 309},
            "Invalid participant split")
    require(split.set_index("pid").loc["574", "split"] == "train", "574 must remain in training")
    manifest = pd.read_csv(cache_paths["training_manifest.csv"], dtype={"pid": "string", "qid": "string"})
    dataset = Dataset.from_file(str(cache_paths["train.arrow"]))
    require(0 < len(dataset) == len(manifest) == prep["examples"] == completion["training_examples"],
            "Source training cache/manifest counts differ")
    require(not manifest[["pid", "qid"]].duplicated().any()
            and set(manifest.pid) == set(split.loc[split["split"].eq("train"), "pid"])
            and set(manifest.qid) == set(base.QIDS) and manifest.label_b.isin([1, 2]).all(),
            "Training manifest contains invalid IDs or B labels")
    require(prep["participants"] == 1440 and manifest.input_tokens.between(1, source["max_input_tokens"]).all(),
            "Training participant count/token budget differs")
    counts = {qid: {str(label): int((manifest.qid.eq(qid) & manifest.label_b.eq(label)).sum())
                    for label in (1, 2)} for qid in base.QIDS}
    require(counts == prep["counts_by_question_label"], "Recorded training B counts differ")
    records = manifest.to_dict("records")
    for index, row in enumerate(dataset):
        prefix_from_training_example(row, records[index], tokenizer)
        if (index + 1) % 1000 == 0:
            print(f"Verified training masks and B alignment: {index + 1}/{len(dataset)}", flush=True)
    print(f"Verified all {len(dataset)} A -> B examples from 1440 training participants.", flush=True)
    anchor = {"path": str(args.source_run),
              "run_config_sha256": base.digest_file(args.source_run / "run_config.json"),
              "training_complete_sha256": base.digest_file(args.source_run / "training_complete.json"),
              "preprocessing_summary_sha256": base.digest_file(prep_path),
              "cache_files_sha256": {name: base.digest_file(path) for name, path in cache_paths.items()}}
    return dataset, anchor


def build_training_arguments(args, bf16):
    from transformers import TrainingArguments
    loader = {"dataloader_num_workers": args.num_workers, "dataloader_pin_memory": True,
              "dataloader_persistent_workers": args.num_workers > 0}
    if args.num_workers:
        loader["dataloader_prefetch_factor"] = 2
    return TrainingArguments(output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs, per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, warmup_steps=0.05, weight_decay=0.0,
        optim="adamw_torch", lr_scheduler_type="linear", max_grad_norm=1.0,
        bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no", save_strategy="epoch", save_total_limit=args.epochs,
        logging_strategy="steps", logging_steps=10, logging_first_step=True,
        logging_nan_inf_filter=False, seed=42, data_seed=42, report_to=[],
        remove_unused_columns=False, label_names=["labels"], **loader)


def finite_trainer_class():
    from transformers import Trainer
    import torch

    class FiniteLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                          num_items_in_batch=num_items_in_batch)
            loss = result[0] if return_outputs else result
            require(bool(torch.isfinite(loss.detach()).all()),
                    "Non-finite loss; stopped without marking training complete")
            return result

    return FiniteLossTrainer


def collect_candidates(out, epochs):
    """Require one intact full-model checkpoint for each completed epoch."""
    candidates = []
    for path in sorted((out / "checkpoints").glob("checkpoint-*"),
                       key=lambda p: int(p.name.rsplit("-", 1)[-1])):
        require_checkpoint_state(path)
        state = read_json(path / "trainer_state.json")
        epoch = float(state["epoch"])
        step = int(state["global_step"])
        require(math.isclose(epoch, round(epoch), rel_tol=0, abs_tol=1e-6)
                and 1 <= round(epoch) <= epochs and step == int(path.name.rsplit("-", 1)[-1]),
                "Checkpoint is not an expected complete epoch")
        hashes = model_files_sha256(path)
        require("config.json" in hashes and any(name.endswith(".safetensors") or name == "pytorch_model.bin"
                                                for name in hashes), "Checkpoint inference files missing")
        candidates.append({"name": path.name, "path": str(path.relative_to(out)),
                           "epoch": epoch, "global_step": step, "model_files_sha256": hashes})
    require(len(candidates) == epochs and {int(c["epoch"]) for c in candidates} == set(range(1, epochs + 1)),
            "Expected an intact checkpoint for every epoch")
    return candidates


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.num_workers:
        method = mp.get_start_method(allow_none=True)
        require(method in (None, "spawn"), "Training DataLoader workers require spawn")
        if method is None:
            mp.set_start_method("spawn")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    source = read_json(args.source_run / "run_config.json")
    completion = read_json(args.source_run / "training_complete.json")
    model_path = Path(source["model"]).resolve()
    require(model_path != (args.source_run / "merged").resolve()
            and model_path != args.out and args.source_run not in model_path.parents,
            "Initialize from the original pretrained model, not a trained checkpoint")
    require(model_files_sha256(model_path) == source["model_files_sha256"],
            "Original model/tokenizer changed since the source baseline")
    helpers = {Path(p).name: base.digest_file(p) for p in
               (base.__file__, plain.__file__, plain.diag.__file__)}
    require(helpers == source["helper_scripts_sha256"], "Original prompt helper scripts changed")
    versions = {name: importlib.metadata.version(name) for name in source["versions"]}
    for name in ("torch", "transformers", "tokenizers"):
        require(versions[name] == source["versions"][name], f"Use the recorded Liquid {name} version")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="right")
    require(tokenizer.eos_token_id is not None and
            base.digest_text(tokenizer.chat_template) == source["chat_template_hash"],
            "Original tokenizer/chat template changed")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset, cache_anchor = load_verified_training_cache(args, tokenizer, source, completion)
    baseline_anchor = source["validation_baseline"]
    for filename, key in (("config.json", "config_sha256"), ("prepared_prompts.jsonl", "prepared_prompts_sha256"),
                          ("summary.json", "summary_sha256")):
        require(base.digest_file(Path(baseline_anchor["path"]) / filename) == baseline_anchor[key],
                "Frozen validation anchor changed")
    import diagnose_liquid_fit as fit
    import train_liquid_lora as lora
    require(base.digest_file(lora.__file__) == source["script_sha256"],
            "Original answer-only training helper changed since the source run")
    config = {"protocol": PROTOCOL, "root": str(args.root), "source_run": str(args.source_run),
        "model": str(model_path),
        "initialization": "original pretrained weights; no adapter or prior fine-tuned weights",
        "qids": list(base.QIDS), "system": source["system"], "context_policy": source["context_policy"],
        "max_input_tokens": source["max_input_tokens"], "epochs": args.epochs,
        "micro_batch_size": args.micro_batch_size, "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate, "seed": 42, "warmup_fraction": 0.05,
        "warmup_steps_argument": 0.05, "optimizer": "adamw_torch", "weight_decay": 0.0,
        "max_grad_norm": 1.0, "lr_scheduler_type": "linear",
        "gradient_checkpointing": {"use_reentrant": False},
        "parameter_dtype": "torch.float32", "precision": "bf16-if-supported-otherwise-fp16 autocast",
        "loss": source["loss"], "trainable_parameters_expected": EXPECTED_PARAMETERS,
        "save_strategy": "epoch", "save_total_limit": args.epochs,
        "dataloader_num_workers": args.num_workers, "dataloader_start_method": "spawn",
        "dataloader_pin_memory": True, "dataloader_persistent_workers": args.num_workers > 0,
        "dataloader_prefetch_factor": 2 if args.num_workers else None,
        "chat_template_hash": source["chat_template_hash"],
        "inputs_sha256": source["inputs_sha256"], "source_training_cache": cache_anchor,
        "model_files_sha256": source["model_files_sha256"], "helper_scripts_sha256": helpers,
        "reused_training_helpers_sha256": {Path(p).name: base.digest_file(p) for p in (fit.__file__, lora.__file__)},
        "validation_baseline": baseline_anchor, "script_sha256": base.digest_file(__file__), "versions": versions}
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "run_config.json"
    if config_path.exists():
        require(read_json(config_path) == config,
                "Immutable run configuration differs; choose a new --out. Nothing was overwritten.")
    else:
        require(not any(args.out.iterdir()), "Output directory is not empty; choose a new --out")
        base.write_json(config_path, config)
    config_hash = base.digest_text(base.dumps(config))
    if args.prepare_only:
        print("Full fine-tuning configuration and all training masks verified. No model loaded.", flush=True)
        return
    done_path = args.out / "training_complete.json"
    if done_path.exists():
        done = read_json(done_path)
        require(done["protocol"] == PROTOCOL and done["config_hash"] == config_hash,
                "Completed-run configuration mismatch")
        require(collect_candidates(args.out, args.epochs) == done["candidate_checkpoints"],
                "A completed checkpoint changed")
        print("Full fine-tuning already complete:", args.out, flush=True)
        return
    require(torch.cuda.is_available(), "CUDA is required for full fine-tuning")
    set_seed(42)
    bf16 = torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                                               dtype=torch.float32, attn_implementation="sdpa")
    require(model.config.model_type == "lfm2" and
            source["max_input_tokens"] + 2 <= model.config.max_position_embeddings,
            "Unexpected model architecture/context length")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(total == trainable == EXPECTED_PARAMETERS < 500_000_000
            and all(p.dtype == torch.float32 for p in model.parameters()),
            "Full-parameter training must optimize all 354,483,968 FP32 parameters")
    model.config.use_cache = False
    print(f"Full fine-tuning: {trainable:,}/{total:,} trainable parameters; "
          f"FP32 weights, {'BF16' if bf16 else 'FP16'} autocast", flush=True)
    checkpoints = args.out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    checkpoint = get_last_checkpoint(str(checkpoints))
    if checkpoint:
        require_checkpoint_state(Path(checkpoint))
        print("Resuming full model and optimizer:", checkpoint, flush=True)
    training_args = build_training_arguments(args, bf16)
    trainer = finite_trainer_class()(model=model, args=training_args, train_dataset=dataset,
                                    data_collator=AnswerOnlyCollator(tokenizer.pad_token_id),
                                    processing_class=tokenizer)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    require(math.isfinite(float(result.metrics["train_loss"]))
            and trainer.state.epoch is not None
            and math.isclose(float(trainer.state.epoch), args.epochs, abs_tol=1e-6),
            "Training did not complete the requested epochs with finite loss")
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    candidates = collect_candidates(args.out, args.epochs)
    require(candidates[-1]["global_step"] == trainer.state.global_step,
            "Last saved checkpoint is not the final training state")
    base.write_json(done_path, {"protocol": PROTOCOL, "config_hash": config_hash,
        "training_examples": len(dataset), "global_step": trainer.state.global_step,
        "base_parameters": total, "trainable_parameters": trainable, "total_training_parameters": total,
        "parameter_dtype": "torch.float32", "autocast_dtype": "torch.bfloat16" if bf16 else "torch.float16",
        "training_metrics": result.metrics, "candidate_checkpoints": candidates})
    print("Training metrics:", result.metrics, flush=True)
    print("Full fine-tuning complete. Epoch checkpoints:", [c["name"] for c in candidates], flush=True)
    print("Evaluate with evaluate_liquid_full.py; select by validation macro accuracy against C.", flush=True)


if __name__ == "__main__":
    main()
