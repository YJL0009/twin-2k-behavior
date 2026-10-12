#!/usr/bin/env python3
"""Full-parameter Qwen2.5-0.5B-Instruct A -> B training on one GPU.

qwen_full_data prepares Qwen-tokenized examples while preserving the source
Liquid run's selected A content and plain target text. All 494,032,768 model
parameters are optimized. No adapter or prior fine-tuned model is initialized.
Validation scoring is a separate program; this trainer reads no held-out labels.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

PROTOCOL = "qwen-nine-binary-a-to-b-full-fixed-content-v1"
EXPECTED_PARAMETERS = 494_032_768


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def config_digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--model", type=Path,
                        default=Path("/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--source-run", type=Path, default=Path("outputs/liquid_lora_v1"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qwen_full_v1"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=32760,
                        help="Safety ceiling only; no source prompt content is truncated")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare and audit Qwen tokenization without loading GPU weights")
    args = parser.parse_args()
    require(min(args.epochs, args.learning_rate, args.micro_batch_size,
                args.gradient_accumulation, args.max_input_tokens) > 0
            and math.isfinite(args.learning_rate), "Training arguments must be positive and finite")
    require(args.preprocess_workers >= 1 and args.num_workers >= 0,
            "--preprocess-workers must be at least 1; --num-workers must be nonnegative")
    require(args.max_input_tokens <= 32760, "Qwen input guard cannot exceed 32,760 tokens")
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1,
            "Launch with plain python on one GPU, not torchrun")
    args.root = args.root.resolve()
    for key in ("model", "source_run", "out"):
        path = getattr(args, key)
        setattr(args, key, (path if path.is_absolute() else args.root / path).resolve())
    require(args.out != args.source_run and args.source_run not in args.out.parents,
            "Use a separate output directory outside the source Liquid run")
    require(args.model != args.out and args.model not in args.out.parents
            and args.out not in args.model.parents,
            "Output and original model directories must be separate")
    return args


class AnswerOnlyCollator:
    """Right-pad inputs and preserve supervision on exactly the digit and EOS."""
    def __init__(self, pad_token_id):
        require(pad_token_id is not None, "A padding token is required")
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples):
        import torch
        size = max(len(row["input_ids"]) for row in examples)
        ids, masks, labels = [], [], []
        for row in examples:
            tokens = list(row["input_ids"])
            attention = list(row["attention_mask"])
            targets = list(row["labels"])
            require(len(tokens) == len(attention) == len(targets)
                    and all(x == -100 for x in targets[:-2])
                    and targets[-2:] == tokens[-2:]
                    and all(x != -100 for x in targets[-2:])
                    and all(x == 1 for x in attention),
                    "Each training example must mask the prompt and supervise only digit + EOS")
            padding = size - len(tokens)
            ids.append(tokens + [self.pad_token_id] * padding)
            masks.append(attention + [0] * padding)
            labels.append(targets + [-100] * padding)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


def step_counts(args, training_examples):
    require(training_examples > 0, "The training dataset is empty")
    batches = math.ceil(training_examples / args.micro_batch_size)
    updates = math.ceil(batches / args.gradient_accumulation)
    total = updates * args.epochs
    return {"optimizer_steps_per_epoch": updates, "total_optimizer_steps": total,
            "warmup_steps": math.ceil(0.05 * total)}


def build_training_arguments(args, bf16, training_examples):
    from transformers import TrainingArguments
    schedule = step_counts(args, training_examples)
    loader = {"dataloader_num_workers": args.num_workers, "dataloader_pin_memory": True,
              "dataloader_persistent_workers": args.num_workers > 0}
    if args.num_workers:
        loader["dataloader_prefetch_factor"] = 2
    return TrainingArguments(output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs, per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, warmup_steps=schedule["warmup_steps"], weight_decay=0.0,
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


def model_files_sha256(folder):
    """Hash inference files separately from optimizer/RNG/trainer state."""
    files = sorted(path for path in Path(folder).iterdir() if path.is_file()
                   and (path.suffix in {".json", ".safetensors", ".model", ".jinja"}
                        or path.name in {"merges.txt", "vocab.txt", "pytorch_model.bin"}))
    return {path.name: digest_file(path) for path in files
            if path.name not in {"trainer_state.json", "training_args.json"}}


def require_checkpoint_state(path):
    path = Path(path)
    require(all((path / name).is_file() for name in
                ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json")),
            f"Incomplete checkpoint; inspect before resuming: {path}")
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    require(weights and all(p.stat().st_size > 0 for p in weights),
            f"Checkpoint weights are missing: {path}")
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (path / filename).exists():
            for relative in set(read_json(path / filename)["weight_map"].values()):
                target = (path / relative).resolve()
                require(path.resolve() in target.parents and target.is_file() and target.stat().st_size > 0,
                        f"Checkpoint weight shard missing: {relative}")


def collect_candidates(out, epochs):
    out = Path(out)
    candidates = []
    for path in sorted((out / "checkpoints").glob("checkpoint-*"),
                       key=lambda p: int(p.name.rsplit("-", 1)[-1])):
        require_checkpoint_state(path)
        state = read_json(path / "trainer_state.json")
        epoch, step = float(state["epoch"]), int(state["global_step"])
        require(math.isclose(epoch, round(epoch), rel_tol=0, abs_tol=1e-6)
                and 1 <= round(epoch) <= epochs and step == int(path.name.rsplit("-", 1)[-1]),
                "Checkpoint is not an expected complete epoch")
        hashes = model_files_sha256(path)
        require("config.json" in hashes and any(name.endswith(".safetensors")
                or name == "pytorch_model.bin" for name in hashes), "Checkpoint inference files missing")
        candidates.append({"name": path.name, "path": str(path.relative_to(out)),
                           "epoch": epoch, "global_step": step, "model_files_sha256": hashes})
    require(len(candidates) == epochs
            and {int(candidate["epoch"]) for candidate in candidates} == set(range(1, epochs + 1)),
            "Expected an intact full-model checkpoint for every epoch")
    return candidates


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.num_workers or args.preprocess_workers:
        method = mp.get_start_method(allow_none=True)
        require(method in (None, "spawn"), "Worker processes require spawn")
        if method is None:
            mp.set_start_method("spawn")
    import qwen_full_data as data
    require(data.PROTOCOL == PROTOCOL and data.EXPECTED_PARAMETERS == EXPECTED_PARAMETERS,
            "Qwen data preparation and trainer versions differ")
    dataset, tokenizer, config = data.prepare_run(args)
    require(config["protocol"] == PROTOCOL and read_json(args.out / "run_config.json") == config,
            "Prepared configuration was not recorded exactly")
    schedule = step_counts(args, len(dataset))
    for key, value in schedule.items():
        require(config[key] == value, f"Prepared training schedule differs: {key}")
    config_hash = config_digest(config)
    if args.prepare_only:
        print(f"Qwen preparation verified: {len(dataset)} A -> B examples; "
              f"{schedule}. No model weights loaded.", flush=True)
        return
    done_path = args.out / "training_complete.json"
    if done_path.exists():
        done = read_json(done_path)
        require(done["protocol"] == PROTOCOL and done["config_hash"] == config_hash,
                "Completed-run configuration mismatch")
        require(collect_candidates(args.out, args.epochs) == done["candidate_checkpoints"],
                "A completed checkpoint changed")
        print("Qwen full fine-tuning already complete:", args.out, flush=True)
        return
    import torch
    from transformers import AutoModelForCausalLM, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    require(torch.cuda.is_available(), "CUDA is required for full fine-tuning")
    set_seed(42)
    bf16 = torch.cuda.is_bf16_supported()
    model_path = Path(config["model"])
    require(model_files_sha256(model_path) == config["model_files_sha256"],
            "Original model/tokenizer files changed after preparation")
    require(not (model_path / "adapter_config.json").exists(), "Initialize from the original Qwen model")
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                                               dtype=torch.float32, attn_implementation="sdpa")
    require(model.config.model_type == "qwen2"
            and args.max_input_tokens + 2 <= model.config.max_position_embeddings,
            "Unexpected model architecture or context length")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    require(total == trainable == EXPECTED_PARAMETERS < 500_000_000
            and all(parameter.dtype == torch.float32 for parameter in model.parameters()),
            "Full-parameter training must optimize all 494,032,768 FP32 parameters")
    model.config.use_cache = False
    print(f"Full fine-tuning: {trainable:,}/{total:,} trainable parameters; FP32 weights, "
          f"{'BF16' if bf16 else 'FP16'} autocast; {schedule}", flush=True)
    checkpoints = args.out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    checkpoint = get_last_checkpoint(str(checkpoints))
    if checkpoint:
        require_checkpoint_state(Path(checkpoint))
        print("Resuming full model and optimizer:", checkpoint, flush=True)
    trainer = finite_trainer_class()(model=model,
        args=build_training_arguments(args, bf16, len(dataset)), train_dataset=dataset,
        data_collator=AnswerOnlyCollator(tokenizer.pad_token_id), processing_class=tokenizer)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    require(math.isfinite(float(result.metrics["train_loss"]))
            and trainer.state.epoch is not None
            and math.isclose(float(trainer.state.epoch), args.epochs, abs_tol=1e-6)
            and trainer.state.global_step == schedule["total_optimizer_steps"],
            "Training did not complete the expected epochs/steps with finite loss")
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    candidates = collect_candidates(args.out, args.epochs)
    require(candidates[-1]["global_step"] == trainer.state.global_step,
            "Last saved checkpoint is not the final training state")
    write_json(done_path, {"protocol": PROTOCOL, "config_hash": config_hash,
        "training_examples": len(dataset), "global_step": trainer.state.global_step,
        "base_parameters": total, "trainable_parameters": trainable, "total_training_parameters": total,
        "parameter_dtype": "torch.float32", "autocast_dtype": "torch.bfloat16" if bf16 else "torch.float16",
        "training_metrics": result.metrics, "candidate_checkpoints": candidates})
    print("Training metrics:", result.metrics, flush=True)
    print("Full fine-tuning complete. Epoch checkpoints:", [c["name"] for c in candidates], flush=True)
    print("Evaluate with evaluate_qwen_full.py; select by validation macro accuracy against C.", flush=True)


if __name__ == "__main__":
    main()
