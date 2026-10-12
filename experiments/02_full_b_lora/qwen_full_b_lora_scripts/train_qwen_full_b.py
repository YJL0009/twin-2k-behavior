#!/usr/bin/env python3
"""Full-parameter Qwen training on every valid training-participant B item.

Initializes the original 494M model. All prompts use A plus answer-free target
metadata; only training B answer tokens and EOS receive loss. Validation and
test answer values are never read by this trainer.
"""
import argparse
import fcntl
import importlib.metadata
import math
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import run_liquid_baseline as base
import train_qwen_full as legacy

PROTOCOL = "qwen-full-b-a-to-b-full-v1"
EXPECTED_PARAMETERS = 494_032_768
require, read_json, write_json = legacy.require, legacy.read_json, legacy.write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--selection-model", type=Path, help="Inferred from the existing Liquid training config")
    parser.add_argument("--out", type=Path, default=Path("outputs/qwen_full_b_v1"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=500)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true")
    actions.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.model = (args.model if args.model.is_absolute() else args.root / args.model).resolve()
    args.out = (args.out if args.out.is_absolute() else args.root / args.out).resolve()
    if args.selection_model is None:
        args.selection_model = Path(read_json(args.root / "outputs/liquid_lora_v1/run_config.json")["model"])
    args.selection_model = args.selection_model.resolve()
    require(args.epochs == 3, "This first full-B protocol freezes three epochs")
    require(math.isfinite(args.learning_rate) and args.learning_rate > 0 and
            min(args.gradient_accumulation, args.preprocess_workers, args.save_steps) > 0 and args.num_workers >= 0,
            "Invalid training argument")
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "Launch plain python on one GPU")
    require(args.out != args.model and args.model not in args.out.parents and args.out not in args.model.parents,
            "Training output must be separate from the original model")
    return args


def preflight(args):
    import full_b_data as data
    import full_b_schema as schema
    import torch
    data.read_split(args.root)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected one visible CUDA GPU")
    require(importlib.metadata.version("transformers") == "5.2.0", "Use the existing liquid_tf52 Python environment")
    for name in ("datasets", "pyarrow", "pandas", "numpy", "accelerate"):
        importlib.metadata.version(name)
    chunks = list((args.root / "data/Twin-2K-500/wave_split/chunks").glob("*.parquet"))
    require(len(chunks) == 7, "Expected seven dataset chunks")
    require((args.root / "data/Twin-2K-500/question_catalog_and_human_response_csv/wave1_3_response.csv").is_file()
            and (args.root / "data/Twin-2K-500/question_catalog_and_human_response_csv/wave4_response.csv").is_file(),
            "Missing B/C response CSV files")
    require(not (args.model / "adapter_config.json").exists() and
            read_json(args.model / "config.json")["model_type"] == "qwen2", "Use the original Qwen model")
    original_run = args.root / "outputs/qwen_full_v1/run_config.json"
    prior = read_json(original_run)
    require(Path(prior["model"]).resolve() == args.model and
            legacy.model_files_sha256(args.model) == prior["model_files_sha256"],
            "Original Qwen weights/tokenizer differ from the earlier experiment")
    liquid = read_json(args.root / "outputs/liquid_lora_v1/run_config.json")
    require(Path(liquid["model"]).resolve() == args.selection_model, "Liquid selection tokenizer path changed")
    for name, digest in liquid["model_files_sha256"].items():
        if not (name.endswith(".safetensors") or name == "pytorch_model.bin"):
            require(base.digest_file(args.selection_model / name) == digest, "Original Liquid tokenizer changed")
    require(len(schema.catalog_items(read_json(Path(__file__).with_name("full_b_question_catalog.json")))) == 126,
            "Unexpected full-B catalog")
    # Conservative space for ~141k long supervised rows, manifests, two full
    # optimizer checkpoints and three independently retained epoch models.
    free = shutil.disk_usage(args.root).free / 2**30
    require(free >= 40, f"Need at least 40 GiB free; available {free:.1f} GiB")
    print(f"Preflight passed: full-B training; 1440 train / 309 validation; {free:.1f} GiB free.", flush=True)


def validate_supervised_row(row):
    ids, mask, labels = row["input_ids"], row["attention_mask"], row["labels"]
    require(len(ids) == len(mask) == len(labels) and mask == [1] * len(ids), "Training row lengths/mask differ")
    first = next((i for i, target in enumerate(labels) if target != -100), None)
    require(first is not None and first > 0 and 2 <= len(ids) - first <= 16,
            "Expected a nonempty prompt and answer-plus-EOS suffix")
    require(all(target == -100 for target in labels[:first]) and labels[first:] == ids[first:],
            "Only the contiguous answer-plus-EOS suffix may receive loss")
    return first


class AnswerOnlyCollator:
    def __init__(self, pad_token_id, eos_token_id):
        self.pad_token_id, self.eos_token_id = pad_token_id, eos_token_id

    def __call__(self, examples):
        import torch
        size = max(len(row["input_ids"]) for row in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in examples:
            first = validate_supervised_row(row)
            require(row["labels"][-1] == self.eos_token_id and
                    row["labels"][first:].count(self.eos_token_id) == 1, "Expected exactly one final answer EOS")
            padding = size - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
            batch["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def answer_loss(model, inputs):
    """Causal answer-only CE, applying the vocabulary head only where needed.

    The decoder still processes and backpropagates through the whole prompt.
    Avoiding vocabulary logits at masked prompt positions reduces memory use.
    Micro-batch size is fixed at one, so each example contributes its mean
    answer-token loss, including EOS, to gradient accumulation.
    """
    import torch
    import torch.nn.functional as F
    require(inputs["input_ids"].shape[0] == 1, "This protocol fixes micro-batch size one")
    decoded = model.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                          use_cache=False, return_dict=True)
    targets = inputs["labels"][:, 1:]
    selected = targets != -100
    require(bool(selected.any()), "No supervised answer tokens")
    hidden = decoded.last_hidden_state[:, :-1, :][selected]
    logits = model.lm_head(hidden)
    loss = F.cross_entropy(logits.float(), targets[selected], reduction="mean")
    require(bool(torch.isfinite(loss.detach())), "Non-finite answer loss")
    return loss


def trainer_class():
    from transformers import Trainer

    class AnswerTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # We normalize each micro-batch ourselves. Trainer must divide by
            # the actual accumulation window, including a final short window.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # The optimized decoder/head path bypasses Accelerate's wrapper on
            # the top-level model.forward, so enter autocast explicitly.
            with self.accelerator.autocast():
                loss = answer_loss(model, inputs)
            return (loss, {"loss": loss}) if return_outputs else loss

    return AnswerTrainer


def retain_epoch(run, checkpoint, epoch, step):
    """Retain immutable inference files while rotating optimizer checkpoints."""
    require(1 <= epoch <= 3, "Unexpected epoch snapshot")
    hashes = legacy.model_files_sha256(checkpoint)
    require("config.json" in hashes and any(name.endswith(".safetensors") or name == "pytorch_model.bin"
            for name in hashes), "Epoch model files are incomplete")
    target = run / "epoch_models" / f"epoch-{epoch}"
    target.parent.mkdir(exist_ok=True)
    candidate = {"name": target.name, "path": str(target.relative_to(run)), "epoch": float(epoch),
                 "global_step": int(step), "model_files_sha256": hashes}
    if target.exists():
        require(read_json(target.parent / f"epoch-{epoch}.json") == candidate and
                legacy.model_files_sha256(target) == hashes, "Previously retained epoch changed")
        return candidate
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    for name in hashes:
        try:
            os.link(checkpoint / name, temporary / name)
        except OSError:
            shutil.copy2(checkpoint / name, temporary / name)
    require(legacy.model_files_sha256(temporary) == hashes, "Epoch snapshot copy failed")
    write_json(target.parent / f"epoch-{epoch}.json", candidate)
    temporary.replace(target)
    return candidate


def collect_epochs(run):
    candidates = []
    for epoch in range(1, 4):
        descriptor = run / "epoch_models" / f"epoch-{epoch}.json"
        if descriptor.exists():
            candidate = read_json(descriptor)
            path = (run / candidate["path"]).resolve()
            require(run in path.parents and candidate["epoch"] == epoch and
                    legacy.model_files_sha256(path) == candidate["model_files_sha256"], "Epoch snapshot changed")
            candidates.append(candidate)
    return candidates


def recover_epoch_snapshots(run):
    """Recover a snapshot if interruption occurred after an epoch checkpoint save."""
    for checkpoint in sorted((run / "checkpoints").glob("checkpoint-*")):
        state_path = checkpoint / "trainer_state.json"
        if not state_path.is_file():
            continue
        state = read_json(state_path)
        epoch = float(state.get("epoch") or 0)
        if math.isclose(epoch, round(epoch), abs_tol=1e-7) and 1 <= round(epoch) <= 3:
            retain_epoch(run, checkpoint, round(epoch), int(state["global_step"]))


def callbacks(run, total_steps):
    from transformers import TrainerCallback

    class RetainEpochCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            epoch = float(state.epoch or 0)
            if math.isclose(epoch, round(epoch), abs_tol=1e-7) and 1 <= round(epoch) <= 3:
                path = run / "checkpoints" / f"checkpoint-{state.global_step}"
                retain_epoch(run, path, round(epoch), state.global_step)
                print(f"Retained full-B epoch {round(epoch)} model at step {state.global_step}", flush=True)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            write_json(run / "status.json", {"state": "training", "updated_unix": time.time(),
                       "global_step": state.global_step, "total_steps": total_steps, "epoch": state.epoch,
                       "test_answers_read": False, "logs": logs or {}})
            return control

    return [RetainEpochCallback()]


def training_arguments(args, n_examples, bf16):
    from transformers import TrainingArguments
    total_steps = args.epochs * math.ceil(n_examples / args.gradient_accumulation)
    loader = {"dataloader_num_workers": args.num_workers, "dataloader_pin_memory": True,
              "dataloader_persistent_workers": args.num_workers > 0}
    if args.num_workers:
        loader["dataloader_prefetch_factor"] = 2
    return TrainingArguments(output_dir=str(args.out / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, warmup_steps=math.ceil(0.05 * total_steps), weight_decay=0.0,
        optim="adamw_torch", lr_scheduler_type="linear", max_grad_norm=1.0, bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no", save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        logging_strategy="steps", logging_steps=10, logging_first_step=True, logging_nan_inf_filter=False,
        seed=42, data_seed=42, report_to=[], remove_unused_columns=False, label_names=["labels"], **loader)


def main():
    args = parse_args()
    preflight(args)
    if args.check_only:
        print("No participant questions or answer values loaded; no training launched.", flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".training.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Full-B training is already running")
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    require(mp.get_start_method() == "spawn", "Preparation and loader workers require spawn")
    import full_b_train_data as data
    write_json(args.out / "status.json", {"state": "preparing", "test_answers_read": False})
    train, validation, manifest, preparation = data.prepare_training(args.root, args.out, args.model,
        args.selection_model, workers=args.preprocess_workers)
    baselines = read_json(args.out / "training_baselines.json")
    require(len(baselines) == 126 and all(item["n"] > 0 for item in baselines.values()),
            "At least one full-B scalar item has no valid training responses")
    require(set(preparation["training_label_counts_by_item"]) == set(baselines),
            "Training metadata does not cover all 126 global scalar B items")
    require(set(train.column_names) == {"input_ids", "attention_mask", "labels"} and
            set(validation.column_names) == {"input_ids", "attention_mask"} and len(validation) == len(manifest),
            "Prepared training/validation columns differ")
    for row in train:
        validate_supervised_row(row)
    total_steps = args.epochs * math.ceil(len(train) / args.gradient_accumulation)
    config = {"protocol": PROTOCOL, "root": str(args.root), "model": str(args.model),
        "selection_model": str(args.selection_model), "initialization": "Original Qwen2.5-0.5B-Instruct",
        "training_labels": "B", "training_participants": 1440, "validation_participants": 309,
        "target_scope": "All answerable B scalar items with valid training B responses",
        "training_examples": len(train), "validation_examples": len(validation), "epochs": args.epochs,
        "trained_question_ids": sorted(preparation["coverage"]["train"]["qid_examples"]),
        "learning_rate": args.learning_rate, "micro_batch_size": 1,
        "gradient_accumulation": args.gradient_accumulation, "total_optimizer_steps": total_steps,
        "optimizer_steps_per_epoch": math.ceil(len(train) / args.gradient_accumulation),
        "warmup_steps": math.ceil(0.05 * total_steps), "optimizer": "adamw_torch", "seed": 42,
        "parameter_dtype": "torch.float32", "all_parameters_expected": EXPECTED_PARAMETERS,
        "loss": "Mean answer-token plus EOS causal cross-entropy per example; prompt/padding masked",
        "efficient_head": "Qwen decoder then vocabulary head at supervised causal positions only",
        "selection_token_budget": 8192, "qwen_input_safety_ceiling": 32760,
        "preprocess_workers": args.preprocess_workers, "dataloader_num_workers": args.num_workers,
        "periodic_save_steps": args.save_steps, "optimizer_checkpoint_limit": 2,
        "epoch_model_retention": "All three complete inference snapshots",
        "validation_selection": "Lowest C answer-only token NLL: mean per item, then per QID, then QID macro; ties earliest epoch",
        "test_policy": "No test participant question records or test label values read; no final test evaluation",
        "preparation_config_sha256": base.digest_file(args.out / "preparation_config.json"),
        "preparation_summary_sha256": base.digest_file(args.out / "preparation_summary.json"),
        "model_files_sha256": legacy.model_files_sha256(args.model),
        "versions": {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "tokenizers", "datasets", "pyarrow", "pandas", "numpy", "accelerate")},
        "scripts_sha256": {p.name: base.digest_file(p) for p in [Path(__file__), Path(data.__file__),
                          Path(legacy.__file__), Path(base.__file__)]}}
    config_path = args.out / "run_config.json"
    if config_path.exists():
        require(read_json(config_path) == config, "Frozen training settings changed; resume with original settings")
    else:
        write_json(config_path, config)
    config_hash = base.digest_text(base.dumps(config))
    if args.prepare_only:
        print(f"Prepared {len(train):,} training examples and {len(validation):,} validation prompts; "
              f"{total_steps:,} optimizer updates across three epochs. No GPU training launched.", flush=True)
        return
    completion_path = args.out / "training_complete.json"
    if completion_path.exists():
        completion = read_json(completion_path)
        require(completion["config_hash"] == config_hash and
                collect_epochs(args.out) == completion["candidate_checkpoints"], "Completed run changed")
        print("Full-B training already complete.", flush=True)
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    set_seed(42)
    bf16 = torch.cuda.is_bf16_supported()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
              dtype=torch.float32, attn_implementation="sdpa")
    require(model.config.model_type == "qwen2" and model.config.max_position_embeddings >= 32768,
            "Unexpected Qwen architecture/context")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    total = sum(p.numel() for p in model.parameters())
    require(total == EXPECTED_PARAMETERS and all(p.requires_grad and p.dtype == torch.float32
            for p in model.parameters()), "Expected all 494M parameters trainable in FP32")
    model.config.use_cache = False
    checkpoints = args.out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    recover_epoch_snapshots(args.out)
    last = get_last_checkpoint(str(checkpoints))
    if last:
        legacy.require_checkpoint_state(Path(last))
        print("Resuming model, optimizer and schedule:", last, flush=True)
    trainer = trainer_class()(model=model, args=training_arguments(args, len(train), bf16),
        train_dataset=train, data_collator=AnswerOnlyCollator(tokenizer.pad_token_id, tokenizer.eos_token_id),
        processing_class=tokenizer, callbacks=callbacks(args.out, total_steps))
    print(f"Full-B training: {len(train):,} examples; all {total:,} parameters; three epochs; "
          f"{total_steps:,} updates. Validation/test labels are absent from the trainer.", flush=True)
    result = trainer.train(resume_from_checkpoint=last)
    require(math.isfinite(float(result.metrics["train_loss"])) and trainer.state.epoch is not None and
            math.isclose(float(trainer.state.epoch), 3, abs_tol=1e-6) and trainer.state.global_step == total_steps,
            "Training did not finish the expected three epochs with finite loss")
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    candidates = collect_epochs(args.out)
    require(len(candidates) == 3 and candidates[-1]["global_step"] == total_steps,
            "Missing complete epoch model snapshots")
    write_json(completion_path, {"protocol": PROTOCOL, "config_hash": config_hash,
        "training_examples": len(train), "global_step": total_steps, "base_parameters": total,
        "trainable_parameters": total, "total_training_parameters": total,
        "training_metrics": result.metrics, "candidate_checkpoints": candidates})
    write_json(args.out / "status.json", {"state": "training_complete", "global_step": total_steps,
                                       "test_answers_read": False})
    print("Full-B training complete. Three retained epoch models are ready for validation.", flush=True)


if __name__ == "__main__":
    main()
