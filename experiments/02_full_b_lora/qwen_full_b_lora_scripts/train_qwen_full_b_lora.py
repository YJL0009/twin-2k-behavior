#!/usr/bin/env python3
"""LoRA adaptation of original Qwen on every valid training-participant B item."""
import argparse
from collections import Counter
import fcntl
import heapq
import importlib.metadata
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import run_liquid_baseline as base
import train_qwen_full as legacy
import train_qwen_full_b as shared

PROTOCOL = "qwen-full-b-a-to-b-lora-v1"
BASE_PARAMETERS = 494_032_768
ADAPTER_PARAMETERS = 4_399_104
TOTAL_PARAMETERS = BASE_PARAMETERS + ADAPTER_PARAMETERS
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_SETTINGS = {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": list(TARGET_MODULES),
                 "bias": "none", "modules_to_save": None, "use_dora": False, "use_rslora": False}
require, read_json, write_json = legacy.require, legacy.read_json, legacy.write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--selection-model", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/qwen_full_b_lora_v4_b16_a2"))
    parser.add_argument("--prepared-from", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
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
    if args.prepared_from is not None:
        args.prepared_from = (args.prepared_from if args.prepared_from.is_absolute()
                              else args.root / args.prepared_from).resolve()
        require(args.prepared_from != args.out, "Prepared source must differ from the new run folder")
    if args.selection_model is None:
        args.selection_model = Path(read_json(args.root / "outputs/liquid_lora_v1/run_config.json")["model"])
    args.selection_model = args.selection_model.resolve()
    require(args.epochs == 3 and math.isfinite(args.learning_rate) and args.learning_rate > 0
            and min(args.micro_batch_size, args.gradient_accumulation, args.preprocess_workers, args.save_steps) > 0
            and args.num_workers >= 0, "Use three epochs and positive finite training settings")
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "Use plain python on one GPU")
    require(args.out != args.model and args.model not in args.out.parents and args.out not in args.model.parents,
            "Keep output separate from original model")
    return args


def preflight(args):
    shared.preflight(args)
    require(importlib.metadata.version("peft") == "0.18.0", "Use PEFT 0.18.0 in the existing liquid_tf52 environment")
    require(TOTAL_PARAMETERS < 500_000_000, "Base plus adapter must be below 500M")
    print(f"LoRA preflight: rank 8 on attention/MLP; {ADAPTER_PARAMETERS:,} trainable; "
          f"{TOTAL_PARAMETERS:,} total parameters.", flush=True)
    print(f"Batch settings: {args.micro_batch_size} examples per GPU forward; "
          f"accumulation {args.gradient_accumulation}; effective batch "
          f"{args.micro_batch_size * args.gradient_accumulation} on one GPU.", flush=True)
    if args.prepared_from is not None and not (args.out / "preparation_summary.json").is_file():
        require((args.prepared_from / "preparation_summary.json").is_file(),
                f"Prepared source is incomplete: {args.prepared_from}")


def projection_budget(model):
    projections = {name: module for name, module in model.named_modules()
                   if name.rsplit(".", 1)[-1] in TARGET_MODULES}
    counts = Counter(name.rsplit(".", 1)[-1] for name in projections)
    require(counts == {name: 24 for name in TARGET_MODULES}, "Unexpected Qwen projection coverage")
    shapes = {"q_proj": (896, 896), "k_proj": (896, 128), "v_proj": (896, 128),
              "o_proj": (896, 896), "gate_proj": (896, 4864), "up_proj": (896, 4864),
              "down_proj": (4864, 896)}
    for name, module in projections.items():
        require((module.in_features, module.out_features) == shapes[name.rsplit(".", 1)[-1]],
                f"Unexpected Qwen projection dimensions: {name}")
    expected = 8 * sum(module.in_features + module.out_features for module in projections.values())
    require(expected == ADAPTER_PARAMETERS, "LoRA parameter budget changed")
    return expected


def validate_trainable(model):
    named = list(model.named_parameters())
    total = sum(p.numel() for _, p in named)
    trainable = sum(p.numel() for _, p in named if p.requires_grad)
    require(total == TOTAL_PARAMETERS < 500_000_000 and trainable == ADAPTER_PARAMETERS,
            "Unexpected LoRA/total parameter count")
    require(all((".lora_A." in name or ".lora_B." in name) == p.requires_grad for name, p in named),
            "Only LoRA A/B matrices may be trainable; all original weights must be frozen")
    return total, trainable


def validate_precision(model, base_dtype):
    import torch
    require(all(p.dtype == (torch.float32 if p.requires_grad else base_dtype)
                for p in model.parameters()),
            "Frozen base must retain its recorded low precision; trainable LoRA must remain FP32")


def audit_training_rows(train, batch_size):
    # Collect the longest rows during the full supervision audit. Only the
    # prepared training split is used for the padded-batch memory check.
    longest = []
    for index, row in enumerate(train):
        shared.validate_supervised_row(row)
        candidate = (len(row["input_ids"]), index, row)
        if len(longest) < batch_size:
            heapq.heappush(longest, candidate)
        elif candidate[:2] > longest[0][:2]:
            heapq.heapreplace(longest, candidate)
    require(longest, "No training examples for memory probe")
    return [entry[2] for entry in sorted(longest, reverse=True)]


def answer_loss(model, inputs):
    # Standard LoRA inserts layers into this exact underlying Qwen in place.
    # Its active adapters therefore remain in every decoder projection. The
    # protocol disallows prompt tuning, routed adapters and modules_to_save.
    qwen = model.get_base_model()
    require(qwen.config.model_type == "qwen2" and hasattr(qwen, "model") and hasattr(qwen, "lm_head"),
            "Expected underlying Qwen with injected LoRA projections")
    import torch
    import torch.nn.functional as F
    decoded = qwen.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                         use_cache=False, return_dict=True)
    targets = inputs["labels"][:, 1:]
    selected = targets != -100
    counts = selected.sum(dim=1)
    require(bool((counts > 0).all()), "Each example needs supervised answer tokens")
    hidden = decoded.last_hidden_state[:, :-1, :][selected]
    logits = qwen.lm_head(hidden)
    token_losses = F.cross_entropy(logits.float(), targets[selected], reduction="none")
    row_indices = torch.where(selected)[0]
    # Preserve the original objective: every example contributes its own mean
    # answer-token loss, including EOS, regardless of answer length or padding.
    sums = torch.zeros(targets.shape[0], device=token_losses.device, dtype=token_losses.dtype)
    sums = sums.index_add(0, row_indices, token_losses)
    loss = (sums / counts).mean()
    require(bool(torch.isfinite(loss.detach())), "Non-finite answer loss")
    return loss


def steps_per_epoch(n_examples, micro_batch_size, gradient_accumulation):
    require(min(n_examples, micro_batch_size, gradient_accumulation) > 0,
            "Example count and batch settings must be positive")
    loader_batches = math.ceil(n_examples / micro_batch_size)
    return math.ceil(loader_batches / gradient_accumulation)


def training_arguments(args, n_examples, bf16):
    from transformers import TrainingArguments
    total_steps = args.epochs * steps_per_epoch(n_examples, args.micro_batch_size, args.gradient_accumulation)
    loader = {"dataloader_num_workers": args.num_workers, "dataloader_pin_memory": True,
              "dataloader_persistent_workers": args.num_workers > 0}
    if args.num_workers:
        loader["dataloader_prefetch_factor"] = 2
    return TrainingArguments(output_dir=str(args.out / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch_size, gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, warmup_steps=math.ceil(0.05 * total_steps), weight_decay=0.0,
        optim="adamw_torch", lr_scheduler_type="linear", max_grad_norm=1.0, bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no", save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        logging_strategy="steps", logging_steps=10, logging_first_step=True, logging_nan_inf_filter=False,
        seed=42, data_seed=42, report_to=[], remove_unused_columns=False, label_names=["labels"], **loader)


def trainer_class():
    from transformers import Trainer

    class LoRAAnswerTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False
            # Transformers 5.2 already divides by the actual number of batches
            # in each accumulation window and controls gradient synchronization.
            # Accelerate 1.12 would otherwise divide again in backward(). Keep
            # accumulation in Trainer.args; disable only that second division.
            self.accelerator.gradient_accumulation_steps = 1

        def get_batch_samples(self, epoch_iterator, num_batches, device):
            batches, _ = super().get_batch_samples(epoch_iterator, num_batches, device)
            examples = sum(int(batch["input_ids"].shape[0]) for batch in batches)
            require(not batches or examples > 0, "Accumulation window has no examples")
            return batches, examples

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            with self.accelerator.autocast():
                loss = answer_loss(model, inputs)
            if num_items_in_batch is not None:
                batch_examples = int(inputs["input_ids"].shape[0])
                window_batches = int(self.current_gradient_accumulation_steps)
                require(type(num_items_in_batch) is int and num_items_in_batch > 0
                        and 0 < batch_examples <= num_items_in_batch and window_batches > 0,
                        "Invalid example counts for accumulation-window weighting")
                # training_step divides by window_batches. The resulting
                # contribution is sum(example mean losses) / window_examples,
                # including unequal full/partial minibatches in the same window.
                weight = window_batches * batch_examples / num_items_in_batch
                require(math.isfinite(weight) and weight > 0, "Invalid accumulated loss weight")
                loss = loss * weight
            return (loss, {"loss": loss}) if return_outputs else loss

    return LoRAAnswerTrainer


def require_adapter_checkpoint(path):
    require(all((path / name).is_file() for name in
            ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json")),
            f"Incomplete LoRA optimizer checkpoint: {path}")
    require((path / "adapter_model.safetensors").is_file() and
            (path / "adapter_model.safetensors").stat().st_size > 0, "Missing adapter weights")


def retain_epoch(run, checkpoint, epoch, step):
    hashes = legacy.model_files_sha256(checkpoint)
    require("adapter_config.json" in hashes and "adapter_model.safetensors" in hashes
            and "model.safetensors" not in hashes, "Expected adapter-only checkpoint")
    require(1 <= epoch <= 3, "Unexpected adapter epoch")
    target = run / "epoch_adapters" / f"epoch-{epoch}"
    target.parent.mkdir(exist_ok=True)
    candidate = {"name": target.name, "path": str(target.relative_to(run)), "epoch": float(epoch),
                 "global_step": int(step), "artifact_type": "lora_adapter", "model_files_sha256": hashes}
    if target.exists():
        require(read_json(target.parent / f"epoch-{epoch}.json") == candidate and
                legacy.model_files_sha256(target) == hashes, "Retained adapter changed")
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
    require(legacy.model_files_sha256(temporary) == hashes, "Adapter snapshot copy failed")
    write_json(target.parent / f"epoch-{epoch}.json", candidate)
    temporary.replace(target)
    return candidate


def collect_epochs(run):
    candidates = []
    for epoch in range(1, 4):
        descriptor = run / "epoch_adapters" / f"epoch-{epoch}.json"
        if descriptor.exists():
            candidate = read_json(descriptor)
            path = (run / candidate["path"]).resolve()
            require(run in path.parents and candidate["epoch"] == epoch and
                    legacy.model_files_sha256(path) == candidate["model_files_sha256"], "Adapter snapshot changed")
            candidates.append(candidate)
    return candidates


def recover_epoch_snapshots(run):
    for checkpoint in sorted((run / "checkpoints").glob("checkpoint-*")):
        state_path = checkpoint / "trainer_state.json"
        if state_path.exists():
            state = read_json(state_path)
            epoch = float(state.get("epoch") or 0)
            if math.isclose(epoch, round(epoch), abs_tol=1e-7) and 1 <= round(epoch) <= 3:
                retain_epoch(run, checkpoint, round(epoch), state["global_step"])


def callbacks(run, total_steps, base_dtype):
    from transformers import TrainerCallback

    class RetainAdapterCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            # Trainer reloads a resumed adapter before this event.
            validate_trainable(model)
            validate_precision(model, base_dtype)
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            epoch = float(state.epoch or 0)
            if math.isclose(epoch, round(epoch), abs_tol=1e-7) and 1 <= round(epoch) <= 3:
                retain_epoch(run, run / "checkpoints" / f"checkpoint-{state.global_step}",
                             round(epoch), state.global_step)
                print(f"Retained epoch-{round(epoch)} LoRA adapter", flush=True)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            write_json(run / "status.json", {"state": "training_lora", "updated_unix": time.time(),
                "global_step": state.global_step, "total_steps": total_steps, "epoch": state.epoch,
                "test_answers_read": False, "logs": logs or {}})
            return control

    return [RetainAdapterCallback()]


def main():
    args = parse_args()
    preflight(args)
    if args.check_only:
        print("No participant questions/answer values loaded; no LoRA training launched.", flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".training.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Full-B LoRA training is already running")
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    require(mp.get_start_method() == "spawn", "Workers require spawn")
    import full_b_train_data as data
    write_json(args.out / "status.json", {"state": "preparing", "test_answers_read": False})
    if args.prepared_from is not None:
        from reuse_full_b_prepared import reuse_prepared
        reused = reuse_prepared(args.prepared_from, args.out)
        if not (args.out / "preparation_reuse.json").exists():
            write_json(args.out / "preparation_reuse.json", reused)
        print("Prepared data reuse:", reused["action"], flush=True)
    train, validation, manifest, preparation = data.prepare_training(args.root, args.out, args.model,
        args.selection_model, workers=args.preprocess_workers)
    baselines = read_json(args.out / "training_baselines.json")
    require(len(baselines) == 126 and all(item["n"] > 0 for item in baselines.values())
            and set(preparation["training_label_counts_by_item"]) == set(baselines),
            "Full-B training coverage/baselines incomplete")
    require(set(train.column_names) == {"input_ids", "attention_mask", "labels"} and
            set(validation.column_names) == {"input_ids", "attention_mask"} and len(validation) == len(manifest),
            "Prepared supervision schema differs")
    longest_examples = audit_training_rows(train, args.micro_batch_size)
    import torch
    bf16 = torch.cuda.is_bf16_supported()
    base_dtype = torch.bfloat16 if bf16 else torch.float16
    base_dtype_name = "bfloat16" if bf16 else "float16"
    epoch_steps = steps_per_epoch(len(train), args.micro_batch_size, args.gradient_accumulation)
    total_steps = args.epochs * epoch_steps
    config = {"protocol": PROTOCOL, "root": str(args.root), "model": str(args.model),
        "selection_model": str(args.selection_model), "initialization": "Original Qwen2.5-0.5B-Instruct plus fresh LoRA",
        "training_labels": "B", "training_participants": 1440, "validation_participants": 309,
        "target_scope": "All answerable B scalar items with valid training B responses",
        "training_examples": len(train), "validation_examples": len(validation), "epochs": args.epochs,
        "trained_question_ids": sorted(preparation["coverage"]["train"]["qid_examples"]),
        "learning_rate": args.learning_rate, "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.micro_batch_size * args.gradient_accumulation,
        "total_optimizer_steps": total_steps, "optimizer_steps_per_epoch": epoch_steps,
        "warmup_steps": math.ceil(0.05 * total_steps), "optimizer": "adamw_torch", "seed": 42,
        "lora": LORA_SETTINGS, "base_parameters_expected": BASE_PARAMETERS,
        "adapter_parameters_expected": ADAPTER_PARAMETERS, "total_training_parameters_expected": TOTAL_PARAMETERS,
        "parameter_dtype": f"{base_dtype_name} frozen base and FP32 trainable LoRA; matching autocast",
        "base_parameter_dtype": base_dtype_name, "adapter_parameter_dtype": "float32",
        "memory_probe": "Longest real training batch: checkpointed forward/backward; no optimizer update; restores RNG",
        "loss": "Mean answer-token plus EOS causal cross-entropy per example; prompt/padding masked",
        "accumulated_loss_weighting": "Actual example count per window; full/partial minibatches weighted by row count",
        "accumulation_owner": "Transformers Trainer; Accelerator backward accumulation divisor fixed at one",
        "efficient_head": "Underlying Qwen with active injected LoRA; vocabulary head at supervised positions only",
        "selection_token_budget": 8192, "qwen_input_safety_ceiling": 32760,
        "preprocess_workers": args.preprocess_workers, "dataloader_num_workers": args.num_workers,
        "periodic_save_steps": args.save_steps, "optimizer_checkpoint_limit": 2,
        "epoch_model_retention": "All three adapter-only inference snapshots",
        "validation_selection": "Lowest C answer-only token NLL: mean per item, then per QID, then QID macro; ties earliest epoch",
        "test_policy": "No test participant question records or test label values used; no final test evaluation",
        "preparation_config_sha256": base.digest_file(args.out / "preparation_config.json"),
        "preparation_summary_sha256": base.digest_file(args.out / "preparation_summary.json"),
        "model_files_sha256": legacy.model_files_sha256(args.model),
        "versions": {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "tokenizers", "datasets", "pyarrow", "pandas", "numpy", "accelerate", "peft")},
        "scripts_sha256": {p.name: base.digest_file(p) for p in [Path(__file__), Path(shared.__file__),
                          Path(__file__).with_name("reuse_full_b_prepared.py"),
                          Path(__file__).with_name("lora_memory_probe.py"),
                          Path(data.__file__), Path(legacy.__file__), Path(base.__file__)]}}
    config_path = args.out / "run_config.json"
    if config_path.exists():
        require(read_json(config_path) == config, "Frozen LoRA training settings changed")
    else:
        write_json(config_path, config)
    config_hash = base.digest_text(base.dumps(config))
    if args.prepare_only:
        print(f"Prepared {len(train):,} all-B training examples; {total_steps:,} LoRA updates across three epochs.", flush=True)
        return
    completion_path = args.out / "training_complete.json"
    if completion_path.exists():
        completion = read_json(completion_path)
        require(completion["config_hash"] == config_hash and collect_epochs(args.out) == completion["candidate_checkpoints"],
                "Completed LoRA run changed")
        print("Full-B LoRA training already complete.", flush=True)
        return
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    original = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        dtype=base_dtype, attn_implementation="sdpa")
    require(original.config.model_type == "qwen2" and sum(p.numel() for p in original.parameters()) == BASE_PARAMETERS,
            "Unexpected original Qwen model")
    projection_budget(original)
    original.config.use_cache = False
    model = get_peft_model(original, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
              lora_dropout=0.05, target_modules=list(TARGET_MODULES), bias="none", modules_to_save=None,
              use_dora=False, use_rslora=False), autocast_adapter_dtype=True)
    total, trainable = validate_trainable(model)
    validate_precision(model, base_dtype)
    # Frozen embeddings need gradient-bearing outputs for checkpointed decoder
    # layers; this hook does not make the embedding parameters trainable.
    model.enable_input_require_grads()
    validate_trainable(model)
    checkpoints = args.out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    recover_epoch_snapshots(args.out)
    last = get_last_checkpoint(str(checkpoints))
    if last:
        require_adapter_checkpoint(Path(last))
        print("Resuming LoRA adapter, optimizer and schedule:", last, flush=True)
    trainer = trainer_class()(model=model, args=training_arguments(args, len(train), bf16),
        train_dataset=train, data_collator=shared.AnswerOnlyCollator(tokenizer.pad_token_id, tokenizer.eos_token_id),
        processing_class=tokenizer, callbacks=callbacks(args.out, total_steps, base_dtype))
    probe_path = args.out / "memory_probe.json"
    gpu = {"name": torch.cuda.get_device_name(0),
           "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    if last and probe_path.is_file() and read_json(probe_path).get("config_hash") == config_hash \
            and read_json(probe_path).get("gpu") == gpu and read_json(probe_path).get("passed") is True:
        print("Reusing successful same-config memory probe on the same GPU type.", flush=True)
    else:
        from lora_memory_probe import probe_batch
        report = probe_batch(model, longest_examples, trainer.data_collator, answer_loss, bf16,
                             {"config_hash": config_hash, "gpu": gpu})
        require(report["trainable_parameters_with_grad"] == ADAPTER_PARAMETERS,
                "Memory probe did not connect all LoRA matrices to the supervised loss")
        write_json(probe_path, {"passed": True, "config_hash": config_hash, "gpu": gpu, **report})
    print(f"Full-B LoRA: {len(train):,} examples; {trainable:,} trainable / {total:,} total parameters; "
          f"micro-batch {args.micro_batch_size}, accumulation {args.gradient_accumulation}; "
          f"three epochs, {total_steps:,} updates.", flush=True)
    result = trainer.train(resume_from_checkpoint=last)
    require(math.isfinite(float(result.metrics["train_loss"])) and trainer.state.epoch is not None and
            math.isclose(float(trainer.state.epoch), 3, abs_tol=1e-6) and trainer.state.global_step == total_steps,
            "LoRA training did not finish the expected three epochs")
    validate_trainable(model)
    validate_precision(model, base_dtype)
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    candidates = collect_epochs(args.out)
    require(len(candidates) == 3 and candidates[-1]["global_step"] == total_steps, "Missing completed epoch adapters")
    write_json(completion_path, {"protocol": PROTOCOL, "config_hash": config_hash,
        "training_examples": len(train), "global_step": total_steps, "base_parameters": BASE_PARAMETERS,
        "trainable_parameters": trainable, "adapter_parameters": trainable, "total_training_parameters": total,
        "training_metrics": result.metrics, "candidate_checkpoints": candidates})
    write_json(args.out / "status.json", {"state": "training_complete", "test_answers_read": False})
    print("All-B LoRA training complete; three epoch adapters retained for validation.", flush=True)


if __name__ == "__main__":
    main()
