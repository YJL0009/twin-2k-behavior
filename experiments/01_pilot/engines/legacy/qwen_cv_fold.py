#!/usr/bin/env python3
"""Train one participant CV fold from original Qwen weights; score its final epoch.

No C labels are used for training or epoch selection. The fixed three-epoch
recipe is shared across all folds. Re-running resumes optimizer state or skips
completed work. Final-test participants are never scored.
"""
import argparse
import gc
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace

import train_qwen_full as core
import run_liquid_baseline as base

PROTOCOL = "qwen-participant-cv-full-final-epoch-v1"
EXPECTED_PARAMETERS = 494_032_768
require = core.require
read_json = core.read_json
write_json = core.write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", type=Path, default=Path("/root/autodl-tmp/twin_qwen/outputs/qwen_cv_v1"))
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--c-response", type=Path, help="Numeric wave4_response.csv; normally inferred")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--train-only", action="store_true")
    modes.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    args.cv_root = args.cv_root.resolve()
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "Use plain python on one GPU, not torchrun")
    return args


def train_arguments(config, out):
    return SimpleNamespace(out=out, epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]), micro_batch_size=int(config["micro_batch_size"]),
        gradient_accumulation=int(config["gradient_accumulation"]),
        num_workers=int(config["dataloader_num_workers"]), max_input_tokens=int(config["max_input_tokens"]))


def make_training_config(cv, fold, out, n):
    args = train_arguments(cv, out)
    require(args.epochs == 3 and cv["seed"] == 42,
            "This CV protocol freezes three epochs and seed 42 for all folds")
    config = {"protocol": PROTOCOL, "fold_number": fold["fold_number"],
        "cv_config_hash": core.config_digest(cv), "fold_config_hash": core.config_digest(fold),
        "model": cv["model"], "model_files_sha256": cv["model_files_sha256"],
        "training_examples": n, "epochs": args.epochs, "learning_rate": args.learning_rate,
        "micro_batch_size": args.micro_batch_size, "gradient_accumulation": args.gradient_accumulation,
        "dataloader_num_workers": args.num_workers, "seed": 42,
        "parameter_dtype": "torch.float32", "precision": "bf16-if-supported-otherwise-fp16 autocast",
        "trainable_parameters_expected": EXPECTED_PARAMETERS, "optimizer": "adamw_torch",
        "weight_decay": 0.0, "lr_scheduler_type": "linear", "max_grad_norm": 1.0,
        "warmup_fraction": 0.05, "gradient_checkpointing": {"use_reentrant": False},
        "save_strategy": "epoch", "save_total_limit": 1,
        "evaluation_policy": "final epoch only; no fold-C-based epoch selection",
        "versions": {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "tokenizers", "datasets", "accelerate", "pandas")},
        "scripts_sha256": {Path(p).name: core.digest_file(p) for p in (__file__, core.__file__)},
        **core.step_counts(args, n)}
    for name, current in config["versions"].items():
        if name in cv["versions"]:
            require(cv["versions"][name] == current, f"CV preparation/training {name} versions differ")
    require(fold["cv_config_hash"] == config["cv_config_hash"], "CV and fold configurations differ")
    path = out / "run_config.json"
    if path.exists():
        require(read_json(path) == config, "Fold training configuration changed; use a fresh CV output folder")
    else:
        require(not (out / "checkpoints").exists() and not (out / "training_complete.json").exists(),
                "Untracked fold checkpoint exists")
        write_json(path, config)
    return config


def final_candidate(out, epochs, expected_steps):
    checkpoints = list((out / "checkpoints").glob("checkpoint-*"))
    require(checkpoints, "No saved fold checkpoint")
    path = max(checkpoints, key=lambda item: int(item.name.rsplit("-", 1)[-1]))
    core.require_checkpoint_state(path)
    state = read_json(path / "trainer_state.json")
    require(math.isclose(float(state["epoch"]), epochs, rel_tol=0, abs_tol=1e-6)
            and int(state["global_step"]) == expected_steps
            and path.name == f"checkpoint-{expected_steps}", "Fold final checkpoint is incomplete")
    return {"name": path.name, "path": str(path.relative_to(out)),
        "epoch": float(state["epoch"]), "global_step": int(state["global_step"]),
        "model_files_sha256": core.model_files_sha256(path)}


def train_fold(cv, fold, dataset, out, config):
    done_path = out / "training_complete.json"
    if done_path.exists():
        done = read_json(done_path)
        require(done["protocol"] == PROTOCOL and done["config_hash"] == core.config_digest(config),
                "Fold completion configuration changed")
        require(done["final_candidate"] == final_candidate(out, config["epochs"], config["total_optimizer_steps"]),
                "Completed fold checkpoint changed")
        print(f"Fold {fold['fold_number']}: training already complete", flush=True)
        return done
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    require(torch.cuda.is_available(), "CUDA is required for full fine-tuning")
    model_path = Path(cv["model"])
    require(core.model_files_sha256(model_path) == cv["model_files_sha256"], "Original Qwen files changed")
    require(not (model_path / "adapter_config.json").exists(), "Each fold must initialize original full Qwen weights")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                               dtype=torch.float32, attn_implementation="sdpa")
    require(model.config.model_type == "qwen2" and cv["max_input_tokens"] + 8 <= model.config.max_position_embeddings,
            "Unexpected Qwen architecture/context")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(total == trainable == EXPECTED_PARAMETERS < 500_000_000
            and all(p.dtype == torch.float32 for p in model.parameters()), "Full Qwen parameter audit failed")
    model.config.use_cache = False
    args = train_arguments(cv, out)
    bf16 = torch.cuda.is_bf16_supported()
    training_args = core.build_training_arguments(args, bf16, len(dataset))
    training_args.save_total_limit = 1
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint = get_last_checkpoint(str(checkpoint_dir))
    if checkpoint:
        core.require_checkpoint_state(checkpoint)
        print("Resuming fold model and optimizer:", checkpoint, flush=True)
    print(f"Fold {fold['fold_number']}: full FT {total:,} parameters, {len(dataset)} examples, "
          f"{config['epochs']} fixed epochs", flush=True)
    trainer = core.finite_trainer_class()(model=model, args=training_args, train_dataset=dataset,
        data_collator=core.AnswerOnlyCollator(tokenizer.pad_token_id), processing_class=tokenizer)
    # A saved final checkpoint can survive interruption before the completion marker.
    already_final = False
    if checkpoint:
        state = read_json(Path(checkpoint) / "trainer_state.json")
        already_final = (int(state["global_step"]) == config["total_optimizer_steps"]
                         and math.isclose(float(state["epoch"]), args.epochs, abs_tol=1e-6))
    if already_final:
        metrics_path = out / "checkpoints/train_results.json"
        metrics = read_json(metrics_path) if metrics_path.exists() else {"recovered_from_complete_final_checkpoint": True}
    else:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        require(math.isfinite(float(result.metrics["train_loss"]))
                and trainer.state.epoch is not None
                and math.isclose(float(trainer.state.epoch), args.epochs, abs_tol=1e-6)
                and trainer.state.global_step == config["total_optimizer_steps"], "Fold training did not finish")
        metrics = result.metrics
        trainer.save_state()
        trainer.save_metrics("train", metrics)
    candidate = final_candidate(out, args.epochs, config["total_optimizer_steps"])
    done = {"protocol": PROTOCOL, "config_hash": core.config_digest(config),
        "training_examples": len(dataset), "global_step": candidate["global_step"],
        "base_parameters": total, "trainable_parameters": trainable, "total_training_parameters": total,
        "training_metrics": metrics, "final_candidate": candidate,
        "evaluation_policy": "final epoch only; no fold-C-based epoch selection"}
    write_json(done_path, done)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return done


def read_selected_labels(path, pids, qids):
    """Read ID column first, skip every other participant before parsing labels."""
    import pandas as pd
    ids = pd.read_csv(path, usecols=["pid"], dtype={"pid": "string"})["pid"]
    require(ids.notna().all() and ids.is_unique and set(pids).issubset(set(ids)), "Response participant IDs invalid")
    skipped = [i + 1 for i, pid in enumerate(ids) if pid not in pids]
    labels = pd.read_csv(path, skiprows=skipped, usecols=["pid", *qids], dtype={"pid": "string"}).set_index("pid")
    require(labels.index.is_unique and set(labels.index) == set(pids), "Heldout-only label selection failed")
    for qid in qids:
        require(labels[qid].dropna().isin([1, 2]).all(), f"Invalid numeric response labels for {qid}: {path}")
    return labels


def finite_or_none(value):
    number = float(value)
    return number if math.isfinite(number) else None


def score_predictions(records, val_manifest, labels_b, labels_c, majority, out, qids):
    import pandas as pd
    keys = list(val_manifest[["pid", "qid"]].itertuples(index=False, name=None))
    require(len(records) == len(keys) and len(set(keys)) == len(keys), "Incomplete prediction set")
    record_frame = pd.DataFrame(records)
    require(not record_frame[["pid", "qid"]].duplicated().any()
            and set(record_frame[["pid", "qid"]].itertuples(index=False, name=None)) == set(keys),
            "Prediction identities do not match fold heldout identities")
    scores = val_manifest[["pid", "qid"]].merge(record_frame, on=["pid", "qid"], validate="one_to_one")
    scores["target_c"] = [labels_c.at[r.pid, r.qid] for r in scores.itertuples()]
    scores["reference_b"] = [labels_b.at[r.pid, r.qid] for r in scores.itertuples()]
    scores["majority_prediction"] = scores.qid.map(majority)
    require(scores.majority_prediction.isin([1, 2]).all(), "Missing fold-training majority label")
    scores["valid"] = scores.prediction.isin([1, 2])
    has_c, has_b = scores.target_c.notna(), scores.reference_b.notna()
    paired = has_b & has_c
    scores["correct"] = (scores.valid & scores.prediction.eq(scores.target_c)).astype(float).where(has_c)
    scores["majority_correct"] = scores.majority_prediction.eq(scores.target_c).astype(float).where(has_c)
    scores["b_correct"] = (scores.valid & scores.prediction.eq(scores.reference_b)).astype(float).where(has_b)
    scores["majority_b_correct"] = scores.majority_prediction.eq(scores.reference_b).astype(float).where(has_b)
    scores["human_correct"] = scores.reference_b.eq(scores.target_c).astype(float).where(paired)
    scores["model_correct_paired"] = scores.correct.where(paired)
    scores["majority_correct_paired"] = scores.majority_correct.where(paired)
    metrics = scores.groupby("qid").agg(total_prompts=("pid", "size"), examples=("correct", "count"),
        model_accuracy=("correct", "mean"), valid_rate=("valid", "mean"),
        majority_accuracy=("majority_correct", "mean"), b_examples=("b_correct", "count"),
        model_B_accuracy=("b_correct", "mean"), majority_B_accuracy=("majority_b_correct", "mean"),
        human_pairs=("human_correct", "count"), human_reference=("human_correct", "mean"),
        model_accuracy_paired=("model_correct_paired", "mean"),
        majority_accuracy_paired=("majority_correct_paired", "mean"))
    require(set(metrics.index) == set(qids) and metrics.examples.gt(0).all(),
            "Every target question needs an observed C label for comparable macro accuracy")
    metrics["excluded_missing_c"] = metrics.total_prompts - metrics.examples
    counts = scores.groupby("qid").agg(examples=("pid", "size"),
        predicted_1=("prediction", lambda x: int(x.eq(1).sum())),
        predicted_2=("prediction", lambda x: int(x.eq(2).sum())),
        invalid=("prediction", lambda x: int((~x.isin([1, 2])).sum())))
    summary = {"evaluation_split": "participant_cross_validation", "evaluation_labels": "C",
        "question_subset": list(qids), "total_prompts": len(scores), "examples": int(has_c.sum()),
        "excluded_missing_c": int((~has_c).sum()), "participants": int(scores.pid.nunique()),
        "question_macro_accuracy": float(metrics.model_accuracy.mean()), "pooled_accuracy": float(scores.correct.mean()),
        "invalid_outputs": int((~scores.valid).sum()), "invalid_rate": float((~scores.valid).mean()),
        "majority_macro_accuracy": float(metrics.majority_accuracy.mean()),
        "majority_pooled_accuracy": float(scores.majority_correct.mean()),
        "gain_over_majority_pp": 100 * float(metrics.model_accuracy.mean() - metrics.majority_accuracy.mean()),
        "B_question_macro_accuracy": finite_or_none(metrics.model_B_accuracy.mean()),
        "B_pooled_accuracy": finite_or_none(scores.b_correct.mean()),
        "B_majority_macro_accuracy": finite_or_none(metrics.majority_B_accuracy.mean()),
        "human_reference_macro_paired": finite_or_none(metrics.human_reference.mean()),
        "model_macro_accuracy_paired": finite_or_none(metrics.model_accuracy_paired.mean()),
        "majority_macro_accuracy_paired": finite_or_none(metrics.majority_accuracy_paired.mean()),
        "epoch_selection": "fixed final epoch; no selection on fold C", "test_evaluated": False}
    scores.to_csv(out / "scored_predictions.csv", index=False)
    metrics.to_csv(out / "metrics.csv")
    counts.to_csv(out / "prediction_counts.csv")
    print(metrics.round(4).to_string(), flush=True)
    print("Prediction counts per question:\n" + counts.to_string(), flush=True)
    return summary


def read_resumable_records(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    pieces = raw.splitlines(keepends=True)
    records, complete_bytes = [], 0
    for index, line in enumerate(pieces):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            require(index == len(pieces) - 1 and not line.endswith(b"\n"),
                    "Saved predictions contain corruption before their final partial line")
            backup = path.with_suffix(".interrupted_tail.bin")
            require(not backup.exists(), "Multiple interrupted prediction tails; inspect saved outputs")
            backup.write_bytes(line)
            with path.open("r+b") as handle:
                handle.truncate(complete_bytes)
            print("Recovered one incomplete final prediction line; preserved its bytes:", backup, flush=True)
            break
        records.append(record)
        complete_bytes += len(line)
    if records and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return records


def evaluate_fold(cv, fold, pool, manifest, out, training, c_response=None):
    from transformers import AutoTokenizer
    done = read_json(out / "training_complete.json")
    require(done["protocol"] == PROTOCOL and done["config_hash"] == core.config_digest(training)
            and done["base_parameters"] == done["trainable_parameters"] == EXPECTED_PARAMETERS,
            "Fold training completion/configuration mismatch")
    candidate = final_candidate(out, training["epochs"], training["total_optimizer_steps"])
    require(candidate == done["final_candidate"], "Final fold checkpoint changed")
    model_path = (out / candidate["path"]).resolve()
    require(out in model_path.parents, "Invalid fold checkpoint path")
    val_indices = [int(i) for i in fold["val_indices"]]
    val_manifest = manifest.iloc[val_indices].copy().reset_index(drop=True)
    require(len(val_manifest) == fold["val_examples"] and set(val_manifest.pid) == set(fold["val_pids"]),
            "Fold validation manifest differs")
    root = Path(cv["root"])
    response_dir = root / "data/Twin-2K-500/question_catalog_and_human_response_csv"
    c_response = (c_response if c_response is not None else response_dir / "wave4_response.csv").resolve()
    b_response = response_dir / "wave1_3_response.csv"
    require(c_response.is_file() and b_response.is_file(), "Numeric wave1_3_response.csv/wave4_response.csv missing")
    # Fingerprint bytes without parsing any answer values before generation.
    response_hashes = {str(path): core.digest_file(path) for path in (b_response, c_response)}
    eval_out = out / "validation"
    eval_out.mkdir(exist_ok=True)
    evaluation = {"protocol": PROTOCOL, "fold_number": fold["fold_number"],
        "training_config_hash": core.config_digest(training), "cv_config_hash": core.config_digest(cv),
        "fold_config_hash": core.config_digest(fold), "checkpoint": candidate,
        "training_complete_sha256": core.digest_file(out / "training_complete.json"),
        "b_response": str(b_response), "c_response": str(c_response), "batch_size": 1,
        "label_source_files_sha256": response_hashes,
        "max_new_tokens": 8, "do_sample": False, "examples": len(val_manifest),
        "evaluation_policy": "final epoch only", "parsing": "fullmatch-whitespace-[12]-whitespace; invalid-is-wrong",
        "script_sha256": core.digest_file(__file__)}
    eval_config = eval_out / "config.json"
    if eval_config.exists():
        require(read_json(eval_config) == evaluation, "Fold evaluation configuration changed")
    else:
        require(not any(eval_out.iterdir()), "Untracked fold evaluation files")
        write_json(eval_config, evaluation)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    original = AutoTokenizer.from_pretrained(cv["model"], local_files_only=True, padding_side="left")
    for tok in (tokenizer, original):
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    require(tokenizer.get_vocab() == original.get_vocab() and tokenizer.chat_template == original.chat_template
            and all(getattr(tokenizer, k) == getattr(original, k) for k in
                    ("bos_token_id", "eos_token_id", "pad_token_id")), "Checkpoint tokenizer changed")
    by_key = {(str(row.pid), str(row.qid)): (index, row)
              for index, row in zip(val_indices, val_manifest.itertuples())}
    raw_path = eval_out / "raw_predictions.jsonl"
    records = read_resumable_records(raw_path)
    seen = set()
    for record in records:
        key = (record["pid"], record["qid"])
        require(key in by_key and key not in seen, "Unexpected or duplicate saved fold prediction")
        _, metadata = by_key[key]
        require(record["prompt_hash"] == metadata.prompt_hash and record["input_tokens"] == int(metadata.input_tokens)
                and record["prediction"] == base.parse_prediction(record["raw_output"]), "Saved fold prediction changed")
        require(all(isinstance(record.get(k), (int, float)) and math.isfinite(record[k]) and 0 <= record[k] <= 1
                    for k in ("first_token_p1", "first_token_p2")), "Invalid saved digit probabilities")
        seen.add(key)
    pending = [(key, index, metadata) for key, (index, metadata) in by_key.items() if key not in seen]
    if pending:
        import torch
        from transformers import AutoModelForCausalLM, GenerationConfig
        require(torch.cuda.is_available(), "CUDA is required for fold generation")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
        require(model.config.model_type == "qwen2" and sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS,
                "Unexpected fold model architecture/size")
        generation = GenerationConfig(do_sample=False, max_new_tokens=8,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id, use_cache=True, return_dict_in_generate=True, output_scores=True)
        digit_ids = {label: tokenizer.encode(str(label), add_special_tokens=False) for label in (1, 2)}
        require(all(len(ids) == 1 for ids in digit_ids.values()), "Expected one token per answer digit")
        with raw_path.open("a", encoding="utf-8") as handle:
            for key, index, metadata in pending:
                row = pool[index]
                require(set(row) == {"input_ids", "attention_mask"} and
                        len(row["input_ids"]) == int(metadata.input_tokens) <= cv["max_input_tokens"]
                        and len(row["input_ids"]) == len(row["attention_mask"])
                        and all(x == 1 for x in row["attention_mask"]), "Invalid label-free validation token prefix")
                inputs = tokenizer.pad([row], padding=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=generation)
                raw = tokenizer.decode(generated.sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                logits = generated.scores[0][0].float()
                require(bool(torch.isfinite(logits).all()), "Nonfinite fold generation scores")
                probabilities = logits.softmax(dim=-1)
                record = {"pid": key[0], "qid": key[1], "prompt_hash": str(metadata.prompt_hash),
                    "input_tokens": int(metadata.input_tokens), "raw_output": raw,
                    "prediction": base.parse_prediction(raw),
                    "first_token_p1": float(probabilities[digit_ids[1][0]]),
                    "first_token_p2": float(probabilities[digit_ids[2][0]])}
                handle.write(base.dumps(record) + "\n")
                handle.flush()
                records.append(record)
                if len(records) % 50 == 0:
                    print(f"Fold {fold['fold_number']}: generated {len(records)}/{len(val_manifest)}", flush=True)
                del generated, inputs, logits, probabilities
        del model
        gc.collect()
        torch.cuda.empty_cache()
    require(len(records) == len(val_manifest), "Fold predictions incomplete; scoring refused")
    # First parsing of heldout answer values. All other participants are skipped.
    source_hashes = {str(path): core.digest_file(path) for path in (b_response, c_response)}
    require(source_hashes == response_hashes, "Response source changed during fold generation")
    if (eval_out / "summary.json").exists():
        previous = read_json(eval_out / "summary.json")
        require(previous["config_hash"] == core.config_digest(evaluation)
                and previous["label_source_files_sha256"] == source_hashes,
                "Previously scored fold configuration or response source changed")
    pids = set(fold["val_pids"])
    qids = cv.get("qids", list(base.QIDS))
    labels_b = read_selected_labels(b_response, pids, qids)
    labels_c = read_selected_labels(c_response, pids, qids)
    summary = score_predictions(records, val_manifest, labels_b, labels_c,
                                fold["majority_prediction"], eval_out, qids)
    summary.update(fold_number=fold["fold_number"], checkpoint=candidate["name"], epoch=candidate["epoch"],
        config_hash=core.config_digest(evaluation),
        label_source_files_sha256=source_hashes)
    write_json(eval_out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    method = mp.get_start_method(allow_none=True)
    require(method in (None, "spawn"), "Worker processes require spawn")
    if method is None:
        mp.set_start_method("spawn")
    import qwen_cv_data as data
    cv, pool, manifest, _ = data.load_cv(args.cv_root)
    fold, dataset = data.prepare_fold(args.cv_root, args.fold)
    out = args.cv_root / f"fold_{args.fold}"
    require(set(fold["train_pids"]).isdisjoint(fold["val_pids"]) and "574" not in fold["val_pids"],
            "Participant leakage or training-only participant in heldout fold")
    config = make_training_config(cv, fold, out, len(dataset))
    if args.prepare_only:
        print(f"Fold {args.fold} verified: {len(dataset)} training examples, "
              f"{fold['val_examples']} heldout prompts. No GPU weights loaded.", flush=True)
        return
    if not args.evaluate_only:
        train_fold(cv, fold, dataset, out, config)
    if not args.train_only:
        evaluate_fold(cv, fold, pool, manifest, out, config, args.c_response)


if __name__ == "__main__":
    main()
