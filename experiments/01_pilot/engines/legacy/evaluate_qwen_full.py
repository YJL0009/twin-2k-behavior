#!/usr/bin/env python3
"""Evaluate Qwen base and full-FT epochs on frozen participant validation prompts.

Messages and selected A answers are identical to Liquid plain-target v3. They
are retokenized with Qwen's tokenizer. Greedy generation, the eight-token cap,
and strict complete-answer parsing remain fixed. No test participants are read.
"""
import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path

import run_liquid_baseline as base
import run_liquid_plain_baseline as plain

PROTOCOL = "qwen-nine-binary-a-to-b-full-fixed-content-v1"
EXPECTED_PARAMETERS = 494_032_768


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_files(folder, hashes, description):
    folder = Path(folder).resolve()
    base.require(bool(hashes), f"Missing {description} file hashes")
    for filename, expected in hashes.items():
        path = (folder / filename).resolve()
        base.require(folder in path.parents and path.is_file()
                     and base.digest_file(path) == expected,
                     f"{description} artifact missing or changed: {filename}")


def verify_frozen_prompt(prompt, tokenizer, original_tokenizer, budget):
    base.require(base.digest_text(base.dumps(prompt["messages"])) == prompt["prompt_hash"],
                 "Frozen validation prompt hash changed")
    kwargs = dict(tokenize=True, add_generation_prompt=True,
                  return_dict=False, truncation=False)
    original = original_tokenizer.apply_chat_template(prompt["messages"], **kwargs)
    current = tokenizer.apply_chat_template(prompt["messages"], **kwargs)
    base.require(isinstance(current, list) and all(isinstance(i, int) for i in current),
                 "Expected a flat list of token IDs")
    base.require(current == original, "Checkpoint tokenizer changes validation token IDs")
    base.require(0 < len(current) == prompt["input_tokens"] <= budget,
                 "Frozen validation input count changed or exceeds budget")


def load_protocol(root, train_run):
    """Verify manifests without reading held-out B/C answer values."""
    import pandas as pd
    config_path = train_run / "run_config.json"
    training = read_json(config_path)
    base.require(training["protocol"] == PROTOCOL, "Expected the Qwen full-parameter protocol")
    base.require(set(training["qids"]) == set(base.QIDS), "Unexpected question subset")
    budget, max_new = training["max_input_tokens"], training["max_new_tokens"]
    base.require(isinstance(budget, int) and 0 < budget <= 32760 and max_new == 8,
                 "Unexpected Qwen input or generation budget")
    preparation = read_json(train_run / "preprocessing_summary.json")
    base.require(preparation["config_hash"] == base.digest_text(base.dumps(training)),
                 "Prepared data no longer matches the training configuration")
    required_artifacts = {"train.arrow", "training_manifest.csv", "mask_audit.json", "validation_prompts.jsonl"}
    base.require(required_artifacts.issubset(preparation["artifacts_sha256"]),
                 "Prepared-data artifact hashes are incomplete")
    for filename in required_artifacts:
        folder = train_run / "prepared" if filename in {"train.arrow", "validation_prompts.jsonl"} else train_run
        verify_files(folder, {filename: preparation["artifacts_sha256"][filename]}, "Prepared data")
    verify_files(Path(training["model"]), training["model_files_sha256"], "Original Qwen model")
    anchor = training["validation_baseline"]
    liquid_run = Path(anchor["path"]).resolve()
    for filename, key in (("config.json", "config_sha256"),
                          ("prepared_prompts.jsonl", "prepared_prompts_sha256"),
                          ("summary.json", "summary_sha256")):
        base.require(base.digest_file(liquid_run / filename) == anchor[key],
                     f"Frozen Liquid prompt source changed: {filename}")
    source_config = read_json(liquid_run / "config.json")
    base.require(source_config["protocol"] == "liquid-nine-binary-fixed-a-plain-target-v3",
                 "Expected the Liquid plain-target v3 prompt source")
    split_path = root / "data/participant_split.csv"
    manifest_path = root / "outputs/majority_baseline_v1/predictions.csv"
    for path in (split_path, manifest_path):
        base.require(base.digest_file(path) == training["inputs_sha256"][str(path)],
                     "Validation split or scoring manifest changed")
        base.require(training["inputs_sha256"][str(path)] == source_config["inputs_sha256"][str(path)],
                     "Qwen and Liquid do not use the same participant/scoring manifests")
    split = pd.read_csv(split_path, dtype={"pid": "string"})
    base.require(split.pid.notna().all() and split.pid.is_unique
                 and split["split"].value_counts().to_dict()
                 == {"train": 1440, "validation": 309, "test": 309}, "Invalid participant split")
    manifest = pd.read_csv(manifest_path, usecols=["pid", "qid"], dtype="string")
    expected_keys = list(manifest.itertuples(index=False, name=None))
    base.require(len(expected_keys) == 1545 and len(set(expected_keys)) == 1545
                 and manifest.notna().all().all() and set(manifest.qid) == set(base.QIDS),
                 "Validation manifest mismatch")
    base.require(set(manifest.pid) == set(split.loc[split["split"] == "validation", "pid"]),
                 "Only the 309 validation participants may be evaluated")
    prompts = base.read_jsonl(train_run / "prepared/validation_prompts.jsonl")
    source_prompts = base.read_jsonl(liquid_run / "prepared_prompts.jsonl")
    base.require([(p["pid"], p["qid"]) for p in prompts] == expected_keys
                 == [(p["pid"], p["qid"]) for p in source_prompts],
                 "Frozen prompt manifest mismatch")
    for current, source in zip(prompts, source_prompts):
        base.require(current["messages"] == source["messages"]
                     and current["prompt_hash"] == source["prompt_hash"]
                     and current["source_input_tokens"] == source["input_tokens"],
                     "Qwen must use exactly the Liquid v3 messages and selected A content")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "tokenizers", "pandas")}
    for name in ("torch", "transformers", "tokenizers"):
        base.require(versions[name] == training["versions"][name],
                     f"Use the Qwen training run's {name} version")
    helpers = {Path(p).name: base.digest_file(p) for p in
               (base.__file__, plain.__file__, plain.diag.__file__)}
    if "helper_scripts_sha256" in training:
        for name, digest in helpers.items():
            base.require(training["helper_scripts_sha256"].get(name) == digest,
                         f"Evaluation helper changed: {name}")
    return training, prompts, manifest_path, versions


def completed_candidates(train_run, training):
    completion = read_json(train_run / "training_complete.json")
    base.require(completion["protocol"] == PROTOCOL
                 and completion["config_hash"] == base.digest_text(base.dumps(training)),
                 "Training completion/config mismatch")
    base.require(completion["base_parameters"] == EXPECTED_PARAMETERS
                 and completion["trainable_parameters"] == EXPECTED_PARAMETERS
                 and completion["total_training_parameters"] == EXPECTED_PARAMETERS,
                 "Unexpected completed model parameter count")
    candidates = sorted(completion["candidate_checkpoints"], key=lambda c: c["global_step"])
    base.require(len(candidates) == training["epochs"] > 0
                 and len({c["name"] for c in candidates}) == len(candidates),
                 "Expected one distinct checkpoint per completed epoch")
    base.require([float(c["epoch"]) for c in candidates] == list(range(1, training["epochs"] + 1)),
                 "Checkpoint epochs differ from the training schedule")
    return candidates


def evaluate_model(train_run, training, prompts, manifest_path, versions,
                   candidate=None, prepare_only=False):
    from transformers import AutoTokenizer
    is_baseline = candidate is None
    name = "base" if is_baseline else candidate["name"]
    out = train_run / "validation" / name
    model_path = Path(training["model"]).resolve() if is_baseline else (train_run / candidate["path"]).resolve()
    if not is_baseline:
        base.require(train_run in model_path.parents and model_path.name == name,
                     "Invalid full-model checkpoint path")
        verify_files(model_path, candidate["model_files_sha256"], "Full-model checkpoint")
    base.require((model_path / "config.json").is_file(), "Full-model config missing")
    budget, max_new = training["max_input_tokens"], training["max_new_tokens"]
    config = {"protocol": "qwen-full-fixed-content-validation-v1", "training_run": str(train_run),
        "model": str(model_path), "stage": "zero_shot" if is_baseline else "full_finetuning",
        "training_config_sha256": base.digest_file(train_run / "run_config.json"),
        "preprocessing_summary_sha256": base.digest_file(train_run / "preprocessing_summary.json"),
        "checkpoint": candidate, "validation_baseline": training["validation_baseline"],
        "qids": list(base.QIDS), "examples": len(prompts), "batch_size": 1,
        "max_input_tokens": budget, "max_new_tokens": max_new, "do_sample": False,
        "evaluation_split": "validation", "evaluation_labels": "C",
        "parsing": "fullmatch-whitespace-[12]-whitespace; invalid-is-wrong",
        "script_files_sha256": {Path(p).name: base.digest_file(p) for p in
            (__file__, base.__file__, plain.__file__, plain.diag.__file__)}, "versions": versions}
    if not is_baseline:
        config["training_complete_sha256"] = base.digest_file(train_run / "training_complete.json")
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists():
        base.require(read_json(config_path) == config,
                     "Evaluation configuration changed; preserve existing results and use a new training run")
    else:
        base.require(not any(out.iterdir()), "Evaluation directory contains untracked files")
        base.write_json(config_path, config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    original = AutoTokenizer.from_pretrained(training["model"], local_files_only=True, padding_side="left")
    for item in (tokenizer, original):
        if item.pad_token_id is None:
            item.pad_token = item.eos_token
    base.require(tokenizer.chat_template == original.chat_template,
                 "Checkpoint tokenizer chat template changed")
    base.require(all(getattr(tokenizer, k) == getattr(original, k)
                     for k in ("bos_token_id", "eos_token_id", "pad_token_id")),
                 "Checkpoint tokenizer special token IDs changed")
    for index, prompt in enumerate(prompts, 1):
        verify_frozen_prompt(prompt, tokenizer, original, budget)
        base.tokenize_batch(tokenizer, [prompt], budget)
        if index % 300 == 0:
            print(f"Verified Qwen validation token IDs: {index}/{len(prompts)}", flush=True)
    print(f"All {len(prompts)} prompts retain the same messages and Qwen token IDs.", flush=True)
    if prepare_only:
        print("Verification complete; no GPU inference or scoring performed.", flush=True)
        return None
    raw_path = out / "raw_predictions.jsonl"
    records = base.read_jsonl(raw_path)
    by_key = {(p["pid"], p["qid"]): p for p in prompts}
    seen = set()
    for record in records:
        key = (record["pid"], record["qid"])
        base.require(key in by_key and key not in seen, "Invalid or duplicate saved prediction")
        base.require(record["prompt_hash"] == by_key[key]["prompt_hash"]
                     and record["input_tokens"] == by_key[key]["input_tokens"], "Saved prompt changed")
        base.require(record["prediction"] == base.parse_prediction(record["raw_output"]),
                     "Saved prediction parser differs")
        base.require(all(isinstance(record.get(k), (int, float)) and math.isfinite(record[k])
                         and 0 <= record[k] <= 1 for k in ("first_token_p1", "first_token_p2")),
                     "Saved digit probabilities are missing or invalid")
        seen.add(key)
    pending = [p for p in prompts if (p["pid"], p["qid"]) not in seen]
    if pending:
        import torch
        from transformers import AutoModelForCausalLM, GenerationConfig
        base.require(torch.cuda.is_available(), "CUDA unavailable")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                    dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
        base.require(model.config.model_type == "qwen2"
                     and sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS,
                     "Unexpected Qwen model architecture or parameter count")
        base.require(budget + max_new <= model.config.max_position_embeddings, "Model context too short")
        generation = GenerationConfig(do_sample=False, max_new_tokens=max_new,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id, use_cache=True,
            return_dict_in_generate=True, output_scores=True)
        digit_ids = {label: tokenizer.encode(str(label), add_special_tokens=False) for label in (1, 2)}
        base.require(all(len(ids) == 1 for ids in digit_ids.values()), "Expected single-token answer digits")
        with raw_path.open("a", encoding="utf-8") as handle:
            for prompt in pending:
                inputs = base.tokenize_batch(tokenizer, [prompt], budget, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=generation)
                raw = tokenizer.decode(generated.sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                logits = generated.scores[0][0].float()
                base.require(bool(torch.isfinite(logits).all()), "Non-finite first-token scores")
                probabilities = logits.softmax(dim=-1)
                record = {key: prompt[key] for key in ("pid", "qid", "prompt_hash", "input_tokens")}
                record.update(raw_output=raw, prediction=base.parse_prediction(raw),
                              first_token_p1=float(probabilities[digit_ids[1][0]]),
                              first_token_p2=float(probabilities[digit_ids[2][0]]))
                records.append(record)
                handle.write(base.dumps(record) + "\n")
                handle.flush()
                if len(records) % 50 == 0:
                    print(f"{name}: generated {len(records)}/{len(prompts)}", flush=True)
                del generated, inputs, logits, probabilities
        del model
        gc.collect()
        torch.cuda.empty_cache()
    base.require(len(records) == len(prompts), "Incomplete predictions; scoring refused")
    # The scoring helper is the first operation that reads held-out answer values.
    plain.score_complete(manifest_path, records, out)
    after = read_json(out / "summary.json")
    if not is_baseline:
        baseline_dir = train_run / "validation/base"
        base.require(read_json(baseline_dir / "config.json")["preprocessing_summary_sha256"]
                     == config["preprocessing_summary_sha256"], "Qwen baseline used different prepared inputs")
        before = read_json(baseline_dir / "summary.json")
        comparison = {"zero_shot_model": "Qwen2.5-0.5B-Instruct; identical fixed-content prompts",
            "zero_shot_macro_accuracy": before["question_macro_accuracy"],
            "full_ft_macro_accuracy": after["question_macro_accuracy"],
            "majority_macro_accuracy": after["majority_macro_accuracy"],
            "gain_over_zero_shot_pp": 100 * (after["question_macro_accuracy"] - before["question_macro_accuracy"]),
            "gain_over_majority_pp": 100 * (after["question_macro_accuracy"] - after["majority_macro_accuracy"]),
            "invalid_outputs": after["invalid_outputs"], "identical_validation_prompts": True,
            "identical_validation_messages_to_liquid": True}
        base.write_json(out / "comparison.json", comparison)
        print("Comparison:", json.dumps(comparison, indent=2), flush=True)
    return after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--train-run", type=Path, default=Path("outputs/qwen_full_v1"))
    parser.add_argument("--checkpoint", default="all", help="all (default) or a completed checkpoint name")
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate original Qwen weights only")
    parser.add_argument("--prepare-only", action="store_true", help="Verify inputs/tokenizers without GPU or scoring")
    args = parser.parse_args()
    root = args.root.resolve()
    train_run = (args.train_run if args.train_run.is_absolute() else root / args.train_run).resolve()
    training, prompts, manifest_path, versions = load_protocol(root, train_run)
    candidates = [] if args.baseline_only else completed_candidates(train_run, training)
    chosen = candidates if args.checkpoint == "all" else [c for c in candidates if c["name"] == args.checkpoint]
    base.require(args.baseline_only or bool(chosen), "Checkpoint name is not registered in this completed run")
    print("Evaluating matched Qwen zero-shot baseline", flush=True)
    evaluate_model(train_run, training, prompts, manifest_path, versions, prepare_only=args.prepare_only)
    if args.baseline_only:
        return
    results = []
    for candidate in chosen:
        print(f"Evaluating epoch {candidate['epoch']}: {candidate['name']}", flush=True)
        summary = evaluate_model(train_run, training, prompts, manifest_path, versions,
                                 candidate, args.prepare_only)
        if summary is not None:
            results.append({"checkpoint": candidate["name"], "model_path": str(train_run / candidate["path"]),
                "epoch": candidate["epoch"], "question_macro_accuracy": summary["question_macro_accuracy"],
                "pooled_accuracy": summary["pooled_accuracy"], "invalid_outputs": summary["invalid_outputs"],
                "majority_macro_accuracy": summary["majority_macro_accuracy"],
                "gain_over_majority_pp": 100 * (summary["question_macro_accuracy"] - summary["majority_macro_accuracy"])})
    if args.prepare_only:
        return
    import pandas as pd
    table = pd.DataFrame(results)
    print(table.to_string(index=False), flush=True)
    if args.checkpoint == "all":
        table.to_csv(train_run / "epoch_comparison.csv", index=False)
        best = max(results, key=lambda r: (r["question_macro_accuracy"], -r["epoch"]))
        selection = {"selection_split": "validation", "selection_labels": "C",
            "criterion": "highest question-macro accuracy; ties choose earliest epoch",
            "best": best, "all_epochs": results, "test_evaluated": False,
            "training_complete_sha256": base.digest_file(train_run / "training_complete.json")}
        base.write_json(train_run / "best_checkpoint.json", selection)
        print("Selected checkpoint:", json.dumps(best, indent=2), flush=True)
        print("Keep this checkpoint and the scoring rule fixed for the final test evaluation.", flush=True)


if __name__ == "__main__":
    main()
