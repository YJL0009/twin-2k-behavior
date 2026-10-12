#!/usr/bin/env python3
"""Evaluate every full-FT epoch on the frozen plain-target validation prompts.

The default evaluates all completed epoch checkpoints, preserving unrestricted
8-token greedy decoding and strict output parsing. Select by validation C
question-macro accuracy; the test split is never evaluated. Rerunning resumes
partial predictions. No thresholds are fitted or changed.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path

import run_liquid_baseline as base
import run_liquid_plain_baseline as plain


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


def evaluate_checkpoint(root, train_run, candidate, prepare_only=False):
    import pandas as pd
    from transformers import AutoTokenizer
    out = train_run / "validation" / candidate["name"]
    run_config_path = train_run / "run_config.json"
    completion_path = train_run / "training_complete.json"
    base.require(completion_path.exists(), "Training has not completed; no validation evaluation started")
    training = json.loads(run_config_path.read_text())
    completion = json.loads(completion_path.read_text())
    base.require(training["protocol"] == "liquid-nine-binary-a-to-b-full-plain-target-v1",
                 "Expected the Liquid full-parameter training protocol")
    base.require(completion["config_hash"] == base.digest_text(base.dumps(training)),
                 "Training configuration no longer matches its completion record")
    base.require(completion["base_parameters"] == 354_483_968
                 and completion["trainable_parameters"] == 354_483_968
                 and completion["total_training_parameters"] == 354_483_968,
                 "Unexpected completed model parameter count")
    model_path = (train_run / candidate["path"]).resolve()
    base.require(train_run in model_path.parents and model_path.name == candidate["name"],
                 "Invalid full-model checkpoint path")
    for filename, expected in candidate["model_files_sha256"].items():
        path = (model_path / filename).resolve()
        base.require(model_path in path.parents and path.is_file()
                     and base.digest_file(path) == expected,
                     f"Full-model checkpoint artifact missing or changed: {filename}")
    base.require((model_path / "config.json").is_file(), "Full-model config missing")
    anchor = training["validation_baseline"]
    baseline_run = Path(anchor["path"]).resolve()
    base.require(out != baseline_run and baseline_run not in out.parents,
                 "Use an output folder outside the frozen baseline")
    for filename, key in (("config.json", "config_sha256"),
                          ("prepared_prompts.jsonl", "prepared_prompts_sha256"),
                          ("summary.json", "summary_sha256")):
        base.require(base.digest_file(baseline_run / filename) == anchor[key],
                     f"Frozen baseline file changed: {filename}")
    baseline_config = json.loads((baseline_run / "config.json").read_text())
    base.require(baseline_config["protocol"] == "liquid-nine-binary-fixed-a-plain-target-v3",
                 "Expected the completed plain-target v3 baseline")
    base.require(training["model_files_sha256"] == baseline_config["model_files_sha256"],
                 "Training did not start from the same base-model files as v3")
    helper_hashes = {Path(path).name: base.digest_file(path) for path in
                     (base.__file__, plain.__file__, plain.diag.__file__)}
    base.require(helper_hashes == baseline_config["script_files_sha256"]
                 == training["helper_scripts_sha256"], "Baseline helper scripts changed")
    budget, max_new = baseline_config["max_input_tokens"], baseline_config["max_new_tokens"]
    base.require((budget, max_new) == (8192, 8) and training["max_input_tokens"] == budget,
                 "Unexpected baseline/training token limits")
    split_path = root / "data/participant_split.csv"
    manifest_path = root / "outputs/majority_baseline_v1/predictions.csv"
    for path in (split_path, manifest_path):
        base.require(base.digest_file(path) == baseline_config["inputs_sha256"][str(path)],
                     "Validation split or scoring manifest changed")
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
    prompts = base.read_jsonl(baseline_run / "prepared_prompts.jsonl")
    base.require([(p["pid"], p["qid"]) for p in prompts] == expected_keys,
                 "Frozen prompt manifest mismatch")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "tokenizers", "pandas")}
    for name in ("torch", "transformers", "tokenizers"):
        base.require(versions[name] == training["versions"][name]
                     == baseline_config["versions"][name], f"Use the original Liquid {name} version")
    config = {"protocol": "liquid-full-frozen-v3-validation-v1", "training_run": str(train_run),
        "baseline_run": str(baseline_run), "model": str(model_path),
        "training_config_sha256": base.digest_file(run_config_path),
        "training_complete_sha256": base.digest_file(completion_path),
        "checkpoint": candidate,
        "validation_baseline": anchor, "qids": list(base.QIDS), "examples": len(prompts),
        "batch_size": 1, "max_input_tokens": budget, "max_new_tokens": max_new,
        "do_sample": False, "evaluation_split": "validation", "evaluation_labels": "C",
        "parsing": "fullmatch-whitespace-[12]-whitespace; invalid-is-wrong",
        "script_files_sha256": {Path(p).name: base.digest_file(p) for p in
            (__file__, base.__file__, plain.__file__, plain.diag.__file__)}, "versions": versions}
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists():
        base.require(json.loads(config_path.read_text()) == config,
                     "Evaluation configuration differs; choose a new --out")
    else:
        base.require(not any(out.iterdir()), "Output folder is not empty; choose a new --out")
        base.write_json(config_path, config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    original_tokenizer = AutoTokenizer.from_pretrained(training["model"], local_files_only=True, padding_side="left")
    for item in (tokenizer, original_tokenizer):
        if item.pad_token_id is None:
            item.pad_token = item.eos_token
    base.require(tokenizer.chat_template == original_tokenizer.chat_template,
                 "Checkpoint tokenizer chat template changed")
    base.require(all(getattr(tokenizer, k) == getattr(original_tokenizer, k)
                     for k in ("bos_token_id", "eos_token_id", "pad_token_id")),
                 "Checkpoint tokenizer special token IDs changed")
    for index, prompt in enumerate(prompts, 1):
        verify_frozen_prompt(prompt, tokenizer, original_tokenizer, budget)
        base.tokenize_batch(tokenizer, [prompt], budget)
        if index % 300 == 0:
            print(f"Verified frozen prompts: {index}/{len(prompts)}", flush=True)
    print(f"All {len(prompts)} validation prompts and token IDs match the zero-shot baseline.", flush=True)
    if prepare_only:
        print("Verification complete; no GPU inference or scoring performed.", flush=True)
        return
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
        seen.add(key)
    pending = [p for p in prompts if (p["pid"], p["qid"]) not in seen]
    if pending:
        import torch
        from transformers import AutoModelForCausalLM, GenerationConfig
        base.require(torch.cuda.is_available(), "CUDA unavailable")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                    dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
        base.require(sum(p.numel() for p in model.parameters()) == completion["base_parameters"],
                     "Checkpoint model parameter count changed")
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
                    print(f"Generated {len(records)}/{len(prompts)}", flush=True)
                del generated, inputs, logits, probabilities
    base.require(len(records) == len(expected_keys), "Incomplete predictions; scoring refused")
    plain.score_complete(manifest_path, records, out)
    before = json.loads((baseline_run / "summary.json").read_text())
    after = json.loads((out / "summary.json").read_text())
    comparison = {"zero_shot_macro_accuracy": before["question_macro_accuracy"],
        "full_ft_macro_accuracy": after["question_macro_accuracy"],
        "majority_macro_accuracy": after["majority_macro_accuracy"],
        "gain_over_zero_shot_pp": 100 * (after["question_macro_accuracy"] - before["question_macro_accuracy"]),
        "gain_over_majority_pp": 100 * (after["question_macro_accuracy"] - after["majority_macro_accuracy"]),
        "invalid_outputs": after["invalid_outputs"], "identical_validation_prompts": True}
    base.write_json(out / "comparison.json", comparison)
    print("Comparison:", json.dumps(comparison, indent=2), flush=True)
    return after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--train-run", type=Path, default=Path("outputs/liquid_full_v1"))
    parser.add_argument("--checkpoint", default="all", help="all (default) or a recorded checkpoint name")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    train_run = (args.train_run if args.train_run.is_absolute() else root / args.train_run).resolve()
    done = json.loads((train_run / "training_complete.json").read_text())
    training = json.loads((train_run / "run_config.json").read_text())
    base.require(done["config_hash"] == base.digest_text(base.dumps(training)),
                 "Training completion/config mismatch")
    candidates = sorted(done["candidate_checkpoints"], key=lambda c: c["global_step"])
    base.require(len(candidates) == training["epochs"] > 0
                 and len({c["name"] for c in candidates}) == len(candidates),
                 "Expected one distinct checkpoint per completed epoch")
    base.require([float(c["epoch"]) for c in candidates] == list(range(1, training["epochs"] + 1)),
                 "Checkpoint epochs differ from the recorded training schedule")
    chosen = candidates if args.checkpoint == "all" else [c for c in candidates if c["name"] == args.checkpoint]
    base.require(bool(chosen), "Checkpoint name is not registered in this completed training run")
    results = []
    for candidate in chosen:
        print(f"Evaluating epoch {candidate['epoch']}: {candidate['name']}", flush=True)
        summary = evaluate_checkpoint(root, train_run, candidate, args.prepare_only)
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
