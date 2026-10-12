#!/usr/bin/env python3
"""Evaluate Liquid with plain targets and the exact A contexts saved by v2.

Requires run_liquid_baseline.py and diagnose_liquid_outputs.py beside this file.
Use the Liquid Transformers 5.2 environment. The first example per question is
a nine-example preflight; default execution then continues automatically.
--smoke-only stops after that preflight and can later resume in the same folder.
No context reselection, fine-tuning, or test-set evaluation occurs.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path

import run_liquid_baseline as base
import diagnose_liquid_outputs as diag


def prepare_plain_prompt(tokenizer, source, budget):
    """Change only target representation; preserve every selected A unit."""
    diag.split_prompt(source)
    base.require(base.token_count(tokenizer, source["messages"]) == source["input_tokens"] <= budget,
                 "Source prompt token count mismatch")
    case = diag.make_case("plain_target", source, plain=True)
    source_context = source["messages"][1]["content"].split(diag.MARKER)[0]
    new_context = case["messages"][1]["content"].split(diag.MARKER)[0]
    base.require(source_context == new_context, "Selected A context changed")
    prompt = dict(source)
    prompt.update(source_prompt_hash=source["prompt_hash"], source_input_tokens=source["input_tokens"],
                  messages=case["messages"], prompt_hash=base.digest_text(base.dumps(case["messages"])),
                  input_tokens=base.token_count(tokenizer, case["messages"]))
    base.require(0 < prompt["input_tokens"] <= budget, "Plain target prompt exceeds original input budget")
    return prompt


def preflight_keys(prompts):
    return [(next(p for p in prompts if p["qid"] == qid)["pid"], qid) for qid in base.QIDS]


def score_complete(manifest_path, records, out):
    """The first stage allowed to read C and human-reference answer values."""
    import pandas as pd
    labels = pd.read_csv(manifest_path, dtype={"pid": "string", "qid": "string"})
    labels = labels.rename(columns={"prediction": "majority_prediction"})
    base.require(len(records) == len(labels), "Incomplete prediction set; scoring refused")
    scores = labels[["pid", "qid", "target_c", "reference_b", "majority_prediction"]].merge(
        pd.DataFrame(records), on=["pid", "qid"], how="left", validate="one_to_one")
    base.require(scores.target_c.isin([1, 2]).all(), "Invalid validation C labels")
    base.require(scores.reference_b.dropna().isin([1, 2]).all(), "Invalid human reference labels")
    scores["valid"] = scores.prediction.isin([1, 2])
    scores["correct"] = scores.valid & scores.prediction.eq(scores.target_c)
    scores["majority_correct"] = scores.majority_prediction.eq(scores.target_c)
    paired = scores.reference_b.notna()
    scores["human_correct"] = scores.reference_b.eq(scores.target_c).astype(float).where(paired)
    scores["model_correct_paired"] = scores.correct.astype(float).where(paired)
    scores["majority_correct_paired"] = scores.majority_correct.astype(float).where(paired)
    metrics = scores.groupby("qid").agg(examples=("correct", "size"), model_accuracy=("correct", "mean"),
        valid_rate=("valid", "mean"), majority_accuracy=("majority_correct", "mean"),
        human_pairs=("human_correct", "count"), human_reference=("human_correct", "mean"),
        model_accuracy_paired=("model_correct_paired", "mean"),
        majority_accuracy_paired=("majority_correct_paired", "mean"))
    summary = {"evaluation_split": "validation", "evaluation_labels": "C", "question_subset": list(base.QIDS),
        "examples": len(scores), "question_macro_accuracy": float(metrics.model_accuracy.mean()),
        "pooled_accuracy": float(scores.correct.mean()), "invalid_outputs": int((~scores.valid).sum()),
        "invalid_rate": float((~scores.valid).mean()),
        "majority_macro_accuracy": float(metrics.majority_accuracy.mean()),
        "majority_pooled_accuracy": float(scores.majority_correct.mean()),
        "human_reference_macro_paired": float(metrics.human_reference.mean()),
        "model_macro_accuracy_paired": float(metrics.model_accuracy_paired.mean()),
        "majority_macro_accuracy_paired": float(metrics.majority_accuracy_paired.mean())}
    scores.to_csv(out / "scored_predictions.csv", index=False)
    metrics.to_csv(out / "metrics.csv")
    counts = scores.groupby("qid").agg(examples=("pid", "size"),
        predicted_1=("prediction", lambda values: int(values.eq(1).sum())),
        predicted_2=("prediction", lambda values: int(values.eq(2).sum())),
        invalid=("prediction", lambda values: int((~values.isin([1, 2])).sum())))
    counts.to_csv(out / "prediction_counts.csv")
    base.write_json(out / "summary.json", summary)
    print(metrics.round(4).to_string())
    print(json.dumps(summary, indent=2))
    print("Prediction counts per question:")
    print(counts.to_string())
    print("Saved results:", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/models/LFM2.5-350M"))
    parser.add_argument("--source-run", type=Path, default=Path("outputs/liquid_baseline_tf52_v2"))
    parser.add_argument("--out", type=Path, default=Path("outputs/liquid_plain_target_v3"))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    import pandas as pd
    from transformers import AutoTokenizer
    root, model_path = args.root.resolve(), args.model.resolve()
    source_run = (args.source_run if args.source_run.is_absolute() else root / args.source_run).resolve()
    out = (args.out if args.out.is_absolute() else root / args.out).resolve()
    base.require(out != source_run and source_run not in out.parents, "Choose an output folder outside the source run")
    split_path = root / "data/participant_split.csv"
    manifest_path = root / "outputs/majority_baseline_v1/predictions.csv"
    source_config_path = source_run / "config.json"
    source_prepared_path = source_run / "prepared_prompts.jsonl"
    source_config = json.loads(source_config_path.read_text())
    base.require(source_config.get("protocol") == "liquid-nine-binary-a-only-v2", "Expected the v2 source protocol")
    base.require(source_config.get("qids") == list(base.QIDS) and source_config.get("examples") == 1545,
                 "Unexpected source question manifest")
    base.require(source_config.get("do_sample") is False, "Expected greedy source decoding")
    budget, max_new = source_config["max_input_tokens"], source_config["max_new_tokens"]
    base.require(budget == 8192 and max_new == 8, "Expected established 8192/8 source token limits")
    split = pd.read_csv(split_path, dtype={"pid": "string"})
    base.require(split.pid.notna().all() and split.pid.is_unique, "Invalid participant split IDs")
    base.require(split["split"].value_counts().to_dict() == {"train": 1440, "validation": 309, "test": 309},
                 "Unexpected participant split sizes")
    base.require(split.set_index("pid").loc["574", "split"] == "train", "Participant 574 must remain in training")
    manifest = pd.read_csv(manifest_path, usecols=["pid", "qid"], dtype="string")
    base.require(len(manifest) == 1545 and not manifest.duplicated().any() and manifest.notna().all().all(),
                 "Expected 1545 unique validation examples")
    base.require(set(manifest.qid) == set(base.QIDS), "Manifest question mismatch")
    base.require(set(manifest.pid) == set(split.loc[split["split"] == "validation", "pid"]),
                 "Manifest must contain exactly the validation participants")
    input_hashes = {str(p): base.digest_file(p) for p in (split_path, manifest_path)}
    for path, digest in input_hashes.items():
        base.require(source_config["inputs_sha256"].get(path) == digest, "Split or manifest changed since v2")
    model_files = sorted(p for p in model_path.iterdir() if p.is_file() and
                        (p.suffix in {".json", ".safetensors", ".model", ".jinja"} or p.name == "merges.txt"))
    model_hashes = {p.name: base.digest_file(p) for p in model_files}
    base.require(model_hashes and model_hashes == source_config["model_files_sha256"],
                 "Model/tokenizer files differ from v2; this comparison requires the exact same files")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "tokenizers", "pandas")}
    for name in ("torch", "transformers", "tokenizers"):
        if name in source_config.get("versions", {}):
            base.require(versions[name] == source_config["versions"][name],
                         f"{name} version differs from v2; use the same Liquid environment")
    sources = base.read_jsonl(source_prepared_path)
    expected_keys = list(manifest.itertuples(index=False, name=None))
    base.require([(p["pid"], p["qid"]) for p in sources] == expected_keys, "Source prompt manifest/order mismatch")
    config = {"protocol": "liquid-nine-binary-fixed-a-plain-target-v3", "root": str(root),
        "model": str(model_path), "source_run": str(source_run), "qids": list(base.QIDS), "examples": 1545,
        "batch_size": 1, "max_input_tokens": budget, "max_new_tokens": max_new, "do_sample": False,
        "context_policy": "exact-v2-selected-A-units-no-reselection-or-refill",
        "target_policy": "diagnose_liquid_outputs.plain_target; substantive text and ordered options preserved",
        "parsing": "fullmatch-whitespace-[12]-whitespace; invalid-is-wrong",
        "preflight": "first-source-example-per-question; stop-if-all-nine-invalid; reuse-predictions",
        "inputs_sha256": input_hashes, "model_files_sha256": model_hashes,
        "source_files_sha256": {p.name: base.digest_file(p) for p in (source_config_path, source_prepared_path)},
        "script_files_sha256": {Path(p).name: base.digest_file(p) for p in (__file__, base.__file__, diag.__file__)},
        "versions": versions}
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists():
        base.require(json.loads(config_path.read_text()) == config, "Existing configuration differs; choose a new --out")
    else:
        base.require(not any(out.iterdir()), "Output folder is not empty; choose a new --out")
        base.write_json(config_path, config)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = []
    for index, source in enumerate(sources, 1):
        prompts.append(prepare_plain_prompt(tokenizer, source, budget))
        if index % 100 == 0:
            print(f"Prepared {index}/{len(sources)} plain-target prompts", flush=True)
    prepared_path = out / "prepared_prompts.jsonl"
    if prepared_path.exists():
        base.require(base.read_jsonl(prepared_path) == prompts, "Saved transformed prompts differ")
    else:
        temporary = prepared_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for prompt in prompts:
                handle.write(base.dumps(prompt) + "\n")
        temporary.replace(prepared_path)
    for prompt in prompts:
        base.tokenize_batch(tokenizer, [prompt], budget)
    print(f"Verified {len(prompts)} fixed-A plain-target prompts; maximum {max(p['input_tokens'] for p in prompts)} tokens.", flush=True)
    by_key = {(p["pid"], p["qid"]): p for p in prompts}
    raw_path = out / "raw_predictions.jsonl"
    records = base.read_jsonl(raw_path)
    seen = {}
    for record in records:
        key = (record["pid"], record["qid"])
        base.require(key in by_key and key not in seen, "Invalid or duplicate saved prediction")
        base.require(record["prompt_hash"] == by_key[key]["prompt_hash"]
                     and record["input_tokens"] == by_key[key]["input_tokens"], "Saved prediction prompt changed")
        base.require(record["prediction"] == base.parse_prediction(record["raw_output"]), "Saved parsing mismatch")
        seen[key] = record
    smoke_keys = preflight_keys(prompts)
    order = smoke_keys + [key for key in expected_keys if key not in set(smoke_keys)]
    if args.smoke_only:
        order = smoke_keys
    model = None
    generation = None

    def report_preflight():
        valid = 0
        for key in smoke_keys:
            record = seen[key]
            valid += record["prediction"] in (1, 2)
            print(f"Preflight {key[1]} pid={key[0]} | raw={record['raw_output']!r} | valid={record['prediction'] in (1, 2)}", flush=True)
        print(f"Preflight valid: {valid}/9. No accuracy scoring performed.", flush=True)
        return valid > 0

    preflight_complete = all(key in seen for key in smoke_keys)
    if preflight_complete:
        if not report_preflight():
            print("Stopped: all nine preflight responses are invalid. Inspect saved raw outputs before a full run.", flush=True)
            return
    with raw_path.open("a", encoding="utf-8") as handle:
        for key in order:
            if key in seen:
                continue
            if model is None:
                import torch
                from transformers import AutoModelForCausalLM, GenerationConfig
                base.require(torch.cuda.is_available(), "CUDA unavailable")
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                            dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
                parameters = sum(p.numel() for p in model.parameters())
                base.require(parameters < 500_000_000, "Model exceeds parameter limit")
                base.require(budget + max_new <= model.config.max_position_embeddings, "Model context too short")
                generation = GenerationConfig(do_sample=False, max_new_tokens=max_new,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id, use_cache=True)
                print(f"Loaded {model.config.model_type}: {parameters:,} parameters, {model.dtype}", flush=True)
            prompt = by_key[key]
            inputs = base.tokenize_batch(tokenizer, [prompt], budget, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=generation)
            raw = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            record = {k: prompt[k] for k in ("pid", "qid", "prompt_hash", "input_tokens", "source_prompt_hash")}
            record.update(raw_output=raw, prediction=base.parse_prediction(raw))
            records.append(record)
            seen[key] = record
            handle.write(base.dumps(record) + "\n")
            handle.flush()
            if not preflight_complete and all(smoke_key in seen for smoke_key in smoke_keys):
                preflight_complete = True
                if not report_preflight():
                    print("Stopped: all nine preflight responses are invalid. Saved records can be inspected.", flush=True)
                    return
            if len(records) % 50 == 0:
                print(f"Generated {len(records)}/{len(prompts)}", flush=True)
    if args.smoke_only:
        print("Preflight complete; no accuracy scoring performed. Rerun without --smoke-only to continue.", flush=True)
        print("Saved preflight:", out, flush=True)
        return
    base.require(set(seen) == set(expected_keys), "Incomplete prediction set; scoring refused")
    score_complete(manifest_path, records, out)


if __name__ == "__main__":
    main()
