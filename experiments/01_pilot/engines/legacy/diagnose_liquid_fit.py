#!/usr/bin/env python3
"""Compare final Liquid training fit with existing held-out validation results.

Default: uniformly sample 128 training examples for each of QID169/184/193,
with seed 42 and no label balancing. --per-question 0 evaluates all 2140.
Both supervised tokens are removed before ordinary greedy eight-token inference.
Uses the original strict full-output parser, never forced binary decoding.
No training occurs; no test responses are read and validation inference is not rerun.
"""
import argparse
import importlib.metadata
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import run_liquid_baseline as base

QIDS = ("QID169", "QID184", "QID193")


def prefix_from_training_example(row, manifest_row, tokenizer):
    """Validate cache/manifest alignment and remove the gold digit AND EOS."""
    ids, labels, mask = (list(row[key]) for key in ("input_ids", "labels", "attention_mask"))
    n = len(ids)
    base.require(n > 2 and n == len(labels) == len(mask), "Malformed training sequence")
    base.require(all(value == 1 for value in mask), "Cached training example contains padding")
    supervised = [i for i, label in enumerate(labels) if label != -100]
    base.require(supervised == [n - 2, n - 1], "Expected exactly two supervised final tokens")
    label = manifest_row["label_b"]
    base.require(label in (1, 2), "Invalid training B label")
    digit = tokenizer.encode(str(int(label)), add_special_tokens=False)
    base.require(len(digit) == 1 and tokenizer.eos_token_id is not None, "Expected one digit token and EOS")
    expected = digit + [tokenizer.eos_token_id]
    base.require(labels[-2:] == ids[-2:] == expected, "Manifest B label differs from supervised cache tokens")
    base.require(n - 2 == int(manifest_row["input_tokens"])
                 and n == int(manifest_row["total_tokens"]), "Cache/manifest token count mismatch")
    return {"input_ids": ids[:-2], "attention_mask": mask[:-2]}


def select_training_indices(manifest, per_question, seed=42):
    base.require(per_question >= 0, "--per-question must be nonnegative")
    selected = []
    rng = random.Random(seed)
    for qid in QIDS:
        indices = manifest.index[manifest.qid.eq(qid)].tolist()
        base.require(indices, f"No training examples for {qid}")
        selected.extend(sorted(indices if per_question == 0 else
                               rng.sample(indices, min(per_question, len(indices)))))
    return selected


def summarize_fit(manifest, results, validation):
    import pandas as pd
    generated = pd.DataFrame(results)
    rows = []
    for qid in QIDS:
        all_train = manifest.loc[manifest.qid.eq(qid), "label_b"]
        majority = int(all_train.mode().min())
        sample = generated.loc[generated.qid.eq(qid)]
        heldout = validation.loc[validation.qid.eq(qid)]
        base.require(len(sample) > 0 and len(heldout) > 0, "Missing question results")
        base.require(heldout.majority_prediction.eq(majority).all(), "Saved majority differs from training B mode")
        valid = sample.prediction.isin([1, 2])
        rows.append({"qid": qid, "train_sample_n": len(sample),
            "train_sample_B_accuracy": float((valid & sample.prediction.eq(sample.label_b)).mean()),
            "train_sample_majority_B_accuracy": float(sample.label_b.eq(majority).mean()),
            "train_valid_rate": float(valid.mean()),
            "train_predicted_1": int(sample.prediction.eq(1).sum()),
            "train_predicted_2": int(sample.prediction.eq(2).sum()),
            "train_invalid": int((~valid).sum()),
            "train_sample_B_1": int(sample.label_b.eq(1).sum()),
            "train_sample_B_2": int(sample.label_b.eq(2).sum()),
            "train_digit_nll_mean": float(sample.gold_digit_nll.mean()),
            "heldout_n": len(heldout),
            "heldout_B_accuracy": float((heldout.prediction.isin([1, 2]) & heldout.prediction.eq(heldout.reference_b)).mean()),
            "heldout_majority_B_accuracy": float(heldout.reference_b.eq(majority).mean()),
            "heldout_C_accuracy": float((heldout.prediction.isin([1, 2]) & heldout.prediction.eq(heldout.target_c)).mean())})
    return pd.DataFrame(rows).set_index("qid")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--train-run", type=Path, default=Path("outputs/liquid_lora_v1"))
    parser.add_argument("--per-question", type=int, default=128)
    args = parser.parse_args()
    base.require(args.per_question >= 0, "--per-question must be nonnegative")
    import pandas as pd
    from datasets import Dataset
    from transformers import AutoTokenizer
    root = args.root.resolve()
    train_run = (args.train_run if args.train_run.is_absolute() else root / args.train_run).resolve()
    eval_run, model_path = train_run / "validation_eval", train_run / "merged"
    training_path, completion_path = train_run / "run_config.json", train_run / "training_complete.json"
    training = json.loads(training_path.read_text())
    completion = json.loads(completion_path.read_text())
    config_hash = base.digest_text(base.dumps(training))
    base.require(training["protocol"] == "liquid-nine-binary-a-to-b-lora-plain-target-v1"
                 and completion["config_hash"] == config_hash, "Training completion/configuration mismatch")
    base.require(completion["base_parameters"] == 354_483_968
                 and completion["total_training_parameters"] < 500_000_000, "Unexpected model size")
    for relative, expected in completion["artifacts_sha256"].items():
        path = (train_run / relative).resolve()
        base.require(train_run in path.parents and path.is_file() and base.digest_file(path) == expected,
                     f"Completed training artifact changed: {relative}")
    prep_path = train_run / "preprocessing_summary.json"
    prep = json.loads(prep_path.read_text())
    base.require(prep["config_hash"] == config_hash and prep["training_labels"] == "B", "Training cache provenance differs")
    arrow_path, manifest_path = train_run / "prepared/train.arrow", train_run / "training_manifest.csv"
    for path in (arrow_path, manifest_path, train_run / "mask_audit.json"):
        base.require(base.digest_file(path) == prep["artifacts_sha256"][path.name], f"Prepared artifact changed: {path.name}")
    split_path = root / "data/participant_split.csv"
    base.require(base.digest_file(split_path) == training["inputs_sha256"][str(split_path)], "Participant split changed")
    split = pd.read_csv(split_path, dtype={"pid": "string"})
    base.require(split.pid.notna().all() and split.pid.is_unique and
                 split["split"].value_counts().to_dict() == {"train": 1440, "validation": 309, "test": 309}, "Invalid split")
    base.require(split.set_index("pid").loc["574", "split"] == "train", "Participant 574 must remain in training")
    manifest = pd.read_csv(manifest_path, dtype={"pid": "string", "qid": "string"})
    dataset = Dataset.from_file(str(arrow_path))
    base.require(0 < len(manifest) == len(dataset) == prep["examples"] == completion["training_examples"],
                 "Training cache/manifest example counts differ")
    base.require(not manifest[["pid", "qid"]].duplicated().any()
                 and set(manifest.pid) == set(split.loc[split["split"].eq("train"), "pid"])
                 and set(manifest.qid) == set(base.QIDS) and manifest.label_b.isin([1, 2]).all(),
                 "Training manifest contains invalid IDs or labels")
    anchor = training["validation_baseline"]
    baseline_run = Path(anchor["path"])
    for filename, key in (("config.json", "config_sha256"), ("prepared_prompts.jsonl", "prepared_prompts_sha256"),
                          ("summary.json", "summary_sha256")):
        base.require(base.digest_file(baseline_run / filename) == anchor[key], "Frozen validation anchor changed")
    baseline_config = json.loads((baseline_run / "config.json").read_text())
    base.require(base.digest_file(base.__file__) == training["helper_scripts_sha256"][Path(base.__file__).name],
                 "Original parsing helper changed")
    eval_config = json.loads((eval_run / "config.json").read_text())
    eval_summary = json.loads((eval_run / "summary.json").read_text())
    base.require(eval_config["protocol"] == "liquid-lora-frozen-v3-validation-v1"
                 and eval_config["training_config_sha256"] == base.digest_file(training_path)
                 and eval_config["training_complete_sha256"] == base.digest_file(completion_path)
                 and eval_config["validation_baseline"] == anchor, "Validation does not correspond to this checkpoint/anchor")
    base.require(eval_summary["examples"] == eval_config["examples"] == 1545
                 and eval_summary["evaluation_split"] == "validation" and eval_summary["evaluation_labels"] == "C",
                 "Validation evaluation is not complete")
    base.require(eval_config["max_new_tokens"] == 8 and eval_config["batch_size"] == 1
                 and eval_config["do_sample"] is False, "Expected the same greedy eight-token evaluation")
    scored_path = eval_run / "scored_predictions.csv"
    validation = pd.read_csv(scored_path, dtype={"pid": "string", "qid": "string", "raw_output": "string"})
    saved_raw = base.read_jsonl(eval_run / "raw_predictions.jsonl")
    prompts = base.read_jsonl(baseline_run / "prepared_prompts.jsonl")
    raw_by_key = {(r["pid"], r["qid"]): r for r in saved_raw}
    prompt_by_key = {(p["pid"], p["qid"]): p for p in prompts}
    keys = list(validation[["pid", "qid"]].itertuples(index=False, name=None))
    base.require(len(keys) == len(set(keys)) == len(saved_raw) == len(raw_by_key) == len(prompts) == 1545
                 and set(keys) == set(raw_by_key) == set(prompt_by_key)
                 and set(validation.pid) == set(split.loc[split["split"].eq("validation"), "pid"]),
                 "Validation records/participants differ")
    for row in validation.itertuples(index=False):
        raw = raw_by_key[(row.pid, row.qid)]
        prompt = prompt_by_key[(row.pid, row.qid)]
        prediction = None if pd.isna(row.prediction) else int(row.prediction)
        base.require(raw["prompt_hash"] == row.prompt_hash == prompt["prompt_hash"]
                     and raw["input_tokens"] == row.input_tokens == prompt["input_tokens"]
                     and raw["raw_output"] == ("" if pd.isna(row.raw_output) else str(row.raw_output))
                     and raw["prediction"] == prediction == base.parse_prediction(raw["raw_output"]), "Saved validation prediction differs")
    base.require(validation.reference_b.isin([1, 2]).all() and validation.target_c.isin([1, 2]).all(), "Missing validation reference labels")
    reference_path = root / "outputs/majority_baseline_v1/predictions.csv"
    base.require(base.digest_file(reference_path) == baseline_config["inputs_sha256"][str(reference_path)],
                 "Authoritative validation labels changed")
    reference = pd.read_csv(reference_path, dtype={"pid": "string", "qid": "string"}).rename(
        columns={"prediction": "majority_prediction"}).set_index(["pid", "qid"])
    columns = ["reference_b", "target_c", "majority_prediction"]
    actual_labels = validation.set_index(["pid", "qid"])[columns]
    base.require(actual_labels.eq(reference.loc[actual_labels.index, columns]).all().all(),
                 "Scored validation labels differ from their source")
    base.require(int((~validation.prediction.isin([1, 2])).sum()) == eval_summary["invalid_outputs"], "Validation validity summary differs")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "datasets")}
    for name in ("torch", "transformers", "tokenizers"):
        base.require(versions[name] == training["versions"][name] == eval_config["versions"][name], "Use the original Liquid runtime")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selected = select_training_indices(manifest, args.per_question)
    # Full-file hashes above protect the complete cache; validate every sampled
    # sequence before GPU loading without scanning unrelated training rows.
    for index in selected:
        prefix = prefix_from_training_example(dataset[index], manifest.iloc[index], tokenizer)
        base.require(len(prefix["input_ids"]) <= training["max_input_tokens"], "Training prefix exceeds budget")
    print(f"Verified {len(selected)} training prefixes; gold digit and EOS removed.", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / "outputs" / ("liquid_fit_diagnostic_" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    config = {"protocol": "liquid-final-training-fit-greedy8-v1", "training_run": str(train_run),
        "validation_run": str(eval_run), "model": str(model_path), "seed": 42,
        "sampling": "uniform-within-question-without-label-balancing", "per_question": args.per_question,
        "qids": list(QIDS), "selected_row_indices": selected, "examples": len(selected), "max_new_tokens": 8,
        "do_sample": False, "batch_size": 1, "gold_answer_tokens_removed": 2,
        "versions": versions, "validation_baseline": anchor,
        "inputs_sha256": {str(p): base.digest_file(p) for p in
            (training_path, completion_path, prep_path, manifest_path, scored_path, eval_run / "config.json", eval_run / "summary.json")},
        "script_sha256": base.digest_file(__file__), "parser_script_sha256": base.digest_file(base.__file__)}
    base.write_json(out / "config.json", config)
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig
    base.require(torch.cuda.is_available(), "CUDA unavailable")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
    base.require(sum(p.numel() for p in model.parameters()) == completion["base_parameters"], "Merged model size changed")
    generation = GenerationConfig(do_sample=False, max_new_tokens=8, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id, use_cache=True,
        return_dict_in_generate=True, output_scores=True)
    digit_tokens = {label: tokenizer.encode(str(label), add_special_tokens=False) for label in (1, 2)}
    base.require(all(len(ids) == 1 for ids in digit_tokens.values()), "Expected single-token answer digits")
    digit_ids = {label: ids[0] for label, ids in digit_tokens.items()}
    results, failures = [], []
    with (out / "results.jsonl").open("x", encoding="utf-8") as handle:
        for ordinal, index in enumerate(selected, 1):
            metadata = manifest.iloc[index]
            prefix = prefix_from_training_example(dataset[index], metadata, tokenizer)
            inputs = {key: torch.tensor([value], dtype=torch.long, device="cuda") for key, value in prefix.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=generation)
            ids = generated.sequences[0, len(prefix["input_ids"]):].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True)
            prediction = base.parse_prediction(text)
            logits = generated.scores[0][0].float()
            base.require(bool(torch.isfinite(logits).all()), "Non-finite first-token scores")
            logp = logits.log_softmax(dim=-1)
            label = int(metadata.label_b)
            result = {"training_row": index, "pid": str(metadata.pid), "qid": str(metadata.qid),
                "label_b": label, "prediction": prediction, "raw_output": text,
                "input_tokens": len(prefix["input_ids"]), "generated_ids": ids,
                "prompt_hash": metadata.prompt_hash, "prefix_ids_sha256": base.digest_text(base.dumps(prefix["input_ids"])),
                "first_token_p1": float(logp[digit_ids[1]].exp()), "first_token_p2": float(logp[digit_ids[2]].exp()),
                "gold_digit_nll": float(-logp[digit_ids[label]])}
            results.append(result)
            handle.write(base.dumps(result) + "\n")
            handle.flush()
            if prediction != label and sum(f["qid"] == str(metadata.qid) for f in failures) < 2:
                decoded = tokenizer.decode(prefix["input_ids"], skip_special_tokens=True)
                marker = "\n\nTARGET QUESTION:\n"
                target_tail = decoded.rsplit(marker, 1)[1] if marker in decoded else None
                failures.append({"training_row": index, "qid": str(metadata.qid), "pid": str(metadata.pid),
                    "label_b": label, "raw_output": text, "target_tail": target_tail[:3500] if target_tail else None})
                base.write_json(out / "failed_training_target_tails.json", failures)
            if ordinal % 32 == 0 or ordinal == len(selected):
                print(f"Training-fit diagnostic: {ordinal}/{len(selected)}", flush=True)
            del generated, inputs
    metrics = summarize_fit(manifest, results, validation)
    metrics.to_csv(out / "metrics.csv")
    print(metrics.round(4).to_string(), flush=True)
    print("Digit NLL is a separate first-token diagnostic; accuracy uses complete generated answers.", flush=True)
    print("Saved fit diagnostic:", out, flush=True)


if __name__ == "__main__":
    main()
