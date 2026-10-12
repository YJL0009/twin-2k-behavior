#!/usr/bin/env python3
"""Select a full-B Qwen LoRA epoch on validation-C NLL, then validate once.

Selection is predeclared mean answer-token NLL (including EOS), averaged over
eligible items/participants within QID and then equally across QIDs. It is a
checkpoint-selection surrogate, not numeric MAE or a combined task-accuracy
score. All family-specific task metrics are reported for the selected epoch.
No test participant inputs or response cells are read by this evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import run_liquid_baseline as base

PROTOCOL = "qwen-full-b-lora-validation-C-nll-v1"
TRAINING_PROTOCOL = "qwen-full-b-a-to-b-lora-v1"
EXPECTED_PARAMETERS = 494_032_768
EXPECTED_ADAPTER_PARAMETERS = 4_399_104
EXPECTED_TOTAL_PARAMETERS = 498_431_872
LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
SELECTION_RULE = {
    "reference": "C", "split": "validation", "direction": "minimize",
    "criterion": "question_macro_mean_answer_token_nll",
    "answer_tokens": "canonical numeric target followed by one EOS",
    "aggregation": "mean token NLL per item; mean eligible items within QID; equal mean over QIDs",
    "tie_break": "earliest epoch", "test_evaluated": False,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(value):
    return base.digest_text(base.dumps(value))


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        require(read_json(path) == value, f"Frozen evaluation configuration changed: {path}")
    else:
        write_json(path, value)


def answer_prediction_slice(total_tokens, answer_tokens):
    """Positions predicting answer[0] ... answer[-1], including final EOS.

    For [p0,p1,p2,a0,a1,EOS], the predictor positions are [p2,a0,a1].
    The final EOS input position is not a predictor for any scored token.
    """
    require(isinstance(total_tokens, int) and isinstance(answer_tokens, int), "Token lengths must be integers")
    require(0 < answer_tokens < total_tokens, "A nonempty prompt and answer are required")
    return slice(total_tokens - answer_tokens - 1, total_tokens - 1)


def teacher_answer_nll(model, prefix_ids, answer_ids, device):
    """Efficient Qwen2 teacher forcing, with vocabulary projection only on tail.

    model.model is the same causal decoder called by Qwen2ForCausalLM.forward;
    model.lm_head is its unchanged vocabulary projection. No prompt token is
    included in loss. No cache or full-sequence vocabulary logits are retained.
    """
    import torch
    import torch.nn.functional as F
    total = list(prefix_ids) + list(answer_ids)
    predictor_positions = answer_prediction_slice(len(total), len(answer_ids))
    require(len(total) <= model.config.max_position_embeddings, "Validation target exceeds model context")
    ids = torch.tensor([total], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        tail = hidden[:, predictor_positions, :]
        logits = model.lm_head(tail).float()
        targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
        require(logits.shape[:2] == (1, len(answer_ids)), "Teacher-forcing shift or answer length mismatch")
        losses = F.cross_entropy(logits[0], targets, reduction="none")
        require(bool(torch.isfinite(losses).all().item()), "Nonfinite validation answer NLL")
        result = {"answer_token_nll_sum": float(losses.sum().item()),
                  "answer_tokens": len(answer_ids), "mean_answer_token_nll": float(losses.mean().item())}
    del ids, mask, hidden, tail, logits, targets, losses
    return result


def aggregate_nll(records):
    import pandas as pd
    require(records, "No valid validation-C answers for checkpoint selection")
    table = pd.DataFrame(records)
    require(not table.duplicated(["pid", "item_id"]).any(), "Duplicate validation NLL cases")
    by_question = table.groupby("qid", sort=True).agg(
        examples=("mean_answer_token_nll", "size"),
        mean_answer_token_nll=("mean_answer_token_nll", "mean"),
        answer_tokens=("answer_tokens", "sum"),
        answer_token_nll_sum=("answer_token_nll_sum", "sum"),
    ).reset_index()
    score = float(by_question["mean_answer_token_nll"].mean())
    require(math.isfinite(score), "Nonfinite macro selection score")
    return {"question_macro_mean_answer_token_nll": score, "eligible_examples": len(table),
            "eligible_questions": len(by_question), "eligible_participants": int(table["pid"].nunique()),
            "pooled_mean_item_answer_token_nll": float(table["mean_answer_token_nll"].mean()),
            "pooled_token_nll": float(table["answer_token_nll_sum"].sum() / table["answer_tokens"].sum())}, by_question


def validate_adapter_config(path):
    path = Path(path)
    config = read_json(path / "adapter_config.json")
    require(config.get("peft_type") == "LORA" and config.get("task_type") == "CAUSAL_LM"
            and config.get("r") == 8 and config.get("lora_alpha") == 16
            and math.isclose(float(config.get("lora_dropout", -1)), 0.05)
            and set(config.get("target_modules") or []) == LORA_TARGETS
            and config.get("bias", "none") == "none"
            and not config.get("use_dora", False) and not config.get("use_rslora", False)
            and not config.get("modules_to_save") and not config.get("rank_pattern")
            and not config.get("alpha_pattern") and not config.get("layers_to_transform")
            and not config.get("layers_pattern"), "Adapter settings differ from frozen rank-8 all-projection LoRA")
    require((path / "adapter_model.safetensors").is_file(), "Expected the complete safetensors LoRA adapter")
    return config


def _load_model(path, original_tokenizer, base_model_path, base_hashes, device="cuda",
                base_parameter_dtype="float32"):
    """Load the specified adapter on a fresh frozen base; merge only in memory.

    Reconstruct the frozen base in its recorded training dtype before upcasting
    to FP32 on CPU for merging. This preserves the rounded base weights against
    which a low-precision run learned its adapter. FP32 merging before inference
    dtype conversion prevents bypassing adapter modules in efficient scoring.
    The exact loaded adapter state is checked against its safetensors file.
    No merged model is written to disk and no existing base model is modified.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from run_qwen_full_b_test import verify_hashes
    require(base_parameter_dtype in ("float32", "bfloat16", "float16"),
            "Unsupported recorded frozen-base parameter dtype")
    loading_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                     "float16": torch.float16}[base_parameter_dtype]
    path, base_model_path = Path(path).resolve(), Path(base_model_path).resolve()
    verify_hashes(base_model_path, base_hashes)
    validate_adapter_config(path)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, padding_side="left")
    require(tokenizer.get_vocab() == original_tokenizer.get_vocab()
            and tokenizer.chat_template == original_tokenizer.chat_template,
            "Checkpoint tokenizer vocabulary or chat template changed")
    require(all(getattr(tokenizer, key) == getattr(original_tokenizer, key)
                for key in ("eos_token_id", "bos_token_id", "pad_token_id")),
            "Checkpoint tokenizer special tokens changed")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, loading = AutoModelForCausalLM.from_pretrained(str(base_model_path), local_files_only=True,
        dtype=loading_dtype, attn_implementation="sdpa", output_loading_info=True)
    require(all(not loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
            f"Checkpoint loading mismatch: {loading}")
    require(sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS
            and model.config.model_type == "qwen2" and hasattr(model, "model") and hasattr(model, "lm_head"),
            "Expected the complete 494M Qwen2 causal language model")
    require(all(parameter.dtype == loading_dtype for parameter in model.parameters()),
            "Reconstructed base parameters differ from the recorded training dtype")
    model = model.to(device="cpu", dtype=torch.float32)
    adapted = PeftModel.from_pretrained(model, str(path), is_trainable=False, local_files_only=True)
    require(sum(p.numel() for p in adapted.parameters()) == EXPECTED_TOTAL_PARAMETERS < 500_000_000,
            "Base plus adapter parameter count differs or exceeds the 500M budget")
    saved_adapter = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    loaded_adapter = get_peft_model_state_dict(adapted, save_embedding_layers=False)
    require(set(saved_adapter) == set(loaded_adapter)
            and sum(t.numel() for t in saved_adapter.values()) == EXPECTED_ADAPTER_PARAMETERS,
            "Loaded adapter keys or parameter count differ from checkpoint")
    for name, expected in saved_adapter.items():
        actual = loaded_adapter[name].detach().cpu()
        require(actual.shape == expected.shape and torch.equal(actual, expected.to(dtype=actual.dtype)),
                f"Loaded adapter tensor differs from checkpoint: {name}")
    del loaded_adapter, saved_adapter
    merged = adapted.merge_and_unload(safe_merge=True)
    require(sum(p.numel() for p in merged.parameters()) == EXPECTED_PARAMETERS
            and not any("lora_" in name for name, _ in merged.named_parameters()),
            "In-memory adapter merge did not produce a complete 494M Qwen model")
    merged.requires_grad_(False)
    merged = merged.to(device=device, dtype=dtype).eval()
    return merged, tokenizer


def _build_validation_targets(dataset, manifest, labels_c, tokenizer):
    import full_b_train_data as data
    targets, coverage, cache = [], Counter(), {}
    for index, spec in enumerate(manifest):
        key = (str(spec["pid"]), spec["item_id"])
        value = labels_c.get(key)
        state = data.label_state(value, spec)
        coverage[state] += 1
        if state != "valid":
            continue
        text = data.format_target(value, spec)
        if text not in cache:
            suffix = tokenizer.encode(text, add_special_tokens=False)
            require(suffix and tokenizer.eos_token_id not in suffix, "Malformed validation answer tokenization")
            cache[text] = list(suffix) + [tokenizer.eos_token_id]
        answer = cache[text]
        prefix = dataset[index]
        data.validate_prefix(prefix, spec)
        require(len(prefix["input_ids"]) + len(answer) <= 32768, "Validation answer plus prompt exceeds context")
        targets.append({"index": index, "pid": key[0], "item_id": key[1], "qid": spec["qid"],
                        "prompt_hash": spec["prompt_hash"], "target_hash": fingerprint(answer),
                        "answer_ids": answer})
    require(targets and coverage["valid"] == len(targets), "No usable validation-C labels")
    return targets, dict(coverage)


def evaluate_nll(candidate, checkpoint, dataset, targets, tokenizer, out, context, device="cuda"):
    from run_qwen_full_b_test import read_resume, verify_hashes
    out.mkdir(parents=True, exist_ok=True)
    verify_hashes(checkpoint, candidate["model_files_sha256"])
    config = {"protocol": PROTOCOL, "selection_rule": SELECTION_RULE, "context": context,
              "candidate": candidate, "targets_sha256": fingerprint(targets)}
    freeze_json(out / "config.json", config)
    raw_path = out / "answer_nll.jsonl"
    records = read_resume(raw_path)
    expected = {(t["pid"], t["item_id"]): t for t in targets}
    seen = set()
    for record in records:
        key = (str(record["pid"]), record["item_id"])
        require(key in expected and key not in seen, "Duplicate or unexpected saved NLL case")
        target = expected[key]
        require(all(record.get(field) == target[field] for field in ("qid", "prompt_hash", "target_hash")),
                "Saved NLL prompt/target changed")
        require(record["answer_tokens"] == len(target["answer_ids"])
                and all(math.isfinite(float(record[name])) and float(record[name]) >= 0
                        for name in ("answer_token_nll_sum", "mean_answer_token_nll"))
                and math.isclose(record["mean_answer_token_nll"] * record["answer_tokens"],
                                 record["answer_token_nll_sum"], rel_tol=2e-6, abs_tol=1e-6),
                "Saved NLL values are inconsistent")
        seen.add(key)
    pending = [target for target in targets if (target["pid"], target["item_id"]) not in seen]
    if pending:
        model, _ = _load_model(checkpoint, tokenizer, context["base_model_path"],
                               context["base_model_files_sha256"], device=device,
                               base_parameter_dtype=context.get("base_parameter_dtype", "float32"))
        started = time.monotonic()
        try:
            with raw_path.open("a", encoding="utf-8") as handle:
                for done, target in enumerate(pending, 1):
                    prefix = dataset[target["index"]]["input_ids"]
                    score = teacher_answer_nll(model, prefix, target["answer_ids"], device)
                    record = {name: target[name] for name in ("pid", "item_id", "qid", "prompt_hash", "target_hash")}
                    record.update(score)
                    handle.write(base.dumps(record) + "\n")
                    handle.flush()
                    records.append(record)
                    if done % 100 == 0 or done == len(pending):
                        rate = done / max(time.monotonic() - started, 1e-9)
                        print(f"{candidate['name']} validation-C NLL: {len(records)}/{len(targets)} | {rate:.2f} items/s", flush=True)
        finally:
            del model
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    require(len(records) == len(targets), "Validation NLL coverage incomplete")
    summary, by_question = aggregate_nll(records)
    summary.update(checkpoint=candidate["name"], epoch=float(candidate["epoch"]),
                   artifact_type="lora_adapter", adapter_path=str(checkpoint),
                   base_model_path=context["base_model_path"],
                   base_parameter_dtype=context.get("base_parameter_dtype", "float32"),
                   config_hash=fingerprint(config))
    by_question.to_csv(out / "nll_by_question.csv", index=False)
    write_json(out / "summary.json", summary)
    return summary


def make_report(summary, selection, scores_dir):
    import pandas as pd
    family = pd.read_csv(scores_dir / "metrics_by_family.csv")
    questions = pd.read_csv(scores_dir / "metrics_by_question.csv")
    columns = ["reference", "family", "valid_labels", "prediction_valid_rate_on_valid_labels",
               "question_macro_categorical_accuracy", "question_macro_mode_categorical_accuracy",
               "question_macro_ordinal_nmae_invalid_penalty", "question_macro_median_ordinal_nmae",
               "question_macro_bounded_nmae_invalid_penalty", "question_macro_median_bounded_nmae"]
    numeric = questions[questions.family.isin(["bounded_numeric", "unbounded_numeric"])][
        ["reference", "qid", "numeric_valid_rate", "numeric_mae_valid_only",
         "median_numeric_mae_on_model_valid", "numeric_fallback_mae_all", "median_numeric_mae_all"]]
    return "\n".join([
        "Full-B Qwen LoRA training and validation complete", f"Selected epoch: {selection['best']['epoch']}",
        f"Base model: {selection['best']['base_model_path']}",
        f"Frozen base training dtype: {selection['best'].get('base_parameter_dtype', 'float32')}",
        f"Selected adapter: {selection['best']['adapter_path']}",
        f"Training parameters: base {EXPECTED_PARAMETERS:,}; adapter {EXPECTED_ADAPTER_PARAMETERS:,}; total {EXPECTED_TOTAL_PARAMETERS:,}",
        f"Validation-C question-macro mean answer-token NLL: {selection['best']['question_macro_mean_answer_token_nll']:.6f}",
        f"Validation participants: {summary['participants']}; scalar generations: {summary['prompts']}", "",
        "Family metrics (C primary, B secondary; accuracy higher is better, normalized errors lower):",
        family[columns].round(6).to_string(index=False), "",
        "Numeric errors by question (original units; lower is better):", numeric.round(6).to_string(index=False), "",
        "Epoch selection used validation-C canonical answer-token NLL, including EOS; this is a selection surrogate.",
        "Each item's mean-token NLL is averaged within QID, then equally across QIDs; ties choose the earlier epoch.",
        "All question families were included in training. No combined accuracy/error score is reported.",
        "Adapters were loaded on the original base and merged in memory for validation; no merged checkpoint was written.",
        "Categorical invalid outputs are wrong; ordinal/bounded invalid outputs have normalized error 1.",
        "Numeric raw MAE is valid-output-only with a matched-mask baseline; median fallback is a separate system metric.",
        "Human B-vs-C references and matrix-vector exact matches are in the detailed CSV files.",
        "No test participants or test response cells were read by this training/validation workflow.",
        "If the earlier full-B test evaluation was run, further model changes are exploratory relative to that exposed test set.",
        f"Detailed scores: {scores_dir}",
    ]) + "\n"


def select_and_evaluate(root, run, model_path, selection_model, validation_dataset,
                        manifest, preparation, checkpoints, select_only=False):
    import pandas as pd
    from transformers import AutoTokenizer
    import full_b_data as inputs
    import full_b_metrics as metrics
    from run_qwen_full_b_test import infer, validate_records, verify_hashes, verify_retest_metadata
    root, run = Path(root).resolve(), Path(run).resolve()
    training = read_json(run / "run_config.json")
    completion = read_json(run / "training_complete.json")
    require(training["protocol"] == TRAINING_PROTOCOL and int(training["epochs"]) == 3,
            "Expected the predeclared three-epoch full-B run")
    base_parameter_dtype = training.get("base_parameter_dtype", "float32")
    require(base_parameter_dtype in ("float32", "bfloat16", "float16")
            and training.get("adapter_parameter_dtype", "float32") == "float32",
            "Unsupported recorded frozen-base or trainable-adapter dtype")
    require(completion["protocol"] == TRAINING_PROTOCOL
            and completion["config_hash"] == fingerprint(training), "Training completion protocol/hash differs")
    require(completion["base_parameters"] == EXPECTED_PARAMETERS
            and completion["trainable_parameters"] == completion["adapter_parameters"] == EXPECTED_ADAPTER_PARAMETERS
            and completion["total_training_parameters"] == EXPECTED_TOTAL_PARAMETERS < 500_000_000,
            "Completed run does not match the frozen base plus rank-8 adapter parameter budget")
    require(training["preparation_config_sha256"] == base.digest_file(run / "preparation_config.json")
            and training["preparation_summary_sha256"] == base.digest_file(run / "preparation_summary.json"),
            "Training preparation provenance changed")
    require(len(checkpoints) == 3 and sorted(float(c["epoch"]) for c in checkpoints) == [1.0, 2.0, 3.0],
            "Need one frozen candidate from each of the three completed epochs")
    require(checkpoints == completion["candidate_checkpoints"], "Candidate list differs from completed training")
    require(len({c["name"] for c in checkpoints}) == 3, "Candidate checkpoint names must be unique")
    require(all(c.get("artifact_type") == "lora_adapter" for c in checkpoints),
            "Expected adapter-only epoch candidates")
    require(Path(training["model"]).resolve() == Path(model_path).resolve(), "Original base-model path changed")
    verify_hashes(Path(model_path), training["model_files_sha256"])
    split = inputs.read_split(root)
    pids = {str(spec["pid"]) for spec in manifest}
    require(pids == split["validation"] and not (pids & split["test"]), "Validation participant membership differs")
    require(len(manifest) == len(validation_dataset) and set(validation_dataset.column_names) == {"input_ids", "attention_mask"},
            "Validation must be the complete label-free prefix dataset")
    keys = {(str(s["pid"]), s["item_id"]) for s in manifest}
    require(len(keys) == len(manifest), "Duplicate validation item")
    response = root / "data/Twin-2K-500/question_catalog_and_human_response_csv"
    b_path, c_path = response / "wave1_3_response.csv", response / "wave4_response.csv"
    item_specs = {}
    for spec in manifest:
        item = spec["item_id"]
        if item in item_specs:
            require(metrics._schema_signature(spec) == metrics._schema_signature(item_specs[item]),
                    f"Inconsistent item schema: {item}")
        item_specs[item] = spec
    evaluator_paths = [Path(__file__), Path(inputs.__file__), Path(metrics.__file__)]
    config = {"protocol": PROTOCOL, "selection_rule": SELECTION_RULE, "training_config_hash": fingerprint(training),
              "training_complete_sha256": base.digest_file(run / "training_complete.json"),
              "preparation_hash": fingerprint(preparation), "manifest_hash": fingerprint(manifest),
              "labels_sha256": {"B": base.digest_file(b_path), "C": base.digest_file(c_path)},
              "scripts_sha256": {p.name: base.digest_file(p) for p in evaluator_paths},
              "versions": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "tokenizers", "peft", "numpy", "pandas")},
              "model": str(Path(model_path).resolve()), "selection_model": str(Path(selection_model).resolve()),
              "base_model_path": str(Path(model_path).resolve()),
              "base_model_files_sha256": training["model_files_sha256"],
              "base_parameter_dtype": base_parameter_dtype, "adapter_parameter_dtype": "float32",
              "adapter_loading": "fresh base in recorded training dtype; CPU FP32 upcast; exact adapter state check; safe in-memory merge; inference dtype conversion",
              "test_response_cells_read": False}
    require(config["versions"]["transformers"] == "5.2.0", "Use the existing liquid_tf52 Python environment")
    for name in ("torch", "transformers", "tokenizers", "peft"):
        require(config["versions"][name] == training["versions"][name], f"Training/evaluation {name} versions differ")
    evaluation = run / "validation"
    evaluation.mkdir(exist_ok=True)
    freeze_json(evaluation / "evaluation_config.json", config)
    # Reuse TRAIN-B-only baselines already frozen during training preparation.
    baseline_path = run / "training_baselines.json"
    require(baseline_path.is_file()
            and base.digest_file(baseline_path) == preparation["artifacts_sha256"][baseline_path.name],
            "Prepared training-B baselines changed")
    baselines = read_json(baseline_path)
    require(set(item_specs) <= set(baselines), "A validation item lacks a frozen training baseline")
    require(all(baseline["n"] > 0 for baseline in baselines.values()), "An item lacks a training-B baseline")
    freeze_json(evaluation / "training_baselines.json", baselines)
    # C is authorized for validation checkpoint selection; check stimuli before scoring likelihood.
    verify_retest_metadata(root, pids, evaluation)
    labels_c = inputs.read_selected_labels(c_path, pids, list(item_specs))
    original = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="left")
    targets, coverage = _build_validation_targets(validation_dataset, manifest, labels_c, original)
    freeze_json(evaluation / "selection_label_coverage.json", {
        "total_prompts": len(manifest), "states": coverage,
        "policy": "Only valid observed validation C labels enter NLL; missing/invalid labels counted here.",
        "eligible_question_ids": sorted({target["qid"] for target in targets}),
    })
    summaries = []
    for candidate in sorted(checkpoints, key=lambda c: float(c["epoch"])):
        checkpoint = (run / candidate["path"]).resolve()
        require(run in checkpoint.parents, "Candidate checkpoint escapes training run")
        validate_adapter_config(checkpoint)
        summaries.append(evaluate_nll(candidate, checkpoint, validation_dataset, targets, original,
                                     evaluation / "selection" / candidate["name"], config))
    pd.DataFrame(summaries).to_csv(run / "epoch_comparison.csv", index=False)
    best = min(summaries, key=lambda row: (row["question_macro_mean_answer_token_nll"], row["epoch"]))
    selection = {"protocol": PROTOCOL, "selection_split": "validation", "selection_labels": "C",
                 "selection_rule": SELECTION_RULE, "training_complete_sha256": config["training_complete_sha256"],
                 "evaluation_config_hash": fingerprint(config), "test_evaluated": False, "best": best}
    freeze_json(run / "best_checkpoint.json", selection)
    print("Selected full-B LoRA adapter:", json.dumps(best, indent=2), flush=True)
    if select_only:
        return selection
    generation_dir = evaluation / "best"
    generation_dir.mkdir(exist_ok=True)
    freeze_json(generation_dir / "generation_config.json", {
        "protocol": PROTOCOL, "selection_sha256": base.digest_file(run / "best_checkpoint.json"),
        "evaluation_config_hash": fingerprint(config), "do_sample": False, "batch_size": 1,
        "output_caps": "8 for original nine binary QIDs; 16 for all other scalar items",
    })
    from run_qwen_full_b_test import read_resume
    raw_path = generation_dir / "raw_predictions.jsonl"
    records = read_resume(raw_path)
    seen = validate_records(records, manifest)
    candidate = next(c for c in checkpoints if c["name"] == best["checkpoint"])
    checkpoint = Path(best["adapter_path"])
    verify_hashes(checkpoint, candidate["model_files_sha256"])
    if len(seen) < len(manifest):
        model, tokenizer = _load_model(checkpoint, original, config["base_model_path"],
                                        config["base_model_files_sha256"],
                                        base_parameter_dtype=config["base_parameter_dtype"])
        try:
            records = infer(model, tokenizer, validation_dataset, manifest, raw_path, batch_size=1,
                            status_path=generation_dir / "status.json")
        finally:
            del model
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    require(len(validate_records(records, manifest)) == len(manifest), "Validation generation incomplete")
    require(base.digest_file(b_path) == config["labels_sha256"]["B"]
            and base.digest_file(c_path) == config["labels_sha256"]["C"], "Labels changed during validation")
    verify_retest_metadata(root, pids, generation_dir)
    labels_b = inputs.read_selected_labels(b_path, pids, list(item_specs))
    scores_dir = generation_dir / "scores"
    all_trained_qids = set(training["trained_question_ids"])
    require(all_trained_qids == set(preparation["coverage"]["train"]["qid_examples"]),
            "Training question coverage declaration changed")
    summary = metrics.score_full(records, manifest, labels_b, labels_c, baselines, scores_dir,
                                 trained_qids=all_trained_qids, trained_scope="trained_full_b_lora")
    report = make_report(summary, selection, scores_dir)
    (run / "full_b_lora_validation_report.txt").write_text(report, encoding="utf-8")
    write_json(generation_dir / "evaluation_complete.json", {
        "protocol": PROTOCOL, "evaluation_config_hash": fingerprint(config),
        "selected_checkpoint": best["checkpoint"], "base_model_path": best["base_model_path"],
        "adapter_path": best["adapter_path"], "examples": len(records),
        "predictions_sha256": base.digest_file(raw_path), "test_evaluated": False,
    })
    print(report, flush=True)
    return {"selection": selection, "validation_metrics": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--run", type=Path, default=Path("outputs/qwen_full_b_lora_v4_b16_a2"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    run = (args.run if args.run.is_absolute() else root / args.run).resolve()
    training = read_json(run / "run_config.json")
    completion = read_json(run / "training_complete.json")
    require(training["protocol"] == TRAINING_PROTOCOL, "Not a full-B training run")
    import full_b_train_data as data
    _, validation_dataset, manifest, preparation = data.prepare_training(
        root, run, Path(training["model"]), Path(training["selection_model"]), workers=args.workers)
    select_and_evaluate(root, run, Path(training["model"]), Path(training["selection_model"]),
                        validation_dataset, manifest, preparation, completion["candidate_checkpoints"],
                        select_only=args.select_only)


if __name__ == "__main__":
    main()
